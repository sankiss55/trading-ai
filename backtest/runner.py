"""Backtest runner: simulation adapters + ``application.market_flow`` (sec. 45, 8.6).

Wiring (sec. 8.6): ``SimClock`` + ``StaticCalendar`` (stored calendar) +
``HistoricalFeed`` (stored bars) + ``SimulatedBroker`` (stop-first brackets, slippage)
+ :class:`application.market_flow.MarketFlow`. No indicator, rule, sizing, SL/TP or
fill logic is implemented here (sec. 45.2.1): the runner only moves time, routes bars,
submits the orders the flow proposes and records fills.

Loop per session (all decisions use data up to the bar being decided, sec. 45.2.8):

1. Clock to the session open; deferred system exits are submitted (they fill at the
   first bar's open).
2. For each minute ``t`` (all symbols): clock to ``t + 1 min`` (the bar is known),
   ``broker.process_bar`` (fills of orders submitted before ``t``), limit-entry timeout,
   intraday flatten from ``flatten_at`` on (sec. 23.5), then ``on_minute_bar`` and, for
   each closed primary bar, ``on_closed_bar``: an :class:`EntryProposal` becomes a
   bracket (MARKET or LIMIT entry, TIF ``DAY`` intraday / ``GTC`` swing,
   ``client_order_id = backtest-{signal_id[4:20]}-entry``); an :class:`ExitRequest`
   runs the system exit (sec. 23.6: cancel the open legs, query the position, MARKET
   sell of the real quantity with ``client_order_id = backtest-...-exit``). Market
   orders therefore fill at the next minute's open plus slippage.
3. Session end: clock to ``close + grace``, buckets closed by time are decided,
   ``broker.close_session()`` (DAY orders expire). Exits requested while the market is
   closed are deferred to the next open (a DAY market order would only expire).

Periods: one continuous simulation over ``[start_date, end_date]`` (warm-up bars before
``start_date`` are fed to the indicators only). In-sample / out-of-sample and
walk-forward windows are evaluated on that single run. There are no optimisable
parameters yet, so walk-forward = evaluate each test window sequentially and report its
metrics; the train windows are reported for reference only.

Slippage sensitivity: the simulation is repeated with the broker slippage
(``risk.slippage_buffer_bps``) multiplied by 1.5 and 2.0; sizing keeps the configured
buffer.

Parallelism: the base run and the sensitivity runs are independent deterministic
simulations. With ``workers > 1`` they run in separate processes (each loads the
dataset itself); results are collected in the fixed scenario order, so the report is
byte-identical to the sequential run (``workers = 1``: one process, one load).

Refusal: if any OWNER_DECISION the backtest needs is ``null`` the runner raises
:class:`BacktestRefusedError` listing them; no default is invented. The sec. 45.4
thresholds may be ``null``: the report status is then ``PENDING_OWNER_DECISION``.
"""

from __future__ import annotations

import asyncio
import calendar as _calendar
import hashlib
import math
import os
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import groupby
from pathlib import Path
from typing import Final, Literal

from adapters.simulation import SimClock, SimulatedBroker, StaticCalendar
from app.config import (
    AppConfig,
    LoadedConfig,
    exit_params,
    pending_backtest_decisions,
    risk_params,
    session_window_params,
)
from application.market_flow import (
    EntryProposal,
    ExitRequest,
    MarketFlow,
    MarketFlowParams,
    Rejected,
    TradeBook,
)
from backtest.data import BacktestData, load_backtest_data
from backtest.report import (
    DISCLAIMER,
    BacktestReport,
    EquityPoint,
    PeriodMetrics,
    RunCounters,
    SlippageScenario,
    TradeRecord,
    WalkForwardWindow,
    compute_metrics,
    continuation_check,
)
from domain.errors import NonRetryableError
from domain.market.session import SessionWindows, compute_session_windows
from domain.models import (
    Bar,
    BracketOrderRequest,
    DataFeed,
    ExitReason,
    HoldingMode,
    OrderSide,
    OrderType,
    SessionDay,
    SimpleOrderRequest,
    TimeInForce,
    TradeUpdateEvent,
)
from domain.risk.risk_engine import PendingEntry, PositionStop
from domain.strategy.strategy import Strategy

__all__ = [
    "CLIENT_ORDER_PREFIX",
    "SLIPPAGE_MULTIPLIERS",
    "BacktestRefusedError",
    "SimulationResult",
    "add_months",
    "data_fingerprint",
    "default_workers",
    "run_backtest",
    "run_backtest_async",
    "simulate",
    "walk_forward_windows",
]

CLIENT_ORDER_PREFIX: Final = "backtest"
SLIPPAGE_MULTIPLIERS: Final[tuple[Decimal, ...]] = (Decimal("1.5"), Decimal("2.0"))
"""Sensitivity runs of sec. 45.3 (+50 % and +100 % slippage)."""

_ONE_MINUTE = timedelta(minutes=1)
_ONE_SECOND = timedelta(seconds=1)
_REGULAR_SESSION_MINUTES = 390


class BacktestRefusedError(NonRetryableError):
    """The backtest cannot run: required OWNER_DECISIONs are still ``null``."""

    def __init__(self, pending: Sequence[str]) -> None:
        self.pending = tuple(pending)
        super().__init__(
            "backtest refused: these OWNER_DECISION parameters are still null "
            f"(set them in config.yaml, no default is assumed): {', '.join(self.pending)}",
            code="OWNER_DECISION_PENDING",
        )


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Output of one simulation run."""

    trades: tuple[TradeRecord, ...]
    equity_curve: tuple[EquityPoint, ...]
    counters: RunCounters
    sessions: tuple[SessionDay, ...]
    slippage_bps: Decimal
    final_equity: Decimal


@dataclass
class _OpenTrade:
    signal_id: str
    symbol: str
    qty: int
    entry_ref: Decimal
    stop_price: Decimal
    take_profit_price: Decimal
    order_type: OrderType
    parent_id: str
    leg_ids: tuple[str, ...]
    submitted_at: datetime
    entry_price: Decimal | None = None
    entry_filled_at: datetime | None = None
    exit_reason: ExitReason | None = None
    exit_order_id: str | None = None


@dataclass
class _Counters:
    minute_bars: int = 0
    closed_bars: int = 0
    signals: int = 0
    entries_submitted: int = 0
    entries_filled: int = 0
    entries_canceled: int = 0
    entries_expired: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    broker_rejections: Counter[str] = field(default_factory=Counter)
    exits_by_reason: Counter[str] = field(default_factory=Counter)
    deferred_exits: int = 0
    sessions_with_position_after_close: int = 0
    legs_expired_with_position: int = 0


def _short_id(signal_id: str) -> str:
    return signal_id.removeprefix("sig_")[:16]


# --------------------------------------------------------------------------- simulator


class _Simulator:
    """One deterministic simulation over the configured period."""

    def __init__(
        self,
        config: AppConfig,
        data: BacktestData,
        *,
        starting_cash: Decimal,
        commission_per_fill: Decimal,
        slippage_multiplier: Decimal,
    ) -> None:
        backtest = config.backtest
        if backtest.start_date is None or backtest.end_date is None:  # pragma: no cover
            raise BacktestRefusedError(("backtest.start_date", "backtest.end_date"))
        holding = config.strategy.holding_mode
        whitelist = config.universe.whitelist
        feed = config.market_data.feed
        risk = risk_params(config)
        if (
            holding is None
            or whitelist is None
            or feed is None
            or (risk.slippage_buffer_bps is None)
        ):  # pragma: no cover - guarded by pending_backtest_decisions
            raise BacktestRefusedError(pending_backtest_decisions(config))
        self._config = config
        self._holding = holding
        self._symbols = tuple(whitelist)
        self._commission = commission_per_fill
        self._trading = tuple(
            s for s in data.sessions if backtest.start_date <= s.session_date <= backtest.end_date
        )
        if not self._trading:
            raise NonRetryableError(
                f"no stored session between {backtest.start_date} and {backtest.end_date}",
                code="NO_SESSIONS",
            )
        strategy = Strategy(config.strategy, strategy_version=config.strategy_version)
        primary = strategy.params.primary_timeframe
        if primary is None or primary.minutes is None:
            raise NonRetryableError(
                "the Phase 1 backtest supports intraday primary timeframes only",
                code="UNSUPPORTED_TIMEFRAME",
            )
        # The clock starts at the first stored session and only moves forward.
        self._clock = SimClock(data.sessions[0].open_utc)
        self._data = data
        self._calendar = StaticCalendar(data.sessions, self._clock)
        self._slippage_bps = risk.slippage_buffer_bps * slippage_multiplier
        self._broker = SimulatedBroker(
            clock=self._clock,
            starting_cash=starting_cash,
            slippage_bps=self._slippage_bps,
            commission_per_fill=commission_per_fill,
            id_prefix=CLIENT_ORDER_PREFIX,
        )
        self._session_params = session_window_params(config)
        self._grace = timedelta(seconds=float(config.market_data.bar_close_grace_seconds))
        offset = (
            config.execution.limit_entry_offset_bps
            if config.execution.entry_order_type == "limit"
            else None
        )
        self._flow = MarketFlow(
            strategy=strategy,
            exit_params=exit_params(config),
            risk_params=risk,
            params=MarketFlowParams(
                symbols=self._symbols,
                feed=feed,
                bar_close_grace_seconds=config.market_data.bar_close_grace_seconds,
                window_bars=config.market_data.history_warmup_bars,
                session=self._session_params,
                limit_entry_offset_bps=offset,
            ),
            clock=self._clock,
            broker=self._broker,
        )
        per_session = _REGULAR_SESSION_MINUTES // primary.minutes
        warmup_count = math.ceil(self._flow.window_size / per_session) + 1
        earlier = [s for s in data.sessions if s.session_date < backtest.start_date]
        self._warmup = tuple(earlier[-warmup_count:])
        self._tif = TimeInForce.DAY if holding is HoldingMode.INTRADAY else TimeInForce.GTC
        self._entry_timeout = timedelta(seconds=config.execution.entry_timeout_seconds)
        self._open: dict[str, _OpenTrade] = {}
        self._by_order: dict[str, _OpenTrade] = {}
        self._closed: list[TradeRecord] = []
        self._deferred: dict[str, ExitReason] = {}
        self._last_exits: dict[str, datetime] = {}
        self._curve: list[EquityPoint] = []
        self._counters = _Counters()
        self._peak_equity = starting_cash
        self._week_start_equity = starting_cash
        self._week: tuple[int, int] | None = None
        self._windows: SessionWindows | None = None

    # ------------------------------------------------------------------ driver

    async def run(self) -> SimulationResult:
        for session in self._warmup:
            await self._warmup_session(session)
        for session in self._trading:
            stored = await self._calendar.get_session(session.session_date)
            if stored is None:  # pragma: no cover - sessions come from the same calendar
                raise NonRetryableError("calendar lost a session", code="INVALID_CALENDAR")
            await self._trade_session(stored)
        counters = self._counters
        return SimulationResult(
            trades=tuple(self._closed),
            equity_curve=tuple(self._curve),
            counters=RunCounters(
                minute_bars=counters.minute_bars,
                closed_bars=counters.closed_bars,
                signals=counters.signals,
                entries_submitted=counters.entries_submitted,
                entries_filled=counters.entries_filled,
                entries_canceled=counters.entries_canceled,
                entries_expired=counters.entries_expired,
                rejections=dict(sorted(counters.rejections.items())),
                broker_rejections=dict(sorted(counters.broker_rejections.items())),
                exits_by_reason=dict(sorted(counters.exits_by_reason.items())),
                deferred_exits=counters.deferred_exits,
                sessions_with_position_after_close=counters.sessions_with_position_after_close,
                legs_expired_with_position=counters.legs_expired_with_position,
                open_trades_at_end=len(self._open),
            ),
            sessions=self._trading,
            slippage_bps=self._slippage_bps,
            final_equity=self._broker.equity(),
        )

    async def _session_minutes(self, session: SessionDay) -> list[Bar]:
        bars: list[Bar] = []
        for symbol in self._symbols:
            bars.extend(
                await self._data.feed.get_minute_bars(symbol, session.open_utc, session.close_utc)
            )
        bars.sort(key=lambda b: (b.bar_start_utc, b.symbol))
        return bars

    async def _warmup_session(self, session: SessionDay) -> None:
        self._clock.advance_to(session.open_utc)
        self._flow.add_session(session)
        for bar in await self._session_minutes(session):
            self._clock.advance_to(bar.bar_end_utc)
            for closed in self._flow.on_minute_bar(bar).closed_bars:
                self._flow.record_closed_bar(closed)
        self._clock.advance_to(session.close_utc + self._grace + _ONE_SECOND)
        for closed in self._flow.on_clock():
            self._flow.record_closed_bar(closed)

    async def _trade_session(self, session: SessionDay) -> None:
        self._clock.advance_to(session.open_utc)
        self._flow.add_session(session)
        self._windows = compute_session_windows(session, self._session_params)
        week = session.session_date.isocalendar()[:2]
        if week != self._week:
            self._week = (week[0], week[1])
            self._week_start_equity = self._broker.equity()
        for parent_id, reason in sorted(self._deferred.items()):
            trade = self._open.get(parent_id)
            if trade is not None:
                await self._system_exit(trade, reason)
        self._deferred.clear()

        minutes = await self._session_minutes(session)
        for start, group_iter in groupby(minutes, key=lambda b: b.bar_start_utc):
            group = list(group_iter)
            self._clock.advance_to(start + _ONE_MINUTE)
            for bar in group:
                self._broker.process_bar(bar)
                self._counters.minute_bars += 1
            self._drain()
            await self._expire_limit_entries()
            if self._windows.is_flatten_time(self._clock.now_utc()):
                await self._flatten()
            closed: list[Bar] = []
            for bar in group:
                closed.extend(self._flow.on_minute_bar(bar).closed_bars)
            if closed:
                await self._decide(closed)
                await self._sample_equity()

        self._clock.advance_to(session.close_utc + self._grace + _ONE_SECOND)
        late = self._flow.on_clock()
        if late:
            await self._decide(late)
        self._broker.close_session()
        self._drain()
        if self._holding is HoldingMode.INTRADAY and await self._broker.get_positions():
            self._counters.sessions_with_position_after_close += 1
        await self._sample_equity()

    # ------------------------------------------------------------------ decisions

    def _book(self) -> TradeBook:
        filled = [t for t in self._open.values() if t.entry_filled_at is not None]
        return TradeBook(
            week_start_equity=self._week_start_equity,
            peak_equity=self._peak_equity,
            pending_entries=tuple(
                PendingEntry(
                    symbol=t.symbol, qty=t.qty, entry_ref=t.entry_ref, stop_price=t.stop_price
                )
                for t in self._open.values()
                if t.entry_filled_at is None
            ),
            position_stops=tuple(
                PositionStop(symbol=t.symbol, stop_price=t.stop_price) for t in filled
            ),
            entry_fills={t.symbol: t.entry_filled_at for t in filled if t.entry_filled_at},
            last_exits=dict(self._last_exits),
        )

    async def _decide(self, closed: Sequence[Bar]) -> None:
        for bar in closed:
            self._counters.closed_bars += 1
            outcome = await self._flow.on_closed_bar(bar, book=self._book())
            if isinstance(outcome, EntryProposal):
                self._counters.signals += 1
                await self._submit_entry(outcome)
            elif isinstance(outcome, Rejected):
                self._counters.signals += 1
                self._counters.rejections.update(outcome.codes)
            elif isinstance(outcome, ExitRequest):
                trade = next((t for t in self._open.values() if t.symbol == outcome.symbol), None)
                if trade is not None:
                    await self._request_exit(trade, outcome.reason)

    async def _submit_entry(self, proposal: EntryProposal) -> None:
        trade = proposal.trade
        order_type: Literal[OrderType.MARKET, OrderType.LIMIT] = (
            OrderType.MARKET if proposal.limit_price is None else OrderType.LIMIT
        )
        request = BracketOrderRequest(
            symbol=trade.symbol,
            qty=trade.qty,
            order_type=order_type,
            limit_price=proposal.limit_price,
            time_in_force=self._tif,
            take_profit_limit_price=trade.take_profit_price,
            stop_loss_stop_price=trade.stop_price,
            client_order_id=f"{CLIENT_ORDER_PREFIX}-{_short_id(trade.signal_id)}-entry",
        )
        try:
            order = await self._broker.submit_bracket(request)
        except NonRetryableError as exc:
            self._counters.broker_rejections[exc.code or "UNKNOWN"] += 1
            self._drain()
            return
        if order.order_id in self._by_order:  # idempotent re-submission (sec. 22)
            return
        record = _OpenTrade(
            signal_id=trade.signal_id,
            symbol=trade.symbol,
            qty=trade.qty,
            entry_ref=trade.entry_ref,
            stop_price=trade.stop_price,
            take_profit_price=trade.take_profit_price,
            order_type=order_type,
            parent_id=order.order_id,
            leg_ids=tuple(leg.order_id for leg in order.legs),
            submitted_at=self._clock.now_utc(),
        )
        self._open[order.order_id] = record
        for order_id in (order.order_id, *record.leg_ids):
            self._by_order[order_id] = record
        self._counters.entries_submitted += 1
        self._drain()

    async def _request_exit(self, trade: _OpenTrade, reason: ExitReason) -> None:
        if trade.exit_reason is not None or trade.parent_id in self._deferred:
            return
        if self._flow.session_of(self._clock.now_utc()) is None:
            self._deferred[trade.parent_id] = reason
            self._counters.deferred_exits += 1
            return
        await self._system_exit(trade, reason)

    async def _cancel(self, order_id: str) -> None:
        try:
            await self._broker.cancel_order(order_id)
        except NonRetryableError as exc:
            if exc.code != "ORDER_NOT_CANCELABLE":  # filled/expired meanwhile: nothing to do
                raise

    async def _system_exit(self, trade: _OpenTrade, reason: ExitReason) -> None:
        """Procedure 23.6: cancel open legs, query the position, market sell it."""
        if trade.entry_filled_at is None:
            await self._cancel(trade.parent_id)
            self._drain()
            return
        trade.exit_reason = reason
        for leg_id in trade.leg_ids:
            await self._cancel(leg_id)
        self._drain()
        if trade.parent_id not in self._open:
            return  # a leg filled while canceling: the real exit is already recorded
        position = next(
            (p for p in await self._broker.get_positions() if p.symbol == trade.symbol), None
        )
        if position is None:  # pragma: no cover - the trade would have been closed
            return
        order = await self._broker.submit_simple(
            SimpleOrderRequest(
                symbol=trade.symbol,
                qty=int(position.qty),
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id=f"{CLIENT_ORDER_PREFIX}-{_short_id(trade.signal_id)}-exit",
            )
        )
        trade.exit_order_id = order.order_id
        self._by_order[order.order_id] = trade
        self._drain()

    async def _flatten(self) -> None:
        """Sec. 23.5: from ``flatten_at`` on, close everything and cancel pending entries."""
        for trade in list(self._open.values()):
            if trade.entry_filled_at is None:
                await self._cancel(trade.parent_id)
                self._drain()
            elif trade.exit_reason is None:
                await self._system_exit(trade, ExitReason.END_OF_DAY)

    async def _expire_limit_entries(self) -> None:
        now = self._clock.now_utc()
        for trade in list(self._open.values()):
            if (
                trade.order_type is OrderType.LIMIT
                and trade.entry_filled_at is None
                and now - trade.submitted_at >= self._entry_timeout
            ):
                await self._cancel(trade.parent_id)
                self._drain()

    # ------------------------------------------------------------------ fills

    def _drain(self) -> None:
        for update in self._broker.drain_trade_updates():
            order = update.order
            trade = self._by_order.get(order.order_id)
            if trade is None or trade.parent_id not in self._open:
                continue
            if update.event is TradeUpdateEvent.FILL and update.fill is not None:
                fill = update.fill
                if order.order_id == trade.parent_id:
                    trade.entry_price = fill.price
                    trade.entry_filled_at = fill.timestamp_utc
                    self._counters.entries_filled += 1
                    continue
                if order.order_id == trade.exit_order_id and trade.exit_reason is not None:
                    reason = trade.exit_reason
                elif order.order_type is OrderType.STOP:
                    reason = ExitReason.STOP_LOSS
                else:
                    reason = ExitReason.TAKE_PROFIT
                self._close_trade(trade, fill.price, fill.timestamp_utc, reason)
            elif (
                order.order_id == trade.parent_id
                and trade.entry_filled_at is None
                and (update.event in (TradeUpdateEvent.CANCELED, TradeUpdateEvent.EXPIRED))
            ):
                self._discard(trade, expired=update.event is TradeUpdateEvent.EXPIRED)
            elif (
                update.event is TradeUpdateEvent.EXPIRED
                and order.order_id in trade.leg_ids
                and trade.entry_filled_at is not None
            ):
                self._counters.legs_expired_with_position += 1

    def _discard(self, trade: _OpenTrade, *, expired: bool) -> None:
        del self._open[trade.parent_id]
        if expired:
            self._counters.entries_expired += 1
        else:
            self._counters.entries_canceled += 1

    def _close_trade(
        self, trade: _OpenTrade, price: Decimal, when: datetime, reason: ExitReason
    ) -> None:
        entry = trade.entry_price
        entry_at = trade.entry_filled_at
        if entry is None or entry_at is None:  # pragma: no cover - exits need an entry
            raise NonRetryableError("exit fill before entry fill", code="STATE_MISMATCH")
        qty = Decimal(trade.qty)
        gross = (price - entry) * qty
        commissions = self._commission * 2
        net = gross - commissions
        risk = entry - trade.stop_price
        self._closed.append(
            TradeRecord(
                trade_id=trade.signal_id,
                symbol=trade.symbol,
                qty=trade.qty,
                entry_ref=trade.entry_ref,
                stop_price=trade.stop_price,
                take_profit_price=trade.take_profit_price,
                entry_filled_at_utc=entry_at,
                entry_price=entry,
                exit_filled_at_utc=when,
                exit_price=price,
                exit_reason=reason,
                gross_pnl=gross,
                commissions=commissions,
                net_pnl=net,
                return_pct=(net / (entry * qty)).quantize(Decimal("0.000001")),
                result_r=((price - entry) / risk).quantize(Decimal("0.0001")) if risk > 0 else None,
                duration_minutes=Decimal((when - entry_at) // _ONE_MINUTE),
                entry_slippage=entry - trade.entry_ref,
            )
        )
        self._counters.exits_by_reason[reason.value] += 1
        self._last_exits[trade.symbol] = when
        del self._open[trade.parent_id]

    async def _sample_equity(self) -> None:
        equity = self._broker.equity()
        exposure_value = sum(
            (abs(p.market_value) for p in await self._broker.get_positions()), Decimal(0)
        )
        exposure = Decimal(0)
        if equity > 0:
            exposure = (exposure_value / equity).quantize(Decimal("0.000001"))
        self._peak_equity = max(self._peak_equity, equity)
        point = EquityPoint(timestamp_utc=self._clock.now_utc(), equity=equity, exposure=exposure)
        if self._curve and self._curve[-1].timestamp_utc == point.timestamp_utc:
            self._curve[-1] = point
        else:
            self._curve.append(point)


# --------------------------------------------------------------------------- periods


def add_months(day: date, months: int) -> date:
    """``day`` shifted by ``months`` calendar months (day clamped to the month end)."""
    index = day.month - 1 + months
    year, month = day.year + index // 12, index % 12 + 1
    return date(year, month, min(day.day, _calendar.monthrange(year, month)[1]))


def walk_forward_windows(
    start: date, end: date, *, train_months: int, test_months: int
) -> list[tuple[date, date, date, date]]:
    """Rolling ``(train_start, train_end, test_start, test_end)`` windows in ``[start, end]``.

    Window ``k`` trains on ``[start + k*test, + train)`` and tests on the following
    ``test_months`` (the last test window is truncated at ``end``).
    """
    if train_months < 1 or test_months < 1:
        raise ValueError("train_months and test_months must be >= 1")
    windows: list[tuple[date, date, date, date]] = []
    k = 0
    while True:
        train_start = add_months(start, k * test_months)
        test_start = add_months(train_start, train_months)
        if test_start > end:
            return windows
        test_end = min(add_months(test_start, test_months) - timedelta(days=1), end)
        windows.append((train_start, test_start - timedelta(days=1), test_start, test_end))
        k += 1


def data_fingerprint(directory: Path) -> str:
    """sha256 over the relative paths and bytes of every CSV of the dataset."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.csv")):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- public API


def _require_runnable(config: AppConfig) -> None:
    pending = pending_backtest_decisions(config)
    if pending:
        raise BacktestRefusedError(pending)


async def simulate(
    config: AppConfig,
    data: BacktestData,
    *,
    starting_cash: Decimal,
    commission_per_fill: Decimal = Decimal(0),
    slippage_multiplier: Decimal = Decimal(1),
) -> SimulationResult:
    """Run one deterministic simulation over ``[backtest.start_date, backtest.end_date]``.

    Raises:
        BacktestRefusedError: a required OWNER_DECISION is ``null``.
        NonRetryableError: invalid data (no sessions, unsupported timeframe...).
    """
    _require_runnable(config)
    simulator = _Simulator(
        config,
        data,
        starting_cash=starting_cash,
        commission_per_fill=commission_per_fill,
        slippage_multiplier=slippage_multiplier,
    )
    return await simulator.run()


def _sessions_between(result: SimulationResult, start: date, end: date) -> int:
    return sum(1 for s in result.sessions if start <= s.session_date <= end)


def _metrics(
    label: str, result: SimulationResult, start: date, end: date, starting_cash: Decimal
) -> PeriodMetrics:
    return compute_metrics(
        label,
        start=start,
        end=end,
        sessions=_sessions_between(result, start, end),
        trades=result.trades,
        curve=result.equity_curve,
        starting_cash=starting_cash,
    )


def default_workers() -> int:
    """Default process count: one per scenario (base + sensitivity), at most the CPUs."""
    return max(1, min(1 + len(SLIPPAGE_MULTIPLIERS), os.cpu_count() or 1))


_WORKER_DATA: dict[tuple[Path, DataFeed], BacktestData] = {}
"""Dataset loaded by a worker process, reused if the pool gives it another scenario."""


def _simulate_scenario(
    config: AppConfig,
    data_dir: Path,
    feed: DataFeed,
    starting_cash: Decimal,
    commission_per_fill: Decimal,
    slippage_multiplier: Decimal,
) -> SimulationResult:
    """Worker-process entry point: load the dataset (once per process) and simulate."""
    key = (data_dir, feed)
    data = _WORKER_DATA.get(key)
    if data is None:
        data = _WORKER_DATA[key] = load_backtest_data(data_dir, feed=feed)
    return asyncio.run(
        simulate(
            config,
            data,
            starting_cash=starting_cash,
            commission_per_fill=commission_per_fill,
            slippage_multiplier=slippage_multiplier,
        )
    )


async def _run_scenarios(
    config: AppConfig,
    data_dir: Path,
    feed: DataFeed,
    *,
    starting_cash: Decimal,
    commission_per_fill: Decimal,
    multipliers: Sequence[Decimal],
    workers: int,
) -> list[SimulationResult]:
    """One simulation per slippage multiplier, returned in ``multipliers`` order."""
    if workers <= 1:
        data = load_backtest_data(data_dir, feed=feed)
        return [
            await simulate(
                config,
                data,
                starting_cash=starting_cash,
                commission_per_fill=commission_per_fill,
                slippage_multiplier=multiplier,
            )
            for multiplier in multipliers
        ]
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=min(workers, len(multipliers))) as pool:
        futures = [
            loop.run_in_executor(
                pool,
                _simulate_scenario,
                config,
                data_dir,
                feed,
                starting_cash,
                commission_per_fill,
                multiplier,
            )
            for multiplier in multipliers
        ]
        return list(await asyncio.gather(*futures))


async def run_backtest_async(
    loaded: LoadedConfig,
    data_dir: Path,
    *,
    starting_cash: Decimal,
    commission_per_fill: Decimal = Decimal(0),
    workers: int = 1,
) -> BacktestReport:
    """Full backtest (sec. 45): base run, splits, walk-forward, slippage sensitivity.

    Args:
        loaded: Validated configuration.
        data_dir: Dataset directory.
        starting_cash: Initial capital.
        commission_per_fill: Flat commission per fill.
        workers: Processes for the base and sensitivity simulations (``1`` = all in
            this process). The report does not depend on it.

    Raises:
        BacktestRefusedError: a required OWNER_DECISION is ``null`` (nothing is run).
    """
    config = loaded.config
    _require_runnable(config)
    backtest = config.backtest
    start, end, oos = backtest.start_date, backtest.end_date, backtest.out_of_sample_start
    train, test = backtest.walk_forward_train_months, backtest.walk_forward_test_months
    feed, holding, primary = (
        config.market_data.feed,
        config.strategy.holding_mode,
        config.strategy.primary_timeframe,
    )
    if (
        start is None
        or end is None
        or oos is None
        or train is None
        or test is None
        or feed is None
        or holding is None
        or primary is None
    ):  # pragma: no cover - guarded by _require_runnable
        raise BacktestRefusedError(pending_backtest_decisions(config))
    base, *runs = await _run_scenarios(
        config,
        data_dir,
        DataFeed(feed),
        starting_cash=starting_cash,
        commission_per_fill=commission_per_fill,
        multipliers=(Decimal(1), *SLIPPAGE_MULTIPLIERS),
        workers=workers,
    )
    scenarios = []
    for multiplier, run in zip(SLIPPAGE_MULTIPLIERS, runs, strict=True):
        scenarios.append(
            SlippageScenario(
                multiplier=multiplier,
                slippage_bps=run.slippage_bps,
                metrics=_metrics("full", run, start, end, starting_cash),
            )
        )
    out_of_sample = _metrics("out_of_sample", base, oos, end, starting_cash)
    walk_forward = tuple(
        WalkForwardWindow(
            index=index,
            train_start=train_start,
            train_end=train_end,
            test=_metrics(f"walk_forward_{index}", base, test_start, test_end, starting_cash),
        )
        for index, (train_start, train_end, test_start, test_end) in enumerate(
            walk_forward_windows(start, end, train_months=train, test_months=test)
        )
    )
    notes = (
        "Walk-forward: no optimisable parameters exist yet, so each test window is "
        "evaluated sequentially on one continuous run; train windows are informative.",
        "Splits come from one continuous run: positions and equity carry over between "
        "periods; a trade belongs to the period of its entry fill.",
        "Slippage sensitivity scales only the simulated fills (x"
        + ", x".join(map(str, SLIPPAGE_MULTIPLIERS))
        + "); sizing keeps risk.slippage_buffer_bps.",
        "Phase 1 runner: the shared check library (Phase 4) and the execution guard / "
        "state machine (Phase 5) are not wired yet (AC-22 pending).",
    )
    return BacktestReport(
        disclaimer=DISCLAIMER,
        config_version=config.config_version,
        strategy_version=config.strategy_version,
        risk_version=config.risk_version,
        config_hash=loaded.config_hash,
        data_fingerprint=data_fingerprint(data_dir),
        symbols=tuple(config.universe.whitelist or ()),
        holding_mode=holding.value,
        primary_timeframe=primary.value,
        start_date=start,
        end_date=end,
        out_of_sample_start=oos,
        starting_cash=starting_cash,
        commission_per_fill=commission_per_fill,
        slippage_bps=base.slippage_bps,
        full=_metrics("full", base, start, end, starting_cash),
        in_sample=_metrics("in_sample", base, start, oos - timedelta(days=1), starting_cash),
        out_of_sample=out_of_sample,
        walk_forward=walk_forward,
        slippage_sensitivity=tuple(scenarios),
        continuation=continuation_check(backtest, out_of_sample),
        counters=base.counters,
        trades=base.trades,
        notes=notes,
    )


def run_backtest(
    loaded: LoadedConfig,
    data_dir: Path,
    *,
    starting_cash: Decimal,
    commission_per_fill: Decimal = Decimal(0),
    workers: int = 1,
) -> BacktestReport:
    """Synchronous wrapper of :func:`run_backtest_async`."""
    return asyncio.run(
        run_backtest_async(
            loaded,
            data_dir,
            starting_cash=starting_cash,
            commission_per_fill=commission_per_fill,
            workers=workers,
        )
    )
