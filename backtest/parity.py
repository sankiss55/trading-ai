"""Phase 3 exit tool: live daily-path signals vs the backtest on the same sessions (sec. 55).

Phase 3 exit criterion: the signals produced by the live data path over several days
match the backtest's on those same days. For a config with a ``1Day`` primary timeframe
and a dataset (``calendar.csv`` + ``bars/*_1Day.csv``), the last ``K`` sessions that have
a stored next session (default 20; the final stored session is left out because neither
path can price an entry without its next open) are decided twice:

a. **Backtest path**: :class:`backtest.runner._Simulator` over exactly those ``K``
   sessions (same warm-up, execution and accounting as ``simulate(window=...)``). Each
   ``MarketFlow.on_closed_bar`` call is recorded with its stamped bar, the ``TradeBook``
   and broker account/positions it saw, its outcome and the flow window after it.
b. **Live path**: :class:`application.daily_cycle.DailyDataCycle` from a fresh process
   start: ``warm_up`` before the first session, then ``run_session_close`` session by
   session at ``close + data delay`` on a ``SimClock``. Market data is a clocked
   ``HistoricalFeed`` over the stored bars (no lookahead: a bar is known from its session
   close; its label stays the Alpaca midnight New York one) and the calendar is a
   ``StaticCalendar`` of the stored sessions. With ``--live`` the market data is the real
   ``AlpacaMarketData`` (read-only) instead, and the Alpaca calendar is compared with the
   stored one.

Execution is Phase 5: the live path submits nothing, so its broker is a read-only replay
of the backtest's state (positions, account and ``TradeBook`` at each decision). Holding
the execution state equal isolates what Phase 3 delivers: data fetch, validation,
session stamping, calendar registration, warm-up and decision timing.

Compared per session and symbol: the live status (must be ``DECIDED``), the stamped bar,
the whole indicator window after the decision, outcome kind, strategy action, signal id,
signal expiry, rule results (ids, results and values), exit reason, reason / rejection
codes and, for entries, the priced trade, limit price and checks. Exit code ``0`` only on
full parity.

CLI::

    python -m backtest.parity --config research/configs/mr_a1_ibs.yaml \
        --data data/alpaca_sip_daily_split [--sessions 20] [--data-delay-minutes 20] \
        [--starting-cash 100000] [--json parity.json]
    python -m backtest.parity ... --live [--env-file .env]

Exit codes: ``0`` full parity, ``1`` mismatches found, ``2`` refused or invalid input
(pending OWNER_DECISIONs, not a ``1Day`` strategy, dataset or live data unavailable).

Simulation adapters are built here, as in ``backtest/runner.py`` (declared Phase 1
deviation); the Alpaca adapters of ``--live`` come from ``app.container``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final, Literal

from adapters.simulation import HistoricalFeed, SimClock, StaticCalendar
from app.config import (
    AppConfig,
    ConfigError,
    LoadedConfig,
    exit_params,
    load_config,
    pending_backtest_decisions,
    risk_params,
    session_window_params,
)
from application.daily_cycle import (
    DEFAULT_DATA_DELAY_SECONDS,
    DailyCycleParams,
    DailyDataCycle,
    SymbolCycleResult,
    SymbolStatus,
    outcome_fields,
)
from application.market_flow import (
    BarOutcome,
    EntryProposal,
    MarketFlow,
    MarketFlowParams,
    Rejected,
    TradeBook,
)
from backtest.data import BacktestData, load_backtest_data
from backtest.runner import BacktestRefusedError, _Simulator, simulation_gate
from domain.errors import DomainError, NonRetryableError
from domain.models import (
    AccountState,
    Bar,
    BracketOrderRequest,
    BrokerOrder,
    DataFeed,
    DomainModel,
    Position,
    SessionDay,
    SimpleOrderRequest,
    Timeframe,
    TradeUpdate,
)
from domain.ports import IBroker, IClock, IMarketData
from domain.strategy.strategy import Strategy

__all__ = [
    "DEFAULT_SESSIONS",
    "DEFAULT_STARTING_CASH",
    "DecisionRecord",
    "ParityMismatch",
    "ParityReport",
    "ReplayBroker",
    "backtest_decisions",
    "compared_sessions",
    "main",
    "market_flow_for",
    "parity_exit_code",
    "published_daily_feed",
    "render_parity_text",
    "run_parity",
]

DEFAULT_SESSIONS: Final = 20
DEFAULT_STARTING_CASH: Final = Decimal(100000)
"""Only sizes the replayed account: both paths see the same state, parity ignores it."""
_SUBMISSION_FORBIDDEN: Final = "SUBMISSION_FORBIDDEN"

CalendarSource = Callable[[date, date], Awaitable[Sequence[SessionDay]]]
"""Reference calendar for ``[start, end]`` (``--live``: ``AlpacaCalendar.get_sessions``)."""


# --------------------------------------------------------------------------- models


class ParityMismatch(DomainModel):
    """One difference between the backtest and the live path."""

    session_date: date
    symbol: str
    field: str
    backtest: str
    live: str


class ParityReport(DomainModel):
    """Result of :func:`run_parity`.

    Attributes:
        mode: ``dataset`` (stored bars on both paths) or ``live`` (Alpaca market data).
        sessions: The compared session dates.
        warm_up_sessions: Sessions the live path warmed up with.
        decisions_compared: ``(session, symbol)`` pairs compared.
        window_bars_compared: Bars compared in the windows after each decision.
        outcome_counts: Backtest outcome kinds over the compared decisions.
        signals: Backtest BUY decisions over the compared sessions.
        calendar_sessions_compared: Reference calendar sessions compared (``--live``).
        mismatches: Every difference found.
    """

    mode: Literal["dataset", "live"]
    config_version: str
    strategy_version: str
    config_hash: str | None
    data_dir: str
    symbols: tuple[str, ...]
    sessions: tuple[date, ...]
    data_delay_seconds: int
    warm_up_sessions: int
    decisions_compared: int
    window_bars_compared: int
    outcome_counts: dict[str, int]
    signals: int
    calendar_sessions_compared: int
    mismatches: tuple[ParityMismatch, ...]

    @property
    def passed(self) -> bool:
        """``True`` on full parity."""
        return not self.mismatches


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One backtest ``on_closed_bar`` call and the state it saw."""

    session_date: date
    symbol: str
    bar: Bar
    outcome: BarOutcome
    book: TradeBook
    account: AccountState
    positions: tuple[Position, ...]
    window_before: tuple[Bar, ...]
    window: tuple[Bar, ...]


# --------------------------------------------------------------------------- wiring


def market_flow_for(
    config: AppConfig,
    *,
    clock: IClock,
    broker: IBroker,
    sessions: Sequence[SessionDay],
    liquidity_data: IMarketData,
) -> MarketFlow:
    """The ``MarketFlow`` the backtest runner builds for ``config`` (same parameters).

    The check-library gate is :func:`backtest.runner.simulation_gate` over ``sessions``
    (the calendar) with ``liquidity_data`` (the live path's own market data) as the
    daily-bar source of the liquidity filter.

    Raises:
        BacktestRefusedError: a required OWNER_DECISION is ``null``.
    """
    whitelist, feed = config.universe.whitelist, config.market_data.feed
    if whitelist is None or feed is None:
        raise BacktestRefusedError(pending_backtest_decisions(config) or ("universe.whitelist",))
    offset = (
        config.execution.limit_entry_offset_bps
        if config.execution.entry_order_type == "limit"
        else None
    )
    return MarketFlow(
        strategy=Strategy(config.strategy, strategy_version=config.strategy_version),
        exit_params=exit_params(config),
        risk_params=risk_params(config),
        params=MarketFlowParams(
            symbols=tuple(whitelist),
            feed=feed,
            bar_close_grace_seconds=config.market_data.bar_close_grace_seconds,
            window_bars=config.market_data.history_warmup_bars,
            session=session_window_params(config),
            limit_entry_offset_bps=offset,
        ),
        clock=clock,
        broker=broker,
        gate=simulation_gate(config, sessions=sessions, clock=clock, liquidity_data=liquidity_data),
    )


class ReplayBroker:
    """Read-only ``IBroker`` serving the backtest's account and positions.

    The live path submits nothing in Phase 3: :meth:`arm` sets the state the next
    decision sees; every order method refuses (``SUBMISSION_FORBIDDEN``).
    """

    def __init__(self, account: AccountState) -> None:
        self._account = account
        self._positions: tuple[Position, ...] = ()

    def arm(self, account: AccountState, positions: Sequence[Position]) -> None:
        """State served until the next call."""
        self._account = account
        self._positions = tuple(positions)

    async def get_account(self) -> AccountState:
        """The armed account."""
        return self._account

    async def get_positions(self) -> list[Position]:
        """The armed positions."""
        return list(self._positions)

    async def get_open_orders(self) -> list[BrokerOrder]:
        """No order is ever open."""
        return []

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """No order is ever known."""
        return None

    async def submit_bracket(self, request: BracketOrderRequest) -> BrokerOrder:
        """Refused: the parity live path never submits."""
        raise NonRetryableError("parity replay broker never submits", code=_SUBMISSION_FORBIDDEN)

    async def submit_simple(self, request: SimpleOrderRequest) -> BrokerOrder:
        """Refused: the parity live path never submits."""
        raise NonRetryableError("parity replay broker never submits", code=_SUBMISSION_FORBIDDEN)

    async def cancel_order(self, order_id: str) -> None:
        """Refused: the parity live path never cancels."""
        raise NonRetryableError("parity replay broker never cancels", code=_SUBMISSION_FORBIDDEN)

    def stream_trade_updates(self) -> AsyncIterator[TradeUpdate]:
        """No trade update stream.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError("the parity replay broker has no trade updates")


def compared_sessions(data: BacktestData, count: int) -> tuple[SessionDay, ...]:
    """The last ``count`` stored sessions that have a stored next session.

    Raises:
        NonRetryableError: ``NO_SESSIONS`` when the dataset is shorter than ``count + 1``.
    """
    if count < 1:
        raise NonRetryableError("at least one session must be compared", code="NO_SESSIONS")
    if len(data.sessions) < count + 1:
        raise NonRetryableError(
            f"the dataset has {len(data.sessions)} sessions; {count + 1} are needed",
            code="NO_SESSIONS",
        )
    return data.sessions[-(count + 1) : -1]


async def published_daily_feed(
    data: BacktestData, symbols: Sequence[str], clock: SimClock
) -> HistoricalFeed:
    """Clocked ``HistoricalFeed`` that publishes each stored daily bar at its session close.

    The stored convention (``bar_end = label + 1 day``) would hide a bar until midnight
    New York, hours after Alpaca serves it; here a bar becomes known at its session close
    (the cycle still waits its data delay). Labels and prices are unchanged; bars after
    ``clock.now_utc()`` stay unknown (no lookahead).
    """
    by_date = {session.session_date: session for session in data.sessions}
    first, last = data.sessions[0].session_date, data.sessions[-1].session_date
    published: list[Bar] = []
    for symbol in symbols:
        for bar in await data.feed.get_daily_bars(symbol, first, last):
            session = by_date.get(bar.bar_start_utc.date())
            if session is not None and session.close_utc > bar.bar_start_utc:
                bar = bar.model_copy(update={"bar_end_utc": session.close_utc})
            published.append(bar)
    return HistoricalFeed(published, clock=clock)


# --------------------------------------------------------------------------- backtest path


async def backtest_decisions(
    config: AppConfig,
    data: BacktestData,
    sessions: Sequence[SessionDay],
    *,
    starting_cash: Decimal,
) -> dict[tuple[date, str], DecisionRecord]:
    """Run the backtest over ``sessions`` and record every daily decision.

    Raises:
        BacktestRefusedError: a required OWNER_DECISION is ``null``.
        NonRetryableError: invalid data or configuration.
    """
    pending = pending_backtest_decisions(config)
    if pending:
        raise BacktestRefusedError(pending)
    simulator = _Simulator(
        config,
        data,
        starting_cash=starting_cash,
        commission_per_fill=Decimal(0),
        slippage_multiplier=Decimal(1),
        window=(sessions[0].session_date, sessions[-1].session_date),
    )
    flow, broker = simulator._flow, simulator._broker
    decide = flow.on_closed_bar
    records: dict[tuple[date, str], DecisionRecord] = {}

    async def recording(bar: Bar, *, book: TradeBook) -> BarOutcome:
        account = await broker.get_account()
        positions = tuple(await broker.get_positions())
        before = flow.window(bar.symbol)
        outcome = await decide(bar, book=book)
        session = flow.session_of(bar.bar_start_utc)
        if session is None:  # pragma: no cover - daily bars are stamped to a session
            raise NonRetryableError("decided bar outside every session", code="INVALID_BAR")
        records[(session.session_date, bar.symbol)] = DecisionRecord(
            session_date=session.session_date,
            symbol=bar.symbol,
            bar=bar,
            outcome=outcome,
            book=book,
            account=account,
            positions=positions,
            window_before=before,
            window=flow.window(bar.symbol),
        )
        return outcome

    # Observation hook: the runner exposes no per-decision callback yet.
    flow.on_closed_bar = recording  # type: ignore[method-assign]
    await simulator.run()
    return records


# --------------------------------------------------------------------------- live path


@dataclass(frozen=True, slots=True)
class _LiveRun:
    warm_up_sessions: int
    results: dict[tuple[date, str], SymbolCycleResult]
    windows: dict[tuple[date, str], tuple[Bar, ...]]


async def _live_decisions(
    config: AppConfig,
    data: BacktestData,
    sessions: Sequence[SessionDay],
    records: Mapping[tuple[date, str], DecisionRecord],
    *,
    market_data: IMarketData | None,
    starting_cash: Decimal,
    data_delay_seconds: int,
) -> _LiveRun:
    whitelist, feed = config.universe.whitelist, config.market_data.feed
    if whitelist is None or feed is None:  # pragma: no cover - checked by the backtest path
        raise BacktestRefusedError(pending_backtest_decisions(config))
    symbols = tuple(whitelist)
    clock = SimClock(sessions[0].open_utc)  # process start: the first compared session
    source = market_data or await published_daily_feed(data, symbols, clock)
    flat = AccountState(
        equity=starting_cash, last_equity=starting_cash, buying_power=starting_cash, status="ACTIVE"
    )
    broker = ReplayBroker(flat)
    flow = market_flow_for(
        config, clock=clock, broker=broker, sessions=data.sessions, liquidity_data=source
    )
    cycle = DailyDataCycle(
        flow=flow,
        market_data=source,
        calendar=StaticCalendar(data.sessions, clock),
        clock=clock,
        params=DailyCycleParams(symbols=symbols, feed=feed, data_delay_seconds=data_delay_seconds),
    )
    warm = await cycle.warm_up(sessions[0])
    results: dict[tuple[date, str], SymbolCycleResult] = {}
    windows: dict[tuple[date, str], tuple[Bar, ...]] = {}
    for session in sessions:
        day = session.session_date

        async def book_for(symbol: str, day: date = day) -> TradeBook:
            record = records.get((day, symbol))
            if record is None:
                broker.arm(flat, ())
                return TradeBook(
                    week_start_equity=starting_cash,
                    peak_equity=starting_cash,
                    pending_entries=(),
                    position_stops=(),
                    entry_fills={},
                    last_exits={},
                    executed_signal_ids=frozenset(),
                )
            broker.arm(record.account, record.positions)
            return record.book

        clock.advance_to(cycle.data_ready_at(session))
        report = await cycle.run_session_close(session, book=book_for)
        for item in report.symbols:
            results[(day, item.symbol)] = item
            windows[(day, item.symbol)] = flow.window(item.symbol)
    return _LiveRun(warm_up_sessions=len(warm.sessions), results=results, windows=windows)


# --------------------------------------------------------------------------- comparison


def _bar_text(bar: Bar | None) -> str:
    if bar is None:
        return "none"
    return (
        f"{bar.bar_start_utc.isoformat()} o={bar.open} h={bar.high} l={bar.low} c={bar.close} "
        f"v={bar.volume} {bar.status.value} {bar.feed.value}"
    )


def _window_difference(expected: Sequence[Bar], actual: Sequence[Bar]) -> tuple[str, str] | None:
    if tuple(expected) == tuple(actual):
        return None
    if len(expected) != len(actual):
        return f"{len(expected)} bars", f"{len(actual)} bars"
    index = next(i for i, (a, b) in enumerate(zip(expected, actual, strict=True)) if a != b)
    return f"[{index}] {_bar_text(expected[index])}", f"[{index}] {_bar_text(actual[index])}"


def _compare(
    record: DecisionRecord, result: SymbolCycleResult, window: Sequence[Bar]
) -> list[ParityMismatch]:
    found: list[ParityMismatch] = []

    def differ(field: str, backtest: object, live: object) -> None:
        if backtest != live:
            found.append(
                ParityMismatch(
                    session_date=record.session_date,
                    symbol=record.symbol,
                    field=field,
                    backtest=str(backtest),
                    live=str(live),
                )
            )

    if result.status is not SymbolStatus.DECIDED or result.outcome is None:
        differ("status", SymbolStatus.DECIDED.value, f"{result.status.value} {result.reasons}")
        return found
    if record.bar != result.bar:
        differ("bar", _bar_text(record.bar), _bar_text(result.bar))
    window_diff = _window_difference(record.window, window)
    if window_diff is not None:
        differ("window", *window_diff)
    expected, actual = record.outcome, result.outcome
    differ("kind", expected.kind, actual.kind)
    action, signal_id, rules, reasons = outcome_fields(expected)
    differ("action", action, result.action)
    differ("signal_id", signal_id, result.signal_id)
    differ("rule_results", rules, result.rule_results)
    differ("reasons", reasons, result.reasons)
    expected_decision, actual_decision = expected.decision, actual.decision
    if expected_decision is not None and actual_decision is not None:
        differ("bar_status", expected_decision.bar_status, actual_decision.bar_status)
        differ("exit_reason", expected_decision.exit_reason, actual_decision.exit_reason)
        if expected_decision.signal is not None and actual_decision.signal is not None:
            differ(
                "signal_expiry",
                expected_decision.signal.expires_at_utc,
                actual_decision.signal.expires_at_utc,
            )
    if isinstance(expected, EntryProposal) and isinstance(actual, EntryProposal):
        differ("trade", expected.trade, actual.trade)
        differ("limit_price", expected.limit_price, actual.limit_price)
        differ("checks", expected.checks, actual.checks)
    elif isinstance(expected, Rejected) and isinstance(actual, Rejected):
        differ("checks", expected.checks, actual.checks)
    return found


async def _compare_calendar(
    data: BacktestData, first: date, last: date, reference: CalendarSource
) -> tuple[int, list[ParityMismatch]]:
    stored = [s for s in data.sessions if first <= s.session_date <= last]
    fetched = list(await reference(first, last))
    stored_by_day = {s.session_date: s for s in stored}
    fetched_by_day = {s.session_date: s for s in fetched}
    found = [
        ParityMismatch(
            session_date=day,
            symbol="*",
            field="calendar",
            backtest=str(stored_by_day.get(day)),
            live=str(fetched_by_day.get(day)),
        )
        for day in sorted(set(stored_by_day) | set(fetched_by_day))
        if stored_by_day.get(day) != fetched_by_day.get(day)
    ]
    return len(fetched), found


async def run_parity(
    loaded: LoadedConfig,
    data: BacktestData,
    *,
    sessions: int = DEFAULT_SESSIONS,
    starting_cash: Decimal = DEFAULT_STARTING_CASH,
    data_delay_seconds: int | None = None,
    market_data: IMarketData | None = None,
    reference_calendar: CalendarSource | None = None,
) -> ParityReport:
    """Decide the last ``sessions`` sessions on both paths and compare them.

    Args:
        loaded: Validated configuration (``1Day`` primary timeframe).
        data: Dataset (the backtest path always reads it).
        sessions: Sessions compared (see :func:`compared_sessions`).
        starting_cash: Cash of the backtest account (replayed on the live path).
        data_delay_seconds: Live data delay (default: the cycle's default).
        market_data: Live-path market data (default: the stored bars, published at
            each session close). ``AlpacaMarketData`` for ``--live``.
        reference_calendar: Calendar compared with the stored sessions over the
            warm-up and compared range (``--live``).

    Raises:
        BacktestRefusedError: a required OWNER_DECISION is ``null``.
        NonRetryableError: not a ``1Day`` strategy, or invalid data.
    """
    config = loaded.config
    if config.strategy.primary_timeframe is not Timeframe.DAY_1:
        raise NonRetryableError(
            "the parity tool covers 1Day strategies only (Phase 3 scope)",
            code="UNSUPPORTED_TIMEFRAME",
        )
    compared = compared_sessions(data, sessions)
    delay = DEFAULT_DATA_DELAY_SECONDS if data_delay_seconds is None else data_delay_seconds
    records = await backtest_decisions(config, data, compared, starting_cash=starting_cash)
    live = await _live_decisions(
        config,
        data,
        compared,
        records,
        market_data=market_data,
        starting_cash=starting_cash,
        data_delay_seconds=delay,
    )
    mismatches: list[ParityMismatch] = []
    window_bars = 0
    for key in sorted(set(records) | set(live.results)):
        record, result = records.get(key), live.results.get(key)
        if record is None or result is None:
            mismatches.append(
                ParityMismatch(
                    session_date=key[0],
                    symbol=key[1],
                    field="presence",
                    backtest="decided" if record is not None else "absent",
                    live="present" if result is not None else "absent",
                )
            )
            continue
        window_bars += len(record.window)
        mismatches.extend(_compare(record, result, live.windows[key]))
    calendar_count = 0
    if reference_calendar is not None:
        first_day = compared[0].session_date
        earlier = [s for s in data.sessions if s.session_date < first_day]
        start = earlier[-live.warm_up_sessions].session_date if live.warm_up_sessions else first_day
        index = data.sessions.index(compared[-1])
        calendar_count, calendar_mismatches = await _compare_calendar(
            data, start, data.sessions[index + 1].session_date, reference_calendar
        )
        mismatches.extend(calendar_mismatches)
    counts: dict[str, int] = {}
    for record in records.values():
        counts[record.outcome.kind] = counts.get(record.outcome.kind, 0) + 1
    signals = sum(1 for record in records.values() if outcome_fields(record.outcome)[1])
    return ParityReport(
        mode="dataset" if market_data is None else "live",
        config_version=config.config_version,
        strategy_version=config.strategy_version,
        config_hash=loaded.config_hash,
        data_dir=str(data.directory),
        symbols=tuple(config.universe.whitelist or ()),
        sessions=tuple(s.session_date for s in compared),
        data_delay_seconds=delay,
        warm_up_sessions=live.warm_up_sessions,
        decisions_compared=len(records),
        window_bars_compared=window_bars,
        outcome_counts=dict(sorted(counts.items())),
        signals=signals,
        calendar_sessions_compared=calendar_count,
        mismatches=tuple(mismatches),
    )


def parity_exit_code(report: ParityReport) -> int:
    """``0`` on full parity, ``1`` otherwise."""
    return 0 if report.passed else 1


def render_parity_text(report: ParityReport, *, max_mismatches: int = 50) -> str:
    """Human-readable summary of ``report``."""
    lines = [
        f"Signal parity ({report.mode}): {'PASS' if report.passed else 'FAIL'}",
        f"config {report.config_version} strategy {report.strategy_version} "
        f"hash {report.config_hash}",
        f"data {report.data_dir}; symbols {', '.join(report.symbols)}",
        f"sessions {report.sessions[0]}..{report.sessions[-1]} ({len(report.sessions)}); "
        f"warm-up {report.warm_up_sessions} sessions; data delay {report.data_delay_seconds} s",
        f"decisions compared {report.decisions_compared}; window bars compared "
        f"{report.window_bars_compared}; BUY signals {report.signals}",
        "backtest outcomes: "
        + (", ".join(f"{k} {v}" for k, v in report.outcome_counts.items()) or "none"),
    ]
    if report.calendar_sessions_compared:
        lines.append(f"reference calendar sessions compared {report.calendar_sessions_compared}")
    lines.append(f"mismatches {len(report.mismatches)}")
    for item in report.mismatches[:max_mismatches]:
        lines.append(
            f"  {item.session_date} {item.symbol} {item.field}: backtest={item.backtest} "
            f"live={item.live}"
        )
    if len(report.mismatches) > max_mismatches:
        lines.append(f"  ... {len(report.mismatches) - max_mismatches} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected at least 1, got {text!r}")
    return value


def _minutes(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid minutes {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected minutes >= 0, got {text!r}")
    return value


def _cash(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid amount {text!r}") from exc
    if not value.is_finite() or value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive amount, got {text!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.parity",
        description="Phase 3 signal parity: live daily data path vs backtest (sec. 55).",
    )
    parser.add_argument("--config", type=Path, required=True, help="config with a 1Day strategy")
    parser.add_argument("--data", type=Path, required=True, help="dataset directory")
    parser.add_argument(
        "--sessions", type=_positive_int, default=DEFAULT_SESSIONS, help="sessions compared"
    )
    parser.add_argument(
        "--data-delay-minutes",
        type=_minutes,
        default=None,
        help="wait after the close before deciding (default 20)",
    )
    parser.add_argument(
        "--starting-cash",
        type=_cash,
        default=DEFAULT_STARTING_CASH,
        help="backtest account cash (replayed on the live path; parity does not depend on it)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="live path on the real Alpaca market data and calendar (read-only, network)",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="secrets for --live")
    parser.add_argument("--json", type=Path, default=None, help="also write the JSON report")
    return parser


def _dataset_adjustment(directory: Path) -> str | None:
    path = directory / "manifest.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("adjustment")
    return value if isinstance(value, str) else None


async def _run_live(
    loaded: LoadedConfig, data: BacktestData, args: argparse.Namespace
) -> ParityReport:
    """``--live``: Alpaca adapters from the composition root, read-only."""
    from adapters.alpaca import AlpacaCalendar
    from app.container import build_clock, build_exchange_adapters, close_quietly
    from app.secrets import AppEnv, load_secrets

    secrets = load_secrets(args.env_file)
    if secrets.app_env is AppEnv.TEST:
        raise NonRetryableError("--live needs APP_ENV paper or dev", code="LIVE_DATA_UNAVAILABLE")
    exchange = build_exchange_adapters(
        secrets.app_env, secrets, loaded.config, build_clock(secrets.app_env)
    )
    try:
        calendar = exchange.calendar
        if exchange.market_data is None or not isinstance(calendar, AlpacaCalendar):
            raise NonRetryableError(
                "Alpaca market data is not configured (market_data.feed/adjustment)",
                code="LIVE_DATA_UNAVAILABLE",
            )
        return await run_parity(
            loaded,
            data,
            sessions=args.sessions,
            starting_cash=args.starting_cash,
            data_delay_seconds=_delay_seconds(args),
            market_data=exchange.market_data,
            reference_calendar=calendar.get_sessions,
        )
    finally:
        for component in (exchange.broker, exchange.calendar, exchange.market_data):
            if component is not None:
                await close_quietly(component)


def _delay_seconds(args: argparse.Namespace) -> int | None:
    minutes: int | None = args.data_delay_minutes
    return None if minutes is None else minutes * 60


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parity CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        loaded = load_config(args.config)
        config = loaded.config
        if config.market_data.feed is None:
            raise BacktestRefusedError(("market_data.feed",))
        adjustment = _dataset_adjustment(args.data)
        if adjustment is not None and adjustment != config.market_data.adjustment:
            raise NonRetryableError(
                f"dataset adjustment {adjustment!r} != market_data.adjustment "
                f"{config.market_data.adjustment!r} (sec. 44: same convention live/backtest)",
                code="DATASET_ADJUSTMENT_MISMATCH",
            )
        data = load_backtest_data(args.data, feed=DataFeed(config.market_data.feed))
        if args.live:
            report = asyncio.run(_run_live(loaded, data, args))
        else:
            report = asyncio.run(
                run_parity(
                    loaded,
                    data,
                    sessions=args.sessions,
                    starting_cash=args.starting_cash,
                    data_delay_seconds=_delay_seconds(args),
                )
            )
    except (ConfigError, DomainError, OSError, ValueError) as exc:
        print(f"parity refused: {exc}", file=sys.stderr)
        return 2
    print(render_parity_text(report))
    if args.json is not None:
        args.json.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return parity_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
