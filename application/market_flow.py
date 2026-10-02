"""Market flow use cases: ``on_minute_bar`` and ``on_closed_bar`` (sec. 8.7, 10, 13).

**Phase 1 subset.** This module wires the deterministic domain (aggregator, strategy,
exit levels, sizing, risk limits) behind the use cases the backtest runs today. It
submits **no** order: the caller (backtest runner now, execution guard in Phase 5)
acts on the typed outcome. Phase 4 will insert the shared check library (sec. 19) and
Phase 5 the execution guard (sec. 20) between :class:`EntryProposal` and submission;
the only safety checks done here are the minimal ones a backtest cannot run without
(entry window / flatten time of sec. 11.4, signal TTL of sec. 13.7, one entry per
symbol at a time), and they will move to the check library.

Flow per closed primary bar (sec. 13, 15.2, 16):

1. Append the bar to the symbol's window (a bar not newer than the last one is a
   duplicate: ``NoAction(DUPLICATE_BAR)``, so a duplicated bar never yields two
   signals, AC-03).
2. Symbol with a position (broker port): ``Strategy.evaluate_position`` ->
   :class:`Hold` or :class:`ExitRequest` (time stop / signal reversal).
3. Symbol with a pending entry: ``NoAction(ENTRY_PENDING)``.
4. Otherwise ``Strategy.evaluate_entry``; on ``BUY``: entry window and TTL ->
   ``compute_exit_levels`` -> ``size_position`` -> ``build_proposed_trade`` ->
   ``evaluate_limits``. Any failure -> :class:`Rejected` with the failing codes; all
   pass -> :class:`EntryProposal`.

Daily primary timeframe (``1Day``, strategy family v2; ``holding_mode = swing`` and no
confirmation timeframe, enforced by the config cross rules):

* There is no aggregator: the caller passes each closed daily bar, stamped to its
  session with :func:`domain.market.session.session_bar`, straight to
  :meth:`MarketFlow.on_closed_bar` after the session close (``on_minute_bar`` refuses
  minute bars, ``on_clock`` closes nothing). Indicators and the ATR stop use daily bars.
* Entry window: the decision is taken after the close and the entry is valid until the
  NEXT session open. The signal expires at ``next session open + signal_ttl_seconds``
  (instead of ``bar_end + signal_ttl_seconds``, sec. 13.7; declared in DECISIONS.md),
  so the next session must be registered before deciding (``SESSION_UNKNOWN``
  otherwise); a decision before the bar's close is ``OUTSIDE_ENTRY_WINDOW``. The
  intraday window of sec. 11.4 (``no_entry_*``, flatten) does not apply.
* Time stop and cooldown count daily bars (sessions). Exits are :class:`ExitRequest`
  as for intraday; the caller executes them at the next open.

Time comes only from ``IClock``; account and positions only from ``IBroker`` (the
broker is the source of truth). Values the live system will read from the database
(week-start and peak equity, pending entries, recorded stops, entry/exit times) are
passed in by the caller as a :class:`TradeBook` (Phase 2+: ``IUnitOfWork``).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Annotated, Final, Literal

from pydantic import Field

from domain.errors import NonRetryableError, StateCriticalError
from domain.market.aggregator import BarAggregator, IngestResult
from domain.market.session import SessionWindowParams, compute_session_windows
from domain.models import (
    Bar,
    CheckResult,
    DataFeed,
    DomainModel,
    ExitReason,
    Money,
    NonNegativeDecimal,
    ProposedTrade,
    SessionDay,
    Signal,
    StrategyAction,
    Symbol,
    Timeframe,
    UtcDatetime,
)
from domain.ports import IBroker, IClock
from domain.risk.exits import (
    ExitLevelsResult,
    ExitParams,
    compute_exit_levels,
    round_down_to_increment,
)
from domain.risk.position_sizer import build_proposed_trade, size_position
from domain.risk.risk_engine import (
    BPS_PER_UNIT,
    Bps,
    PendingEntry,
    PositionStop,
    RiskParams,
    RiskState,
    evaluate_limits,
)
from domain.strategy.strategy import Strategy, StrategyDecision, count_bars_since

__all__ = [
    "ATR_UNAVAILABLE",
    "DUPLICATE_BAR",
    "ENTRY_PENDING",
    "NO_SIGNAL",
    "OUTSIDE_ENTRY_WINDOW",
    "SESSION_UNKNOWN",
    "SIGNAL_EXPIRED",
    "BarOutcome",
    "EntryProposal",
    "ExitRequest",
    "Hold",
    "MarketFlow",
    "MarketFlowParams",
    "MinuteBarResult",
    "NoAction",
    "Rejected",
    "TradeBook",
]

DUPLICATE_BAR: Final = "DUPLICATE_BAR"
ENTRY_PENDING: Final = "ENTRY_PENDING"
NO_SIGNAL: Final = "NO_SIGNAL"
OUTSIDE_ENTRY_WINDOW: Final = "OUTSIDE_ENTRY_WINDOW"
SESSION_UNKNOWN: Final = "SESSION_UNKNOWN"
SIGNAL_EXPIRED: Final = "SIGNAL_EXPIRED"
ATR_UNAVAILABLE: Final = "ATR_UNAVAILABLE"


# --------------------------------------------------------------------------- inputs


class MarketFlowParams(DomainModel):
    """Technical parameters of the flow (built by the composition root from config).

    Attributes:
        symbols: Whitelist the aggregators handle.
        feed: Market data feed of the incoming minute bars.
        bar_close_grace_seconds: ``market_data.bar_close_grace_seconds`` (sec. 10.3.3).
        window_bars: Closed primary bars kept per symbol
            (``market_data.history_warmup_bars``); the flow keeps at least
            ``time_stop_bars + 1`` and ``cooldown_bars + 1`` bars so bar counts stay exact.
        session: Session-window parameters (sec. 11.4).
        limit_entry_offset_bps: ``None`` for market entries; for limit entries the
            limit price ``close * (1 + bps / 10000)`` rounded down to a valid increment
            becomes ``entry_ref`` (sec. 15.2, 23.3).
    """

    symbols: Annotated[tuple[Symbol, ...], Field(min_length=1)]
    feed: DataFeed
    bar_close_grace_seconds: NonNegativeDecimal
    window_bars: int = Field(ge=1)
    session: SessionWindowParams
    limit_entry_offset_bps: Bps | None


class TradeBook(DomainModel):
    """Caller-maintained trading records the flow needs besides the broker state.

    Attributes:
        week_start_equity: Equity at the start of the week (``equity_snapshots``).
        peak_equity: Highest recorded equity (``equity_snapshots``).
        pending_entries: Submitted, not yet filled entries.
        position_stops: Recorded stop of each open position.
        entry_fills: Entry fill time of each open position, per symbol.
        last_exits: Time of the last exit fill per symbol (cooldown).
    """

    week_start_equity: Money
    peak_equity: Money
    pending_entries: tuple[PendingEntry, ...]
    position_stops: tuple[PositionStop, ...]
    entry_fills: Mapping[Symbol, UtcDatetime]
    last_exits: Mapping[Symbol, UtcDatetime]


# --------------------------------------------------------------------------- outcomes


class MinuteBarResult(DomainModel):
    """Result of :meth:`MarketFlow.on_minute_bar`.

    ``closed_bars`` are the primary-timeframe bars closed by this minute or by the
    clock, sorted by ``(bar_start_utc, symbol)``; pass each to ``on_closed_bar``.
    """

    ingest: IngestResult
    closed_bars: tuple[Bar, ...]


class NoAction(DomainModel):
    """Nothing to do for this bar (no signal, duplicate bar, entry pending...)."""

    kind: Literal["NO_ACTION"] = "NO_ACTION"
    symbol: Symbol
    reason: str
    decision: StrategyDecision | None


class Hold(DomainModel):
    """Open position kept (``evaluate_position`` returned ``HOLD``)."""

    kind: Literal["HOLD"] = "HOLD"
    symbol: Symbol
    decision: StrategyDecision


class ExitRequest(DomainModel):
    """The strategy asks for a system exit (procedure 23.6) with ``reason``."""

    kind: Literal["EXIT"] = "EXIT"
    symbol: Symbol
    reason: ExitReason
    decision: StrategyDecision


class Rejected(DomainModel):
    """A ``BUY`` signal that did not survive pricing, sizing, limits or session checks."""

    kind: Literal["REJECTED"] = "REJECTED"
    symbol: Symbol
    codes: tuple[str, ...]
    signal: Signal
    decision: StrategyDecision
    checks: tuple[CheckResult, ...]


class EntryProposal(DomainModel):
    """A priced, sized entry that passed every Phase 1 check (not submitted).

    ``limit_price`` is set only for limit entries (it equals ``trade.entry_ref``).
    ``checks`` holds the exit-level, sizing and limit results (``risk_events``).
    """

    kind: Literal["ENTRY"] = "ENTRY"
    symbol: Symbol
    trade: ProposedTrade
    signal: Signal
    decision: StrategyDecision
    exit_levels: ExitLevelsResult
    limit_price: Money | None
    checks: tuple[CheckResult, ...]


BarOutcome = NoAction | Hold | ExitRequest | Rejected | EntryProposal
"""Typed outcome of :meth:`MarketFlow.on_closed_bar`."""


# --------------------------------------------------------------------------- use cases


class MarketFlow:
    """Stateful market-flow use cases for one strategy and one whitelist.

    State: one aggregator per used intraday timeframe (none for a ``1Day`` primary), a
    bounded window of closed bars per symbol and timeframe, and the registered sessions.
    It is deterministic: no wall clock, no randomness.

    Args:
        strategy: Built strategy (all its OWNER_DECISIONs set).
        exit_params: Risk-engine exit parameters (sec. 16).
        risk_params: Risk parameters (sec. 15).
        params: Technical flow parameters.
        clock: Time source.
        broker: Broker port (account and positions).
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        exit_params: ExitParams,
        risk_params: RiskParams,
        params: MarketFlowParams,
        clock: IClock,
        broker: IBroker,
    ) -> None:
        strategy_params = strategy.params
        primary = strategy_params.primary_timeframe
        if primary is None:  # pragma: no cover - Strategy refuses to build without it
            raise NonRetryableError("strategy has no primary timeframe", code="CONFIG_INVALID")
        self._strategy = strategy
        self._exit_params = exit_params
        self._risk_params = risk_params
        self._params = params
        self._clock = clock
        self._broker = broker
        self._primary = primary
        self._confirmation = strategy_params.confirmation_timeframe
        self._daily = primary is Timeframe.DAY_1
        if self._daily and self._confirmation is not None:
            raise NonRetryableError(
                "a 1Day primary timeframe takes no confirmation timeframe",
                code="CONFIG_INVALID",
            )
        self._signal_ttl = timedelta(seconds=strategy_params.signal_ttl_seconds or 0)
        grace = float(params.bar_close_grace_seconds)
        self._primary_agg: BarAggregator | None = None
        if not self._daily:
            self._primary_agg = BarAggregator(
                timeframe=primary,
                symbols=params.symbols,
                feed=params.feed,
                bar_close_grace_seconds=grace,
            )
        self._confirm_agg = (
            None
            if self._confirmation is None
            else BarAggregator(
                timeframe=self._confirmation,
                symbols=params.symbols,
                feed=params.feed,
                bar_close_grace_seconds=grace,
            )
        )
        size = max(
            params.window_bars,
            (strategy_params.exit.time_stop_bars or 0) + 1,
            (strategy_params.cooldown_bars or 0) + 1,
        )
        self._window_size = size
        self._windows: dict[str, deque[Bar]] = {s: deque(maxlen=size) for s in params.symbols}
        self._confirm_windows: dict[str, deque[Bar]] = {
            s: deque(maxlen=size) for s in params.symbols
        }
        self._sessions: list[SessionDay] = []
        self._session_opens: list[datetime] = []
        self._session_closes: list[datetime] = []

    # ------------------------------------------------------------------ accessors

    @property
    def window_size(self) -> int:
        """Maximum closed primary bars kept per symbol."""
        return self._window_size

    def window(self, symbol: Symbol) -> tuple[Bar, ...]:
        """Closed primary bars of ``symbol`` currently in the window, oldest first."""
        return tuple(self._windows[symbol])

    # ------------------------------------------------------------------ sessions

    def add_session(self, session: SessionDay) -> None:
        """Register the next calendar session before any of its minutes arrive.

        With a ``1Day`` primary, register it before its daily bar is decided and, before
        deciding, the session after it (the signal expires at its open).

        Raises:
            NonRetryableError: code ``SESSION_OUT_OF_ORDER`` (aggregator error for
                intraday primaries) when the session does not start after the previous
                one closes.
        """
        if self._primary_agg is not None:
            self._primary_agg.add_session(session)
        elif self._sessions and session.open_utc < self._sessions[-1].close_utc:
            raise NonRetryableError(
                f"session {session.session_date} must start after the previous session "
                f"{self._sessions[-1].session_date} closes",
                code="SESSION_OUT_OF_ORDER",
            )
        if self._confirm_agg is not None:
            self._confirm_agg.add_session(session)
        self._sessions.append(session)
        self._session_opens.append(session.open_utc)
        self._session_closes.append(session.close_utc)

    def session_of(self, instant: datetime) -> SessionDay | None:
        """Registered session containing ``instant`` (``open <= instant < close``)."""
        # Sessions are registered in order and disjoint (add_session refuses anything
        # else): only the last session opening at or before ``instant`` can contain it.
        index = bisect_right(self._session_opens, instant) - 1
        if index >= 0 and instant < self._sessions[index].close_utc:
            return self._sessions[index]
        return None

    def _session_after(self, instant: datetime) -> SessionDay | None:
        """First registered session opening after ``instant``."""
        index = bisect_right(self._session_opens, instant)
        return self._sessions[index] if index < len(self._sessions) else None

    # ------------------------------------------------------------------ on_minute_bar

    def on_minute_bar(self, bar: Bar) -> MinuteBarResult:
        """Use case ``on_minute_bar`` (sec. 10.3): aggregate one 1-minute bar.

        The bar is ingested by every aggregator, then buckets whose
        ``end + grace`` has passed at ``clock.now_utc()`` are closed. Confirmation bars
        are stored internally (before the primary bars are returned, so a confirmation
        bar closing at the same instant is visible to the primary decision).
        Duplicates and late bars are reported in ``ingest`` and close nothing.

        Raises:
            NonRetryableError: code ``UNSUPPORTED_TIMEFRAME`` with a ``1Day`` primary
                (daily bars are not aggregated from minutes).
        """
        primary_agg = self._primary_agg
        if primary_agg is None:
            raise NonRetryableError(
                "1Day primary timeframe: closed daily bars go straight to on_closed_bar",
                code="UNSUPPORTED_TIMEFRAME",
            )
        ingest = primary_agg.ingest(bar)
        closed = list(ingest.closed_bars)
        if self._confirm_agg is not None:
            self._store_confirmation(self._confirm_agg.ingest(bar).closed_bars)
        closed.extend(self.on_clock())
        closed.sort(key=lambda b: (b.bar_start_utc, b.symbol))
        return MinuteBarResult(ingest=ingest, closed_bars=tuple(closed))

    def on_clock(self) -> tuple[Bar, ...]:
        """Close buckets by time at ``clock.now_utc()`` (call on every clock step).

        Returns the closed primary bars, sorted by ``(bar_start_utc, symbol)``; always
        empty with a ``1Day`` primary (nothing is aggregated).
        """
        if self._primary_agg is None:
            return ()
        now = self._clock.now_utc()
        if self._confirm_agg is not None:
            self._store_confirmation(self._confirm_agg.on_time(now))
        return self._primary_agg.on_time(now)

    def _store_confirmation(self, bars: tuple[Bar, ...]) -> None:
        for closed in bars:
            window = self._confirm_windows[closed.symbol]
            if not window or closed.bar_start_utc > window[-1].bar_start_utc:
                window.append(closed)

    # ------------------------------------------------------------------ on_closed_bar

    def record_closed_bar(self, bar: Bar) -> bool:
        """Append a closed primary bar to its window without deciding (warm-up).

        Returns ``False`` (and changes nothing) when the bar is not newer than the last
        bar of the window, i.e. a duplicate or out-of-order bar.

        Raises:
            NonRetryableError: code ``INVALID_BAR`` for a bar of another timeframe or an
                unknown symbol, or a daily bar not stamped to a registered session;
                ``FEED_MISMATCH`` for a daily bar of another feed than ``params.feed``.
        """
        if bar.timeframe is not self._primary or bar.symbol not in self._windows:
            raise NonRetryableError(
                f"{bar.symbol} {bar.timeframe} is not a primary bar of the whitelist",
                code="INVALID_BAR",
            )
        if self._daily:
            session = self.session_of(bar.bar_start_utc)
            if (
                session is None
                or bar.bar_start_utc != session.open_utc
                or bar.bar_end_utc != session.close_utc
            ):
                raise NonRetryableError(
                    f"{bar.symbol} daily bar {bar.bar_start_utc.isoformat()} is not stamped "
                    "to a registered session (domain.market.session.session_bar)",
                    code="INVALID_BAR",
                )
            if bar.feed is not self._params.feed:  # sec. 10.2.2: same feed live/backtest
                raise NonRetryableError(
                    f"{bar.symbol} daily bar from feed {bar.feed.value}, the strategy feed is "
                    f"{self._params.feed.value}",
                    code="FEED_MISMATCH",
                )
        window = self._windows[bar.symbol]
        if window and bar.bar_start_utc <= window[-1].bar_start_utc:
            return False
        window.append(bar)
        return True

    async def on_closed_bar(self, bar: Bar, *, book: TradeBook) -> BarOutcome:
        """Use case ``on_closed_bar`` (sec. 13): decide on one closed primary bar.

        Raises:
            NonRetryableError: code ``INVALID_BAR`` (see :meth:`record_closed_bar`).
            StateCriticalError: code ``STATE_MISMATCH`` when the broker reports a
                position whose entry fill time is not in ``book``.
        """
        symbol = bar.symbol
        if not self.record_closed_bar(bar):
            return NoAction(symbol=symbol, reason=DUPLICATE_BAR, decision=None)
        window = self._windows[symbol]
        context = self._strategy.build_context(
            tuple(window),
            tuple(self._confirm_windows[symbol]) if self._confirm_agg is not None else None,
            sessions=self._sessions_covering(window[0].bar_start_utc),
        )

        positions = await self._broker.get_positions()
        if any(p.symbol == symbol and p.qty != 0 for p in positions):
            entry_fill = book.entry_fills.get(symbol)
            if entry_fill is None:
                raise StateCriticalError(
                    f"broker reports a {symbol} position with no recorded entry fill",
                    code="STATE_MISMATCH",
                )
            decision = self._strategy.evaluate_position(
                context, bars_held=count_bars_since(window, entry_fill)
            )
            if decision.action is StrategyAction.CLOSE and decision.exit_reason is not None:
                return ExitRequest(symbol=symbol, reason=decision.exit_reason, decision=decision)
            return Hold(symbol=symbol, decision=decision)

        if any(entry.symbol == symbol for entry in book.pending_entries):
            return NoAction(symbol=symbol, reason=ENTRY_PENDING, decision=None)

        last_exit = book.last_exits.get(symbol)
        now = self._clock.now_utc()
        decision = self._strategy.evaluate_entry(
            context,
            bars_since_last_exit=None if last_exit is None else count_bars_since(window, last_exit),
            created_at_utc=now,
            expires_at_utc=self._daily_expiry(bar) if self._daily else None,
        )
        if decision.action is not StrategyAction.BUY or decision.signal is None:
            return NoAction(symbol=symbol, reason=NO_SIGNAL, decision=decision)
        return await self._price_entry(bar, decision, decision.signal, context.latest_atr(), book)

    def _daily_expiry(self, bar: Bar) -> datetime | None:
        """Daily signal expiry: next session open + TTL (``None`` if not registered)."""
        following = self._session_after(bar.bar_start_utc)
        return None if following is None else following.open_utc + self._signal_ttl

    def _sessions_covering(self, since: datetime) -> tuple[SessionDay, ...]:
        """Sessions overlapping the window plus the one before (``gap_pct``)."""
        # Sessions are registered in strictly increasing order (the aggregator refuses
        # anything else), so the first session closing after ``since`` is a bisection.
        first = bisect_right(self._session_closes, since)
        return tuple(self._sessions[max(first - 1, 0) :])

    async def _price_entry(
        self,
        bar: Bar,
        decision: StrategyDecision,
        signal: Signal,
        atr: Decimal | None,
        book: TradeBook,
    ) -> BarOutcome:
        symbol = bar.symbol
        now = self._clock.now_utc()

        def rejected(codes: tuple[str, ...], checks: tuple[CheckResult, ...] = ()) -> Rejected:
            return Rejected(
                symbol=symbol, codes=codes, signal=signal, decision=decision, checks=checks
            )

        session = self.session_of(bar.bar_start_utc)
        if session is None:
            return rejected((SESSION_UNKNOWN,))
        if self._daily:
            # Decided after the close; valid until the next session open (+ TTL).
            if self._session_after(bar.bar_start_utc) is None:
                return rejected((SESSION_UNKNOWN,))
            if now < bar.bar_end_utc:
                return rejected((OUTSIDE_ENTRY_WINDOW,))
        else:
            windows = compute_session_windows(session, self._params.session)
            if not windows.is_entry_window(now) or windows.is_flatten_time(now):
                return rejected((OUTSIDE_ENTRY_WINDOW,))
        if now > signal.expires_at_utc:
            return rejected((SIGNAL_EXPIRED,))
        if atr is None or bar.close is None:
            return rejected((ATR_UNAVAILABLE,))

        limit_price: Decimal | None = None
        entry_ref = bar.close
        offset = self._params.limit_entry_offset_bps
        if offset is not None:
            limit_price = round_down_to_increment(bar.close * (1 + offset / BPS_PER_UNIT))
            entry_ref = limit_price

        levels = compute_exit_levels(entry_ref, atr, self._exit_params)
        if (
            not levels.check.passed
            or levels.stop_price is None
            or (levels.take_profit_price is None)
        ):
            return rejected((levels.check.code,), (levels.check,))

        account = await self._broker.get_account()
        sizing = size_position(
            entry_ref=entry_ref,
            stop_price=levels.stop_price,
            equity=account.equity,
            buying_power=account.buying_power,
            params=self._risk_params,
        )
        if not sizing.check.passed:
            return rejected((sizing.check.code,), (levels.check, sizing.check))
        trade = build_proposed_trade(
            sizing,
            signal_id=signal.signal_id,
            symbol=symbol,
            take_profit_price=levels.take_profit_price,
        )
        state = RiskState(
            equity=account.equity,
            last_equity=account.last_equity,
            week_start_equity=book.week_start_equity,
            peak_equity=book.peak_equity,
            open_positions=tuple(await self._broker.get_positions()),
            pending_entries=book.pending_entries,
            position_stops=book.position_stops,
        )
        limits = evaluate_limits(self._risk_params, state, trade)
        checks = (levels.check, sizing.check, *limits)
        failed = tuple(dict.fromkeys(c.code for c in limits if not c.passed))
        if failed:
            return rejected(failed, checks)
        return EntryProposal(
            symbol=symbol,
            trade=trade,
            signal=signal,
            decision=decision,
            exit_levels=levels,
            limit_price=limit_price,
            checks=checks,
        )
