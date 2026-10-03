"""Daily data cycle: live daily bars -> ``MarketFlow`` once per session (Phase 3).

Scope (Phase 3, declared in ``docs/RUNBOOK.md`` "Daily data cycle"): live market data for
strategies with a ``1Day`` primary timeframe only (strategy family v2). There is no
websocket minute stream: the decision is taken once per session, after the close, when
the SIP daily bar of the session is final (the Alpaca free plan serves SIP history only
when it is at least 15 minutes old, so the cycle waits ``data_delay_seconds``, default
20 minutes, after the calendar close). Nothing is submitted: the caller receives a typed
:class:`CycleReport` and execution stays Phase 5.

Use cases (ports only: ``IMarketData``, ``IMarketCalendar``, ``IClock``; sec. 8.3, 8.7):

* :meth:`DailyDataCycle.warm_up` (sec. 25, process start): the ``N`` sessions before the
  first session to decide (``N = MarketFlow.window_size``, the window the backtest warms
  up with) are registered and their daily bars recorded as history, with no decision.
* :meth:`DailyDataCycle.run_session_close` (sec. 10, 13): after ``close + data delay``
  of the session that just closed, every whitelisted symbol, in sorted order (the
  backtest order): fetch ``IMarketData.get_daily_bars(symbol, day, day)``, validate,
  stamp with :func:`domain.market.session.session_bar` and pass the bar to
  :meth:`MarketFlow.on_closed_bar`.

History replay (warm-up and gap refill, sec. 10.4) is one code path, identical to the
backtest warm-up: sessions are registered in calendar order (contiguously, so ``gap_pct``
sees the real previous session) and each session's bar is recorded with
``MarketFlow.record_closed_bar``; a session without a bar becomes an EMPTY session bar
(the backtest does the same, so time stops and cooldowns still count it). A history bar
that is present but invalid stops the replay of that symbol: it is neither decided nor
advanced, and the next cycle retries from its last recorded session.

Validation of the decision bar (sec. 10.6, fail closed; a failing symbol gets **no
decision** this session, the reason is recorded, the other symbols are decided):

=================  ==============================================================
status             cause
=================  ==============================================================
``STALE``          before ``close + data_delay_seconds``: the cycle refuses to
                   decide at all (no state change)
``MISSING``        no bar labelled with the session date
``INCOMPLETE``     the bar is not ``COMPLETE``
``FEED_MISMATCH``  the bar comes from another feed than the strategy feed
``INVALID``        several bars for the date, another timeframe, OHLC not sane, or a
                   label that does not belong to the session
``FETCH_FAILED``   the market-data adapter raised (retries exhausted, bad data)
``BACKFILL_FAILED`` the history before the session could not be completed
=================  ==============================================================

A missing or invalid decision bar is never recorded, so a bar published late is picked
up as history by the next cycle (gap refill) and the window converges to the backtest's.

Gaps (sec. 10.4, process down): sessions between the last processed session and the
session being decided are replayed as history before deciding; only the latest closed
session is decided (``NOT_LATEST_SESSION`` when a later session is already final).

Calendar (sec. 11): sessions come only from ``IMarketCalendar.get_session``. Warm-up
scans the calendar day by day backwards; the Alpaca adapter caches days, so the
composition root should prefetch the range (``AlpacaCalendar.get_sessions``) before
``warm_up`` to keep it to one request.

The cycle owns its ``MarketFlow``: nothing else may register sessions or feed bars to it.
Calls are serialized by a lock. Time comes only from ``IClock``.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_right
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Final

from pydantic import Field

from application.market_flow import (
    BarOutcome,
    EntryProposal,
    ExitRequest,
    MarketFlow,
    NoAction,
    Rejected,
    TradeBook,
)
from domain.errors import NonRetryableError, RetryableError
from domain.market.session import SessionBarMismatchError, empty_session_bar, session_bar
from domain.models import (
    Bar,
    BarStatus,
    DataFeed,
    DomainModel,
    RuleResult,
    SessionDay,
    StrategyAction,
    Symbol,
    Timeframe,
    UtcDatetime,
)
from domain.ports import IClock, IMarketCalendar, IMarketData

__all__ = [
    "ALREADY_STARTED",
    "BACKFILL_EMPTY",
    "CALENDAR_UNAVAILABLE",
    "DATA_NOT_FINAL",
    "DEFAULT_DATA_DELAY_SECONDS",
    "DUPLICATE_DAILY_BAR",
    "FEED_MISMATCH",
    "INCOMPLETE_BAR",
    "INVALID_OHLC",
    "MISSING_BAR",
    "NEXT_SESSION_UNKNOWN",
    "NOT_LATEST_SESSION",
    "SESSION_BAR_MISMATCH",
    "SESSION_NOT_IN_CALENDAR",
    "SESSION_OUT_OF_ORDER",
    "WRONG_TIMEFRAME",
    "BookSource",
    "CycleReport",
    "CycleStatus",
    "DailyCycleParams",
    "DailyDataCycle",
    "SymbolCycleResult",
    "SymbolHistory",
    "SymbolStatus",
    "WarmUpReport",
    "outcome_fields",
]

DEFAULT_DATA_DELAY_SECONDS: Final = 20 * 60
"""Wait after the calendar close before the daily bar is final (SIP history >= 15 min)."""

# Cycle-level reasons.
DATA_NOT_FINAL: Final = "DATA_NOT_FINAL"
NOT_LATEST_SESSION: Final = "NOT_LATEST_SESSION"
NEXT_SESSION_UNKNOWN: Final = "NEXT_SESSION_UNKNOWN"
SESSION_NOT_IN_CALENDAR: Final = "SESSION_NOT_IN_CALENDAR"
SESSION_OUT_OF_ORDER: Final = "SESSION_OUT_OF_ORDER"
CALENDAR_UNAVAILABLE: Final = "CALENDAR_UNAVAILABLE"
ALREADY_STARTED: Final = "ALREADY_STARTED"

# Bar-level reasons.
MISSING_BAR: Final = "MISSING_BAR"
DUPLICATE_DAILY_BAR: Final = "DUPLICATE_DAILY_BAR"
WRONG_TIMEFRAME: Final = "WRONG_TIMEFRAME"
FEED_MISMATCH: Final = "FEED_MISMATCH"
INCOMPLETE_BAR: Final = "INCOMPLETE_BAR"
INVALID_OHLC: Final = "INVALID_OHLC"
SESSION_BAR_MISMATCH: Final = "SESSION_BAR_MISMATCH"
BACKFILL_EMPTY: Final = "BACKFILL_EMPTY"

_NEXT_SESSION_SCAN_DAYS: Final = 14
"""Calendar days scanned for the next session (longest market closure + weekends)."""


# --------------------------------------------------------------------------- models


class DailyCycleParams(DomainModel):
    """Parameters of the daily cycle (built by the composition root from config).

    Attributes:
        symbols: Whitelist (decided in sorted order, like the backtest).
        feed: Strategy feed (``market_data.feed``); a bar of another feed is refused.
        data_delay_seconds: Wait after the calendar close before the daily bar is final.
    """

    symbols: Annotated[tuple[Symbol, ...], Field(min_length=1)]
    feed: DataFeed
    data_delay_seconds: int = Field(default=DEFAULT_DATA_DELAY_SECONDS, ge=0)


class SymbolStatus(StrEnum):
    """Per-symbol result of one cycle (only ``DECIDED`` reached ``MarketFlow``)."""

    DECIDED = "DECIDED"
    STALE = "STALE"
    MISSING = "MISSING"
    INCOMPLETE = "INCOMPLETE"
    FEED_MISMATCH = "FEED_MISMATCH"
    INVALID = "INVALID"
    FETCH_FAILED = "FETCH_FAILED"
    BACKFILL_FAILED = "BACKFILL_FAILED"
    NOT_EVALUATED = "NOT_EVALUATED"


class CycleStatus(StrEnum):
    """``COMPLETED``: every symbol decided; ``PARTIAL``: some failed closed;
    ``REFUSED``: nothing decided and no state changed."""

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    REFUSED = "REFUSED"


class SymbolHistory(DomainModel):
    """History replayed for one symbol (warm-up or gap refill).

    Attributes:
        symbol: Symbol.
        recorded: Session dates recorded (EMPTY bars included), oldest first.
        empty_sessions: Recorded dates without a bar (EMPTY session bars).
        failed_session: First date that could not be recorded (replay stopped there).
        reasons: Codes, e.g. ``BACKFILL_EMPTY 2026-06-01`` or ``FEED_MISMATCH 2026-06-02``.
    """

    symbol: Symbol
    recorded: tuple[date, ...] = ()
    empty_sessions: tuple[date, ...] = ()
    failed_session: date | None = None
    reasons: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """``True`` when every requested session was recorded."""
        return self.failed_session is None


class SymbolCycleResult(DomainModel):
    """Result of one symbol in :meth:`DailyDataCycle.run_session_close`.

    Attributes:
        symbol: Symbol.
        status: See :class:`SymbolStatus`.
        bar: The session-stamped bar passed to ``MarketFlow`` (``DECIDED`` only).
        outcome: Typed ``MarketFlow`` outcome (``DECIDED`` only).
        outcome_kind: ``outcome.kind`` (``NO_ACTION``, ``HOLD``, ``EXIT``, ``REJECTED``...).
        action: Strategy action (``None`` when the flow decided without the strategy,
            e.g. ``ENTRY_PENDING`` or ``DUPLICATE_BAR``).
        signal_id: Deterministic signal id of a BUY (sec. 13.7).
        rule_results: Every evaluated rule of the strategy decision.
        reasons: Data-validation codes, or the outcome's reason / rejection codes.
        history: History replayed for the symbol before deciding (gap refill).
    """

    symbol: Symbol
    status: SymbolStatus
    bar: Bar | None = None
    outcome: Annotated[BarOutcome, Field(discriminator="kind")] | None = None
    outcome_kind: str | None = None
    action: StrategyAction | None = None
    signal_id: str | None = None
    rule_results: tuple[RuleResult, ...] = ()
    reasons: tuple[str, ...] = ()
    history: SymbolHistory | None = None


class CycleReport(DomainModel):
    """Typed result of one session close; nothing in it has been submitted.

    Attributes:
        session: The session decided.
        next_session: The following calendar session (signal expiry), if known.
        evaluated_at_utc: Clock time of the cycle.
        data_ready_at_utc: ``session.close_utc + data_delay_seconds``.
        status: See :class:`CycleStatus`.
        reasons: Cycle-level codes (``DATA_NOT_FINAL``, ``NEXT_SESSION_UNKNOWN``...).
        history_sessions: Sessions replayed as history in this cycle (gap refill).
        symbols: One result per whitelisted symbol, in decision (sorted) order.
    """

    session: SessionDay
    next_session: SessionDay | None
    evaluated_at_utc: UtcDatetime
    data_ready_at_utc: UtcDatetime
    status: CycleStatus
    reasons: tuple[str, ...] = ()
    history_sessions: tuple[date, ...] = ()
    symbols: tuple[SymbolCycleResult, ...] = ()

    def result(self, symbol: str) -> SymbolCycleResult:
        """The result of ``symbol``.

        Raises:
            KeyError: ``symbol`` is not in the report.
        """
        for item in self.symbols:
            if item.symbol == symbol:
                return item
        raise KeyError(symbol)


class WarmUpReport(DomainModel):
    """Result of :meth:`DailyDataCycle.warm_up` (no decision is ever emitted).

    Attributes:
        first_session: The first session that will be decided.
        sessions: The history sessions registered, oldest first.
        symbols: History recorded per symbol, in sorted order.
    """

    first_session: SessionDay
    sessions: tuple[SessionDay, ...]
    symbols: tuple[SymbolHistory, ...]

    @property
    def complete(self) -> bool:
        """``True`` when every symbol recorded every session."""
        return all(item.complete for item in self.symbols)


BookSource = Callable[[str], Awaitable[TradeBook]]
"""Per-symbol ``TradeBook`` provider, awaited right before that symbol's decision."""


class _BarCheck(DomainModel):
    """Validated, stamped bar or the failure of one session's bars."""

    bar: Bar | None
    status: SymbolStatus
    reason: str | None


# --------------------------------------------------------------------------- validation


def _check_session_bars(bars: Sequence[Bar], session: SessionDay, feed: DataFeed) -> _BarCheck:
    """Exactly one COMPLETE, sane bar of ``feed`` for ``session``, stamped to it."""
    if not bars:
        return _BarCheck(bar=None, status=SymbolStatus.MISSING, reason=MISSING_BAR)
    if len(bars) > 1:
        return _BarCheck(bar=None, status=SymbolStatus.INVALID, reason=DUPLICATE_DAILY_BAR)
    bar = bars[0]
    if bar.timeframe is not Timeframe.DAY_1:
        return _BarCheck(bar=None, status=SymbolStatus.INVALID, reason=WRONG_TIMEFRAME)
    if bar.feed is not feed:  # sec. 10.2.2: same feed live and in backtest
        return _BarCheck(bar=None, status=SymbolStatus.FEED_MISMATCH, reason=FEED_MISMATCH)
    if bar.status is not BarStatus.COMPLETE:
        return _BarCheck(bar=None, status=SymbolStatus.INCOMPLETE, reason=INCOMPLETE_BAR)
    if not _sane_ohlc(bar):
        return _BarCheck(bar=None, status=SymbolStatus.INVALID, reason=INVALID_OHLC)
    try:
        stamped = session_bar(bar, session)
    except SessionBarMismatchError:
        return _BarCheck(bar=None, status=SymbolStatus.INVALID, reason=SESSION_BAR_MISMATCH)
    return _BarCheck(bar=stamped, status=SymbolStatus.DECIDED, reason=None)


def _sane_ohlc(bar: Bar) -> bool:
    """Positive prices with ``low <= open, close <= high`` (re-checked: adapters may
    build bars without validation)."""
    o, h, low, c = bar.open, bar.high, bar.low, bar.close
    if o is None or h is None or low is None or c is None:
        return False
    if min(o, h, low, c) <= 0:
        return False
    return low <= min(o, c) and h >= max(o, c)


def _by_label_date(bars: Sequence[Bar]) -> dict[date, list[Bar]]:
    grouped: dict[date, list[Bar]] = {}
    for bar in bars:
        grouped.setdefault(bar.bar_start_utc.date(), []).append(bar)
    return grouped


def outcome_fields(
    outcome: BarOutcome,
) -> tuple[StrategyAction | None, str | None, tuple[RuleResult, ...], tuple[str, ...]]:
    """``(action, signal_id, rule_results, reasons)`` of a ``MarketFlow`` outcome.

    ``reasons`` is the ``NoAction`` reason, the ``Rejected`` codes or the exit reason.
    """
    decision = outcome.decision
    action = None if decision is None else decision.action
    rules: tuple[RuleResult, ...] = () if decision is None else decision.rule_results
    signal_id: str | None = None
    if decision is not None and decision.signal is not None:
        signal_id = decision.signal.signal_id
    reasons: tuple[str, ...] = ()
    if isinstance(outcome, NoAction):
        reasons = (outcome.reason,)
    elif isinstance(outcome, Rejected):
        reasons = outcome.codes
    elif isinstance(outcome, ExitRequest):
        reasons = (outcome.reason.value,)
    elif isinstance(outcome, EntryProposal):
        signal_id = outcome.signal.signal_id
    return action, signal_id, rules, reasons


# --------------------------------------------------------------------------- use cases


class DailyDataCycle:
    """Warm-up, gap refill and once-per-session decision for a ``1Day`` strategy.

    Args:
        flow: The strategy's ``MarketFlow`` (``1Day`` primary), used by this cycle only.
        market_data: Daily bars (``get_daily_bars``), already domain ``Bar`` with feed.
        calendar: Market calendar (sessions and early closes, sec. 11).
        clock: Time source.
        params: Whitelist, strategy feed and data delay.
    """

    def __init__(
        self,
        *,
        flow: MarketFlow,
        market_data: IMarketData,
        calendar: IMarketCalendar,
        clock: IClock,
        params: DailyCycleParams,
    ) -> None:
        self._flow = flow
        self._market_data = market_data
        self._calendar = calendar
        self._clock = clock
        self._params = params
        self._symbols = tuple(sorted(params.symbols))
        self._delay = timedelta(seconds=params.data_delay_seconds)
        self._sessions: list[SessionDay] = []
        self._opens: list[datetime] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ accessors

    @property
    def registered_sessions(self) -> tuple[SessionDay, ...]:
        """Sessions registered in the flow so far, oldest first."""
        return tuple(self._sessions)

    def data_ready_at(self, session: SessionDay) -> datetime:
        """Instant from which the session's daily bar is considered final."""
        return session.close_utc + self._delay

    # ------------------------------------------------------------------ calendar

    async def _next_session(self, session: SessionDay) -> SessionDay | None:
        for offset in range(1, _NEXT_SESSION_SCAN_DAYS + 1):
            found = await self._calendar.get_session(session.session_date + timedelta(days=offset))
            if found is not None:
                return found
        return None

    async def _sessions_between(self, after: date, before: date) -> list[SessionDay]:
        """Calendar sessions with ``after < session_date < before``, oldest first."""
        found: list[SessionDay] = []
        day = after + timedelta(days=1)
        while day < before:
            session = await self._calendar.get_session(day)
            if session is not None:
                found.append(session)
            day += timedelta(days=1)
        return found

    async def _sessions_before(self, session: SessionDay, count: int) -> list[SessionDay]:
        """The ``count`` calendar sessions before ``session`` (fewer if the calendar has
        none that far back), oldest first."""
        found: list[SessionDay] = []
        day = session.session_date
        limit = session.session_date - timedelta(days=2 * count + _NEXT_SESSION_SCAN_DAYS)
        while len(found) < count and day > limit:
            day -= timedelta(days=1)
            earlier = await self._calendar.get_session(day)
            if earlier is not None:
                found.append(earlier)
        found.reverse()
        return found

    async def latest_closed_session(self) -> SessionDay | None:
        """Latest calendar session whose data is final at ``clock.now_utc()``."""
        now = self._clock.now_utc()
        day = now.date()
        for _ in range(_NEXT_SESSION_SCAN_DAYS + 1):
            session = await self._calendar.get_session(day)
            if session is not None and self.data_ready_at(session) <= now:
                return session
            day -= timedelta(days=1)
        return None

    # ------------------------------------------------------------------ registration

    def _is_registered(self, session: SessionDay) -> bool:
        index = bisect_right(self._opens, session.open_utc) - 1
        return index >= 0 and self._sessions[index] == session

    def _register(self, session: SessionDay) -> None:
        """Register ``session`` in the flow unless it already is (sessions only grow)."""
        if self._is_registered(session):
            return
        self._flow.add_session(session)  # raises SESSION_OUT_OF_ORDER on an older one
        self._sessions.append(session)
        self._opens.append(session.open_utc)

    # ------------------------------------------------------------------ history

    def _history_for(self, symbol: str, session: SessionDay) -> list[SessionDay]:
        """Registered sessions before ``session`` missing from ``symbol``'s window
        (the last ``window_size`` at most: older bars would leave the window anyway)."""
        window = self._flow.window(symbol)
        last_start = window[-1].bar_start_utc if window else None
        missing = [
            s
            for s in self._sessions
            if s.open_utc < session.open_utc and (last_start is None or s.open_utc > last_start)
        ]
        return missing[-self._flow.window_size :]

    def _record_history(
        self, symbol: str, sessions: Sequence[SessionDay], by_date: dict[date, list[Bar]]
    ) -> SymbolHistory:
        """Record ``sessions`` in order (EMPTY when a session has no bar); stop at the
        first session whose bars are present but invalid."""
        recorded: list[date] = []
        empty: list[date] = []
        reasons: list[str] = []
        for session in sessions:
            day = session.session_date
            bars = by_date.get(day, [])
            if not bars:
                bar = empty_session_bar(symbol, session, feed=self._params.feed)
                empty.append(day)
                reasons.append(f"{BACKFILL_EMPTY} {day.isoformat()}")
            else:
                check = _check_session_bars(bars, session, self._params.feed)
                if check.bar is None:
                    reasons.append(f"{check.reason} {day.isoformat()}")
                    return SymbolHistory(
                        symbol=symbol,
                        recorded=tuple(recorded),
                        empty_sessions=tuple(empty),
                        failed_session=day,
                        reasons=tuple(reasons),
                    )
                bar = check.bar
            self._flow.record_closed_bar(bar)
            recorded.append(day)
        return SymbolHistory(
            symbol=symbol,
            recorded=tuple(recorded),
            empty_sessions=tuple(empty),
            reasons=tuple(reasons),
        )

    async def _fetch(self, symbol: str, first: date, last: date) -> dict[date, list[Bar]]:
        bars = await self._market_data.get_daily_bars(symbol, first, last)
        return _by_label_date([bar for bar in bars if bar.symbol == symbol])

    # ------------------------------------------------------------------ warm-up

    async def warm_up(self, first_session: SessionDay) -> WarmUpReport:
        """Register and record the ``window_size`` sessions before ``first_session``.

        Identical to the backtest warm-up (same sessions, EMPTY bars for sessions
        without a bar); no decision is emitted. Call it once, at process start, before
        any :meth:`run_session_close`.

        Raises:
            NonRetryableError: ``ALREADY_STARTED`` (sessions already registered) or
                ``DATA_NOT_FINAL`` (the last history session is not final yet).
            RetryableError: the calendar could not be read (retries exhausted).
        """
        async with self._lock:
            if self._sessions:
                raise NonRetryableError(
                    "warm_up must run before any session is registered", code=ALREADY_STARTED
                )
            history = await self._sessions_before(first_session, self._flow.window_size)
            now = self._clock.now_utc()
            if history and now < self.data_ready_at(history[-1]):
                raise NonRetryableError(
                    f"warm-up session {history[-1].session_date} is not final before "
                    f"{self.data_ready_at(history[-1]).isoformat()}",
                    code=DATA_NOT_FINAL,
                )
            for session in history:
                self._register(session)
            results: list[SymbolHistory] = []
            for symbol in self._symbols:
                results.append(await self._replay(symbol, history))
            return WarmUpReport(
                first_session=first_session, sessions=tuple(history), symbols=tuple(results)
            )

    async def _replay(self, symbol: str, sessions: Sequence[SessionDay]) -> SymbolHistory:
        if not sessions:
            return SymbolHistory(symbol=symbol)
        try:
            by_date = await self._fetch(symbol, sessions[0].session_date, sessions[-1].session_date)
        except (RetryableError, NonRetryableError) as exc:
            return SymbolHistory(
                symbol=symbol,
                failed_session=sessions[0].session_date,
                reasons=(f"FETCH_FAILED {exc.code or type(exc).__name__}",),
            )
        return self._record_history(symbol, sessions, by_date)

    # ------------------------------------------------------------------ session close

    async def run_session_close(
        self, session: SessionDay, *, book: TradeBook | BookSource
    ) -> CycleReport:
        """Use case: decide once on the daily bar of ``session`` (no order is sent).

        Args:
            session: The session that just closed (the latest final session).
            book: Trading records for the decisions, or a per-symbol provider awaited
                right before each symbol's decision.

        Raises:
            StateCriticalError: from ``MarketFlow`` (broker position without a recorded
                entry fill): the system must halt.
        """
        async with self._lock:
            return await self._run_session_close(session, book)

    async def _run_session_close(
        self, session: SessionDay, book: TradeBook | BookSource
    ) -> CycleReport:
        now = self._clock.now_utc()
        ready_at = self.data_ready_at(session)

        def refused(
            reason: str,
            status: SymbolStatus = SymbolStatus.NOT_EVALUATED,
            next_session: SessionDay | None = None,
        ) -> CycleReport:
            return CycleReport(
                session=session,
                next_session=next_session,
                evaluated_at_utc=now,
                data_ready_at_utc=ready_at,
                status=CycleStatus.REFUSED,
                reasons=(reason,),
                symbols=tuple(
                    SymbolCycleResult(symbol=s, status=status, reasons=(reason,))
                    for s in self._symbols
                ),
            )

        if now < ready_at:  # sec. 10.6: the bar may still be forming or not published
            return refused(DATA_NOT_FINAL, SymbolStatus.STALE)
        try:
            if await self._calendar.get_session(session.session_date) != session:
                return refused(SESSION_NOT_IN_CALENDAR)
            following = await self._next_session(session)
            if following is not None and self.data_ready_at(following) <= now:
                return refused(NOT_LATEST_SESSION, next_session=following)
            if self._sessions:
                last = self._sessions[-1]
                gap = await self._sessions_between(last.session_date, session.session_date)
            else:
                gap = await self._sessions_before(session, self._flow.window_size)
        except (RetryableError, NonRetryableError):
            return refused(CALENDAR_UNAVAILABLE)  # sec. 11.5: no calendar, no new trade
        if (
            self._sessions
            and not self._is_registered(session)
            and (session.open_utc < self._sessions[-1].close_utc)
        ):
            return refused(SESSION_OUT_OF_ORDER)

        for item in (*gap, session, *(() if following is None else (following,))):
            self._register(item)
        replayed: set[date] = set()
        results: list[SymbolCycleResult] = []
        for symbol in self._symbols:
            result = await self._decide_symbol(symbol, session, book)
            if result.history is not None:
                replayed.update(result.history.recorded)
            results.append(result)
        decided = all(r.status is SymbolStatus.DECIDED for r in results)
        return CycleReport(
            session=session,
            next_session=following,
            evaluated_at_utc=now,
            data_ready_at_utc=ready_at,
            status=CycleStatus.COMPLETED if decided else CycleStatus.PARTIAL,
            reasons=() if following is not None else (NEXT_SESSION_UNKNOWN,),
            history_sessions=tuple(sorted(replayed)),
            symbols=tuple(results),
        )

    async def _decide_symbol(
        self, symbol: str, session: SessionDay, book: TradeBook | BookSource
    ) -> SymbolCycleResult:
        missing = self._history_for(symbol, session)
        first = missing[0].session_date if missing else session.session_date
        try:
            by_date = await self._fetch(symbol, first, session.session_date)
        except (RetryableError, NonRetryableError) as exc:
            return SymbolCycleResult(
                symbol=symbol,
                status=SymbolStatus.FETCH_FAILED,
                reasons=(f"FETCH_FAILED {exc.code or type(exc).__name__}",),
            )
        history: SymbolHistory | None = None
        if missing:
            history = self._record_history(symbol, missing, by_date)
            if not history.complete:
                return SymbolCycleResult(
                    symbol=symbol,
                    status=SymbolStatus.BACKFILL_FAILED,
                    reasons=history.reasons,
                    history=history,
                )
        check = _check_session_bars(
            by_date.get(session.session_date, []), session, self._params.feed
        )
        if check.bar is None:
            return SymbolCycleResult(
                symbol=symbol,
                status=check.status,
                reasons=(check.reason or MISSING_BAR,),
                history=history,
            )
        current = book if isinstance(book, TradeBook) else await book(symbol)
        outcome = await self._flow.on_closed_bar(check.bar, book=current)
        action, signal_id, rules, reasons = outcome_fields(outcome)
        return SymbolCycleResult(
            symbol=symbol,
            status=SymbolStatus.DECIDED,
            bar=check.bar,
            outcome=outcome,
            outcome_kind=outcome.kind,
            action=action,
            signal_id=signal_id,
            rule_results=rules,
            reasons=reasons,
            history=history,
        )
