"""Single check library (sec. 19), shared by the pre-AI step and the Execution Guard (sec. 20).

Every safety and validity check of sec. 19 lives here, once. A check is a pure function
of an immutable :class:`CheckContext` (no I/O, no clock, no randomness) that returns one
or more :class:`~domain.models.CheckResult`. The two entry points evaluate EVERY check and
never short-circuit, so ``risk_events`` records all results, not only the first failure:

* :func:`run_pre_ai_checks`: the checks of the sec. 19 table, in table order.
* :func:`run_execution_checks`: the same checks again (with the state the Execution Guard
  refreshed from the broker) plus the execution preconditions of sec. 20 / 22 the caller
  provides (:class:`ExecutionFacts`: idempotency, symbol lock, AI verdict in ``ACTIVE``).

Result convention (stable codes, recorded in ``risk_events``):

* ``detail["check"]`` is always the sec. 19 check code (:class:`CheckCode`) the result
  belongs to; group by it to see one row per check.
* ``code`` is the outcome: the check code itself when a library-native check passes
  (e.g. ``ENTRY_WINDOW``), the specific failure code when it fails (e.g.
  ``OUTSIDE_ENTRY_WINDOW``). Results produced by the wrapped domain functions keep their
  own stable codes (``domain.market.universe``: ``SYMBOL_NOT_WHITELISTED``,
  ``AVG_VOLUME_TOO_LOW``...; ``domain.risk.risk_engine``: ``MAX_DRAWDOWN``...;
  ``domain.risk.exits``: ``EXIT_LEVELS_VALID`` / ``INVALID_EXIT_LEVELS``;
  ``domain.market.quality``: ``DATA_FRESH`` / ``STALE``).
* A composite check contributes one result per independent condition (for example
  ``RISK_LIMITS_OK`` contributes the eight limits of sec. 15.1 and ``TRADING_ENABLED``
  one result per active kill-switch reason), so co-occurring failures are all visible.
* Details never carry the decision instant itself (``risk_events.occurred_at_utc``
  records it), so the same decision taken at two instants yields equal results.
* Fail closed (sec. 3.4): a missing input (``None``) never passes. It fails with an
  ``*_UNAVAILABLE`` code, or ``PARAM_PENDING`` for an ``OWNER_DECISION`` still ``null``.

Reuse (sec. 19: never two implementations of one check): listing, tradability, price,
average volume and spread wrap ``domain.market.universe``; intraday freshness wraps
``domain.market.quality.check_staleness``; entry windows use
``domain.market.session.SessionWindows``; bar state and cooldown use
``domain.strategy.strategy.bar_allows_entry`` / ``cooldown_active``; exit levels delegate to
``domain.risk.exits.check_exit_levels`` and risk limits to
``domain.risk.risk_engine.evaluate_limits``. The kill-switch rule (:func:`kill_switch_reasons`)
is the single implementation also used by ``application.control_rules``.

Timing semantics (:class:`IntradayTiming` / :class:`DailyTiming`):

* Intraday: entries inside ``[entries_allowed_from, entries_allowed_until)`` and before
  ``flatten_at`` of the signal bar's session (sec. 11.4), with the broker clock open.
* Daily (``1Day`` primary): the decision is taken after the session close and the order
  executes at the NEXT session open, so ``MARKET_OPEN`` and ``ENTRY_WINDOW`` require the
  next session to be known (``SESSION_UNKNOWN`` otherwise) and the decision to be at or
  after the signal session close; expiry is the signal's own ``expires_at_utc``.

Circuit breaker: ``CIRCUIT_BREAKER_NORMAL`` passes only in ``NORMAL`` (sec. 19 and
invariant 51.13). Sec. 30.1 lists ``WARNING`` as allowing entries; the stricter rule of
sec. 19 / 51 is applied (invariants rank first, sec. 50) and reported as a deviation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, model_validator

from domain.market import universe
from domain.market.quality import check_staleness
from domain.market.session import SessionWindows
from domain.models import (
    AccountState,
    AIMode,
    AIValidity,
    AIVerdictKind,
    AIVerdictResult,
    Bar,
    BrokerOrder,
    BrokerOrderStatus,
    CheckResult,
    CircuitBreakerState,
    DataFeed,
    DomainModel,
    Money,
    NonEmptyStr,
    NonNegativeDecimal,
    OrderSide,
    Position,
    Price,
    ProposedTrade,
    Quote,
    SessionDay,
    Signal,
    Symbol,
    SystemMode,
    UtcDatetime,
)
from domain.risk.exits import PARAM_PENDING, check_exit_levels
from domain.risk.risk_engine import (
    BPS_PER_UNIT,
    PendingEntry,
    PositionStop,
    RiskParams,
    RiskState,
    evaluate_limits,
)
from domain.strategy.strategy import bar_allows_entry, cooldown_active

__all__ = [
    "ASSET_STATUS_UNAVAILABLE",
    "BAR_NOT_CLOSED",
    "BAR_NOT_ENTERABLE",
    "BROKER_STATE_UNAVAILABLE",
    "BUYING_POWER_INSUFFICIENT",
    "BUYING_POWER_UNAVAILABLE",
    "CIRCUIT_BREAKER_NOT_NORMAL",
    "CIRCUIT_BREAKER_UNAVAILABLE",
    "CONFIG_INCOMPLETE",
    "CONFIG_STATUS_UNAVAILABLE",
    "CONTROL_STATE_UNAVAILABLE",
    "COOLDOWN_ACTIVE",
    "DAILY_BAR_SUPERSEDED",
    "DUPLICATE_CLIENT_ORDER_ID",
    "DUPLICATE_SIGNAL",
    "EMERGENCY_CLOSE_ACTIVE",
    "ENTRY_PENDING",
    "EXECUTION_CHECKS",
    "EXIT_LEVELS_UNAVAILABLE",
    "FEED_DISCONNECTED",
    "FEED_STATUS_UNAVAILABLE",
    "KILL_SWITCH_CODES",
    "LIQUIDITY_FEED_MISMATCH",
    "LIQUIDITY_UNAVAILABLE",
    "MARKET_CLOCK_UNAVAILABLE",
    "MARKET_CLOSED",
    "ORDER_HISTORY_UNAVAILABLE",
    "OUTSIDE_ENTRY_WINDOW",
    "PARAM_PENDING",
    "POSITION_EXISTS",
    "PRE_AI_CHECKS",
    "RECONCILIATION_STALE",
    "RECONCILIATION_UNAVAILABLE",
    "RISK_LIMITS_UNAVAILABLE",
    "SESSION_UNKNOWN",
    "SIGNAL_EXPIRED",
    "SIGNAL_HISTORY_UNAVAILABLE",
    "SPREAD_NOT_APPLIED",
    "STATE_MISMATCH_UNRESOLVED",
    "STOP_FILE_PRESENT",
    "SYMBOL_LOCK_NOT_HELD",
    "SYSTEM_MODE_UNAVAILABLE",
    "SYSTEM_NOT_RUNNING",
    "TRADING_DISABLED",
    "AIApprovalCode",
    "Check",
    "CheckCode",
    "CheckContext",
    "CheckParams",
    "ControlFacts",
    "DailyTiming",
    "EntryTiming",
    "ExecutionFacts",
    "IntradayTiming",
    "MarketFacts",
    "PortfolioFacts",
    "ReconciliationFacts",
    "RuntimeFacts",
    "SpreadFilter",
    "all_passed",
    "checks_of",
    "entry_pending_in_book",
    "failed_codes",
    "kill_switch_reasons",
    "pending_entry_order",
    "run_execution_checks",
    "run_pre_ai_checks",
]


class CheckCode(StrEnum):
    """Sec. 19 check codes (``detail["check"]`` of every result), plus sec. 20 preconditions."""

    SYMBOL_ALLOWED = "SYMBOL_ALLOWED"
    MARKET_OPEN = "MARKET_OPEN"
    ENTRY_WINDOW = "ENTRY_WINDOW"
    BAR_CLOSED = "BAR_CLOSED"
    SIGNAL_NOT_EXPIRED = "SIGNAL_NOT_EXPIRED"
    DATA_FRESH = "DATA_FRESH"
    NO_EXISTING_POSITION = "NO_EXISTING_POSITION"
    NO_PENDING_ORDER = "NO_PENDING_ORDER"
    NOT_DUPLICATE_SIGNAL = "NOT_DUPLICATE_SIGNAL"
    COOLDOWN_OK = "COOLDOWN_OK"
    PRICE_RANGE = "PRICE_RANGE"
    LIQUIDITY_OK = "LIQUIDITY_OK"
    SPREAD_OK = "SPREAD_OK"
    EXIT_LEVELS_VALID = "EXIT_LEVELS_VALID"
    RISK_LIMITS_OK = "RISK_LIMITS_OK"
    BUYING_POWER_OK = "BUYING_POWER_OK"
    CIRCUIT_BREAKER_NORMAL = "CIRCUIT_BREAKER_NORMAL"
    TRADING_ENABLED = "TRADING_ENABLED"
    SYSTEM_RUNNING = "SYSTEM_RUNNING"
    STATE_RECONCILED = "STATE_RECONCILED"
    CONFIG_COMPLETE = "CONFIG_COMPLETE"
    # Execution Guard preconditions (sec. 20.1, 20.5, 20.6, 22).
    SYMBOL_LOCK_HELD = "SYMBOL_LOCK_HELD"
    CLIENT_ORDER_ID_UNUSED = "CLIENT_ORDER_ID_UNUSED"
    AI_VERDICT_OK = "AI_VERDICT_OK"


# --------------------------------------------------------------------------- failure codes

MARKET_CLOCK_UNAVAILABLE: Final = "MARKET_CLOCK_UNAVAILABLE"
MARKET_CLOSED: Final = "MARKET_CLOSED"
SESSION_UNKNOWN: Final = "SESSION_UNKNOWN"
OUTSIDE_ENTRY_WINDOW: Final = "OUTSIDE_ENTRY_WINDOW"
BAR_NOT_CLOSED: Final = "BAR_NOT_CLOSED"
BAR_NOT_ENTERABLE: Final = "BAR_NOT_ENTERABLE"
SIGNAL_EXPIRED: Final = "SIGNAL_EXPIRED"
FEED_STATUS_UNAVAILABLE: Final = "FEED_STATUS_UNAVAILABLE"
FEED_DISCONNECTED: Final = "FEED_DISCONNECTED"
DAILY_BAR_SUPERSEDED: Final = "DAILY_BAR_SUPERSEDED"
BROKER_STATE_UNAVAILABLE: Final = "BROKER_STATE_UNAVAILABLE"
POSITION_EXISTS: Final = "POSITION_EXISTS"
ENTRY_PENDING: Final = "ENTRY_PENDING"
SIGNAL_HISTORY_UNAVAILABLE: Final = "SIGNAL_HISTORY_UNAVAILABLE"
DUPLICATE_SIGNAL: Final = "DUPLICATE_SIGNAL"
COOLDOWN_ACTIVE: Final = "COOLDOWN_ACTIVE"
ASSET_STATUS_UNAVAILABLE: Final = "ASSET_STATUS_UNAVAILABLE"
LIQUIDITY_UNAVAILABLE: Final = "LIQUIDITY_UNAVAILABLE"
LIQUIDITY_FEED_MISMATCH: Final = "LIQUIDITY_FEED_MISMATCH"
SPREAD_NOT_APPLIED: Final = "SPREAD_NOT_APPLIED"
EXIT_LEVELS_UNAVAILABLE: Final = "EXIT_LEVELS_UNAVAILABLE"
RISK_LIMITS_UNAVAILABLE: Final = "RISK_LIMITS_UNAVAILABLE"
BUYING_POWER_UNAVAILABLE: Final = "BUYING_POWER_UNAVAILABLE"
BUYING_POWER_INSUFFICIENT: Final = "BUYING_POWER_INSUFFICIENT"
CIRCUIT_BREAKER_UNAVAILABLE: Final = "CIRCUIT_BREAKER_UNAVAILABLE"
CIRCUIT_BREAKER_NOT_NORMAL: Final = "CIRCUIT_BREAKER_NOT_NORMAL"
CONTROL_STATE_UNAVAILABLE: Final = "CONTROL_STATE_UNAVAILABLE"
STOP_FILE_PRESENT: Final = "STOP_FILE_PRESENT"
EMERGENCY_CLOSE_ACTIVE: Final = "EMERGENCY_CLOSE_ACTIVE"
TRADING_DISABLED: Final = "TRADING_DISABLED"
SYSTEM_MODE_UNAVAILABLE: Final = "SYSTEM_MODE_UNAVAILABLE"
SYSTEM_NOT_RUNNING: Final = "SYSTEM_NOT_RUNNING"
RECONCILIATION_UNAVAILABLE: Final = "RECONCILIATION_UNAVAILABLE"
STATE_MISMATCH_UNRESOLVED: Final = "STATE_MISMATCH_UNRESOLVED"
RECONCILIATION_STALE: Final = "RECONCILIATION_STALE"
CONFIG_STATUS_UNAVAILABLE: Final = "CONFIG_STATUS_UNAVAILABLE"
CONFIG_INCOMPLETE: Final = "CONFIG_INCOMPLETE"
SYMBOL_LOCK_NOT_HELD: Final = "SYMBOL_LOCK_NOT_HELD"
ORDER_HISTORY_UNAVAILABLE: Final = "ORDER_HISTORY_UNAVAILABLE"
DUPLICATE_CLIENT_ORDER_ID: Final = "DUPLICATE_CLIENT_ORDER_ID"

KILL_SWITCH_CODES: Final[frozenset[str]] = frozenset(
    {STOP_FILE_PRESENT, EMERGENCY_CLOSE_ACTIVE, TRADING_DISABLED}
)
"""Kill-switch codes of ``TRADING_ENABLED`` (sec. 31.1-31.3, AC-15)."""


class AIApprovalCode(StrEnum):
    """Failure codes of ``AI_VERDICT_OK`` (sec. 20.5: ``ACTIVE`` needs a valid APPROVE)."""

    AI_APPROVAL_MISSING = "AI_APPROVAL_MISSING"
    AI_RESPONSE_NOT_VALID = "AI_RESPONSE_NOT_VALID"
    AI_SIGNAL_MISMATCH = "AI_SIGNAL_MISMATCH"
    AI_VETOED = "AI_VETOED"


_FINAL_ORDER_STATUSES: Final[frozenset[BrokerOrderStatus]] = frozenset(
    {
        BrokerOrderStatus.FILLED,
        BrokerOrderStatus.CANCELED,
        BrokerOrderStatus.EXPIRED,
        BrokerOrderStatus.REJECTED,
        BrokerOrderStatus.REPLACED,
    }
)
"""Broker statuses after which an order can no longer open a position. Anything else
(including ``DONE_FOR_DAY``, ``HELD``, ``SUSPENDED``) still counts as pending: fail closed."""


# --------------------------------------------------------------------------- context


class SpreadFilter(StrEnum):
    """How ``SPREAD_OK`` applies (sec. 10.5, 12: "si aplica").

    ``ENFORCED``: a quote is required and ``max_spread_bps`` is a hard filter.
    ``NOT_APPLIED``: the environment has no quote source (backtest on stored bars); the
    check passes with ``SPREAD_NOT_APPLIED`` so the omission is explicit in every result.
    """

    ENFORCED = "ENFORCED"
    NOT_APPLIED = "NOT_APPLIED"


class CheckParams(DomainModel):
    """Configuration the checks compare against (``None`` = OWNER_DECISION pending).

    Attributes:
        whitelist / blacklist / min_price / max_price / min_avg_daily_volume /
        avg_volume_lookback_days / liquidity_feed / max_spread_bps: ``universe.*``.
        spread_filter: Whether the environment enforces the spread filter.
        max_bar_age_seconds: ``market_data.max_bar_age_seconds`` (sec. 10.6).
        max_reconcile_age_seconds: Maximum age of the last clean reconciliation.
        cooldown_bars / min_minutes_per_bar: ``strategy.*``.
        min_tp_distance_ticks: ``strategy.exit.min_tp_distance_ticks`` (sec. 16.4).
        risk: ``risk.*`` (sec. 15).
    """

    whitelist: tuple[Symbol, ...] | None
    blacklist: tuple[Symbol, ...]
    min_price: Price | None
    max_price: Price | None
    min_avg_daily_volume: Annotated[int, Field(ge=0)] | None
    avg_volume_lookback_days: Annotated[int, Field(ge=1)] | None
    liquidity_feed: DataFeed | None
    max_spread_bps: NonNegativeDecimal | None
    spread_filter: SpreadFilter
    max_bar_age_seconds: Annotated[int, Field(ge=1)] | None
    max_reconcile_age_seconds: Annotated[int, Field(ge=1)] | None
    cooldown_bars: Annotated[int, Field(ge=0)] | None
    min_minutes_per_bar: Annotated[int, Field(ge=1)] | None
    min_tp_distance_ticks: Annotated[int, Field(ge=1)] | None
    risk: RiskParams


class ControlFacts(DomainModel):
    """``system_control`` row plus the STOP file (sec. 31.1, 31.2)."""

    trading_enabled: bool
    emergency_close: bool
    stop_file_present: bool
    ai_mode: AIMode


class ReconciliationFacts(DomainModel):
    """Latest reconciliation (sec. 24): when it last ran cleanly and any open mismatch."""

    last_clean_at_utc: UtcDatetime | None
    """End of the last reconciliation without mismatch (``None``: never reconciled)."""
    unresolved_mismatch: bool


class RuntimeFacts(DomainModel):
    """Runtime safety state (control, breaker, mode, broker status, config, clock, feed).

    Every field may be ``None`` (unknown) and then fails its check closed.

    Attributes:
        control: Control table and STOP file.
        breaker_state: Circuit breaker state (sec. 30).
        system_mode: Lifecycle mode (sec. 32).
        reconciliation: Latest reconciliation outcome.
        pending_owner_decisions: Required OWNER_DECISION paths still ``null`` for this
            environment (``CONFIG_COMPLETE``).
        tradable_symbols: Symbols the broker Asset API reports ``tradable`` (sec. 12).
        market_open: Market open now according to the broker clock (sec. 11.1).
        feed_connected: Market data source connected (sec. 10.6, 33).
    """

    control: ControlFacts | None
    breaker_state: CircuitBreakerState | None
    system_mode: SystemMode | None
    reconciliation: ReconciliationFacts | None
    pending_owner_decisions: tuple[str, ...] | None
    tradable_symbols: frozenset[Symbol] | None
    market_open: bool | None
    feed_connected: bool | None

    @classmethod
    def unavailable(cls) -> Self:
        """Nothing known: every runtime check fails closed."""
        return cls(
            control=None,
            breaker_state=None,
            system_mode=None,
            reconciliation=None,
            pending_owner_decisions=None,
            tradable_symbols=None,
            market_open=None,
            feed_connected=None,
        )


class IntradayTiming(DomainModel):
    """Intraday entry timing: the signal bar's session windows and the last minute bar."""

    kind: Literal["INTRADAY"] = "INTRADAY"
    windows: SessionWindows | None
    """Windows of the signal bar's session (``None``: session not in the calendar)."""
    last_minute_bar_end_utc: UtcDatetime | None
    """End of the last 1-minute bar received for the symbol (``None``: none yet)."""


class DailyTiming(DomainModel):
    """Daily next-open timing: the signal bar's session and the session after it."""

    kind: Literal["DAILY"] = "DAILY"
    signal_session: SessionDay | None
    next_session: SessionDay | None


EntryTiming = Annotated[IntradayTiming | DailyTiming, Field(discriminator="kind")]


class MarketFacts(DomainModel):
    """Market data the universe checks need.

    Attributes:
        daily_bars: Daily bars of the liquidity feed for the symbol (``None``: the source
            is unavailable). Only bars that ended at or before the signal bar's start are
            used, i.e. sessions before the decision session (no lookahead).
        quote: Latest quote (``None``: no quote).
    """

    daily_bars: tuple[Bar, ...] | None
    quote: Quote | None


class PortfolioFacts(DomainModel):
    """Broker state (source of truth, sec. 3.3) and the DB records the checks need.

    ``None`` for a broker field means it could not be read: every check that needs it
    fails closed. ``executed_signal_ids`` are the signal ids that already produced an
    entry submission (DB, sec. 22).
    """

    account: AccountState | None
    positions: tuple[Position, ...] | None
    open_orders: tuple[BrokerOrder, ...] | None
    pending_entries: tuple[PendingEntry, ...]
    position_stops: tuple[PositionStop, ...]
    week_start_equity: Money
    peak_equity: Money
    executed_signal_ids: frozenset[str] | None


class CheckContext(DomainModel):
    """Everything the checks of one signal see, frozen at one instant ``now_utc``.

    Attributes:
        now_utc: Decision instant (from ``IClock``).
        signal: The strategy's BUY signal.
        signal_bar: The closed primary bar the signal was decided on.
        bars_since_last_exit: Closed primary bars since the last exit (``None``: none).
        trade: The Risk Engine's priced and sized trade, ``None`` if pricing failed.
        params / timing / runtime / market / portfolio: See the field types.
    """

    now_utc: UtcDatetime
    signal: Signal
    signal_bar: Bar
    bars_since_last_exit: Annotated[int, Field(ge=0)] | None
    trade: ProposedTrade | None
    params: CheckParams
    timing: EntryTiming
    runtime: RuntimeFacts
    market: MarketFacts
    portfolio: PortfolioFacts

    @model_validator(mode="after")
    def _same_symbol(self) -> Self:
        symbol = self.signal.symbol
        if self.signal_bar.symbol != symbol:
            raise ValueError("signal_bar.symbol must equal signal.symbol")
        if self.trade is not None and (
            self.trade.symbol != symbol or self.trade.signal_id != self.signal.signal_id
        ):
            raise ValueError("trade must belong to the signal (symbol and signal_id)")
        return self

    @property
    def symbol(self) -> Symbol:
        """Symbol of the signal."""
        return self.signal.symbol


class ExecutionFacts(DomainModel):
    """Preconditions only the Execution Guard can provide (sec. 20, 22).

    Attributes:
        client_order_id: Deterministic entry ``client_order_id`` (sec. 22).
        client_order_id_seen: Whether DB or broker already hold an order with that id
            (``None``: could not be determined).
        symbol_lock_held: Whether the caller holds the symbol lock (sec. 28).
        ai_result: Validated AI result for this signal (``None``: none). Required to be a
            valid ``APPROVE`` only in ``ACTIVE`` (sec. 20.5); ignored otherwise (AC-17).
    """

    client_order_id: NonEmptyStr
    client_order_id_seen: bool | None
    symbol_lock_held: bool
    ai_result: AIVerdictResult | None


# --------------------------------------------------------------------------- helpers

Check = Callable[[CheckContext], tuple[CheckResult, ...]]
"""A check: pure function of the context, one or more results."""


def _result(check: CheckCode, passed: bool, code: str, **detail: Any) -> CheckResult:
    return CheckResult(passed=passed, code=code, detail={**detail, "check": check.value})


def _passed(check: CheckCode, **detail: Any) -> CheckResult:
    return _result(check, True, check.value, **detail)


def _failed(check: CheckCode, code: str, **detail: Any) -> CheckResult:
    return _result(check, False, code, **detail)


def _pending(check: CheckCode, *params: str) -> CheckResult:
    return _failed(check, PARAM_PENDING, pending=list(params))


def _tag(check: CheckCode, result: CheckResult) -> CheckResult:
    """A wrapped domain result tagged with the sec. 19 check it belongs to."""
    return CheckResult(
        passed=result.passed, code=result.code, detail={**result.detail, "check": check.value}
    )


def failed_codes(results: Iterable[CheckResult]) -> tuple[str, ...]:
    """Codes of the failed results, in order, without repetitions."""
    return tuple(dict.fromkeys(result.code for result in results if not result.passed))


def all_passed(results: Iterable[CheckResult]) -> bool:
    """``True`` when every result passed (an empty set never counts as approval)."""
    items = tuple(results)
    return bool(items) and all(result.passed for result in items)


def kill_switch_reasons(control: ControlFacts) -> tuple[str, ...]:
    """Why the runtime controls block new entries (empty: they do not).

    The single implementation of the kill switch (sec. 31.1-31.3): the STOP file and
    ``emergency_close`` override ``trading_enabled``. Order: STOP file, emergency close,
    trading disabled.
    """
    reasons: list[str] = []
    if control.stop_file_present:
        reasons.append(STOP_FILE_PRESENT)
    if control.emergency_close:
        reasons.append(EMERGENCY_CLOSE_ACTIVE)
    if not control.trading_enabled:
        reasons.append(TRADING_DISABLED)
    return tuple(reasons)


def entry_pending_in_book(symbol: Symbol, pending_entries: Sequence[PendingEntry]) -> bool:
    """Whether the DB records a pending (unfilled) entry for ``symbol``."""
    return any(entry.symbol == symbol for entry in pending_entries)


def pending_entry_order(symbol: Symbol, open_orders: Sequence[BrokerOrder]) -> BrokerOrder | None:
    """First broker order that may still open a position in ``symbol`` (a BUY, not final).

    Long-only MVP: entries are BUY orders; bracket legs and system exits are SELL orders.
    """
    for order in open_orders:
        if (
            order.symbol == symbol
            and order.side is OrderSide.BUY
            and order.status not in _FINAL_ORDER_STATUSES
        ):
            return order
    return None


# --------------------------------------------------------------------------- universe


def check_symbol_allowed(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``SYMBOL_ALLOWED``: whitelisted, not blacklisted, broker ``tradable`` (sec. 12)."""
    code, params, symbol = CheckCode.SYMBOL_ALLOWED, ctx.params, ctx.symbol
    if params.whitelist is None:
        listed = _pending(code, "universe.whitelist")
    else:
        listed = _tag(
            code,
            universe.check_symbol_listed(
                symbol, whitelist=params.whitelist, blacklist=params.blacklist
            ),
        )
    tradable_set = ctx.runtime.tradable_symbols
    if tradable_set is None:
        tradable = _failed(code, ASSET_STATUS_UNAVAILABLE, symbol=symbol)
    else:
        tradable = _tag(code, universe.check_tradable(symbol, tradable=symbol in tradable_set))
    return (listed, tradable)


def check_price_range(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``PRICE_RANGE``: signal bar close within ``[min_price, max_price]`` (sec. 12)."""
    code, params = CheckCode.PRICE_RANGE, ctx.params
    if params.min_price is None or params.max_price is None:
        missing = [
            name
            for name, value in (
                ("universe.min_price", params.min_price),
                ("universe.max_price", params.max_price),
            )
            if value is None
        ]
        return (_pending(code, *missing),)
    return (
        _tag(
            code,
            universe.check_price_range(
                ctx.symbol,
                ctx.signal_bar.close,
                min_price=params.min_price,
                max_price=params.max_price,
            ),
        ),
    )


def check_liquidity(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``LIQUIDITY_OK``: average daily volume of previous sessions (sec. 10.2.1, 12).

    Bars must come from ``universe.liquidity_feed`` (owner decision 2026-10-02: SIP daily
    bars) and only bars that ended at or before the signal bar's start count, so the
    decision session itself is never used (no lookahead).
    """
    code, params, symbol = CheckCode.LIQUIDITY_OK, ctx.params, ctx.symbol
    missing = [
        name
        for name, value in (
            ("universe.min_avg_daily_volume", params.min_avg_daily_volume),
            ("universe.avg_volume_lookback_days", params.avg_volume_lookback_days),
            ("universe.liquidity_feed", params.liquidity_feed),
        )
        if value is None
    ]
    if missing:
        return (_pending(code, *missing),)
    bars = ctx.market.daily_bars
    if bars is None:
        return (_failed(code, LIQUIDITY_UNAVAILABLE, symbol=symbol),)
    own = [bar for bar in bars if bar.symbol == symbol]
    feeds = sorted({bar.feed.value for bar in own if bar.feed is not params.liquidity_feed})
    if feeds:
        return (
            _failed(
                code,
                LIQUIDITY_FEED_MISMATCH,
                symbol=symbol,
                feeds=feeds,
                liquidity_feed=str(params.liquidity_feed),
            ),
        )
    lookback, minimum = params.avg_volume_lookback_days, params.min_avg_daily_volume
    if lookback is None or minimum is None:  # pragma: no cover - reported as pending above
        return (_pending(code, "universe.avg_volume_lookback_days"),)
    return (
        _tag(
            code,
            universe.check_avg_daily_volume(
                symbol,
                own,
                lookback_days=lookback,
                min_avg_daily_volume=minimum,
                as_of_utc=ctx.signal_bar.bar_start_utc,
            ),
        ),
    )


def check_spread(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``SPREAD_OK``: quote spread ``<= max_spread_bps`` when the filter applies (10.5)."""
    code, params = CheckCode.SPREAD_OK, ctx.params
    if params.spread_filter is SpreadFilter.NOT_APPLIED:
        return (_result(code, True, SPREAD_NOT_APPLIED, reason="no quote source"),)
    if params.max_spread_bps is None:
        return (_pending(code, "universe.max_spread_bps"),)
    return (
        _tag(
            code,
            universe.check_spread(
                ctx.symbol, ctx.market.quote, max_spread_bps=params.max_spread_bps
            ),
        ),
    )


# --------------------------------------------------------------------------- market / session


def check_market_open(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``MARKET_OPEN``: broker clock open (intraday) / next session known (daily)."""
    code, timing, market_open = CheckCode.MARKET_OPEN, ctx.timing, ctx.runtime.market_open
    if isinstance(timing, DailyTiming):
        if timing.next_session is None:
            return (_failed(code, SESSION_UNKNOWN, mode=timing.kind),)
        if market_open is None:
            return (_failed(code, MARKET_CLOCK_UNAVAILABLE, mode=timing.kind),)
        return (
            _passed(
                code,
                mode=timing.kind,
                executes_at_utc=timing.next_session.open_utc.isoformat(),
            ),
        )
    if market_open is None:
        return (_failed(code, MARKET_CLOCK_UNAVAILABLE, mode=timing.kind),)
    if not market_open:
        return (_failed(code, MARKET_CLOSED, mode=timing.kind),)
    return (_passed(code, mode=timing.kind),)


def check_entry_window(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``ENTRY_WINDOW``: inside the session's entry window (sec. 11.4) / after the close.

    Intraday: ``entries_allowed_from <= now < entries_allowed_until`` and not past
    ``flatten_at``. Daily: the signal session and the next one are known and
    ``now >= signal session close`` (valid until the signal expiry, checked apart).
    """
    code, timing, now = CheckCode.ENTRY_WINDOW, ctx.timing, ctx.now_utc
    detail: dict[str, Any] = {"mode": timing.kind}
    if isinstance(timing, DailyTiming):
        session, following = timing.signal_session, timing.next_session
        if session is None or following is None:
            return (_failed(code, SESSION_UNKNOWN, **detail),)
        detail.update(
            allowed_from_utc=session.close_utc.isoformat(),
            executes_at_utc=following.open_utc.isoformat(),
        )
        if now < session.close_utc:
            return (_failed(code, OUTSIDE_ENTRY_WINDOW, **detail),)
        return (_passed(code, **detail),)
    windows = timing.windows
    if windows is None:
        return (_failed(code, SESSION_UNKNOWN, **detail),)
    detail.update(
        allowed_from_utc=windows.entries_allowed_from_utc.isoformat(),
        allowed_until_utc=windows.entries_allowed_until_utc.isoformat(),
        flatten_at_utc=None
        if windows.flatten_at_utc is None
        else windows.flatten_at_utc.isoformat(),
    )
    if not windows.is_entry_window(now) or windows.is_flatten_time(now):
        return (_failed(code, OUTSIDE_ENTRY_WINDOW, **detail),)
    return (_passed(code, **detail),)


def check_bar_closed(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``BAR_CLOSED``: the signal bar ended and its state allows entries (10.3.4-10.3.5)."""
    code, bar = CheckCode.BAR_CLOSED, ctx.signal_bar
    minimum = ctx.params.min_minutes_per_bar
    detail = {
        "bar_end_utc": bar.bar_end_utc.isoformat(),
        "bar_status": bar.status.value,
        "minutes_present": bar.minutes_present,
        "min_minutes_per_bar": minimum,
    }
    closed = (
        _failed(code, BAR_NOT_CLOSED, **detail)
        if bar.bar_end_utc > ctx.now_utc
        else _passed(code, **detail)
    )
    if minimum is None:
        state = _pending(code, "strategy.min_minutes_per_bar")
    elif not bar_allows_entry(bar, minimum):
        state = _failed(code, BAR_NOT_ENTERABLE, **detail)
    else:
        state = _passed(code, **detail)
    return (closed, state)


def check_signal_not_expired(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``SIGNAL_NOT_EXPIRED``: ``now < expires_at_utc`` (sec. 13.7, 19)."""
    code, expires = CheckCode.SIGNAL_NOT_EXPIRED, ctx.signal.expires_at_utc
    detail = {"expires_at_utc": expires.isoformat()}
    if ctx.now_utc < expires:
        return (_passed(code, **detail),)
    return (_failed(code, SIGNAL_EXPIRED, **detail),)


def check_data_fresh(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``DATA_FRESH``: feed connected and the symbol not STALE (sec. 10.6).

    Intraday: ``domain.market.quality.check_staleness`` on the last 1-minute bar.
    Daily: the signal bar is still the latest complete daily bar, i.e. no session after
    the signal session has closed yet (``DAILY_BAR_SUPERSEDED`` otherwise).
    """
    code, timing, symbol = CheckCode.DATA_FRESH, ctx.timing, ctx.symbol
    connected = ctx.runtime.feed_connected
    if connected is None:
        feed = _failed(code, FEED_STATUS_UNAVAILABLE)
    elif not connected:
        feed = _failed(code, FEED_DISCONNECTED)
    else:
        feed = _passed(code, part="feed")
    if isinstance(timing, DailyTiming):
        session, following = timing.signal_session, timing.next_session
        if session is None:
            fresh = _failed(code, SESSION_UNKNOWN, mode=timing.kind)
        elif ctx.signal_bar.bar_end_utc != session.close_utc or (
            following is not None and ctx.now_utc >= following.close_utc
        ):
            fresh = _failed(code, DAILY_BAR_SUPERSEDED, mode=timing.kind, symbol=symbol)
        else:
            fresh = _passed(code, mode=timing.kind, part="symbol")
        return (feed, fresh)
    max_age = ctx.params.max_bar_age_seconds
    if max_age is None:
        return (feed, _pending(code, "market_data.max_bar_age_seconds"))
    stale = check_staleness(
        symbol,
        timing.last_minute_bar_end_utc,
        now_utc=ctx.now_utc,
        max_bar_age_seconds=float(max_age),
    )
    return (feed, _tag(code, stale))


# --------------------------------------------------------------------------- positions / signal


def check_no_existing_position(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``NO_EXISTING_POSITION``: the broker reports no position in the symbol."""
    code, positions = CheckCode.NO_EXISTING_POSITION, ctx.portfolio.positions
    if positions is None:
        return (_failed(code, BROKER_STATE_UNAVAILABLE, source="positions"),)
    held = [p for p in positions if p.symbol == ctx.symbol and p.qty != 0]
    if held:
        return (_failed(code, POSITION_EXISTS, symbol=ctx.symbol, qty=str(held[0].qty)),)
    return (_passed(code, symbol=ctx.symbol),)


def check_no_pending_order(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``NO_PENDING_ORDER``: no pending entry in the symbol, broker and DB (sec. 19, 21.2.4)."""
    code, portfolio, symbol = CheckCode.NO_PENDING_ORDER, ctx.portfolio, ctx.symbol
    if portfolio.open_orders is None:
        broker = _failed(code, BROKER_STATE_UNAVAILABLE, source="open_orders")
    else:
        order = pending_entry_order(symbol, portfolio.open_orders)
        broker = (
            _passed(code, source="broker")
            if order is None
            else _failed(
                code,
                ENTRY_PENDING,
                source="broker",
                client_order_id=order.client_order_id,
                status=order.status.value,
            )
        )
    database = (
        _failed(code, ENTRY_PENDING, source="db")
        if entry_pending_in_book(symbol, portfolio.pending_entries)
        else _passed(code, source="db")
    )
    return (broker, database)


def check_not_duplicate_signal(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``NOT_DUPLICATE_SIGNAL``: the ``signal_id`` never produced an entry (sec. 22)."""
    code, executed = CheckCode.NOT_DUPLICATE_SIGNAL, ctx.portfolio.executed_signal_ids
    signal_id = ctx.signal.signal_id
    if executed is None:
        return (_failed(code, SIGNAL_HISTORY_UNAVAILABLE, signal_id=signal_id),)
    if signal_id in executed:
        return (_failed(code, DUPLICATE_SIGNAL, signal_id=signal_id),)
    return (_passed(code, signal_id=signal_id),)


def check_cooldown(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``COOLDOWN_OK``: ``bars_since_last_exit >= cooldown_bars`` (sec. 5.4, 13.5)."""
    code, cooldown = CheckCode.COOLDOWN_OK, ctx.params.cooldown_bars
    if cooldown is None:
        return (_pending(code, "strategy.cooldown_bars"),)
    detail = {"bars_since_last_exit": ctx.bars_since_last_exit, "cooldown_bars": cooldown}
    if cooldown_active(ctx.bars_since_last_exit, cooldown):
        return (_failed(code, COOLDOWN_ACTIVE, **detail),)
    return (_passed(code, **detail),)


# --------------------------------------------------------------------------- trade / risk


def check_exit_levels_valid(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``EXIT_LEVELS_VALID``: sec. 16.4 on the trade (``domain.risk.exits.check_exit_levels``)."""
    code, trade = CheckCode.EXIT_LEVELS_VALID, ctx.trade
    if trade is None:
        return (_failed(code, EXIT_LEVELS_UNAVAILABLE, reason="no priced trade"),)
    return (
        _tag(
            code,
            check_exit_levels(
                trade.entry_ref,
                trade.stop_price,
                trade.take_profit_price,
                ctx.params.min_tp_distance_ticks,
            ),
        ),
    )


def check_risk_limits(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``RISK_LIMITS_OK``: every limit of sec. 15.1 with the new trade included.

    Delegates to ``domain.risk.risk_engine.evaluate_limits`` (eight results, one per
    limit, codes such as ``DAILY_LOSS_LIMIT`` or ``MAX_DRAWDOWN``).
    """
    code, trade, portfolio = CheckCode.RISK_LIMITS_OK, ctx.trade, ctx.portfolio
    missing = [
        name
        for name, value in (
            ("trade", trade),
            ("account", portfolio.account),
            ("positions", portfolio.positions),
        )
        if value is None
    ]
    if trade is None or portfolio.account is None or portfolio.positions is None:
        return (_failed(code, RISK_LIMITS_UNAVAILABLE, missing=missing),)
    state = RiskState(
        equity=portfolio.account.equity,
        last_equity=portfolio.account.last_equity,
        week_start_equity=portfolio.week_start_equity,
        peak_equity=portfolio.peak_equity,
        open_positions=portfolio.positions,
        pending_entries=portfolio.pending_entries,
        position_stops=portfolio.position_stops,
    )
    return tuple(_tag(code, result) for result in evaluate_limits(ctx.params.risk, state, trade))


def check_buying_power(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``BUYING_POWER_OK``: ``qty * (entry_ref + slippage buffer) <= buying_power``.

    The slippage buffer is ``entry_ref * slippage_buffer_bps / 10000`` as in the sizing
    of sec. 15.2 (whose ``qty_buying_power`` this verifies again with the current
    account, e.g. in the Execution Guard after a refresh).
    """
    code, trade, account = CheckCode.BUYING_POWER_OK, ctx.trade, ctx.portfolio.account
    if trade is None or account is None:
        return (
            _failed(
                code,
                BUYING_POWER_UNAVAILABLE,
                missing=[n for n, v in (("trade", trade), ("account", account)) if v is None],
            ),
        )
    bps = ctx.params.risk.slippage_buffer_bps
    if bps is None:
        return (_pending(code, "risk.slippage_buffer_bps"),)
    required = trade.qty * (trade.entry_ref + trade.entry_ref * bps / BPS_PER_UNIT)
    detail = {"required": str(required), "buying_power": str(account.buying_power)}
    if required > account.buying_power:
        return (_failed(code, BUYING_POWER_INSUFFICIENT, **detail),)
    return (_passed(code, **detail),)


# --------------------------------------------------------------------------- system state


def check_circuit_breaker(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``CIRCUIT_BREAKER_NORMAL``: breaker ``NORMAL`` (sec. 19, invariant 51.13)."""
    code, state = CheckCode.CIRCUIT_BREAKER_NORMAL, ctx.runtime.breaker_state
    if state is None:
        return (_failed(code, CIRCUIT_BREAKER_UNAVAILABLE),)
    if state is not CircuitBreakerState.NORMAL:
        return (_failed(code, CIRCUIT_BREAKER_NOT_NORMAL, state=state.value),)
    return (_passed(code, state=state.value),)


def check_trading_enabled(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``TRADING_ENABLED``: ``trading_enabled``, no STOP file, no emergency close (31.1-31.3).

    One failing result per active kill-switch reason (:data:`KILL_SWITCH_CODES`).
    """
    code, control = CheckCode.TRADING_ENABLED, ctx.runtime.control
    if control is None:
        return (_failed(code, CONTROL_STATE_UNAVAILABLE),)
    reasons = kill_switch_reasons(control)
    if reasons:
        return tuple(_failed(code, reason, kill_switch=True) for reason in reasons)
    return (_passed(code),)


def check_system_running(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``SYSTEM_RUNNING``: only ``RUNNING`` allows new entries (sec. 32)."""
    code, mode = CheckCode.SYSTEM_RUNNING, ctx.runtime.system_mode
    if mode is None:
        return (_failed(code, SYSTEM_MODE_UNAVAILABLE),)
    if mode is not SystemMode.RUNNING:
        return (_failed(code, SYSTEM_NOT_RUNNING, mode=mode.value),)
    return (_passed(code, mode=mode.value),)


def check_state_reconciled(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``STATE_RECONCILED``: no unresolved mismatch and a recent clean reconciliation."""
    code, facts = CheckCode.STATE_RECONCILED, ctx.runtime.reconciliation
    if facts is None:
        return (_failed(code, RECONCILIATION_UNAVAILABLE),)
    mismatch = (
        _failed(code, STATE_MISMATCH_UNRESOLVED)
        if facts.unresolved_mismatch
        else _passed(code, part="mismatch")
    )
    max_age = ctx.params.max_reconcile_age_seconds
    if max_age is None:
        return (mismatch, _pending(code, "execution.reconcile_interval_seconds"))
    last = facts.last_clean_at_utc
    if last is None:
        return (mismatch, _failed(code, RECONCILIATION_STALE, age_seconds=None))
    age = (ctx.now_utc - last).total_seconds()
    detail = {"age_seconds": age, "max_age_seconds": max_age}
    recent = (
        _failed(code, RECONCILIATION_STALE, **detail)
        if age > max_age
        else _passed(code, part="recency", **detail)
    )
    return (mismatch, recent)


def check_config_complete(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """``CONFIG_COMPLETE``: no required OWNER_DECISION is ``null`` (sec. 7.4, AC-20)."""
    code, pending = CheckCode.CONFIG_COMPLETE, ctx.runtime.pending_owner_decisions
    if pending is None:
        return (_failed(code, CONFIG_STATUS_UNAVAILABLE),)
    if pending:
        return (_failed(code, CONFIG_INCOMPLETE, pending=list(pending)),)
    return (_passed(code),)


# --------------------------------------------------------------------------- execution only


def _check_symbol_lock(ctx: CheckContext, facts: ExecutionFacts) -> tuple[CheckResult, ...]:
    code = CheckCode.SYMBOL_LOCK_HELD
    if not facts.symbol_lock_held:
        return (_failed(code, SYMBOL_LOCK_NOT_HELD, symbol=ctx.symbol),)
    return (_passed(code, symbol=ctx.symbol),)


def _check_client_order_id(ctx: CheckContext, facts: ExecutionFacts) -> tuple[CheckResult, ...]:
    code, seen = CheckCode.CLIENT_ORDER_ID_UNUSED, facts.client_order_id_seen
    detail = {"client_order_id": facts.client_order_id}
    if seen is None:
        return (_failed(code, ORDER_HISTORY_UNAVAILABLE, **detail),)
    if seen:
        return (_failed(code, DUPLICATE_CLIENT_ORDER_ID, **detail),)
    return (_passed(code, **detail),)


def _check_ai_verdict(ctx: CheckContext, facts: ExecutionFacts) -> tuple[CheckResult, ...]:
    code, control = CheckCode.AI_VERDICT_OK, ctx.runtime.control
    if control is None:
        return (_failed(code, CONTROL_STATE_UNAVAILABLE),)
    mode = control.ai_mode
    if mode is not AIMode.ACTIVE:  # DISABLED / SHADOW: the verdict never gates (AC-17)
        return (_passed(code, ai_mode=mode.value, required=False),)
    result = facts.ai_result
    detail: dict[str, Any] = {"ai_mode": mode.value, "required": True}
    if result is None:
        return (_failed(code, AIApprovalCode.AI_APPROVAL_MISSING.value, **detail),)
    if result.validity is not AIValidity.VALID or result.verdict is None:
        detail["validity"] = result.validity.value
        return (_failed(code, AIApprovalCode.AI_RESPONSE_NOT_VALID.value, **detail),)
    if result.verdict.signal_id != ctx.signal.signal_id:
        return (_failed(code, AIApprovalCode.AI_SIGNAL_MISMATCH.value, **detail),)
    if result.verdict.verdict is not AIVerdictKind.APPROVE:
        return (_failed(code, AIApprovalCode.AI_VETOED.value, **detail),)
    return (_passed(code, **detail),)


# --------------------------------------------------------------------------- entry points

PRE_AI_CHECKS: Final[tuple[tuple[CheckCode, Check], ...]] = (
    (CheckCode.SYMBOL_ALLOWED, check_symbol_allowed),
    (CheckCode.MARKET_OPEN, check_market_open),
    (CheckCode.ENTRY_WINDOW, check_entry_window),
    (CheckCode.BAR_CLOSED, check_bar_closed),
    (CheckCode.SIGNAL_NOT_EXPIRED, check_signal_not_expired),
    (CheckCode.DATA_FRESH, check_data_fresh),
    (CheckCode.NO_EXISTING_POSITION, check_no_existing_position),
    (CheckCode.NO_PENDING_ORDER, check_no_pending_order),
    (CheckCode.NOT_DUPLICATE_SIGNAL, check_not_duplicate_signal),
    (CheckCode.COOLDOWN_OK, check_cooldown),
    (CheckCode.PRICE_RANGE, check_price_range),
    (CheckCode.LIQUIDITY_OK, check_liquidity),
    (CheckCode.SPREAD_OK, check_spread),
    (CheckCode.EXIT_LEVELS_VALID, check_exit_levels_valid),
    (CheckCode.RISK_LIMITS_OK, check_risk_limits),
    (CheckCode.BUYING_POWER_OK, check_buying_power),
    (CheckCode.CIRCUIT_BREAKER_NORMAL, check_circuit_breaker),
    (CheckCode.TRADING_ENABLED, check_trading_enabled),
    (CheckCode.SYSTEM_RUNNING, check_system_running),
    (CheckCode.STATE_RECONCILED, check_state_reconciled),
    (CheckCode.CONFIG_COMPLETE, check_config_complete),
)
"""Every check of the sec. 19 table, in table order."""

ExecutionCheck = Callable[[CheckContext, ExecutionFacts], tuple[CheckResult, ...]]

EXECUTION_CHECKS: Final[tuple[tuple[CheckCode, ExecutionCheck], ...]] = (
    (CheckCode.SYMBOL_LOCK_HELD, _check_symbol_lock),
    (CheckCode.CLIENT_ORDER_ID_UNUSED, _check_client_order_id),
    (CheckCode.AI_VERDICT_OK, _check_ai_verdict),
)
"""Execution Guard preconditions evaluated after the sec. 19 checks (sec. 20)."""


def run_pre_ai_checks(ctx: CheckContext) -> tuple[CheckResult, ...]:
    """Evaluate every sec. 19 check (no short-circuit) and return all results in order."""
    return tuple(result for _, check in PRE_AI_CHECKS for result in check(ctx))


def run_execution_checks(ctx: CheckContext, facts: ExecutionFacts) -> tuple[CheckResult, ...]:
    """Execution Guard (sec. 20.4-20.6): every sec. 19 check again, then the preconditions.

    ``ctx`` must be built from the state refreshed just before the submission (broker
    account, positions and open orders, ``system_control`` and the STOP file). Nothing is
    skipped: the result holds every sec. 19 result followed by the lock, idempotency and
    AI-verdict results.
    """
    extra = tuple(result for _, check in EXECUTION_CHECKS for result in check(ctx, facts))
    return (*run_pre_ai_checks(ctx), *extra)


def checks_of(results: Iterable[CheckResult], check: CheckCode) -> tuple[CheckResult, ...]:
    """Results that belong to ``check`` (by ``detail["check"]``)."""
    return tuple(r for r in results if r.detail.get("check") == check.value)
