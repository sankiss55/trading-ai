"""Deterministic strategy: bars -> BUY / NO_ACTION / HOLD / CLOSE (sec. 5.2, 13).

Responsibilities (and non-responsibilities):

* Entries (:meth:`Strategy.evaluate_entry`, called when the symbol has **no** position):
  ``BUY`` only if **all** entry rules are true and **no** no-trade rule is true (13.4).
  Owner no-trade rules come from config; these built-in no-trade rules always apply:

  - ``NOTRADE_INCOMPLETE_BAR``: the signal bar does not allow entries, see
    :func:`bar_allows_entry` (sec. 10.3.4/10.3.5, 13.5).
  - ``NOTRADE_COOLDOWN``: ``bars_since_last_exit < cooldown_bars`` (sec. 5.4, 13.5).
  - ``NOTRADE_DATA_UNAVAILABLE``: some owner no-trade rule could not be evaluated
    because a value was ``None``. Sec. 14.3 makes such a rule ``False``, which for a
    no-trade rule would silently disable the protection; this built-in restores the
    fail-closed default of sec. 3.4 (NO NEW TRADE under data uncertainty).

  If the signal bar is EMPTY no owner rule is evaluated at all (sec. 10.3.4): only the
  built-ins are recorded and the action is ``NO_ACTION``.
* Positions (:meth:`Strategy.evaluate_position`, called when the symbol **has** a
  position): ``CLOSE`` with the :class:`~domain.models.ExitReason` of the system-run
  exits of sec. 13.6, else ``HOLD``:

  - ``EXIT_TIME_STOP`` -> ``TIME_STOP``: ``bars_held >= time_stop_bars``
    (``time_stop_bars = None`` disables it). Evaluated on every bar, EMPTY included.
  - Owner ``exit_rules`` -> ``SIGNAL_REVERSAL``, only if ``exit_on_signal_reversal`` is
    true; the position is closed if **any** exit rule is true. Not evaluated on EMPTY
    bars. Time stop wins when both fire.

  Stop loss / take profit are broker-side bracket legs and end-of-day flatten /
  emergency close are system procedures (23.5, 31.3): none is evaluated here.
* Safety checks (stale data, market closed, entry window, risk, existing position,
  duplicate signal...) are **not** strategy rules (13.5, 19); they live in the single
  check library. :func:`bar_allows_entry` and :func:`cooldown_active` are exported so
  that library reuses them instead of reimplementing them.

Counting bars: ``bars_since_last_exit`` and ``bars_held`` are numbers of closed
primary-timeframe bars (EMPTY included, they are elapsed time) whose ``bar_end_utc`` is
after the event; :func:`count_bars_since` computes that from a bar sequence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Final, Literal, Self, TypeVar

from pydantic import Field, model_validator

from domain.market.session import OwnerDecisionPendingError
from domain.models import (
    Bar,
    BarStatus,
    DomainModel,
    ExitReason,
    HoldingMode,
    Money,
    PositiveDecimal,
    RuleResult,
    SessionDay,
    Signal,
    StrategyAction,
    Symbol,
    Timeframe,
    UtcDatetime,
)
from domain.strategy.features import (
    SERIES_PERIOD_PARAM,
    IndicatorContext,
    IndicatorParams,
    build_indicator_context,
)
from domain.strategy.rules import (
    Identifier,
    RuleSpec,
    RuleTimeframe,
    StrategyConfigError,
    StrategyInputError,
    evaluate_rule,
    has_missing_values,
)
from domain.strategy.signals import build_signal

__all__ = [
    "BUILTIN_RULE_IDS",
    "EXIT_TIME_STOP",
    "NOTRADE_COOLDOWN",
    "NOTRADE_DATA_UNAVAILABLE",
    "NOTRADE_INCOMPLETE_BAR",
    "Strategy",
    "StrategyDecision",
    "StrategyExitParams",
    "StrategyParams",
    "bar_allows_entry",
    "cooldown_active",
    "count_bars_since",
    "pending_strategy_params",
    "required_warmup_bars",
]

NOTRADE_INCOMPLETE_BAR: Final = "NOTRADE_INCOMPLETE_BAR"
NOTRADE_COOLDOWN: Final = "NOTRADE_COOLDOWN"
NOTRADE_DATA_UNAVAILABLE: Final = "NOTRADE_DATA_UNAVAILABLE"
EXIT_TIME_STOP: Final = "EXIT_TIME_STOP"
BUILTIN_RULE_IDS: Final[frozenset[str]] = frozenset(
    {NOTRADE_INCOMPLETE_BAR, NOTRADE_COOLDOWN, NOTRADE_DATA_UNAVAILABLE, EXIT_TIME_STOP}
)

EntryRules = Annotated[tuple[RuleSpec, ...], Field(min_length=1)]


# --------------------------------------------------------------------------- params


class StrategyExitParams(DomainModel):
    """``strategy.exit`` of config.yaml. ``None`` = OWNER_DECISION pending.

    Only ``time_stop_bars`` (``None`` = no time stop, per config) and
    ``exit_on_signal_reversal`` are used by the strategy. The stop/take-profit fields are
    mapped here so the whole section validates, but they are consumed by the risk engine
    (``domain.risk.exits.ExitParams``), which owns their pending checks.
    """

    stop_method: Literal["atr"] | None
    stop_atr_multiplier: PositiveDecimal | None
    take_profit_r_multiple: PositiveDecimal | None
    time_stop_bars: int | None = Field(ge=1)
    exit_on_signal_reversal: bool | None
    min_tp_distance_ticks: int | None = Field(ge=0)


class StrategyParams(DomainModel):
    """The ``strategy`` section of config.yaml (sec. 7.3), validated structurally.

    ``None`` values are accepted here (observation mode may start with pending
    OWNER_DECISIONs, sec. 7.4.3); :class:`Strategy` refuses to be built while a value it
    needs is ``None`` (see :func:`pending_strategy_params`). No default is invented.

    ``exit_rules`` (owner signal-reversal rules) is not yet a key of config.yaml; it is
    optional here and required only when ``exit.exit_on_signal_reversal`` is true.
    ``holding_mode`` / ``flatten_minutes_before_close`` are mapped for completeness and
    consumed by ``domain.market.session``.
    """

    holding_mode: HoldingMode | None
    flatten_minutes_before_close: int | None = Field(ge=0)
    primary_timeframe: Timeframe | None
    confirmation_timeframe: Timeframe | None
    signal_ttl_seconds: int | None = Field(gt=0)
    cooldown_bars: int | None = Field(ge=0)
    min_minutes_per_bar: int | None = Field(ge=1)
    indicators: IndicatorParams
    entry_rules: EntryRules | None
    no_trade_rules: tuple[RuleSpec, ...] | None
    exit_rules: tuple[RuleSpec, ...] | None = None
    no_trade_thresholds: dict[Identifier, Money | None]
    exit: StrategyExitParams

    @model_validator(mode="after")
    def _check_structure(self) -> Self:
        rules = _all_rules(self)
        ids = [rule.rule_id for rule in rules]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate rule ids: {duplicates}")
        builtin = sorted(set(ids) & BUILTIN_RULE_IDS)
        if builtin:
            raise ValueError(f"rule ids reserved for built-in rules: {builtin}")
        unknown = sorted(
            {p for rule in rules for p in rule.referenced_params()} - set(self.no_trade_thresholds)
        )
        if unknown:
            raise ValueError(f"rules reference unknown no_trade_thresholds: {unknown}")
        minutes = self.primary_timeframe.minutes if self.primary_timeframe else None
        if (
            minutes is not None
            and self.min_minutes_per_bar is not None
            and self.min_minutes_per_bar > minutes
        ):
            raise ValueError(
                f"min_minutes_per_bar={self.min_minutes_per_bar} exceeds the "
                f"{minutes} minutes of primary_timeframe {self.primary_timeframe}"
            )
        return self


def _all_rules(params: StrategyParams) -> tuple[RuleSpec, ...]:
    return (params.entry_rules or ()) + (params.no_trade_rules or ()) + (params.exit_rules or ())


def _active_rules(params: StrategyParams) -> tuple[RuleSpec, ...]:
    """Rules that will actually be evaluated (exit rules only if reversal is enabled)."""
    exits = (params.exit_rules or ()) if params.exit.exit_on_signal_reversal is True else ()
    return (params.entry_rules or ()) + (params.no_trade_rules or ()) + exits


def pending_strategy_params(params: StrategyParams) -> tuple[str, ...]:
    """Dotted names of the OWNER_DECISION parameters still ``None`` that the strategy needs.

    Always required: ``primary_timeframe``, ``signal_ttl_seconds``, ``cooldown_bars``,
    ``min_minutes_per_bar``, ``entry_rules``, ``no_trade_rules``,
    ``exit.exit_on_signal_reversal`` and ``indicators.atr_period`` (the ATR stop of
    sec. 16.1 needs it). Conditionally required: ``exit_rules`` when reversal is enabled;
    ``confirmation_timeframe``, other indicator periods and thresholds when an active
    rule references them (sec. 13.2: no indicator that no rule uses).
    """
    pending: list[str] = []
    for name in (
        "primary_timeframe",
        "signal_ttl_seconds",
        "cooldown_bars",
        "min_minutes_per_bar",
        "entry_rules",
        "no_trade_rules",
    ):
        if getattr(params, name) is None:
            pending.append(f"strategy.{name}")
    if params.exit.exit_on_signal_reversal is None:
        pending.append("strategy.exit.exit_on_signal_reversal")
    elif params.exit.exit_on_signal_reversal and params.exit_rules is None:
        pending.append("strategy.exit_rules")
    rules = _active_rules(params)
    resolved = [pair for rule in rules for pair in rule.resolved_series()]
    if params.confirmation_timeframe is None and any(
        tf is RuleTimeframe.CONFIRMATION for tf, _ in resolved
    ):
        pending.append("strategy.confirmation_timeframe")
    needed_periods = {"atr_period"} | {
        SERIES_PERIOD_PARAM[base] for _, base in resolved if base in SERIES_PERIOD_PARAM
    }
    for name in sorted(needed_periods):
        if getattr(params.indicators, name) is None:
            pending.append(f"strategy.indicators.{name}")
    for name in sorted({p for rule in rules for p in rule.referenced_params()}):
        if params.no_trade_thresholds.get(name) is None:
            pending.append(f"strategy.no_trade_thresholds.{name}")
    return tuple(pending)


def required_warmup_bars(params: StrategyParams) -> int:
    """Primary bars needed before every active rule can produce a non-``None`` value.

    ``max(min_bars of each configured indicator) + (max |offset| - 1)``; compare it with
    ``market_data.history_warmup_bars`` (sec. 7.4.4).
    """
    periods = params.indicators
    longest = max(
        (periods.min_bars(name) for name in IndicatorParams.model_fields),
        default=0,
    )
    offsets = [o.offset for rule in _active_rules(params) for o in rule.series_operands()]
    deepest = max((-offset for offset in offsets), default=1)
    return max(longest, 1) + deepest - 1


# --------------------------------------------------------------------------- helpers


def bar_allows_entry(bar: Bar, min_minutes_per_bar: int) -> bool:
    """Whether a closed bar may produce an entry (sec. 10.3.4, 10.3.5, 13.5).

    * ``COMPLETE`` -> True.
    * ``EMPTY`` -> False (never evaluated).
    * ``INCOMPLETE`` -> True only if ``minutes_present >= min_minutes_per_bar``.
    """
    if bar.status is BarStatus.COMPLETE:
        return True
    if bar.status is BarStatus.EMPTY:
        return False
    return bar.minutes_present is not None and bar.minutes_present >= min_minutes_per_bar


def cooldown_active(bars_since_last_exit: int | None, cooldown_bars: int) -> bool:
    """True while the symbol is still in cooldown (``None`` = no previous exit)."""
    return bars_since_last_exit is not None and bars_since_last_exit < cooldown_bars


def count_bars_since(bars: Iterable[Bar], since_utc: datetime) -> int:
    """Number of closed bars whose ``bar_end_utc`` is after ``since_utc`` (EMPTY included).

    Only the given bars are counted: pass a window that covers ``since_utc``.
    """
    if since_utc.tzinfo is None:
        raise StrategyInputError("since_utc must be timezone-aware (UTC)")
    return sum(1 for bar in bars if bar.bar_end_utc > since_utc)


# --------------------------------------------------------------------------- decision


class StrategyDecision(DomainModel):
    """Output of one strategy evaluation on one closed primary bar.

    ``signal`` is set iff ``action`` is BUY; ``exit_reason`` iff ``action`` is CLOSE.
    ``rule_results`` holds every evaluated rule (owner rules first, then built-ins).
    """

    action: StrategyAction
    symbol: Symbol
    timeframe: Timeframe
    bar_start_utc: UtcDatetime
    bar_end_utc: UtcDatetime
    bar_status: BarStatus
    rule_results: tuple[RuleResult, ...]
    signal: Signal | None = None
    exit_reason: ExitReason | None = None
    strategy_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if (self.action is StrategyAction.BUY) != (self.signal is not None):
            raise ValueError("signal must be set if and only if action is BUY")
        if (self.action is StrategyAction.CLOSE) != (self.exit_reason is not None):
            raise ValueError("exit_reason must be set if and only if action is CLOSE")
        return self


_T = TypeVar("_T")


def _require(value: _T | None, name: str) -> _T:
    if value is None:
        raise OwnerDecisionPendingError(name)
    return value


class Strategy:
    """Deterministic rule-based strategy for one ``strategy_version``.

    Args:
        params: The ``strategy`` config section.
        strategy_version: Top-level ``strategy_version`` of config.yaml (part of the
            signal id, sec. 13.7).

    Raises:
        OwnerDecisionPendingError: a parameter the strategy needs is still ``None``;
            names the first one of :func:`pending_strategy_params`.
        StrategyConfigError: empty ``strategy_version``.
    """

    def __init__(self, params: StrategyParams, *, strategy_version: str) -> None:
        if not strategy_version:
            raise StrategyConfigError("strategy_version must not be empty")
        pending = pending_strategy_params(params)
        if pending:
            raise OwnerDecisionPendingError(pending[0])
        self._params = params
        self._version = strategy_version
        self._primary_timeframe: Timeframe = _require(
            params.primary_timeframe, "strategy.primary_timeframe"
        )
        self._ttl: int = _require(params.signal_ttl_seconds, "strategy.signal_ttl_seconds")
        self._cooldown: int = _require(params.cooldown_bars, "strategy.cooldown_bars")
        self._min_minutes: int = _require(
            params.min_minutes_per_bar, "strategy.min_minutes_per_bar"
        )
        self._entry_rules: tuple[RuleSpec, ...] = _require(
            params.entry_rules, "strategy.entry_rules"
        )
        self._no_trade_rules: tuple[RuleSpec, ...] = _require(
            params.no_trade_rules, "strategy.no_trade_rules"
        )
        reversal = _require(
            params.exit.exit_on_signal_reversal, "strategy.exit.exit_on_signal_reversal"
        )
        self._exit_rules: tuple[RuleSpec, ...] = (
            _require(params.exit_rules, "strategy.exit_rules") if reversal else ()
        )
        self._thresholds: dict[str, Decimal | None] = dict(params.no_trade_thresholds)

    @property
    def params(self) -> StrategyParams:
        """The validated parameters."""
        return self._params

    @property
    def strategy_version(self) -> str:
        """Version string embedded in every signal id."""
        return self._version

    def build_context(
        self,
        primary_bars: Sequence[Bar],
        confirmation_bars: Sequence[Bar] | None = None,
        *,
        sessions: Sequence[SessionDay] = (),
    ) -> IndicatorContext:
        """Build the indicator context from closed bars of one symbol.

        Args:
            primary_bars: Closed primary bars, oldest first; the last one is the bar the
                decision refers to (it may be EMPTY or INCOMPLETE).
            confirmation_bars: Closed confirmation bars (ignored bars ending after the
                signal bar are dropped). Required if a confirmation timeframe is set.
            sessions: Consecutive calendar sessions covering the bars (``gap_pct``).
        """
        return build_indicator_context(
            primary_bars,
            confirmation_bars,
            primary_timeframe=self._primary_timeframe,
            confirmation_timeframe=self._params.confirmation_timeframe,
            periods=self._params.indicators,
            sessions=sessions,
        )

    def _signal_bar(self, context: IndicatorContext) -> Bar:
        bar = context.signal_bar
        if bar is None:
            raise StrategyInputError("no primary bar to evaluate")
        if bar.timeframe is not self._primary_timeframe:
            raise StrategyInputError(
                f"signal bar timeframe {bar.timeframe} != {self._primary_timeframe}"
            )
        return bar

    def _decision(
        self,
        bar: Bar,
        action: StrategyAction,
        results: Sequence[RuleResult],
        *,
        signal: Signal | None = None,
        exit_reason: ExitReason | None = None,
    ) -> StrategyDecision:
        return StrategyDecision(
            action=action,
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            bar_start_utc=bar.bar_start_utc,
            bar_end_utc=bar.bar_end_utc,
            bar_status=bar.status,
            rule_results=tuple(results),
            signal=signal,
            exit_reason=exit_reason,
            strategy_version=self._version,
        )

    def evaluate_entry(
        self,
        context: IndicatorContext,
        *,
        bars_since_last_exit: int | None,
        created_at_utc: datetime,
    ) -> StrategyDecision:
        """Decide ``BUY`` or ``NO_ACTION`` for a symbol without an open position.

        Args:
            context: Output of :meth:`build_context`.
            bars_since_last_exit: Closed primary bars since the last exit of this symbol
                (``None`` = no previous exit); see :func:`count_bars_since`.
            created_at_utc: Decision time (from ``IClock``), stored in the signal.

        Raises:
            StrategyInputError: empty context, wrong timeframe or negative bar count.
        """
        bar = self._signal_bar(context)
        if bars_since_last_exit is not None and bars_since_last_exit < 0:
            raise StrategyInputError("bars_since_last_exit must be >= 0")
        incomplete = RuleResult(
            rule_id=NOTRADE_INCOMPLETE_BAR,
            result=not bar_allows_entry(bar, self._min_minutes),
            values={
                "bar_status": bar.status.value,
                "minutes_present": bar.minutes_present,
                "min_minutes_per_bar": self._min_minutes,
            },
        )
        cooldown = RuleResult(
            rule_id=NOTRADE_COOLDOWN,
            result=cooldown_active(bars_since_last_exit, self._cooldown),
            values={"bars_since_last_exit": bars_since_last_exit, "cooldown_bars": self._cooldown},
        )
        if bar.status is BarStatus.EMPTY:
            return self._decision(bar, StrategyAction.NO_ACTION, (incomplete, cooldown))

        entry = [evaluate_rule(r, context, self._thresholds) for r in self._entry_rules]
        no_trade = [evaluate_rule(r, context, self._thresholds) for r in self._no_trade_rules]
        unevaluable = [r.rule_id for r in no_trade if has_missing_values(r)]
        data_unavailable = RuleResult(
            rule_id=NOTRADE_DATA_UNAVAILABLE,
            result=bool(unevaluable),
            values={"rules": ",".join(unevaluable)},
        )
        results = (*entry, *no_trade, incomplete, cooldown, data_unavailable)
        blocked = any(r.result for r in (*no_trade, incomplete, cooldown, data_unavailable))
        if not all(r.result for r in entry) or blocked:
            return self._decision(bar, StrategyAction.NO_ACTION, results)
        signal = build_signal(
            strategy_version=self._version,
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            bar_start_utc=bar.bar_start_utc,
            bar_end_utc=bar.bar_end_utc,
            created_at_utc=created_at_utc,
            signal_ttl_seconds=self._ttl,
            rule_results=results,
        )
        return self._decision(bar, StrategyAction.BUY, results, signal=signal)

    def evaluate_position(self, context: IndicatorContext, *, bars_held: int) -> StrategyDecision:
        """Decide ``HOLD`` or ``CLOSE`` (with exit reason) for an open position.

        Args:
            context: Output of :meth:`build_context`.
            bars_held: Closed primary bars since the entry fill (see
                :func:`count_bars_since`).

        Raises:
            StrategyInputError: empty context, wrong timeframe or negative bar count.
        """
        bar = self._signal_bar(context)
        if bars_held < 0:
            raise StrategyInputError("bars_held must be >= 0")
        limit = self._params.exit.time_stop_bars
        time_stop = RuleResult(
            rule_id=EXIT_TIME_STOP,
            result=limit is not None and bars_held >= limit,
            values={"bars_held": bars_held, "time_stop_bars": limit},
        )
        reversal: list[RuleResult] = []
        if bar.status is not BarStatus.EMPTY:
            reversal = [evaluate_rule(r, context, self._thresholds) for r in self._exit_rules]
        results = (*reversal, time_stop)
        if time_stop.result:
            return self._decision(
                bar, StrategyAction.CLOSE, results, exit_reason=ExitReason.TIME_STOP
            )
        if any(r.result for r in reversal):
            return self._decision(
                bar, StrategyAction.CLOSE, results, exit_reason=ExitReason.SIGNAL_REVERSAL
            )
        return self._decision(bar, StrategyAction.HOLD, results)
