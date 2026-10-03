"""Circuit breaker (sec. 30): states, triggers and the pure transition function.

States (sec. 30.1): ``NORMAL`` < ``WARNING`` < ``HALTED`` < ``EMERGENCY`` (severity
order). The breaker state is the most severe of:

* the **latched** triggers (``HALTED`` / ``EMERGENCY``), which stay tripped after their
  condition disappears until their reset rule allows it, and
* the **warning** triggers (``WARNING``), active exactly while their condition holds.

Triggers and reset rules (sec. 30.2, :data:`TRIGGER_RULES`):

==============================  ===========  ==============  ==========================
Trigger                         State        Reset           Procedure
==============================  ===========  ==============  ==========================
DAILY_LOSS_LIMIT                HALTED       next session
WEEKLY_LOSS_LIMIT               HALTED       manual
MAX_DRAWDOWN                    HALTED       manual
ORDER_REJECTIONS                HALTED       next session
API_ERRORS_REPEATED             WARNING      automatic
API_ERRORS_PERSISTENT           HALTED       manual
MARKET_DATA_STALE_MAJORITY      WARNING      automatic
MARKET_DATA_STREAM_DOWN         HALTED       manual
DB_NOT_WRITABLE                 HALTED       manual
STATE_MISMATCH                  HALTED       manual
ORPHANED_POSITION               HALTED       manual
UNPROTECTED_POSITION            HALTED       manual          23.7 protect position
BROKER_STATE_UNKNOWN            HALTED       manual
CLOCK_SKEW                      WARNING      automatic       (system mode DEGRADED)
AI_UNAVAILABLE_PERSISTENT       WARNING      automatic
EMERGENCY_CLOSE                 EMERGENCY    manual          31.3 emergency close
==============================  ===========  ==============  ==========================

"Next session" triggers are released automatically by the first evaluation of a later
session in which the condition is gone. "Manual" triggers are released only by
:func:`clear_halt` (the ``clear-halt`` control command, sec. 31.4), which
:func:`can_clear` gates: a recent clean reconciliation, ``emergency_close`` back to
``false`` and no latched condition still present. Where sec. 30.2 names a HALTED
trigger without a reset rule, manual reset applies (sec. 32: ``HALTED -> READY`` only by
a manual ``clear-halt``).

New entries (sec. 19, invariant 51.13): only ``NORMAL`` allows them
(:func:`blocks_new_entries`); the check library's ``CIRCUIT_BREAKER_NORMAL`` enforces it.
Sec. 30.1 lists ``WARNING`` as allowing entries; the stricter invariant is applied.
System exits and protection continue in every state; ``EMERGENCY`` closes everything.

Everything here is pure: the caller provides the observations (:class:`BreakerSignals`)
and persists the returned :class:`BreakerStatus` and transition (sec. 30.2: every
transition is recorded with its cause and notified).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Final, Self

from pydantic import Field, model_validator

from domain.errors import NonRetryableError
from domain.market.quality import StaleReport
from domain.models import (
    CheckResult,
    CircuitBreakerState,
    DomainModel,
    NonNegativeDecimal,
    PositiveDecimal,
    ReconciliationOutcome,
    Symbol,
    UtcDatetime,
)
from domain.risk.risk_engine import DAILY_LOSS_LIMIT, MAX_DRAWDOWN, WEEKLY_LOSS_LIMIT

__all__ = [
    "CLEAR_HALT_REFUSED",
    "SEVERITY",
    "TRIGGER_RULES",
    "ApiErrorLevel",
    "BreakerDecision",
    "BreakerParams",
    "BreakerSignals",
    "BreakerStatus",
    "BreakerTransition",
    "BreakerTrigger",
    "ClearDecision",
    "ClearHaltRefusedError",
    "ClearRefusal",
    "LatchedTrigger",
    "Procedure",
    "ResetPolicy",
    "TriggerRule",
    "blocks_new_entries",
    "can_clear",
    "clear_halt",
    "detect_triggers",
    "evaluate_breaker",
    "requires_close_all",
]

CLEAR_HALT_REFUSED: Final = "CLEAR_HALT_REFUSED"

_S = CircuitBreakerState

SEVERITY: Final[Mapping[CircuitBreakerState, int]] = MappingProxyType(
    {_S.NORMAL: 0, _S.WARNING: 1, _S.HALTED: 2, _S.EMERGENCY: 3}
)
"""Severity order of the breaker states."""


class BreakerTrigger(StrEnum):
    """Trigger of sec. 30.2 (stable code recorded with every transition)."""

    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    WEEKLY_LOSS_LIMIT = "WEEKLY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    ORDER_REJECTIONS = "ORDER_REJECTIONS"
    API_ERRORS_REPEATED = "API_ERRORS_REPEATED"
    API_ERRORS_PERSISTENT = "API_ERRORS_PERSISTENT"
    MARKET_DATA_STALE_MAJORITY = "MARKET_DATA_STALE_MAJORITY"
    MARKET_DATA_STREAM_DOWN = "MARKET_DATA_STREAM_DOWN"
    DB_NOT_WRITABLE = "DB_NOT_WRITABLE"
    STATE_MISMATCH = "STATE_MISMATCH"
    ORPHANED_POSITION = "ORPHANED_POSITION"
    UNPROTECTED_POSITION = "UNPROTECTED_POSITION"
    BROKER_STATE_UNKNOWN = "BROKER_STATE_UNKNOWN"
    CLOCK_SKEW = "CLOCK_SKEW"
    AI_UNAVAILABLE_PERSISTENT = "AI_UNAVAILABLE_PERSISTENT"
    EMERGENCY_CLOSE = "EMERGENCY_CLOSE"


class ResetPolicy(StrEnum):
    """How a tripped trigger is released."""

    AUTOMATIC = "AUTOMATIC"
    """Active only while the condition holds (``WARNING`` triggers)."""
    NEXT_SESSION = "NEXT_SESSION"
    """Released by the first evaluation of a later session without the condition."""
    MANUAL = "MANUAL"
    """Released only by :func:`clear_halt` (``clear-halt``, sec. 31.4)."""


class Procedure(StrEnum):
    """Procedure the caller must run while the trigger is active."""

    PROTECT_POSITION = "PROTECT_POSITION"
    """Sec. 23.7: protect an unprotected position immediately (invariant 51.6)."""
    EMERGENCY_CLOSE = "EMERGENCY_CLOSE"
    """Sec. 31.3: cancel entries, close every system position, verify, notify."""


class TriggerRule(DomainModel):
    """Resulting state, reset policy and procedure of one trigger."""

    state: CircuitBreakerState
    reset: ResetPolicy
    procedure: Procedure | None = None


_T = BreakerTrigger
_R = ResetPolicy

TRIGGER_RULES: Final[Mapping[BreakerTrigger, TriggerRule]] = MappingProxyType(
    {
        _T.DAILY_LOSS_LIMIT: TriggerRule(state=_S.HALTED, reset=_R.NEXT_SESSION),
        _T.WEEKLY_LOSS_LIMIT: TriggerRule(state=_S.HALTED, reset=_R.MANUAL),
        _T.MAX_DRAWDOWN: TriggerRule(state=_S.HALTED, reset=_R.MANUAL),
        _T.ORDER_REJECTIONS: TriggerRule(state=_S.HALTED, reset=_R.NEXT_SESSION),
        _T.API_ERRORS_REPEATED: TriggerRule(state=_S.WARNING, reset=_R.AUTOMATIC),
        _T.API_ERRORS_PERSISTENT: TriggerRule(state=_S.HALTED, reset=_R.MANUAL),
        _T.MARKET_DATA_STALE_MAJORITY: TriggerRule(state=_S.WARNING, reset=_R.AUTOMATIC),
        _T.MARKET_DATA_STREAM_DOWN: TriggerRule(state=_S.HALTED, reset=_R.MANUAL),
        _T.DB_NOT_WRITABLE: TriggerRule(state=_S.HALTED, reset=_R.MANUAL),
        _T.STATE_MISMATCH: TriggerRule(state=_S.HALTED, reset=_R.MANUAL),
        _T.ORPHANED_POSITION: TriggerRule(state=_S.HALTED, reset=_R.MANUAL),
        _T.UNPROTECTED_POSITION: TriggerRule(
            state=_S.HALTED, reset=_R.MANUAL, procedure=Procedure.PROTECT_POSITION
        ),
        _T.BROKER_STATE_UNKNOWN: TriggerRule(state=_S.HALTED, reset=_R.MANUAL),
        _T.CLOCK_SKEW: TriggerRule(state=_S.WARNING, reset=_R.AUTOMATIC),
        _T.AI_UNAVAILABLE_PERSISTENT: TriggerRule(state=_S.WARNING, reset=_R.AUTOMATIC),
        _T.EMERGENCY_CLOSE: TriggerRule(
            state=_S.EMERGENCY, reset=_R.MANUAL, procedure=Procedure.EMERGENCY_CLOSE
        ),
    }
)
"""Sec. 30.2, one rule per trigger (see the module table)."""

_LOSS_TRIGGERS: Final[Mapping[str, BreakerTrigger]] = MappingProxyType(
    {
        DAILY_LOSS_LIMIT: _T.DAILY_LOSS_LIMIT,
        WEEKLY_LOSS_LIMIT: _T.WEEKLY_LOSS_LIMIT,
        MAX_DRAWDOWN: _T.MAX_DRAWDOWN,
    }
)


class ApiErrorLevel(StrEnum):
    """Broker API error level as judged by the caller's error counters (sec. 30.2)."""

    NONE = "NONE"
    REPEATED = "REPEATED"
    PERSISTENT = "PERSISTENT"


# --------------------------------------------------------------------------- inputs


class BreakerParams(DomainModel):
    """Breaker thresholds from configuration.

    Attributes:
        max_order_rejections_per_day: ``execution.max_order_rejections_per_day``; the
            breaker trips when the day's rejections reach it (``>=``, sec. 30.2).
        max_clock_skew_seconds: ``system.max_clock_skew_seconds`` (sec. 11.6).
    """

    max_order_rejections_per_day: Annotated[int, Field(ge=0)]
    max_clock_skew_seconds: PositiveDecimal


class BreakerSignals(DomainModel):
    """Observations of one evaluation. Every field is required: nothing is assumed.

    Attributes:
        observed_at_utc: When the observations were taken (``IClock``).
        session_date: Trading session the observation belongs to (next-session resets).
        loss_limit_results: Results of ``risk_engine`` ``check_daily_loss``,
            ``check_weekly_loss`` and ``check_drawdown`` (e.g. the ``RISK_LIMITS_OK``
            results of the check library). A failed ``DAILY_LOSS_LIMIT`` /
            ``WEEKLY_LOSS_LIMIT`` / ``MAX_DRAWDOWN`` result trips the matching trigger.
        order_rejections_today: Broker order rejections in the session.
        api_errors: Broker API error level.
        stale_report: ``quality.assess_staleness`` of the whitelist (``None``: not
            evaluated in this observation, e.g. outside the session).
        market_data_stream_connected: The market data source is connected.
        db_writable: The operational database accepts writes.
        reconciliation_outcome: Latest reconciliation result (``None``: none yet).
        unprotected_symbols: Positions without an active stop in the broker.
        broker_state_known: Account, positions and orders could be read.
        clock_skew_seconds: ``|host - broker|`` clock difference (``None``: not
            measured; ``MARKET_OPEN`` already blocks entries without a broker clock).
        ai_unavailable_persistent: AI errors persist with ``ACTIVE`` + ``block``.
        emergency_close: ``system_control.emergency_close`` (sec. 31.3).
    """

    observed_at_utc: UtcDatetime
    session_date: date
    loss_limit_results: tuple[CheckResult, ...]
    order_rejections_today: Annotated[int, Field(ge=0)]
    api_errors: ApiErrorLevel
    stale_report: StaleReport | None
    market_data_stream_connected: bool
    db_writable: bool
    reconciliation_outcome: ReconciliationOutcome | None
    unprotected_symbols: tuple[Symbol, ...]
    broker_state_known: bool
    clock_skew_seconds: NonNegativeDecimal | None
    ai_unavailable_persistent: bool
    emergency_close: bool


# --------------------------------------------------------------------------- state


class LatchedTrigger(DomainModel):
    """A tripped ``HALTED`` / ``EMERGENCY`` trigger and when it tripped."""

    trigger: BreakerTrigger
    state: CircuitBreakerState
    reset: ResetPolicy
    tripped_at_utc: UtcDatetime
    session_date: date


class BreakerStatus(DomainModel):
    """Persistent breaker state: latched triggers plus the active warnings.

    ``state`` must equal the most severe state of ``latched`` and ``warnings``.
    """

    state: CircuitBreakerState
    latched: tuple[LatchedTrigger, ...]
    warnings: tuple[BreakerTrigger, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        expected = _state_of(self.latched, self.warnings)
        if self.state is not expected:
            raise ValueError(f"breaker state {self.state} inconsistent with triggers ({expected})")
        triggers = [item.trigger for item in self.latched]
        if len(triggers) != len(set(triggers)):
            raise ValueError("a trigger may be latched only once")
        if any(TRIGGER_RULES[t].reset is not ResetPolicy.AUTOMATIC for t in self.warnings):
            raise ValueError("warnings must be automatic-reset triggers")
        if any(item.reset is ResetPolicy.AUTOMATIC for item in self.latched):
            raise ValueError("automatic-reset triggers are never latched")
        return self

    @classmethod
    def normal(cls) -> Self:
        """No trigger: ``NORMAL``."""
        return cls(state=_S.NORMAL, latched=(), warnings=())


class BreakerTransition(DomainModel):
    """A state change with its causes (recorded in ``system_events`` and notified)."""

    from_state: CircuitBreakerState
    to_state: CircuitBreakerState
    causes: tuple[str, ...]
    occurred_at_utc: UtcDatetime


class BreakerDecision(DomainModel):
    """Output of :func:`evaluate_breaker` / :func:`clear_halt`.

    Attributes:
        status: The new state to persist.
        transition: Set only when the state changed.
        active: Triggers whose condition holds in this observation.
        tripped: Triggers latched by this evaluation.
        released: Triggers released by this evaluation (next session or manual clear).
        procedures: Procedures the caller must run now (sec. 23.7, 31.3).
    """

    status: BreakerStatus
    transition: BreakerTransition | None
    active: tuple[BreakerTrigger, ...]
    tripped: tuple[BreakerTrigger, ...]
    released: tuple[BreakerTrigger, ...]
    procedures: tuple[Procedure, ...]


# --------------------------------------------------------------------------- rules


def blocks_new_entries(state: CircuitBreakerState) -> bool:
    """Whether ``state`` blocks new entries: every state but ``NORMAL`` (19, 51.13)."""
    return state is not _S.NORMAL


def requires_close_all(state: CircuitBreakerState) -> bool:
    """Whether ``state`` requires closing every system position (``EMERGENCY``, 31.3)."""
    return state is _S.EMERGENCY


def _state_of(
    latched: tuple[LatchedTrigger, ...], warnings: tuple[BreakerTrigger, ...]
) -> CircuitBreakerState:
    states = [item.state for item in latched] + [TRIGGER_RULES[t].state for t in warnings]
    return max(states, key=SEVERITY.__getitem__, default=_S.NORMAL)


def detect_triggers(signals: BreakerSignals, params: BreakerParams) -> tuple[BreakerTrigger, ...]:
    """Triggers whose condition holds in ``signals``, in :class:`BreakerTrigger` order."""
    found: set[BreakerTrigger] = set()
    for result in signals.loss_limit_results:
        trigger = _LOSS_TRIGGERS.get(result.code)
        if trigger is not None and not result.passed:
            found.add(trigger)
    if signals.order_rejections_today >= params.max_order_rejections_per_day:
        found.add(_T.ORDER_REJECTIONS)
    if signals.api_errors is ApiErrorLevel.REPEATED:
        found.add(_T.API_ERRORS_REPEATED)
    elif signals.api_errors is ApiErrorLevel.PERSISTENT:
        found.add(_T.API_ERRORS_PERSISTENT)
    if signals.stale_report is not None and signals.stale_report.circuit_breaker_warning:
        found.add(_T.MARKET_DATA_STALE_MAJORITY)
    if not signals.market_data_stream_connected:
        found.add(_T.MARKET_DATA_STREAM_DOWN)
    if not signals.db_writable:
        found.add(_T.DB_NOT_WRITABLE)
    if signals.reconciliation_outcome is ReconciliationOutcome.STATE_MISMATCH:
        found.add(_T.STATE_MISMATCH)
    elif signals.reconciliation_outcome is ReconciliationOutcome.ORPHANED:
        found.add(_T.ORPHANED_POSITION)
    if signals.unprotected_symbols:
        found.add(_T.UNPROTECTED_POSITION)
    if not signals.broker_state_known:
        found.add(_T.BROKER_STATE_UNKNOWN)
    skew = signals.clock_skew_seconds
    if skew is not None and skew > params.max_clock_skew_seconds:
        found.add(_T.CLOCK_SKEW)
    if signals.ai_unavailable_persistent:
        found.add(_T.AI_UNAVAILABLE_PERSISTENT)
    if signals.emergency_close:
        found.add(_T.EMERGENCY_CLOSE)
    return tuple(trigger for trigger in BreakerTrigger if trigger in found)


def _ordered(triggers: set[BreakerTrigger]) -> tuple[BreakerTrigger, ...]:
    return tuple(trigger for trigger in BreakerTrigger if trigger in triggers)


def _decision(
    previous: BreakerStatus,
    latched: list[LatchedTrigger],
    active: tuple[BreakerTrigger, ...],
    *,
    tripped: tuple[BreakerTrigger, ...],
    released: tuple[BreakerTrigger, ...],
    at: UtcDatetime,
    extra_causes: tuple[str, ...] = (),
) -> BreakerDecision:
    warnings = tuple(t for t in active if TRIGGER_RULES[t].reset is ResetPolicy.AUTOMATIC)
    ordered = tuple(sorted(latched, key=lambda item: list(BreakerTrigger).index(item.trigger)))
    state = _state_of(ordered, warnings)
    status = BreakerStatus(state=state, latched=ordered, warnings=warnings)
    transition = None
    if state is not previous.state:
        causes = (*extra_causes, *(t.value for t in (*tripped, *released)))
        if not causes:  # a warning appeared or disappeared
            appeared = set(warnings) ^ set(previous.warnings)
            causes = tuple(t.value for t in _ordered(appeared))
        transition = BreakerTransition(
            from_state=previous.state, to_state=state, causes=causes, occurred_at_utc=at
        )
    procedures = tuple(
        dict.fromkeys(
            rule.procedure for t in active if (rule := TRIGGER_RULES[t]).procedure is not None
        )
    )
    return BreakerDecision(
        status=status,
        transition=transition,
        active=active,
        tripped=tripped,
        released=released,
        procedures=procedures,
    )


def evaluate_breaker(
    status: BreakerStatus, signals: BreakerSignals, params: BreakerParams
) -> BreakerDecision:
    """Pure transition function of the breaker (sec. 30.2).

    1. Detect the triggers active in ``signals``.
    2. Release latched ``NEXT_SESSION`` triggers tripped in an earlier session whose
       condition is gone. ``MANUAL`` triggers are never released here.
    3. Latch every active ``HALTED`` / ``EMERGENCY`` trigger not latched yet.
    4. Warnings are exactly the active automatic triggers.
    5. The state is the most severe of latched triggers and warnings; a change yields a
       :class:`BreakerTransition` with its causes.
    """
    active = detect_triggers(signals, params)
    active_set = set(active)
    released: set[BreakerTrigger] = set()
    latched: list[LatchedTrigger] = []
    for item in status.latched:
        if (
            item.reset is ResetPolicy.NEXT_SESSION
            and signals.session_date > item.session_date
            and item.trigger not in active_set
        ):
            released.add(item.trigger)
        else:
            latched.append(item)
    already = {item.trigger for item in latched}
    tripped: set[BreakerTrigger] = set()
    for trigger in active:
        rule = TRIGGER_RULES[trigger]
        if rule.reset is ResetPolicy.AUTOMATIC or trigger in already:
            continue
        tripped.add(trigger)
        latched.append(
            LatchedTrigger(
                trigger=trigger,
                state=rule.state,
                reset=rule.reset,
                tripped_at_utc=signals.observed_at_utc,
                session_date=signals.session_date,
            )
        )
    return _decision(
        status,
        latched,
        active,
        tripped=_ordered(tripped),
        released=_ordered(released),
        at=signals.observed_at_utc,
    )


# --------------------------------------------------------------------------- manual reset


class ClearRefusal(StrEnum):
    """Why ``clear-halt`` is refused (sec. 31.3, 31.4)."""

    NOTHING_TO_CLEAR = "NOTHING_TO_CLEAR"
    RECONCILIATION_NOT_CLEAN = "RECONCILIATION_NOT_CLEAN"
    EMERGENCY_CLOSE_STILL_SET = "EMERGENCY_CLOSE_STILL_SET"
    CONDITION_STILL_PRESENT = "CONDITION_STILL_PRESENT"


class ClearDecision(DomainModel):
    """Whether ``clear-halt`` may run, with every refusal at once."""

    allowed: bool
    refusals: tuple[ClearRefusal, ...]
    still_present: tuple[BreakerTrigger, ...] = ()


class ClearHaltRefusedError(NonRetryableError):
    """:func:`clear_halt` was called although :func:`can_clear` refuses."""

    def __init__(self, decision: ClearDecision) -> None:
        super().__init__(
            "clear-halt refused: " + ", ".join(r.value for r in decision.refusals),
            code=CLEAR_HALT_REFUSED,
        )
        self.decision = decision


def can_clear(
    status: BreakerStatus,
    signals: BreakerSignals,
    params: BreakerParams,
    *,
    reconciliation_clean_recent: bool,
) -> ClearDecision:
    """Whether a manual ``clear-halt`` may release the latched triggers (sec. 31.4).

    Refused (every reason listed) when nothing is latched, when there is no recent clean
    reconciliation (sec. 31.4), while ``emergency_close`` is still ``true`` (sec. 31.3.7:
    it returns to ``false`` only by a manual action, before clearing) or while a latched
    trigger's condition is still present (it would trip again at once).
    """
    refusals: list[ClearRefusal] = []
    if not status.latched:
        refusals.append(ClearRefusal.NOTHING_TO_CLEAR)
    if not reconciliation_clean_recent:
        refusals.append(ClearRefusal.RECONCILIATION_NOT_CLEAN)
    if signals.emergency_close:
        refusals.append(ClearRefusal.EMERGENCY_CLOSE_STILL_SET)
    active = set(detect_triggers(signals, params))
    present = tuple(
        item.trigger
        for item in status.latched
        if item.trigger in active and item.trigger is not _T.EMERGENCY_CLOSE
    )
    if present:
        refusals.append(ClearRefusal.CONDITION_STILL_PRESENT)
    return ClearDecision(allowed=not refusals, refusals=tuple(refusals), still_present=present)


def clear_halt(
    status: BreakerStatus,
    signals: BreakerSignals,
    params: BreakerParams,
    *,
    reconciliation_clean_recent: bool,
) -> BreakerDecision:
    """Manual reset (``python -m app.control clear-halt``, sec. 31.4): release every latch.

    The resulting state is ``NORMAL``, or ``WARNING`` if a warning condition holds.

    Raises:
        ClearHaltRefusedError: :func:`can_clear` refuses.
    """
    decision = can_clear(
        status, signals, params, reconciliation_clean_recent=reconciliation_clean_recent
    )
    if not decision.allowed:
        raise ClearHaltRefusedError(decision)
    released = _ordered({item.trigger for item in status.latched})
    return _decision(
        status,
        [],
        detect_triggers(signals, params),
        tripped=(),
        released=released,
        at=signals.observed_at_utc,
        extra_causes=("MANUAL_CLEAR",),
    )
