"""Circuit breaker (sec. 30): every trigger, latching, resets and manual clear (sec. 49.1)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from domain.guards.checks import CIRCUIT_BREAKER_NOT_NORMAL, failed_codes, run_pre_ai_checks
from domain.market.quality import StaleReport
from domain.models import CheckResult, CircuitBreakerState, ReconciliationOutcome
from domain.risk.circuit_breaker import (
    TRIGGER_RULES,
    ApiErrorLevel,
    BreakerParams,
    BreakerSignals,
    BreakerStatus,
    BreakerTrigger,
    ClearHaltRefusedError,
    ClearRefusal,
    LatchedTrigger,
    Procedure,
    ResetPolicy,
    blocks_new_entries,
    can_clear,
    clear_halt,
    detect_triggers,
    evaluate_breaker,
    requires_close_all,
)
from domain.risk.risk_engine import DAILY_LOSS_LIMIT, MAX_DRAWDOWN, PARAM_PENDING, WEEKLY_LOSS_LIMIT
from tests.unit.guards import factories as f

S = CircuitBreakerState
T = BreakerTrigger
DAY = date(2025, 3, 10)
AT = datetime(2025, 3, 10, 15, 0, tzinfo=UTC)
PARAMS = BreakerParams(max_order_rejections_per_day=3, max_clock_skew_seconds=Decimal(2))


def signals(**overrides: Any) -> BreakerSignals:
    """Observations with no trigger active."""
    values: dict[str, Any] = {
        "observed_at_utc": AT,
        "session_date": DAY,
        "loss_limit_results": (),
        "order_rejections_today": 0,
        "api_errors": ApiErrorLevel.NONE,
        "stale_report": None,
        "market_data_stream_connected": True,
        "db_writable": True,
        "reconciliation_outcome": ReconciliationOutcome.RECONCILED_OK,
        "unprotected_symbols": (),
        "broker_state_known": True,
        "clock_skew_seconds": Decimal("0.1"),
        "ai_unavailable_persistent": False,
        "emergency_close": False,
    }
    values.update(overrides)
    return BreakerSignals(**values)


def loss(code: str, *, passed: bool = False) -> CheckResult:
    return CheckResult(passed=passed, code=code, detail={"limit": code})


def stale(warning: bool) -> StaleReport:
    return StaleReport(
        stale_symbols=frozenset({"A", "B"}) if warning else frozenset(),
        whitelist_size=3,
        stale_count=2 if warning else 0,
        circuit_breaker_warning=warning,
    )


TRIGGER_SIGNALS: dict[BreakerTrigger, dict[str, Any]] = {
    T.DAILY_LOSS_LIMIT: {"loss_limit_results": (loss(DAILY_LOSS_LIMIT),)},
    T.WEEKLY_LOSS_LIMIT: {"loss_limit_results": (loss(WEEKLY_LOSS_LIMIT),)},
    T.MAX_DRAWDOWN: {"loss_limit_results": (loss(MAX_DRAWDOWN),)},
    T.ORDER_REJECTIONS: {"order_rejections_today": 3},
    T.API_ERRORS_REPEATED: {"api_errors": ApiErrorLevel.REPEATED},
    T.API_ERRORS_PERSISTENT: {"api_errors": ApiErrorLevel.PERSISTENT},
    T.MARKET_DATA_STALE_MAJORITY: {"stale_report": stale(True)},
    T.MARKET_DATA_STREAM_DOWN: {"market_data_stream_connected": False},
    T.DB_NOT_WRITABLE: {"db_writable": False},
    T.STATE_MISMATCH: {"reconciliation_outcome": ReconciliationOutcome.STATE_MISMATCH},
    T.ORPHANED_POSITION: {"reconciliation_outcome": ReconciliationOutcome.ORPHANED},
    T.UNPROTECTED_POSITION: {"unprotected_symbols": ("SPY",)},
    T.BROKER_STATE_UNKNOWN: {"broker_state_known": False},
    T.CLOCK_SKEW: {"clock_skew_seconds": Decimal("2.5")},
    T.AI_UNAVAILABLE_PERSISTENT: {"ai_unavailable_persistent": True},
    T.EMERGENCY_CLOSE: {"emergency_close": True},
}

EXPECTED: dict[BreakerTrigger, tuple[CircuitBreakerState, ResetPolicy]] = {
    T.DAILY_LOSS_LIMIT: (S.HALTED, ResetPolicy.NEXT_SESSION),
    T.WEEKLY_LOSS_LIMIT: (S.HALTED, ResetPolicy.MANUAL),
    T.MAX_DRAWDOWN: (S.HALTED, ResetPolicy.MANUAL),
    T.ORDER_REJECTIONS: (S.HALTED, ResetPolicy.NEXT_SESSION),
    T.API_ERRORS_REPEATED: (S.WARNING, ResetPolicy.AUTOMATIC),
    T.API_ERRORS_PERSISTENT: (S.HALTED, ResetPolicy.MANUAL),
    T.MARKET_DATA_STALE_MAJORITY: (S.WARNING, ResetPolicy.AUTOMATIC),
    T.MARKET_DATA_STREAM_DOWN: (S.HALTED, ResetPolicy.MANUAL),
    T.DB_NOT_WRITABLE: (S.HALTED, ResetPolicy.MANUAL),
    T.STATE_MISMATCH: (S.HALTED, ResetPolicy.MANUAL),
    T.ORPHANED_POSITION: (S.HALTED, ResetPolicy.MANUAL),
    T.UNPROTECTED_POSITION: (S.HALTED, ResetPolicy.MANUAL),
    T.BROKER_STATE_UNKNOWN: (S.HALTED, ResetPolicy.MANUAL),
    T.CLOCK_SKEW: (S.WARNING, ResetPolicy.AUTOMATIC),
    T.AI_UNAVAILABLE_PERSISTENT: (S.WARNING, ResetPolicy.AUTOMATIC),
    T.EMERGENCY_CLOSE: (S.EMERGENCY, ResetPolicy.MANUAL),
}
"""Sec. 30.2 table."""


def test_rules_match_the_spec_table() -> None:
    assert set(TRIGGER_RULES) == set(BreakerTrigger) == set(TRIGGER_SIGNALS)
    for trigger, (state, reset) in EXPECTED.items():
        assert (TRIGGER_RULES[trigger].state, TRIGGER_RULES[trigger].reset) == (state, reset)


def test_quiet_signals_keep_normal() -> None:
    decision = evaluate_breaker(BreakerStatus.normal(), signals(), PARAMS)
    assert decision.status == BreakerStatus.normal()
    assert decision.transition is None
    assert decision.active == ()


@pytest.mark.parametrize("trigger", list(BreakerTrigger))
def test_each_trigger_moves_normal_to_its_state(trigger: BreakerTrigger) -> None:
    decision = evaluate_breaker(BreakerStatus.normal(), signals(**TRIGGER_SIGNALS[trigger]), PARAMS)
    state, reset = EXPECTED[trigger]
    assert detect_triggers(signals(**TRIGGER_SIGNALS[trigger]), PARAMS) == (trigger,)
    assert decision.status.state is state
    assert decision.transition is not None
    assert decision.transition.from_state is S.NORMAL
    assert decision.transition.to_state is state
    assert decision.transition.causes == (trigger.value,)
    assert decision.transition.occurred_at_utc == AT
    if reset is ResetPolicy.AUTOMATIC:
        assert decision.status.warnings == (trigger,)
        assert decision.status.latched == ()
        assert decision.tripped == ()
    else:
        assert [item.trigger for item in decision.status.latched] == [trigger]
        assert decision.tripped == (trigger,)


def test_procedures_are_requested_while_their_condition_holds() -> None:
    unprotected = signals(**TRIGGER_SIGNALS[T.UNPROTECTED_POSITION])
    first = evaluate_breaker(BreakerStatus.normal(), unprotected, PARAMS)
    assert first.procedures == (Procedure.PROTECT_POSITION,)
    again = evaluate_breaker(first.status, unprotected, PARAMS)
    assert again.procedures == (Procedure.PROTECT_POSITION,)
    assert again.transition is None
    emergency = evaluate_breaker(first.status, signals(emergency_close=True), PARAMS)
    assert emergency.procedures == (Procedure.EMERGENCY_CLOSE,)


def test_emergency_dominates_and_requires_closing_everything() -> None:
    decision = evaluate_breaker(
        BreakerStatus.normal(),
        signals(emergency_close=True, db_writable=False, api_errors=ApiErrorLevel.REPEATED),
        PARAMS,
    )
    assert decision.status.state is S.EMERGENCY
    assert requires_close_all(decision.status.state)
    assert not requires_close_all(S.HALTED)


def test_passed_or_pending_loss_results_do_not_trip() -> None:
    results = (
        loss(DAILY_LOSS_LIMIT, passed=True),
        CheckResult(passed=False, code=PARAM_PENDING, detail={"limit": MAX_DRAWDOWN}),
    )
    assert detect_triggers(signals(loss_limit_results=results), PARAMS) == ()


@pytest.mark.parametrize(("rejections", "trips"), [(2, False), (3, True), (4, True)])
def test_order_rejections_trip_when_reaching_the_maximum(rejections: int, trips: bool) -> None:
    found = detect_triggers(signals(order_rejections_today=rejections), PARAMS)
    assert (T.ORDER_REJECTIONS in found) is trips


@pytest.mark.parametrize(("skew", "trips"), [("2", False), ("2.01", True), (None, False)])
def test_clock_skew_above_the_maximum_warns(skew: str | None, trips: bool) -> None:
    value = None if skew is None else Decimal(skew)
    found = detect_triggers(signals(clock_skew_seconds=value), PARAMS)
    assert (T.CLOCK_SKEW in found) is trips


def test_stale_minority_does_not_warn() -> None:
    assert detect_triggers(signals(stale_report=stale(False)), PARAMS) == ()


# --------------------------------------------------------------------------- resets


def test_warning_clears_automatically_with_its_condition() -> None:
    warned = evaluate_breaker(
        BreakerStatus.normal(), signals(stale_report=stale(True)), PARAMS
    ).status
    assert warned.state is S.WARNING
    decision = evaluate_breaker(warned, signals(), PARAMS)
    assert decision.status.state is S.NORMAL
    assert decision.transition is not None
    assert decision.transition.causes == (T.MARKET_DATA_STALE_MAJORITY.value,)


def test_next_session_trigger_holds_for_the_session_then_releases() -> None:
    halted = evaluate_breaker(
        BreakerStatus.normal(), signals(**TRIGGER_SIGNALS[T.DAILY_LOSS_LIMIT]), PARAMS
    ).status
    later_today = evaluate_breaker(halted, signals(observed_at_utc=AT + timedelta(hours=1)), PARAMS)
    assert later_today.status.state is S.HALTED  # condition gone, same session: still halted
    assert later_today.released == ()
    next_session = evaluate_breaker(
        later_today.status, signals(session_date=DAY + timedelta(days=1)), PARAMS
    )
    assert next_session.status.state is S.NORMAL
    assert next_session.released == (T.DAILY_LOSS_LIMIT,)
    assert next_session.transition is not None
    assert next_session.transition.to_state is S.NORMAL


def test_next_session_trigger_stays_while_its_condition_persists() -> None:
    rejected = signals(**TRIGGER_SIGNALS[T.ORDER_REJECTIONS])
    halted = evaluate_breaker(BreakerStatus.normal(), rejected, PARAMS).status
    tomorrow = evaluate_breaker(
        halted, rejected.model_copy(update={"session_date": DAY + timedelta(days=1)}), PARAMS
    )
    assert tomorrow.status.state is S.HALTED
    assert tomorrow.status.latched[0].session_date == DAY  # the original trip is kept


@pytest.mark.parametrize(
    "trigger", [t for t, (_, reset) in EXPECTED.items() if reset is ResetPolicy.MANUAL]
)
def test_manual_triggers_never_release_by_themselves(trigger: BreakerTrigger) -> None:
    status = evaluate_breaker(
        BreakerStatus.normal(), signals(**TRIGGER_SIGNALS[trigger]), PARAMS
    ).status
    for days in (0, 1, 30):
        decision = evaluate_breaker(
            status, signals(session_date=DAY + timedelta(days=days)), PARAMS
        )
        assert decision.status.state is EXPECTED[trigger][0]
        assert decision.released == ()
        status = decision.status


def test_clear_halt_releases_every_latch_after_a_clean_reconciliation() -> None:
    status = evaluate_breaker(
        BreakerStatus.normal(),
        signals(loss_limit_results=(loss(MAX_DRAWDOWN),), db_writable=False),
        PARAMS,
    ).status
    quiet = signals()
    assert can_clear(status, quiet, PARAMS, reconciliation_clean_recent=True).allowed
    decision = clear_halt(status, quiet, PARAMS, reconciliation_clean_recent=True)
    assert decision.status == BreakerStatus.normal()
    assert decision.released == (T.MAX_DRAWDOWN, T.DB_NOT_WRITABLE)
    assert decision.transition is not None
    assert decision.transition.causes[0] == "MANUAL_CLEAR"


def test_clear_halt_keeps_an_active_warning() -> None:
    status = evaluate_breaker(
        BreakerStatus.normal(), signals(db_writable=False, clock_skew_seconds=Decimal(5)), PARAMS
    ).status
    decision = clear_halt(
        status, signals(clock_skew_seconds=Decimal(5)), PARAMS, reconciliation_clean_recent=True
    )
    assert decision.status.state is S.WARNING
    assert decision.status.warnings == (T.CLOCK_SKEW,)


@pytest.mark.parametrize(
    ("status_signals", "now_signals", "clean", "refusals"),
    [
        ({}, {}, True, (ClearRefusal.NOTHING_TO_CLEAR,)),
        ({"db_writable": False}, {}, False, (ClearRefusal.RECONCILIATION_NOT_CLEAN,)),
        (
            {"emergency_close": True},
            {"emergency_close": True},
            True,
            (ClearRefusal.EMERGENCY_CLOSE_STILL_SET,),
        ),
        (
            {"loss_limit_results": (loss(WEEKLY_LOSS_LIMIT),)},
            {"loss_limit_results": (loss(WEEKLY_LOSS_LIMIT),)},
            True,
            (ClearRefusal.CONDITION_STILL_PRESENT,),
        ),
        (
            {"broker_state_known": False},
            {"broker_state_known": False},
            False,
            (ClearRefusal.RECONCILIATION_NOT_CLEAN, ClearRefusal.CONDITION_STILL_PRESENT),
        ),
    ],
)
def test_clear_is_refused_with_every_reason(
    status_signals: dict[str, Any],
    now_signals: dict[str, Any],
    clean: bool,
    refusals: tuple[ClearRefusal, ...],
) -> None:
    status = evaluate_breaker(BreakerStatus.normal(), signals(**status_signals), PARAMS).status
    decision = can_clear(status, signals(**now_signals), PARAMS, reconciliation_clean_recent=clean)
    assert not decision.allowed
    assert decision.refusals == refusals
    with pytest.raises(ClearHaltRefusedError) as info:
        clear_halt(status, signals(**now_signals), PARAMS, reconciliation_clean_recent=clean)
    assert info.value.code == "CLEAR_HALT_REFUSED"
    assert info.value.decision == decision


def test_emergency_clears_only_after_the_flag_is_reset_manually() -> None:
    status = evaluate_breaker(BreakerStatus.normal(), signals(emergency_close=True), PARAMS).status
    assert status.state is S.EMERGENCY
    decision = clear_halt(status, signals(), PARAMS, reconciliation_clean_recent=True)
    assert decision.status.state is S.NORMAL


# --------------------------------------------------------------------------- state model


def test_status_must_match_its_triggers() -> None:
    latched = LatchedTrigger(
        trigger=T.DB_NOT_WRITABLE,
        state=S.HALTED,
        reset=ResetPolicy.MANUAL,
        tripped_at_utc=AT,
        session_date=DAY,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        BreakerStatus(state=S.NORMAL, latched=(latched,), warnings=())
    with pytest.raises(ValueError, match="automatic-reset"):
        BreakerStatus(state=S.HALTED, latched=(), warnings=(T.DB_NOT_WRITABLE,))
    with pytest.raises(ValueError, match="only once"):
        BreakerStatus(state=S.HALTED, latched=(latched, latched), warnings=())


# --------------------------------------------------------------------------- invariant 51.13


@pytest.mark.parametrize("state", [S.WARNING, S.HALTED, S.EMERGENCY])
def test_no_new_entry_passes_the_library_unless_normal(state: CircuitBreakerState) -> None:
    assert blocks_new_entries(state)
    ctx = f.context(runtime=f.runtime(breaker_state=state))
    assert failed_codes(run_pre_ai_checks(ctx)) == (CIRCUIT_BREAKER_NOT_NORMAL,)


def test_normal_allows_new_entries() -> None:
    assert not blocks_new_entries(S.NORMAL)
    assert failed_codes(run_pre_ai_checks(f.context())) == ()


def test_a_tripped_breaker_blocks_the_library_end_to_end() -> None:
    """Breaker decision -> runtime facts -> check library: HALTED blocks the entry."""
    decision = evaluate_breaker(
        BreakerStatus.normal(), signals(loss_limit_results=(loss(DAILY_LOSS_LIMIT),)), PARAMS
    )
    ctx = f.context(runtime=f.runtime(breaker_state=decision.status.state))
    assert CIRCUIT_BREAKER_NOT_NORMAL in failed_codes(run_pre_ai_checks(ctx))
