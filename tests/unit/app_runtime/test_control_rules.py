"""Control rules (sec. 7.4.3, 31, 46.4; AC-20)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import load_config, pending_owner_decisions
from application.control_rules import (
    RefusalCode,
    can_enable_trading,
    can_set_ai_mode,
    effective_trading_enabled,
    emergency_close_requested,
    trading_block_reasons,
)
from application.health import (
    CHECK_BROKER,
    CHECK_CLOCK_SKEW,
    CHECK_OWNER_DECISIONS,
    CHECK_STOP_FILE,
    HealthCheck,
    HealthReport,
    HealthStatus,
)
from domain.models import AIMode, CircuitBreakerState, SystemControl
from tests.unit.app_runtime.fakes import at

ROOT = Path(__file__).resolve().parents[3]
PENDING_CONFIG = ROOT / "tests" / "fixtures" / "config.pending.yaml"


def _health(*checks: tuple[str, HealthStatus]) -> HealthReport:
    return HealthReport(
        checked_at_utc=at(14),
        checks=tuple(
            HealthCheck(name=name, status=status, detail=f"{name} {status}")
            for name, status in checks
        ),
    )


HEALTHY = _health(
    (CHECK_BROKER, HealthStatus.OK),
    (CHECK_CLOCK_SKEW, HealthStatus.OK),
    ("database", HealthStatus.OK),
    (CHECK_OWNER_DECISIONS, HealthStatus.OK),
    (CHECK_STOP_FILE, HealthStatus.OK),
)


def test_complete_config_healthy_paper_is_allowed() -> None:
    decision = can_enable_trading(
        pending_decisions=(),
        environment="paper",
        control=SystemControl(),
        stop_file_present=False,
        health=HEALTHY,
    )
    assert decision.allowed
    assert decision.refusals == ()
    assert decision.missing_parameters == ()


def test_pending_owner_decisions_refuse_with_the_exact_missing_list() -> None:
    config = load_config(PENDING_CONFIG).config
    pending = pending_owner_decisions(config)
    assert pending  # the fixture keeps every OWNER_DECISION null
    decision = can_enable_trading(
        pending_decisions=pending,
        environment="paper",
        control=SystemControl(),
        stop_file_present=False,
        health=HEALTHY,
    )
    assert not decision.allowed
    assert decision.codes == (RefusalCode.PENDING_OWNER_DECISIONS,)
    assert decision.missing_parameters == pending
    assert "universe.whitelist" in decision.missing_parameters


@pytest.mark.parametrize("environment", ["live", "LIVE", " live "])
def test_live_environment_is_refused(environment: str) -> None:
    decision = can_enable_trading(
        pending_decisions=(),
        environment=environment,
        control=SystemControl(),
        stop_file_present=False,
        health=HEALTHY,
    )
    assert decision.codes == (RefusalCode.LIVE_ENVIRONMENT_BLOCKED,)


def test_stop_file_and_emergency_close_refuse() -> None:
    decision = can_enable_trading(
        pending_decisions=(),
        environment="paper",
        control=SystemControl(emergency_close=True),
        stop_file_present=True,
        health=HEALTHY,
    )
    assert decision.codes == (
        RefusalCode.STOP_FILE_PRESENT,
        RefusalCode.EMERGENCY_CLOSE_ACTIVE,
    )


def test_every_reason_is_reported_at_once() -> None:
    decision = can_enable_trading(
        pending_decisions=("market_data.feed",),
        environment="live",
        control=SystemControl(emergency_close=True),
        stop_file_present=True,
        health=None,
        circuit_breaker=CircuitBreakerState.HALTED,
        unresolved_mismatch=True,
    )
    assert decision.codes == (
        RefusalCode.PENDING_OWNER_DECISIONS,
        RefusalCode.LIVE_ENVIRONMENT_BLOCKED,
        RefusalCode.STOP_FILE_PRESENT,
        RefusalCode.EMERGENCY_CLOSE_ACTIVE,
        RefusalCode.CIRCUIT_BREAKER_NOT_NORMAL,
        RefusalCode.UNRESOLVED_STATE_MISMATCH,
        RefusalCode.HEALTH_UNKNOWN,
    )
    assert decision.missing_parameters == ("market_data.feed",)


def test_unknown_health_is_refused() -> None:
    decision = can_enable_trading(
        pending_decisions=(),
        environment="paper",
        control=SystemControl(),
        stop_file_present=False,
        health=None,
    )
    assert decision.codes == (RefusalCode.HEALTH_UNKNOWN,)


@pytest.mark.parametrize("status", [HealthStatus.DEGRADED, HealthStatus.FAILED])
def test_broker_or_clock_health_not_ok_is_refused(status: HealthStatus) -> None:
    health = _health((CHECK_BROKER, HealthStatus.OK), (CHECK_CLOCK_SKEW, status))
    decision = can_enable_trading(
        pending_decisions=(),
        environment="paper",
        control=SystemControl(),
        stop_file_present=False,
        health=health,
    )
    assert decision.codes == (RefusalCode.HEALTH_NOT_OK,)
    assert decision.refusals[0].details == (f"{CHECK_CLOCK_SKEW}: {status}: clock_skew {status}",)


def test_stop_and_pending_checks_are_not_reported_twice_as_health() -> None:
    health = _health(
        (CHECK_BROKER, HealthStatus.OK),
        (CHECK_OWNER_DECISIONS, HealthStatus.DEGRADED),
        (CHECK_STOP_FILE, HealthStatus.DEGRADED),
    )
    decision = can_enable_trading(
        pending_decisions=("risk.max_positions",),
        environment="paper",
        control=SystemControl(),
        stop_file_present=True,
        health=health,
    )
    assert decision.codes == (RefusalCode.PENDING_OWNER_DECISIONS, RefusalCode.STOP_FILE_PRESENT)


def test_stop_file_overrides_trading_enabled() -> None:
    enabled = SystemControl(trading_enabled=True)
    assert effective_trading_enabled(enabled, stop_file_present=False)
    assert not effective_trading_enabled(enabled, stop_file_present=True)
    assert trading_block_reasons(enabled, stop_file_present=True) == ("STOP_FILE_PRESENT",)


def test_emergency_close_is_a_flag_that_blocks_trading() -> None:
    control = SystemControl(trading_enabled=True, emergency_close=True)
    assert emergency_close_requested(control)
    assert not effective_trading_enabled(control, stop_file_present=False)
    assert not emergency_close_requested(SystemControl())


def test_default_control_is_fail_closed() -> None:
    assert trading_block_reasons(SystemControl(), stop_file_present=False) == ("TRADING_DISABLED",)


def test_ai_mode_active_requires_shadow_approval() -> None:
    refused = can_set_ai_mode(AIMode.ACTIVE, shadow_approval_recorded=False)
    assert refused.codes == (RefusalCode.SHADOW_APPROVAL_MISSING,)
    assert can_set_ai_mode(AIMode.ACTIVE, shadow_approval_recorded=True).allowed


@pytest.mark.parametrize("mode", [AIMode.DISABLED, AIMode.SHADOW])
def test_ai_mode_disabled_and_shadow_are_allowed(mode: AIMode) -> None:
    assert can_set_ai_mode(mode, shadow_approval_recorded=False).allowed
