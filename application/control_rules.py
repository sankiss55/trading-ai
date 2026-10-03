"""Runtime control rules (sec. 7.4.3, 31, 46.4, AC-20). Pure: no I/O, no clock.

The control CLI (``app/control.py``) and the runtime ask these functions before acting;
the functions only decide, they never write.

* :func:`can_enable_trading` refuses, listing EVERY reason at once, while:
  any ``OWNER_DECISION`` is ``null`` (AC-20; the refusal carries the exact dotted paths),
  the environment is ``live`` (sec. 7.2, 48), the STOP file exists (sec. 31.2),
  ``emergency_close`` is set (sec. 31.3), the circuit breaker is not ``NORMAL`` or a
  state mismatch is unresolved (sec. 31.4), or the health of the broker, broker clock
  and database is unknown or not ``OK`` (sec. 42).
* :func:`effective_trading_enabled`: the STOP file and ``emergency_close`` override the
  ``trading_enabled`` flag (sec. 31.2, 31.3).
* :func:`can_set_ai_mode`: ``ACTIVE`` requires an approved shadow evaluation recorded in
  ``system_events`` (sec. 31.4, 46.4). ``DISABLED`` and ``SHADOW`` are always allowed.
* Emergency close is a flag only here (:func:`emergency_close_requested`); the
  procedure of sec. 31.3 (cancel entries, close positions, verify) is Phase 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from application.health import (
    CHECK_OWNER_DECISIONS,
    CHECK_STOP_FILE,
    HealthReport,
)
from domain.guards.checks import ControlFacts, kill_switch_reasons
from domain.models import AIMode, CircuitBreakerState, SystemControl
from domain.models.base import DomainModel

__all__ = [
    "LIVE_ENVIRONMENT",
    "ControlDecision",
    "Refusal",
    "RefusalCode",
    "can_enable_trading",
    "can_set_ai_mode",
    "effective_trading_enabled",
    "emergency_close_requested",
    "trading_block_reasons",
]

LIVE_ENVIRONMENT: Final = "live"
_HEALTH_CHECKS_REPORTED_ELSEWHERE: Final = frozenset({CHECK_OWNER_DECISIONS, CHECK_STOP_FILE})


class RefusalCode(StrEnum):
    """Stable machine-readable reasons of a refused control change."""

    PENDING_OWNER_DECISIONS = "PENDING_OWNER_DECISIONS"
    LIVE_ENVIRONMENT_BLOCKED = "LIVE_ENVIRONMENT_BLOCKED"
    STOP_FILE_PRESENT = "STOP_FILE_PRESENT"
    EMERGENCY_CLOSE_ACTIVE = "EMERGENCY_CLOSE_ACTIVE"
    CIRCUIT_BREAKER_NOT_NORMAL = "CIRCUIT_BREAKER_NOT_NORMAL"
    UNRESOLVED_STATE_MISMATCH = "UNRESOLVED_STATE_MISMATCH"
    HEALTH_UNKNOWN = "HEALTH_UNKNOWN"
    HEALTH_NOT_OK = "HEALTH_NOT_OK"
    SHADOW_APPROVAL_MISSING = "SHADOW_APPROVAL_MISSING"


class Refusal(DomainModel):
    """One reason why a control change is refused.

    Attributes:
        code: Stable reason code.
        message: Human-readable explanation (secret-free).
        details: Items the operator must act on (e.g. the missing config paths).
    """

    code: RefusalCode
    message: str
    details: tuple[str, ...] = ()


class ControlDecision(DomainModel):
    """Outcome of a control rule: allowed, or refused with every reason."""

    allowed: bool
    refusals: tuple[Refusal, ...] = ()

    @property
    def missing_parameters(self) -> tuple[str, ...]:
        """Dotted paths of the pending ``OWNER_DECISION`` parameters (AC-20)."""
        for refusal in self.refusals:
            if refusal.code is RefusalCode.PENDING_OWNER_DECISIONS:
                return refusal.details
        return ()

    @property
    def codes(self) -> tuple[RefusalCode, ...]:
        """Codes of every refusal, in evaluation order."""
        return tuple(refusal.code for refusal in self.refusals)


def _decide(refusals: Sequence[Refusal]) -> ControlDecision:
    return ControlDecision(allowed=not refusals, refusals=tuple(refusals))


def emergency_close_requested(control: SystemControl) -> bool:
    """Whether the emergency close flag is set (sec. 31.3; execution is Phase 5)."""
    return control.emergency_close


def trading_block_reasons(control: SystemControl, *, stop_file_present: bool) -> tuple[str, ...]:
    """Why new entries are blocked by the runtime controls (empty: not blocked by them).

    Delegates to :func:`domain.guards.checks.kill_switch_reasons`, the single
    implementation of the kill switch also evaluated by ``TRADING_ENABLED`` (sec. 19).
    """
    return kill_switch_reasons(
        ControlFacts(
            trading_enabled=control.trading_enabled,
            emergency_close=control.emergency_close,
            stop_file_present=stop_file_present,
            ai_mode=control.ai_mode,
        )
    )


def effective_trading_enabled(control: SystemControl, *, stop_file_present: bool) -> bool:
    """``trading_enabled`` after the STOP file and ``emergency_close`` overrides."""
    return not trading_block_reasons(control, stop_file_present=stop_file_present)


def _health_refusal(health: HealthReport | None) -> Refusal | None:
    if health is None:
        return Refusal(
            code=RefusalCode.HEALTH_UNKNOWN,
            message="no health check result: broker, broker clock and database status unknown",
        )
    failing = tuple(
        f"{check.name}: {check.status}: {check.detail}"
        for check in health.not_ok()
        if check.name not in _HEALTH_CHECKS_REPORTED_ELSEWHERE
    )
    if failing:
        return Refusal(
            code=RefusalCode.HEALTH_NOT_OK,
            message=f"health checks not OK (overall {health.status})",
            details=failing,
        )
    return None


def can_enable_trading(
    *,
    pending_decisions: Sequence[str],
    environment: str,
    control: SystemControl,
    stop_file_present: bool,
    health: HealthReport | None,
    circuit_breaker: CircuitBreakerState = CircuitBreakerState.NORMAL,
    unresolved_mismatch: bool = False,
) -> ControlDecision:
    """Whether ``trading_enabled`` may become ``true`` (sec. 7.4.3, 31.4, AC-20).

    Args:
        pending_decisions: Dotted paths of the ``null`` OWNER_DECISION parameters
            (``app.config.pending_owner_decisions``), in config order.
        environment: ``APP_ENV`` value (``dev | test | paper | live``).
        control: Current ``system_control`` row.
        stop_file_present: Whether ``control/STOP`` exists.
        health: Latest health report, or ``None`` if it could not be obtained.
        circuit_breaker: Circuit breaker state (Phase 4; ``NORMAL`` until it exists).
        unresolved_mismatch: Whether a reconciliation mismatch is unresolved (Phase 5).

    Returns:
        ``allowed=True`` only when no refusal applies; otherwise every refusal.
    """
    refusals: list[Refusal] = []
    if pending_decisions:
        refusals.append(
            Refusal(
                code=RefusalCode.PENDING_OWNER_DECISIONS,
                message=(
                    f"{len(pending_decisions)} OWNER_DECISION parameter(s) are null in "
                    "config.yaml; only the owner may set them (sec. 7.4.3)"
                ),
                details=tuple(pending_decisions),
            )
        )
    if environment.strip().lower() == LIVE_ENVIRONMENT:
        refusals.append(
            Refusal(
                code=RefusalCode.LIVE_ENVIRONMENT_BLOCKED,
                message="APP_ENV=live is blocked in the MVP (sec. 7.2, 48)",
            )
        )
    if stop_file_present:
        refusals.append(
            Refusal(
                code=RefusalCode.STOP_FILE_PRESENT,
                message="the STOP file exists: remove it first (sec. 31.2)",
            )
        )
    if control.emergency_close:
        refusals.append(
            Refusal(
                code=RefusalCode.EMERGENCY_CLOSE_ACTIVE,
                message="emergency_close is set: clear it manually first (sec. 31.3)",
            )
        )
    if circuit_breaker is not CircuitBreakerState.NORMAL:
        refusals.append(
            Refusal(
                code=RefusalCode.CIRCUIT_BREAKER_NOT_NORMAL,
                message=f"circuit breaker is {circuit_breaker}, not NORMAL (sec. 31.4)",
            )
        )
    if unresolved_mismatch:
        refusals.append(
            Refusal(
                code=RefusalCode.UNRESOLVED_STATE_MISMATCH,
                message="a reconciliation mismatch is unresolved (sec. 24.3, 31.4)",
            )
        )
    health_refusal = _health_refusal(health)
    if health_refusal is not None:
        refusals.append(health_refusal)
    return _decide(refusals)


def can_set_ai_mode(target: AIMode, *, shadow_approval_recorded: bool) -> ControlDecision:
    """Whether ``ai_mode`` may be set to ``target`` (sec. 31.4, 46.4).

    Args:
        target: Requested mode.
        shadow_approval_recorded: Whether an approved shadow evaluation report
            (``shadow/report.py``) is recorded in ``system_events``.
    """
    if target is AIMode.ACTIVE and not shadow_approval_recorded:
        return _decide(
            [
                Refusal(
                    code=RefusalCode.SHADOW_APPROVAL_MISSING,
                    message=(
                        "ai_mode ACTIVE requires an approved shadow-mode evaluation "
                        "(sec. 46.4) recorded in system_events; none exists"
                    ),
                )
            ]
        )
    return _decide([])
