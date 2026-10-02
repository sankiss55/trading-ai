"""Scripted AI veto filter for tests and deterministic simulation (sec. 8.6, 49.5).

``ScriptedAIFilter`` implements ``IAIFilter`` without any model call. It returns
pre-recorded ``AIVerdictResult`` objects in order, or computes them with a callable,
and NEVER raises (sec. 8.5): an exhausted script or a failing callable is reported as
``UNAVAILABLE``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from domain.models import (
    AIReasonCode,
    AIRiskFlag,
    AIValidity,
    AIVerdict,
    AIVerdictKind,
    AIVerdictResult,
    Snapshot,
)

__all__ = [
    "SCRIPT_EXHAUSTED",
    "SCRIPT_FAILED",
    "ScriptedAIFilter",
    "always_approve",
    "approve_result",
    "unavailable_result",
    "veto_result",
]

SCRIPT_EXHAUSTED = "SCRIPT_EXHAUSTED"
"""``invalid_reason`` returned once a result sequence has been consumed."""
SCRIPT_FAILED = "SCRIPT_FAILED"
"""``invalid_reason`` prefix returned when the scripted callable raises."""

ScriptFn = Callable[[Snapshot], AIVerdictResult]


def approve_result(
    signal_id: str,
    *,
    confidence: float = 0.8,
    rationale: str = "scripted approval",
    risk_flags: Sequence[AIRiskFlag] = (),
) -> AIVerdictResult:
    """VALID ``APPROVE`` result for ``signal_id``."""
    return AIVerdictResult(
        validity=AIValidity.VALID,
        verdict=AIVerdict(
            signal_id=signal_id,
            verdict=AIVerdictKind.APPROVE,
            reason_code=AIReasonCode.SIGNAL_CONFIRMED,
            risk_flags=tuple(risk_flags),
            confidence=confidence,
            rationale=rationale,
        ),
        latency_ms=0,
        stop_reason="end_turn",
        model_id="scripted",
    )


def veto_result(
    signal_id: str,
    *,
    reason_code: AIReasonCode = AIReasonCode.CONTEXT_ADVERSE,
    confidence: float = 0.8,
    rationale: str = "scripted veto",
    risk_flags: Sequence[AIRiskFlag] = (),
) -> AIVerdictResult:
    """VALID ``VETO`` result for ``signal_id``."""
    return AIVerdictResult(
        validity=AIValidity.VALID,
        verdict=AIVerdict(
            signal_id=signal_id,
            verdict=AIVerdictKind.VETO,
            reason_code=reason_code,
            risk_flags=tuple(risk_flags),
            confidence=confidence,
            rationale=rationale,
        ),
        latency_ms=0,
        stop_reason="end_turn",
        model_id="scripted",
    )


def unavailable_result(reason: str) -> AIVerdictResult:
    """``UNAVAILABLE`` result with ``invalid_reason=reason``."""
    return AIVerdictResult(
        validity=AIValidity.UNAVAILABLE, invalid_reason=reason, model_id="scripted"
    )


def always_approve(snapshot: Snapshot) -> AIVerdictResult:
    """Script function approving every snapshot with its own ``signal_id``."""
    return approve_result(snapshot.payload.signal_id)


class ScriptedAIFilter:
    """``IAIFilter`` returning scripted results; never raises.

    Args:
        script: Either a sequence of results returned one per call, in order, or a
            callable computing the result from the snapshot.
        exhausted_result: Result returned after a sequence is consumed. Defaults to
            ``UNAVAILABLE`` with reason ``SCRIPT_EXHAUSTED``.
    """

    def __init__(
        self,
        script: Sequence[AIVerdictResult] | ScriptFn,
        *,
        exhausted_result: AIVerdictResult | None = None,
    ) -> None:
        self._fn: ScriptFn | None = script if callable(script) else None
        self._results: tuple[AIVerdictResult, ...] = () if callable(script) else tuple(script)
        self._index = 0
        self._exhausted = exhausted_result or unavailable_result(SCRIPT_EXHAUSTED)
        self._calls: list[Snapshot] = []

    @property
    def calls(self) -> tuple[Snapshot, ...]:
        """Snapshots received so far, in call order."""
        return tuple(self._calls)

    async def evaluate(self, snapshot: Snapshot) -> AIVerdictResult:
        """Return the next scripted result for ``snapshot``."""
        self._calls.append(snapshot)
        if self._fn is not None:
            try:
                return self._fn(snapshot)
            except Exception as exc:  # noqa: BLE001 - the port must never raise (sec. 8.5)
                return unavailable_result(f"{SCRIPT_FAILED}: {type(exc).__name__}")
        if self._index >= len(self._results):
            return self._exhausted
        result = self._results[self._index]
        self._index += 1
        return result
