"""System modes and their transitions (sec. 32, 54.1).

Allowed transitions, exactly as in sec. 32::

    BOOTING -> SYNCING -> READY -> RUNNING
    RUNNING -> DEGRADED -> RUNNING        (automatic recovery of a non-critical anomaly)
    RUNNING | DEGRADED -> HALTED          (circuit breaker trigger)
    any -> EMERGENCY                      (emergency_close)
    HALTED -> READY                       (manual clear-halt after a clean reconciliation)
    any -> SHUTTING_DOWN

``SHUTTING_DOWN`` is terminal. A self-transition is never a transition. Anything else
raises :class:`LifecycleError` (code ``INVALID_MODE_TRANSITION``).

Every transition is recorded as a ``MODE_TRANSITION`` row of ``system_events`` (sec. 32,
"each transition is recorded") and logged. If the event cannot be written, a transition
towards safety (``HALTED``, ``EMERGENCY``, ``SHUTTING_DOWN``) still happens and the failure
is logged; any other transition is refused by re-raising the error (fail closed: the
system never becomes more permissive without an audit trail).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from app.logging_setup import fields
from domain.errors import NonRetryableError
from domain.models import SystemEvent, SystemMode
from domain.ports import IClock, IUnitOfWork

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ANY_SOURCE_TARGETS",
    "EVENT_MODE_TRANSITION",
    "INVALID_MODE_TRANSITION",
    "SAFETY_TARGETS",
    "LifecycleError",
    "SystemLifecycle",
    "Transition",
    "UnitOfWorkFactory",
    "is_allowed_transition",
    "record_system_event",
]

UnitOfWorkFactory = Callable[[], IUnitOfWork]
"""Returns the unit of work for one transaction (a fresh or a shared instance)."""

EVENT_MODE_TRANSITION: Final = "MODE_TRANSITION"
INVALID_MODE_TRANSITION: Final = "INVALID_MODE_TRANSITION"

_M = SystemMode
ALLOWED_TRANSITIONS: Final[Mapping[SystemMode, frozenset[SystemMode]]] = MappingProxyType(
    {
        _M.BOOTING: frozenset({_M.SYNCING}),
        _M.SYNCING: frozenset({_M.READY}),
        _M.READY: frozenset({_M.RUNNING}),
        _M.RUNNING: frozenset({_M.DEGRADED, _M.HALTED}),
        _M.DEGRADED: frozenset({_M.RUNNING, _M.HALTED}),
        _M.HALTED: frozenset({_M.READY}),
        _M.EMERGENCY: frozenset(),
        _M.SHUTTING_DOWN: frozenset(),
    }
)
"""Source-specific transitions of sec. 32 (see also :data:`ANY_SOURCE_TARGETS`)."""

ANY_SOURCE_TARGETS: Final = frozenset({_M.EMERGENCY, _M.SHUTTING_DOWN})
"""Reachable from every mode except ``SHUTTING_DOWN`` (terminal)."""

SAFETY_TARGETS: Final = frozenset({_M.HALTED, _M.EMERGENCY, _M.SHUTTING_DOWN})
"""Transitions applied even when their audit event cannot be written."""

_LOGGER = logging.getLogger(__name__)


class LifecycleError(NonRetryableError):
    """A transition not allowed by sec. 32."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=INVALID_MODE_TRANSITION)


def is_allowed_transition(source: SystemMode, target: SystemMode) -> bool:
    """Whether ``source -> target`` is a transition of sec. 32."""
    if source is target or source is SystemMode.SHUTTING_DOWN:
        return False
    return target in ANY_SOURCE_TARGETS or target in ALLOWED_TRANSITIONS[source]


async def record_system_event(uow_factory: UnitOfWorkFactory, event: SystemEvent) -> None:
    """Append ``event`` to ``system_events`` in its own committed transaction."""
    async with uow_factory() as uow:
        await uow.system_events.append(event)
        await uow.commit()


@dataclass(frozen=True, slots=True)
class Transition:
    """One applied transition."""

    source: SystemMode
    target: SystemMode
    reason: str
    recorded: bool


class SystemLifecycle:
    """Current system mode; applies and records transitions.

    Args:
        uow_factory: Unit of work used to record each transition.
        clock: Time source of the event timestamps.
        initial: Starting mode (``BOOTING``).
    """

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        clock: IClock,
        initial: SystemMode = SystemMode.BOOTING,
    ) -> None:
        self._uow_factory = uow_factory
        self._clock = clock
        self._mode = initial
        self._history: list[Transition] = []

    @property
    def mode(self) -> SystemMode:
        """Current mode."""
        return self._mode

    @property
    def history(self) -> tuple[Transition, ...]:
        """Transitions applied so far, oldest first."""
        return tuple(self._history)

    async def transition(
        self,
        target: SystemMode,
        *,
        reason: str,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        """Move to ``target`` and record a ``MODE_TRANSITION`` system event.

        Raises:
            LifecycleError: the transition is not allowed by sec. 32.
            Exception: the event could not be written and ``target`` is not a safety
                mode (the mode is left unchanged).
        """
        source = self._mode
        if not is_allowed_transition(source, target):
            raise LifecycleError(f"transition {source} -> {target} is not allowed (sec. 32)")
        event = SystemEvent(
            occurred_at_utc=self._clock.now_utc(),
            event_type=EVENT_MODE_TRANSITION,
            detail={"from": str(source), "to": str(target), "reason": reason, **(detail or {})},
        )
        recorded = True
        try:
            await record_system_event(self._uow_factory, event)
        except Exception as exc:
            if target not in SAFETY_TARGETS:
                _LOGGER.error(
                    "mode transition refused: its system event could not be written",
                    extra=fields(
                        event="MODE_TRANSITION_FAILED",
                        from_mode=str(source),
                        to_mode=str(target),
                        error_type=type(exc).__name__,
                    ),
                )
                raise
            recorded = False
            _LOGGER.error(
                "mode transition applied without its system event (write failed)",
                extra=fields(
                    event="MODE_TRANSITION_UNRECORDED",
                    from_mode=str(source),
                    to_mode=str(target),
                    error_type=type(exc).__name__,
                ),
            )
        self._mode = target
        self._history.append(Transition(source, target, reason, recorded))
        _LOGGER.info(
            "system mode changed",
            extra=fields(
                event=EVENT_MODE_TRANSITION,
                from_mode=str(source),
                to_mode=str(target),
                reason=reason,
            ),
        )
