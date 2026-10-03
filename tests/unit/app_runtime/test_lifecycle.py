"""System mode transitions (sec. 32, 54.1)."""

from __future__ import annotations

import itertools

import pytest

from adapters.simulation.sim_clock import FixedClock
from app.lifecycle import (
    EVENT_MODE_TRANSITION,
    INVALID_MODE_TRANSITION,
    LifecycleError,
    SystemLifecycle,
    is_allowed_transition,
)
from domain.models import SystemMode
from tests.unit.app_runtime.fakes import DatabaseDownError, FakeUnitOfWork, at

M = SystemMode

SPEC_TRANSITIONS = {
    (M.BOOTING, M.SYNCING),
    (M.SYNCING, M.READY),
    (M.READY, M.RUNNING),
    (M.RUNNING, M.DEGRADED),
    (M.DEGRADED, M.RUNNING),
    (M.RUNNING, M.HALTED),
    (M.DEGRADED, M.HALTED),
    (M.HALTED, M.READY),
} | {
    (source, target)
    for source in M
    for target in (M.EMERGENCY, M.SHUTTING_DOWN)
    if source is not target and source is not M.SHUTTING_DOWN
}


@pytest.mark.parametrize(("source", "target"), list(itertools.product(M, M)))
def test_allowed_transitions_are_exactly_those_of_sec_32(
    source: SystemMode, target: SystemMode
) -> None:
    assert is_allowed_transition(source, target) == ((source, target) in SPEC_TRANSITIONS)


def test_shutting_down_is_terminal() -> None:
    assert not any(is_allowed_transition(M.SHUTTING_DOWN, target) for target in M)


def _lifecycle(uow: FakeUnitOfWork, initial: SystemMode = M.BOOTING) -> SystemLifecycle:
    return SystemLifecycle(uow_factory=uow.factory, clock=FixedClock(at(14)), initial=initial)


async def test_each_transition_is_recorded_as_a_system_event() -> None:
    uow = FakeUnitOfWork()
    lifecycle = _lifecycle(uow)
    await lifecycle.transition(M.SYNCING, reason="loaded")
    await lifecycle.transition(M.READY, reason="synced", detail={"observation_only": True})
    assert lifecycle.mode is M.READY
    events = uow.events(EVENT_MODE_TRANSITION)
    assert [(e.detail["from"], e.detail["to"]) for e in events] == [
        ("BOOTING", "SYNCING"),
        ("SYNCING", "READY"),
    ]
    assert events[1].detail["reason"] == "synced"
    assert events[1].detail["observation_only"] is True
    assert events[0].occurred_at_utc == at(14)
    assert [t.recorded for t in lifecycle.history] == [True, True]


async def test_forbidden_transition_raises_and_records_nothing() -> None:
    uow = FakeUnitOfWork()
    lifecycle = _lifecycle(uow)
    with pytest.raises(LifecycleError) as caught:
        await lifecycle.transition(M.RUNNING, reason="skip the sync")
    assert caught.value.code == INVALID_MODE_TRANSITION
    assert lifecycle.mode is M.BOOTING
    assert uow.events() == []


async def test_non_safety_transition_is_refused_when_its_event_cannot_be_written() -> None:
    uow = FakeUnitOfWork()
    lifecycle = _lifecycle(uow)
    uow.fail_writes = True
    with pytest.raises(DatabaseDownError):
        await lifecycle.transition(M.SYNCING, reason="loaded")
    assert lifecycle.mode is M.BOOTING


@pytest.mark.parametrize("target", [M.EMERGENCY, M.SHUTTING_DOWN])
async def test_safety_transition_applies_even_without_database(target: SystemMode) -> None:
    uow = FakeUnitOfWork()
    lifecycle = _lifecycle(uow, initial=M.READY)
    uow.fail_writes = True
    await lifecycle.transition(target, reason="db down")
    assert lifecycle.mode is target
    assert lifecycle.history[-1].recorded is False


async def test_halted_can_only_return_to_ready() -> None:
    uow = FakeUnitOfWork()
    lifecycle = _lifecycle(uow, initial=M.HALTED)
    with pytest.raises(LifecycleError):
        await lifecycle.transition(M.RUNNING, reason="skip clear-halt")
    await lifecycle.transition(M.READY, reason="clear-halt")
    assert lifecycle.mode is M.READY
