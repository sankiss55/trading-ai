"""Health checks (sec. 42) with fakes: broker, clock skew, DB, pending decisions, STOP."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from adapters.simulation.sim_clock import FixedClock
from application.health import (
    CHECK_BROKER,
    CHECK_CLOCK_SKEW,
    CHECK_DATABASE,
    CHECK_OWNER_DECISIONS,
    CHECK_STOP_FILE,
    HealthStatus,
    check_broker,
    check_clock_skew,
    check_database,
    run_health_checks,
)
from domain.errors import NonRetryableError
from domain.models import AccountState, SystemControl
from tests.unit.app_runtime.fakes import FakeBroker, FakeCalendar, FakeUnitOfWork, at, unreachable

MAX_SKEW = Decimal(2)


async def test_all_ok() -> None:
    clock = FixedClock(at(14))
    report = await run_health_checks(
        broker=FakeBroker(),
        calendar=FakeCalendar(clock),
        clock=clock,
        uow=FakeUnitOfWork(),
        max_clock_skew_seconds=MAX_SKEW,
        pending_decisions=(),
        stop_file_present=False,
    )
    assert report.status is HealthStatus.OK
    assert [check.name for check in report.checks] == [
        CHECK_BROKER,
        CHECK_CLOCK_SKEW,
        CHECK_DATABASE,
        CHECK_OWNER_DECISIONS,
        CHECK_STOP_FILE,
    ]
    assert report.checked_at_utc == at(14)
    assert report.not_ok() == ()


@pytest.mark.parametrize(
    ("skew", "expected"),
    [
        (timedelta(seconds=2), HealthStatus.OK),
        (timedelta(seconds=-2), HealthStatus.OK),
        (timedelta(seconds=2, milliseconds=1), HealthStatus.DEGRADED),
        (timedelta(seconds=-5), HealthStatus.DEGRADED),
    ],
)
async def test_clock_skew_above_limit_is_degraded(skew: timedelta, expected: HealthStatus) -> None:
    clock = FixedClock(at(14))
    check = await check_clock_skew(FakeCalendar(clock, skew=skew), clock, MAX_SKEW)
    assert check.status is expected
    assert Decimal(check.data["skew_seconds"]) == Decimal(str(skew.total_seconds()))


async def test_clock_skew_degrades_the_overall_report() -> None:
    clock = FixedClock(at(14))
    report = await run_health_checks(
        broker=FakeBroker(),
        calendar=FakeCalendar(clock, skew=timedelta(seconds=10)),
        clock=clock,
        uow=FakeUnitOfWork(),
        max_clock_skew_seconds=MAX_SKEW,
        pending_decisions=(),
        stop_file_present=False,
    )
    assert report.status is HealthStatus.DEGRADED
    assert [check.name for check in report.not_ok()] == [CHECK_CLOCK_SKEW]


async def test_broker_clock_unavailable_fails() -> None:
    clock = FixedClock(at(14))
    check = await check_clock_skew(FakeCalendar(clock, error=unreachable()), clock, MAX_SKEW)
    assert check.status is HealthStatus.FAILED
    assert "NETWORK" in check.detail


async def test_unreachable_broker_fails_without_raising() -> None:
    check = await check_broker(FakeBroker(error=unreachable()))
    assert check.status is HealthStatus.FAILED
    assert check.detail == "broker unreachable: RetryableError: [NETWORK] connection timed out"


async def test_non_domain_error_detail_carries_only_the_class_name() -> None:
    check = await check_broker(FakeBroker(error=RuntimeError("secret-looking text")))
    assert check.status is HealthStatus.FAILED
    assert check.detail == "broker unreachable: RuntimeError"


async def test_inactive_account_is_degraded() -> None:
    account = AccountState(
        equity=Decimal(1), last_equity=Decimal(1), buying_power=Decimal(1), status="SUSPENDED"
    )
    check = await check_broker(FakeBroker(account=account))
    assert check.status is HealthStatus.DEGRADED
    assert check.data == {"account_status": "SUSPENDED"}


async def test_database_not_writable_fails() -> None:
    uow = FakeUnitOfWork()
    uow.fail_writes = True
    check = await check_database(uow)
    assert check.status is HealthStatus.FAILED


async def test_database_probe_persists_nothing() -> None:
    uow = FakeUnitOfWork()
    async with uow:
        await uow.control.set(SystemControl(trading_enabled=True, updated_by="cli:owner"))
        await uow.commit()
    commits = uow.commits
    check = await check_database(uow)
    assert check.status is HealthStatus.OK
    assert uow.commits == commits
    assert uow.committed.control == SystemControl(trading_enabled=True, updated_by="cli:owner")
    assert uow.events() == []


async def test_pending_decisions_and_stop_file_degrade() -> None:
    clock = FixedClock(at(14))
    report = await run_health_checks(
        broker=FakeBroker(),
        calendar=FakeCalendar(clock),
        clock=clock,
        uow=FakeUnitOfWork(),
        max_clock_skew_seconds=MAX_SKEW,
        pending_decisions=("universe.whitelist", "market_data.feed"),
        stop_file_present=True,
    )
    assert report.status is HealthStatus.DEGRADED
    pending = report.get(CHECK_OWNER_DECISIONS)
    assert pending is not None
    assert pending.data == {"pending": ["universe.whitelist", "market_data.feed"]}
    stop = report.get(CHECK_STOP_FILE)
    assert stop is not None
    assert stop.status is HealthStatus.DEGRADED


async def test_failed_dominates_degraded_and_signature_tracks_changes() -> None:
    clock = FixedClock(at(14))
    healthy = await run_health_checks(
        broker=FakeBroker(),
        calendar=FakeCalendar(clock),
        clock=clock,
        uow=FakeUnitOfWork(),
        max_clock_skew_seconds=MAX_SKEW,
        pending_decisions=(),
        stop_file_present=True,
    )
    failing = await run_health_checks(
        broker=FakeBroker(error=NonRetryableError("unauthorized", code="AUTH")),
        calendar=FakeCalendar(clock),
        clock=clock,
        uow=FakeUnitOfWork(),
        max_clock_skew_seconds=MAX_SKEW,
        pending_decisions=(),
        stop_file_present=True,
    )
    assert healthy.status is HealthStatus.DEGRADED
    assert failing.status is HealthStatus.FAILED
    assert healthy.signature() != failing.signature()
    assert healthy.model_dump(mode="json")["checks"][0]["status"] == "OK"
