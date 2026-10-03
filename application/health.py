"""Health monitor checks (sec. 42), aggregated into a typed :class:`HealthReport`.

**Phase 2 subset.** The checks available before the live streams exist:

========================  ======================================================  ==========
check                     OK when                                                 otherwise
========================  ======================================================  ==========
``broker``                ``IBroker.get_account`` answers and the account is      FAILED /
                          ``ACTIVE``                                              DEGRADED
``clock_skew``            ``|broker clock - IClock| <= max_clock_skew_seconds``   DEGRADED
                          (sec. 30.2: skew is WARNING/DEGRADED); broker clock     (FAILED if
                          measured at the midpoint of the request                 unreachable)
``database``              a write transaction succeeds (rolled back, nothing is   FAILED
                          persisted)
``owner_decisions``       no ``OWNER_DECISION`` is ``null`` (AC-20)               DEGRADED
``stop_file``             ``control/STOP`` is absent (sec. 31.2)                  DEGRADED
========================  ======================================================  ==========

``DEGRADED`` means "running, but trading must not be enabled/continue"; ``FAILED`` means
an integration is not usable. The overall status is the worst check. Market-data and
trade-update streams (Phase 3/5), AI (Phase 8), disk space and SMTP reachability are
added by later phases.

Everything goes through ports; the caller passes facts it read itself (pending owner
decisions from the validated config, STOP file presence from the file system), so this
module performs no I/O of its own and never reads the wall clock (sec. 8.3).
A check never raises: any error becomes a ``FAILED`` check whose detail carries the
error class and, for domain errors, their (secret-free) message (sec. 43.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final

from pydantic import Field

from domain.errors import DomainError
from domain.models.base import DomainModel, UtcDatetime
from domain.ports import IBroker, IClock, IMarketCalendar, IUnitOfWork

__all__ = [
    "ACCOUNT_STATUS_ACTIVE",
    "CHECK_BROKER",
    "CHECK_CLOCK_SKEW",
    "CHECK_DATABASE",
    "CHECK_OWNER_DECISIONS",
    "CHECK_STOP_FILE",
    "HealthCheck",
    "HealthReport",
    "HealthStatus",
    "check_broker",
    "check_clock_skew",
    "check_database",
    "check_owner_decisions",
    "check_stop_file",
    "run_health_checks",
]

CHECK_BROKER: Final = "broker"
CHECK_CLOCK_SKEW: Final = "clock_skew"
CHECK_DATABASE: Final = "database"
CHECK_OWNER_DECISIONS: Final = "owner_decisions"
CHECK_STOP_FILE: Final = "stop_file"
ACCOUNT_STATUS_ACTIVE: Final = "ACTIVE"


class HealthStatus(StrEnum):
    """Result of one check, ordered from best to worst."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"

    @property
    def rank(self) -> int:
        """0 (OK) to 2 (FAILED)."""
        return _RANK[self]


_RANK: Final[dict[HealthStatus, int]] = {
    HealthStatus.OK: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.FAILED: 2,
}


class HealthCheck(DomainModel):
    """One check result. ``detail`` is human readable and secret-free."""

    name: str
    status: HealthStatus
    detail: str
    data: dict[str, Any] = Field(default_factory=dict)


class HealthReport(DomainModel):
    """All check results of one health cycle."""

    checked_at_utc: UtcDatetime
    checks: tuple[HealthCheck, ...]

    @property
    def status(self) -> HealthStatus:
        """Worst status among the checks (``OK`` for an empty report)."""
        return max(
            (check.status for check in self.checks),
            key=lambda status: status.rank,
            default=HealthStatus.OK,
        )

    def get(self, name: str) -> HealthCheck | None:
        """The check called ``name``, if it ran."""
        return next((check for check in self.checks if check.name == name), None)

    def not_ok(self) -> tuple[HealthCheck, ...]:
        """Checks whose status is not ``OK``."""
        return tuple(check for check in self.checks if check.status is not HealthStatus.OK)

    def signature(self) -> tuple[tuple[str, HealthStatus], ...]:
        """``(name, status)`` pairs: equal signatures mean "no change" (sec. 42 records
        the result in ``system_events`` only when it changes)."""
        return tuple((check.name, check.status) for check in self.checks)


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, DomainError):
        return f"{type(exc).__name__}: {exc}"
    return type(exc).__name__


async def check_broker(broker: IBroker) -> HealthCheck:
    """Alpaca REST reachable and the account ``ACTIVE``."""
    try:
        account = await broker.get_account()
    except Exception as exc:  # noqa: BLE001 - a health check reports, it never raises
        return HealthCheck(
            name=CHECK_BROKER,
            status=HealthStatus.FAILED,
            detail=f"broker unreachable: {_error_detail(exc)}",
        )
    data = {"account_status": account.status}
    if account.status.upper() != ACCOUNT_STATUS_ACTIVE:
        return HealthCheck(
            name=CHECK_BROKER,
            status=HealthStatus.DEGRADED,
            detail=f"broker reachable but account status is {account.status}",
            data=data,
        )
    return HealthCheck(name=CHECK_BROKER, status=HealthStatus.OK, detail="reachable", data=data)


async def check_clock_skew(
    calendar: IMarketCalendar, clock: IClock, max_skew_seconds: Decimal
) -> HealthCheck:
    """Local clock vs broker clock; skew above ``max_skew_seconds`` is ``DEGRADED``."""
    before = clock.now_utc()
    try:
        market_clock = await calendar.get_clock()
    except Exception as exc:  # noqa: BLE001 - a health check reports, it never raises
        return HealthCheck(
            name=CHECK_CLOCK_SKEW,
            status=HealthStatus.FAILED,
            detail=f"broker clock unavailable: {_error_detail(exc)}",
        )
    after = clock.now_utc()
    local_mid = before + (after - before) / 2
    skew = market_clock.now_utc - local_mid
    skew_seconds = Decimal(str(round(skew / timedelta(seconds=1), 3)))
    data = {
        "skew_seconds": str(skew_seconds),
        "max_skew_seconds": str(max_skew_seconds),
        "market_open": market_clock.is_open,
    }
    if abs(skew_seconds) > max_skew_seconds:
        return HealthCheck(
            name=CHECK_CLOCK_SKEW,
            status=HealthStatus.DEGRADED,
            detail=f"clock skew {skew_seconds}s exceeds {max_skew_seconds}s",
            data=data,
        )
    return HealthCheck(
        name=CHECK_CLOCK_SKEW,
        status=HealthStatus.OK,
        detail=f"clock skew {skew_seconds}s",
        data=data,
    )


async def check_database(uow: IUnitOfWork) -> HealthCheck:
    """The operational DB accepts a write.

    The probe rewrites the current ``system_control`` row unchanged inside a transaction
    and leaves it WITHOUT committing (the unit of work rolls back), so nothing is
    persisted and a concurrent control change can never be overwritten.
    """
    try:
        async with uow:
            control = await uow.control.get()
            await uow.control.set(control)
    except Exception as exc:  # noqa: BLE001 - a health check reports, it never raises
        return HealthCheck(
            name=CHECK_DATABASE,
            status=HealthStatus.FAILED,
            detail=f"database not writable: {_error_detail(exc)}",
        )
    return HealthCheck(name=CHECK_DATABASE, status=HealthStatus.OK, detail="writable")


def check_owner_decisions(pending: Sequence[str]) -> HealthCheck:
    """``DEGRADED`` while any ``OWNER_DECISION`` is ``null``: trading cannot be enabled."""
    if pending:
        return HealthCheck(
            name=CHECK_OWNER_DECISIONS,
            status=HealthStatus.DEGRADED,
            detail=f"{len(pending)} owner decision(s) pending: trading cannot be enabled",
            data={"pending": list(pending)},
        )
    return HealthCheck(
        name=CHECK_OWNER_DECISIONS, status=HealthStatus.OK, detail="no pending decision"
    )


def check_stop_file(present: bool) -> HealthCheck:
    """``DEGRADED`` while the STOP file exists (trading disabled whatever the DB says)."""
    if present:
        return HealthCheck(
            name=CHECK_STOP_FILE,
            status=HealthStatus.DEGRADED,
            detail="STOP file present: trading disabled (sec. 31.2)",
        )
    return HealthCheck(name=CHECK_STOP_FILE, status=HealthStatus.OK, detail="absent")


async def run_health_checks(
    *,
    broker: IBroker,
    calendar: IMarketCalendar,
    clock: IClock,
    uow: IUnitOfWork,
    max_clock_skew_seconds: Decimal,
    pending_decisions: Sequence[str],
    stop_file_present: bool,
) -> HealthReport:
    """Run every Phase 2 check once and aggregate them (checks run sequentially)."""
    checks = (
        await check_broker(broker),
        await check_clock_skew(calendar, clock, max_clock_skew_seconds),
        await check_database(uow),
        check_owner_decisions(pending_decisions),
        check_stop_file(stop_file_present),
    )
    return HealthReport(checked_at_utc=clock.now_utc(), checks=checks)
