"""Deterministic clocks for backtest and simulation (sec. 8.6).

* ``SimClock`` (backtest): advances with the bars and never goes backwards.
* ``FixedClock`` (tests): fully controllable, may be set to any instant.

Neither clock reads the system wall clock: the initial instant is always injected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

from domain.errors import NonRetryableError

__all__ = ["AdvanceableClock", "FixedClock", "SimClock", "ensure_utc"]


def ensure_utc(value: datetime, *, name: str = "datetime") -> datetime:
    """Return ``value`` normalized to ``datetime.UTC``.

    Raises:
        ValueError: if ``value`` is naive or has a non-zero UTC offset.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{name} must be timezone-aware (UTC); got a naive datetime")
    if offset != timedelta(0):
        raise ValueError(f"{name} must be in UTC, got offset {offset}")
    return value.replace(tzinfo=UTC)


class AdvanceableClock(Protocol):
    """A clock that simulation adapters (e.g. ``HistoricalFeed``) can move forward."""

    def now_utc(self) -> datetime:
        """Current simulated time in UTC."""
        ...

    def advance_to(self, instant: datetime) -> None:
        """Move the clock to ``instant``."""
        ...


class SimClock:
    """Monotonic simulated clock implementing ``IClock``.

    Args:
        start: Initial simulated instant (timezone-aware UTC).
    """

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start, name="start")

    def now_utc(self) -> datetime:
        """Current simulated time in UTC."""
        return self._now

    def advance_to(self, instant: datetime) -> None:
        """Move the clock forward to ``instant``. Staying at the same instant is a no-op.

        Raises:
            NonRetryableError: code ``CLOCK_BACKWARDS`` if ``instant`` is before now.
            ValueError: if ``instant`` is naive or not UTC.
        """
        target = ensure_utc(instant, name="instant")
        if target < self._now:
            raise NonRetryableError(
                f"simulated clock cannot go backwards: {target.isoformat()} < "
                f"{self._now.isoformat()}",
                code="CLOCK_BACKWARDS",
            )
        self._now = target

    def advance_by(self, delta: timedelta) -> None:
        """Move the clock forward by a non-negative ``delta``."""
        if delta < timedelta(0):
            raise NonRetryableError("delta must be non-negative", code="CLOCK_BACKWARDS")
        self._now = self._now + delta


class FixedClock:
    """Controllable test clock implementing ``IClock``. It never moves by itself.

    Unlike ``SimClock`` it may be set backwards, which tests use to simulate skew.

    Args:
        now: Initial instant (timezone-aware UTC).
    """

    def __init__(self, now: datetime) -> None:
        self._now = ensure_utc(now, name="now")

    def now_utc(self) -> datetime:
        """The instant last set."""
        return self._now

    def set(self, instant: datetime) -> None:
        """Set the clock to any instant (forwards or backwards)."""
        self._now = ensure_utc(instant, name="instant")

    def advance(self, delta: timedelta) -> None:
        """Shift the clock by ``delta`` (may be negative)."""
        self._now = self._now + delta

    def advance_to(self, instant: datetime) -> None:
        """Alias of ``set`` so that ``FixedClock`` satisfies ``AdvanceableClock``."""
        self.set(instant)
