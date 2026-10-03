"""System wall clock (sec. 8.6: ``IClock`` for ``paper`` / ``dev``).

``SystemClock`` is the only ``IClock`` implementation that reads the operating-system
time. ``domain/`` and ``application/`` never read the wall clock (sec. 8.3.6, enforced
by ``tests/architecture/``); they receive this clock through the composition root.
The host is expected to keep its clock synchronized (NTP, sec. 58.1); the health
monitor compares it with the broker clock (``max_clock_skew_seconds``, sec. 42).
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["SystemClock"]


class SystemClock:
    """``IClock`` over the operating-system clock, always timezone-aware UTC."""

    def now_utc(self) -> datetime:
        """Current wall-clock time in UTC."""
        return datetime.now(UTC)
