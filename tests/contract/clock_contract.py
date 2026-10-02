"""Reusable ``IClock`` contract (sec. 49.2).

Subclass ``ClockContract`` in a ``test_*.py`` module and provide a ``clock`` fixture.
"""

from __future__ import annotations

from datetime import timedelta

from domain.ports import IClock


class ClockContract:
    """Behaviors every ``IClock`` implementation must have."""

    def test_now_is_timezone_aware_utc(self, clock: IClock) -> None:
        now = clock.now_utc()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)

    def test_now_never_goes_backwards_between_calls(self, clock: IClock) -> None:
        first = clock.now_utc()
        second = clock.now_utc()
        assert second >= first
