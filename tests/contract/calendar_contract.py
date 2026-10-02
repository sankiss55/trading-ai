"""Reusable ``IMarketCalendar`` contract (sec. 11, 49.2).

Subclass ``CalendarContract`` and provide a ``calendar_harness`` fixture returning a
``CalendarHarness``. A real broker calendar can be plugged in by pointing the harness
at known trading and non-trading dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from domain.models import SessionDay
from domain.ports import IMarketCalendar


@dataclass(frozen=True)
class CalendarHarness:
    """Calendar under test plus dates whose nature is known."""

    calendar: IMarketCalendar
    trading_day: date
    non_trading_day: date


class CalendarContract:
    """Behaviors every ``IMarketCalendar`` implementation must have."""

    async def test_trading_day_has_a_utc_session(self, calendar_harness: CalendarHarness) -> None:
        session = await calendar_harness.calendar.get_session(calendar_harness.trading_day)
        assert isinstance(session, SessionDay)
        assert session.session_date == calendar_harness.trading_day
        assert session.open_utc.utcoffset() == timedelta(0)
        assert session.open_utc < session.close_utc

    async def test_non_trading_day_has_no_session(self, calendar_harness: CalendarHarness) -> None:
        assert await calendar_harness.calendar.get_session(calendar_harness.non_trading_day) is None

    async def test_clock_is_internally_consistent(self, calendar_harness: CalendarHarness) -> None:
        clock = await calendar_harness.calendar.get_clock()
        assert clock.now_utc.utcoffset() == timedelta(0)
        assert clock.next_open_utc > clock.now_utc
        assert clock.next_close_utc > clock.now_utc
        if clock.is_open:
            assert clock.next_open_utc > clock.next_close_utc
        else:
            assert clock.next_open_utc < clock.next_close_utc
