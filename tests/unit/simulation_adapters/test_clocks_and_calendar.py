"""Unit tests for ``SimClock``, ``FixedClock`` and ``StaticCalendar``."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from adapters.simulation import static_calendar
from adapters.simulation.sim_clock import FixedClock, SimClock
from adapters.simulation.static_calendar import (
    StaticCalendar,
    build_regular_sessions,
    market_time_to_utc,
)
from domain.errors import NonRetryableError
from domain.ports import IClock, IMarketCalendar
from tests.unit.simulation_adapters.builders import T0


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# --------------------------------------------------------------------------- clocks


def test_sim_clock_is_monotonic() -> None:
    clock: IClock = SimClock(T0)
    assert isinstance(clock, SimClock)
    clock.advance_to(T0)  # same instant: no-op
    clock.advance_to(T0 + timedelta(minutes=1))
    assert clock.now_utc() == T0 + timedelta(minutes=1)
    with pytest.raises(NonRetryableError) as excinfo:
        clock.advance_to(T0)
    assert excinfo.value.code == "CLOCK_BACKWARDS"
    clock.advance_by(timedelta(seconds=30))
    assert clock.now_utc() == T0 + timedelta(minutes=1, seconds=30)
    with pytest.raises(NonRetryableError):
        clock.advance_by(timedelta(seconds=-1))


@pytest.mark.parametrize(
    "bad",
    [
        datetime(2026, 10, 1, 13, 30),  # noqa: DTZ001 - naive on purpose
        datetime(2026, 10, 1, 9, 30, tzinfo=timezone(-timedelta(hours=4))),
    ],
)
def test_clocks_reject_naive_or_non_utc(bad: datetime) -> None:
    with pytest.raises(ValueError, match="UTC"):
        SimClock(bad)
    with pytest.raises(ValueError, match="UTC"):
        FixedClock(bad)


def test_fixed_clock_can_move_both_ways() -> None:
    clock = FixedClock(T0)
    clock.advance(timedelta(minutes=-5))
    assert clock.now_utc() == T0 - timedelta(minutes=5)
    clock.set(T0)
    clock.advance_to(T0 - timedelta(hours=1))
    assert clock.now_utc() == T0 - timedelta(hours=1)


# --------------------------------------------------------------------------- time zones


@pytest.fixture(params=["zoneinfo", "builtin-rule"])
def tz_mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run conversions both with ``zoneinfo`` (if available) and the built-in rule."""
    if request.param == "zoneinfo":
        try:
            ZoneInfo("America/New_York")
        except ZoneInfoNotFoundError:
            pytest.skip("IANA time zone database not installed (tzdata)")
    else:
        monkeypatch.setattr(static_calendar, "_default_market_tz", lambda: None)
    return str(request.param)


@pytest.mark.usefixtures("tz_mode")
@pytest.mark.parametrize(
    ("day", "expected_open", "expected_close"),
    [
        (date(2026, 10, 1), utc(2026, 10, 1, 13, 30), utc(2026, 10, 1, 20, 0)),  # EDT
        (date(2026, 12, 1), utc(2026, 12, 1, 14, 30), utc(2026, 12, 1, 21, 0)),  # EST
        (date(2026, 3, 6), utc(2026, 3, 6, 14, 30), utc(2026, 3, 6, 21, 0)),  # before DST
        (date(2026, 3, 9), utc(2026, 3, 9, 13, 30), utc(2026, 3, 9, 20, 0)),  # DST from 3/8
        (date(2026, 10, 30), utc(2026, 10, 30, 13, 30), utc(2026, 10, 30, 20, 0)),
        (date(2026, 11, 2), utc(2026, 11, 2, 14, 30), utc(2026, 11, 2, 21, 0)),  # EST from 11/1
    ],
)
def test_regular_session_times_in_utc(
    day: date, expected_open: datetime, expected_close: datetime
) -> None:
    session = build_regular_sessions([day])[day]
    assert (session.open_utc, session.close_utc) == (expected_open, expected_close)
    assert session.is_early_close is False


@pytest.mark.usefixtures("tz_mode")
def test_early_close_session() -> None:
    day = date(2026, 11, 27)
    session = build_regular_sessions([day], early_closes={day: time(13, 0)})[day]
    assert session.close_utc == utc(2026, 11, 27, 18, 0)
    assert session.is_early_close is True


def test_early_close_for_unknown_day_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-trading"):
        build_regular_sessions([date(2026, 10, 1)], early_closes={date(2026, 10, 2): time(13)})


def test_explicit_tzinfo_is_used() -> None:
    fixed = timezone(timedelta(hours=-5))
    assert market_time_to_utc(date(2026, 7, 1), time(9, 30), fixed) == utc(2026, 7, 1, 14, 30)


def test_builtin_rule_refuses_years_before_2007(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(static_calendar, "_default_market_tz", lambda: None)
    with pytest.raises(NonRetryableError):
        market_time_to_utc(date(2006, 7, 3), time(9, 30))


# --------------------------------------------------------------------------- calendar

DAYS = [date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5)]


def calendar_at(now: datetime) -> IMarketCalendar:
    sessions = build_regular_sessions(DAYS, tz=timezone(timedelta(hours=-4)))
    return StaticCalendar(sessions, FixedClock(now))


@pytest.mark.parametrize(
    ("now", "is_open", "next_open", "next_close"),
    [
        # before the open of 10/01
        (utc(2026, 10, 1, 12, 0), False, utc(2026, 10, 1, 13, 30), utc(2026, 10, 1, 20, 0)),
        # exactly at the open: open (inclusive)
        (utc(2026, 10, 1, 13, 30), True, utc(2026, 10, 2, 13, 30), utc(2026, 10, 1, 20, 0)),
        # exactly at the close: closed (exclusive)
        (utc(2026, 10, 1, 20, 0), False, utc(2026, 10, 2, 13, 30), utc(2026, 10, 2, 20, 0)),
        # Friday after the close -> Monday
        (utc(2026, 10, 2, 22, 0), False, utc(2026, 10, 5, 13, 30), utc(2026, 10, 5, 20, 0)),
        # Saturday
        (utc(2026, 10, 3, 15, 0), False, utc(2026, 10, 5, 13, 30), utc(2026, 10, 5, 20, 0)),
    ],
)
async def test_get_clock(
    now: datetime, is_open: bool, next_open: datetime, next_close: datetime
) -> None:
    clock = await calendar_at(now).get_clock()
    assert (clock.is_open, clock.now_utc, clock.next_open_utc, clock.next_close_utc) == (
        is_open,
        now,
        next_open,
        next_close,
    )


async def test_get_clock_without_future_sessions_raises() -> None:
    with pytest.raises(NonRetryableError) as excinfo:
        await calendar_at(utc(2026, 10, 5, 15, 0)).get_clock()
    assert excinfo.value.code == "CALENDAR_EXHAUSTED"


async def test_get_session() -> None:
    calendar = calendar_at(T0)
    session = await calendar.get_session(date(2026, 10, 2))
    assert session is not None
    assert session.open_utc == utc(2026, 10, 2, 13, 30)
    assert await calendar.get_session(date(2026, 10, 3)) is None


def test_calendar_rejects_inconsistent_sessions() -> None:
    sessions = build_regular_sessions(DAYS, tz=timezone(timedelta(hours=-4)))
    with pytest.raises(ValueError, match="session key"):
        StaticCalendar({date(2026, 10, 9): sessions[date(2026, 10, 1)]}, FixedClock(T0))
    duplicated = [sessions[date(2026, 10, 1)], sessions[date(2026, 10, 1)]]
    with pytest.raises(ValueError, match="duplicate"):
        StaticCalendar(duplicated, FixedClock(T0))
