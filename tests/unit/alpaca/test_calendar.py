"""AlpacaCalendar: New York wall times -> UTC (DST), early closes, cache, retries."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Calendar
from pydantic import SecretStr

from adapters.alpaca import AlpacaCalendar, AlpacaCredentials, RetryPolicy
from adapters.alpaca._http import _TimeoutAdapter
from adapters.alpaca.calendar import convert_calendar_day
from domain.errors import NonRetryableError, RetryableError
from tests.unit.alpaca.fakes import (
    FAKE_KEY,
    FAKE_SECRET,
    FakeCalendarClient,
    SleepRecorder,
    api_error,
    calendar_payload,
)

DAYS = [
    calendar_payload(date(2025, 3, 7)),  # Friday, EST (UTC-5)
    calendar_payload(date(2025, 3, 10)),  # Monday after the DST switch, EDT (UTC-4)
    calendar_payload(date(2025, 7, 3), close_hhmm="13:00"),  # early close
    calendar_payload(date(2025, 7, 7)),  # July 4th holiday is absent
    calendar_payload(date(2025, 11, 28), close_hhmm="13:00"),  # day after Thanksgiving, EST
]
POLICY = RetryPolicy(max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=10.0)


def _calendar(client: FakeCalendarClient, sleep: SleepRecorder | None = None) -> AlpacaCalendar:
    return AlpacaCalendar(
        client, retry_policy=POLICY, sleep=sleep or SleepRecorder(), unit_random=lambda: 0.0
    )


async def test_sessions_are_converted_to_utc_across_dst() -> None:
    sessions = await _calendar(FakeCalendarClient(DAYS)).get_sessions(
        date(2025, 3, 1), date(2025, 3, 31)
    )
    est, edt = sessions
    assert est.open_utc == datetime(2025, 3, 7, 14, 30, tzinfo=UTC)
    assert est.close_utc == datetime(2025, 3, 7, 21, 0, tzinfo=UTC)
    assert edt.open_utc == datetime(2025, 3, 10, 13, 30, tzinfo=UTC)
    assert edt.close_utc == datetime(2025, 3, 10, 20, 0, tzinfo=UTC)
    assert not est.is_early_close
    assert not edt.is_early_close


async def test_early_closes_are_flagged() -> None:
    calendar = _calendar(FakeCalendarClient(DAYS))
    july = await calendar.get_session(date(2025, 7, 3))
    assert july is not None
    assert july.is_early_close
    assert july.close_utc == datetime(2025, 7, 3, 17, 0, tzinfo=UTC)  # 13:00 EDT
    november = await calendar.get_session(date(2025, 11, 28))
    assert november is not None
    assert november.is_early_close
    assert november.close_utc == datetime(2025, 11, 28, 18, 0, tzinfo=UTC)  # 13:00 EST


async def test_get_sessions_is_ordered_and_bounded() -> None:
    client = FakeCalendarClient(list(reversed(DAYS)))
    sessions = await _calendar(client).get_sessions(date(2025, 7, 1), date(2025, 7, 31))
    assert [s.session_date for s in sessions] == [date(2025, 7, 3), date(2025, 7, 7)]
    request = client.requests[0]
    assert request is not None
    assert (request.start, request.end) == (date(2025, 7, 1), date(2025, 7, 31))


async def test_non_trading_day_is_none_and_cached() -> None:
    client = FakeCalendarClient(DAYS)
    calendar = _calendar(client)
    await calendar.get_sessions(date(2025, 7, 1), date(2025, 7, 8))
    assert await calendar.get_session(date(2025, 7, 4)) is None
    assert await calendar.get_session(date(2025, 7, 7)) is not None
    assert len(client.requests) == 1  # answered from the in-memory calendar (sec. 11.3)


async def test_calendar_retries_rate_limit() -> None:
    sleep = SleepRecorder()
    client = FakeCalendarClient(DAYS, failures=[api_error(429)])
    session = await _calendar(client, sleep).get_session(date(2025, 3, 7))
    assert session is not None
    assert sleep.delays == [1.0]


async def test_calendar_gives_up_after_bounded_attempts() -> None:
    client = FakeCalendarClient(DAYS, failures=[api_error(500)] * 5)
    with pytest.raises(RetryableError):
        await _calendar(client).get_session(date(2025, 3, 7))
    assert len(client.requests) == POLICY.max_attempts


async def test_auth_error_is_not_retried() -> None:
    client = FakeCalendarClient(DAYS, failures=[api_error(401)])
    with pytest.raises(NonRetryableError) as info:
        await _calendar(client).get_session(date(2025, 3, 7))
    assert info.value.code == "ALPACA_AUTH"
    assert len(client.requests) == 1


def test_inconsistent_calendar_day_is_rejected() -> None:
    day = Calendar(**calendar_payload(date(2025, 3, 7), open_hhmm="16:00", close_hhmm="09:30"))
    with pytest.raises(NonRetryableError) as info:
        convert_calendar_day(day)
    assert info.value.code == "INVALID_CALENDAR"


async def test_clock_is_phase_2() -> None:
    with pytest.raises(NotImplementedError, match="Phase 2"):
        await _calendar(FakeCalendarClient(DAYS)).get_clock()


def test_from_credentials_forces_paper_trading_client() -> None:
    credentials = AlpacaCredentials(api_key=SecretStr(FAKE_KEY), secret_key=SecretStr(FAKE_SECRET))
    calendar = AlpacaCalendar.from_credentials(credentials)
    client = calendar._client
    assert isinstance(client, TradingClient)
    assert client._base_url == "https://paper-api.alpaca.markets"
    assert isinstance(
        client._session.get_adapter("https://paper-api.alpaca.markets"), _TimeoutAdapter
    )
