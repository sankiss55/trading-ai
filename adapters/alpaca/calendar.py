"""``IMarketCalendar`` sessions from the Alpaca trading calendar (sec. 11).

Scope (declared early pull-forward of part of Phase 2): trading days and early closes
only, used to store the historical calendar of the backtest dataset. The live market
clock is Phase 2 (``get_clock`` raises ``NotImplementedError``).

SDK facts (verified against alpaca-py 0.44.0 source):

* ``TradingClient(api_key, secret_key, paper=True).get_calendar(GetCalendarRequest(
  start=date, end=date))`` returns ``list[alpaca.trading.models.Calendar]`` (market
  days from 1970 to 2029 per the SDK docstring).
* ``Calendar.open`` / ``Calendar.close`` are **naive** datetimes built from the API's
  ``date`` + ``"HH:MM"`` strings, in America/New_York wall time. They are localized with
  ``zoneinfo`` (DST-aware) and converted to UTC here.
* A session is an early close when its local close is before 16:00.

The trading client is always built with ``paper=True`` (sec. 7.2); ``APP_ENV=live`` is
refused by the composition root before any adapter is built.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.models import Calendar
from alpaca.trading.requests import GetCalendarRequest
from pydantic import ValidationError

from adapters.alpaca._http import AlpacaCredentials, HttpTimeouts, install_timeouts
from adapters.alpaca._retry import RequestPacer, RetryPolicy, Sleep, call_with_retry
from domain.errors import NonRetryableError
from domain.models import MarketClock, SessionDay

__all__ = [
    "MARKET_TIMEZONE",
    "REGULAR_CLOSE_LOCAL",
    "AlpacaCalendar",
    "CalendarClient",
    "convert_calendar_day",
]

MARKET_TIMEZONE = ZoneInfo("America/New_York")
REGULAR_CLOSE_LOCAL = time(16, 0)
"""Regular close in market local time; anything earlier is an early close."""


class CalendarClient(Protocol):
    """The subset of ``TradingClient`` used here (injectable in tests)."""

    def get_calendar(
        self, filters: GetCalendarRequest | None = None
    ) -> list[Calendar] | dict[str, Any]:
        """Market days in the filter range."""
        ...


def _to_utc(local: datetime) -> datetime:
    if local.tzinfo is None:
        return local.replace(tzinfo=MARKET_TIMEZONE).astimezone(UTC)
    return local.astimezone(UTC)


def _local_time(moment: datetime) -> time:
    if moment.tzinfo is None:
        return moment.time()
    return moment.astimezone(MARKET_TIMEZONE).time()


def convert_calendar_day(day: Calendar) -> SessionDay:
    """Convert one SDK calendar day into a UTC ``SessionDay``.

    Raises:
        NonRetryableError: code ``INVALID_CALENDAR`` for an inconsistent day.
    """
    try:
        open_utc, close_utc = _to_utc(day.open), _to_utc(day.close)
        if day.open.date() != day.date or day.close.date() != day.date:
            raise ValueError("open/close are not on the session date")
        return SessionDay(
            session_date=day.date,
            open_utc=open_utc,
            close_utc=close_utc,
            is_early_close=_local_time(day.close) < REGULAR_CLOSE_LOCAL,
        )
    except (ValueError, ValidationError) as exc:
        raise NonRetryableError(f"calendar day {day.date}: {exc}", code="INVALID_CALENDAR") from exc


class AlpacaCalendar:
    """Trading sessions from the Alpaca calendar, cached in memory (sec. 11.3).

    Args:
        client: alpaca-py ``TradingClient`` (paper) or a test double.
        retry_policy: Bounded retry policy for every request.
        sleep: Async sleep used by retries and the pacer.
        pacer: Optional local request pacer.
        unit_random: ``U[0, 1)`` source of the retry jitter.
    """

    def __init__(
        self,
        client: CalendarClient,
        *,
        retry_policy: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
        pacer: RequestPacer | None = None,
        unit_random: Callable[[], float] = random.random,
    ) -> None:
        self._client = client
        self._policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._pacer = pacer
        self._unit_random = unit_random
        self._cache: dict[date, SessionDay | None] = {}

    @classmethod
    def from_credentials(
        cls,
        credentials: AlpacaCredentials,
        *,
        retry_policy: RetryPolicy | None = None,
        timeouts: HttpTimeouts | None = None,
        min_request_interval_seconds: float = 0.3,
    ) -> AlpacaCalendar:
        """Build the adapter over a real ``TradingClient(paper=True)`` (network).

        Only the composition root may call this (sec. 8.3.4).
        """
        client = TradingClient(
            api_key=credentials.api_key.get_secret_value(),
            secret_key=credentials.secret_key.get_secret_value(),
            paper=True,
        )
        install_timeouts(client, timeouts or HttpTimeouts())
        return cls(
            client,
            retry_policy=retry_policy,
            pacer=RequestPacer(min_request_interval_seconds),
        )

    async def get_sessions(self, start: date, end: date) -> list[SessionDay]:
        """Trading sessions with ``start <= session_date <= end``, ordered by date.

        Raises:
            RetryableError: retries exhausted (rate limit, 5xx, network).
            NonRetryableError: auth, rejected request, invalid calendar data.
        """
        if end < start:
            return []
        request = GetCalendarRequest(start=start, end=end)
        operation = f"get_calendar {start.isoformat()}..{end.isoformat()}"
        result = await call_with_retry(
            lambda: self._client.get_calendar(request),
            operation=operation,
            policy=self._policy,
            sleep=self._sleep,
            pacer=self._pacer,
            unit_random=self._unit_random,
        )
        if not isinstance(result, list):
            raise NonRetryableError(
                f"{operation}: expected a list, got {type(result).__name__}",
                code="INVALID_SCHEMA",
            )
        sessions: dict[date, SessionDay] = {}
        for item in result:
            session = convert_calendar_day(item)
            if session.session_date in sessions:
                raise NonRetryableError(
                    f"{operation}: duplicate day {session.session_date}", code="INVALID_CALENDAR"
                )
            if start <= session.session_date <= end:
                sessions[session.session_date] = session
        ordered = [sessions[key] for key in sorted(sessions)]
        for offset in range((end - start).days + 1):
            day = date.fromordinal(start.toordinal() + offset)
            self._cache[day] = sessions.get(day)
        return ordered

    async def get_session(self, day: date) -> SessionDay | None:
        """Session of ``day`` (with early close), or ``None`` if not a trading day."""
        if day not in self._cache:
            await self.get_sessions(day, day)
        return self._cache.get(day)

    async def get_clock(self) -> MarketClock:
        """The live market clock (``TradingClient.get_clock``) is Phase 2.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError("Phase 2")
