"""Static market calendar for backtest and simulation (sec. 8.6, 11).

``StaticCalendar`` answers ``IMarketCalendar`` from a stored mapping of trading days to
``SessionDay`` and an injected ``IClock``. ``build_regular_sessions`` creates regular
09:30-16:00 America/New_York sessions (converted to UTC) for a list of dates, with
optional early closes.

Time zone source: ``zoneinfo`` (``America/New_York``). On platforms without an IANA
database (e.g. Windows without the ``tzdata`` package) a built-in US Eastern rule
(DST from the second Sunday of March to the first Sunday of November, in force since
2007) is used instead. Callers may also pass an explicit ``tzinfo``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from domain.errors import NonRetryableError
from domain.models import MarketClock, SessionDay
from domain.ports import IClock

__all__ = [
    "MARKET_TIMEZONE",
    "REGULAR_CLOSE",
    "REGULAR_OPEN",
    "StaticCalendar",
    "build_regular_sessions",
    "market_time_to_utc",
]

MARKET_TIMEZONE = "America/New_York"
REGULAR_OPEN = time(9, 30)
"""Regular session open in market local time (sec. 5.5)."""
REGULAR_CLOSE = time(16, 0)
"""Regular session close in market local time (sec. 5.5)."""

_EST = timedelta(hours=-5)
_EDT = timedelta(hours=-4)
_FALLBACK_FIRST_YEAR = 2007


def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    days_to_sunday = (6 - first.weekday()) % 7
    return first + timedelta(days=days_to_sunday + 7 * (n - 1))


def _us_eastern_offset(day: date) -> timedelta:
    """UTC offset of US Eastern time on ``day`` for local times between 03:00 and 24:00.

    DST transitions happen at 02:00 local on Sundays, so for session hours the offset
    only depends on the date.
    """
    if day.year < _FALLBACK_FIRST_YEAR:
        raise NonRetryableError(
            f"built-in US Eastern rule only covers {_FALLBACK_FIRST_YEAR}+; "
            "install tzdata or pass tz explicitly",
            code="TIMEZONE_UNAVAILABLE",
        )
    dst_start = _nth_sunday(day.year, 3, 2)
    dst_end = _nth_sunday(day.year, 11, 1)
    return _EDT if dst_start <= day < dst_end else _EST


def _default_market_tz() -> tzinfo | None:
    try:
        return ZoneInfo(MARKET_TIMEZONE)
    except ZoneInfoNotFoundError:
        return None


def market_time_to_utc(day: date, local_time: time, tz: tzinfo | None = None) -> datetime:
    """Convert a market-local wall time on ``day`` to an aware UTC datetime.

    Args:
        day: Calendar date in market local time.
        local_time: Wall time in market local time (naive ``time``).
        tz: Market time zone. ``None`` uses ``zoneinfo`` America/New_York or, if the
            IANA database is unavailable, the built-in US Eastern rule.
    """
    zone = tz if tz is not None else _default_market_tz()
    if zone is not None:
        return datetime.combine(day, local_time, tzinfo=zone).astimezone(UTC)
    if local_time < time(3, 0):
        raise NonRetryableError(
            "built-in US Eastern rule only supports local times from 03:00",
            code="TIMEZONE_UNAVAILABLE",
        )
    local = datetime.combine(day, local_time, tzinfo=UTC)
    return local - _us_eastern_offset(day)


def build_regular_sessions(
    days: Iterable[date],
    *,
    early_closes: Mapping[date, time] | None = None,
    open_time: time = REGULAR_OPEN,
    close_time: time = REGULAR_CLOSE,
    tz: tzinfo | None = None,
) -> dict[date, SessionDay]:
    """Build regular sessions for ``days``.

    Args:
        days: Trading dates (holidays must simply be left out).
        early_closes: Early close wall times in market local time, e.g.
            ``{date(2026, 11, 27): time(13, 0)}``. Every key must be in ``days``.
        open_time: Session open in market local time.
        close_time: Regular session close in market local time.
        tz: Market time zone (see ``market_time_to_utc``).

    Returns:
        Mapping of date to ``SessionDay`` in UTC, ordered by date.
    """
    ordered = sorted(set(days))
    closes = dict(early_closes or {})
    unknown = sorted(set(closes) - set(ordered))
    if unknown:
        raise ValueError(f"early closes for non-trading days: {unknown}")
    sessions: dict[date, SessionDay] = {}
    for day in ordered:
        is_early = day in closes
        sessions[day] = SessionDay(
            session_date=day,
            open_utc=market_time_to_utc(day, open_time, tz),
            close_utc=market_time_to_utc(day, closes[day] if is_early else close_time, tz),
            is_early_close=is_early,
        )
    return sessions


class StaticCalendar:
    """``IMarketCalendar`` backed by stored sessions and an injected clock.

    ``get_clock`` derives the market state from ``clock.now_utc()``:

    * ``is_open`` is true when ``open_utc <= now < close_utc`` for some session.
    * ``next_close_utc`` is the close of the current session when open, otherwise the
      close of the next session.
    * ``next_open_utc`` is the first session open strictly after ``now`` (so while the
      market is open it is the next day's open, like the broker clock).

    Args:
        sessions: Trading days keyed by date, or an iterable of ``SessionDay``.
        clock: Time source.
    """

    def __init__(
        self,
        sessions: Mapping[date, SessionDay] | Iterable[SessionDay],
        clock: IClock,
    ) -> None:
        values = sessions.values() if isinstance(sessions, Mapping) else sessions
        by_date: dict[date, SessionDay] = {}
        for session in values:
            if session.session_date in by_date:
                raise ValueError(f"duplicate session for {session.session_date}")
            by_date[session.session_date] = session
        if isinstance(sessions, Mapping):
            for key, session in sessions.items():
                if key != session.session_date:
                    raise ValueError(f"session key {key} != session_date {session.session_date}")
        self._sessions = by_date
        self._ordered = sorted(by_date.values(), key=lambda s: s.open_utc)
        for previous, current in zip(self._ordered, self._ordered[1:], strict=False):
            if current.open_utc < previous.close_utc:
                raise ValueError(
                    f"overlapping sessions {previous.session_date} and {current.session_date}"
                )
        self._clock = clock

    async def get_session(self, day: date) -> SessionDay | None:
        """Session of ``day`` or ``None`` if it is not a stored trading day."""
        return self._sessions.get(day)

    async def get_clock(self) -> MarketClock:
        """Market state at ``clock.now_utc()``.

        Raises:
            NonRetryableError: code ``CALENDAR_EXHAUSTED`` if no future session is stored
                to compute ``next_open_utc`` / ``next_close_utc``.
        """
        now = self._clock.now_utc()
        current = next((s for s in self._ordered if s.open_utc <= now < s.close_utc), None)
        next_open = next((s.open_utc for s in self._ordered if s.open_utc > now), None)
        next_close: datetime | None
        if current is not None:
            next_close = current.close_utc
        else:
            next_close = next((s.close_utc for s in self._ordered if s.open_utc > now), None)
        if next_open is None or next_close is None:
            raise NonRetryableError(
                f"static calendar has no session after {now.isoformat()}",
                code="CALENDAR_EXHAUSTED",
            )
        return MarketClock(
            is_open=current is not None,
            now_utc=now,
            next_open_utc=next_open,
            next_close_utc=next_close,
        )
