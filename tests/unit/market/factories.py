"""Test factories for market-domain tests (explicit fixture values, no OWNER_DECISIONs)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from domain.models import Bar, BarStatus, DataFeed, SessionDay, Timeframe

# Summer (EDT, UTC-4) regular session: 09:30-16:00 ET = 13:30-20:00 UTC.
SUMMER_DAY = SessionDay(
    session_date=date(2026, 6, 15),
    open_utc=datetime(2026, 6, 15, 13, 30, tzinfo=UTC),
    close_utc=datetime(2026, 6, 15, 20, 0, tzinfo=UTC),
    is_early_close=False,
)
# Next session (Tuesday).
SUMMER_NEXT_DAY = SessionDay(
    session_date=date(2026, 6, 16),
    open_utc=datetime(2026, 6, 16, 13, 30, tzinfo=UTC),
    close_utc=datetime(2026, 6, 16, 20, 0, tzinfo=UTC),
    is_early_close=False,
)
# Winter (EST, UTC-5) regular session: 09:30-16:00 ET = 14:30-21:00 UTC.
WINTER_DAY = SessionDay(
    session_date=date(2026, 1, 12),
    open_utc=datetime(2026, 1, 12, 14, 30, tzinfo=UTC),
    close_utc=datetime(2026, 1, 12, 21, 0, tzinfo=UTC),
    is_early_close=False,
)
# Early close (EST): 09:30-13:00 ET = 14:30-18:00 UTC.
EARLY_CLOSE_DAY = SessionDay(
    session_date=date(2026, 11, 27),
    open_utc=datetime(2026, 11, 27, 14, 30, tzinfo=UTC),
    close_utc=datetime(2026, 11, 27, 18, 0, tzinfo=UTC),
    is_early_close=True,
)


def minute_bar(
    start: datetime,
    *,
    symbol: str = "SPY",
    open_: str = "100",
    high: str = "101",
    low: str = "99",
    close: str = "100.5",
    volume: int = 1000,
    feed: DataFeed = DataFeed.IEX,
) -> Bar:
    """A COMPLETE 1Min bar starting at ``start``."""
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.MIN_1,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(minutes=1),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=volume,
        feed=feed,
        status=BarStatus.COMPLETE,
    )


def daily_bar(day: date, volume: int, *, symbol: str = "SPY") -> Bar:
    """A COMPLETE 1Day bar labelled at 04:00 UTC of ``day``."""
    start = datetime(day.year, day.month, day.day, 4, 0, tzinfo=UTC)
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.DAY_1,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(days=1),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=volume,
        feed=DataFeed.IEX,
        status=BarStatus.COMPLETE,
    )
