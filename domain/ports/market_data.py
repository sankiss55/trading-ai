"""Market data and calendar ports (sec. 8.5)."""

from collections.abc import AsyncIterator, Sequence
from datetime import date, datetime
from typing import Protocol

from domain.models import Bar, MarketClock, Quote, SessionDay

__all__ = ["IMarketCalendar", "IMarketData"]


class IMarketData(Protocol):
    """Market data source: live minute bars, historical bars and quotes."""

    def stream_minute_bars(self, symbols: Sequence[str]) -> AsyncIterator[Bar]:
        """Real-time 1-minute bars, already converted to domain models."""
        ...

    async def get_minute_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        """Historical 1-minute bars for warm-up, gap filling and backtest."""
        ...

    async def get_daily_bars(self, symbol: str, start: date, end: date) -> list[Bar]:
        """Daily bars for the liquidity filters."""
        ...

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        """Latest available quote, or ``None`` if the feed does not offer one."""
        ...


class IMarketCalendar(Protocol):
    """Market clock and trading calendar as reported by the broker (sec. 11)."""

    async def get_clock(self) -> MarketClock:
        """Current market state according to the broker."""
        ...

    async def get_session(self, day: date) -> SessionDay | None:
        """Session of the day (with early close if any), or ``None`` if not a trading day."""
        ...
