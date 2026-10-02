"""Market data models: bars, quotes, market clock and session days (sec. 8.4, 10, 11)."""

from datetime import date
from typing import Self

from pydantic import Field, model_validator

from domain.models.base import DomainModel, NonNegativeDecimal, Price, Symbol, UtcDatetime
from domain.models.enums import BarStatus, DataFeed, Timeframe

__all__ = ["Bar", "MarketClock", "Quote", "SessionDay"]


class Bar(DomainModel):
    """An OHLCV bar covering ``[bar_start_utc, bar_end_utc)``.

    * ``COMPLETE`` / ``INCOMPLETE`` bars carry all four prices.
    * ``INCOMPLETE`` bars must report ``minutes_present`` (>= 1), the number of
      1-minute bars that contributed to the aggregate (sec. 10.3.5).
    * ``EMPTY`` bars (aggregate window closed without any 1-minute bar, sec. 10.3.4)
      carry no prices, zero volume and ``minutes_present`` of ``None`` or ``0``.
      The strategy never evaluates entries on them.
    """

    symbol: Symbol
    timeframe: Timeframe
    bar_start_utc: UtcDatetime
    bar_end_utc: UtcDatetime
    open: Price | None
    high: Price | None
    low: Price | None
    close: Price | None
    volume: int = Field(ge=0)
    feed: DataFeed
    status: BarStatus
    minutes_present: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.bar_end_utc <= self.bar_start_utc:
            raise ValueError("bar_end_utc must be after bar_start_utc")
        prices = (self.open, self.high, self.low, self.close)
        if self.status is BarStatus.EMPTY:
            if any(p is not None for p in prices):
                raise ValueError("EMPTY bars must not carry prices")
            if self.volume != 0:
                raise ValueError("EMPTY bars must have zero volume")
            if self.minutes_present not in (None, 0):
                raise ValueError("EMPTY bars must have minutes_present None or 0")
            return self
        if self.open is None or self.high is None or self.low is None or self.close is None:
            raise ValueError(f"{self.status} bars must carry open, high, low and close")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC inconsistent: low <= open, close <= high is required")
        if self.status is BarStatus.INCOMPLETE and (
            self.minutes_present is None or self.minutes_present < 1
        ):
            raise ValueError("INCOMPLETE bars must report minutes_present >= 1")
        if self.status is BarStatus.COMPLETE and self.minutes_present == 0:
            raise ValueError("COMPLETE bars cannot have minutes_present == 0")
        return self


class Quote(DomainModel):
    """Latest top-of-book quote for a symbol. Sizes are Decimal because feeds may report
    them as non-integers (VERIFICAR)."""

    symbol: Symbol
    bid_price: NonNegativeDecimal
    ask_price: NonNegativeDecimal
    bid_size: NonNegativeDecimal
    ask_size: NonNegativeDecimal
    timestamp_utc: UtcDatetime


class MarketClock(DomainModel):
    """Market state as reported by the broker clock (sec. 11.1)."""

    is_open: bool
    now_utc: UtcDatetime
    next_open_utc: UtcDatetime
    next_close_utc: UtcDatetime


class SessionDay(DomainModel):
    """One regular trading session from the broker calendar, including early closes."""

    session_date: date
    open_utc: UtcDatetime
    close_utc: UtcDatetime
    is_early_close: bool

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.close_utc <= self.open_utc:
            raise ValueError("close_utc must be after open_utc")
        return self
