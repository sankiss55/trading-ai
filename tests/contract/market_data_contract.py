"""Reusable ``IMarketData`` contract (sec. 10, 49.2).

Subclass ``MarketDataContract`` and provide a ``market_data_harness`` fixture. For a
live feed set ``stream_sample`` to 0 outside market hours to skip the stream check.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pytest

from domain.models import Bar, Quote, Timeframe
from domain.ports import IMarketData


@dataclass(frozen=True)
class MarketDataHarness:
    """Feed under test plus ranges known to contain data."""

    feed: IMarketData
    symbol: str
    minute_start: datetime
    minute_end: datetime
    day_start: date
    day_end: date
    stream_symbols: tuple[str, ...]
    stream_sample: int


class MarketDataContract:
    """Behaviors every ``IMarketData`` implementation must have."""

    async def test_minute_bars_are_sorted_and_in_range(
        self, market_data_harness: MarketDataHarness
    ) -> None:
        h = market_data_harness
        bars = await h.feed.get_minute_bars(h.symbol, h.minute_start, h.minute_end)
        assert bars
        assert all(isinstance(bar, Bar) for bar in bars)
        assert all(bar.symbol == h.symbol and bar.timeframe is Timeframe.MIN_1 for bar in bars)
        assert all(h.minute_start <= bar.bar_start_utc < h.minute_end for bar in bars)
        starts = [bar.bar_start_utc for bar in bars]
        assert starts == sorted(set(starts))
        assert all(bar.bar_end_utc - bar.bar_start_utc == timedelta(minutes=1) for bar in bars)

    async def test_daily_bars_are_daily_and_in_range(
        self, market_data_harness: MarketDataHarness
    ) -> None:
        h = market_data_harness
        bars = await h.feed.get_daily_bars(h.symbol, h.day_start, h.day_end)
        assert bars
        assert all(bar.symbol == h.symbol and bar.timeframe is Timeframe.DAY_1 for bar in bars)
        assert all(h.day_start <= bar.bar_start_utc.date() <= h.day_end for bar in bars)

    async def test_latest_quote_is_quote_or_none(
        self, market_data_harness: MarketDataHarness
    ) -> None:
        quote = await market_data_harness.feed.get_latest_quote(market_data_harness.symbol)
        assert quote is None or (
            isinstance(quote, Quote) and quote.symbol == market_data_harness.symbol
        )

    async def test_stream_yields_requested_minute_bars_only(
        self, market_data_harness: MarketDataHarness
    ) -> None:
        h = market_data_harness
        if h.stream_sample == 0:
            pytest.skip("stream sampling disabled for this implementation")
        received: list[Bar] = []
        async for bar in h.feed.stream_minute_bars(list(h.stream_symbols)):
            received.append(bar)
            if len(received) >= h.stream_sample:
                break
        assert len(received) == h.stream_sample
        assert all(bar.symbol in h.stream_symbols for bar in received)
        assert all(bar.timeframe is Timeframe.MIN_1 for bar in received)
