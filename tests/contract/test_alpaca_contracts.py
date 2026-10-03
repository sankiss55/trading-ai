"""Port contracts run against the Alpaca adapters over fake SDK transports (no network).

``MarketDataContract`` / ``CalendarContract`` (sec. 49.2) run against ``AlpacaMarketData``
and ``AlpacaCalendar``. The pagination test drives the REAL ``StockHistoricalDataClient``
with its HTTP ``get`` replaced, to pin the SDK behavior the adapter relies on: pages are
followed through ``next_page_token`` inside one ``get_stock_bars`` call.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from alpaca.data.historical.stock import StockHistoricalDataClient

from adapters.alpaca import AlpacaCalendar, AlpacaMarketData, BarAdjustment
from domain.models import DataFeed
from domain.ports import IMarketCalendar, IMarketData
from tests.contract.calendar_contract import CalendarContract, CalendarHarness
from tests.contract.market_data_contract import MarketDataContract, MarketDataHarness
from tests.unit.alpaca.fakes import (
    FAKE_KEY,
    FAKE_SECRET,
    FakeBarsClient,
    FakeCalendarClient,
    SleepRecorder,
    bar_payload,
    calendar_payload,
    clock_payload,
    minute_payloads,
)

T0 = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)


class TestAlpacaMarketDataContract(MarketDataContract):
    @pytest.fixture
    def market_data_harness(self) -> MarketDataHarness:
        daily = [
            bar_payload(datetime(2026, 9, 28 + i, 4, 0, tzinfo=UTC), 500.0 + i, volume=1e6)
            for i in range(3)
        ]
        client = FakeBarsClient({("SPY", "1Min"): minute_payloads(T0, 20), ("SPY", "1Day"): daily})
        feed: IMarketData = AlpacaMarketData(
            client, feed=DataFeed.IEX, adjustment=BarAdjustment.SPLIT, sleep=SleepRecorder()
        )
        return MarketDataHarness(
            feed=feed,
            symbol="SPY",
            minute_start=T0,
            minute_end=T0 + timedelta(minutes=10),
            day_start=date(2026, 9, 28),
            day_end=date(2026, 9, 30),
            stream_symbols=("SPY",),
            stream_sample=0,  # the real-time stream is Phase 3
        )


MARKET_OPEN_CLOCK = clock_payload(
    "2026-10-02T10:00:00-04:00",
    is_open=True,
    next_open="2026-10-05T09:30:00-04:00",
    next_close="2026-10-02T16:00:00-04:00",
)
MARKET_CLOSED_CLOCK = clock_payload()  # Saturday 2026-10-03


class TestAlpacaCalendarContract(CalendarContract):
    @pytest.fixture(params=[MARKET_OPEN_CLOCK, MARKET_CLOSED_CLOCK], ids=["open", "closed"])
    def calendar_harness(self, request: pytest.FixtureRequest) -> CalendarHarness:
        days = [calendar_payload(date(2026, 10, d)) for d in (1, 2, 5)]
        client = FakeCalendarClient(days, clock=request.param)
        calendar: IMarketCalendar = AlpacaCalendar(client, sleep=SleepRecorder())
        return CalendarHarness(
            calendar=calendar, trading_day=date(2026, 10, 2), non_trading_day=date(2026, 10, 3)
        )


class _PagedDataClient(StockHistoricalDataClient):
    """Real SDK client whose HTTP GET returns scripted pages (no network)."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        super().__init__(api_key=FAKE_KEY, secret_key=FAKE_SECRET)
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def get(self, path: str, data: Any = None, **kwargs: Any) -> Any:
        self.calls.append({"path": path, **dict(data or {})})
        return self.pages[len(self.calls) - 1]


async def test_sdk_follows_next_page_token_within_one_chunk() -> None:
    bars = minute_payloads(T0, 5)
    client = _PagedDataClient(
        [
            {"bars": {"SPY": bars[:3]}, "next_page_token": "page-2"},
            {"bars": {"SPY": bars[3:]}, "next_page_token": None},
        ]
    )
    adapter = AlpacaMarketData(
        client, feed=DataFeed.IEX, adjustment=BarAdjustment.RAW, sleep=SleepRecorder()
    )
    result = await adapter.get_minute_bars("SPY", T0, T0 + timedelta(minutes=5))
    assert [b.bar_start_utc for b in result] == [T0 + timedelta(minutes=i) for i in range(5)]
    assert [c["page_token"] for c in client.calls] == [None, "page-2"]
    first = client.calls[0]
    assert first["path"] == "/stocks/bars"
    assert first["feed"] == "iex"
    assert first["adjustment"] == "raw"
    assert str(first["timeframe"]) == "1Min"  # requests encodes the TimeFrame via str()
    assert first["limit"] == 10_000  # SDK page size
