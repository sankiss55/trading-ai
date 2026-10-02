"""Unit tests for ``HistoricalFeed``, CSV loading and ``ScriptedFeed``."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from adapters.simulation.historical_feed import HistoricalFeed, ScriptedFeed, load_bars_csv
from adapters.simulation.sim_clock import SimClock
from domain.errors import NonRetryableError
from domain.models import Bar, BarStatus, DataFeed, Quote, Timeframe
from domain.ports import IMarketData
from tests.unit.simulation_adapters.builders import T0, minute_bar

MINUTE = timedelta(minutes=1)


def bars_for(symbol: str, count: int, *, start: datetime = T0) -> list[Bar]:
    return [
        minute_bar(start + i * MINUTE, "100", "101", "99", "100.5", symbol=symbol)
        for i in range(count)
    ]


async def collect(feed: IMarketData, symbols: list[str]) -> list[Bar]:
    return [bar async for bar in feed.stream_minute_bars(symbols)]


# --------------------------------------------------------------------------- CSV


def test_load_csv_parses_decimals_and_bar_end(tmp_path: Path) -> None:
    path = tmp_path / "SPY_5Min.csv"
    path.write_text(
        "t,o,h,l,c,v\n2026-10-01T13:30:00Z,500.10,501.00,499.90,500.50,1000\n",
        encoding="utf-8",
    )
    [bar] = load_bars_csv(path, symbol="SPY", timeframe=Timeframe.MIN_5, feed=DataFeed.IEX)
    assert bar.bar_start_utc == T0
    assert bar.bar_end_utc == T0 + timedelta(minutes=5)
    assert (bar.open, bar.high, bar.low, bar.close) == (
        Decimal("500.10"),
        Decimal("501.00"),
        Decimal("499.90"),
        Decimal("500.50"),
    )
    assert bar.volume == 1000


def test_daily_csv_bar_covers_one_day(tmp_path: Path) -> None:
    path = tmp_path / "SPY_1Day.csv"
    path.write_text("t,o,h,l,c,v\n2026-10-01T04:00:00+00:00,1,2,1,2,10\n", encoding="utf-8")
    [bar] = load_bars_csv(path, symbol="SPY", timeframe=Timeframe.DAY_1, feed=DataFeed.SIP)
    assert bar.bar_end_utc - bar.bar_start_utc == timedelta(days=1)


@pytest.mark.parametrize(
    "content",
    [
        "time,o,h,l,c,v\n2026-10-01T13:30:00Z,1,2,1,2,10\n",  # bad header
        "t,o,h,l,c,v\n2026-10-01T09:30:00-04:00,1,2,1,2,10\n",  # not UTC
        "t,o,h,l,c,v\n2026-10-01T13:30:00,1,2,1,2,10\n",  # naive
        "t,o,h,l,c,v\n2026-10-01T13:30:00Z,1,2,abc,2,10\n",  # bad decimal
        "t,o,h,l,c,v\n2026-10-01T13:30:00Z,1,2,1.5,2,10\n",  # low above open
    ],
)
def test_invalid_csv_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "SPY_1Min.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(NonRetryableError) as excinfo:
        load_bars_csv(path, symbol="SPY", timeframe=Timeframe.MIN_1, feed=DataFeed.IEX)
    assert excinfo.value.code == "INVALID_BAR_DATA"


def test_csv_dir_rejects_unknown_timeframe(tmp_path: Path) -> None:
    (tmp_path / "SPY_2Min.csv").write_text("t,o,h,l,c,v\n", encoding="utf-8")
    with pytest.raises(NonRetryableError):
        HistoricalFeed.from_csv_dir(tmp_path, feed=DataFeed.IEX)


# --------------------------------------------------------------------------- HistoricalFeed


def test_duplicate_bars_are_rejected() -> None:
    with pytest.raises(NonRetryableError) as excinfo:
        HistoricalFeed(bars_for("SPY", 2) + bars_for("SPY", 1))
    assert excinfo.value.code == "DUPLICATE_BAR"


async def test_minute_range_returns_only_fully_closed_bars_inside_the_window() -> None:
    feed = HistoricalFeed(bars_for("SPY", 5))
    bars = await feed.get_minute_bars("SPY", T0 + MINUTE, T0 + 3 * MINUTE + timedelta(seconds=30))
    assert [b.bar_start_utc for b in bars] == [T0 + MINUTE, T0 + 2 * MINUTE]
    assert await feed.get_minute_bars("QQQ", T0, T0 + timedelta(hours=1)) == []


async def test_stream_is_time_ordered_with_symbol_tie_break() -> None:
    feed = HistoricalFeed(list(reversed(bars_for("SPY", 2) + bars_for("AAPL", 2))))
    received = await collect(feed, ["SPY", "AAPL"])
    assert [(b.bar_start_utc, b.symbol) for b in received] == [
        (T0, "AAPL"),
        (T0, "SPY"),
        (T0 + MINUTE, "AAPL"),
        (T0 + MINUTE, "SPY"),
    ]


async def test_stream_advances_the_clock_to_each_bar_end() -> None:
    clock = SimClock(T0)
    feed = HistoricalFeed(bars_for("SPY", 3), clock=clock)
    seen: list[datetime] = []
    async for bar in feed.stream_minute_bars(["SPY"]):
        assert clock.now_utc() == bar.bar_end_utc
        seen.append(clock.now_utc())
    assert seen == [T0 + MINUTE, T0 + 2 * MINUTE, T0 + 3 * MINUTE]


async def test_stream_skips_bars_already_in_the_past() -> None:
    clock = SimClock(T0 + 2 * MINUTE)
    feed = HistoricalFeed(bars_for("SPY", 4), clock=clock)
    received = await collect(feed, ["SPY"])
    assert [b.bar_start_utc for b in received] == [T0 + MINUTE, T0 + 2 * MINUTE, T0 + 3 * MINUTE]


async def test_clock_prevents_lookahead_in_history_and_quotes() -> None:
    clock = SimClock(T0 + 2 * MINUTE)
    quotes = {
        "SPY": [
            Quote(
                symbol="SPY",
                bid_price=Decimal("100.00"),
                ask_price=Decimal("100.02"),
                bid_size=Decimal(1),
                ask_size=Decimal(1),
                timestamp_utc=T0 + offset * MINUTE,
            )
            for offset in (3, 1)
        ]
    }
    feed = HistoricalFeed(bars_for("SPY", 5), clock=clock, quotes=quotes)
    bars = await feed.get_minute_bars("SPY", T0, T0 + timedelta(hours=1))
    assert [b.bar_end_utc for b in bars] == [T0 + MINUTE, T0 + 2 * MINUTE]
    quote = await feed.get_latest_quote("SPY")
    assert quote is not None
    assert quote.timestamp_utc == T0 + MINUTE
    assert await feed.get_latest_quote("QQQ") is None


async def test_daily_bars_filter_by_inclusive_utc_dates() -> None:
    daily = [
        Bar(
            symbol="SPY",
            timeframe=Timeframe.DAY_1,
            bar_start_utc=datetime(2026, 9, day, 4, tzinfo=UTC),
            bar_end_utc=datetime(2026, 9, day, 4, tzinfo=UTC) + timedelta(days=1),
            open=Decimal(1),
            high=Decimal(2),
            low=Decimal(1),
            close=Decimal(2),
            volume=1,
            feed=DataFeed.IEX,
            status=BarStatus.COMPLETE,
        )
        for day in (28, 29, 30)
    ]
    feed = HistoricalFeed(daily)
    bars = await feed.get_daily_bars("SPY", date(2026, 9, 29), date(2026, 9, 30))
    assert [b.bar_start_utc.day for b in bars] == [29, 30]


# --------------------------------------------------------------------------- ScriptedFeed


async def test_scripted_feed_replays_duplicates_and_out_of_order_bars() -> None:
    first, second = bars_for("SPY", 2)
    clock = SimClock(T0)
    feed = ScriptedFeed([second, first, second, *bars_for("QQQ", 1)], clock=clock)
    received = await collect(feed, ["SPY"])
    assert received == [second, first, second]
    assert clock.now_utc() == second.bar_end_utc  # never moved backwards


async def test_scripted_feed_history_and_quotes() -> None:
    history = bars_for("SPY", 3)
    feed = ScriptedFeed(history=history, quotes={"SPY": None})
    assert await feed.get_minute_bars("SPY", T0, T0 + 2 * MINUTE) == history[:2]
    assert await feed.get_daily_bars("SPY", date(2026, 10, 1), date(2026, 10, 1)) == []
    assert await feed.get_latest_quote("SPY") is None
