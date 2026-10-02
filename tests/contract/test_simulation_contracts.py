"""Port contracts run against the simulation adapters (sec. 8.6, 49.2)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from adapters.simulation.historical_feed import HistoricalFeed, ScriptedFeed
from adapters.simulation.sim_clock import FixedClock, SimClock
from adapters.simulation.simulated_broker import SimulatedBroker
from adapters.simulation.static_calendar import StaticCalendar, build_regular_sessions
from domain.models import (
    Bar,
    BarStatus,
    BracketOrderRequest,
    DataFeed,
    OrderType,
    Timeframe,
    TimeInForce,
)
from domain.ports import IBroker, IClock, IMarketCalendar, IMarketData
from tests.contract.broker_contract import BracketSemanticsContract, BrokerContract
from tests.contract.calendar_contract import CalendarContract, CalendarHarness
from tests.contract.clock_contract import ClockContract
from tests.contract.market_data_contract import MarketDataContract, MarketDataHarness

T0 = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)  # 09:30 America/New_York (EDT)
TRADING_DAYS = [date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5)]

# --------------------------------------------------------------------------- IClock


class TestSimClockContract(ClockContract):
    @pytest.fixture
    def clock(self) -> IClock:
        return SimClock(T0)


class TestFixedClockContract(ClockContract):
    @pytest.fixture
    def clock(self) -> IClock:
        return FixedClock(T0)


# --------------------------------------------------------------------------- IMarketCalendar


class TestStaticCalendarContract(CalendarContract):
    @pytest.fixture(
        params=[T0 + timedelta(minutes=30), datetime(2026, 10, 1, 21, 0, tzinfo=UTC)],
        ids=["market-open", "market-closed"],
    )
    def calendar_harness(self, request: pytest.FixtureRequest) -> CalendarHarness:
        calendar: IMarketCalendar = StaticCalendar(
            build_regular_sessions(TRADING_DAYS), FixedClock(request.param)
        )
        return CalendarHarness(
            calendar=calendar, trading_day=date(2026, 10, 2), non_trading_day=date(2026, 10, 3)
        )


# --------------------------------------------------------------------------- IMarketData


def _minute_bar(symbol: str, start: datetime, price: Decimal) -> Bar:
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.MIN_1,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(minutes=1),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=100,
        feed=DataFeed.IEX,
        status=BarStatus.COMPLETE,
    )


def _daily_bar(symbol: str, day: date, price: Decimal) -> Bar:
    start = datetime(day.year, day.month, day.day, 4, 0, tzinfo=UTC)
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.DAY_1,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(days=1),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1_000_000,
        feed=DataFeed.IEX,
        status=BarStatus.COMPLETE,
    )


def _sample_bars() -> list[Bar]:
    bars = [
        _minute_bar(symbol, T0 + timedelta(minutes=i), Decimal(100 + i))
        for symbol in ("SPY", "QQQ")
        for i in range(10)
    ]
    bars += [
        _daily_bar("SPY", date(2026, 9, 28) + timedelta(days=i), Decimal(500 + i)) for i in range(4)
    ]
    return bars


def _write_csv(directory: Path, bars: list[Bar]) -> None:
    files: dict[str, list[str]] = {}
    for bar in bars:
        name = f"{bar.symbol}_{bar.timeframe.value}.csv"
        row = (
            f"{bar.bar_start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')},{bar.open},{bar.high},"
            f"{bar.low},{bar.close},{bar.volume}"
        )
        files.setdefault(name, ["t,o,h,l,c,v"]).append(row)
    for name, lines in files.items():
        (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestMarketDataContracts(MarketDataContract):
    @pytest.fixture(params=["historical", "historical-csv", "scripted"])
    def market_data_harness(
        self, request: pytest.FixtureRequest, tmp_path: Path
    ) -> MarketDataHarness:
        bars = _sample_bars()
        feed: IMarketData
        if request.param == "historical":
            feed = HistoricalFeed(bars)
        elif request.param == "historical-csv":
            _write_csv(tmp_path, bars)
            feed = HistoricalFeed.from_csv_dir(tmp_path, feed=DataFeed.IEX)
        else:
            feed = ScriptedFeed([b for b in bars if b.timeframe is Timeframe.MIN_1], history=bars)
        return MarketDataHarness(
            feed=feed,
            symbol="SPY",
            minute_start=T0,
            minute_end=T0 + timedelta(minutes=10),
            day_start=date(2026, 9, 28),
            day_end=date(2026, 9, 30),
            stream_symbols=("SPY", "QQQ"),
            stream_sample=3,
        )


# --------------------------------------------------------------------------- IBroker


class SimBrokerHarness:
    """Bracket harness over ``SimulatedBroker`` with scripted 1-minute bars."""

    entry_ref = Decimal(100)
    stop_price = Decimal(98)
    take_profit_price = Decimal(102)

    def __init__(self) -> None:
        self.clock = SimClock(T0)
        self.sim = SimulatedBroker(
            clock=self.clock, starting_cash=Decimal(100_000), slippage_bps=Decimal(5)
        )
        self._sequence = 0

    @property
    def broker(self) -> IBroker:
        return self.sim

    @property
    def symbol(self) -> str:
        return "SPY"

    def client_order_id(self, tag: str) -> str:
        self._sequence += 1
        return f"test-{tag}-{self._sequence}"

    def bracket_request(
        self, client_order_id: str, *, qty: int = 1, tif: TimeInForce = TimeInForce.DAY
    ) -> BracketOrderRequest:
        return BracketOrderRequest(
            symbol=self.symbol,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=tif,
            take_profit_limit_price=self.take_profit_price,
            stop_loss_stop_price=self.stop_price,
            client_order_id=client_order_id,
        )

    def resting_bracket_request(self, client_order_id: str) -> BracketOrderRequest:
        return BracketOrderRequest(
            symbol=self.symbol,
            qty=1,
            order_type=OrderType.LIMIT,
            limit_price=Decimal(50),
            time_in_force=TimeInForce.DAY,
            take_profit_limit_price=Decimal(150),
            stop_loss_stop_price=Decimal(49),
            client_order_id=client_order_id,
        )

    async def run_bar(self, open_: Decimal, high: Decimal, low: Decimal, close: Decimal) -> None:
        start = self.clock.now_utc()
        bar = Bar(
            symbol=self.symbol,
            timeframe=Timeframe.MIN_1,
            bar_start_utc=start,
            bar_end_utc=start + timedelta(minutes=1),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=100,
            feed=DataFeed.IEX,
            status=BarStatus.COMPLETE,
        )
        self.sim.process_bar(bar)
        self.clock.advance_to(bar.bar_end_utc)

    async def fill_pending_market_orders(self) -> None:
        await self.run_bar(
            self.entry_ref,
            self.entry_ref + Decimal("0.5"),
            self.entry_ref - Decimal("0.5"),
            self.entry_ref,
        )

    async def end_session(self) -> None:
        self.sim.close_session()


class TestSimulatedBrokerContract(BrokerContract):
    @pytest.fixture
    def broker_harness(self) -> SimBrokerHarness:
        return SimBrokerHarness()


class TestSimulatedBrokerBracketSemantics(BracketSemanticsContract):
    @pytest.fixture
    def broker_harness(self) -> SimBrokerHarness:
        return SimBrokerHarness()
