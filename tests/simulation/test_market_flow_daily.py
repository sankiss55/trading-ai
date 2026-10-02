"""``application.market_flow`` with a ``1Day`` primary timeframe (strategy family v2)."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from adapters.simulation import SimClock, SimulatedBroker
from app.config import AppConfig, exit_params, risk_params, session_window_params
from application.market_flow import (
    OUTSIDE_ENTRY_WINDOW,
    SESSION_UNKNOWN,
    SIGNAL_EXPIRED,
    EntryProposal,
    MarketFlow,
    MarketFlowParams,
    Rejected,
    TradeBook,
)
from domain.errors import NonRetryableError
from domain.market.session import session_bar
from domain.models import Bar, DataFeed, SessionDay, Timeframe
from domain.strategy.signals import signal_id_for
from domain.strategy.strategy import Strategy
from tests.simulation.daily_helpers import (
    Ohlc,
    daily_bar,
    daily_loaded,
    flat_day,
    sessions_for,
    weekdays,
)

CASH = Decimal(100000)
DAYS = weekdays(date(2025, 3, 3), 22)
SESSIONS = sessions_for(DAYS)
SIGNAL_DAY = 20
"""The last-but-one session: IBS 0.1 after 20 neutral sessions (ATR warm)."""


def _series() -> list[Ohlc]:
    series = [flat_day() for _ in DAYS]
    series[SIGNAL_DAY] = Ohlc("100", "101", "99", "99.2")
    return series


def _stamped(index: int, ohlc: Ohlc, *, feed: DataFeed = DataFeed.SIP) -> Bar:
    return session_bar(daily_bar("AAA", DAYS[index], ohlc, feed=feed), SESSIONS[index])


def _book() -> TradeBook:
    return TradeBook(
        week_start_equity=CASH,
        peak_equity=CASH,
        pending_entries=(),
        position_stops=(),
        entry_fills={},
        last_exits={},
    )


class DailyHarness:
    """A daily flow wired to a SimClock and a SimulatedBroker (no runner involved)."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or daily_loaded(whitelist=["AAA"]).config
        self.clock = SimClock(SESSIONS[0].open_utc)
        self.broker = SimulatedBroker(clock=self.clock, starting_cash=CASH, slippage_bps=Decimal(5))
        self.flow = MarketFlow(
            strategy=Strategy(self.config.strategy, strategy_version=self.config.strategy_version),
            exit_params=exit_params(self.config),
            risk_params=risk_params(self.config),
            params=MarketFlowParams(
                symbols=("AAA",),
                feed=DataFeed.SIP,
                bar_close_grace_seconds=self.config.market_data.bar_close_grace_seconds,
                window_bars=self.config.market_data.history_warmup_bars,
                session=session_window_params(self.config),
                limit_entry_offset_bps=None,
            ),
            clock=self.clock,
            broker=self.broker,
        )

    def warm_up(self, until: int) -> None:
        """Register and record the sessions before ``until`` (no decision)."""
        series = _series()
        for index in range(until):
            self.flow.add_session(SESSIONS[index])
            self.clock.advance_to(SESSIONS[index].close_utc + timedelta(seconds=11))
            assert self.flow.record_closed_bar(_stamped(index, series[index]))

    def at_close(self, index: int) -> None:
        self.clock.advance_to(SESSIONS[index].close_utc + timedelta(seconds=11))


async def test_daily_signal_expires_at_the_next_open_plus_ttl() -> None:
    harness = DailyHarness()
    harness.warm_up(SIGNAL_DAY)
    harness.flow.add_session(SESSIONS[SIGNAL_DAY])
    harness.flow.add_session(SESSIONS[SIGNAL_DAY + 1])
    harness.at_close(SIGNAL_DAY)
    bar = _stamped(SIGNAL_DAY, _series()[SIGNAL_DAY])
    outcome = await harness.flow.on_closed_bar(bar, book=_book())
    assert isinstance(outcome, EntryProposal)
    signal = outcome.signal
    assert signal.timeframe is Timeframe.DAY_1
    assert signal.bar_start_utc == SESSIONS[SIGNAL_DAY].open_utc
    assert signal.bar_end_utc == SESSIONS[SIGNAL_DAY].close_utc
    assert signal.created_at_utc == harness.clock.now_utc()
    assert signal.expires_at_utc == SESSIONS[SIGNAL_DAY + 1].open_utc + timedelta(seconds=120)
    assert signal.signal_id == signal_id_for(
        strategy_version=harness.config.strategy_version,
        symbol="AAA",
        timeframe=Timeframe.DAY_1,
        bar_start_utc=SESSIONS[SIGNAL_DAY].open_utc,
    )
    assert outcome.trade.entry_ref == Decimal("99.2")
    assert outcome.trade.stop_price == Decimal("93.20")  # 99.2 - 3 x ATR(14) of 2


async def test_daily_entry_needs_the_next_session() -> None:
    harness = DailyHarness()
    harness.warm_up(SIGNAL_DAY)
    harness.flow.add_session(SESSIONS[SIGNAL_DAY])  # the next session is not registered
    harness.at_close(SIGNAL_DAY)
    outcome = await harness.flow.on_closed_bar(
        _stamped(SIGNAL_DAY, _series()[SIGNAL_DAY]), book=_book()
    )
    assert isinstance(outcome, Rejected)
    assert outcome.codes == (SESSION_UNKNOWN,)


async def test_daily_decision_before_the_close_is_outside_the_entry_window() -> None:
    harness = DailyHarness()
    harness.warm_up(SIGNAL_DAY)
    harness.flow.add_session(SESSIONS[SIGNAL_DAY])
    harness.flow.add_session(SESSIONS[SIGNAL_DAY + 1])
    harness.clock.advance_to(SESSIONS[SIGNAL_DAY].close_utc - timedelta(minutes=1))
    outcome = await harness.flow.on_closed_bar(
        _stamped(SIGNAL_DAY, _series()[SIGNAL_DAY]), book=_book()
    )
    assert isinstance(outcome, Rejected)
    assert outcome.codes == (OUTSIDE_ENTRY_WINDOW,)


async def test_daily_signal_expired_after_the_next_open() -> None:
    harness = DailyHarness()
    harness.warm_up(SIGNAL_DAY)
    harness.flow.add_session(SESSIONS[SIGNAL_DAY])
    harness.flow.add_session(SESSIONS[SIGNAL_DAY + 1])
    # Decided too late: after the next open + TTL (e.g. a restart during the session).
    harness.clock.advance_to(SESSIONS[SIGNAL_DAY + 1].open_utc + timedelta(seconds=121))
    outcome = await harness.flow.on_closed_bar(
        _stamped(SIGNAL_DAY, _series()[SIGNAL_DAY]), book=_book()
    )
    assert isinstance(outcome, Rejected)
    assert outcome.codes == (SIGNAL_EXPIRED,)


def test_daily_flow_refuses_minute_bars_and_closes_nothing_by_time() -> None:
    harness = DailyHarness()
    harness.flow.add_session(SESSIONS[0])
    assert harness.flow.on_clock() == ()
    with pytest.raises(NonRetryableError) as info:
        harness.flow.on_minute_bar(_stamped(0, flat_day()))
    assert info.value.code == "UNSUPPORTED_TIMEFRAME"


def test_daily_bars_must_be_stamped_to_a_registered_session() -> None:
    harness = DailyHarness()
    harness.flow.add_session(SESSIONS[0])
    unstamped = daily_bar("AAA", DAYS[0], flat_day())  # labelled at midnight New York
    with pytest.raises(NonRetryableError) as info:
        harness.flow.record_closed_bar(unstamped)
    assert info.value.code == "INVALID_BAR"
    unregistered = _stamped(1, flat_day())
    with pytest.raises(NonRetryableError) as info:
        harness.flow.record_closed_bar(unregistered)
    assert info.value.code == "INVALID_BAR"
    assert harness.flow.window("AAA") == ()


def test_daily_bars_must_come_from_the_strategy_feed() -> None:
    harness = DailyHarness()
    harness.flow.add_session(SESSIONS[0])
    with pytest.raises(NonRetryableError) as info:
        harness.flow.record_closed_bar(_stamped(0, flat_day(), feed=DataFeed.IEX))
    assert info.value.code == "FEED_MISMATCH"


def test_daily_sessions_must_be_registered_in_order() -> None:
    harness = DailyHarness()
    harness.flow.add_session(SESSIONS[1])
    for session in (SESSIONS[0], SESSIONS[1]):
        with pytest.raises(NonRetryableError) as info:
            harness.flow.add_session(session)
        assert info.value.code == "SESSION_OUT_OF_ORDER"


def test_daily_primary_refuses_a_confirmation_timeframe() -> None:
    config = daily_loaded(
        whitelist=["AAA"], strategy={"confirmation_timeframe": Timeframe.MIN_15}
    ).config
    with pytest.raises(NonRetryableError) as info:
        DailyHarness(config)
    assert info.value.code == "CONFIG_INVALID"


def test_session_of_finds_the_containing_session_only() -> None:
    harness = DailyHarness()
    for session in SESSIONS[:3]:
        harness.flow.add_session(session)
    first: SessionDay = SESSIONS[0]
    assert harness.flow.session_of(first.open_utc) == first
    assert harness.flow.session_of(first.close_utc - timedelta(seconds=1)) == first
    assert harness.flow.session_of(first.close_utc) is None  # after the close
    assert harness.flow.session_of(first.open_utc - timedelta(seconds=1)) is None
    assert harness.flow.session_of(SESSIONS[2].open_utc) == SESSIONS[2]
    assert harness.flow.session_of(SESSIONS[2].close_utc + timedelta(days=1)) is None
