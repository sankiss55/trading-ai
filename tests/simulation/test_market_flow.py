"""``application.market_flow`` use cases against the simulation adapters (sec. 8.7)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from adapters.simulation import SimClock, SimulatedBroker
from app.config import AppConfig, exit_params, risk_params, session_window_params
from application.market_flow import (
    DUPLICATE_BAR,
    ENTRY_PENDING,
    OUTSIDE_ENTRY_WINDOW,
    BarOutcome,
    EntryProposal,
    ExitRequest,
    Hold,
    MarketFlow,
    MarketFlowParams,
    NoAction,
    Rejected,
    TradeBook,
)
from backtest.data import BacktestData, load_backtest_data
from backtest.runner import simulation_gate
from domain.errors import StateCriticalError
from domain.market.aggregator import IngestStatus
from domain.market.session import compute_session_windows
from domain.models import (
    Bar,
    BracketOrderRequest,
    DataFeed,
    OrderType,
    SessionDay,
    TimeInForce,
)
from domain.risk.exits import round_down_to_increment
from domain.risk.risk_engine import PendingEntry
from domain.strategy.strategy import Strategy
from tests.simulation.helpers import fixture_loaded, make_dataset

CASH = Decimal(100000)
END = date(2025, 1, 6)

Pair = tuple[Bar, BarOutcome]


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> BacktestData:
    directory: Path = make_dataset(tmp_path_factory.mktemp("flow"), end=END)
    return load_backtest_data(directory, feed=DataFeed.IEX)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return fixture_loaded().config


class Harness:
    """A flow wired to a SimClock and a SimulatedBroker (no runner involved)."""

    def __init__(self, config: AppConfig, data: BacktestData, *, limit_bps: Decimal | None = None):
        self.config = config
        self.data = data
        self.clock = SimClock(data.sessions[0].open_utc)
        self.broker = SimulatedBroker(clock=self.clock, starting_cash=CASH, slippage_bps=Decimal(5))
        self.flow = MarketFlow(
            strategy=Strategy(config.strategy, strategy_version=config.strategy_version),
            exit_params=exit_params(config),
            risk_params=risk_params(config),
            params=MarketFlowParams(
                symbols=("SYNTH",),
                feed=DataFeed.IEX,
                bar_close_grace_seconds=config.market_data.bar_close_grace_seconds,
                window_bars=config.market_data.history_warmup_bars,
                session=session_window_params(config),
                limit_entry_offset_bps=limit_bps,
            ),
            clock=self.clock,
            broker=self.broker,
            gate=simulation_gate(
                config, sessions=data.sessions, clock=self.clock, liquidity_data=data.feed
            ),
        )

    async def minutes(self, session: SessionDay) -> list[Bar]:
        return await self.data.feed.get_minute_bars("SYNTH", session.open_utc, session.close_utc)

    async def replay(
        self,
        sessions: tuple[SessionDay, ...],
        book: Callable[[], TradeBook],
        *,
        duplicate: bool = False,
    ) -> list[Pair]:
        pairs: list[Pair] = []
        for session in sessions:
            self.clock.advance_to(session.open_utc)
            self.flow.add_session(session)
            for bar in await self.minutes(session):
                self.clock.advance_to(bar.bar_end_utc)
                result = self.flow.on_minute_bar(bar)
                if duplicate:
                    again = self.flow.on_minute_bar(bar)
                    assert again.ingest.status is IngestStatus.DUPLICATE
                    assert again.closed_bars == ()
                for closed in result.closed_bars:
                    pairs.append((closed, await self.flow.on_closed_bar(closed, book=book())))
        return pairs


def empty_book() -> TradeBook:
    return TradeBook(
        week_start_equity=CASH,
        peak_equity=CASH,
        pending_entries=(),
        position_stops=(),
        entry_fills={},
        last_exits={},
        executed_signal_ids=frozenset(),
    )


async def test_duplicate_minute_bars_produce_the_same_decisions(
    config: AppConfig, data: BacktestData
) -> None:
    """AC-03: a duplicated bar never produces a second signal."""
    once = await Harness(config, data).replay(data.sessions, empty_book)
    twice = await Harness(config, data).replay(data.sessions, empty_book, duplicate=True)
    assert [(b, o.model_dump()) for b, o in once] == [(b, o.model_dump()) for b, o in twice]
    signal_ids = [o.signal.signal_id for _, o in twice if isinstance(o, EntryProposal | Rejected)]
    assert signal_ids
    assert len(signal_ids) == len(set(signal_ids))


async def test_duplicate_closed_bar_is_ignored(config: AppConfig, data: BacktestData) -> None:
    harness = Harness(config, data)
    pairs = await harness.replay(data.sessions[:1], empty_book)
    bar, _ = pairs[-1]
    outcome = await harness.flow.on_closed_bar(bar, book=empty_book())
    assert isinstance(outcome, NoAction)
    assert outcome.reason == DUPLICATE_BAR


async def test_entry_proposals_come_from_the_domain_pipeline(
    config: AppConfig, data: BacktestData
) -> None:
    pairs = await Harness(config, data).replay(data.sessions, empty_book)
    proposals = [(b, o) for b, o in pairs if isinstance(o, EntryProposal)]
    assert proposals
    for bar, proposal in proposals:
        trade = proposal.trade
        assert trade.entry_ref == bar.close  # market entry: signal bar close (15.2)
        assert trade.stop_price == proposal.exit_levels.stop_price
        assert trade.take_profit_price == proposal.exit_levels.take_profit_price
        assert trade.signal_id == proposal.signal.signal_id
        assert trade.qty >= 1
        assert all(check.passed for check in proposal.checks)
        assert proposal.limit_price is None
        session = next(s for s in data.sessions if s.open_utc <= bar.bar_start_utc < s.close_utc)
        windows = compute_session_windows(session, session_window_params(config))
        created = proposal.signal.created_at_utc
        assert windows.is_entry_window(created)
        assert not windows.is_flatten_time(created)
    rejected = [o for _, o in pairs if isinstance(o, Rejected)]
    assert any(OUTSIDE_ENTRY_WINDOW in o.codes for o in rejected)


async def test_limit_entries_use_the_rounded_limit_price_as_entry_ref(
    config: AppConfig, data: BacktestData
) -> None:
    harness = Harness(config, data, limit_bps=Decimal(10))
    pairs = await harness.replay(data.sessions, empty_book)
    proposals = [(b, o) for b, o in pairs if isinstance(o, EntryProposal)]
    assert proposals
    for bar, proposal in proposals:
        assert bar.close is not None
        expected = round_down_to_increment(bar.close * Decimal("1.001"))
        assert proposal.limit_price == expected
        assert proposal.trade.entry_ref == expected


async def test_pending_entry_blocks_new_entries(config: AppConfig, data: BacktestData) -> None:
    def pending_book() -> TradeBook:
        entry = PendingEntry(symbol="SYNTH", qty=1, entry_ref=Decimal(100), stop_price=Decimal(99))
        return empty_book().model_copy(update={"pending_entries": (entry,)})

    pairs = await Harness(config, data).replay(data.sessions, pending_book)
    assert pairs
    assert all(isinstance(o, NoAction) and o.reason == ENTRY_PENDING for _, o in pairs)


async def test_position_requires_a_recorded_entry_fill(
    config: AppConfig, data: BacktestData
) -> None:
    harness = Harness(config, data)
    *warmup, session = data.sessions
    await harness.replay(tuple(warmup), empty_book)
    harness.clock.advance_to(session.open_utc)
    harness.flow.add_session(session)
    minutes = await harness.minutes(session)
    first = minutes[0]
    assert first.open is not None
    await harness.broker.submit_bracket(
        BracketOrderRequest(
            symbol="SYNTH",
            qty=1,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            take_profit_limit_price=round_down_to_increment(first.open * 2),
            stop_loss_stop_price=round_down_to_increment(first.open / 2),
            client_order_id="test-entry",
        )
    )
    closed: list[Bar] = []
    consumed = 0
    for bar in minutes:
        consumed += 1
        harness.clock.advance_to(bar.bar_end_utc)
        harness.broker.process_bar(bar)
        closed.extend(harness.flow.on_minute_bar(bar).closed_bars)
        if closed:
            break
    assert await harness.broker.get_positions()
    with pytest.raises(StateCriticalError) as info:
        await harness.flow.on_closed_bar(closed[0], book=empty_book())
    assert info.value.code == "STATE_MISMATCH"

    # With the entry fill recorded the position is managed (HOLD or system exit).
    nxt: list[Bar] = []
    for bar in minutes[consumed:]:
        harness.clock.advance_to(bar.bar_end_utc)
        nxt.extend(harness.flow.on_minute_bar(bar).closed_bars)
        if nxt:
            break
    book = empty_book().model_copy(update={"entry_fills": {"SYNTH": first.bar_start_utc}})
    outcome = await harness.flow.on_closed_bar(nxt[0], book=book)
    assert isinstance(outcome, Hold | ExitRequest)
    assert harness.flow.window("SYNTH")[-1] == nxt[0]
    assert nxt[0].bar_start_utc - closed[0].bar_start_utc == timedelta(minutes=5)
