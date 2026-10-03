"""``application.daily_cycle``: live daily bars -> ``MarketFlow`` once per session (Phase 3).

Synthetic bars only (no network). Fixture config ``tests/fixtures/config.daily.yaml``
(window 20 sessions, entry ``ibs < 0.2``, SIP feed). The backtest oracle is the real
runner (``backtest.parity.backtest_decisions`` records every runner decision).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from adapters.simulation import SimClock, SimulatedBroker, StaticCalendar
from application.daily_cycle import (
    ALREADY_STARTED,
    BACKFILL_EMPTY,
    DATA_NOT_FINAL,
    DUPLICATE_DAILY_BAR,
    FEED_MISMATCH,
    INCOMPLETE_BAR,
    INVALID_OHLC,
    MISSING_BAR,
    NEXT_SESSION_UNKNOWN,
    NOT_LATEST_SESSION,
    SESSION_BAR_MISMATCH,
    SESSION_NOT_IN_CALENDAR,
    CycleReport,
    CycleStatus,
    DailyCycleParams,
    DailyDataCycle,
    SymbolStatus,
    outcome_fields,
)
from application.market_flow import (
    DUPLICATE_BAR,
    SESSION_UNKNOWN,
    SIGNAL_EXPIRED,
    BarOutcome,
    EntryProposal,
    Rejected,
    TradeBook,
)
from backtest.parity import DecisionRecord, backtest_decisions, market_flow_for
from domain.errors import NonRetryableError, RetryableError
from domain.models import Bar, BarStatus, DataFeed, Quote, SessionDay, StrategyAction
from tests.simulation.daily_helpers import (
    Ohlc,
    daily_bar,
    daily_loaded,
    dataset,
    flat_day,
    sessions_for,
    weekdays,
)

CASH = Decimal(100000)
DAYS = weekdays(date(2025, 2, 3), 40)
SESSIONS = sessions_for(DAYS)
START = 25
"""First decided session; the 20 sessions before it are the warm-up."""
WINDOW = 20
SYMBOLS = ("AAA", "BBB")
SIGNAL = Ohlc("100", "101", "99", "99.2")
"""IBS 0.1: BUY after the close (entry rule ``ibs < 0.2``)."""
CONFIG = daily_loaded(
    whitelist=list(SYMBOLS),
    backtest={"start_date": DAYS[START], "end_date": DAYS[-1], "out_of_sample_start": DAYS[30]},
).config
BOOK = TradeBook(
    week_start_equity=CASH,
    peak_equity=CASH,
    pending_entries=(),
    position_stops=(),
    entry_fills={},
    last_exits={},
    executed_signal_ids=frozenset(),
)


def _series(signal_at: int | None = None) -> list[Ohlc | None]:
    series: list[Ohlc | None] = [flat_day() for _ in DAYS]
    if signal_at is not None:
        series[signal_at] = SIGNAL
    return series


def _bars(series: Mapping[str, Sequence[Ohlc | None]]) -> list[Bar]:
    return [
        daily_bar(symbol, day, ohlc)
        for symbol, items in series.items()
        for day, ohlc in zip(DAYS, items, strict=True)
        if ohlc is not None
    ]


class MutableFeed:
    """``IMarketData`` over a mutable list of daily bars (late bars, outages)."""

    def __init__(self, bars: Sequence[Bar]) -> None:
        self.bars = list(bars)
        self.failing: set[str] = set()
        self.calls: list[tuple[str, date, date]] = []

    async def get_daily_bars(self, symbol: str, start: date, end: date) -> list[Bar]:
        self.calls.append((symbol, start, end))
        if symbol in self.failing:
            raise RetryableError("market data unavailable", code="RATE_LIMITED")
        return [
            bar
            for bar in self.bars
            if bar.symbol == symbol and start <= bar.bar_start_utc.date() <= end
        ]

    async def get_minute_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        return []

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        return None

    def stream_minute_bars(self, symbols: Sequence[str]) -> AsyncIterator[Bar]:
        raise NotImplementedError


class Harness:
    """A daily cycle on a SimClock, a static calendar and an empty simulated account."""

    def __init__(
        self,
        bars: Sequence[Bar],
        *,
        delay_seconds: int = 1200,
        calendar: Sequence[SessionDay] = SESSIONS,
        clock_start: datetime | None = None,
    ) -> None:
        self.clock = SimClock(clock_start or SESSIONS[START].open_utc)
        self.broker = SimulatedBroker(clock=self.clock, starting_cash=CASH, slippage_bps=Decimal(5))
        self.feed = MutableFeed(bars)
        # Liquidity filter source (check library): the same bars, its own call log, so
        # ``feed.calls`` records only the cycle's fetches.
        self.liquidity = MutableFeed(())
        self.liquidity.bars = self.feed.bars
        self.flow = market_flow_for(
            CONFIG,
            clock=self.clock,
            broker=self.broker,
            sessions=calendar,
            liquidity_data=self.liquidity,
        )
        self.decided = 0
        decide = self.flow.on_closed_bar

        async def counting(bar: Bar, *, book: TradeBook) -> BarOutcome:
            self.decided += 1
            return await decide(bar, book=book)

        self.flow.on_closed_bar = counting  # type: ignore[method-assign]
        self.cycle = DailyDataCycle(
            flow=self.flow,
            market_data=self.feed,
            calendar=StaticCalendar(calendar, self.clock),
            clock=self.clock,
            params=DailyCycleParams(
                symbols=("BBB", "AAA"), feed=DataFeed.SIP, data_delay_seconds=delay_seconds
            ),
        )

    async def close(self, index: int) -> CycleReport:
        """Run the cycle of session ``index`` at its data-ready instant."""
        self.clock.advance_to(self.cycle.data_ready_at(SESSIONS[index]))
        return await self.cycle.run_session_close(SESSIONS[index], book=BOOK)


async def _backtest(
    series: Mapping[str, Sequence[Ohlc | None]], tmp_path: Path, last: int
) -> dict[tuple[date, str], DecisionRecord]:
    data = dataset(series, DAYS, tmp_path)
    return await backtest_decisions(CONFIG, data, SESSIONS[START : last + 1], starting_cash=CASH)


# --------------------------------------------------------------------------- parity


async def test_run_session_close_decides_like_the_backtest(tmp_path: Path) -> None:
    signal_day = START + 3
    series = {"AAA": _series(signal_day), "BBB": _series()}
    records = await _backtest(series, tmp_path, signal_day)
    harness = Harness(_bars(series))
    await harness.cycle.warm_up(SESSIONS[START])
    for index in range(START, signal_day + 1):
        report = await harness.close(index)
        assert report.status is CycleStatus.COMPLETED
        assert [r.symbol for r in report.symbols] == ["AAA", "BBB"]  # sorted, like the backtest
        for result in report.symbols:
            record = records[(DAYS[index], result.symbol)]
            assert result.status is SymbolStatus.DECIDED
            assert result.bar == record.bar
            assert result.outcome is not None
            assert result.outcome_kind == record.outcome.kind
            action, signal_id, rules, reasons = outcome_fields(record.outcome)
            assert (result.action, result.signal_id) == (action, signal_id)
            assert result.rule_results == rules
            assert result.reasons == reasons
            assert harness.flow.window(result.symbol) == record.window
    entry = report.result("AAA")
    assert isinstance(entry.outcome, EntryProposal)
    expected_entry = records[(DAYS[signal_day], "AAA")].outcome
    assert isinstance(expected_entry, EntryProposal)
    assert entry.signal_id == expected_entry.signal.signal_id
    assert entry.outcome.trade == expected_entry.trade
    assert entry.outcome.signal.expires_at_utc == expected_entry.signal.expires_at_utc
    assert entry.action is StrategyAction.BUY
    assert report.result("BBB").reasons == ("NO_SIGNAL",)


async def test_warm_up_emits_no_decision_and_leaves_the_backtest_window(tmp_path: Path) -> None:
    series = {"AAA": _series(START - 2), "BBB": _series()}  # a BUY bar inside the warm-up
    records = await _backtest(series, tmp_path, START)
    harness = Harness(_bars(series))
    warm = await harness.cycle.warm_up(SESSIONS[START])
    assert warm.complete
    assert warm.sessions == SESSIONS[START - WINDOW : START]
    assert harness.decided == 0
    assert await harness.broker.get_open_orders() == []
    for symbol in SYMBOLS:
        assert harness.flow.window(symbol) == records[(DAYS[START], symbol)].window_before
    assert [h.symbol for h in warm.symbols] == ["AAA", "BBB"]
    assert all(len(h.recorded) == WINDOW for h in warm.symbols)


async def test_without_warm_up_the_first_cycle_warms_up_through_the_same_path(
    tmp_path: Path,
) -> None:
    series = {"AAA": _series(START), "BBB": _series()}
    records = await _backtest(series, tmp_path, START)
    harness = Harness(_bars(series))
    report = await harness.close(START)
    assert report.history_sessions == tuple(DAYS[START - WINDOW : START])
    assert harness.decided == 2
    for symbol in SYMBOLS:
        assert harness.flow.window(symbol) == records[(DAYS[START], symbol)].window
    assert isinstance(report.result("AAA").outcome, EntryProposal)


# --------------------------------------------------------------------------- fail closed


async def test_the_cycle_refuses_to_decide_before_close_plus_delay() -> None:
    harness = Harness(_bars({"AAA": _series(START), "BBB": _series()}))
    await harness.cycle.warm_up(SESSIONS[START])
    before = {s: harness.flow.window(s) for s in SYMBOLS}
    ready = SESSIONS[START].close_utc + timedelta(minutes=20)
    harness.clock.advance_to(ready - timedelta(seconds=1))
    report = await harness.cycle.run_session_close(SESSIONS[START], book=BOOK)
    assert report.status is CycleStatus.REFUSED
    assert report.reasons == (DATA_NOT_FINAL,)
    assert report.data_ready_at_utc == ready
    assert {r.status for r in report.symbols} == {SymbolStatus.STALE}
    assert all(r.outcome is None and r.reasons == (DATA_NOT_FINAL,) for r in report.symbols)
    assert {s: harness.flow.window(s) for s in SYMBOLS} == before
    assert harness.decided == 0
    harness.clock.advance_to(ready)
    report = await harness.cycle.run_session_close(SESSIONS[START], book=BOOK)
    assert report.status is CycleStatus.COMPLETED
    assert isinstance(report.result("AAA").outcome, EntryProposal)


async def test_the_data_delay_is_configurable() -> None:
    harness = Harness(_bars({"AAA": _series(), "BBB": _series()}), delay_seconds=0)
    await harness.cycle.warm_up(SESSIONS[START])
    harness.clock.advance_to(SESSIONS[START].close_utc)
    report = await harness.cycle.run_session_close(SESSIONS[START], book=BOOK)
    assert report.status is CycleStatus.COMPLETED


def _incomplete(bar: Bar) -> Bar:
    return bar.model_copy(update={"status": BarStatus.INCOMPLETE, "minutes_present": 200})


def _insane(bar: Bar) -> Bar:
    return Bar.model_construct(**{**dict(bar), "high": Decimal("98"), "low": Decimal("99")})


def _after_open(bar: Bar) -> Bar:
    late = datetime.combine(bar.bar_start_utc.date(), datetime.min.time(), tzinfo=UTC)
    return bar.model_copy(update={"bar_start_utc": late + timedelta(hours=15)})


@pytest.mark.parametrize(
    ("mutate", "status", "reason"),
    [
        (None, SymbolStatus.MISSING, MISSING_BAR),
        (_incomplete, SymbolStatus.INCOMPLETE, INCOMPLETE_BAR),
        (
            lambda bar: bar.model_copy(update={"feed": DataFeed.IEX}),
            SymbolStatus.FEED_MISMATCH,
            FEED_MISMATCH,
        ),
        (_insane, SymbolStatus.INVALID, INVALID_OHLC),
        (_after_open, SymbolStatus.INVALID, SESSION_BAR_MISMATCH),
    ],
    ids=["missing", "incomplete", "feed_mismatch", "insane_ohlc", "label_after_open"],
)
async def test_a_bad_bar_means_no_decision_for_that_symbol(
    mutate: Callable[[Bar], Bar] | None, status: SymbolStatus, reason: str
) -> None:
    bars = _bars({"AAA": _series(START), "BBB": _series(START)})
    target = next(b for b in bars if b.symbol == "AAA" and b.bar_start_utc.date() == DAYS[START])
    bars.remove(target)
    if mutate is not None:
        bars.append(mutate(target))
    harness = Harness(bars)
    await harness.cycle.warm_up(SESSIONS[START])
    window_before = harness.flow.window("AAA")
    report = await harness.close(START)
    assert report.status is CycleStatus.PARTIAL
    aaa = report.result("AAA")
    assert aaa.status is status
    assert aaa.reasons == (reason,)
    assert aaa.outcome is None
    assert aaa.signal_id is None
    assert harness.flow.window("AAA") == window_before  # nothing recorded for AAA
    assert isinstance(report.result("BBB").outcome, EntryProposal)
    assert harness.decided == 1


async def test_two_bars_for_one_session_are_refused() -> None:
    bars = _bars({"AAA": _series(START), "BBB": _series()})
    target = next(b for b in bars if b.symbol == "AAA" and b.bar_start_utc.date() == DAYS[START])
    bars.append(target.model_copy(update={"close": Decimal("100.5")}))
    harness = Harness(bars)
    await harness.cycle.warm_up(SESSIONS[START])
    aaa = (await harness.close(START)).result("AAA")
    assert aaa.status is SymbolStatus.INVALID
    assert aaa.reasons == (DUPLICATE_DAILY_BAR,)


async def test_a_feed_mismatch_is_refused_in_the_history_too() -> None:
    bars = [
        bar.model_copy(update={"feed": DataFeed.IEX}) if bar.symbol == "AAA" else bar
        for bar in _bars({"AAA": _series(), "BBB": _series()})
    ]
    harness = Harness(bars)
    warm = await harness.cycle.warm_up(SESSIONS[START])
    assert not warm.complete
    aaa = next(h for h in warm.symbols if h.symbol == "AAA")
    assert aaa.failed_session == DAYS[START - WINDOW]
    assert aaa.reasons == (f"{FEED_MISMATCH} {DAYS[START - WINDOW].isoformat()}",)
    assert harness.flow.window("AAA") == ()
    report = await harness.close(START)
    assert report.result("AAA").status is SymbolStatus.BACKFILL_FAILED
    assert report.result("BBB").status is SymbolStatus.DECIDED


async def test_a_market_data_outage_fails_closed_per_symbol() -> None:
    harness = Harness(_bars({"AAA": _series(START), "BBB": _series(START)}))
    await harness.cycle.warm_up(SESSIONS[START])
    harness.feed.failing.add("AAA")
    report = await harness.close(START)
    aaa = report.result("AAA")
    assert aaa.status is SymbolStatus.FETCH_FAILED
    assert aaa.reasons == ("FETCH_FAILED RATE_LIMITED",)
    assert report.result("BBB").status is SymbolStatus.DECIDED
    harness.feed.failing.clear()  # recovered: the next cycle refills the session first
    report = await harness.close(START + 1)
    history = report.result("AAA").history
    assert history is not None
    assert history.recorded == (DAYS[START],)
    assert report.result("AAA").status is SymbolStatus.DECIDED


# --------------------------------------------------------------------------- gaps


async def test_a_late_bar_is_refilled_and_the_window_converges_to_the_backtest(
    tmp_path: Path,
) -> None:
    series = {"AAA": _series(START + 1), "BBB": _series()}
    records = await _backtest(series, tmp_path, START + 1)
    bars = _bars(series)
    late = next(b for b in bars if b.symbol == "AAA" and b.bar_start_utc.date() == DAYS[START])
    bars.remove(late)
    harness = Harness(bars)
    await harness.cycle.warm_up(SESSIONS[START])
    assert (await harness.close(START)).result("AAA").status is SymbolStatus.MISSING
    harness.feed.bars.append(late)  # published after the cycle
    report = await harness.close(START + 1)
    aaa = report.result("AAA")
    assert aaa.history is not None
    assert aaa.history.recorded == (DAYS[START],)
    assert aaa.history.empty_sessions == ()
    assert report.history_sessions == (DAYS[START],)
    assert harness.flow.window("AAA") == records[(DAYS[START + 1], "AAA")].window
    expected = records[(DAYS[START + 1], "AAA")].outcome
    assert isinstance(aaa.outcome, EntryProposal)
    assert isinstance(expected, EntryProposal)
    assert aaa.signal_id == expected.signal.signal_id


async def test_a_session_still_missing_at_refill_becomes_an_empty_bar(tmp_path: Path) -> None:
    series: dict[str, Sequence[Ohlc | None]] = {"AAA": _series(), "BBB": _series()}
    gappy = list(series["AAA"])
    gappy[START] = None  # no bar at all (e.g. a halted symbol): the backtest stores EMPTY
    series["AAA"] = gappy
    records = await _backtest(series, tmp_path, START + 1)
    harness = Harness(_bars(series))
    await harness.cycle.warm_up(SESSIONS[START])
    assert (await harness.close(START)).result("AAA").status is SymbolStatus.MISSING
    report = await harness.close(START + 1)
    history = report.result("AAA").history
    assert history is not None
    assert history.empty_sessions == (DAYS[START],)
    assert history.reasons == (f"{BACKFILL_EMPTY} {DAYS[START].isoformat()}",)
    window = harness.flow.window("AAA")
    assert window[-2].status is BarStatus.EMPTY
    assert window == records[(DAYS[START + 1], "AAA")].window


async def test_missed_sessions_are_backfilled_in_order_before_deciding(tmp_path: Path) -> None:
    decide_at = START + 3
    series = {"AAA": _series(decide_at), "BBB": _series()}
    records = await _backtest(series, tmp_path, decide_at)
    harness = Harness(_bars(series))
    await harness.cycle.warm_up(SESSIONS[START])
    await harness.close(START)
    decided = harness.decided
    harness.feed.calls.clear()
    report = await harness.close(decide_at)  # the process was down for two sessions
    assert report.status is CycleStatus.COMPLETED
    assert report.history_sessions == (DAYS[START + 1], DAYS[START + 2])
    for symbol in SYMBOLS:
        result = report.result(symbol)
        assert result.history is not None
        assert result.history.recorded == (DAYS[START + 1], DAYS[START + 2])
        assert harness.flow.window(symbol) == records[(DAYS[decide_at], symbol)].window
    assert harness.decided == decided + 2  # the refilled sessions were not decided
    assert harness.feed.calls == [
        ("AAA", DAYS[START + 1], DAYS[decide_at]),
        ("BBB", DAYS[START + 1], DAYS[decide_at]),
    ]
    entry = report.result("AAA")
    expected = records[(DAYS[decide_at], "AAA")].outcome
    assert isinstance(entry.outcome, EntryProposal)
    assert isinstance(expected, EntryProposal)
    assert entry.signal_id == expected.signal.signal_id
    assert entry.outcome.trade == expected.trade


async def test_only_the_latest_closed_session_is_decided() -> None:
    harness = Harness(_bars({"AAA": _series(), "BBB": _series()}))
    await harness.cycle.warm_up(SESSIONS[START])
    harness.clock.advance_to(harness.cycle.data_ready_at(SESSIONS[START + 1]))
    assert await harness.cycle.latest_closed_session() == SESSIONS[START + 1]
    report = await harness.cycle.run_session_close(SESSIONS[START], book=BOOK)
    assert report.status is CycleStatus.REFUSED
    assert report.reasons == (NOT_LATEST_SESSION,)
    assert report.next_session == SESSIONS[START + 1]
    assert harness.decided == 0
    report = await harness.cycle.run_session_close(SESSIONS[START + 1], book=BOOK)
    assert report.status is CycleStatus.COMPLETED
    assert report.history_sessions == (DAYS[START],)


async def test_a_repeated_close_never_yields_a_second_signal() -> None:
    harness = Harness(_bars({"AAA": _series(START), "BBB": _series()}))
    await harness.cycle.warm_up(SESSIONS[START])
    first = await harness.close(START)
    assert isinstance(first.result("AAA").outcome, EntryProposal)
    again = await harness.cycle.run_session_close(SESSIONS[START], book=BOOK)
    assert [r.reasons for r in again.symbols] == [(DUPLICATE_BAR,), (DUPLICATE_BAR,)]
    assert all(r.signal_id is None for r in again.symbols)


# --------------------------------------------------------------------------- calendar, book


async def test_the_last_calendar_session_has_no_next_open_so_entries_are_rejected() -> None:
    harness = Harness(
        _bars({"AAA": _series(START), "BBB": _series()}), calendar=SESSIONS[: START + 1]
    )
    await harness.cycle.warm_up(SESSIONS[START])
    report = await harness.close(START)
    assert report.next_session is None
    assert report.reasons == (NEXT_SESSION_UNKNOWN,)
    aaa = report.result("AAA").outcome
    assert isinstance(aaa, Rejected)
    # Every check runs (sec. 19): without a next open the signal also keeps its fallback
    # expiry ``bar_end + signal_ttl_seconds``, already past after the data delay.
    assert aaa.codes == (SESSION_UNKNOWN, SIGNAL_EXPIRED)


async def test_a_session_the_calendar_does_not_know_is_refused() -> None:
    harness = Harness(_bars({"AAA": _series(), "BBB": _series()}))
    await harness.cycle.warm_up(SESSIONS[START])
    moved = SESSIONS[START].model_copy(
        update={"close_utc": SESSIONS[START].close_utc - timedelta(hours=3)}
    )
    harness.clock.advance_to(harness.cycle.data_ready_at(SESSIONS[START]))
    report = await harness.cycle.run_session_close(moved, book=BOOK)
    assert report.status is CycleStatus.REFUSED
    assert report.reasons == (SESSION_NOT_IN_CALENDAR,)


async def test_warm_up_runs_once_and_only_on_final_data() -> None:
    harness = Harness(
        _bars({"AAA": _series(), "BBB": _series()}), clock_start=SESSIONS[START - 1].open_utc
    )
    harness.clock.advance_to(SESSIONS[START - 1].close_utc + timedelta(minutes=5))
    with pytest.raises(NonRetryableError) as info:
        await harness.cycle.warm_up(SESSIONS[START])
    assert info.value.code == DATA_NOT_FINAL
    harness.clock.advance_to(SESSIONS[START].open_utc)
    await harness.cycle.warm_up(SESSIONS[START])
    with pytest.raises(NonRetryableError) as info:
        await harness.cycle.warm_up(SESSIONS[START])
    assert info.value.code == ALREADY_STARTED


async def test_a_book_provider_is_awaited_per_symbol_in_decision_order() -> None:
    harness = Harness(_bars({"AAA": _series(), "BBB": _series()}))
    await harness.cycle.warm_up(SESSIONS[START])
    asked: list[str] = []

    async def book_for(symbol: str) -> TradeBook:
        asked.append(symbol)
        return BOOK

    harness.clock.advance_to(harness.cycle.data_ready_at(SESSIONS[START]))
    await harness.cycle.run_session_close(SESSIONS[START], book=book_for)
    assert asked == ["AAA", "BBB"]
