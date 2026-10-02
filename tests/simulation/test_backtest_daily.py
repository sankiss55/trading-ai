"""End-to-end daily (``1Day``, swing) backtests on synthetic daily bars (family v2 engine).

Scenario: 20 neutral warm-up sessions (February 2025, IBS 0.5, range 2, so ATR(14) = 2),
trading from 2025-03-03. Session ``D`` closes at IBS 0.1 (entry rule ``ibs < 0.2``): the
signal is decided after its close and the market entry fills at the open of ``D + 1``.
Fixture rules: exit when ``ibs > 0.8`` or after ``time_stop_bars = 5`` sessions; stop =
entry_ref - 3 x ATR = 99.2 - 6 = 93.20.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from adapters.simulation import HistoricalFeed
from backtest.__main__ import main as cli_main
from backtest.data import BacktestData
from backtest.report import BacktestReport
from backtest.research import run_research
from backtest.runner import SimulationResult, run_backtest, simulate
from domain.errors import NonRetryableError
from domain.models import Bar, ExitReason, HoldingMode, Timeframe
from domain.strategy.signals import signal_id_for
from tests.simulation.daily_helpers import (
    DAILY_CONFIG,
    Ohlc,
    daily_loaded,
    dataset,
    flat_day,
    random_walk,
    sessions_for,
    weekdays,
    write_dataset,
)

CASH = Decimal(100000)
DAYS = weekdays(date(2025, 2, 3), 60)
SESSIONS = sessions_for(DAYS)
D = 25
"""Index of the signal session (2025-03-10)."""
PERIOD = {"start_date": DAYS[20], "end_date": DAYS[-1], "out_of_sample_start": DAYS[40]}
SLIP = Decimal(5) / Decimal(10000)
TICK = Decimal("0.0001")
SIGNAL_DAY = Ohlc("100", "101", "99", "99.2")
"""IBS 0.1: BUY after the close."""


def _series(**changes: Ohlc | None) -> list[Ohlc | None]:
    """Neutral days, the signal day ``D`` and ``changes`` keyed ``d<offset from D>``."""
    series: list[Ohlc | None] = [flat_day() for _ in DAYS]
    series[D] = SIGNAL_DAY
    for key, ohlc in changes.items():
        series[D + int(key.removeprefix("d"))] = ohlc
    return series


async def _run(
    series: Sequence[Ohlc | None],
    tmp_path: Path,
    *,
    exit_: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    backtest: dict[str, Any] | None = None,
) -> SimulationResult:
    loaded = daily_loaded(
        whitelist=["AAA"], backtest=backtest or PERIOD, exit_=exit_, execution=execution
    )
    data = dataset({"AAA": series}, DAYS, tmp_path)
    return await simulate(loaded.config, data, starting_cash=CASH)


def _buy(price: str) -> Decimal:
    return (Decimal(price) * (1 + SLIP)).quantize(TICK, rounding=ROUND_CEILING)


def _sell(price: str | Decimal) -> Decimal:
    return (Decimal(price) * (1 - SLIP)).quantize(TICK, rounding=ROUND_FLOOR)


# --------------------------------------------------------------------------- entries / exits


async def test_entry_fills_at_the_next_session_open_with_adverse_slippage(tmp_path: Path) -> None:
    result = await _run(_series(d1=Ohlc("99.5", "100.5", "98.5", "99.5")), tmp_path)
    trade = result.trades[0]
    assert trade.trade_id == signal_id_for(
        strategy_version="fixture-daily-0.1.0",
        symbol="AAA",
        timeframe=Timeframe.DAY_1,
        bar_start_utc=SESSIONS[D].open_utc,
    )
    assert trade.entry_ref == Decimal("99.2")  # close of the signal session
    assert trade.stop_price == Decimal("93.20")
    assert trade.entry_filled_at_utc == SESSIONS[D + 1].open_utc
    assert trade.entry_price == _buy("99.5")
    assert trade.entry_slippage == trade.entry_price - trade.entry_ref
    assert result.counters.minute_bars == 0
    assert result.counters.closed_bars == len(DAYS) - 20  # one daily bar per session


async def test_rule_exit_fills_at_the_next_open_after_overnight_holds(tmp_path: Path) -> None:
    series = _series(d3=Ohlc("99.5", "100.5", "98.5", "100.4"))  # IBS 0.95 -> exit
    result = await _run(series, tmp_path)
    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.SIGNAL_REVERSAL
    assert trade.exit_filled_at_utc == SESSIONS[D + 4].open_utc
    assert trade.exit_price == _sell("100")
    # Held overnight three times with its GTC legs alive: one exposed sample per close.
    exposed = [p for p in result.equity_curve if p.exposure > 0]
    assert [p.timestamp_utc.date() for p in exposed] == [DAYS[D + 1], DAYS[D + 2], DAYS[D + 3]]
    assert all(p.timestamp_utc > SESSIONS[D + 1].close_utc for p in exposed[1:])
    assert result.counters.legs_expired_with_position == 0
    assert result.counters.sessions_with_position_after_close == 0  # intraday-only counter
    assert result.counters.open_trades_at_end == 0


@pytest.mark.parametrize(("limit", "exit_index"), [(5, 6), (2, 3)])
async def test_time_stop_counts_sessions(tmp_path: Path, limit: int, exit_index: int) -> None:
    result = await _run(_series(), tmp_path, exit_={"time_stop_bars": limit})
    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.TIME_STOP
    # Entry at the open of D+1; ``limit`` closes later the exit fills at the next open.
    assert trade.entry_filled_at_utc == SESSIONS[D + 1].open_utc
    assert trade.exit_filled_at_utc == SESSIONS[D + exit_index].open_utc
    assert trade.exit_price == _sell("100")


async def test_gap_through_the_stop_fills_at_the_open(tmp_path: Path) -> None:
    result = await _run(_series(d2=Ohlc("80", "81", "79", "80.5")), tmp_path)
    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_filled_at_utc == SESSIONS[D + 2].open_utc
    assert trade.exit_price == _sell("80")  # open x (1 - slippage), not the stop


async def test_intraday_stop_touch_fills_at_the_stop_by_the_close(tmp_path: Path) -> None:
    result = await _run(_series(d2=Ohlc("99", "99.5", "93", "95")), tmp_path)
    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_filled_at_utc == SESSIONS[D + 2].close_utc  # intrabar: bar end
    assert trade.exit_price == _sell(Decimal("93.20"))


async def test_stop_first_when_one_daily_bar_spans_stop_and_take_profit(tmp_path: Path) -> None:
    result = await _run(_series(d2=Ohlc("100", "170", "90", "100")), tmp_path)
    trade = result.trades[0]
    assert trade.take_profit_price < Decimal(170)
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_price == _sell(Decimal("93.20"))


async def test_limit_entry_lives_for_one_session(tmp_path: Path) -> None:
    execution = {"entry_order_type": "limit", "limit_entry_offset_bps": Decimal(0)}
    filled = await _run(_series(), tmp_path, execution=execution)
    # Limit 99.2 (the signal close): the neutral D+1 opens at 100 and trades down to 99.
    assert filled.trades[0].entry_price == Decimal("99.20")
    assert filled.trades[0].entry_filled_at_utc == SESSIONS[D + 1].close_utc  # intrabar
    above = _series(d1=Ohlc("100", "101", "99.5", "100"))  # never trades at 99.2
    result = await _run(above, tmp_path, execution=execution)
    assert result.trades == ()
    assert result.counters.entries_submitted == 1
    assert result.counters.entries_canceled == 1
    assert result.counters.open_trades_at_end == 0


# --------------------------------------------------------------------------- no lookahead


async def test_bars_after_the_decision_session_do_not_change_the_decision(
    tmp_path: Path,
) -> None:
    base = (await _run(_series(), tmp_path)).trades[0]
    future = _series(**{f"d{k}": Ohlc("100", "140", "60", "120") for k in range(2, 30)})
    future[D + 1] = Ohlc("100", "130", "70", "125")  # same open, different H/L/C
    changed = (await _run(future, tmp_path)).trades[0]
    decision = ("trade_id", "qty", "entry_ref", "stop_price", "take_profit_price")
    assert {k: getattr(changed, k) for k in decision} == {k: getattr(base, k) for k in decision}
    assert changed.entry_price == base.entry_price
    assert changed.entry_filled_at_utc == base.entry_filled_at_utc
    # Sanity: the decision bar itself does matter.
    no_signal = await _run(_series(d0=flat_day()), tmp_path)
    assert all(t.entry_filled_at_utc != SESSIONS[D + 1].open_utc for t in no_signal.trades)


class _RecordingFeed(HistoricalFeed):
    """``HistoricalFeed`` that records the date ranges of ``get_daily_bars``."""

    def __init__(self, bars: Sequence[Bar]) -> None:
        super().__init__(bars)
        self.ranges: list[tuple[date, date]] = []

    async def get_daily_bars(self, symbol: str, start: date, end: date) -> list[Bar]:
        self.ranges.append((start, end))
        return await super().get_daily_bars(symbol, start, end)


async def test_daily_bars_after_the_window_are_never_read(tmp_path: Path) -> None:
    source = dataset({"AAA": _series()}, DAYS, tmp_path)
    all_bars = await source.feed.get_daily_bars("AAA", DAYS[0], DAYS[-1])
    feed = _RecordingFeed(all_bars)
    data = BacktestData(feed=feed, sessions=source.sessions, directory=tmp_path)
    end = DAYS[D + 3]
    loaded = daily_loaded(whitelist=["AAA"], backtest={**PERIOD, "end_date": end})
    result = await simulate(loaded.config, data, starting_cash=CASH)
    assert feed.ranges
    assert all(last <= end for _, last in feed.ranges)
    assert result.sessions[-1].session_date == end
    assert result.counters.open_trades_at_end == 1  # held past the window end


# --------------------------------------------------------------------------- data gaps


async def test_missing_daily_bar_is_an_elapsed_session(tmp_path: Path) -> None:
    result = await _run(_series(d3=None), tmp_path)
    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.TIME_STOP
    assert trade.exit_filled_at_utc == SESSIONS[D + 6].open_utc  # the gap counted


async def test_exit_waits_for_a_session_with_a_bar(tmp_path: Path) -> None:
    result = await _run(_series(d6=None), tmp_path)
    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.TIME_STOP
    assert trade.exit_filled_at_utc == SESSIONS[D + 7].open_utc  # no bar at D+6


async def test_entry_signal_on_the_last_stored_session_is_rejected(tmp_path: Path) -> None:
    series = _series()
    series[-1] = SIGNAL_DAY
    result = await _run(series, tmp_path)
    assert result.counters.rejections == {"SESSION_UNKNOWN": 1}
    assert len(result.trades) == 1


async def test_daily_requires_swing(tmp_path: Path) -> None:
    loaded = daily_loaded(whitelist=["AAA"], backtest=PERIOD)
    config = loaded.config.model_copy(
        update={
            "strategy": loaded.config.strategy.model_copy(
                update={"holding_mode": HoldingMode.INTRADAY, "flatten_minutes_before_close": 5}
            )
        }
    )
    with pytest.raises(NonRetryableError) as info:
        await simulate(config, dataset({"AAA": _series()}, DAYS, tmp_path), starting_cash=CASH)
    assert info.value.code == "UNSUPPORTED_TIMEFRAME"


async def test_window_warm_up_gives_the_continuous_run_decisions(tmp_path: Path) -> None:
    """A window starting mid-run decides like the continuous run (full warm-up window)."""
    days = weekdays(date(2025, 1, 6), 120)
    bars = {"AAA": random_walk(11, len(days)), "BBB": random_walk(12, len(days), start="50")}
    data = dataset(bars, days, tmp_path)
    period = {"start_date": days[25], "end_date": days[-1], "out_of_sample_start": days[90]}
    config = daily_loaded(backtest=period).config
    full = await simulate(config, data, starting_cash=CASH)
    exposure = {p.timestamp_utc.date(): p.exposure for p in full.equity_curve}
    entries = {t.entry_filled_at_utc.date() for t in full.trades}
    # The first session after a flat close that does not fill an entry decided before it.
    cut = next(
        day
        for previous, day in pairwise(days[60:])
        if exposure[previous] == 0 and day not in entries
    )
    window = await simulate(config, data, starting_cash=CASH, window=(cut, days[-1]))
    signals = [t.trade_id for t in full.trades if t.entry_filled_at_utc.date() > cut]
    assert signals
    assert [t.trade_id for t in window.trades if t.entry_filled_at_utc.date() > cut] == signals


# --------------------------------------------------------------------------- reports


def _calendar_days(start: date, end: date) -> list[date]:
    count = (end - start).days + 1
    return [d for d in (start + timedelta(days=i) for i in range(count)) if d.weekday() < 5]


@pytest.fixture(scope="module")
def daily_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory dataset matching the fixture period (2025-03-03..2025-06-30) + warm-up."""
    days = _calendar_days(date(2025, 1, 6), date(2025, 6, 30))
    bars = {"AAA": random_walk(21, len(days)), "BBB": random_walk(22, len(days), start="50")}
    return write_dataset(tmp_path_factory.mktemp("daily"), bars, days)


@pytest.fixture(scope="module")
def daily_report(daily_dir: Path) -> BacktestReport:
    return run_backtest(daily_loaded(), daily_dir, starting_cash=CASH, workers=1)


def test_daily_report_has_trades_and_every_section(daily_report: BacktestReport) -> None:
    report = daily_report
    assert report.primary_timeframe == "1Day"
    assert report.holding_mode == "swing"
    assert report.full.trades >= 5
    assert {t.symbol for t in report.trades} == {"AAA", "BBB"}
    for trade in report.trades:
        assert trade.exit_filled_at_utc > trade.entry_filled_at_utc
        entry: datetime = trade.entry_filled_at_utc
        assert (entry.hour, entry.minute) in ((13, 30), (14, 30))  # a session open
    assert len(report.slippage_sensitivity) == 2


@pytest.mark.parametrize("workers", [2, 3])
def test_daily_report_does_not_depend_on_workers(
    daily_report: BacktestReport, daily_dir: Path, workers: int
) -> None:
    parallel = run_backtest(daily_loaded(), daily_dir, starting_cash=CASH, workers=workers)
    assert parallel.model_dump_json(indent=2) == daily_report.model_dump_json(indent=2)


def test_daily_research_mode_does_not_depend_on_workers(daily_dir: Path) -> None:
    sequential = run_research(daily_loaded(), daily_dir, starting_cash=CASH, workers=1)
    parallel = run_research(daily_loaded(), daily_dir, starting_cash=CASH, workers=3)
    assert parallel.model_dump_json() == sequential.model_dump_json()
    assert any(segment.metrics.trades for segment in sequential.segments)


@pytest.mark.parametrize("mode", ["official", "segmented"])
def test_daily_cli_runs_both_modes(daily_dir: Path, tmp_path: Path, mode: str) -> None:
    args = ["--config", str(DAILY_CONFIG), "--data", str(daily_dir), "--starting-cash", "100000"]
    one, three = tmp_path / "w1.json", tmp_path / "w3.json"
    assert cli_main([*args, "--mode", mode, "--workers", "1", "--out", str(one)]) == 0
    assert cli_main([*args, "--mode", mode, "--workers", "3", "--out", str(three)]) == 0
    assert one.read_bytes() == three.read_bytes()
    assert json.loads(one.read_text(encoding="utf-8"))["primary_timeframe"] == "1Day"
