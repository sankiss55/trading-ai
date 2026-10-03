"""``backtest.parity``: Phase 3 signal parity between the backtest and the live data path.

Synthetic daily bars only (seeded random walks, not market data); no network.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from adapters.simulation import HistoricalFeed, SimClock
from app.config import LoadedConfig
from backtest.data import BacktestData, load_backtest_data
from backtest.parity import (
    ReplayBroker,
    compared_sessions,
    main,
    parity_exit_code,
    published_daily_feed,
    run_parity,
)
from domain.errors import NonRetryableError
from domain.models import (
    AccountState,
    Bar,
    DataFeed,
    OrderSide,
    OrderType,
    SimpleOrderRequest,
    TimeInForce,
)
from tests.simulation.daily_helpers import (
    DAILY_CONFIG,
    daily_loaded,
    random_walk,
    weekdays,
    write_dataset,
)

ROOT = Path(__file__).resolve().parents[3]
INTRADAY_CONFIG = ROOT / "tests" / "fixtures" / "config.backtest.yaml"
DAYS = weekdays(date(2025, 2, 3), 70)
SERIES = {"AAA": random_walk(3, 70), "BBB": random_walk(5, 70, start="50")}
COMPARED = 15


def _dataset(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "data", SERIES, DAYS)


def _loaded() -> LoadedConfig:
    return daily_loaded(
        backtest={"start_date": DAYS[30], "end_date": DAYS[-1], "out_of_sample_start": DAYS[50]}
    )


def _data(tmp_path: Path) -> BacktestData:
    return load_backtest_data(_dataset(tmp_path), feed=DataFeed.SIP)


async def test_matching_synthetic_data_is_full_parity(tmp_path: Path) -> None:
    report = await run_parity(_loaded(), _data(tmp_path), sessions=COMPARED)
    assert report.passed, report.mismatches
    assert parity_exit_code(report) == 0
    assert report.sessions == tuple(DAYS[-(COMPARED + 1) : -1])
    assert report.decisions_compared == COMPARED * 2
    assert report.warm_up_sessions == 20
    assert report.window_bars_compared == COMPARED * 2 * 20
    # Positions were held: the replayed execution state was exercised, not only entries.
    assert report.outcome_counts.get("HOLD", 0) + report.outcome_counts.get("EXIT", 0) > 0
    assert report.signals > 0


async def test_a_mutated_bar_is_reported_as_a_mismatch(tmp_path: Path) -> None:
    data = _data(tmp_path)
    target_day = DAYS[-5]
    bars: list[Bar] = []
    for symbol in ("AAA", "BBB"):
        for bar in await data.feed.get_daily_bars(symbol, DAYS[0], DAYS[-1]):
            if symbol == "AAA" and bar.bar_start_utc.date() == target_day:
                assert bar.low is not None
                bar = bar.model_copy(update={"close": bar.low})  # IBS 0: still a sane bar
            bars.append(bar)
    report = await run_parity(_loaded(), data, sessions=COMPARED, market_data=HistoricalFeed(bars))
    assert not report.passed
    assert parity_exit_code(report) == 1
    fields = {(m.session_date, m.symbol, m.field) for m in report.mismatches}
    assert (target_day, "AAA", "bar") in fields
    assert (target_day, "AAA", "window") in fields
    assert all(m.symbol == "AAA" for m in report.mismatches)
    assert all(m.session_date >= target_day for m in report.mismatches)


async def test_a_mutated_warm_up_bar_shows_in_the_first_window(tmp_path: Path) -> None:
    data = _data(tmp_path)
    warm_day = DAYS[-(COMPARED + 1) - 3]
    bars: list[Bar] = []
    for symbol in ("AAA", "BBB"):
        for bar in await data.feed.get_daily_bars(symbol, DAYS[0], DAYS[-1]):
            if symbol == "BBB" and bar.bar_start_utc.date() == warm_day:
                bar = bar.model_copy(update={"volume": bar.volume + 1})
            bars.append(bar)
    report = await run_parity(_loaded(), data, sessions=COMPARED, market_data=HistoricalFeed(bars))
    first = report.sessions[0]
    assert ("BBB", "window") in {
        (m.symbol, m.field) for m in report.mismatches if m.session_date == first
    }
    assert not any(m.field == "bar" for m in report.mismatches)


def test_the_cli_exits_0_on_parity_and_writes_the_json_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "parity.json"
    code = main(
        [
            "--config",
            str(DAILY_CONFIG),
            "--data",
            str(_dataset(tmp_path)),
            "--sessions",
            str(COMPARED),
            "--json",
            str(out),
        ]
    )
    assert code == 0
    assert "Signal parity (dataset): PASS" in capsys.readouterr().out
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["mismatches"] == []
    assert stored["decisions_compared"] == COMPARED * 2


def test_the_cli_refuses_an_intraday_strategy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", str(INTRADAY_CONFIG), "--data", str(_dataset(tmp_path))])
    assert code == 2
    assert "parity refused" in capsys.readouterr().err


def test_the_cli_refuses_a_dataset_with_another_adjustment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _dataset(tmp_path)
    manifest = {"version": 2, "daily_only": True, "feeds": {"1Day": "sip"}, "adjustment": "raw"}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    code = main(["--config", str(DAILY_CONFIG), "--data", str(directory)])
    assert code == 2
    assert "DATASET_ADJUSTMENT_MISMATCH" in capsys.readouterr().err


def test_compared_sessions_leave_out_the_final_stored_session(tmp_path: Path) -> None:
    data = _data(tmp_path)
    assert compared_sessions(data, 3) == data.sessions[-4:-1]
    with pytest.raises(NonRetryableError) as info:
        compared_sessions(data, len(data.sessions))
    assert info.value.code == "NO_SESSIONS"


async def test_the_published_feed_knows_a_bar_only_from_its_session_close(
    tmp_path: Path,
) -> None:
    data = _data(tmp_path)
    session = data.sessions[10]
    clock = SimClock(session.close_utc - timedelta(seconds=1))
    feed = await published_daily_feed(data, ("AAA",), clock)
    day = session.session_date
    assert await feed.get_daily_bars("AAA", day, day) == []
    clock.advance_to(session.close_utc)
    (bar,) = await feed.get_daily_bars("AAA", day, day)
    assert bar.bar_start_utc.date() == day  # the midnight New York label is kept
    assert bar.bar_end_utc == session.close_utc
    later = data.sessions[11].session_date
    assert await feed.get_daily_bars("AAA", later, later) == []  # no lookahead


async def test_the_replay_broker_never_submits() -> None:
    cash = Decimal(1000)
    broker = ReplayBroker(
        AccountState(equity=cash, last_equity=cash, buying_power=cash, status="ACTIVE")
    )
    assert await broker.get_positions() == []
    assert await broker.get_open_orders() == []
    with pytest.raises(NonRetryableError) as info:
        await broker.submit_simple(
            SimpleOrderRequest(
                symbol="AAA",
                qty=1,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id="parity-test",
            )
        )
    assert info.value.code == "SUBMISSION_FORBIDDEN"
    with pytest.raises(NonRetryableError):
        await broker.cancel_order("any")
