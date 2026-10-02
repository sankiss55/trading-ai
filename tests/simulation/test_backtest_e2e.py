"""End-to-end deterministic backtests on synthetic data (sec. 45, 49.5)."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import pytest

from app.config import load_config
from backtest.__main__ import main as cli_main
from backtest.data import BARS_DIR, load_backtest_data
from backtest.report import BacktestReport, render_text
from backtest.runner import BacktestRefusedError, SimulationResult, run_backtest, simulate
from domain.models import DataFeed, ExitReason
from tests.simulation.helpers import FIXTURE_CONFIG, PENDING_CONFIG, fixture_loaded, make_dataset

CASH = Decimal(100000)
SHORT = {
    "start_date": date(2025, 1, 2),
    "end_date": date(2025, 1, 15),
    "out_of_sample_start": date(2025, 1, 10),
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("synthetic"), end=date(2025, 1, 15))


@pytest.fixture(scope="module")
def report(dataset: Path) -> BacktestReport:
    return run_backtest(fixture_loaded(backtest=SHORT), dataset, starting_cash=CASH)


async def _simulate(dataset: Path, *, exit_: dict[str, object] | None = None) -> SimulationResult:
    loaded = fixture_loaded(backtest=SHORT, exit_=exit_)
    data = load_backtest_data(dataset, feed=DataFeed.IEX)
    return await simulate(loaded.config, data, starting_cash=CASH)


def test_report_has_trades_and_every_section(report: BacktestReport) -> None:
    assert report.full.trades >= 1
    assert report.full.trades == len(report.trades)
    assert report.in_sample.trades + report.out_of_sample.trades == report.full.trades
    assert report.in_sample.end_date == date(2025, 1, 9)
    assert [s.multiplier for s in report.slippage_sensitivity] == [Decimal("1.5"), Decimal("2.0")]
    assert [s.slippage_bps for s in report.slippage_sensitivity] == [
        Decimal("7.5"),
        Decimal("10.0"),
    ]
    assert report.continuation.status in ("PASS", "FAIL")
    assert report.counters.open_trades_at_end == 0
    for trade in report.trades:
        assert trade.exit_filled_at_utc >= trade.entry_filled_at_utc
        assert trade.stop_price < trade.entry_ref < trade.take_profit_price
        assert trade.net_pnl == trade.gross_pnl - trade.commissions
        if trade.result_r is not None:
            expected = (trade.exit_price - trade.entry_price) / (
                trade.entry_price - trade.stop_price
            )
            assert trade.result_r == expected.quantize(Decimal("0.0001"))
    text = render_text(report)
    assert "Continuation check" in text
    assert "do not imply future profitability" in text


def test_higher_slippage_never_improves_the_result(report: BacktestReport) -> None:
    returns = [report.full.total_return] + [
        s.metrics.total_return for s in report.slippage_sensitivity
    ]
    assert returns == sorted(returns, reverse=True)


def test_same_seed_gives_identical_report(report: BacktestReport, tmp_path: Path) -> None:
    other = make_dataset(tmp_path / "again", end=date(2025, 1, 15))
    again = run_backtest(fixture_loaded(backtest=SHORT), other, starting_cash=CASH)
    assert again.model_dump_json() == report.model_dump_json()


def test_entries_fill_after_the_signal_bar(report: BacktestReport) -> None:
    """No lookahead: market entries fill at a minute open after the decision (45.2.8)."""
    for trade in report.trades:
        assert trade.entry_filled_at_utc.second == 0
        assert trade.entry_filled_at_utc.minute % 5 == 0  # first minute after a 5Min close
        assert trade.entry_slippage > 0  # bought at open * (1 + slippage)


async def test_intraday_flatten_leaves_no_position_after_close(dataset: Path) -> None:
    """AC-16 analogue: wide exits so positions survive until flatten_at."""
    result = await _simulate(
        dataset,
        exit_={
            "stop_atr_multiplier": Decimal(20),
            "take_profit_r_multiple": Decimal(20),
            "time_stop_bars": None,
            "exit_on_signal_reversal": False,
        },
    )
    assert result.trades
    assert result.counters.sessions_with_position_after_close == 0
    assert result.counters.open_trades_at_end == 0
    eod = [t for t in result.trades if t.exit_reason is ExitReason.END_OF_DAY]
    assert eod
    sessions = {s.session_date: s for s in result.sessions}
    for trade in eod:
        session = sessions[trade.exit_filled_at_utc.date()]
        flatten_at = session.close_utc - timedelta(minutes=10)
        assert flatten_at <= trade.exit_filled_at_utc < session.close_utc


async def test_stop_first_when_one_bar_touches_stop_and_take_profit(
    dataset: Path, tmp_path: Path
) -> None:
    """Sec. 45.2.6: rewrite the entry-fill minute so it spans both legs -> STOP_LOSS."""
    first = (await _simulate(dataset)).trades[0]
    copy_dir = make_dataset(tmp_path / "whipsaw", end=date(2025, 1, 15))
    path = copy_dir / BARS_DIR / "SYNTH_1Min.csv"
    stamp = first.entry_filled_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    for row in rows[1:]:
        if row[0] == stamp:
            row[2] = str(first.take_profit_price + 1)  # high above the take profit
            row[3] = str(first.stop_price - 1)  # low below the stop
            break
    else:  # pragma: no cover - the fill minute always exists
        pytest.fail("entry minute not found")
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)

    data = load_backtest_data(copy_dir, feed=DataFeed.IEX)
    result = await simulate(fixture_loaded(backtest=SHORT).config, data, starting_cash=CASH)
    trade = result.trades[0]
    assert trade.trade_id == first.trade_id
    assert trade.entry_price == first.entry_price  # the past is unchanged
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_filled_at_utc == first.entry_filled_at_utc + timedelta(minutes=1)
    slip = Decimal(5) / Decimal(10000)
    expected = (first.stop_price * (1 - slip)).quantize(Decimal("0.0001"), rounding=ROUND_FLOOR)
    assert trade.exit_price == expected


def test_refuses_to_run_with_pending_owner_decisions(dataset: Path) -> None:
    with pytest.raises(BacktestRefusedError) as info:
        run_backtest(load_config(PENDING_CONFIG), dataset, starting_cash=CASH)
    assert "strategy.entry_rules" in info.value.pending
    assert "risk.risk_per_trade_pct" in info.value.pending
    assert "strategy.entry_rules" in str(info.value)


def test_cli_refuses_pending_config_and_runs_fixture(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--data", str(dataset), "--starting-cash", "100000"]
    assert cli_main(["--config", str(PENDING_CONFIG), *args]) == 2
    assert "OWNER_DECISION" in capsys.readouterr().err
    out = tmp_path / "report.json"
    assert cli_main(["--config", str(FIXTURE_CONFIG), *args, "--out", str(out)]) == 0
    assert "BACKTEST REPORT" in capsys.readouterr().out
    parsed = BacktestReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert parsed.full.trades >= 1
