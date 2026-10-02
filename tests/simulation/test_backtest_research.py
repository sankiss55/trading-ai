"""RESEARCH mode (segmented, fresh account per segment) on synthetic data."""

from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from backtest.__main__ import main as cli_main
from backtest.data import BARS_DIR, load_backtest_data
from backtest.research import (
    CONTINUATION_LABEL,
    RESEARCH_BANNER,
    ResearchReport,
    SegmentReport,
    render_research_text,
    research_segments,
    run_research,
)
from backtest.runner import SimulationResult, run_backtest_async, simulate
from domain.errors import NonRetryableError
from domain.models import DataFeed
from tests.simulation.helpers import FIXTURE_CONFIG, fixture_loaded, make_dataset

CASH = Decimal(100000)
DATA_FROM = date(2024, 11, 18)
DATA_TO = date(2025, 1, 24)
PERIOD: dict[str, Any] = {
    "start_date": date(2024, 12, 16),
    "end_date": DATA_TO,
    "out_of_sample_start": date(2025, 1, 13),
    "walk_forward_train_months": 1,
    "walk_forward_test_months": 1,
}
TIGHT_DRAWDOWN: dict[str, Any] = {"max_drawdown_pct": Decimal("0.002")}
"""TEST FIXTURE value: a 0.2 % drawdown halts every account after its first losses."""
SCALE = Decimal("1.5")
CENT = Decimal("0.01")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("research"), start=DATA_FROM, end=DATA_TO)


@pytest.fixture(scope="module")
def halted(dataset: Path) -> ResearchReport:
    loaded = fixture_loaded(backtest=PERIOD, risk=TIGHT_DRAWDOWN)
    return run_research(loaded, dataset, starting_cash=CASH, workers=1)


def _segment(report: ResearchReport, label: str) -> SegmentReport:
    return next(s for s in report.segments if s.label == label)


async def _simulate_window(
    dataset: Path, window: tuple[date, date], *, risk: dict[str, Any] | None = None
) -> SimulationResult:
    config = fixture_loaded(backtest=PERIOD, risk=risk).config
    data = load_backtest_data(dataset, feed=DataFeed.IEX)
    return await simulate(config, data, starting_cash=CASH, window=window)


def _scale_bars(directory: Path, keep: Callable[[date], bool]) -> None:
    """Multiply the OHLC of every stored bar whose date fails ``keep`` by ``SCALE``."""
    for path in sorted((directory / BARS_DIR).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        for row in rows[1:]:
            if not keep(date.fromisoformat(row[0][:10])):
                row[1:5] = [str((Decimal(v) * SCALE).quantize(CENT)) for v in row[1:5]]
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)


# --------------------------------------------------------------------------- segments


def test_segments_cover_walk_forward_splits_years_and_sensitivity() -> None:
    segments = research_segments(
        date(2022, 1, 1), date(2026, 6, 30), date(2025, 7, 1), train_months=12, test_months=3
    )
    labels = [s.label for s in segments]
    years = [f"year_{y}" for y in range(2022, 2027)]
    assert labels == [
        *(f"walk_forward_{i}" for i in range(14)),
        "in_sample",
        "out_of_sample",
        *years,
        "out_of_sample_x1.5",
        *(f"{y}_x1.5" for y in years),
        "out_of_sample_x2.0",
        *(f"{y}_x2.0" for y in years),
    ]
    by_label = {s.label: s for s in segments}
    assert (by_label["walk_forward_0"].start, by_label["walk_forward_0"].end) == (
        date(2023, 1, 1),
        date(2023, 3, 31),
    )
    assert by_label["walk_forward_13"].end == date(2026, 6, 30)
    assert by_label["in_sample"].end == date(2025, 6, 30)
    assert (by_label["year_2026"].start, by_label["year_2026"].end) == (
        date(2026, 1, 1),
        date(2026, 6, 30),
    )
    assert by_label["year_2022_x2.0"].slippage_multiplier == Decimal("2.0")


# --------------------------------------------------------------------------- independence


async def test_a_halt_in_one_segment_does_not_affect_the_next(
    halted: ResearchReport, dataset: Path
) -> None:
    official = await run_backtest_async(
        fixture_loaded(backtest=PERIOD, risk=TIGHT_DRAWDOWN), dataset, starting_cash=CASH
    )
    assert official.counters.rejections["MAX_DRAWDOWN"] > 0
    assert official.out_of_sample.trades == 0  # the continuous account stays halted

    in_sample = _segment(halted, "in_sample")
    out_of_sample = _segment(halted, "out_of_sample")
    assert in_sample.halt_rejections["MAX_DRAWDOWN"] > 0
    assert in_sample.drawdown_limit_reached_on is not None
    assert out_of_sample.trades  # a fresh account trades again
    assert out_of_sample.metrics.starting_equity == CASH

    alone = await _simulate_window(
        dataset, (PERIOD["out_of_sample_start"], PERIOD["end_date"]), risk=TIGHT_DRAWDOWN
    )
    assert alone.trades == out_of_sample.trades
    assert alone.counters == out_of_sample.counters


def test_every_segment_starts_from_the_starting_cash(halted: ResearchReport) -> None:
    for segment in halted.segments:
        assert segment.metrics.starting_equity == CASH
        assert segment.counters.open_trades_at_end == 0
        for trade in segment.trades:
            assert segment.metrics.start_date <= trade.entry_filled_at_utc.date()
            assert trade.exit_filled_at_utc.date() <= segment.metrics.end_date


def test_parallel_segments_give_the_sequential_report(
    halted: ResearchReport, dataset: Path
) -> None:
    loaded = fixture_loaded(backtest=PERIOD, risk=TIGHT_DRAWDOWN)
    parallel = run_research(loaded, dataset, starting_cash=CASH, workers=3)
    assert parallel.model_dump_json(indent=2) == halted.model_dump_json(indent=2)


# --------------------------------------------------------------------------- warm-up


async def test_window_warm_up_is_the_official_warm_up(dataset: Path) -> None:
    """Same code path: ``window`` = the configured period gives the official run."""
    official = await _simulate_window(dataset, (PERIOD["start_date"], PERIOD["end_date"]))
    config = fixture_loaded(backtest=PERIOD).config
    data = load_backtest_data(dataset, feed=DataFeed.IEX)
    assert await simulate(config, data, starting_cash=CASH) == official


async def test_segment_ignores_data_after_its_end(dataset: Path, tmp_path: Path) -> None:
    """No lookahead: rewriting every bar after the segment end changes nothing."""
    window = (date(2024, 12, 16), date(2024, 12, 31))
    before = await _simulate_window(dataset, window)
    changed = make_dataset(tmp_path / "future", start=DATA_FROM, end=DATA_TO)
    _scale_bars(changed, keep=lambda day: day <= window[1])
    after = await _simulate_window(changed, window)
    assert before.trades
    assert after == before
    assert after.sessions[0].session_date >= window[0]
    assert after.sessions[-1].session_date <= window[1]


async def test_segment_warms_up_on_the_sessions_before_its_start(
    dataset: Path, tmp_path: Path
) -> None:
    """The stored sessions just before the segment feed the indicators."""
    window = (date(2025, 1, 13), DATA_TO)
    before = await _simulate_window(dataset, window)
    changed = make_dataset(tmp_path / "past", start=DATA_FROM, end=DATA_TO)
    _scale_bars(changed, keep=lambda day: not date(2025, 1, 6) <= day < window[0])
    after = await _simulate_window(changed, window)
    assert after.trades != before.trades or after.counters != before.counters


async def test_invalid_window_is_refused(dataset: Path) -> None:
    with pytest.raises(NonRetryableError) as info:
        await _simulate_window(dataset, (date(2025, 1, 10), date(2025, 1, 9)))
    assert info.value.code == "INVALID_WINDOW"


# --------------------------------------------------------------------------- report


def test_report_is_labelled_research_and_has_every_section(halted: ResearchReport) -> None:
    assert halted.mode == "segmented"
    assert halted.banner == RESEARCH_BANNER
    assert halted.continuation_label == CONTINUATION_LABEL
    assert "fresh-account" in halted.continuation.basis
    labels = [s.label for s in halted.segments]
    assert labels[:4] == ["walk_forward_0", "in_sample", "out_of_sample", "year_2024"]
    assert "year_2025_x2.0" in labels
    assert [a.label for a in halted.aggregates] == [
        "walk_forward_tests",
        "years",
        "years_x1.5",
        "years_x2.0",
    ]
    text = render_research_text(halted)
    assert text.splitlines()[0] == RESEARCH_BANNER
    assert CONTINUATION_LABEL in text
    assert "MAX_DRAWDOWN" in text
    assert "take-profit hit rate" in text
    assert "do not imply future profitability" in text
    parsed = ResearchReport.model_validate_json(halted.model_dump_json())
    assert parsed == halted


def test_aggregates_pool_the_trades_of_their_segments(halted: ResearchReport) -> None:
    years = [s for s in halted.segments if s.kind == "year" and s.slippage_multiplier == 1]
    pooled = next(a for a in halted.aggregates if a.label == "years")
    trades = [t for s in years for t in s.trades]
    assert pooled.segments == tuple(s.label for s in years)
    assert pooled.trades == len(trades) == sum(s.metrics.trades for s in years)
    r_values = [t.result_r for t in trades if t.result_r is not None]
    assert r_values
    expected = (sum(r_values, Decimal(0)) / len(r_values)).quantize(Decimal("0.000001"))
    assert pooled.expectancy_r == expected
    positive = sum(1 for s in years if (s.metrics.expectancy_r or 0) > 0)
    assert pooled.positive_expectancy_segments == positive
    assert pooled.halted_segments == len(years)
    for segment in halted.segments:
        take_profits = segment.exits_by_reason.get("TAKE_PROFIT", 0)
        if segment.trades:
            assert segment.take_profit_rate == (
                Decimal(take_profits) / len(segment.trades)
            ).quantize(Decimal("0.000001"))


# --------------------------------------------------------------------------- CLI


def _short_config(tmp_path: Path) -> Path:
    text = FIXTURE_CONFIG.read_text(encoding="utf-8")
    for key, old, new in (
        ("start_date", "2025-01-02", "2025-01-13"),
        ("end_date", "2025-03-31", "2025-01-17"),
        ("out_of_sample_start", "2025-03-01", "2025-01-15"),
    ):
        assert f"{key}: {old}" in text
        text = text.replace(f"{key}: {old}", f"{key}: {new}")
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_mode_flag(dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--config", str(_short_config(tmp_path)), "--data", str(dataset)]
    args += ["--starting-cash", "100000", "--workers", "1"]
    with pytest.raises(SystemExit) as info:
        cli_main([*args, "--mode", "walk"])
    assert info.value.code == 2
    assert "--mode" in capsys.readouterr().err

    out = tmp_path / "research.json"
    assert cli_main([*args, "--mode", "segmented", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.splitlines()[0] == RESEARCH_BANNER
    parsed = ResearchReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert parsed.mode == "segmented"
    assert {s.label for s in parsed.segments} >= {"in_sample", "out_of_sample", "year_2025"}

    official = tmp_path / "official.json"
    assert cli_main([*args, "--out", str(official)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("BACKTEST REPORT")
    assert "RESEARCH MODE" not in text
    assert '"mode"' not in official.read_text(encoding="utf-8")
