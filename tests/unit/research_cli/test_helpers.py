"""Helpers of the research CLI: break-even cost, session marks, halts override, selection,
git status, dev windows and the trial count."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from adapters.simulation.static_calendar import build_regular_sessions
from app.config import AppConfig, load_config
from backtest.registry import BacktestResearchError, TrialRecord, load_hypotheses
from backtest.report import EquityPoint
from backtest.research_cli import (
    break_even_cost,
    cost_metric_name,
    dev_windows,
    halts_disabled,
    load_protocol,
    read_git_status,
    select_hypotheses,
    session_marks,
    trials_after,
)
from backtest.stats import session_close_equity
from domain.models import SessionDay

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = load_protocol(ROOT / "research" / "protocol.yaml")
CONFIG = load_config(ROOT / "research" / "configs" / "mr_a1_ibs.yaml").config


# --------------------------------------------------------------------------- break-even


def test_break_even_interpolates_the_sign_change() -> None:
    result = break_even_cost([(5.0, 0.10), (10.0, 0.02), (15.0, -0.06)])
    assert result.method == "interpolated"
    assert result.cost_bps_per_side == pytest.approx(11.25)


def test_break_even_at_a_measured_zero() -> None:
    assert break_even_cost([(5.0, 0.1), (10.0, 0.0), (15.0, -0.1)]).cost_bps_per_side == 10.0
    assert break_even_cost([(5.0, -0.1), (10.0, 0.0), (15.0, 0.1)]).cost_bps_per_side == 10.0


def test_break_even_extrapolates_beyond_the_range() -> None:
    above = break_even_cost([(5.0, 0.30), (10.0, 0.25), (15.0, 0.20)])
    assert above.method == "extrapolated"
    assert above.cost_bps_per_side == pytest.approx(35.0)
    below = break_even_cost([(5.0, -0.05), (10.0, -0.15), (15.0, -0.25)])
    assert below.method == "extrapolated"
    assert below.cost_bps_per_side == pytest.approx(2.5)


@pytest.mark.parametrize(
    "points",
    [
        [(5.0, 0.1), (10.0, None), (15.0, None)],
        [(5.0, 0.1), (10.0, 0.2), (15.0, 0.3)],  # expectancy does not fall with cost
        [(5.0, None), (10.0, None), (15.0, None)],
    ],
)
def test_break_even_undefined(points: Sequence[tuple[float, float | None]]) -> None:
    result = break_even_cost(points)
    assert (result.method, result.cost_bps_per_side) == ("undefined", None)


def test_cost_metric_names() -> None:
    assert cost_metric_name(Decimal(2)) == "expectancy_r_at_2x_cost"
    assert cost_metric_name(Decimal("1.5")) == "expectancy_r_at_1.5x_cost"


# --------------------------------------------------------------------------- session marks


def _sessions(count: int) -> list[SessionDay]:
    days = [date(2024, 3, 4) + timedelta(days=i) for i in range(count + 4)]
    return list(build_regular_sessions([d for d in days if d.weekday() < 5][:count]).values())


def test_session_marks_take_the_sample_after_each_close() -> None:
    sessions = _sessions(3)
    curve = [
        EquityPoint(
            timestamp_utc=s.close_utc + timedelta(seconds=11),
            equity=Decimal(100 + 10 * i),
            exposure=Decimal("0.5") if i == 1 else Decimal(0),
        )
        for i, s in enumerate(sessions)
    ]
    equity, flags = session_marks(curve, sessions, Decimal(100))
    assert equity == [Decimal(100), Decimal(100), Decimal(110), Decimal(120)]
    assert flags == [False, True, False]
    # stats.session_close_equity cuts at close_utc: it lags one session on runner curves.
    assert session_close_equity(curve, sessions, Decimal(100))[2] == Decimal(100)


def test_session_marks_carry_forward_without_samples() -> None:
    sessions = _sessions(3)
    curve = [
        EquityPoint(timestamp_utc=sessions[0].close_utc, equity=Decimal(90), exposure=Decimal(0))
    ]
    equity, flags = session_marks(curve, sessions, Decimal(100))
    assert equity == [Decimal(100), Decimal(90), Decimal(90), Decimal(90)]
    assert flags == [False, False, False]
    assert session_marks([], sessions, Decimal(100))[0] == [Decimal(100)] * 4


# --------------------------------------------------------------------------- halts override


def test_halts_disabled_only_changes_the_loss_limits() -> None:
    research = halts_disabled(CONFIG)
    assert research.risk.max_daily_loss_pct == 1
    assert research.risk.max_weekly_loss_pct == 1
    assert research.risk.max_drawdown_pct == 1
    changed = {"max_daily_loss_pct", "max_weekly_loss_pct", "max_drawdown_pct"}
    original = CONFIG.risk.model_dump()
    assert {k: v for k, v in research.risk.model_dump().items() if k not in changed} == {
        k: v for k, v in original.items() if k not in changed
    }
    assert research.model_dump(exclude={"risk"}) == CONFIG.model_dump(exclude={"risk"})
    # The loaded (official) config is not modified.
    assert CONFIG.risk.max_drawdown_pct == Decimal("0.10")
    assert CONFIG.risk.max_daily_loss_pct == Decimal("0.02")


# --------------------------------------------------------------------------- selection


def test_select_hypotheses() -> None:
    registered = load_hypotheses(ROOT / "research" / "hypotheses")
    assert [h.id for h in select_hypotheses(registered, "all")] == sorted(h.id for h in registered)
    chosen = select_hypotheses(registered, "tf_t1_sma200, mr_a1_ibs")
    assert [h.id for h in chosen] == ["mr_a1_ibs", "tf_t1_sma200"]
    for selection, code in (
        ("nope", "HYPOTHESIS_UNKNOWN"),
        ("mr_a1_ibs,mr_a1_ibs", "HYPOTHESIS_SELECTION_INVALID"),
        (" , ", "HYPOTHESIS_SELECTION_INVALID"),
    ):
        with pytest.raises(BacktestResearchError) as caught:
            select_hypotheses(registered, selection)
        assert caught.value.code == code
    with pytest.raises(BacktestResearchError):
        select_hypotheses((), "all")


# --------------------------------------------------------------------------- git


def test_git_status_reads_commit_and_dirty_flag() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(args: Sequence[str]) -> str:
        calls.append(tuple(args))
        return "abc123\n" if args[0] == "rev-parse" else " M backtest/x.py\n"

    status = read_git_status(runner)
    assert (status.commit, status.dirty) == ("abc123", True)
    assert calls == [("rev-parse", "HEAD"), ("status", "--porcelain")]
    clean = read_git_status(lambda args: "abc123\n" if args[0] == "rev-parse" else "")
    assert (clean.commit, clean.dirty) == ("abc123", False)


@pytest.mark.parametrize("error", [OSError("no git"), subprocess.CalledProcessError(128, ["git"])])
def test_git_failure_is_unknown_and_dirty(error: Exception) -> None:
    def runner(args: Sequence[str]) -> str:
        raise error

    status = read_git_status(runner)
    assert (status.commit, status.dirty) == (None, True)


# --------------------------------------------------------------------------- windows


def _with_backtest(**changes: object) -> AppConfig:
    return CONFIG.model_copy(update={"backtest": CONFIG.backtest.model_copy(update=changes)})


def test_dev_windows_of_a_research_config() -> None:
    window, tests = dev_windows(CONFIG, PROTOCOL)
    assert (window.start, window.end) == (date(2016, 11, 1), date(2022, 12, 31))
    assert len(tests) == 21
    assert (tests[-1].start, tests[-1].end) == (date(2022, 11, 1), date(2022, 12, 31))


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"end_date": date(2023, 1, 1)}, "LOCKBOX_TOUCHED"),
        ({"end_date": date(2026, 6, 30)}, "LOCKBOX_TOUCHED"),
        ({"start_date": date(2016, 1, 4)}, "DEV_WINDOW_MISMATCH"),
        ({"end_date": date(2022, 6, 30)}, "DEV_WINDOW_MISMATCH"),
        ({"walk_forward_test_months": 6}, "DEV_WINDOW_MISMATCH"),
    ],
)
def test_dev_refuses_windows_outside_the_dev_window(changes: dict[str, object], code: str) -> None:
    with pytest.raises(BacktestResearchError) as caught:
        dev_windows(_with_backtest(**changes), PROTOCOL)
    assert caught.value.code == code


# --------------------------------------------------------------------------- trial count


def _record(hypothesis_id: str, config_hash: str, trial_id: str) -> TrialRecord:
    return TrialRecord.model_validate(
        {
            "trial_id": trial_id,
            "hypothesis_id": hypothesis_id,
            "config_hash": config_hash,
            "data_fingerprint": "d",
            "window": {"start": date(2016, 11, 1), "end": date(2022, 12, 31)},
            "mode": "dev",
            "git_commit": "abc",
            "git_dirty": False,
            "created_at_utc": datetime(2026, 10, 2, tzinfo=UTC),
        }
    )


def test_trials_after_counts_distinct_pairs_and_v1() -> None:
    existing = [_record("a", "h1", "t1"), _record("a", "h1", "t2"), _record("b", "h2", "t3")]
    assert trials_after(existing, (), 1) == 3
    assert trials_after(existing, [("a", "h1")], 1) == 3  # a rerun is not a new trial
    assert trials_after(existing, [("a", "h9"), ("c", "h3")], 1) == 5
