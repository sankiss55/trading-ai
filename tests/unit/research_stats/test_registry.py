"""Hypothesis registry, trial log and lockbox guard (:mod:`backtest.registry`).

Every trial log here lives in ``tmp_path``: tests never create ``research/trials.jsonl``.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from backtest.registry import (
    TRIAL_BUDGET,
    V1_TRIAL_COUNT,
    BacktestResearchError,
    GitStatus,
    Hypothesis,
    TrialRecord,
    TrialWindow,
    append_trial,
    assert_lockbox_allowed,
    count_trials,
    load_hypotheses,
    load_hypothesis,
    read_trials,
)
from domain.errors import NonRetryableError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HYPOTHESES_DIR = PROJECT_ROOT / "research" / "hypotheses"
REGISTERED_IDS = {
    "mr_a1_ibs",
    "mr_a2_ibs_sma200",
    "mr_b1_rsi2_lt5",
    "mr_b2_rsi2_lt10",
    "mr_b3_rsi2_lt15",
    "tf_t1_sma200",
    "tf_t2_sma210",
}
CLEAN = GitStatus(commit="0123abc", dirty=False)


def _record(
    trial_id: str = "t1",
    *,
    hypothesis_id: str = "mr_a1_ibs",
    config_hash: str = "hash-a",
    mode: str = "dev",
    metrics: dict[str, float | int | str | bool | None] | None = None,
) -> TrialRecord:
    return TrialRecord.model_validate(
        {
            "trial_id": trial_id,
            "hypothesis_id": hypothesis_id,
            "config_hash": config_hash,
            "data_fingerprint": "data-1",
            "window": {"start": date(2016, 1, 1), "end": date(2022, 12, 31)},
            "mode": mode,
            "git_commit": "0123abc",
            "git_dirty": False,
            "created_at_utc": datetime(2026, 10, 2, 12, 0, tzinfo=UTC),
            "metrics": metrics or {"expectancy_r": 0.12, "trades": 140},
        }
    )


# --------------------------------------------------------------------------- hypotheses


def test_registered_hypotheses_parse_with_distinct_ids() -> None:
    hypotheses = load_hypotheses(HYPOTHESES_DIR)
    ids = [h.id for h in hypotheses]
    assert set(ids) == REGISTERED_IDS
    assert len(ids) == len(set(ids))
    assert len(ids) + V1_TRIAL_COUNT <= TRIAL_BUDGET


@pytest.mark.parametrize("hypothesis_id", sorted(REGISTERED_IDS))
def test_registered_hypothesis_content(hypothesis_id: str) -> None:
    hypothesis = load_hypothesis(HYPOTHESES_DIR / f"{hypothesis_id}.yaml")
    assert hypothesis.status == "registered"
    assert hypothesis.created_at == date(2026, 10, 2)
    assert hypothesis.config_path == f"research/configs/{hypothesis_id}.yaml"
    assert hypothesis.symbols == ("SPY", "QQQ", "IWM")
    assert hypothesis.costs.base_bps_per_side == Decimal(5)
    assert hypothesis.costs.stress_multipliers == (Decimal(2), Decimal(3))
    assert all(source.url.startswith("https://") for source in hypothesis.sources)
    dev = hypothesis.pass_rule["dev"]
    assert dev["expectancy_r"] == {"op": ">=", "value": 0.10}
    assert dev["t_stat"] == {"op": ">=", "value": 2.0}
    assert dev["deflated_sharpe_ratio"] == {"op": ">=", "value": 0.95}
    assert dev["pbo"] == {"op": "<", "value": 0.5}
    assert len(dev["vs_buy_and_hold"]["any_of"]) == 2
    lockbox = hypothesis.pass_rule["lockbox"]
    assert lockbox["profit_factor"] == {"op": ">", "value": 1.1}
    assert lockbox["max_drawdown"] == {"op": "<=", "value": 0.15}


def test_hypothesis_file_must_match_its_id(tmp_path: Path) -> None:
    source = (HYPOTHESES_DIR / "mr_a1_ibs.yaml").read_text(encoding="utf-8")
    renamed = tmp_path / "other_id.yaml"
    renamed.write_text(source, encoding="utf-8")
    with pytest.raises(BacktestResearchError) as excinfo:
        load_hypothesis(renamed)
    assert excinfo.value.code == "HYPOTHESIS_INVALID"


@pytest.mark.parametrize(
    "text",
    [
        "id: [unclosed",
        "id: x\nfamily: f\n",  # missing required fields
        "- just\n- a list\n",
    ],
)
def test_invalid_hypothesis_files_are_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "x.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(BacktestResearchError) as excinfo:
        load_hypothesis(path)
    assert isinstance(excinfo.value, NonRetryableError)


def test_hypothesis_rejects_non_http_sources() -> None:
    data = load_hypothesis(HYPOTHESES_DIR / "mr_a1_ibs.yaml").model_dump()
    data["sources"] = [{"citation": "x", "url": "ftp://example.org/paper.pdf"}]
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(data)


# --------------------------------------------------------------------------- trial log


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trials.jsonl"
    assert read_trials(path) == ()
    first = _record("t1", metrics={"profit_factor": math.inf, "ok": True, "note": None})
    second = _record("t2", config_hash="hash-b")
    append_trial(path, first)
    line_before = path.read_text(encoding="utf-8")
    append_trial(path, second)
    assert path.read_text(encoding="utf-8").startswith(line_before)  # append-only
    assert read_trials(path) == (first, second)
    assert read_trials(path)[0].metrics["profit_factor"] == math.inf
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_duplicate_trial_id_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    append_trial(path, _record("t1"))
    with pytest.raises(BacktestResearchError) as excinfo:
        append_trial(path, _record("t1", config_hash="other"))
    assert excinfo.value.code == "TRIAL_DUPLICATE"
    assert len(read_trials(path)) == 1


def test_invalid_trial_line_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    append_trial(path, _record("t1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n{not json}\n")
    with pytest.raises(BacktestResearchError, match=r":3: invalid trial record"):
        read_trials(path)


def test_trial_record_validation() -> None:
    with pytest.raises(ValidationError):
        TrialWindow(start=date(2022, 1, 2), end=date(2022, 1, 1))
    data = _record().model_dump()
    data["created_at_utc"] = datetime(2026, 10, 2, 12, 0)  # noqa: DTZ001 - naive on purpose
    with pytest.raises(ValidationError):
        TrialRecord.model_validate(data)
    data = _record().model_dump()
    data["mode"] = "production"
    with pytest.raises(ValidationError):
        TrialRecord.model_validate(data)


def test_count_trials(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    assert count_trials(path) == V1_TRIAL_COUNT == 1
    assert count_trials(path, include_v1=False) == 0
    append_trial(path, _record("t1"))
    append_trial(path, _record("t2"))  # rerun of the same config: not a new trial
    append_trial(path, _record("t3", config_hash="hash-a2"))  # changed config: new trial
    append_trial(path, _record("t4", hypothesis_id="mr_b1_rsi2_lt5", config_hash="hash-b"))
    append_trial(path, _record("t5", config_hash="hash-a", mode="lockbox"))
    assert count_trials(path) == 4
    assert count_trials(path, include_v1=False) == 3


# --------------------------------------------------------------------------- lockbox guard


def _hypothesis(**changes: object) -> Hypothesis:
    base = load_hypothesis(HYPOTHESES_DIR / "mr_a2_ibs_sma200.yaml")
    return base.model_copy(update=changes)


def test_lockbox_allowed_when_every_condition_holds() -> None:
    trials = [
        _record("t1", hypothesis_id="mr_a2_ibs_sma200"),  # dev trial: fine
        _record("t2", hypothesis_id="mr_b1_rsi2_lt5", mode="lockbox"),  # other hypothesis
    ]
    assert_lockbox_allowed(_hypothesis(), trials, CLEAN)


@pytest.mark.parametrize(
    ("hypothesis", "trials", "git", "code"),
    [
        (None, [], CLEAN, "LOCKBOX_NOT_REGISTERED"),
        ("no_rule", [], CLEAN, "LOCKBOX_NO_PASS_RULE"),
        ("empty_lockbox_rule", [], CLEAN, "LOCKBOX_NO_PASS_RULE"),
        ("ok", [], GitStatus(commit=None, dirty=False), "LOCKBOX_NO_COMMIT"),
        ("ok", [], GitStatus(commit="", dirty=False), "LOCKBOX_NO_COMMIT"),
        ("ok", [], GitStatus(commit="0123abc", dirty=True), "LOCKBOX_DIRTY_TREE"),
        ("ok", ["opened"], CLEAN, "LOCKBOX_ALREADY_OPENED"),
    ],
)
def test_lockbox_refusals(
    hypothesis: str | None, trials: list[str], git: GitStatus, code: str
) -> None:
    candidates = {
        "ok": _hypothesis(),
        "no_rule": _hypothesis(pass_rule={}),
        "empty_lockbox_rule": _hypothesis(pass_rule={"dev": {"t_stat": 2}, "lockbox": {}}),
    }
    records = [_record("t9", hypothesis_id="mr_a2_ibs_sma200", mode="lockbox") for _ in trials]
    with pytest.raises(BacktestResearchError) as excinfo:
        assert_lockbox_allowed(None if hypothesis is None else candidates[hypothesis], records, git)
    assert excinfo.value.code == code
    assert isinstance(excinfo.value, NonRetryableError)
