"""End-to-end research CLI (``python -m backtest.research_cli``) on SYNTHETIC daily data.

Every run uses a temporary research tree: a protocol with synthetic windows (dev
2024-03..2024-12, lockbox 2025-01..2025-06), copies of registered hypotheses whose
``config_path`` points to temporary copies of their research configs (indicator periods
and walk-forward months shortened to fit 18 months of synthetic bars), seeded random-walk
bars for SPY/QQQ/IWM and a temporary trial registry. Nothing here reads the real dataset
or writes ``research/trials.jsonl``. Git is a fake runner and the clock is fixed.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from backtest.__main__ import main as backtest_main
from backtest.registry import TrialRecord, count_trials, read_trials
from backtest.research_cli import DEV_BASE_METRICS, main
from tests.simulation.daily_helpers import random_walk, write_dataset

REPO = Path(__file__).resolve().parents[2]
IDS = ("mr_a1_ibs", "mr_a2_ibs_sma200", "mr_b3_rsi2_lt15", "tf_t1_sma200")
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
COMMIT = "abc123def4567890"
PROTOCOL_CHANGES = (
    ("{start: 2016-11-01, end: 2022-12-31}", "{start: 2024-03-01, end: 2024-12-31}"),
    ("{start: 2023-01-01, end: 2026-06-30}", "{start: 2025-01-01, end: 2025-06-30}"),
    ("embargo_start: 2026-07-01", "embargo_start: 2025-07-01"),
    ("{train_months: 12, test_months: 3}", "{train_months: 2, test_months: 1}"),
    ("resamples: 10000", "resamples: 300"),
    ("sims: 10000", "sims: 300"),
    ("sims: 1000", "sims: 50"),
    ("blocks: 16", "blocks: 4"),
)
CONFIG_CHANGES = (
    ("history_warmup_bars: 250", "history_warmup_bars: 30"),
    ("sma_long_period: 200", "sma_long_period: 20"),
    ("sma_long_period: 210", "sma_long_period: 21"),
    ("start_date: 2016-11-01", "start_date: 2024-03-01"),
    ("end_date: 2022-12-31", "end_date: 2024-12-31"),
    ("out_of_sample_start: 2017-11-01", "out_of_sample_start: 2024-05-01"),
    ("walk_forward_train_months: 12", "walk_forward_train_months: 2"),
    ("walk_forward_test_months: 3", "walk_forward_test_months: 1"),
    # Liquidity filter (check library, sec. 12): synthetic volumes are 0.5-2M shares/day.
    ("min_avg_daily_volume: 5000000", "min_avg_daily_volume: 100000"),
)


def clean_git(args: Sequence[str]) -> str:
    return COMMIT + "\n" if args[0] == "rev-parse" else ""


def dirty_git(args: Sequence[str]) -> str:
    return COMMIT + "\n" if args[0] == "rev-parse" else " M backtest/research_cli.py\n"


def broken_git(args: Sequence[str]) -> str:
    raise OSError("git is not installed")


def _replace(text: str, changes: Sequence[tuple[str, str]], *, strict: bool) -> str:
    for old, new in changes:
        assert not strict or old in text, old
        text = text.replace(old, new)
    return text


@dataclass(frozen=True)
class Tree:
    """A temporary research tree."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    def common(
        self, registry: Path, hypotheses: str = "hypotheses", protocol: str = "protocol.yaml"
    ) -> list[str]:
        return [
            "--protocol",
            str(self.root / protocol),
            "--hypotheses-dir",
            str(self.root / hypotheses),
            "--registry",
            str(registry),
        ]

    def dev(
        self,
        registry: Path,
        *,
        workers: int = 1,
        out: Path | None = None,
        extra: Sequence[str] = (),
        git: Callable[[Sequence[str]], str] = clean_git,
        hypotheses: str = "hypotheses",
        protocol: str = "protocol.yaml",
    ) -> int:
        args = [
            "dev",
            *self.common(registry, hypotheses, protocol),
            "--data",
            str(self.data),
            "--starting-cash",
            "100000",
            "--workers",
            str(workers),
            *extra,
        ]
        if out is not None:
            args += ["--out", str(out)]
        return main(args, git_runner=git, now=lambda: NOW)

    def lockbox(
        self,
        registry: Path,
        hypothesis: str,
        *,
        out: Path | None = None,
        git: Callable[[Sequence[str]], str] = clean_git,
        hypotheses: str = "hypotheses",
        workers: int = 1,
    ) -> int:
        args = [
            "lockbox",
            *self.common(registry, hypotheses),
            "--hypothesis",
            hypothesis,
            "--data",
            str(self.data),
            "--starting-cash",
            "100000",
            "--workers",
            str(workers),
        ]
        if out is not None:
            args += ["--out", str(out)]
        return main(args, git_runner=git, now=lambda: LATER)


def build_tree(
    root: Path,
    *,
    ids: Sequence[str] = IDS,
    config_changes: Sequence[tuple[str, str]] = (),
    hypotheses: str = "hypotheses",
    drop_lockbox_rule: bool = False,
) -> Tree:
    """Synthetic dataset + protocol + hypotheses + configs under ``root``."""
    if not (root / "data").exists():
        days: list[date] = []
        day = date(2024, 1, 2)
        while day <= date(2025, 6, 30):
            if day.weekday() < 5:
                days.append(day)
            day += timedelta(days=1)
        write_dataset(
            root / "data",
            {
                "SPY": random_walk(1, len(days), start="400"),
                "QQQ": random_walk(2, len(days), start="350"),
                "IWM": random_walk(3, len(days), start="200"),
            },
            days,
        )
        protocol = (REPO / "research" / "protocol.yaml").read_text(encoding="utf-8")
        (root / "protocol.yaml").write_text(
            _replace(protocol, PROTOCOL_CHANGES, strict=True), encoding="utf-8"
        )
    configs = root / f"configs_{hypotheses}"
    configs.mkdir(exist_ok=True)
    (root / hypotheses).mkdir(exist_ok=True)
    for hypothesis_id in ids:
        text = (REPO / "research" / "configs" / f"{hypothesis_id}.yaml").read_text(encoding="utf-8")
        text = _replace(text, CONFIG_CHANGES, strict=False)
        text = _replace(text, config_changes, strict=True)
        config = configs / f"{hypothesis_id}.yaml"
        config.write_text(text, encoding="utf-8")
        raw: dict[str, Any] = yaml.safe_load(
            (REPO / "research" / "hypotheses" / f"{hypothesis_id}.yaml").read_text(encoding="utf-8")
        )
        raw["config_path"] = str(config)
        if drop_lockbox_rule:
            del raw["pass_rule"]["lockbox"]
        (root / hypotheses / f"{hypothesis_id}.yaml").write_text(
            yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
        )
    return Tree(root)


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Tree:
    return build_tree(tmp_path_factory.mktemp("research"))


@dataclass(frozen=True)
class DevRun:
    registry: Path
    out: Path

    def report(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(
            (self.out / "dev_report.json").read_text(encoding="utf-8")
        )
        return loaded

    def text(self) -> str:
        return (self.out / "dev_report.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dev_run(tree: Tree, tmp_path_factory: pytest.TempPathFactory) -> DevRun:
    base = tmp_path_factory.mktemp("dev_w1")
    registry, out = base / "trials.jsonl", base / "out"
    assert tree.dev(registry, workers=1, out=out) == 0
    return DevRun(registry=registry, out=out)


# --------------------------------------------------------------------------- dev


def test_dev_appends_one_trial_per_hypothesis_with_git_fields(dev_run: DevRun) -> None:
    records = read_trials(dev_run.registry)
    assert [r.hypothesis_id for r in records] == sorted(IDS)
    for record in records:
        assert record.mode == "dev"
        assert (record.git_commit, record.git_dirty) == (COMMIT, False)
        assert record.created_at_utc == NOW
        assert (record.window.start, record.window.end) == (date(2024, 3, 1), date(2024, 12, 31))
        assert (
            record.trial_id
            == f"{record.hypothesis_id}-dev-20261002T120000Z-{record.config_hash[:12]}"
        )
        assert set(DEV_BASE_METRICS) <= set(record.metrics)
        assert {"expectancy_r_at_2x_cost", "expectancy_r_at_3x_cost"} <= set(record.metrics)
        assert record.metrics["n_trials"] == len(IDS) + 1
        assert isinstance(record.metrics["gate_passed"], bool)
    assert len({r.data_fingerprint for r in records}) == 1


def test_dev_report_has_every_section(dev_run: DevRun) -> None:
    report = dev_run.report()
    assert report["mode"] == "dev"
    assert report["batch"]["n_trials"] == len(IDS) + 1
    assert report["batch"]["pbo"]["n_configs"] == len(IDS)
    assert set(report["batch"]["pbo_by_family"]) == {"MR-A (IBS)"}
    assert report["batch"]["trials_sr_variance"] > 0
    for result in report["hypotheses"]:
        assert result["walk_forward"]["windows"]
        assert result["walk_forward"]["bootstrap"]["n_resamples"] == 300
        assert [r["slippage_multiplier"] for r in result["cost_stress"]] == ["2", "3"]
        assert result["break_even"]["method"] in ("interpolated", "extrapolated", "undefined")
        assert {b["label"] for b in result["benchmarks"]} == {
            "IWM",
            "QQQ",
            "SPY",
            "equal_weight(IWM,QQQ,SPY)",
        }
        assert result["gate_benchmark"] == "equal_weight(IWM,QQQ,SPY)"
        assert result["drawdown_risk"]["owner_halt"] == pytest.approx(0.10)
        assert result["drawdown_risk"]["gate_limit"] == pytest.approx(0.15)
        regimes = result["regimes"]
        assert regimes["by_year"]
        assert regimes["by_symbol"]
        assert {r["key"] for r in regimes["by_spy_sma200"]} <= {"above", "below", "unknown"}
        assert result["gate"]["stage"] == "dev"
        assert result["gate_sample"] == "continuous_walk_forward_span"
        metrics = result["metrics"]
        assert metrics["gate_sample"] == "continuous_walk_forward_span"
        assert metrics["expectancy_r"] == metrics["span_expectancy_r"]
        assert metrics["gate_trades"] == metrics["span_trades"]
        span = result["continuous_in_walk_forward_span"]
        assert metrics["span_trades"] == span["trades"]
        assert result["continuous_in_walk_forward_span_bootstrap"]["n_trades"] == span["trades"]
        assert result["gate"]["passed"] is result["metrics"]["gate_passed"]
        names = [row["name"] for row in result["gate"]["rows"]]
        assert names[0] == "expectancy_r"
        assert names[-1] == "vs_buy_and_hold"
    traded = [r for r in report["hypotheses"] if r["continuous"]["trades"]["trades"]]
    assert traded
    assert all(r["random_entry"] for r in traded)


def test_research_runs_disable_halts_and_are_labelled(dev_run: DevRun) -> None:
    report = dev_run.report()
    assert report["halts_label"] == "halts disabled for research statistics"
    for result in report["hypotheses"]:
        runs = [result["continuous"], *result["cost_stress"]]
        assert all(run["halts"] == "disabled" for run in runs)
        assert all(not any(run["halt_rejections"].values()) for run in runs)
        for window in result["walk_forward"]["windows"]:
            assert not any(window["halt_rejections"].values())


def test_text_report_is_labelled_development_with_lockbox_untouched(dev_run: DevRun) -> None:
    lines = dev_run.text().splitlines()
    banner = "DEVELOPMENT EVALUATION 2024-03..2024-12 — lockbox untouched"
    assert lines[0] == banner
    assert lines[-1] == banner
    text = "\n".join(lines)
    assert "halts disabled for research statistics" in text
    assert "Gate summary (pass_rule.dev):" in text
    for hypothesis_id in IDS:
        assert f"[{hypothesis_id}]" in text
    assert "vs_buy_and_hold (any of)" in text
    assert "PRIMARY gate sample" in text


def test_dev_is_identical_for_one_and_three_workers(
    tree: Tree, dev_run: DevRun, tmp_path: Path
) -> None:
    registry, out = tmp_path / "trials.jsonl", tmp_path / "out"
    assert tree.dev(registry, workers=3, out=out) == 0
    assert (out / "dev_report.json").read_bytes() == (dev_run.out / "dev_report.json").read_bytes()
    assert registry.read_bytes() == dev_run.registry.read_bytes()


def test_gate_sample_switch_uses_the_fresh_window_trades(
    tree: Tree, dev_run: DevRun, tmp_path: Path
) -> None:
    protocol = (tree.root / "protocol.yaml").read_text(encoding="utf-8")
    switched = protocol.replace(
        "gate_sample: continuous_walk_forward_span", "gate_sample: walk_forward_fresh_accounts"
    )
    assert switched != protocol
    (tree.root / "protocol_span.yaml").write_text(switched, encoding="utf-8")
    registry, out = tmp_path / "trials.jsonl", tmp_path / "out"
    assert tree.dev(registry, out=out, protocol="protocol_span.yaml") == 0
    report = json.loads((out / "dev_report.json").read_text(encoding="utf-8"))
    default = {r["hypothesis_id"]: r for r in dev_run.report()["hypotheses"]}
    for result in report["hypotheses"]:
        metrics = result["metrics"]
        assert metrics["gate_sample"] == "walk_forward_fresh_accounts"
        assert metrics["expectancy_r"] == metrics["wf_expectancy_r"]
        assert metrics["gate_trades"] == metrics["wf_trades"]
        boot = result["walk_forward"]["bootstrap"]
        assert metrics["bootstrap_expectancy_ci_low"] == boot["expectancy_ci_low"]
        # The simulations are the same: only the gate sample changed.
        same = default[result["hypothesis_id"]]
        assert result["continuous"] == same["continuous"]
        assert result["walk_forward"] == same["walk_forward"]
    text = (out / "dev_report.txt").read_text(encoding="utf-8")
    assert "per-trade gate sample: walk_forward_fresh_accounts" in text


def _fake_record(index: int) -> TrialRecord:
    return TrialRecord.model_validate(
        {
            "trial_id": f"old-{index}",
            "hypothesis_id": f"old_hypothesis_{index}",
            "config_hash": f"hash-{index}",
            "data_fingerprint": "d",
            "window": {"start": date(2024, 3, 1), "end": date(2024, 12, 31)},
            "mode": "dev",
            "git_commit": COMMIT,
            "git_dirty": False,
            "created_at_utc": NOW,
        }
    )


def test_dev_refuses_a_batch_beyond_the_trial_budget(
    tree: Tree, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = tmp_path / "trials.jsonl"
    lines = [_fake_record(i).model_dump_json() for i in range(8)]  # 1 + 8 + 4 = 13 > 12
    registry.write_text("\n".join(lines) + "\n", encoding="utf-8")
    before = registry.read_bytes()
    assert tree.dev(registry, out=tmp_path / "out") == 2
    assert "TRIAL_BUDGET_EXCEEDED" in capsys.readouterr().err
    assert registry.read_bytes() == before
    assert not (tmp_path / "out").exists()
    # Exactly at the budget (7 earlier trials: 1 + 7 + 4 = 12) the batch runs.
    registry.write_text("\n".join(lines[:7]) + "\n", encoding="utf-8")
    assert tree.dev(registry) == 0
    assert count_trials(registry) == 12


def test_dev_refuses_a_config_touching_the_lockbox(
    tree: Tree, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_tree(
        tree.root,
        ids=("mr_a1_ibs",),
        hypotheses="touching",
        config_changes=(("end_date: 2024-12-31", "end_date: 2025-01-31"),),
    )
    registry = tmp_path / "trials.jsonl"
    assert tree.dev(registry, hypotheses="touching") == 2
    assert "LOCKBOX_TOUCHED" in capsys.readouterr().err
    assert not registry.exists()


def test_dev_records_a_dirty_tree(tree: Tree, tmp_path: Path) -> None:
    registry = tmp_path / "trials.jsonl"
    assert tree.dev(registry, extra=["--hypotheses", "tf_t1_sma200"], git=dirty_git) == 0
    (record,) = read_trials(registry)
    assert (record.git_commit, record.git_dirty) == (COMMIT, True)
    report_registry = tmp_path / "unknown.jsonl"
    assert tree.dev(report_registry, extra=["--hypotheses", "tf_t1_sma200"], git=broken_git) == 0
    (unknown,) = read_trials(report_registry)
    assert (unknown.git_commit, unknown.git_dirty) == (None, True)


def test_official_cli_keeps_the_halts(tree: Tree, tmp_path: Path) -> None:
    """The official run of the same config still halts; the research run never does."""
    build_tree(
        tree.root,
        ids=("mr_a1_ibs",),
        hypotheses="tight",
        config_changes=(("max_daily_loss_pct: 0.02", "max_daily_loss_pct: 0.001"),),
    )
    config = tree.root / "configs_tight" / "mr_a1_ibs.yaml"
    official = tmp_path / "official.json"
    args = ["--config", str(config), "--data", str(tree.data), "--starting-cash", "100000"]
    assert backtest_main([*args, "--workers", "1", "--out", str(official)]) == 0
    rejections = json.loads(official.read_text(encoding="utf-8"))["counters"]["rejections"]
    assert rejections.get("DAILY_LOSS_LIMIT", 0) > 0
    registry, out = tmp_path / "trials.jsonl", tmp_path / "out"
    assert tree.dev(registry, hypotheses="tight", out=out) == 0
    report = json.loads((out / "dev_report.json").read_text(encoding="utf-8"))
    (result,) = report["hypotheses"]
    assert not any(result["continuous"]["halt_rejections"].values())
    assert result["continuous"]["trades"]["trades"] > 0


# --------------------------------------------------------------------------- lockbox


@pytest.fixture
def dev_registry(dev_run: DevRun, tmp_path: Path) -> Path:
    registry = tmp_path / "trials.jsonl"
    shutil.copyfile(dev_run.registry, registry)
    return registry


def test_lockbox_opens_once_and_records_the_trial(
    tree: Tree, dev_registry: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = len(read_trials(dev_registry))
    out = tmp_path / "out"
    assert tree.lockbox(dev_registry, "mr_a2_ibs_sma200", out=out, workers=2) == 0
    records = read_trials(dev_registry)
    assert len(records) == before + 1
    record = records[-1]
    assert record.mode == "lockbox"
    assert (record.window.start, record.window.end) == (date(2025, 1, 1), date(2025, 6, 30))
    assert (record.git_commit, record.git_dirty) == (COMMIT, False)
    assert isinstance(record.metrics["gate_passed"], bool)
    dev_record = next(r for r in records if r.hypothesis_id == "mr_a2_ibs_sma200")
    assert record.config_hash == dev_record.config_hash
    low, high = (
        dev_record.metrics["bootstrap_expectancy_ci_low"],
        dev_record.metrics["bootstrap_expectancy_ci_high"],
    )
    expectancy = record.metrics["expectancy_r"]
    assert isinstance(low, float)
    assert isinstance(high, float)
    if isinstance(expectancy, float):
        inside = low <= expectancy <= high
        assert record.metrics["expectancy_r_inside_dev_bootstrap_ci"] is inside
    report = json.loads((out / "lockbox_mr_a2_ibs_sma200.json").read_text(encoding="utf-8"))
    assert report["official"]["halts"] == "enabled"
    assert report["statistics"]["halts"] == "disabled"
    assert report["dev_trial_id"] == dev_record.trial_id
    text = (out / "lockbox_mr_a2_ibs_sma200.txt").read_text(encoding="utf-8")
    assert text.startswith("LOCKBOX EVALUATION 2025-01..2025-06 — opened once for mr_a2_ibs_sma200")
    assert "gate (pass_rule.lockbox):" in text
    # The trial count does not grow: same (hypothesis, config_hash) pair.
    assert count_trials(dev_registry) == len(IDS) + 1
    capsys.readouterr()
    assert tree.lockbox(dev_registry, "mr_a2_ibs_sma200") == 2
    assert "LOCKBOX_ALREADY_OPENED" in capsys.readouterr().err
    assert len(read_trials(dev_registry)) == before + 1


@pytest.mark.parametrize(
    ("hypothesis", "git", "code"),
    [
        ("not_registered", clean_git, "LOCKBOX_NOT_REGISTERED"),
        ("mr_a1_ibs", dirty_git, "LOCKBOX_DIRTY_TREE"),
        ("mr_a1_ibs", broken_git, "LOCKBOX_NO_COMMIT"),
    ],
)
def test_lockbox_guard_refuses(
    tree: Tree,
    dev_registry: Path,
    capsys: pytest.CaptureFixture[str],
    hypothesis: str,
    git: Callable[[Sequence[str]], str],
    code: str,
) -> None:
    before = dev_registry.read_bytes()
    assert tree.lockbox(dev_registry, hypothesis, git=git) == 2
    assert code in capsys.readouterr().err
    assert dev_registry.read_bytes() == before


def test_lockbox_refuses_without_a_lockbox_pass_rule(
    tree: Tree, dev_registry: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_tree(tree.root, ids=("mr_a1_ibs",), hypotheses="no_lockbox", drop_lockbox_rule=True)
    before = dev_registry.read_bytes()
    assert tree.lockbox(dev_registry, "mr_a1_ibs", hypotheses="no_lockbox") == 2
    assert "LOCKBOX_NO_PASS_RULE" in capsys.readouterr().err
    assert dev_registry.read_bytes() == before


def test_lockbox_refuses_without_a_dev_trial_of_the_same_config(
    tree: Tree, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "trials.jsonl"
    assert tree.lockbox(empty, "mr_a1_ibs") == 2
    assert "LOCKBOX_NO_DEV_TRIAL" in capsys.readouterr().err
    assert not empty.exists()


# --------------------------------------------------------------------------- list / report


def test_list_and_report(tree: Tree, dev_run: DevRun, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", *tree.common(dev_run.registry)]) == 0
    listed = capsys.readouterr().out
    assert f"Registered hypotheses: {len(IDS)}" in listed
    assert f"Trials used: {len(IDS) + 1} of 12" in listed
    assert "lockbox closed" in listed
    assert main(["report", *tree.common(dev_run.registry)]) == 0
    reported = capsys.readouterr().out
    assert f"{len(IDS)} records, {len(IDS) + 1} trials of 12" in reported
    for hypothesis_id in IDS:
        assert f"{hypothesis_id}-dev-20261002T120000Z" in reported
