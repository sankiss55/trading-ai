"""Research CLI of strategy family v2 (plan WU4; ``research/README.md``; DECISIONS.md
"Strategy v1.0.0 result and family v2").

Usage::

    python -m backtest.research_cli list    [--registry F]
    python -m backtest.research_cli dev     --data DIR --starting-cash N
                                            [--hypotheses all|id,...] [--workers W]
                                            [--registry F] [--out DIR]
    python -m backtest.research_cli lockbox --hypothesis ID --data DIR --starting-cash N
                                            [--workers W] [--registry F] [--out DIR]
    python -m backtest.research_cli report  [--registry F]

Common options: ``--protocol`` (default ``research/protocol.yaml``), ``--hypotheses-dir``
(default ``research/hypotheses``) and ``--root`` (base of relative ``config_path`` values,
default: the working directory). ``--registry`` defaults to ``research/trials.jsonl``.

**dev** evaluates registered hypotheses on the DEVELOPMENT window only (refused when a
simulated window touches the lockbox or the config window is not the protocol's dev
window). Per hypothesis, all in one process pool (:func:`backtest.research.run_config_jobs`):

a) one continuous dev-window run with the loss-limit halts DISABLED (research-only copy
   of the config: daily / weekly / max-drawdown limits set to 1.0; exposure and position
   limits kept) -> session marks -> daily returns -> CAGR, Sharpe, Sortino, max DD,
   exposure, PSR, DSR and a stationary-bootstrap Monte Carlo of the max drawdown;
b) every walk-forward test window as a fresh account (halts disabled likewise): the
   pooled out-of-sample trades, and the continuous run's trades entered in the same
   span; the protocol's ``gate_sample`` picks the one the per-trade criteria use
   (per-trade R statistics, ISO-week block bootstrap of expectancy and profit factor);
c) the continuous run again at each cost-stress multiplier (broker slippage scaled,
   sizing unchanged) and the break-even cost by linear interpolation;
d) buy-and-hold of every ETF and the equal-weight basket from the same bars and costs;
e) the random-entry benchmark of the continuous-run trades;
f) regime splits of the continuous-run trades: calendar year, symbol and SPY above /
   below its SMA200 at the decision session (the session before the entry fill);
g) batch statistics: PBO (CSCV) on the aligned daily-return matrix of the batch (and per
   family with >= 2 configs), DSR with N = registry trials including this batch and V =
   variance of the batch's per-session Sharpe ratios, Holm / BHY adjusted p-values of the
   per-trade t-tests.

Each hypothesis's ``pass_rule["dev"]`` is then evaluated (``{op, value}`` criteria, all
required; ``any_of`` = at least one alternative mapping passes in full), a JSON and a text
report are written and one :class:`~backtest.registry.TrialRecord` per hypothesis is
appended to the registry. The batch is refused BEFORE any simulation when it would exceed
the trial budget.

**lockbox** opens the lockbox window once for one hypothesis
(:func:`~backtest.registry.assert_lockbox_allowed`, plus a recorded dev trial of the same
config hash): one official continuous run (halts ENABLED, the pass rule is evaluated on
it) and one halts-disabled statistics run, then ``pass_rule["lockbox"]`` (including
"expectancy inside the dev bootstrap CI" read from the dev trial) and a lockbox trial
record.

Determinism: every randomised statistic uses the protocol seeds and results are merged
in job order, so reports do not depend on ``--workers``. The wall clock (``created_at``)
and ``git rev-parse HEAD`` / ``git status --porcelain`` are read here, in the CLI layer,
and passed down. Exit codes: ``0`` done, ``2`` refused or invalid input.
Backtest results do not imply future profitability.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import math
import os
import statistics
import subprocess
import sys
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.config import AppConfig, ConfigError, LoadedConfig, load_config, pending_backtest_decisions
from backtest.benchmark import (
    DailyBar,
    buy_and_hold,
    daily_bars_from_bars,
    equal_weight_basket,
)
from backtest.data import BacktestData, load_backtest_data
from backtest.random_entry import random_entry_benchmark, trade_specs_from_records
from backtest.registry import (
    DEFAULT_HYPOTHESES_DIR,
    DEFAULT_TRIALS_PATH,
    TRIAL_BUDGET,
    V1_TRIAL_COUNT,
    BacktestResearchError,
    GitStatus,
    Hypothesis,
    TrialRecord,
    TrialWindow,
    append_trial,
    assert_lockbox_allowed,
    load_hypotheses,
    read_trials,
)
from backtest.report import DISCLAIMER, EquityPoint, TradeRecord
from backtest.research import HALT_CODES, ConfigJob, run_config_jobs
from backtest.runner import (
    BacktestRefusedError,
    SimulationJob,
    SimulationResult,
    data_fingerprint,
    walk_forward_windows,
)
from backtest.stats import (
    BootstrapResult,
    DeflatedSharpe,
    PBOResult,
    PerformanceSummary,
    ReturnMoments,
    bhy_adjust,
    block_bootstrap,
    daily_returns,
    deflated_sharpe_ratio,
    holm_adjust,
    monte_carlo_drawdown,
    performance_summary,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    return_moments,
    trade_r_stats,
)
from domain.errors import NonRetryableError
from domain.market.indicators import sma
from domain.models import DataFeed, SessionDay, UtcDatetime

__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "DEV_BASE_METRICS",
    "HALTS_DISABLED_LABEL",
    "LOCKBOX_METRICS",
    "BreakEven",
    "DateWindow",
    "DevReport",
    "GateResult",
    "GateRow",
    "GitRunner",
    "LockboxReport",
    "MetricValue",
    "ResearchProtocol",
    "Study",
    "break_even_cost",
    "cost_metric_name",
    "dev_banner",
    "dev_metric_names",
    "dev_windows",
    "evaluate_pass_rule",
    "halts_disabled",
    "load_protocol",
    "load_study",
    "main",
    "read_git_status",
    "render_dev_text",
    "render_lockbox_text",
    "run_dev_async",
    "run_lockbox_async",
    "select_hypotheses",
    "session_marks",
    "subprocess_git",
    "trials_after",
    "validate_pass_rule",
]

DEFAULT_PROTOCOL_PATH: Final = Path("research/protocol.yaml")
HALTS_DISABLED_LABEL: Final = "halts disabled for research statistics"
REGIME_SYMBOL: Final = "SPY"
REGIME_SMA_PERIOD: Final = 200
SHARPE_UNITS: Final = (
    "per-session (daily) Sharpe ratios: mean / sd of session-close returns, not annualised"
)
_NO_HALT: Final = Decimal(1)
_OPS: Final[tuple[str, ...]] = (">=", ">", "<=", "<", "==")

MetricValue = float | int | str | bool | None
GitRunner = Callable[[Sequence[str]], str]
"""Runs ``git <args>`` and returns its standard output (raises on failure)."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")


# --------------------------------------------------------------------------- protocol


class DateWindow(_Model):
    """Inclusive date window."""

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("window end must not be before its start")
        return self

    def label(self) -> str:
        """``YYYY-MM..YYYY-MM``."""
        return f"{self.start:%Y-%m}..{self.end:%Y-%m}"


class CostProtocol(_Model):
    """Base cost per side and the cost-stress multipliers (each > 1, increasing)."""

    base_bps_per_side: Decimal = Field(ge=0)
    stress_multipliers: tuple[Decimal, ...] = Field(min_length=1)

    @field_validator("stress_multipliers")
    @classmethod
    def _increasing(cls, value: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        if any(m <= 1 for m in value) or any(b <= a for a, b in pairwise(value)):
            raise ValueError("stress multipliers must be > 1 and strictly increasing")
        return value


class WalkForwardProtocol(_Model):
    train_months: int = Field(ge=1)
    test_months: int = Field(ge=1)


class BootstrapProtocol(_Model):
    resamples: int = Field(ge=1)
    confidence: float = Field(gt=0, lt=1)
    seed: int = Field(ge=0)


class MonteCarloProtocol(_Model):
    sims: int = Field(ge=1)
    mean_block_length: float = Field(ge=1)
    seed: int = Field(ge=0)


class RandomEntryProtocol(_Model):
    sims: int = Field(ge=1)
    seed: int = Field(ge=0)


class PBOProtocol(_Model):
    blocks: int = Field(ge=2)

    @field_validator("blocks")
    @classmethod
    def _even(cls, value: int) -> int:
        if value % 2:
            raise ValueError("PBO blocks must be even")
        return value


GateSample = Literal["walk_forward_fresh_accounts", "continuous_walk_forward_span"]
"""Trades the per-trade dev criteria are measured on: the pooled walk-forward test windows
as fresh accounts (a trade open at a window end is never closed, so it is not counted),
or the continuous dev run's trades entered inside the walk-forward test span."""


class ResearchProtocol(_Model):
    """``research/protocol.yaml``: windows, trial budget, costs and statistic settings."""

    protocol_version: str = Field(min_length=1)
    dev_window: DateWindow
    lockbox_window: DateWindow
    embargo_start: date
    trial_budget: int = Field(ge=1)
    v1_trials: int = Field(ge=0)
    costs: CostProtocol
    gate_sample: GateSample
    walk_forward: WalkForwardProtocol
    bootstrap: BootstrapProtocol
    monte_carlo: MonteCarloProtocol
    random_entry: RandomEntryProtocol
    pbo: PBOProtocol

    @model_validator(mode="after")
    def _ordered_windows(self) -> Self:
        if self.dev_window.end >= self.lockbox_window.start:
            raise ValueError("the development window must end before the lockbox starts")
        if self.lockbox_window.end >= self.embargo_start:
            raise ValueError("the lockbox must end before the embargo starts")
        if self.v1_trials > self.trial_budget:
            raise ValueError("v1_trials exceeds the trial budget")
        return self


def load_protocol(path: Path) -> ResearchProtocol:
    """Load and validate the research protocol; it must agree with the registry
    constants (:data:`~backtest.registry.TRIAL_BUDGET`, ``V1_TRIAL_COUNT``).

    Raises:
        BacktestResearchError: code ``PROTOCOL_INVALID``.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        protocol = ResearchProtocol.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise BacktestResearchError(
            f"invalid research protocol {path}: {exc}", code="PROTOCOL_INVALID"
        ) from exc
    if protocol.trial_budget != TRIAL_BUDGET or protocol.v1_trials != V1_TRIAL_COUNT:
        raise BacktestResearchError(
            f"{path}: trial_budget {protocol.trial_budget} / v1_trials {protocol.v1_trials} "
            f"differ from the registry ({TRIAL_BUDGET} / {V1_TRIAL_COUNT})",
            code="PROTOCOL_INVALID",
        )
    return protocol


# --------------------------------------------------------------------------- metrics


DEV_BASE_METRICS: Final[tuple[str, ...]] = (
    "expectancy_r",
    "t_stat",
    "bootstrap_expectancy_ci_low",
    "profit_factor",
    "mc_max_drawdown_p95",
    "deflated_sharpe_ratio",
    "pbo",
    "random_entry_percentile",
    "sharpe_minus_bh_sharpe",
    "max_drawdown_over_bh_max_drawdown",
    "cagr_over_bh_cagr",
    "bootstrap_expectancy_ci_high",
    "bootstrap_profit_factor_ci_low",
    "bootstrap_profit_factor_ci_high",
    "gate_sample",
    "gate_trades",
    "wf_trades",
    "wf_r_trades",
    "wf_expectancy_r",
    "wf_t_stat",
    "wf_profit_factor",
    "wf_open_trades_at_end",
    "span_trades",
    "span_expectancy_r",
    "span_t_stat",
    "span_profit_factor",
    "p_value",
    "holm_p_value",
    "bhy_p_value",
    "sharpe",
    "sharpe_daily",
    "sortino",
    "cagr",
    "max_drawdown",
    "exposure",
    "psr",
    "dsr_sr0",
    "n_trials",
    "trials_sr_variance",
    "mc_max_drawdown_p50",
    "prob_drawdown_ge_owner_halt",
    "prob_drawdown_ge_gate_limit",
    "break_even_cost_bps",
    "bh_sharpe",
    "bh_cagr",
    "bh_max_drawdown",
    "pbo_family",
    "continuous_trades",
    "continuous_expectancy_r",
    "gate_passed",
)
"""Metrics of a dev trial record (plus :func:`cost_metric_name` of every multiplier).
The first eleven are the acceptance-table gate values; the per-trade ones (expectancy,
t, bootstrap CI, PF, p-values) are measured on the protocol's ``gate_sample``; ``wf_*``
and ``span_*`` give both samples."""

LOCKBOX_METRICS: Final[tuple[str, ...]] = (
    "expectancy_r",
    "profit_factor",
    "max_drawdown",
    "expectancy_r_inside_dev_bootstrap_ci",
    "trades",
    "t_stat",
    "sharpe",
    "cagr",
    "stats_trades",
    "stats_expectancy_r",
    "stats_profit_factor",
    "stats_max_drawdown",
    "stats_sharpe",
    "stats_cagr",
    "gate_passed",
)
"""Metrics of a lockbox trial record; the first four are measured on the official
(halts-enabled) run, ``stats_*`` on the halts-disabled run."""


def _multiplier_text(multiplier: Decimal) -> str:
    if multiplier == multiplier.to_integral_value():
        return str(int(multiplier))
    return format(multiplier.normalize(), "f")


def cost_metric_name(multiplier: Decimal) -> str:
    """``expectancy_r_at_<m>x_cost`` (e.g. ``expectancy_r_at_2x_cost``)."""
    return f"expectancy_r_at_{_multiplier_text(multiplier)}x_cost"


def dev_metric_names(protocol: ResearchProtocol) -> frozenset[str]:
    """Every metric a dev pass rule may reference."""
    return frozenset(DEV_BASE_METRICS) | {
        cost_metric_name(m) for m in protocol.costs.stress_multipliers
    }


# --------------------------------------------------------------------------- pass rules


class GateRow(_Model):
    """One evaluated criterion; ``op = "any_of"`` holds its evaluated alternatives."""

    name: str
    op: str
    threshold: float | bool | None
    value: MetricValue
    passed: bool
    alternatives: tuple[tuple[GateRow, ...], ...] = ()


class GateResult(_Model):
    """A pass rule evaluated on the metrics of one trial."""

    stage: Literal["dev", "lockbox"]
    passed: bool
    rows: tuple[GateRow, ...]


def _rule_error(where: str, message: str) -> BacktestResearchError:
    return BacktestResearchError(f"invalid pass rule {where}: {message}", code="PASS_RULE_INVALID")


def validate_pass_rule(rule: object, known: Iterable[str], *, where: str) -> None:
    """Check the structure of a pass rule before anything runs.

    A rule is a non-empty mapping ``name -> {op, value}`` (``op`` in ``>= > <= < ==``; a
    boolean ``value`` needs ``==``; ``name`` must be a known metric) or ``name -> {any_of:
    [rule, ...]}`` (``name`` is then a label and each alternative is itself a rule).

    Raises:
        BacktestResearchError: code ``PASS_RULE_INVALID``.
    """
    known_names = frozenset(known)
    if not isinstance(rule, Mapping) or not rule:
        raise _rule_error(where, "must be a non-empty mapping of criteria")
    for name, spec in rule.items():
        label = f"{where}.{name}"
        if not isinstance(spec, Mapping):
            raise _rule_error(label, "must be {op, value} or {any_of: [...]}")
        if set(spec) == {"any_of"}:
            alternatives = spec["any_of"]
            if not isinstance(alternatives, list) or not alternatives:
                raise _rule_error(label, "any_of must be a non-empty list of rules")
            for number, alternative in enumerate(alternatives):
                validate_pass_rule(alternative, known_names, where=f"{label}.any_of[{number}]")
            continue
        if set(spec) != {"op", "value"}:
            raise _rule_error(label, f"expected keys op and value, got {sorted(spec)}")
        if name not in known_names:
            raise _rule_error(label, f"unknown metric {name!r}")
        op, value = spec["op"], spec["value"]
        if op not in _OPS:
            raise _rule_error(label, f"unknown op {op!r} (allowed: {', '.join(_OPS)})")
        if isinstance(value, bool):
            if op != "==":
                raise _rule_error(label, "a boolean threshold needs op ==")
        elif not isinstance(value, int | float | Decimal) or not math.isfinite(float(value)):
            raise _rule_error(label, f"threshold must be a finite number or a boolean: {value!r}")


def _threshold(value: object) -> float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float | Decimal):
        return float(value)
    raise _rule_error("<value>", f"invalid threshold {value!r}")  # pragma: no cover - validated


def _compare(value: MetricValue, op: str, threshold: float | bool) -> bool:
    """Fail closed: a missing, non-numeric or NaN value never passes."""
    if isinstance(threshold, bool):
        return isinstance(value, bool) and value is threshold
    if value is None or isinstance(value, bool | str):
        return False
    number = float(value)
    if math.isnan(number):
        return False
    if op == ">=":
        return number >= threshold
    if op == ">":
        return number > threshold
    if op == "<=":
        return number <= threshold
    if op == "<":
        return number < threshold
    return number == threshold


def _evaluate(rule: Mapping[str, Any], metrics: Mapping[str, MetricValue]) -> tuple[GateRow, ...]:
    rows: list[GateRow] = []
    for name, spec in rule.items():
        if "any_of" in spec:
            alternatives = tuple(_evaluate(alt, metrics) for alt in spec["any_of"])
            rows.append(
                GateRow(
                    name=name,
                    op="any_of",
                    threshold=None,
                    value=None,
                    passed=any(all(r.passed for r in alt) for alt in alternatives),
                    alternatives=alternatives,
                )
            )
            continue
        threshold = _threshold(spec["value"])
        value = metrics.get(name)
        rows.append(
            GateRow(
                name=name,
                op=spec["op"],
                threshold=threshold,
                value=value,
                passed=_compare(value, spec["op"], threshold),
            )
        )
    return tuple(rows)


def evaluate_pass_rule(
    rule: Mapping[str, Any],
    metrics: Mapping[str, MetricValue],
    *,
    stage: Literal["dev", "lockbox"],
) -> GateResult:
    """Evaluate a (validated) pass rule: PASS only when every criterion passes."""
    rows = _evaluate(rule, metrics)
    return GateResult(stage=stage, passed=all(r.passed for r in rows), rows=rows)


# --------------------------------------------------------------------------- git


def subprocess_git(cwd: Path) -> GitRunner:
    """A :data:`GitRunner` that runs the ``git`` executable in ``cwd`` (``shell=False``)."""

    def run(args: Sequence[str]) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=60
        )
        return completed.stdout

    return run


def read_git_status(runner: GitRunner) -> GitStatus:
    """``git rev-parse HEAD`` and ``git status --porcelain``. A failing git (no
    repository, no executable) gives an unknown commit and a dirty tree (fail closed)."""
    try:
        commit = runner(["rev-parse", "HEAD"]).strip() or None
        dirty = bool(runner(["status", "--porcelain"]).strip())
    except (OSError, subprocess.SubprocessError):
        return GitStatus(commit=None, dirty=True)
    return GitStatus(commit=commit, dirty=dirty)


# --------------------------------------------------------------------------- studies


@dataclass(frozen=True, slots=True)
class Study:
    """A registered hypothesis and its loaded research configuration."""

    hypothesis: Hypothesis
    loaded: LoadedConfig


def halts_disabled(config: AppConfig) -> AppConfig:
    """Research-only copy of ``config`` with the loss-limit halts disabled
    (:data:`HALTS_DISABLED_LABEL`): ``max_daily_loss_pct``, ``max_weekly_loss_pct`` and
    ``max_drawdown_pct`` set to 1.0 (a 100 % loss is never reached). Exposure, position
    and open-risk limits are kept. Never used by the official run."""
    risk = config.risk.model_copy(
        update={
            "max_daily_loss_pct": _NO_HALT,
            "max_weekly_loss_pct": _NO_HALT,
            "max_drawdown_pct": _NO_HALT,
        }
    )
    return config.model_copy(update={"risk": risk})


def select_hypotheses(available: Sequence[Hypothesis], selection: str) -> tuple[Hypothesis, ...]:
    """``all`` or comma-separated ids, returned sorted by id (the batch order).

    Raises:
        BacktestResearchError: ``HYPOTHESIS_UNKNOWN`` / ``HYPOTHESIS_SELECTION_INVALID``.
    """
    by_id = {h.id: h for h in available}
    if selection.strip() == "all":
        ids = sorted(by_id)
    else:
        ids = [part.strip() for part in selection.split(",") if part.strip()]
        if not ids or len(set(ids)) != len(ids):
            raise BacktestResearchError(
                f"invalid hypothesis selection {selection!r}", code="HYPOTHESIS_SELECTION_INVALID"
            )
        ids = sorted(ids)
    unknown = [i for i in ids if i not in by_id]
    if unknown or not ids:
        raise BacktestResearchError(
            f"hypotheses not registered: {', '.join(unknown) or '(none registered)'}",
            code="HYPOTHESIS_UNKNOWN",
        )
    return tuple(by_id[i] for i in ids)


def load_study(hypothesis: Hypothesis, root: Path, protocol: ResearchProtocol) -> Study:
    """Load the hypothesis's config and check it matches the registration and protocol
    (symbols, cost per side = ``risk.slippage_buffer_bps``, stress multipliers).

    Raises:
        ConfigError, BacktestRefusedError, BacktestResearchError
        (``HYPOTHESIS_CONFIG_MISMATCH``).
    """
    path = Path(hypothesis.config_path)
    loaded = load_config(path if path.is_absolute() else root / path)
    config = loaded.config
    pending = pending_backtest_decisions(config)
    if pending:
        raise BacktestRefusedError(pending)
    problems: list[str] = []
    if set(config.universe.whitelist or ()) != set(hypothesis.symbols):
        problems.append(f"whitelist {config.universe.whitelist} != symbols {hypothesis.symbols}")
    costs = protocol.costs
    if hypothesis.costs.base_bps_per_side != costs.base_bps_per_side:
        problems.append("hypothesis base cost differs from the protocol")
    if hypothesis.costs.stress_multipliers != costs.stress_multipliers:
        problems.append("hypothesis stress multipliers differ from the protocol")
    if config.risk.slippage_buffer_bps != costs.base_bps_per_side:
        problems.append("risk.slippage_buffer_bps differs from the protocol cost per side")
    if problems:
        raise BacktestResearchError(
            f"{hypothesis.id}: {'; '.join(problems)}", code="HYPOTHESIS_CONFIG_MISMATCH"
        )
    return Study(hypothesis=hypothesis, loaded=loaded)


def dev_windows(
    config: AppConfig, protocol: ResearchProtocol
) -> tuple[DateWindow, tuple[DateWindow, ...]]:
    """The continuous window and the walk-forward test windows of a dev evaluation.

    Raises:
        BacktestResearchError: ``LOCKBOX_TOUCHED`` when any window ends on or after the
            lockbox start; ``DEV_WINDOW_MISMATCH`` when the config window or walk-forward
            months differ from the protocol.
    """
    backtest = config.backtest
    start, end = backtest.start_date, backtest.end_date
    train, test = backtest.walk_forward_train_months, backtest.walk_forward_test_months
    if start is None or end is None or train is None or test is None:  # pragma: no cover
        raise BacktestRefusedError(pending_backtest_decisions(config))
    tests = tuple(
        DateWindow(start=test_start, end=test_end)
        for _, _, test_start, test_end in walk_forward_windows(
            start, end, train_months=train, test_months=test
        )
    )
    lockbox = protocol.lockbox_window
    for window in (DateWindow(start=start, end=end), *tests):
        if window.end >= lockbox.start:
            raise BacktestResearchError(
                f"dev refused: window {window.start}..{window.end} touches the lockbox "
                f"{lockbox.start}..{lockbox.end}; dev uses {protocol.dev_window.start}.."
                f"{protocol.dev_window.end} only",
                code="LOCKBOX_TOUCHED",
            )
    dev = protocol.dev_window
    if (start, end) != (dev.start, dev.end):
        raise BacktestResearchError(
            f"dev refused: backtest window {start}..{end} is not the protocol development "
            f"window {dev.start}..{dev.end}",
            code="DEV_WINDOW_MISMATCH",
        )
    wf = protocol.walk_forward
    if (train, test) != (wf.train_months, wf.test_months):
        raise BacktestResearchError(
            f"dev refused: walk-forward {train}/{test} months differ from the protocol "
            f"{wf.train_months}/{wf.test_months}",
            code="DEV_WINDOW_MISMATCH",
        )
    return DateWindow(start=start, end=end), tests


def trials_after(
    existing: Sequence[TrialRecord], batch: Iterable[tuple[str, str]], v1_trials: int
) -> int:
    """Trials counted for the budget / DSR after adding ``batch`` ``(hypothesis_id,
    config_hash)`` pairs (same rule as :func:`backtest.registry.count_trials`)."""
    pairs = {(r.hypothesis_id, r.config_hash) for r in existing} | set(batch)
    return len(pairs) + v1_trials


# --------------------------------------------------------------------------- series


def session_marks(
    curve: Sequence[EquityPoint], sessions: Sequence[SessionDay], starting_equity: Decimal
) -> tuple[list[Decimal], list[bool]]:
    """Equity after every session (``[starting_equity, mark_1, ..., mark_n]``) and, per
    session, whether a sample in ``[open_k, open_(k+1))`` shows a position.

    The runner samples the equity after each session's close (``close + grace + 1 s``),
    so the mark of session ``k`` is the last sample before the NEXT session's open (the
    last sample for the last session). ``backtest.stats.session_close_equity`` cuts at
    ``close_utc`` and would return the previous session's value for these curves.
    """
    points = sorted(curve, key=lambda p: p.timestamp_utc)
    stamps = [p.timestamp_utc for p in points]
    equity = [starting_equity]
    flags: list[bool] = []
    begin = bisect_left(stamps, sessions[0].open_utc) if sessions else 0
    for index in range(len(sessions)):
        end = (
            bisect_left(stamps, sessions[index + 1].open_utc)
            if index + 1 < len(sessions)
            else len(points)
        )
        equity.append(points[end - 1].equity if end > 0 else starting_equity)
        flags.append(any(p.exposure > 0 for p in points[begin:end]))
        begin = max(begin, end)
    return equity, flags


class TradeSampleStats(_Model):
    """Per-trade statistics of a set of trades (R from ``result_r``; PF from net P&L)."""

    trades: int
    r_trades: int
    expectancy_r: float | None
    sd_r: float | None
    t_stat: float | None
    p_value: float | None
    """One-sided p-value of ``mean R > 0`` (exact Student t)."""
    ci_low: float | None
    ci_high: float | None
    win_rate: float | None
    profit_factor: float | None
    """``inf`` with gains and no loss; ``None`` without trades or P&L."""
    net_pnl: float


def _profit_factor(trades: Sequence[TradeRecord]) -> float | None:
    gain = sum((t.net_pnl for t in trades if t.net_pnl > 0), Decimal(0))
    loss = -sum((t.net_pnl for t in trades if t.net_pnl < 0), Decimal(0))
    if loss > 0:
        return float(gain / loss)
    return math.inf if gain > 0 else None


def trade_sample_stats(trades: Sequence[TradeRecord], *, confidence: float) -> TradeSampleStats:
    """:class:`TradeSampleStats` of ``trades``."""
    r_values = [float(t.result_r) for t in trades if t.result_r is not None]
    stats = trade_r_stats(r_values, confidence=confidence)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    return TradeSampleStats(
        trades=len(trades),
        r_trades=stats.n,
        expectancy_r=stats.mean,
        sd_r=stats.sd,
        t_stat=stats.t_stat,
        p_value=stats.p_value,
        ci_low=stats.ci_low,
        ci_high=stats.ci_high,
        win_rate=wins / len(trades) if trades else None,
        profit_factor=_profit_factor(trades),
        net_pnl=float(sum((t.net_pnl for t in trades), Decimal(0))),
    )


# --------------------------------------------------------------------------- report models


class ContinuousRun(_Model):
    """One continuous account over a window (statistics on its session marks)."""

    label: str
    halts: Literal["enabled", "disabled"]
    window: DateWindow
    slippage_multiplier: Decimal
    cost_bps_per_side: Decimal
    sessions: int
    performance: PerformanceSummary
    moments: ReturnMoments
    psr: float | None
    """P(true per-session Sharpe > 0) (probabilistic Sharpe ratio, benchmark 0)."""
    trades: TradeSampleStats
    open_trades_at_end: int
    halt_rejections: dict[str, int]


class DrawdownRisk(_Model):
    """Stationary-bootstrap Monte Carlo of the max drawdown of the daily returns."""

    n_sims: int
    mean_block_length: float
    seed: int
    observed: float
    p5: float
    p50: float
    p95: float
    owner_halt: float
    prob_ge_owner_halt: float
    gate_limit: float | None
    prob_ge_gate_limit: float | None


class WindowSummary(_Model):
    label: str
    window: DateWindow
    trades: int
    expectancy_r: float | None
    net_pnl: float
    open_trades_at_end: int
    halt_rejections: dict[str, int]


class WalkForward(_Model):
    """Walk-forward test windows as fresh accounts; pooled trades = primary gate sample."""

    train_months: int
    test_months: int
    windows: tuple[WindowSummary, ...]
    pooled: TradeSampleStats
    bootstrap: BootstrapResult
    open_trades_at_end: int
    """Trades still open at a window end: never closed, so not counted."""


class BreakEven(_Model):
    """Cost per side at which the continuous-run expectancy reaches 0 R."""

    cost_bps_per_side: float | None
    method: Literal["interpolated", "extrapolated", "undefined"]
    points: tuple[tuple[float, float | None], ...]


class BenchmarkSummary(_Model):
    label: str
    shares: tuple[tuple[str, int], ...]
    performance: PerformanceSummary


class RandomEntrySummary(_Model):
    n_sims: int
    seed: int
    n_trades: int
    cost_bps: float
    real_expectancy: float
    real_total_return: float
    expectancy_quantiles: tuple[float, float, float]
    total_return_quantiles: tuple[float, float, float]
    expectancy_percentile: float
    total_return_percentile: float


class SplitRow(_Model):
    key: str
    stats: TradeSampleStats


class RegimeSplits(_Model):
    sample: str
    by_year: tuple[SplitRow, ...]
    by_symbol: tuple[SplitRow, ...]
    by_spy_sma200: tuple[SplitRow, ...]
    spy_rule: str


class MultipleTesting(_Model):
    p_value: float | None
    holm_p_value: float
    bhy_p_value: float


class HypothesisDevResult(_Model):
    hypothesis_id: str
    family: str
    title: str
    config_path: str
    config_hash: str
    strategy_version: str
    trial_id: str
    continuous: ContinuousRun
    drawdown_risk: DrawdownRisk | None
    gate_sample: GateSample
    walk_forward: WalkForward
    continuous_in_walk_forward_span: TradeSampleStats
    """Continuous-run trades entered inside the walk-forward test span."""
    continuous_in_walk_forward_span_bootstrap: BootstrapResult
    cost_stress: tuple[ContinuousRun, ...]
    break_even: BreakEven
    benchmarks: tuple[BenchmarkSummary, ...]
    gate_benchmark: str | None
    random_entry: RandomEntrySummary | None
    regimes: RegimeSplits
    deflated_sharpe: DeflatedSharpe | None
    multiple_testing: MultipleTesting
    metrics: dict[str, MetricValue]
    gate: GateResult
    notes: tuple[str, ...]


class BatchStatistics(_Model):
    hypotheses: tuple[str, ...]
    n_trials: int
    trial_budget: int
    sharpe_units: str
    trials_sr_variance: float | None
    trials_sr_variance_source: str
    pbo: PBOResult | None
    pbo_by_family: dict[str, PBOResult]
    multiple_testing: str
    notes: tuple[str, ...]


class DevReport(_Model):
    """Development evaluation report (JSON via ``model_dump_json``)."""

    mode: Literal["dev"]
    banner: str
    halts_label: str
    disclaimer: str
    protocol: ResearchProtocol
    data_fingerprint: str
    starting_cash: Decimal
    git_commit: str | None
    git_dirty: bool
    created_at_utc: UtcDatetime
    batch: BatchStatistics
    hypotheses: tuple[HypothesisDevResult, ...]
    notes: tuple[str, ...]


class LockboxReport(_Model):
    """Lockbox evaluation report (one hypothesis, opened once)."""

    mode: Literal["lockbox"]
    banner: str
    disclaimer: str
    protocol: ResearchProtocol
    hypothesis_id: str
    family: str
    title: str
    config_hash: str
    strategy_version: str
    data_fingerprint: str
    starting_cash: Decimal
    git_commit: str | None
    git_dirty: bool
    created_at_utc: UtcDatetime
    dev_trial_id: str
    dev_data_fingerprint: str
    dev_expectancy_ci: tuple[float, float]
    dev_gate_passed: MetricValue
    official: ContinuousRun
    statistics: ContinuousRun
    metrics: dict[str, MetricValue]
    gate: GateResult
    trial_id: str
    notes: tuple[str, ...]


def dev_banner(protocol: ResearchProtocol) -> str:
    """``DEVELOPMENT EVALUATION 2016-11..2022-12 — lockbox untouched``."""
    return f"DEVELOPMENT EVALUATION {protocol.dev_window.label()} — lockbox untouched"


def _lockbox_banner(protocol: ResearchProtocol, hypothesis_id: str) -> str:
    return (
        f"LOCKBOX EVALUATION {protocol.lockbox_window.label()} — opened once for "
        f"{hypothesis_id} (official run: halts enabled)"
    )


# --------------------------------------------------------------------------- analysis


def _psr(moments: ReturnMoments) -> float | None:
    if moments.sharpe is None or moments.skewness is None or moments.kurtosis is None:
        return None
    try:
        return probabilistic_sharpe_ratio(
            moments.sharpe, 0.0, moments.n, moments.skewness, moments.kurtosis
        )
    except ValueError:
        return None


def _continuous_run(
    result: SimulationResult,
    *,
    label: str,
    halts: Literal["enabled", "disabled"],
    window: DateWindow,
    multiplier: Decimal,
    base_bps: Decimal,
    starting_cash: Decimal,
    confidence: float,
) -> tuple[ContinuousRun, list[float]]:
    equity, flags = session_marks(result.equity_curve, result.sessions, starting_cash)
    returns = daily_returns(equity)
    moments = return_moments(returns)
    run = ContinuousRun(
        label=label,
        halts=halts,
        window=window,
        slippage_multiplier=multiplier,
        cost_bps_per_side=base_bps * multiplier,
        sessions=len(result.sessions),
        performance=performance_summary(returns, exposure_flags=flags),
        moments=moments,
        psr=_psr(moments),
        trades=trade_sample_stats(result.trades, confidence=confidence),
        open_trades_at_end=result.counters.open_trades_at_end,
        halt_rejections={code: result.counters.rejections.get(code, 0) for code in HALT_CODES},
    )
    return run, returns


def _drawdown_risk(
    returns: Sequence[float], protocol: ResearchProtocol, config: AppConfig
) -> DrawdownRisk | None:
    if not returns:
        return None
    mc = protocol.monte_carlo
    owner_halt = config.risk.max_drawdown_pct
    gate_limit = config.backtest.max_drawdown_pct
    if owner_halt is None:  # pragma: no cover - guarded by pending_backtest_decisions
        return None
    halt = monte_carlo_drawdown(
        returns,
        mean_block_length=mc.mean_block_length,
        n_sims=mc.sims,
        threshold=float(owner_halt),
        seed=mc.seed,
    )
    gate = (
        None
        if gate_limit is None
        else monte_carlo_drawdown(
            returns,
            mean_block_length=mc.mean_block_length,
            n_sims=mc.sims,
            threshold=float(gate_limit),
            seed=mc.seed,
        )
    )
    return DrawdownRisk(
        n_sims=halt.n_sims,
        mean_block_length=halt.mean_block_length,
        seed=halt.seed,
        observed=halt.observed,
        p5=halt.p5,
        p50=halt.p50,
        p95=halt.p95,
        owner_halt=halt.threshold,
        prob_ge_owner_halt=halt.prob_drawdown_ge_threshold,
        gate_limit=None if gate is None else gate.threshold,
        prob_ge_gate_limit=None if gate is None else gate.prob_drawdown_ge_threshold,
    )


def _root(b0: float, e0: float, b1: float, e1: float) -> float:
    return b0 - e0 * (b1 - b0) / (e1 - e0)


def break_even_cost(points: Sequence[tuple[float, float | None]]) -> BreakEven:
    """Cost per side (bps) where the expectancy crosses 0 R, by linear interpolation of
    ``(cost_bps, expectancy_r)`` points; outside the measured range the segment nearest
    to zero is extrapolated (only when the expectancy falls as the cost grows)."""
    pts = tuple((float(b), None if e is None else float(e)) for b, e in points)
    known = sorted((b, e) for b, e in pts if e is not None)
    if len(known) < 2:
        return BreakEven(cost_bps_per_side=None, method="undefined", points=pts)
    for (b0, e0), (b1, e1) in pairwise(known):
        if e0 == 0.0:
            return BreakEven(cost_bps_per_side=b0, method="interpolated", points=pts)
        if (e0 > 0) != (e1 > 0) or e1 == 0.0:
            return BreakEven(
                cost_bps_per_side=_root(b0, e0, b1, e1), method="interpolated", points=pts
            )
    (b0, e0), (b1, e1) = known[-2:] if known[0][1] > 0 else known[:2]
    if e1 >= e0:
        return BreakEven(cost_bps_per_side=None, method="undefined", points=pts)
    return BreakEven(cost_bps_per_side=_root(b0, e0, b1, e1), method="extrapolated", points=pts)


def _benchmarks(
    bars: Mapping[str, Sequence[DailyBar]],
    symbols: Sequence[str],
    window: DateWindow,
    *,
    starting_cash: Decimal,
    cost_bps: Decimal,
    notes: list[str],
) -> tuple[tuple[BenchmarkSummary, ...], BenchmarkSummary | None]:
    summaries: list[BenchmarkSummary] = []
    for symbol in sorted(symbols):
        try:
            result = buy_and_hold(
                symbol,
                bars.get(symbol, ()),
                starting_cash=starting_cash,
                cost_bps=cost_bps,
                start=window.start,
                end=window.end,
            )
        except ValueError as exc:
            notes.append(f"buy-and-hold {symbol} unavailable: {exc}")
            continue
        summaries.append(
            BenchmarkSummary(label=result.label, shares=result.shares, performance=result.summary)
        )
    try:
        basket_result = equal_weight_basket(
            {s: bars.get(s, ()) for s in symbols},
            starting_cash=starting_cash,
            cost_bps=cost_bps,
            start=window.start,
            end=window.end,
        )
    except ValueError as exc:
        notes.append(f"equal-weight basket unavailable (B&H gate metrics are n/a): {exc}")
        return tuple(summaries), None
    basket = BenchmarkSummary(
        label=basket_result.label, shares=basket_result.shares, performance=basket_result.summary
    )
    return (*summaries, basket), basket


def _random_entry(
    bars: Mapping[str, Sequence[DailyBar]],
    result: SimulationResult,
    window: DateWindow,
    protocol: ResearchProtocol,
    notes: list[str],
) -> RandomEntrySummary | None:
    try:
        specs = trade_specs_from_records(result.trades, [s.session_date for s in result.sessions])
        bench = random_entry_benchmark(
            bars,
            specs,
            cost_bps=protocol.costs.base_bps_per_side,
            n_sims=protocol.random_entry.sims,
            seed=protocol.random_entry.seed,
            start=window.start,
            end=window.end,
        )
    except ValueError as exc:
        notes.append(f"random-entry benchmark unavailable: {exc}")
        return None
    return RandomEntrySummary(
        n_sims=bench.n_sims,
        seed=bench.seed,
        n_trades=bench.n_trades,
        cost_bps=bench.cost_bps,
        real_expectancy=bench.real_expectancy,
        real_total_return=bench.real_total_return,
        expectancy_quantiles=bench.expectancy_quantiles,
        total_return_quantiles=bench.total_return_quantiles,
        expectancy_percentile=bench.expectancy_percentile,
        total_return_percentile=bench.total_return_percentile,
    )


def _spy_regimes(bars: Sequence[DailyBar]) -> dict[date, str]:
    """SPY above / below its SMA200 at each session close (``None`` SMA = unknown)."""
    averages = sma([float(b.close) for b in bars], REGIME_SMA_PERIOD)
    regimes: dict[date, str] = {}
    for bar, average in zip(bars, averages, strict=True):
        if average is not None:
            regimes[bar.session_date] = "above" if float(bar.close) > average else "below"
    return regimes


def _split_rows(
    groups: Mapping[str, Sequence[TradeRecord]], confidence: float
) -> tuple[SplitRow, ...]:
    return tuple(
        SplitRow(key=key, stats=trade_sample_stats(groups[key], confidence=confidence))
        for key in sorted(groups)
    )


def _regime_splits(
    trades: Sequence[TradeRecord],
    all_sessions: Sequence[SessionDay],
    regimes: Mapping[date, str],
    confidence: float,
) -> RegimeSplits:
    index = {s.session_date: i for i, s in enumerate(all_sessions)}
    by_year: dict[str, list[TradeRecord]] = {}
    by_symbol: dict[str, list[TradeRecord]] = {}
    by_regime: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        entry_day = trade.entry_filled_at_utc.date()
        by_year.setdefault(str(entry_day.year), []).append(trade)
        by_symbol.setdefault(trade.symbol, []).append(trade)
        position = index.get(entry_day)
        decision = all_sessions[position - 1].session_date if position else None
        regime = regimes.get(decision, "unknown") if decision is not None else "unknown"
        by_regime.setdefault(regime, []).append(trade)
    return RegimeSplits(
        sample="continuous dev run (" + HALTS_DISABLED_LABEL + ")",
        by_year=_split_rows(by_year, confidence),
        by_symbol=_split_rows(by_symbol, confidence),
        by_spy_sma200=_split_rows(by_regime, confidence),
        spy_rule=(
            f"{REGIME_SYMBOL} close vs SMA{REGIME_SMA_PERIOD} of its closes at the decision "
            "session (the stored session before the entry fill); 'unknown' when the "
            "SMA or the bar is unavailable"
        ),
    )


def _bootstrap(trades: Sequence[TradeRecord], protocol: ResearchProtocol) -> BootstrapResult:
    boot = protocol.bootstrap
    return block_bootstrap(
        [t.entry_filled_at_utc.date() for t in trades],
        [None if t.result_r is None else float(t.result_r) for t in trades],
        [float(t.net_pnl) for t in trades],
        n_resamples=boot.resamples,
        confidence=boot.confidence,
        seed=boot.seed,
    )


@dataclass(frozen=True, slots=True)
class _Partial:
    """Per-hypothesis results before the batch statistics (PBO, DSR, multiple testing)."""

    study: Study
    continuous: ContinuousRun
    returns: list[float]
    drawdown_risk: DrawdownRisk | None
    walk_forward: WalkForward
    in_span: TradeSampleStats
    in_span_bootstrap: BootstrapResult
    cost_stress: tuple[ContinuousRun, ...]
    break_even: BreakEven
    benchmarks: tuple[BenchmarkSummary, ...]
    basket: BenchmarkSummary | None
    random_entry: RandomEntrySummary | None
    regimes: RegimeSplits
    notes: tuple[str, ...]


def _analyse(
    study: Study,
    results: Sequence[SimulationResult],
    *,
    protocol: ResearchProtocol,
    window: DateWindow,
    tests: Sequence[DateWindow],
    data: BacktestData,
    bars: Mapping[str, Sequence[DailyBar]],
    regimes: Mapping[date, str],
    starting_cash: Decimal,
) -> _Partial:
    config = study.loaded.config
    confidence = protocol.bootstrap.confidence
    base_bps = protocol.costs.base_bps_per_side
    multipliers = (Decimal(1), *protocol.costs.stress_multipliers)
    runs: list[ContinuousRun] = []
    returns: list[float] = []
    for number, (multiplier, result) in enumerate(zip(multipliers, results, strict=False)):
        run, series = _continuous_run(
            result,
            label="continuous" if number == 0 else f"continuous_x{_multiplier_text(multiplier)}",
            halts="disabled",
            window=window,
            multiplier=multiplier,
            base_bps=base_bps,
            starting_cash=starting_cash,
            confidence=confidence,
        )
        runs.append(run)
        if number == 0:
            returns = series
    continuous_result = results[0]
    wf_results = results[len(multipliers) :]
    pooled = [t for r in wf_results for t in r.trades]
    walk_forward = WalkForward(
        train_months=protocol.walk_forward.train_months,
        test_months=protocol.walk_forward.test_months,
        windows=tuple(
            WindowSummary(
                label=f"walk_forward_{index}",
                window=test,
                trades=len(result.trades),
                expectancy_r=trade_sample_stats(result.trades, confidence=confidence).expectancy_r,
                net_pnl=float(sum((t.net_pnl for t in result.trades), Decimal(0))),
                open_trades_at_end=result.counters.open_trades_at_end,
                halt_rejections={
                    code: result.counters.rejections.get(code, 0) for code in HALT_CODES
                },
            )
            for index, (test, result) in enumerate(zip(tests, wf_results, strict=True))
        ),
        pooled=trade_sample_stats(pooled, confidence=confidence),
        bootstrap=_bootstrap(pooled, protocol),
        open_trades_at_end=sum(r.counters.open_trades_at_end for r in wf_results),
    )
    span_start = tests[0].start if tests else window.end
    span_trades = [
        t for t in continuous_result.trades if t.entry_filled_at_utc.date() >= span_start
    ]
    notes: list[str] = []
    symbols = tuple(config.universe.whitelist or ())
    benchmarks, basket = _benchmarks(
        bars, symbols, window, starting_cash=starting_cash, cost_bps=base_bps, notes=notes
    )
    return _Partial(
        study=study,
        continuous=runs[0],
        returns=returns,
        drawdown_risk=_drawdown_risk(returns, protocol, config),
        walk_forward=walk_forward,
        in_span=trade_sample_stats(span_trades, confidence=confidence),
        in_span_bootstrap=_bootstrap(span_trades, protocol),
        cost_stress=tuple(runs[1:]),
        break_even=break_even_cost(
            [(float(r.cost_bps_per_side), r.trades.expectancy_r) for r in runs]
        ),
        benchmarks=benchmarks,
        basket=basket,
        random_entry=_random_entry(bars, continuous_result, window, protocol, notes),
        regimes=_regime_splits(continuous_result.trades, data.sessions, regimes, confidence),
        notes=tuple(notes),
    )


def _gate_trades(partial: _Partial, sample: GateSample) -> tuple[TradeSampleStats, BootstrapResult]:
    """Statistics and bootstrap of the protocol's per-trade gate sample."""
    if sample == "continuous_walk_forward_span":
        return partial.in_span, partial.in_span_bootstrap
    return partial.walk_forward.pooled, partial.walk_forward.bootstrap


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


@dataclass(frozen=True, slots=True)
class _Batch:
    n_trials: int
    variance: float | None
    variance_source: str
    pbo: PBOResult | None
    pbo_by_family: dict[str, PBOResult]
    notes: tuple[str, ...]


def _trials_variance(
    partials: Sequence[_Partial], existing: Sequence[TrialRecord]
) -> tuple[float | None, str]:
    batch = [p.continuous.moments.sharpe for p in partials]
    sharpes = [s for s in batch if s is not None]
    if len(sharpes) >= 2:
        return statistics.variance(sharpes), (
            f"sample variance (ddof 1) of the {len(sharpes)} per-session Sharpe ratios of "
            "this batch"
        )
    in_batch = {(p.study.hypothesis.id, p.study.loaded.config_hash) for p in partials}
    prior: dict[tuple[str, str], float] = {}
    for record in existing:
        key = (record.hypothesis_id, record.config_hash)
        value = record.metrics.get("sharpe_daily")
        if record.mode == "dev" and key not in in_batch and isinstance(value, float):
            prior[key] = value  # the latest record of a trial wins
    combined = sharpes + [prior[k] for k in sorted(prior)]
    if len(combined) >= 2:
        return statistics.variance(combined), (
            f"sample variance (ddof 1) of {len(sharpes)} per-session Sharpe ratio(s) of this "
            f"batch and {len(prior)} prior dev trial(s) of the registry (batch < 2)"
        )
    return None, "undefined: fewer than 2 per-session Sharpe ratios (batch and registry)"


def _pbo(
    partials: Sequence[_Partial], blocks: int, notes: list[str], label: str
) -> PBOResult | None:
    if len(partials) < 2:
        return None
    lengths = {len(p.returns) for p in partials}
    if len(lengths) != 1:
        notes.append(f"PBO {label}: daily-return series are not aligned ({sorted(lengths)})")
        return None
    rows = lengths.pop()
    if rows < 2 * blocks:
        notes.append(f"PBO {label}: {rows} sessions are fewer than 2 x {blocks} blocks")
        return None
    matrix = [[p.returns[t] for p in partials] for t in range(rows)]
    return probability_of_backtest_overfitting(matrix, n_blocks=blocks)


def _batch_statistics(
    partials: Sequence[_Partial],
    existing: Sequence[TrialRecord],
    protocol: ResearchProtocol,
) -> _Batch:
    notes: list[str] = []
    n_trials = trials_after(
        existing,
        [(p.study.hypothesis.id, p.study.loaded.config_hash) for p in partials],
        protocol.v1_trials,
    )
    variance, source = _trials_variance(partials, existing)
    blocks = protocol.pbo.blocks
    pbo = _pbo(partials, blocks, notes, "batch")
    if pbo is None and len(partials) < 2:
        notes.append("PBO needs at least 2 configurations in the batch: n/a (gate fails)")
    families: dict[str, list[_Partial]] = {}
    for partial in partials:
        families.setdefault(partial.study.hypothesis.family, []).append(partial)
    by_family: dict[str, PBOResult] = {}
    for family in sorted(families):
        members = families[family]
        if len(members) >= 2:
            result = _pbo(members, blocks, notes, f"family {family}")
            if result is not None:
                by_family[family] = result
    return _Batch(
        n_trials=n_trials,
        variance=variance,
        variance_source=source,
        pbo=pbo,
        pbo_by_family=by_family,
        notes=tuple(notes),
    )


def _deflated(partial: _Partial, batch: _Batch) -> DeflatedSharpe | None:
    moments = partial.continuous.moments
    if (
        moments.sharpe is None
        or moments.skewness is None
        or moments.kurtosis is None
        or batch.variance is None
    ):
        return None
    try:
        return deflated_sharpe_ratio(
            moments.sharpe,
            moments.n,
            moments.skewness,
            moments.kurtosis,
            n_trials=batch.n_trials,
            trials_sr_variance=batch.variance,
        )
    except ValueError:
        return None


def _dev_metrics(
    partial: _Partial,
    batch: _Batch,
    deflated: DeflatedSharpe | None,
    testing: MultipleTesting,
    protocol: ResearchProtocol,
) -> dict[str, MetricValue]:
    wf = partial.walk_forward
    gate_stats, gate_boot = _gate_trades(partial, protocol.gate_sample)
    perf = partial.continuous.performance
    basket = None if partial.basket is None else partial.basket.performance
    risk = partial.drawdown_risk
    family = partial.study.hypothesis.family
    family_pbo = batch.pbo_by_family.get(family)
    metrics: dict[str, MetricValue] = {
        "expectancy_r": gate_stats.expectancy_r,
        "t_stat": gate_stats.t_stat,
        "bootstrap_expectancy_ci_low": gate_boot.expectancy_ci_low,
        "profit_factor": gate_stats.profit_factor,
        "mc_max_drawdown_p95": None if risk is None else risk.p95,
        "deflated_sharpe_ratio": None if deflated is None else deflated.dsr,
        "pbo": None if batch.pbo is None else batch.pbo.pbo,
        "random_entry_percentile": (
            None if partial.random_entry is None else partial.random_entry.expectancy_percentile
        ),
        "sharpe_minus_bh_sharpe": _difference(
            perf.sharpe, None if basket is None else basket.sharpe
        ),
        "max_drawdown_over_bh_max_drawdown": _ratio(
            perf.max_drawdown, None if basket is None else basket.max_drawdown
        ),
        "cagr_over_bh_cagr": _ratio(perf.cagr, None if basket is None else basket.cagr),
        "bootstrap_expectancy_ci_high": gate_boot.expectancy_ci_high,
        "bootstrap_profit_factor_ci_low": gate_boot.profit_factor_ci_low,
        "bootstrap_profit_factor_ci_high": gate_boot.profit_factor_ci_high,
        "gate_sample": protocol.gate_sample,
        "gate_trades": gate_stats.trades,
        "wf_trades": wf.pooled.trades,
        "wf_r_trades": wf.pooled.r_trades,
        "wf_expectancy_r": wf.pooled.expectancy_r,
        "wf_t_stat": wf.pooled.t_stat,
        "wf_profit_factor": wf.pooled.profit_factor,
        "span_trades": partial.in_span.trades,
        "span_expectancy_r": partial.in_span.expectancy_r,
        "span_t_stat": partial.in_span.t_stat,
        "span_profit_factor": partial.in_span.profit_factor,
        "wf_open_trades_at_end": wf.open_trades_at_end,
        "p_value": testing.p_value,
        "holm_p_value": testing.holm_p_value,
        "bhy_p_value": testing.bhy_p_value,
        "sharpe": perf.sharpe,
        "sharpe_daily": partial.continuous.moments.sharpe,
        "sortino": perf.sortino,
        "cagr": perf.cagr,
        "max_drawdown": perf.max_drawdown,
        "exposure": perf.exposure,
        "psr": partial.continuous.psr,
        "dsr_sr0": None if deflated is None else deflated.sr0,
        "n_trials": batch.n_trials,
        "trials_sr_variance": batch.variance,
        "mc_max_drawdown_p50": None if risk is None else risk.p50,
        "prob_drawdown_ge_owner_halt": None if risk is None else risk.prob_ge_owner_halt,
        "prob_drawdown_ge_gate_limit": None if risk is None else risk.prob_ge_gate_limit,
        "break_even_cost_bps": partial.break_even.cost_bps_per_side,
        "bh_sharpe": None if basket is None else basket.sharpe,
        "bh_cagr": None if basket is None else basket.cagr,
        "bh_max_drawdown": None if basket is None else basket.max_drawdown,
        "pbo_family": None if family_pbo is None else family_pbo.pbo,
        "continuous_trades": partial.continuous.trades.trades,
        "continuous_expectancy_r": partial.continuous.trades.expectancy_r,
    }
    for multiplier, run in zip(protocol.costs.stress_multipliers, partial.cost_stress, strict=True):
        metrics[cost_metric_name(multiplier)] = run.trades.expectancy_r
    return metrics


def _trial_id(hypothesis_id: str, mode: str, created_at: datetime, config_hash: str) -> str:
    return f"{hypothesis_id}-{mode}-{created_at:%Y%m%dT%H%M%SZ}-{config_hash[:12]}"


def _submit_order(jobs: Sequence[ConfigJob]) -> list[int]:
    """Longest windows first (keeps the pool busy); stable for equal lengths."""

    def length(index: int) -> int:
        window = jobs[index].job.window
        return 0 if window is None else (window[1] - window[0]).days

    return sorted(range(len(jobs)), key=lambda i: -length(i))


def _single_feed(studies: Sequence[Study]) -> DataFeed:
    feeds = {s.loaded.config.market_data.feed for s in studies}
    feed = feeds.pop() if len(feeds) == 1 else None
    if feed is None:
        raise BacktestResearchError(
            "the batch must share one market_data.feed",
            code="HYPOTHESIS_CONFIG_MISMATCH",
        )
    return DataFeed(feed)


async def _daily_bars(
    data: BacktestData, symbols: Iterable[str], last: date
) -> dict[str, list[DailyBar]]:
    """Daily bars of ``symbols`` from the first stored session up to ``last`` only."""
    first = data.sessions[0].session_date
    return {
        symbol: daily_bars_from_bars(await data.feed.get_daily_bars(symbol, first, last))
        for symbol in sorted(set(symbols))
    }


async def run_dev_async(
    *,
    protocol: ResearchProtocol,
    studies: Sequence[Study],
    data_dir: Path,
    starting_cash: Decimal,
    workers: int,
    existing_trials: Sequence[TrialRecord],
    git_status: GitStatus,
    created_at: datetime,
) -> tuple[DevReport, tuple[TrialRecord, ...]]:
    """Development evaluation of ``studies`` (see the module docstring).

    Nothing is written: the caller writes the reports and appends the returned records.

    Raises:
        BacktestResearchError: ``TRIAL_BUDGET_EXCEEDED``, ``LOCKBOX_TOUCHED``,
            ``DEV_WINDOW_MISMATCH``, ``PASS_RULE_INVALID`` / ``PASS_RULE_MISSING``
            (all checked before any simulation).
        NonRetryableError: invalid data.
    """
    if not studies:
        raise BacktestResearchError("no hypothesis selected", code="HYPOTHESIS_UNKNOWN")
    n_trials = trials_after(
        existing_trials,
        [(s.hypothesis.id, s.loaded.config_hash) for s in studies],
        protocol.v1_trials,
    )
    if n_trials > protocol.trial_budget:
        raise BacktestResearchError(
            f"dev refused: the registry would hold {n_trials} trials (v1 included), above "
            f"the budget of {protocol.trial_budget}",
            code="TRIAL_BUDGET_EXCEEDED",
        )
    known = dev_metric_names(protocol)
    windows: list[tuple[DateWindow, tuple[DateWindow, ...]]] = []
    for study in studies:
        rule = study.hypothesis.pass_rule.get("dev")
        if not rule:
            raise BacktestResearchError(
                f"{study.hypothesis.id} has no dev pass rule", code="PASS_RULE_MISSING"
            )
        validate_pass_rule(rule, known, where=f"{study.hypothesis.id}.pass_rule.dev")
        windows.append(dev_windows(study.loaded.config, protocol))
    feed = _single_feed(studies)
    data = load_backtest_data(data_dir, feed=feed)

    multipliers = (Decimal(1), *protocol.costs.stress_multipliers)
    jobs: list[ConfigJob] = []
    spans: list[tuple[int, int]] = []
    for study, (window, tests) in zip(studies, windows, strict=True):
        research = halts_disabled(study.loaded.config)
        first = len(jobs)
        jobs += [
            ConfigJob(research, SimulationJob(m, (window.start, window.end))) for m in multipliers
        ]
        jobs += [ConfigJob(research, SimulationJob(Decimal(1), (t.start, t.end))) for t in tests]
        spans.append((first, len(jobs)))
    results = await run_config_jobs(
        jobs,
        data,
        feed,
        starting_cash=starting_cash,
        workers=workers,
        submit_order=_submit_order(jobs),
    )
    dev = protocol.dev_window
    symbols = [s for study in studies for s in study.loaded.config.universe.whitelist or ()]
    bars = await _daily_bars(data, [*symbols, REGIME_SYMBOL], dev.end)
    regimes = _spy_regimes(bars.get(REGIME_SYMBOL, []))
    partials = [
        _analyse(
            study,
            results[first:last],
            protocol=protocol,
            window=window,
            tests=tests,
            data=data,
            bars=bars,
            regimes=regimes,
            starting_cash=starting_cash,
        )
        for study, (window, tests), (first, last) in zip(studies, windows, spans, strict=True)
    ]
    batch = _batch_statistics(partials, existing_trials, protocol)
    raw_p = [_gate_trades(p, protocol.gate_sample)[0].p_value for p in partials]
    adjusted_inputs = [1.0 if p is None else p for p in raw_p]
    holm, bhy = holm_adjust(adjusted_inputs), bhy_adjust(adjusted_inputs)
    fingerprint = data_fingerprint(data_dir)
    hypotheses: list[HypothesisDevResult] = []
    records: list[TrialRecord] = []
    for index, partial in enumerate(partials):
        study = partial.study
        hypothesis = study.hypothesis
        deflated = _deflated(partial, batch)
        testing = MultipleTesting(
            p_value=raw_p[index], holm_p_value=holm[index], bhy_p_value=bhy[index]
        )
        metrics = _dev_metrics(partial, batch, deflated, testing, protocol)
        gate = evaluate_pass_rule(hypothesis.pass_rule["dev"], metrics, stage="dev")
        metrics["gate_passed"] = gate.passed
        trial_id = _trial_id(hypothesis.id, "dev", created_at, study.loaded.config_hash)
        hypotheses.append(
            HypothesisDevResult(
                hypothesis_id=hypothesis.id,
                family=hypothesis.family,
                title=hypothesis.title,
                config_path=hypothesis.config_path,
                config_hash=study.loaded.config_hash,
                strategy_version=study.loaded.config.strategy_version,
                trial_id=trial_id,
                continuous=partial.continuous,
                drawdown_risk=partial.drawdown_risk,
                walk_forward=partial.walk_forward,
                gate_sample=protocol.gate_sample,
                continuous_in_walk_forward_span=partial.in_span,
                continuous_in_walk_forward_span_bootstrap=partial.in_span_bootstrap,
                cost_stress=partial.cost_stress,
                break_even=partial.break_even,
                benchmarks=partial.benchmarks,
                gate_benchmark=None if partial.basket is None else partial.basket.label,
                random_entry=partial.random_entry,
                regimes=partial.regimes,
                deflated_sharpe=deflated,
                multiple_testing=testing,
                metrics=metrics,
                gate=gate,
                notes=partial.notes,
            )
        )
        records.append(
            TrialRecord(
                trial_id=trial_id,
                hypothesis_id=hypothesis.id,
                config_hash=study.loaded.config_hash,
                data_fingerprint=fingerprint,
                window=TrialWindow(start=dev.start, end=dev.end),
                mode="dev",
                git_commit=git_status.commit,
                git_dirty=git_status.dirty,
                created_at_utc=created_at,
                metrics=metrics,
            )
        )
    report = DevReport(
        mode="dev",
        banner=dev_banner(protocol),
        halts_label=HALTS_DISABLED_LABEL,
        disclaimer=DISCLAIMER,
        protocol=protocol,
        data_fingerprint=fingerprint,
        starting_cash=starting_cash,
        git_commit=git_status.commit,
        git_dirty=git_status.dirty,
        created_at_utc=created_at,
        batch=BatchStatistics(
            hypotheses=tuple(s.hypothesis.id for s in studies),
            n_trials=batch.n_trials,
            trial_budget=protocol.trial_budget,
            sharpe_units=SHARPE_UNITS,
            trials_sr_variance=batch.variance,
            trials_sr_variance_source=batch.variance_source,
            pbo=batch.pbo,
            pbo_by_family=batch.pbo_by_family,
            multiple_testing=(
                "Holm (FWER) and BHY (FDR) adjustments of the one-sided per-trade t-test "
                "p-values of the pooled walk-forward trades across this batch (a missing "
                "p-value counts as 1)"
            ),
            notes=batch.notes,
        ),
        hypotheses=tuple(hypotheses),
        notes=_DEV_NOTES,
    )
    return report, tuple(records)


_DEV_NOTES: Final[tuple[str, ...]] = (
    "Research statistics use research-only copies of the configs with the loss-limit "
    "halts disabled (daily, weekly and max-drawdown limits = 1.0); exposure, position "
    "and open-risk limits are kept. The official run (python -m backtest) is unaffected.",
    "Per-trade criteria (expectancy, t, bootstrap CI, PF, Holm/BHY) use the protocol's "
    "gate_sample: walk_forward_fresh_accounts = pooled walk-forward test windows, each a "
    "fresh account (no parameter is optimised, train windows are informative; a trade "
    "still open at a window end is never closed and not counted, see "
    "wf_open_trades_at_end, which cuts long-holding trend trades); "
    "continuous_walk_forward_span = continuous-run trades entered in the walk-forward "
    "span. Both samples are reported.",
    "Account-level criteria (Sharpe, DSR, Monte-Carlo drawdown, cost stress, random "
    "entry, buy-and-hold, PBO) use the continuous dev-window run. Session marks are the "
    "equity after each session close; exposure = share of sessions ending with a "
    "position.",
    "Buy-and-hold gate benchmark: the equal-weight basket of the hypothesis symbols (no "
    "rebalancing), same bars and cost per side; dividends are not credited (split "
    "adjustment) for the strategy and the benchmarks alike.",
    "Cost stress scales only the simulated broker slippage; sizing keeps "
    "risk.slippage_buffer_bps. Break-even = linear interpolation of the continuous-run "
    "expectancy (R) against the cost per side.",
    "PBO and DSR depend on the batch: rerunning a subset changes them.",
)


# --------------------------------------------------------------------------- lockbox


def _latest_dev_trial(
    trials: Sequence[TrialRecord], hypothesis_id: str, config_hash: str
) -> TrialRecord | None:
    matches = [
        t
        for t in trials
        if t.hypothesis_id == hypothesis_id and t.mode == "dev" and t.config_hash == config_hash
    ]
    return matches[-1] if matches else None


def _float_metric(record: TrialRecord, name: str) -> float | None:
    value = record.metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


async def run_lockbox_async(
    *,
    protocol: ResearchProtocol,
    hypothesis_id: str,
    hypotheses: Sequence[Hypothesis],
    root: Path,
    data_dir: Path,
    starting_cash: Decimal,
    workers: int,
    existing_trials: Sequence[TrialRecord],
    git_status: GitStatus,
    created_at: datetime,
) -> tuple[LockboxReport, TrialRecord]:
    """Open the lockbox once for ``hypothesis_id`` (see the module docstring).

    Raises:
        BacktestResearchError: the lockbox guard codes (``LOCKBOX_NOT_REGISTERED``,
            ``LOCKBOX_NO_PASS_RULE``, ``LOCKBOX_NO_COMMIT``, ``LOCKBOX_DIRTY_TREE``,
            ``LOCKBOX_ALREADY_OPENED``), ``LOCKBOX_NO_DEV_TRIAL`` (no dev trial of this
            config hash with a bootstrap CI), ``PASS_RULE_INVALID``; all before any
            simulation.
    """
    hypothesis = next((h for h in hypotheses if h.id == hypothesis_id), None)
    assert_lockbox_allowed(hypothesis, existing_trials, git_status)
    if hypothesis is None:  # pragma: no cover - refused by assert_lockbox_allowed
        raise BacktestResearchError("lockbox refused", code="LOCKBOX_NOT_REGISTERED")
    rule = hypothesis.pass_rule["lockbox"]
    validate_pass_rule(rule, LOCKBOX_METRICS, where=f"{hypothesis.id}.pass_rule.lockbox")
    study = load_study(hypothesis, root, protocol)
    dev_windows(study.loaded.config, protocol)  # the registered dev config, unchanged
    dev_trial = _latest_dev_trial(existing_trials, hypothesis.id, study.loaded.config_hash)
    ci_low = None if dev_trial is None else _float_metric(dev_trial, "bootstrap_expectancy_ci_low")
    ci_high = (
        None if dev_trial is None else _float_metric(dev_trial, "bootstrap_expectancy_ci_high")
    )
    if dev_trial is None or ci_low is None or ci_high is None:
        raise BacktestResearchError(
            f"lockbox refused: no dev trial of {hypothesis.id} with config sha256 "
            f"{study.loaded.config_hash[:12]} and a bootstrap expectancy CI in the registry",
            code="LOCKBOX_NO_DEV_TRIAL",
        )
    feed = _single_feed([study])
    data = load_backtest_data(data_dir, feed=feed)
    window = protocol.lockbox_window
    official_config = study.loaded.config
    span = (window.start, window.end)
    official_result, stats_result = await run_config_jobs(
        [
            ConfigJob(official_config, SimulationJob(Decimal(1), span)),
            ConfigJob(halts_disabled(official_config), SimulationJob(Decimal(1), span)),
        ],
        data,
        feed,
        starting_cash=starting_cash,
        workers=workers,
    )
    confidence = protocol.bootstrap.confidence
    base_bps = protocol.costs.base_bps_per_side
    official, _ = _continuous_run(
        official_result,
        label="lockbox_official",
        halts="enabled",
        window=window,
        multiplier=Decimal(1),
        base_bps=base_bps,
        starting_cash=starting_cash,
        confidence=confidence,
    )
    stats_run, _ = _continuous_run(
        stats_result,
        label="lockbox_statistics",
        halts="disabled",
        window=window,
        multiplier=Decimal(1),
        base_bps=base_bps,
        starting_cash=starting_cash,
        confidence=confidence,
    )
    expectancy = official.trades.expectancy_r
    metrics: dict[str, MetricValue] = {
        "expectancy_r": expectancy,
        "profit_factor": official.trades.profit_factor,
        "max_drawdown": official.performance.max_drawdown,
        "expectancy_r_inside_dev_bootstrap_ci": (
            None if expectancy is None else ci_low <= expectancy <= ci_high
        ),
        "trades": official.trades.trades,
        "t_stat": official.trades.t_stat,
        "sharpe": official.performance.sharpe,
        "cagr": official.performance.cagr,
        "stats_trades": stats_run.trades.trades,
        "stats_expectancy_r": stats_run.trades.expectancy_r,
        "stats_profit_factor": stats_run.trades.profit_factor,
        "stats_max_drawdown": stats_run.performance.max_drawdown,
        "stats_sharpe": stats_run.performance.sharpe,
        "stats_cagr": stats_run.performance.cagr,
    }
    gate = evaluate_pass_rule(rule, metrics, stage="lockbox")
    metrics["gate_passed"] = gate.passed
    fingerprint = data_fingerprint(data_dir)
    trial_id = _trial_id(hypothesis.id, "lockbox", created_at, study.loaded.config_hash)
    notes = [
        "The pass rule is evaluated on the official run (loss-limit halts enabled); the "
        f"statistics run ({HALTS_DISABLED_LABEL}) is informative.",
        "Expectancy inside the dev CI: the official lockbox expectancy lies in the dev "
        "trial's block-bootstrap expectancy interval.",
    ]
    if fingerprint != dev_trial.data_fingerprint:
        notes.append(
            "The dataset fingerprint differs from the dev trial's: the development bars may "
            "not be the ones evaluated in development."
        )
    report = LockboxReport(
        mode="lockbox",
        banner=_lockbox_banner(protocol, hypothesis.id),
        disclaimer=DISCLAIMER,
        protocol=protocol,
        hypothesis_id=hypothesis.id,
        family=hypothesis.family,
        title=hypothesis.title,
        config_hash=study.loaded.config_hash,
        strategy_version=study.loaded.config.strategy_version,
        data_fingerprint=fingerprint,
        starting_cash=starting_cash,
        git_commit=git_status.commit,
        git_dirty=git_status.dirty,
        created_at_utc=created_at,
        dev_trial_id=dev_trial.trial_id,
        dev_data_fingerprint=dev_trial.data_fingerprint,
        dev_expectancy_ci=(ci_low, ci_high),
        dev_gate_passed=dev_trial.metrics.get("gate_passed"),
        official=official,
        statistics=stats_run,
        metrics=metrics,
        gate=gate,
        trial_id=trial_id,
        notes=tuple(notes),
    )
    record = TrialRecord(
        trial_id=trial_id,
        hypothesis_id=hypothesis.id,
        config_hash=study.loaded.config_hash,
        data_fingerprint=fingerprint,
        window=TrialWindow(start=window.start, end=window.end),
        mode="lockbox",
        git_commit=git_status.commit,
        git_dirty=git_status.dirty,
        created_at_utc=created_at,
        metrics=metrics,
    )
    return report, record


# --------------------------------------------------------------------------- text


def _fmt(value: MetricValue | Decimal, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | str):
        return str(value)
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _pct(value: float | Decimal | None) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def _row_lines(row: GateRow, indent: str) -> list[str]:
    verdict = "PASS" if row.passed else "FAIL"
    if row.op != "any_of":
        return [
            f"{indent}{row.name} {row.op} {_fmt(row.threshold)}: value {_fmt(row.value)} "
            f"-> {verdict}"
        ]
    lines = [f"{indent}{row.name} (any of) -> {verdict}"]
    for number, alternative in enumerate(row.alternatives, start=1):
        passed = all(r.passed for r in alternative)
        lines.append(f"{indent}  alternative {number} -> {'PASS' if passed else 'FAIL'}")
        for child in alternative:
            lines += _row_lines(child, indent + "    ")
    return lines


def _gate_lines(gate: GateResult, indent: str) -> list[str]:
    lines: list[str] = []
    for row in gate.rows:
        lines += _row_lines(row, indent)
    lines.append(f"{indent}overall -> {'PASS' if gate.passed else 'FAIL'}")
    return lines


def _stats_text(stats: TradeSampleStats) -> str:
    return (
        f"trades {stats.trades} (R {stats.r_trades}), E[R] {_fmt(stats.expectancy_r)}, "
        f"t {_fmt(stats.t_stat, 2)}, p {_fmt(stats.p_value)}, 95% t-CI "
        f"[{_fmt(stats.ci_low)}, {_fmt(stats.ci_high)}], PF {_fmt(stats.profit_factor, 3)}, "
        f"win rate {_pct(stats.win_rate)}, net P&L {stats.net_pnl:.2f}"
    )


def _run_text(run: ContinuousRun) -> str:
    perf = run.performance
    return (
        f"{run.window.start}..{run.window.end} ({run.sessions} sessions, "
        f"{run.cost_bps_per_side} bps/side): return {_pct(perf.total_return)}, CAGR "
        f"{_pct(perf.cagr)}, Sharpe {_fmt(perf.sharpe, 3)}, Sortino {_fmt(perf.sortino, 3)}, "
        f"max DD {_pct(perf.max_drawdown)}, exposure {_pct(perf.exposure)}, PSR "
        f"{_fmt(run.psr)}"
    )


def _failed(gate: GateResult) -> str:
    names = [row.name for row in gate.rows if not row.passed]
    return ", ".join(names) if names else "-"


def _bootstrap_text(boot: BootstrapResult) -> str:
    return (
        f"block bootstrap (ISO weeks, {boot.n_blocks} blocks, {boot.n_resamples} resamples):"
        f" E[R] CI [{_fmt(boot.expectancy_ci_low)}, {_fmt(boot.expectancy_ci_high)}], "
        f"P(E <= 0) {_fmt(boot.prob_expectancy_le_zero)}, PF CI "
        f"[{_fmt(boot.profit_factor_ci_low, 3)}, {_fmt(boot.profit_factor_ci_high, 3)}]"
    )


def _primary(result: HypothesisDevResult, sample: GateSample) -> str:
    return " — PRIMARY gate sample" if result.gate_sample == sample else " (informative)"


def _hypothesis_lines(result: HypothesisDevResult) -> list[str]:
    wf = result.walk_forward
    lines = [
        "",
        f"[{result.hypothesis_id}] {result.family}: {result.title}",
        f"  strategy {result.strategy_version}, config sha256 {result.config_hash[:12]}, "
        f"trial {result.trial_id}",
        f"  continuous dev run ({HALTS_DISABLED_LABEL}): {_run_text(result.continuous)}",
        f"    {_stats_text(result.continuous.trades)}; open at end "
        f"{result.continuous.open_trades_at_end}; halt rejections "
        f"{result.continuous.halt_rejections}",
    ]
    risk = result.drawdown_risk
    if risk is not None:
        lines.append(
            f"  Monte-Carlo max DD (stationary bootstrap, mean block {risk.mean_block_length:g}"
            f" sessions, {risk.n_sims} sims): observed {_pct(risk.observed)}, p50 "
            f"{_pct(risk.p50)}, p95 {_pct(risk.p95)}; P(DD >= {_pct(risk.owner_halt)} owner "
            f"halt) {_fmt(risk.prob_ge_owner_halt)}; P(DD >= {_pct(risk.gate_limit)}) "
            f"{_fmt(risk.prob_ge_gate_limit)}"
        )
    lines += [
        f"  walk-forward OOS ({len(wf.windows)} fresh-account test windows, train "
        f"{wf.train_months} / test {wf.test_months} months, {HALTS_DISABLED_LABEL})"
        f"{_primary(result, 'walk_forward_fresh_accounts')}:",
        f"    {_stats_text(wf.pooled)}",
        f"    {_bootstrap_text(wf.bootstrap)}",
        f"    trades open at a window end (not counted): {wf.open_trades_at_end}",
        "    windows: "
        + "; ".join(
            f"{w.window.start}..{w.window.end} {w.trades} tr E {_fmt(w.expectancy_r, 3)}"
            for w in wf.windows
        ),
        "  continuous-run trades entered in the walk-forward span"
        f"{_primary(result, 'continuous_walk_forward_span')}:",
        f"    {_stats_text(result.continuous_in_walk_forward_span)}",
        f"    {_bootstrap_text(result.continuous_in_walk_forward_span_bootstrap)}",
        "  cost stress (continuous run):",
        f"    x1 ({result.continuous.cost_bps_per_side} bps): E[R] "
        f"{_fmt(result.continuous.trades.expectancy_r)}, PF "
        f"{_fmt(result.continuous.trades.profit_factor, 3)}, return "
        f"{_pct(result.continuous.performance.total_return)}",
    ]
    for run in result.cost_stress:
        lines.append(
            f"    x{_multiplier_text(run.slippage_multiplier)} ({run.cost_bps_per_side} bps): "
            f"E[R] {_fmt(run.trades.expectancy_r)}, PF {_fmt(run.trades.profit_factor, 3)}, "
            f"return {_pct(run.performance.total_return)}"
        )
    lines.append(
        f"    break-even cost: {_fmt(result.break_even.cost_bps_per_side, 2)} bps per side "
        f"({result.break_even.method})"
    )
    lines.append(
        f"  buy-and-hold (same bars, {result.continuous.cost_bps_per_side} bps/side; gate: "
        f"{result.gate_benchmark or 'n/a'}):"
    )
    for bench in result.benchmarks:
        perf = bench.performance
        lines.append(
            f"    {bench.label}: CAGR {_pct(perf.cagr)}, Sharpe {_fmt(perf.sharpe, 3)}, "
            f"max DD {_pct(perf.max_drawdown)}"
        )
    entry = result.random_entry
    if entry is not None:
        q5, q50, q95 = entry.expectancy_quantiles
        lines.append(
            f"  random entry ({entry.n_sims} sims, {entry.n_trades} trades): real mean return "
            f"{_pct(entry.real_expectancy)} vs p5/p50/p95 {_pct(q5)}/{_pct(q50)}/{_pct(q95)} "
            f"-> percentile {entry.expectancy_percentile:.1f}"
        )
    regimes = result.regimes
    for title, rows in (
        ("years", regimes.by_year),
        ("symbols", regimes.by_symbol),
        (f"{REGIME_SYMBOL} vs SMA{REGIME_SMA_PERIOD}", regimes.by_spy_sma200),
    ):
        lines.append(
            f"  regimes by {title} ({regimes.sample}): "
            + "; ".join(
                f"{r.key} {r.stats.trades} tr E {_fmt(r.stats.expectancy_r, 3)} "
                f"PF {_fmt(r.stats.profit_factor, 2)}"
                for r in rows
            )
        )
    deflated = result.deflated_sharpe
    testing = result.multiple_testing
    lines += [
        f"  deflated Sharpe (per-session): SR {_fmt(None if deflated is None else deflated.sr)}"
        f", SR0 {_fmt(None if deflated is None else deflated.sr0)}, N "
        f"{_fmt(None if deflated is None else deflated.n_trials, 0)}, DSR "
        f"{_fmt(None if deflated is None else deflated.dsr)}",
        f"  multiple testing: p {_fmt(testing.p_value)}, Holm {_fmt(testing.holm_p_value)}, "
        f"BHY {_fmt(testing.bhy_p_value)}",
        "  gate (pass_rule.dev):",
        *_gate_lines(result.gate, "    "),
    ]
    lines += [f"  note: {note}" for note in result.notes]
    return lines


def render_dev_text(report: DevReport) -> str:
    """Readable development report (banner first and last)."""
    batch = report.batch
    commit = (report.git_commit or "unknown")[:12]
    lines = [
        report.banner,
        f"Research statistics: {report.halts_label} (exposure and position limits kept).",
        f"protocol {report.protocol.protocol_version}, data {report.data_fingerprint[:12]}, "
        f"cash {report.starting_cash} per account, git {commit} "
        f"({'dirty' if report.git_dirty else 'clean'}), "
        f"created {report.created_at_utc:%Y-%m-%dT%H:%M:%SZ}",
        f"trials N = {batch.n_trials} of budget {batch.trial_budget} (registry incl. this "
        f"batch, v1 counted); DSR trial variance V = {_fmt(batch.trials_sr_variance, 8)} "
        f"({batch.trials_sr_variance_source}; {batch.sharpe_units})",
        f"PBO (CSCV, {report.protocol.pbo.blocks} blocks) across the batch: "
        f"{_fmt(None if batch.pbo is None else batch.pbo.pbo)}"
        + "".join(
            f"; {family}: {_fmt(result.pbo)}" for family, result in batch.pbo_by_family.items()
        ),
        f"per-trade gate sample: {report.protocol.gate_sample}",
        "",
        "Gate summary (pass_rule.dev):",
    ]
    for result in report.hypotheses:
        m = result.metrics
        lines.append(
            f"  {result.hypothesis_id:<20} {'PASS' if result.gate.passed else 'FAIL'}  "
            f"gate trades {_fmt(m.get('gate_trades'))}, E[R] {_fmt(m.get('expectancy_r'))}, "
            f"t {_fmt(m.get('t_stat'), 2)}, PF {_fmt(m.get('profit_factor'), 3)}, DSR "
            f"{_fmt(m.get('deflated_sharpe_ratio'), 3)}; failed: {_failed(result.gate)}"
        )
    for result in report.hypotheses:
        lines += _hypothesis_lines(result)
    lines += ["", *batch.notes, *report.notes, report.disclaimer, "", report.banner]
    return "\n".join(lines)


def render_lockbox_text(report: LockboxReport) -> str:
    """Readable lockbox report (banner first and last)."""
    low, high = report.dev_expectancy_ci
    lines = [
        report.banner,
        f"[{report.hypothesis_id}] {report.family}: {report.title}",
        f"strategy {report.strategy_version}, config sha256 {report.config_hash[:12]}, data "
        f"{report.data_fingerprint[:12]}, git {(report.git_commit or 'unknown')[:12]}, "
        f"trial {report.trial_id}",
        f"dev trial {report.dev_trial_id}: gate passed {_fmt(report.dev_gate_passed)}, "
        f"bootstrap E[R] CI [{_fmt(low)}, {_fmt(high)}]",
        f"official run (halts enabled): {_run_text(report.official)}",
        f"  {_stats_text(report.official.trades)}; halt rejections "
        f"{report.official.halt_rejections}",
        f"statistics run ({HALTS_DISABLED_LABEL}): {_run_text(report.statistics)}",
        f"  {_stats_text(report.statistics.trades)}",
        "gate (pass_rule.lockbox):",
        *_gate_lines(report.gate, "  "),
        "",
        *report.notes,
        report.disclaimer,
        "",
        report.banner,
    ]
    return "\n".join(lines)


def _list_text(
    hypotheses: Sequence[Hypothesis], trials: Sequence[TrialRecord], registry: Path, budget: int
) -> str:
    lines = [f"Registered hypotheses: {len(hypotheses)}"]
    for h in hypotheses:
        dev = [t for t in trials if t.hypothesis_id == h.id and t.mode == "dev"]
        opened = any(t.hypothesis_id == h.id and t.mode == "lockbox" for t in trials)
        last = "-" if not dev else _fmt(dev[-1].metrics.get("gate_passed"))
        lines.append(
            f"  {h.id:<20} {h.status:<15} {h.family:<24} dev trials {len(dev)} (last gate "
            f"passed {last}), lockbox {'OPENED' if opened else 'closed'}"
        )
    used = trials_after(trials, (), V1_TRIAL_COUNT)
    lines.append(
        f"Trials used: {used} of {budget} (distinct (hypothesis, config_hash) in {registry} "
        f"+ {V1_TRIAL_COUNT} for strategy v1.0.0)"
    )
    return "\n".join(lines)


def _report_text(trials: Sequence[TrialRecord], registry: Path, budget: int) -> str:
    used = trials_after(trials, (), V1_TRIAL_COUNT)
    lines = [f"Trial registry {registry}: {len(trials)} records, {used} trials of {budget}"]
    for t in trials:
        m = t.metrics
        lines.append(
            f"  {t.trial_id} [{t.mode}] {t.window.start}..{t.window.end} commit "
            f"{(t.git_commit or 'unknown')[:12]}{' DIRTY' if t.git_dirty else ''} "
            f"{t.created_at_utc:%Y-%m-%dT%H:%M:%SZ}: gate {_fmt(m.get('gate_passed'))}, "
            f"E[R] {_fmt(m.get('expectancy_r'))}, PF {_fmt(m.get('profit_factor'), 3)}, "
            f"t {_fmt(m.get('t_stat'), 2)}, DSR {_fmt(m.get('deflated_sharpe_ratio'), 3)}, "
            f"max DD {_pct(_metric_float(m.get('max_drawdown')))}"
        )
    return "\n".join(lines)


def _metric_float(value: MetricValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


# --------------------------------------------------------------------------- CLI


def _decimal(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal {text!r}") from exc
    if not value.is_finite() or value <= 0:
        raise argparse.ArgumentTypeError(f"expected a finite positive amount, got {text!r}")
    return value


def _workers(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid worker count {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected at least 1 worker, got {text!r}")
    return value


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    common.add_argument("--hypotheses-dir", type=Path, default=DEFAULT_HYPOTHESES_DIR)
    common.add_argument("--registry", type=Path, default=DEFAULT_TRIALS_PATH)
    common.add_argument(
        "--root", type=Path, default=Path(), help="base of relative config paths (default: cwd)"
    )
    runs = argparse.ArgumentParser(add_help=False)
    runs.add_argument("--data", type=Path, required=True, help="dataset directory")
    runs.add_argument(
        "--starting-cash", type=_decimal, required=True, help="initial capital (no default)"
    )
    runs.add_argument(
        "--workers",
        type=_workers,
        default=_default_workers(),
        help="processes (default: %(default)s); results do not depend on it",
    )
    runs.add_argument("--out", type=Path, default=None, help="directory for the JSON/text report")
    parser = argparse.ArgumentParser(
        prog="python -m backtest.research_cli",
        description="Research protocol of strategy family v2 (research/README.md).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", parents=[common], help="hypotheses, status, trials used / budget")
    sub.add_parser("report", parents=[common], help="summary of the recorded trials")
    dev = sub.add_parser(
        "dev", parents=[common, runs], help="development evaluation (dev window only)"
    )
    dev.add_argument("--hypotheses", default="all", help="'all' or comma-separated ids")
    lockbox = sub.add_parser(
        "lockbox", parents=[common, runs], help="open the lockbox ONCE for one hypothesis"
    )
    lockbox.add_argument("--hypothesis", required=True)
    return parser


def _write(out: Path | None, stem: str, payload: str, text: str) -> list[str]:
    if out is None:
        return []
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(payload, encoding="utf-8")
    (out / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    return [f"Reports written to {out / (stem + '.json')} and {out / (stem + '.txt')}"]


def _append(registry: Path, records: Sequence[TrialRecord], protocol: ResearchProtocol) -> None:
    existing = read_trials(registry)
    total = trials_after(
        existing, [(r.hypothesis_id, r.config_hash) for r in records], protocol.v1_trials
    )
    if total > protocol.trial_budget:  # the registry changed during the run
        raise BacktestResearchError(
            f"refused to record: {total} trials would exceed the budget {protocol.trial_budget}",
            code="TRIAL_BUDGET_EXCEEDED",
        )
    for record in records:
        append_trial(registry, record)


def _cmd_dev(args: argparse.Namespace, git: GitRunner, now: Callable[[], datetime]) -> str:
    protocol = load_protocol(args.protocol)
    hypotheses = select_hypotheses(load_hypotheses(args.hypotheses_dir), args.hypotheses)
    studies = tuple(load_study(h, args.root, protocol) for h in hypotheses)
    report, records = asyncio.run(
        run_dev_async(
            protocol=protocol,
            studies=studies,
            data_dir=args.data,
            starting_cash=args.starting_cash,
            workers=args.workers,
            existing_trials=read_trials(args.registry),
            git_status=read_git_status(git),
            created_at=now(),
        )
    )
    text = render_dev_text(report)
    written = _write(args.out, "dev_report", report.model_dump_json(indent=2), text)
    _append(args.registry, records, protocol)
    return "\n".join(
        [text, "", *written, f"{len(records)} trial records appended to {args.registry}"]
    )


def _cmd_lockbox(args: argparse.Namespace, git: GitRunner, now: Callable[[], datetime]) -> str:
    protocol = load_protocol(args.protocol)
    hypotheses = load_hypotheses(args.hypotheses_dir)
    report, record = asyncio.run(
        run_lockbox_async(
            protocol=protocol,
            hypothesis_id=args.hypothesis,
            hypotheses=hypotheses,
            root=args.root,
            data_dir=args.data,
            starting_cash=args.starting_cash,
            workers=args.workers,
            existing_trials=read_trials(args.registry),
            git_status=read_git_status(git),
            created_at=now(),
        )
    )
    text = render_lockbox_text(report)
    stem = f"lockbox_{args.hypothesis}"
    written = _write(args.out, stem, report.model_dump_json(indent=2), text)
    append_trial(args.registry, record)
    return "\n".join([text, "", *written, f"lockbox trial appended to {args.registry}"])


def main(
    argv: Sequence[str] | None = None,
    *,
    git_runner: GitRunner | None = None,
    now: Callable[[], datetime] | None = None,
) -> int:
    """Run the research CLI; ``git_runner`` / ``now`` are injectable for tests."""
    args = _parser().parse_args(argv)
    git = git_runner or subprocess_git(args.root)
    clock = now or (lambda: datetime.now(UTC))
    try:
        if args.command == "list":
            text = _list_text(
                load_hypotheses(args.hypotheses_dir),
                read_trials(args.registry),
                args.registry,
                load_protocol(args.protocol).trial_budget,
            )
        elif args.command == "report":
            text = _report_text(
                read_trials(args.registry),
                args.registry,
                load_protocol(args.protocol).trial_budget,
            )
        elif args.command == "dev":
            text = _cmd_dev(args, git, clock)
        else:
            text = _cmd_lockbox(args, git, clock)
    except BacktestRefusedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ConfigError, NonRetryableError) as exc:
        print(f"research {args.command} refused: {exc}", file=sys.stderr)
        return 2
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")  # the banners are not ASCII
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
