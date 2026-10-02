"""Hypothesis registry, append-only trial log and lockbox guard (plan WU3, research/README.md).

* A **hypothesis** is pre-registered in ``research/hypotheses/<id>.yaml`` before any
  run: rationale, sources, exact rules, costs, config path and pass rule.
* Every evaluated run is a **trial** appended as one JSON line to the trial log
  (``research/trials.jsonl`` by default; the path is always passed in). Records are
  never rewritten. Times come from the caller (CLI layer): nothing here reads the clock.
* :func:`count_trials` gives N for the deflated Sharpe ratio: distinct
  ``(hypothesis_id, config_hash)`` pairs in the log plus strategy v1.0.0
  (:data:`V1_TRIAL_COUNT`), which is never written to the log.
* :func:`assert_lockbox_allowed` refuses to open the lockbox unless the hypothesis is
  registered with a lockbox pass rule, the git tree is clean at a recorded commit and
  the lockbox was never opened for that hypothesis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from domain.errors import NonRetryableError
from domain.models import UtcDatetime

__all__ = [
    "DEFAULT_HYPOTHESES_DIR",
    "DEFAULT_TRIALS_PATH",
    "TRIAL_BUDGET",
    "V1_TRIAL_COUNT",
    "BacktestResearchError",
    "CostAssumption",
    "GitStatus",
    "Hypothesis",
    "HypothesisStatus",
    "Source",
    "TrialMode",
    "TrialRecord",
    "TrialWindow",
    "append_trial",
    "assert_lockbox_allowed",
    "count_trials",
    "load_hypotheses",
    "load_hypothesis",
    "read_trials",
]

DEFAULT_HYPOTHESES_DIR: Final = Path("research/hypotheses")
DEFAULT_TRIALS_PATH: Final = Path("research/trials.jsonl")

TRIAL_BUDGET: Final = 12
"""Owner decision 2026-10-02: at most 12 registered configurations in total (v1 included)."""

V1_TRIAL_COUNT: Final = 1
"""Strategy v1.0.0 (5-min EMA trend, rejected 2026-10-02, DECISIONS.md) counts as one
trial of the research history; it is not a record of the log."""

HypothesisStatus = Literal["registered", "finalist", "rejected", "lockbox_passed", "lockbox_failed"]
TrialMode = Literal["dev", "lockbox"]
MetricValue = float | int | str | bool | None


class BacktestResearchError(NonRetryableError):
    """The research protocol refuses the request (invalid registry or lockbox guard)."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Source(_Model):
    """A published source behind a hypothesis."""

    citation: str = Field(min_length=1)
    url: str

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError(f"source url must be http(s), got {value!r}")
        return value


class CostAssumption(_Model):
    """Per-side cost of the evaluation and the stress multipliers."""

    base_bps_per_side: Decimal = Field(ge=0)
    stress_multipliers: tuple[Decimal, ...] = Field(min_length=1)


class Hypothesis(_Model):
    """A pre-registered hypothesis (``research/hypotheses/<id>.yaml``)."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    family: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    sources: tuple[Source, ...] = Field(min_length=1)
    rules: str = Field(min_length=1)
    symbols: tuple[str, ...] = Field(min_length=1)
    costs: CostAssumption
    config_path: str = Field(min_length=1)
    created_at: date
    pass_rule: dict[str, Any] = Field(default_factory=dict)
    """Criteria by stage: ``dev`` (development gate) and ``lockbox`` (written before opening)."""
    status: HypothesisStatus = "registered"


class TrialWindow(_Model):
    """Inclusive date window of a trial."""

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("window end must not be before its start")
        return self


class TrialRecord(_Model):
    """One evaluated run, as stored in the trial log (one JSON object per line)."""

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    trial_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    data_fingerprint: str = Field(min_length=1)
    window: TrialWindow
    mode: TrialMode
    git_commit: str | None
    git_dirty: bool
    created_at_utc: UtcDatetime
    metrics: dict[str, MetricValue] = Field(default_factory=dict)


@dataclass(frozen=True)
class GitStatus:
    """State of the working tree, supplied by the CLI layer (``git rev-parse`` / ``status``)."""

    commit: str | None
    dirty: bool


# --------------------------------------------------------------------------- hypotheses


def load_hypothesis(path: Path) -> Hypothesis:
    """Load and validate one hypothesis file; its stem must equal its ``id``."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        hypothesis = Hypothesis.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise BacktestResearchError(
            f"invalid hypothesis file {path}: {exc}", code="HYPOTHESIS_INVALID"
        ) from exc
    if hypothesis.id != path.stem:
        raise BacktestResearchError(
            f"hypothesis id {hypothesis.id!r} does not match its file name {path.name}",
            code="HYPOTHESIS_INVALID",
        )
    return hypothesis


def load_hypotheses(directory: Path) -> tuple[Hypothesis, ...]:
    """Every ``*.yaml`` hypothesis of ``directory``, sorted by id (ids are unique by
    construction: one file per id)."""
    return tuple(load_hypothesis(path) for path in sorted(directory.glob("*.yaml")))


# --------------------------------------------------------------------------- trial log


def read_trials(path: Path) -> tuple[TrialRecord, ...]:
    """Records of the trial log in file order (empty when the file does not exist)."""
    if not path.exists():
        return ()
    records: list[TrialRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(TrialRecord.model_validate_json(line))
        except ValidationError as exc:
            raise BacktestResearchError(
                f"{path}:{number}: invalid trial record: {exc}", code="TRIALS_INVALID"
            ) from exc
    return tuple(records)


def append_trial(path: Path, record: TrialRecord) -> None:
    """Append ``record`` as one JSON line; refuses a duplicated ``trial_id``."""
    if any(existing.trial_id == record.trial_id for existing in read_trials(path)):
        raise BacktestResearchError(
            f"trial {record.trial_id!r} is already in {path}", code="TRIAL_DUPLICATE"
        )
    line = record.model_dump_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def count_trials(registry_path: Path, include_v1: bool = True) -> int:
    """N for the deflated Sharpe ratio: distinct ``(hypothesis_id, config_hash)`` pairs
    of the log (a rerun of the same configuration is not a new trial; a changed
    configuration is) plus :data:`V1_TRIAL_COUNT` when ``include_v1``."""
    distinct = {(r.hypothesis_id, r.config_hash) for r in read_trials(registry_path)}
    return len(distinct) + (V1_TRIAL_COUNT if include_v1 else 0)


# --------------------------------------------------------------------------- lockbox


def assert_lockbox_allowed(
    hypothesis: Hypothesis | None, trials: Sequence[TrialRecord], git_status: GitStatus
) -> None:
    """Raise :class:`BacktestResearchError` unless the lockbox may be opened.

    Checked in order: the hypothesis is registered (``None`` = not found), it has a
    non-empty ``pass_rule["lockbox"]``, the commit is recorded, the tree is clean and
    no lockbox trial of this hypothesis exists (the lockbox opens once).
    """
    if hypothesis is None:
        raise BacktestResearchError(
            "lockbox refused: the hypothesis is not registered", code="LOCKBOX_NOT_REGISTERED"
        )
    if not hypothesis.pass_rule.get("lockbox"):
        raise BacktestResearchError(
            f"lockbox refused: {hypothesis.id} has no lockbox pass rule written before opening",
            code="LOCKBOX_NO_PASS_RULE",
        )
    if not git_status.commit:
        raise BacktestResearchError(
            "lockbox refused: the git commit is not recorded", code="LOCKBOX_NO_COMMIT"
        )
    if git_status.dirty:
        raise BacktestResearchError(
            "lockbox refused: the git working tree has uncommitted changes",
            code="LOCKBOX_DIRTY_TREE",
        )
    if any(t.hypothesis_id == hypothesis.id and t.mode == "lockbox" for t in trials):
        raise BacktestResearchError(
            f"lockbox refused: the lockbox was already opened for {hypothesis.id}",
            code="LOCKBOX_ALREADY_OPENED",
        )
