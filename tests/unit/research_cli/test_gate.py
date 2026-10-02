"""Pass-rule validation and evaluation of the research CLI (``{op, value}`` and ``any_of``)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from backtest.registry import BacktestResearchError, load_hypotheses
from backtest.research_cli import (
    LOCKBOX_METRICS,
    MetricValue,
    dev_metric_names,
    evaluate_pass_rule,
    load_protocol,
    validate_pass_rule,
)

ROOT = Path(__file__).resolve().parents[3]
KNOWN = {"a", "b", "c", "flag"}
BH_RULE: dict[str, Any] = {
    "vs_bh": {
        "any_of": [
            {"a": {"op": ">=", "value": 0.0}},
            {"b": {"op": "<=", "value": 0.5}, "c": {"op": ">=", "value": 0.6}},
        ]
    }
}


def _passed(rule: dict[str, Any], metrics: dict[str, MetricValue]) -> bool:
    validate_pass_rule(rule, KNOWN, where="test")
    return evaluate_pass_rule(rule, metrics, stage="dev").passed


@pytest.mark.parametrize(
    ("op", "threshold", "value", "expected"),
    [
        (">=", 0.1, 0.1, True),
        (">=", 0.1, 0.0999, False),
        (">", 0.0, 0.0, False),
        (">", 0.0, 1e-12, True),
        ("<=", 0.15, 0.15, True),
        ("<=", 0.15, 0.1500001, False),
        ("<", 0.5, 0.5, False),
        ("<", 0.5, 0.49, True),
        ("==", 1.0, 1, True),
        (">=", 1.2, math.inf, True),
        ("<=", 1.2, math.inf, False),
    ],
)
def test_single_criterion_boundaries(
    op: str, threshold: float, value: float, expected: bool
) -> None:
    assert _passed({"a": {"op": op, "value": threshold}}, {"a": value}) is expected


@pytest.mark.parametrize("value", [None, math.nan, "0.5", True])
def test_missing_or_non_numeric_values_fail_closed(value: MetricValue) -> None:
    assert _passed({"a": {"op": ">=", "value": 0.0}}, {"a": value}) is False


def test_absent_metric_fails_closed() -> None:
    assert _passed({"a": {"op": ">=", "value": 0.0}}, {}) is False


@pytest.mark.parametrize(
    ("value", "expected"), [(True, True), (False, False), (None, False), (1, False)]
)
def test_boolean_criterion(value: MetricValue, expected: bool) -> None:
    assert _passed({"flag": {"op": "==", "value": True}}, {"flag": value}) is expected


def test_all_criteria_must_pass() -> None:
    rule = {"a": {"op": ">=", "value": 1.0}, "b": {"op": "<", "value": 1.0}}
    assert _passed(rule, {"a": 1.0, "b": 0.5}) is True
    assert _passed(rule, {"a": 1.0, "b": 1.0}) is False


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"a": 0.1, "b": 0.9, "c": 0.0}, True),  # first alternative alone
        ({"a": -0.1, "b": 0.4, "c": 0.7}, True),  # second alternative in full
        ({"a": -0.1, "b": 0.4, "c": 0.5}, False),  # second alternative only half
        ({"a": None, "b": 0.4, "c": None}, False),  # missing values fail closed
        ({"a": None, "b": 0.5, "c": 0.6}, True),  # boundaries of the second one
    ],
)
def test_any_of_needs_one_complete_alternative(
    metrics: dict[str, MetricValue], expected: bool
) -> None:
    assert _passed(BH_RULE, metrics) is expected


def test_any_of_rows_keep_every_alternative() -> None:
    result = evaluate_pass_rule(BH_RULE, {"a": -1.0, "b": 0.4, "c": 0.7}, stage="dev")
    (row,) = result.rows
    assert row.op == "any_of"
    assert [[r.passed for r in alt] for alt in row.alternatives] == [[False], [True, True]]
    assert row.passed is True


@pytest.mark.parametrize(
    "rule",
    [
        {},
        {"a": {"op": "=>", "value": 1}},
        {"a": {"op": ">=", "value": "1"}},
        {"a": {"op": ">=", "value": math.inf}},
        {"a": {"op": ">=", "value": True}},
        {"a": {"op": ">=", "value": 1, "extra": 2}},
        {"zzz": {"op": ">=", "value": 1}},
        {"g": {"any_of": []}},
        {"g": {"any_of": [{}]}},
        {"g": {"any_of": [{"zzz": {"op": ">=", "value": 1}}]}},
        {"a": 1},
    ],
)
def test_invalid_rules_are_refused_before_running(rule: dict[str, Any]) -> None:
    with pytest.raises(BacktestResearchError) as caught:
        validate_pass_rule(rule, KNOWN, where="test")
    assert caught.value.code == "PASS_RULE_INVALID"


def test_registered_pass_rules_reference_known_metrics_only() -> None:
    protocol = load_protocol(ROOT / "research" / "protocol.yaml")
    for hypothesis in load_hypotheses(ROOT / "research" / "hypotheses"):
        validate_pass_rule(hypothesis.pass_rule["dev"], dev_metric_names(protocol), where="dev")
        validate_pass_rule(hypothesis.pass_rule["lockbox"], LOCKBOX_METRICS, where="lockbox")
    assert "expectancy_r_at_2x_cost" in dev_metric_names(protocol)
    assert "expectancy_r_at_3x_cost" in dev_metric_names(protocol)
