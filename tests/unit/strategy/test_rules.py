"""Rule language: parsing, validation and evaluation (sec. 13.3, 14.3)."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from domain.models import RuleValue
from domain.strategy.rules import (
    ComparisonOperator,
    ConstOperand,
    ParamOperand,
    RuleSpec,
    RuleTimeframe,
    SeriesOperand,
    StrategyConfigError,
    evaluate_rule,
    has_missing_values,
    resolve_series_name,
)
from tests.unit.strategy.factories import rule, s


class FakeSource:
    """Series source backed by a dict ``{(timeframe, base_name): values}``."""

    def __init__(self, data: Mapping[tuple[RuleTimeframe, str], tuple[RuleValue, ...]]) -> None:
        self._data = data

    def series_value(self, name: str, offset: int, rule_timeframe: RuleTimeframe) -> RuleValue:
        values = self._data[resolve_series_name(name, rule_timeframe)]
        return values[offset] if -offset <= len(values) else None


P = RuleTimeframe.PRIMARY
C = RuleTimeframe.CONFIRMATION


def spec(data: dict[str, Any]) -> RuleSpec:
    return RuleSpec.model_validate(data)


# --------------------------------------------------------------------------- parsing


def test_parses_yaml_rule_spec() -> None:
    text = """
    - rule_id: ENTRY_TREND_01
      timeframe: primary
      left: {series: ema_fast, offset: -1}
      op: ">"
      right: {series: ema_slow}
    - rule_id: NOTRADE_RSI_HIGH
      timeframe: primary
      left: {series: rsi}
      op: ">="
      right: {param: rsi_overbought}
    - rule_id: ENTRY_MOMENTUM_01
      timeframe: confirmation
      left: {series: rsi_confirm, offset: -2}
      op: "<"
      right: {const: "50.5"}
    """
    specs = [RuleSpec.model_validate(item) for item in yaml.safe_load(text)]
    trend, rsi_high, momentum = specs
    assert trend.left == SeriesOperand(series="ema_fast", offset=-1)
    assert trend.right == SeriesOperand(series="ema_slow", offset=-1)  # default offset -1
    assert trend.op is ComparisonOperator.GT
    assert rsi_high.right == ParamOperand(param="rsi_overbought")
    assert rsi_high.referenced_params() == ("rsi_overbought",)
    assert momentum.timeframe is C
    assert momentum.right == ConstOperand(const=Decimal("50.5"))
    assert momentum.resolved_series() == ((C, "rsi"),)


@pytest.mark.parametrize(
    "data",
    [
        rule("R_1", s("macd"), ">", s("close")),  # unknown series
        rule("R_1", s("ema_fast"), "=>", s("close")),  # unknown operator
        rule("R_1", s("ema_fast"), "in", s("close")),
        rule("R_1", s("ema_fast", 0), ">", s("close")),  # lookahead offset
        rule("R_1", s("ema_fast", 1), ">", s("close")),
        rule("R_1", s("close"), ">", {"const": 50.5}),  # float constant
        rule("R_1", s("close"), ">", {"const": "nan"}),
        rule("R_1", {"const": "1"}, ">", {"const": "2"}),  # no series operand
        rule("R_1", {"param": "a"}, ">", {"const": "2"}),
        rule("R_1", s("close"), ">", {"const": "1", "param": "x"}),  # ambiguous operand
        rule("R_1", s("close"), ">", {"expr": "close * 2"}),  # no arbitrary expressions
        rule("R_1", s("close"), ">", {"param": "Bad-Name"}),
        rule("lower_id", s("close"), ">", s("open")),  # bad rule id
        rule("R_1", s("close"), ">", s("open"), timeframe="daily"),  # bad timeframe
        {**rule("R_1", s("close"), ">", s("open")), "extra": 1},
    ],
)
def test_rejects_invalid_specs(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        spec(data)


def test_spec_is_immutable() -> None:
    parsed = spec(rule("R_1", s("close"), ">", s("open")))
    with pytest.raises(ValidationError):
        parsed.rule_id = "R_2"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "timeframe", "expected"),
    [
        ("close", P, (P, "close")),
        ("close", C, (C, "close")),
        ("ema_fast_confirm", P, (C, "ema_fast")),
        ("ema_fast_confirm", C, (C, "ema_fast")),
    ],
)
def test_resolve_series_name(
    name: str, timeframe: RuleTimeframe, expected: tuple[RuleTimeframe, str]
) -> None:
    assert resolve_series_name(name, timeframe) == expected


# --------------------------------------------------------------------------- evaluation


@pytest.mark.parametrize(
    ("op", "left", "right", "expected"),
    [
        (">", 2.0, 1.0, True),
        (">", 1.0, 1.0, False),
        (">=", 1.0, 1.0, True),
        ("<", 1.0, 2.0, True),
        ("<=", 2.0, 2.0, True),
        ("<=", 2.5, 2.0, False),
        ("==", 1.5, 1.5, True),
        ("!=", 1.5, 1.5, False),
        ("!=", 1.5, 1.25, True),
    ],
)
def test_operators(op: str, left: float, right: float, expected: bool) -> None:
    source = FakeSource({(P, "ema_fast"): (left,), (P, "ema_slow"): (right,)})
    result = evaluate_rule(spec(rule("R_1", s("ema_fast"), op, s("ema_slow"))), source, {})
    assert result.result is expected
    assert result.rule_id == "R_1"
    assert result.values == {"ema_fast[-1]": left, "ema_slow[-1]": right}


def test_mixed_decimal_int_float_comparisons() -> None:
    source = FakeSource(
        {(P, "close"): (Decimal("100.10"),), (P, "volume"): (1500,), (P, "rsi"): (50.0000001,)}
    )
    above = evaluate_rule(spec(rule("R_1", s("close"), ">", {"const": "100.1"})), source, {})
    equal = evaluate_rule(spec(rule("R_2", s("close"), "==", {"const": "100.1"})), source, {})
    volume = evaluate_rule(spec(rule("R_3", s("volume"), ">=", {"const": "1500"})), source, {})
    rsi = evaluate_rule(spec(rule("R_4", s("rsi"), ">", {"const": "50"})), source, {})
    assert (above.result, equal.result, volume.result, rsi.result) == (False, True, True, True)
    assert equal.values == {"close[-1]": Decimal("100.10"), "100.1": Decimal("100.1")}


def test_offsets_select_earlier_bars() -> None:
    source = FakeSource({(P, "close"): (Decimal("1"), Decimal("2"), Decimal("3"))})
    rising = evaluate_rule(spec(rule("R_1", s("close", -1), ">", s("close", -2))), source, {})
    older = evaluate_rule(spec(rule("R_2", s("close", -3), "==", {"const": "1"})), source, {})
    assert rising.result is True
    assert rising.values == {"close[-1]": Decimal("3"), "close[-2]": Decimal("2")}
    assert older.result is True


@pytest.mark.parametrize(
    ("left", "right"),
    [(None, 1.0), (1.0, None), (None, None), (float("nan"), 1.0), (float("inf"), 1.0)],
)
def test_none_or_non_finite_operand_is_false(left: float | None, right: float | None) -> None:
    source = FakeSource({(P, "ema_fast"): (left,), (P, "ema_slow"): (right,)})
    for op in (">", "<", "==", "!=", ">=", "<="):
        result = evaluate_rule(spec(rule("R_1", s("ema_fast"), op, s("ema_slow"))), source, {})
        assert result.result is False
        assert has_missing_values(result)


def test_offset_beyond_history_is_none_and_false() -> None:
    source = FakeSource({(P, "close"): (Decimal("1"),)})
    result = evaluate_rule(spec(rule("R_1", s("close", -2), "<", {"const": "5"})), source, {})
    assert result.result is False
    assert result.values["close[-2]"] is None


def test_threshold_reference() -> None:
    source = FakeSource({(P, "rsi"): (75.0,)})
    rsi_high = spec(rule("R_1", s("rsi"), ">", {"param": "rsi_overbought"}))
    assert evaluate_rule(rsi_high, source, {"rsi_overbought": Decimal("70")}).result is True
    assert evaluate_rule(rsi_high, source, {"rsi_overbought": Decimal("80")}).result is False
    pending = evaluate_rule(rsi_high, source, {"rsi_overbought": None})
    assert pending.result is False
    assert has_missing_values(pending)


def test_unknown_threshold_raises() -> None:
    source = FakeSource({(P, "rsi"): (75.0,)})
    with pytest.raises(StrategyConfigError):
        evaluate_rule(spec(rule("R_1", s("rsi"), ">", {"param": "missing"})), source, {})


def test_confirmation_resolution_in_evaluation() -> None:
    source = FakeSource({(P, "ema_fast"): (1.0,), (C, "ema_fast"): (9.0,), (C, "ema_slow"): (5.0,)})
    via_suffix = spec(rule("R_1", s("ema_fast_confirm"), ">", s("ema_slow_confirm")))
    via_timeframe = spec(rule("R_2", s("ema_fast"), ">", s("ema_slow"), timeframe="confirmation"))
    assert evaluate_rule(via_suffix, source, {}).result is True
    assert evaluate_rule(via_timeframe, source, {}).result is True


def test_complete_values_are_not_missing() -> None:
    source = FakeSource({(P, "close"): (Decimal("2"),)})
    result = evaluate_rule(spec(rule("R_1", s("close"), ">", {"const": "1"})), source, {})
    assert not has_missing_values(result)
