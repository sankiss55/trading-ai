"""Unit tests for exit levels: increments, rounding and coherence (sec. 16)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from domain.errors import NonRetryableError
from domain.risk.exits import (
    EXIT_LEVELS_VALID,
    INVALID_EXIT_LEVELS,
    INVALID_RISK_INPUT,
    PARAM_PENDING,
    ExitParams,
    check_exit_levels,
    compute_exit_levels,
    floor_divide,
    price_increment,
    round_down_to_increment,
)

D = Decimal


def exit_params(**overrides: Any) -> ExitParams:
    """Explicit test fixture values (not OWNER_DECISION defaults)."""
    base: dict[str, Any] = {
        "stop_atr_multiplier": D("2"),
        "take_profit_r_multiple": D("2"),
        "min_tp_distance_ticks": 2,
    }
    base.update(overrides)
    return ExitParams(**base)


# ------------------------------------------------------------------ price_increment


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        (D("1"), D("0.01")),
        (D("1.00"), D("0.01")),
        (D("0.9999"), D("0.0001")),
        (D("0.99999999"), D("0.0001")),
        (D("1.0001"), D("0.01")),
        (D("0.0001"), D("0.0001")),
        (D("523.17"), D("0.01")),
    ],
)
def test_price_increment(price: Decimal, expected: Decimal) -> None:
    assert price_increment(price) == expected


@pytest.mark.parametrize("price", [D("0"), D("-1"), D("NaN"), D("Infinity")])
def test_price_increment_rejects_invalid_price(price: Decimal) -> None:
    with pytest.raises(NonRetryableError) as exc:
        price_increment(price)
    assert exc.value.code == INVALID_RISK_INPUT


def test_price_increment_rejects_float() -> None:
    with pytest.raises(NonRetryableError):
        price_increment(1.5)  # type: ignore[arg-type]


# ------------------------------------------------------------------ rounding


@pytest.mark.parametrize(
    ("price", "increment", "expected"),
    [
        (D("10.019"), None, D("10.01")),
        (D("10.01"), None, D("10.01")),  # exact increment stays
        (D("10.0199999"), None, D("10.01")),  # never rounds up
        (D("1.00999"), None, D("1.00")),  # at the $1 threshold
        (D("1"), None, D("1.00")),
        (D("0.99999"), None, D("0.9999")),  # just below $1: 4 decimals
        (D("0.12345"), None, D("0.1234")),
        (D("0.1234"), None, D("0.1234")),
        (D("10.07"), D("0.05"), D("10.05")),  # explicit increment
        (D("10.10"), D("0.05"), D("10.10")),
    ],
)
def test_round_down_to_increment(
    price: Decimal, increment: Decimal | None, expected: Decimal
) -> None:
    result = round_down_to_increment(price, increment)
    assert result == expected
    assert result.as_tuple().exponent == expected.as_tuple().exponent


@pytest.mark.parametrize("increment", [D("0"), D("-0.01")])
def test_round_down_rejects_non_positive_increment(increment: Decimal) -> None:
    with pytest.raises(NonRetryableError):
        round_down_to_increment(D("10"), increment)


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (D("10"), D("3"), 3),
        (D("9"), D("3"), 3),
        (D("-1"), D("3"), -1),
        (D("-9"), D("3"), -3),
        (D("1000"), D("2.05"), 487),
        (D("0"), D("5"), 0),
    ],
)
def test_floor_divide(numerator: Decimal, denominator: Decimal, expected: int) -> None:
    assert floor_divide(numerator, denominator) == expected


def test_floor_divide_by_zero_raises() -> None:
    with pytest.raises(NonRetryableError):
        floor_divide(D("1"), D("0"))


# ------------------------------------------------------------------ compute_exit_levels


@pytest.mark.parametrize(
    ("entry", "atr", "mult", "r", "stop_raw", "stop", "tp_raw", "tp"),
    [
        # stop = 100 - 1.5*2 = 97; tp = 100 + 3*2 = 106
        ("100", "1.5", "2", "2", "97.0", "97.00", "106.00", "106.00"),
        # stop 99.334 -> 99.33; tp = 100 + 0.67*2 = 101.34
        ("100", "0.333", "2", "2", "99.334", "99.33", "101.34", "101.34"),
        # tp = 100 + 0.67*1.5 = 101.005 -> rounded DOWN to 101.00
        ("100", "0.333", "2", "1.5", "99.334", "99.33", "101.005", "101.00"),
        # sub-dollar: stop 0.5 - 0.02468 = 0.47532 -> 0.4753; tp = 0.5 + 0.0247*2 = 0.5494
        ("0.5", "0.01234", "2", "2", "0.47532", "0.4753", "0.5494", "0.5494"),
        # stop crosses below $1: 0.98 uses 0.0001; tp 1.10 uses 0.01
        ("1.02", "0.02", "2", "2", "0.98", "0.9800", "1.1000", "1.10"),
    ],
)
def test_compute_exit_levels_happy_path(
    entry: str, atr: str, mult: str, r: str, stop_raw: str, stop: str, tp_raw: str, tp: str
) -> None:
    params = exit_params(stop_atr_multiplier=D(mult), take_profit_r_multiple=D(r))
    result = compute_exit_levels(D(entry), D(atr), params)
    assert result.check.passed, result.check.detail
    assert result.check.code == EXIT_LEVELS_VALID
    assert result.stop_price_raw == D(stop_raw)
    assert result.stop_price == D(stop)
    assert result.take_profit_price_raw == D(tp_raw)
    assert result.take_profit_price == D(tp)


def test_compute_exit_levels_records_inputs() -> None:
    result = compute_exit_levels(D("100"), D("1.5"), exit_params())
    assert result.entry_ref == D("100")
    assert result.atr == D("1.5")
    assert result.stop_atr_multiplier == D("2")
    assert result.take_profit_r_multiple == D("2")
    assert result.min_tp_distance_ticks == 2
    assert result.check.detail["stop_price_raw"] == D("97.0")


def test_non_positive_raw_stop_is_invalid_exit_levels() -> None:
    # stop = 10 - 6*2 = -2
    result = compute_exit_levels(D("10"), D("6"), exit_params())
    assert not result.check.passed
    assert result.check.code == INVALID_EXIT_LEVELS
    assert result.check.detail["failed"] == ["stop_positive"]
    assert result.stop_price is None
    assert result.take_profit_price is None


def test_raw_stop_exactly_zero_is_invalid() -> None:
    result = compute_exit_levels(D("10"), D("5"), exit_params())
    assert result.check.code == INVALID_EXIT_LEVELS
    assert result.stop_price_raw == D("0")


def test_zero_atr_makes_stop_equal_entry_invalid() -> None:
    result = compute_exit_levels(D("100"), D("0"), exit_params())
    assert result.check.code == INVALID_EXIT_LEVELS
    assert "stop_below_entry" in result.check.detail["failed"]
    assert "tp_above_entry" in result.check.detail["failed"]


@pytest.mark.parametrize(("ticks", "passed"), [(1, True), (2, False)])
def test_min_tp_distance_boundary(ticks: int, passed: bool) -> None:
    # stop 99.995 -> 99.99; tp = 100 + 0.01*1 = 100.01: distance = 1 tick exactly
    params = exit_params(
        stop_atr_multiplier=D("1"), take_profit_r_multiple=D("1"), min_tp_distance_ticks=ticks
    )
    result = compute_exit_levels(D("100"), D("0.005"), params)
    assert result.take_profit_price == D("100.01")
    assert result.check.passed is passed
    if not passed:
        assert result.check.detail["failed"] == ["tp_min_distance"]


@pytest.mark.parametrize(
    "missing", ["stop_atr_multiplier", "take_profit_r_multiple", "min_tp_distance_ticks"]
)
def test_pending_exit_param_fails_closed(missing: str) -> None:
    result = compute_exit_levels(D("100"), D("1"), exit_params(**{missing: None}))
    assert not result.check.passed
    assert result.check.code == PARAM_PENDING
    assert result.check.detail["pending"] == [missing]
    assert result.stop_price is None


def test_compute_rejects_float_inputs() -> None:
    with pytest.raises(NonRetryableError):
        compute_exit_levels(100.0, D("1"), exit_params())  # type: ignore[arg-type]
    with pytest.raises(NonRetryableError):
        compute_exit_levels(D("100"), 1.0, exit_params())  # type: ignore[arg-type]


# ------------------------------------------------------------------ check_exit_levels


@pytest.mark.parametrize(
    ("entry", "stop", "tp", "failed"),
    [
        ("100", "97.00", "106.00", []),
        ("100", "0", "106.00", ["stop_positive"]),
        ("100", "100", "106.00", ["stop_below_entry"]),
        ("100", "101", "106.00", ["stop_below_entry"]),
        ("100", "97", "100", ["tp_above_entry", "tp_min_distance"]),
        ("100", "97", "100.01", ["tp_min_distance"]),
        ("100", "97", "100.02", []),  # exactly 2 ticks
        ("100", "97.005", "106", ["stop_on_increment"]),
        ("100", "97", "106.001", ["tp_on_increment"]),
        ("0.5", "0.4753", "0.5002", []),  # 2 sub-dollar ticks
        ("0.5", "0.4753", "0.5001", ["tp_min_distance"]),
    ],
)
def test_check_exit_levels_table(entry: str, stop: str, tp: str, failed: list[str]) -> None:
    result = check_exit_levels(D(entry), D(stop), D(tp), 2)
    assert result.detail["failed"] == failed
    assert result.passed is (not failed)
    assert result.code == (EXIT_LEVELS_VALID if not failed else INVALID_EXIT_LEVELS)


def test_check_exit_levels_pending_ticks() -> None:
    result = check_exit_levels(D("100"), D("97"), D("106"), None)
    assert not result.passed
    assert result.code == PARAM_PENDING


# ------------------------------------------------------------------ ExitParams


@pytest.mark.parametrize(
    "overrides",
    [
        {"stop_atr_multiplier": D("0")},
        {"stop_atr_multiplier": D("-1")},
        {"take_profit_r_multiple": D("0")},
        {"min_tp_distance_ticks": 0},
        {"stop_atr_multiplier": 2.0},
    ],
)
def test_exit_params_validation(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        exit_params(**overrides)


def test_exit_params_have_no_defaults() -> None:
    with pytest.raises(ValidationError):
        ExitParams()  # type: ignore[call-arg]
