"""Unit tests for position sizing (sec. 15.2). Expectations are hand-calculated."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from domain.errors import NonRetryableError
from domain.risk.exits import INVALID_RISK_INPUT, PARAM_PENDING
from domain.risk.position_sizer import (
    INVALID_STOP,
    QTY_TOO_SMALL,
    SIZING_OK,
    SizingResult,
    build_proposed_trade,
    size_position,
)
from domain.risk.risk_engine import RiskParams

D = Decimal


def risk_params(**overrides: Any) -> RiskParams:
    """Explicit test fixture values (fractions: 0.01 = 1 %), not OWNER_DECISION defaults."""
    base: dict[str, Any] = {
        "risk_per_trade_pct": D("0.01"),
        "max_daily_loss_pct": D("0.02"),
        "max_weekly_loss_pct": D("0.05"),
        "max_drawdown_pct": D("0.10"),
        "max_positions": 3,
        "max_total_exposure_pct": D("0.5"),
        "max_symbol_exposure_pct": D("0.10"),
        "max_aggregate_open_risk_pct": D("0.03"),
        "slippage_buffer_bps": D("5"),
        "min_qty": 1,
    }
    base.update(overrides)
    return RiskParams(**base)


def size(
    entry: str, stop: str, equity: str, buying_power: str, **param_overrides: Any
) -> SizingResult:
    return size_position(
        entry_ref=D(entry),
        stop_price=D(stop),
        equity=D(equity),
        buying_power=D(buying_power),
        params=risk_params(**param_overrides),
    )


# Hand calculations (risk 1 %, symbol exposure 10 %, slippage 5 bps):
#
# A) entry 50, stop 45, equity 100000, bp 100000
#    slippage = 50*5/10000 = 0.025; eff = 5.025; risk_amount = 1000
#    qty_risk = floor(1000/5.025 = 199.004...) = 199
#    qty_exposure = floor(10000/50) = 200
#    qty_bp = floor(100000/50.025 = 1999.0005...) = 1999        -> 199 (risk)
# B) entry 100, stop 98, equity 100000, bp 100000
#    slippage = 0.05; eff = 2.05; qty_risk = floor(487.80...) = 487
#    qty_exposure = floor(10000/100) = 100; qty_bp = floor(999.50...) = 999 -> 100 (exposure)
# C) entry 50, stop 45, equity 100000, bp 5000
#    qty_bp = floor(5000/50.025 = 99.95...) = 99                -> 99 (buying_power)
@pytest.mark.parametrize(
    ("entry", "stop", "equity", "bp", "qty_risk", "qty_exposure", "qty_bp", "qty", "binding"),
    [
        ("50", "45", "100000", "100000", 199, 200, 1999, 199, "risk"),
        ("100", "98", "100000", "100000", 487, 100, 999, 100, "exposure"),
        ("50", "45", "100000", "5000", 199, 200, 99, 99, "buying_power"),
    ],
)
def test_each_min_branch_can_win(
    entry: str,
    stop: str,
    equity: str,
    bp: str,
    qty_risk: int,
    qty_exposure: int,
    qty_bp: int,
    qty: int,
    binding: str,
) -> None:
    result = size(entry, stop, equity, bp)
    assert result.check.passed
    assert result.check.code == SIZING_OK
    assert (result.qty_risk, result.qty_exposure, result.qty_buying_power) == (
        qty_risk,
        qty_exposure,
        qty_bp,
    )
    assert result.qty == qty
    assert result.binding_constraint == binding


def test_intermediate_values_are_recorded() -> None:
    result = size("50", "45", "100000", "100000")
    assert result.stop_distance == D("5")
    assert result.slippage == D("0.025")
    assert result.effective_risk_per_share == D("5.025")
    assert result.risk_amount == D("1000.00")
    assert result.entry_ref == D("50")
    assert result.risk_per_trade_pct == D("0.01")
    assert result.slippage_buffer_bps == D("5")
    assert result.min_qty == 1
    dumped = result.model_dump(mode="json")
    assert dumped["qty"] == 199
    assert dumped["effective_risk_per_share"] == "5.025"


def test_exact_quotients_are_not_reduced_and_ties_prefer_risk() -> None:
    # equity 10000, risk 1 % = 100; entry 10, stop 9, no slippage: eff = 1 -> qty_risk = 100
    # exposure 10 %: 1000/10 = 100 (tie); bp 100000/10 = 10000
    result = size("10", "9", "10000", "100000", slippage_buffer_bps=D("0"))
    assert result.qty_risk == 100
    assert result.qty_exposure == 100
    assert result.qty == 100
    assert result.binding_constraint == "risk"


@pytest.mark.parametrize("stop", ["50", "50.01", "60"])
def test_invalid_stop(stop: str) -> None:
    result = size("50", stop, "100000", "100000")
    assert not result.check.passed
    assert result.check.code == INVALID_STOP
    assert result.qty is None
    assert result.stop_distance == D("50") - D(stop)


@pytest.mark.parametrize(
    ("entry", "stop", "equity", "bp", "expected_qty"),
    [
        # risk_amount 10; eff = 20 + 0.05 = 20.05 -> 0.498... -> 0 (never rounded up)
        ("100", "80", "1000", "100000", 0),
        # bp 50 < entry + slippage 100.05 -> 0
        ("100", "98", "100000", "50", 0),
        # negative buying power: floor(-100/50.025 = -1.99...) = -2
        ("50", "45", "100000", "-100", -2),
        # negative equity
        ("50", "45", "-1000", "100000", -2),
    ],
)
def test_qty_too_small_never_rounds_up(
    entry: str, stop: str, equity: str, bp: str, expected_qty: int
) -> None:
    result = size(entry, stop, equity, bp)
    assert not result.check.passed
    assert result.check.code == QTY_TOO_SMALL
    assert result.qty == expected_qty


def test_min_qty_boundary() -> None:
    # qty = 199 (case A): min_qty 199 passes, 200 is rejected
    assert size("50", "45", "100000", "100000", min_qty=199).check.passed
    rejected = size("50", "45", "100000", "100000", min_qty=200)
    assert rejected.check.code == QTY_TOO_SMALL
    assert rejected.qty == 199


@pytest.mark.parametrize(
    "missing", ["risk_per_trade_pct", "max_symbol_exposure_pct", "slippage_buffer_bps"]
)
def test_pending_param_fails_closed(missing: str) -> None:
    result = size("50", "45", "100000", "100000", **{missing: None})
    assert not result.check.passed
    assert result.check.code == PARAM_PENDING
    assert result.check.detail["pending"] == [missing]
    assert result.qty is None


def test_unrelated_pending_params_do_not_block_sizing() -> None:
    result = size("50", "45", "100000", "100000", max_daily_loss_pct=None)
    assert result.check.passed


@pytest.mark.parametrize("field", ["entry_ref", "stop_price", "equity", "buying_power"])
def test_float_inputs_are_rejected(field: str) -> None:
    kwargs: dict[str, Any] = {
        "entry_ref": D("50"),
        "stop_price": D("45"),
        "equity": D("100000"),
        "buying_power": D("100000"),
        "params": risk_params(),
    }
    kwargs[field] = 1.0
    with pytest.raises(NonRetryableError) as exc:
        size_position(**kwargs)
    assert exc.value.code == INVALID_RISK_INPUT


@pytest.mark.parametrize("entry", ["0", "-1"])
def test_non_positive_entry_raises(entry: str) -> None:
    with pytest.raises(NonRetryableError):
        size(entry, "-5", "100000", "100000")


# ------------------------------------------------------------------ RiskParams units


@pytest.mark.parametrize(
    "overrides",
    [
        {"risk_per_trade_pct": D("1.5")},  # percentage written by mistake
        {"risk_per_trade_pct": D("0")},
        {"max_symbol_exposure_pct": D("-0.1")},
        {"risk_per_trade_pct": 0.01},  # float
        {"slippage_buffer_bps": D("-1")},
        {"slippage_buffer_bps": D("10000")},
        {"min_qty": 0},
        {"max_positions": 0},
    ],
)
def test_risk_params_ranges(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        risk_params(**overrides)


# ------------------------------------------------------------------ build_proposed_trade


def test_build_proposed_trade() -> None:
    sizing = size("50", "45", "100000", "100000")
    trade = build_proposed_trade(
        sizing, signal_id="sig-1", symbol="AAPL", take_profit_price=D("60")
    )
    assert trade.qty == 199
    assert trade.entry_ref == D("50")
    assert trade.stop_price == D("45")
    assert trade.take_profit_price == D("60")
    assert trade.risk_per_share == D("5.025")
    assert trade.risk_amount == D("999.975")  # 199 * 5.025 <= 1000
    assert trade.risk_pct_of_equity == D("0.00999975")
    assert trade.r_multiple == D("2")  # (60 - 50) / 5


def test_build_proposed_trade_refuses_rejected_sizing() -> None:
    sizing = size("100", "80", "1000", "100000")
    with pytest.raises(NonRetryableError) as exc:
        build_proposed_trade(sizing, signal_id="sig-1", symbol="AAPL", take_profit_price=D("120"))
    assert exc.value.code == INVALID_RISK_INPUT
