"""Unit tests for the risk limits (sec. 15.1). Expectations are hand-calculated."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from domain.models import Position, ProposedTrade
from domain.risk.risk_engine import (
    AGGREGATE_OPEN_RISK,
    DAILY_LOSS_LIMIT,
    INVALID_RISK_INPUT,
    LIMIT_CHECKS,
    MAX_DRAWDOWN,
    MAX_POSITIONS,
    PARAM_PENDING,
    RISK_PER_TRADE_EXCEEDED,
    SYMBOL_EXPOSURE,
    TOTAL_EXPOSURE,
    WEEKLY_LOSS_LIMIT,
    LimitCheck,
    PendingEntry,
    PositionStop,
    RiskParams,
    RiskState,
    check_aggregate_open_risk,
    check_daily_loss,
    check_drawdown,
    check_max_positions,
    check_risk_per_trade,
    check_symbol_exposure,
    check_total_exposure,
    check_weekly_loss,
    evaluate_limits,
    pending_risk_params,
)

D = Decimal


def risk_params(**overrides: Any) -> RiskParams:
    """Explicit test fixture values (fractions: 0.01 = 1 %), not OWNER_DECISION defaults."""
    base: dict[str, Any] = {
        "risk_per_trade_pct": D("0.005"),
        "max_daily_loss_pct": D("0.02"),
        "max_weekly_loss_pct": D("0.05"),
        "max_drawdown_pct": D("0.10"),
        "max_positions": 3,
        "max_total_exposure_pct": D("0.5"),
        "max_symbol_exposure_pct": D("0.05"),
        "max_aggregate_open_risk_pct": D("0.015"),
        "slippage_buffer_bps": D("5"),
        "min_qty": 1,
    }
    base.update(overrides)
    return RiskParams(**base)


def state(**overrides: Any) -> RiskState:
    base: dict[str, Any] = {
        "equity": D("100000"),
        "last_equity": D("100000"),
        "week_start_equity": D("100000"),
        "peak_equity": D("100000"),
        "open_positions": (),
        "pending_entries": (),
        "position_stops": (),
    }
    base.update(overrides)
    return RiskState(**base)


def trade(**overrides: Any) -> ProposedTrade:
    # 100 shares at 50, stop 45: notional 5000 (5 % of 100000), risk 500 (0.5 %)
    base: dict[str, Any] = {
        "signal_id": "sig-1",
        "symbol": "MSFT",
        "qty": 100,
        "entry_ref": D("50"),
        "stop_price": D("45"),
        "take_profit_price": D("60"),
        "risk_per_share": D("5"),
        "risk_amount": D("500"),
        "risk_pct_of_equity": D("0.005"),
        "r_multiple": D("2"),
    }
    base.update(overrides)
    return ProposedTrade(**base)


def position(symbol: str, qty: str, avg: str, market_value: str) -> Position:
    return Position(symbol=symbol, qty=D(qty), avg_entry_price=D(avg), market_value=D(market_value))


def pending(
    symbol: str = "NVDA", qty: int = 100, entry: str = "50", stop: str = "45"
) -> PendingEntry:
    return PendingEntry(symbol=symbol, qty=qty, entry_ref=D(entry), stop_price=D(stop))


# ------------------------------------------------------------------ risk per trade


@pytest.mark.parametrize(("risk_amount", "passed"), [("500", True), ("500.01", False)])
def test_risk_per_trade_boundary(risk_amount: str, passed: bool) -> None:
    # 500 / 100000 = 0.005 == limit -> passes; above -> fails
    result = check_risk_per_trade(risk_params(), state(), trade(risk_amount=D(risk_amount)))
    assert result.passed is passed
    assert result.code == RISK_PER_TRADE_EXCEEDED
    assert result.detail["limit"] == RISK_PER_TRADE_EXCEEDED


# ------------------------------------------------------------------ loss limits


@pytest.mark.parametrize(
    ("equity", "passed"),
    [
        ("98000", False),  # (98000-100000)/100000 = -0.02: loss reaches 2 % -> fails
        ("98000.01", True),
        ("97000", False),
        ("101000", True),  # gain
    ],
)
def test_daily_loss_boundary(equity: str, passed: bool) -> None:
    result = check_daily_loss(risk_params(), state(equity=D(equity)), trade())
    assert result.passed is passed
    assert result.code == DAILY_LOSS_LIMIT


def test_daily_loss_is_equity_based_so_includes_unrealized_pnl() -> None:
    # Equity already reflects an open position's unrealized loss of 2000 (no realized P&L).
    open_pos = position("AAPL", "100", "150", "13000")  # cost 15000, worth 13000
    st = state(
        equity=D("98000"),
        open_positions=(open_pos,),
        position_stops=(PositionStop(symbol="AAPL", stop_price=D("120")),),
    )
    result = check_daily_loss(risk_params(), st, trade())
    assert not result.passed
    assert result.detail["value"] == D("-0.02")


@pytest.mark.parametrize(("equity", "passed"), [("95000", False), ("95000.01", True)])
def test_weekly_loss_boundary(equity: str, passed: bool) -> None:
    st = state(equity=D(equity), last_equity=D(equity))
    result = check_weekly_loss(risk_params(), st, trade())
    assert result.passed is passed
    assert result.code == WEEKLY_LOSS_LIMIT


@pytest.mark.parametrize(
    ("equity", "passed"),
    [
        ("99000", False),  # (99000-110000)/110000 = -0.10 -> fails
        ("99000.01", True),
        ("120000", True),  # above a stale peak
    ],
)
def test_drawdown_boundary(equity: str, passed: bool) -> None:
    st = state(equity=D(equity), peak_equity=D("110000"))
    result = check_drawdown(risk_params(), st, trade())
    assert result.passed is passed
    assert result.code == MAX_DRAWDOWN


@pytest.mark.parametrize(
    ("check", "field"),
    [
        (check_daily_loss, "last_equity"),
        (check_weekly_loss, "week_start_equity"),
        (check_drawdown, "peak_equity"),
    ],
)
@pytest.mark.parametrize("reference", ["0", "-1"])
def test_loss_limits_fail_closed_on_non_positive_reference(
    check: LimitCheck, field: str, reference: str
) -> None:
    result = check(risk_params(), state(**{field: D(reference)}), trade())
    assert not result.passed
    assert result.code == INVALID_RISK_INPUT


# ------------------------------------------------------------------ positions and exposure


@pytest.mark.parametrize(
    ("n_positions", "n_pending", "passed"),
    [(0, 0, True), (1, 1, True), (2, 0, True), (2, 1, False), (3, 0, False)],
)
def test_max_positions_boundary(n_positions: int, n_pending: int, passed: bool) -> None:
    # limit 3, counting open + pending + the new trade
    positions = tuple(position(f"S{i}", "1", "10", "10") for i in range(n_positions))
    entries = tuple(pending(symbol=f"P{i}") for i in range(n_pending))
    st = state(open_positions=positions, pending_entries=entries)
    result = check_max_positions(risk_params(), st, trade())
    assert result.passed is passed
    assert result.code == MAX_POSITIONS
    assert result.detail["value"] == n_positions + n_pending + 1


@pytest.mark.parametrize(("market_value", "passed"), [("40000", True), ("40000.01", False)])
def test_total_exposure_boundary(market_value: str, passed: bool) -> None:
    # positions 40000 + pending 100*50 = 5000 + new 100*50 = 5000 -> 50000/100000 = 0.5
    st = state(
        open_positions=(position("AAPL", "100", "400", market_value),),
        pending_entries=(pending(),),
    )
    result = check_total_exposure(risk_params(), st, trade())
    assert result.passed is passed
    assert result.code == TOTAL_EXPOSURE
    assert result.detail["pending_value"] == D("5000")
    assert result.detail["new_value"] == D("5000")


def test_total_exposure_counts_absolute_market_value() -> None:
    st = state(
        open_positions=(position("AAPL", "-100", "400", "-40000.01"),), pending_entries=(pending(),)
    )
    assert not check_total_exposure(risk_params(), st, trade()).passed


@pytest.mark.parametrize(("qty", "passed"), [(100, True), (101, False)])
def test_symbol_exposure_boundary(qty: int, passed: bool) -> None:
    # 100*50 = 5000 / 100000 = 0.05 == limit -> passes; 101 shares = 5050 -> fails
    result = check_symbol_exposure(risk_params(), state(), trade(qty=qty))
    assert result.passed is passed
    assert result.code == SYMBOL_EXPOSURE


# ------------------------------------------------------------------ aggregate open risk


def _aggregate_state(stop: str = "145") -> RiskState:
    return state(
        open_positions=(position("AAPL", "100", "150", "15000"),),
        pending_entries=(pending(),),
        position_stops=(PositionStop(symbol="AAPL", stop_price=D(stop)),),
    )


@pytest.mark.parametrize(("limit", "passed"), [("0.015", True), ("0.0149", False)])
def test_aggregate_open_risk_boundary(limit: str, passed: bool) -> None:
    # open (150-145)*100 = 500 + pending (50-45)*100 = 500 + new 500 = 1500 / 100000 = 0.015
    params = risk_params(max_aggregate_open_risk_pct=D(limit))
    result = check_aggregate_open_risk(params, _aggregate_state(), trade())
    assert result.passed is passed
    assert result.code == AGGREGATE_OPEN_RISK
    assert result.detail["open_risk"] == D("500")
    assert result.detail["pending_risk"] == D("500")
    assert result.detail["new_risk"] == D("500")


def test_aggregate_open_risk_clamps_stop_above_entry_to_zero() -> None:
    result = check_aggregate_open_risk(risk_params(), _aggregate_state(stop="155"), trade())
    assert result.detail["open_risk"] == D("0")
    assert result.passed


def test_aggregate_open_risk_fails_closed_without_stop() -> None:
    st = state(open_positions=(position("AAPL", "100", "150", "15000"),), position_stops=())
    result = check_aggregate_open_risk(risk_params(), st, trade())
    assert not result.passed
    assert result.code == INVALID_RISK_INPUT
    assert result.detail["limit"] == AGGREGATE_OPEN_RISK
    assert result.detail["symbols"] == ["AAPL"]


def test_risk_state_rejects_duplicate_stops() -> None:
    stop = PositionStop(symbol="AAPL", stop_price=D("145"))
    with pytest.raises(ValidationError):
        state(position_stops=(stop, stop))


def test_risk_state_has_no_defaults() -> None:
    with pytest.raises(ValidationError):
        RiskState(equity=D("1"), last_equity=D("1"), week_start_equity=D("1"), peak_equity=D("1"))  # type: ignore[call-arg]


# ------------------------------------------------------------------ fail closed


EQUITY_DENOMINATOR_CHECKS = [
    check_risk_per_trade,
    check_total_exposure,
    check_symbol_exposure,
    check_aggregate_open_risk,
]


@pytest.mark.parametrize("check", EQUITY_DENOMINATOR_CHECKS)
@pytest.mark.parametrize("equity", ["0", "-5000"])
def test_non_positive_equity_fails_closed(check: LimitCheck, equity: str) -> None:
    result = check(risk_params(), state(equity=D(equity)), trade())
    assert not result.passed
    assert result.code == INVALID_RISK_INPUT


PARAM_OF_CHECK: list[tuple[LimitCheck, str, str]] = [
    (check_risk_per_trade, "risk_per_trade_pct", RISK_PER_TRADE_EXCEEDED),
    (check_daily_loss, "max_daily_loss_pct", DAILY_LOSS_LIMIT),
    (check_weekly_loss, "max_weekly_loss_pct", WEEKLY_LOSS_LIMIT),
    (check_drawdown, "max_drawdown_pct", MAX_DRAWDOWN),
    (check_max_positions, "max_positions", MAX_POSITIONS),
    (check_total_exposure, "max_total_exposure_pct", TOTAL_EXPOSURE),
    (check_symbol_exposure, "max_symbol_exposure_pct", SYMBOL_EXPOSURE),
    (check_aggregate_open_risk, "max_aggregate_open_risk_pct", AGGREGATE_OPEN_RISK),
]


@pytest.mark.parametrize(("check", "param", "limit"), PARAM_OF_CHECK)
def test_pending_param_fails_closed(check: LimitCheck, param: str, limit: str) -> None:
    result = check(risk_params(**{param: None}), state(), trade())
    assert not result.passed
    assert result.code == PARAM_PENDING
    assert result.detail == {"limit": limit, "param": param}


# ------------------------------------------------------------------ aggregate evaluation


def test_evaluate_limits_all_pass_in_table_order() -> None:
    results = evaluate_limits(risk_params(), state(), trade())
    assert [r.code for r in results] == [
        RISK_PER_TRADE_EXCEEDED,
        DAILY_LOSS_LIMIT,
        WEEKLY_LOSS_LIMIT,
        MAX_DRAWDOWN,
        MAX_POSITIONS,
        TOTAL_EXPOSURE,
        SYMBOL_EXPOSURE,
        AGGREGATE_OPEN_RISK,
    ]
    assert all(r.passed for r in results)
    assert len(results) == len(LIMIT_CHECKS)


def test_evaluate_limits_reports_every_failure_without_short_circuit() -> None:
    st = state(equity=D("90000"), last_equity=D("100000"), peak_equity=D("100000"))
    results = evaluate_limits(risk_params(), st, trade(qty=200, risk_amount=D("1000")))
    failed = {r.code for r in results if not r.passed}
    # daily -10 %, weekly -10 %, drawdown -10 %, risk 1000/90000, symbol 10000/90000
    assert failed == {
        RISK_PER_TRADE_EXCEEDED,
        DAILY_LOSS_LIMIT,
        WEEKLY_LOSS_LIMIT,
        MAX_DRAWDOWN,
        SYMBOL_EXPOSURE,
    }
    assert len(results) == len(LIMIT_CHECKS)


def test_evaluate_limits_with_all_params_pending_never_passes() -> None:
    params = risk_params(**{param: None for _, param, _ in PARAM_OF_CHECK})
    results = evaluate_limits(params, state(), trade())
    assert results
    assert all(not r.passed and r.code == PARAM_PENDING for r in results)


def test_pending_risk_params() -> None:
    assert pending_risk_params(risk_params()) == ()
    params = risk_params(max_positions=None, slippage_buffer_bps=None)
    assert pending_risk_params(params) == ("max_positions", "slippage_buffer_bps")


def test_risk_params_have_no_defaults() -> None:
    with pytest.raises(ValidationError):
        RiskParams(min_qty=1)  # type: ignore[call-arg]
