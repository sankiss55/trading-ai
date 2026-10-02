"""Deterministic risk limits of the Risk Engine (sec. 15.1, 15.3).

Every function here is pure: it receives explicit, immutable state and returns a
:class:`~domain.models.CheckResult`. Nothing reads the broker, the database or the clock.

Units: every ``*_pct`` parameter is a **fraction of equity**, never a percentage:
``Decimal("0.01")`` means 1 %. Fractions are validated to ``0 < x <= 1`` so a value
mistakenly written as a percentage (``1`` for 1 %, ``2`` for 2 %) is either rejected at
construction or, for ``1``, means 100 % (which the owner must notice in review).
``slippage_buffer_bps`` is in basis points (``1 bps = 0.0001``).

Result-code convention (stable, used by ``risk_events`` and sec. 19):

* Each limit has one stable code (e.g. ``DAILY_LOSS_LIMIT``). The result carries that
  code whether it passed or failed; ``passed`` is the outcome.
* If a limit cannot be evaluated it fails closed with code ``PARAM_PENDING`` (its
  ``OWNER_DECISION`` parameter is ``None``) or ``INVALID_RISK_INPUT`` (e.g. equity
  ``<= 0`` as a denominator, an open position without a known stop).
* ``detail["limit"]`` always names the limit code, so a fail-closed result is still
  attributable to its limit. ``detail["param"]`` names the configuration parameter.

Loss limits (daily, weekly, drawdown) fail when the loss **reaches** the limit
(``loss >= max``, sec. 30.2). Exposure/risk limits fail when the value **exceeds** the
limit (``value > max``). Daily loss is equity-based, so it includes unrealized P&L.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Annotated, Any, Final, Self

from pydantic import Field, model_validator

from domain.models import (
    CheckResult,
    DomainModel,
    Money,
    Position,
    Price,
    ProposedTrade,
    Symbol,
)
from domain.risk.exits import INVALID_RISK_INPUT, PARAM_PENDING

__all__ = [
    "AGGREGATE_OPEN_RISK",
    "BPS_PER_UNIT",
    "DAILY_LOSS_LIMIT",
    "INVALID_RISK_INPUT",
    "LIMIT_CHECKS",
    "MAX_DRAWDOWN",
    "MAX_POSITIONS",
    "PARAM_PENDING",
    "RISK_PER_TRADE_EXCEEDED",
    "SYMBOL_EXPOSURE",
    "TOTAL_EXPOSURE",
    "WEEKLY_LOSS_LIMIT",
    "Bps",
    "Fraction",
    "LimitCheck",
    "PendingEntry",
    "PositionStop",
    "RiskParams",
    "RiskState",
    "check_aggregate_open_risk",
    "check_daily_loss",
    "check_drawdown",
    "check_max_positions",
    "check_risk_per_trade",
    "check_symbol_exposure",
    "check_total_exposure",
    "check_weekly_loss",
    "evaluate_limits",
    "pending_risk_params",
]

RISK_PER_TRADE_EXCEEDED: Final = "RISK_PER_TRADE_EXCEEDED"
DAILY_LOSS_LIMIT: Final = "DAILY_LOSS_LIMIT"
WEEKLY_LOSS_LIMIT: Final = "WEEKLY_LOSS_LIMIT"
MAX_DRAWDOWN: Final = "MAX_DRAWDOWN"
MAX_POSITIONS: Final = "MAX_POSITIONS"
TOTAL_EXPOSURE: Final = "TOTAL_EXPOSURE"
SYMBOL_EXPOSURE: Final = "SYMBOL_EXPOSURE"
AGGREGATE_OPEN_RISK: Final = "AGGREGATE_OPEN_RISK"

BPS_PER_UNIT: Final = Decimal(10000)
"""Basis points in one unit (unit conversion, not a trading parameter)."""

Fraction = Annotated[Money, Field(gt=0, le=1)]
"""Fraction of equity in ``(0, 1]``: ``Decimal("0.01")`` is 1 %."""

Bps = Annotated[Money, Field(ge=0, lt=BPS_PER_UNIT)]
"""Basis points in ``[0, 10000)``."""


class RiskParams(DomainModel):
    """Parameters of the ``risk`` section of ``config.yaml`` (sec. 7.3, 15).

    All fields are required (no defaults: values come from configuration only). A
    ``None`` value means the ``OWNER_DECISION`` is pending; any check that needs it
    fails closed with ``PARAM_PENDING``. Ranges are validated at construction.
    """

    risk_per_trade_pct: Fraction | None
    max_daily_loss_pct: Fraction | None
    max_weekly_loss_pct: Fraction | None
    max_drawdown_pct: Fraction | None
    max_positions: Annotated[int, Field(ge=1)] | None
    max_total_exposure_pct: Fraction | None
    max_symbol_exposure_pct: Fraction | None
    max_aggregate_open_risk_pct: Fraction | None
    slippage_buffer_bps: Bps | None
    min_qty: Annotated[int, Field(ge=1)]
    """FIJO in the MVP (whole shares); still taken from configuration."""


class PendingEntry(DomainModel):
    """An entry order accepted by the broker but not yet filled (sec. 15.1).

    ``entry_ref`` and ``stop_price`` are the planned levels recorded for the order.
    """

    symbol: Symbol
    qty: int = Field(ge=1)
    entry_ref: Price
    stop_price: Price


class PositionStop(DomainModel):
    """Protective stop recorded in the DB for an open position (sec. 15.1, DB + broker)."""

    symbol: Symbol
    stop_price: Price


class RiskState(DomainModel):
    """Explicit portfolio state the limits are evaluated against.

    No field has a default on purpose: forgetting to pass positions or pending entries
    must not silently understate exposure.

    Attributes:
        equity: Current account equity (broker).
        last_equity: Equity at the previous session close (broker), for daily loss.
        week_start_equity: Equity at the start of the week (``equity_snapshots``).
        peak_equity: Highest recorded equity (``equity_snapshots``).
        open_positions: Open positions as reported by the broker.
        pending_entries: Pending (unfilled) entry orders.
        position_stops: DB stop of each open position; every open position must have
            exactly one, otherwise aggregate open risk fails closed.
    """

    equity: Money
    last_equity: Money
    week_start_equity: Money
    peak_equity: Money
    open_positions: tuple[Position, ...]
    pending_entries: tuple[PendingEntry, ...]
    position_stops: tuple[PositionStop, ...]

    @model_validator(mode="after")
    def _unique_stops(self) -> Self:
        symbols = [stop.symbol for stop in self.position_stops]
        if len(symbols) != len(set(symbols)):
            raise ValueError("position_stops must contain at most one stop per symbol")
        return self


def pending_risk_params(params: RiskParams) -> tuple[str, ...]:
    """Names of the ``RiskParams`` fields still ``None`` (pending ``OWNER_DECISION``)."""
    return tuple(name for name in RiskParams.model_fields if getattr(params, name) is None)


def _pending(limit: str, param: str) -> CheckResult:
    return CheckResult(passed=False, code=PARAM_PENDING, detail={"limit": limit, "param": param})


def _invalid(limit: str, param: str, reason: str, **extra: Any) -> CheckResult:
    return CheckResult(
        passed=False,
        code=INVALID_RISK_INPUT,
        detail={"limit": limit, "param": param, "reason": reason, **extra},
    )


def _result(limit: str, param: str, passed: bool, **values: Any) -> CheckResult:
    return CheckResult(passed=passed, code=limit, detail={"limit": limit, "param": param, **values})


def _new_trade_notional(trade: ProposedTrade) -> Decimal:
    return trade.entry_ref * trade.qty


def check_risk_per_trade(params: RiskParams, state: RiskState, trade: ProposedTrade) -> CheckResult:
    """``trade.risk_amount / equity <= risk_per_trade_pct`` (code ``RISK_PER_TRADE_EXCEEDED``).

    ``trade.risk_amount`` includes the slippage buffer (sec. 15.2).
    """
    limit, param = RISK_PER_TRADE_EXCEEDED, "risk_per_trade_pct"
    threshold = params.risk_per_trade_pct
    if threshold is None:
        return _pending(limit, param)
    if state.equity <= 0:
        return _invalid(limit, param, "equity_not_positive", equity=state.equity)
    value = trade.risk_amount / state.equity
    return _result(
        limit,
        param,
        value <= threshold,
        value=value,
        threshold=threshold,
        risk_amount=trade.risk_amount,
        equity=state.equity,
    )


def _loss_check(
    limit: str, param: str, threshold: Decimal | None, equity: Decimal, reference: Decimal
) -> CheckResult:
    """Shared loss rule: ``(equity - reference) / reference``; fails when loss >= threshold."""
    if threshold is None:
        return _pending(limit, param)
    if reference <= 0:
        return _invalid(limit, param, "reference_equity_not_positive", reference=reference)
    change = (equity - reference) / reference
    return _result(
        limit,
        param,
        -change < threshold,
        value=change,
        threshold=threshold,
        equity=equity,
        reference=reference,
    )


def check_daily_loss(params: RiskParams, state: RiskState, trade: ProposedTrade) -> CheckResult:
    """``(equity - last_equity) / last_equity`` vs ``max_daily_loss_pct``.

    Equity-based, so unrealized P&L is included (sec. 15.1). Fails when the daily loss
    reaches the limit. Code ``DAILY_LOSS_LIMIT``. ``trade`` is unused (uniform signature).
    """
    return _loss_check(
        DAILY_LOSS_LIMIT,
        "max_daily_loss_pct",
        params.max_daily_loss_pct,
        state.equity,
        state.last_equity,
    )


def check_weekly_loss(params: RiskParams, state: RiskState, trade: ProposedTrade) -> CheckResult:
    """``(equity - week_start_equity) / week_start_equity`` vs ``max_weekly_loss_pct``.

    Fails when the weekly loss reaches the limit. Code ``WEEKLY_LOSS_LIMIT``.
    """
    return _loss_check(
        WEEKLY_LOSS_LIMIT,
        "max_weekly_loss_pct",
        params.max_weekly_loss_pct,
        state.equity,
        state.week_start_equity,
    )


def check_drawdown(params: RiskParams, state: RiskState, trade: ProposedTrade) -> CheckResult:
    """``(equity - peak_equity) / peak_equity`` vs ``max_drawdown_pct``.

    Fails when the drawdown reaches the limit. Code ``MAX_DRAWDOWN``.
    """
    return _loss_check(
        MAX_DRAWDOWN,
        "max_drawdown_pct",
        params.max_drawdown_pct,
        state.equity,
        state.peak_equity,
    )


def check_max_positions(params: RiskParams, state: RiskState, trade: ProposedTrade) -> CheckResult:
    """``open positions + pending entries + 1 (new trade) <= max_positions``.

    Code ``MAX_POSITIONS``.
    """
    limit, param = MAX_POSITIONS, "max_positions"
    threshold = params.max_positions
    if threshold is None:
        return _pending(limit, param)
    count = len(state.open_positions) + len(state.pending_entries) + 1
    return _result(
        limit,
        param,
        count <= threshold,
        value=count,
        threshold=threshold,
        open_positions=len(state.open_positions),
        pending_entries=len(state.pending_entries),
    )


def check_total_exposure(params: RiskParams, state: RiskState, trade: ProposedTrade) -> CheckResult:
    """``(sum |market_value| + pending notional + new notional) / equity <= max``.

    Pending notional is ``qty * entry_ref`` of each pending entry; the new trade's is
    ``qty * entry_ref``. Code ``TOTAL_EXPOSURE``.
    """
    limit, param = TOTAL_EXPOSURE, "max_total_exposure_pct"
    threshold = params.max_total_exposure_pct
    if threshold is None:
        return _pending(limit, param)
    if state.equity <= 0:
        return _invalid(limit, param, "equity_not_positive", equity=state.equity)
    positions_value = sum((abs(p.market_value) for p in state.open_positions), Decimal(0))
    pending_value = sum((e.entry_ref * e.qty for e in state.pending_entries), Decimal(0))
    new_value = _new_trade_notional(trade)
    total = positions_value + pending_value + new_value
    value = total / state.equity
    return _result(
        limit,
        param,
        value <= threshold,
        value=value,
        threshold=threshold,
        positions_value=positions_value,
        pending_value=pending_value,
        new_value=new_value,
        equity=state.equity,
    )


def check_symbol_exposure(
    params: RiskParams, state: RiskState, trade: ProposedTrade
) -> CheckResult:
    """``new position value / equity <= max_symbol_exposure_pct`` (code ``SYMBOL_EXPOSURE``).

    Only the new position counts: pyramiding and multiple positions per symbol are
    forbidden by other checks (sec. 5.4).
    """
    limit, param = SYMBOL_EXPOSURE, "max_symbol_exposure_pct"
    threshold = params.max_symbol_exposure_pct
    if threshold is None:
        return _pending(limit, param)
    if state.equity <= 0:
        return _invalid(limit, param, "equity_not_positive", equity=state.equity)
    new_value = _new_trade_notional(trade)
    value = new_value / state.equity
    return _result(
        limit,
        param,
        value <= threshold,
        value=value,
        threshold=threshold,
        new_value=new_value,
        equity=state.equity,
    )


def check_aggregate_open_risk(
    params: RiskParams, state: RiskState, trade: ProposedTrade
) -> CheckResult:
    """``(open risk + pending risk + new trade risk) / equity <= max_aggregate_open_risk_pct``.

    * Open risk: ``max(avg_entry_price - stop, 0) * |qty|`` per broker position, with the
      stop taken from ``state.position_stops``. A position without a stop fails closed
      (``INVALID_RISK_INPUT``): its risk is unknown.
    * Pending risk: ``max(entry_ref - stop, 0) * qty`` per pending entry (conservative:
      a pending bracket becomes open risk as soon as it fills).
    * New trade risk: ``trade.risk_amount`` (includes the slippage buffer).

    Code ``AGGREGATE_OPEN_RISK``.
    """
    limit, param = AGGREGATE_OPEN_RISK, "max_aggregate_open_risk_pct"
    threshold = params.max_aggregate_open_risk_pct
    if threshold is None:
        return _pending(limit, param)
    if state.equity <= 0:
        return _invalid(limit, param, "equity_not_positive", equity=state.equity)
    stops = {stop.symbol: stop.stop_price for stop in state.position_stops}
    missing = sorted(p.symbol for p in state.open_positions if p.symbol not in stops)
    if missing:
        return _invalid(limit, param, "position_without_stop", symbols=missing)
    zero = Decimal(0)
    open_risk = sum(
        (max(p.avg_entry_price - stops[p.symbol], zero) * abs(p.qty) for p in state.open_positions),
        zero,
    )
    pending_risk = sum(
        (max(e.entry_ref - e.stop_price, zero) * e.qty for e in state.pending_entries), zero
    )
    total = open_risk + pending_risk + trade.risk_amount
    value = total / state.equity
    return _result(
        limit,
        param,
        value <= threshold,
        value=value,
        threshold=threshold,
        open_risk=open_risk,
        pending_risk=pending_risk,
        new_risk=trade.risk_amount,
        equity=state.equity,
    )


LimitCheck = Callable[[RiskParams, RiskState, ProposedTrade], CheckResult]

LIMIT_CHECKS: Final[tuple[LimitCheck, ...]] = (
    check_risk_per_trade,
    check_daily_loss,
    check_weekly_loss,
    check_drawdown,
    check_max_positions,
    check_total_exposure,
    check_symbol_exposure,
    check_aggregate_open_risk,
)
"""Every limit of sec. 15.1, in table order."""


def evaluate_limits(
    params: RiskParams, state: RiskState, trade: ProposedTrade
) -> tuple[CheckResult, ...]:
    """Evaluate every limit of sec. 15.1 with the new trade included (sec. 15.2.3).

    All limits are always evaluated (no short-circuit) so ``risk_events`` records every
    result (sec. 19). The trade is acceptable only if every result passed.
    """
    return tuple(check(params, state, trade) for check in LIMIT_CHECKS)
