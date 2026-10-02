"""Position sizing for long entries (sec. 15.2).

Exact formula (all ``Decimal``)::

    stop_distance            = entry_ref - stop_price
    slippage                 = entry_ref * slippage_buffer_bps / 10000
    effective_risk_per_share = stop_distance + slippage
    risk_amount              = equity * risk_per_trade_pct
    qty_risk                 = floor(risk_amount / effective_risk_per_share)
    qty_exposure             = floor(equity * max_symbol_exposure_pct / entry_ref)
    qty_buying_power         = floor(buying_power / (entry_ref + slippage))
    qty                      = min(qty_risk, qty_exposure, qty_buying_power)

Units: ``risk_per_trade_pct`` and ``max_symbol_exposure_pct`` are **fractions**
(``Decimal("0.01")`` = 1 %), validated by :class:`~domain.risk.risk_engine.RiskParams`;
``slippage_buffer_bps`` is in basis points.

``stop_price`` must already be rounded to a valid increment (sec. 16.3.4): pass
``ExitLevelsResult.stop_price``.

Rejections never raise; they are returned in ``SizingResult.check``:

* ``INVALID_STOP``: ``stop_distance <= 0``.
* ``QTY_TOO_SMALL``: ``qty < min_qty``. The quantity is NEVER rounded up.
* ``PARAM_PENDING``: a needed ``OWNER_DECISION`` parameter is ``None``.
* ``SIZING_OK``: sized successfully (``qty >= min_qty``).

After sizing, every limit of sec. 15.1 must still be checked with the new position
included (sec. 15.2.3): see :func:`domain.risk.risk_engine.evaluate_limits`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Final, Literal

from domain.errors import NonRetryableError
from domain.models import CheckResult, DomainModel, Money, ProposedTrade
from domain.risk.exits import INVALID_RISK_INPUT, PARAM_PENDING, floor_divide, require_decimal
from domain.risk.risk_engine import BPS_PER_UNIT, RiskParams

__all__ = [
    "INVALID_STOP",
    "QTY_TOO_SMALL",
    "SIZING_OK",
    "BindingConstraint",
    "SizingResult",
    "build_proposed_trade",
    "size_position",
]

INVALID_STOP: Final = "INVALID_STOP"
QTY_TOO_SMALL: Final = "QTY_TOO_SMALL"
SIZING_OK: Final = "SIZING_OK"

BindingConstraint = Literal["risk", "exposure", "buying_power"]
"""Which of the three quantities determined ``qty`` (first in this order on ties)."""

_SIZING_PARAMS: Final = (
    "risk_per_trade_pct",
    "max_symbol_exposure_pct",
    "slippage_buffer_bps",
)


class SizingResult(DomainModel):
    """Every input, intermediate value and the verdict of one sizing (sec. 15.2.4).

    ``model_dump(mode="json")`` is the ``risk_events.sizing`` payload. Intermediate
    fields are ``None`` when the calculation stopped earlier (pending parameter or
    invalid stop). ``qty`` is the computed ``min(...)`` even when it is rejected as too
    small, so the log shows the actual value; use it only if ``check.passed``.
    """

    entry_ref: Money
    stop_price: Money
    equity: Money
    buying_power: Money
    risk_per_trade_pct: Money | None
    max_symbol_exposure_pct: Money | None
    slippage_buffer_bps: Money | None
    min_qty: int
    stop_distance: Money | None = None
    slippage: Money | None = None
    effective_risk_per_share: Money | None = None
    risk_amount: Money | None = None
    qty_risk: int | None = None
    qty_exposure: int | None = None
    qty_buying_power: int | None = None
    qty: int | None = None
    binding_constraint: BindingConstraint | None = None
    check: CheckResult


def size_position(
    *,
    entry_ref: Decimal,
    stop_price: Decimal,
    equity: Decimal,
    buying_power: Decimal,
    params: RiskParams,
) -> SizingResult:
    """Compute the whole-share quantity of a long entry (sec. 15.2).

    Args:
        entry_ref: Entry reference price (signal bar close, or limit price).
        stop_price: Stop price already rounded per sec. 16.3.
        equity: Account equity from the broker.
        buying_power: Available buying power from the broker.
        params: Risk parameters; only ``risk_per_trade_pct``,
            ``max_symbol_exposure_pct``, ``slippage_buffer_bps`` and ``min_qty`` are used.

    Returns:
        A :class:`SizingResult`; accept the trade only if ``result.check.passed``.

    Raises:
        NonRetryableError: code ``INVALID_RISK_INPUT`` if an amount is not a finite
            ``Decimal`` or ``entry_ref <= 0`` (contract violations, not rejections).
    """
    for name, value in (
        ("entry_ref", entry_ref),
        ("stop_price", stop_price),
        ("equity", equity),
        ("buying_power", buying_power),
    ):
        require_decimal(name, value)
    if entry_ref <= 0:
        raise NonRetryableError(f"entry_ref must be > 0, got {entry_ref}", code=INVALID_RISK_INPUT)

    base: dict[str, Any] = {
        "entry_ref": entry_ref,
        "stop_price": stop_price,
        "equity": equity,
        "buying_power": buying_power,
        "risk_per_trade_pct": params.risk_per_trade_pct,
        "max_symbol_exposure_pct": params.max_symbol_exposure_pct,
        "slippage_buffer_bps": params.slippage_buffer_bps,
        "min_qty": params.min_qty,
    }
    risk_pct = params.risk_per_trade_pct
    exposure_pct = params.max_symbol_exposure_pct
    slippage_bps = params.slippage_buffer_bps
    if risk_pct is None or exposure_pct is None or slippage_bps is None:
        pending = [name for name in _SIZING_PARAMS if base[name] is None]
        check = CheckResult(
            passed=False, code=PARAM_PENDING, detail={"check": SIZING_OK, "pending": pending}
        )
        return SizingResult(**base, check=check)

    stop_distance = entry_ref - stop_price
    if stop_distance <= 0:
        check = CheckResult(
            passed=False,
            code=INVALID_STOP,
            detail={
                "stop_distance": stop_distance,
                "entry_ref": entry_ref,
                "stop_price": stop_price,
            },
        )
        return SizingResult(**base, stop_distance=stop_distance, check=check)

    slippage = entry_ref * slippage_bps / BPS_PER_UNIT
    effective = stop_distance + slippage
    risk_amount = equity * risk_pct
    qty_risk = floor_divide(risk_amount, effective)
    qty_exposure = floor_divide(equity * exposure_pct, entry_ref)
    qty_buying_power = floor_divide(buying_power, entry_ref + slippage)
    candidates: tuple[tuple[BindingConstraint, int], ...] = (
        ("risk", qty_risk),
        ("exposure", qty_exposure),
        ("buying_power", qty_buying_power),
    )
    binding, qty = min(candidates, key=lambda item: item[1])

    passed = qty >= params.min_qty
    check = CheckResult(
        passed=passed,
        code=SIZING_OK if passed else QTY_TOO_SMALL,
        detail={"qty": qty, "min_qty": params.min_qty, "binding_constraint": binding},
    )
    return SizingResult(
        **base,
        stop_distance=stop_distance,
        slippage=slippage,
        effective_risk_per_share=effective,
        risk_amount=risk_amount,
        qty_risk=qty_risk,
        qty_exposure=qty_exposure,
        qty_buying_power=qty_buying_power,
        qty=qty,
        binding_constraint=binding,
        check=check,
    )


def build_proposed_trade(
    sizing: SizingResult, *, signal_id: str, symbol: str, take_profit_price: Decimal
) -> ProposedTrade:
    """Build the :class:`~domain.models.ProposedTrade` of an accepted sizing.

    * ``risk_per_share`` = ``effective_risk_per_share`` (stop distance + slippage).
    * ``risk_amount`` = ``qty * risk_per_share`` (the trade's actual monetary risk,
      ``<= equity * risk_per_trade_pct`` by construction).
    * ``risk_pct_of_equity`` = ``risk_amount / equity`` (fraction).
    * ``r_multiple`` = ``(take_profit_price - entry_ref) / stop_distance`` (R as in
      sec. 39; slightly below ``take_profit_r_multiple`` because of rounding down).

    Raises:
        NonRetryableError: code ``INVALID_RISK_INPUT`` if ``sizing`` was rejected; a
            rejected sizing must never become a trade.
    """
    require_decimal("take_profit_price", take_profit_price)
    if (
        not sizing.check.passed
        or sizing.qty is None
        or sizing.effective_risk_per_share is None
        or sizing.stop_distance is None
    ):
        raise NonRetryableError(
            f"cannot build a trade from a rejected sizing ({sizing.check.code})",
            code=INVALID_RISK_INPUT,
        )
    risk_amount = sizing.effective_risk_per_share * sizing.qty
    return ProposedTrade(
        signal_id=signal_id,
        symbol=symbol,
        qty=sizing.qty,
        entry_ref=sizing.entry_ref,
        stop_price=sizing.stop_price,
        take_profit_price=take_profit_price,
        risk_per_share=sizing.effective_risk_per_share,
        risk_amount=risk_amount,
        risk_pct_of_equity=risk_amount / sizing.equity,
        r_multiple=(take_profit_price - sizing.entry_ref) / sizing.stop_distance,
    )
