"""Stop loss and take profit levels for long entries (sec. 16.1-16.4).

Only the Risk Engine computes exit levels; Claude never sends prices (sec. 16).

Design choice (documented once, used by every public function here):

* Business rejections never raise. :func:`compute_exit_levels` always returns a frozen
  :class:`ExitLevelsResult` that carries every input, every intermediate value and a
  :class:`~domain.models.CheckResult` (``check``). Callers inspect ``check.passed``.
  This keeps the full calculation available for ``risk_events`` (sec. 15.2.4, 19) even
  when the levels are rejected.
* Contract violations (a non-``Decimal`` or non-finite number, a non-positive price
  passed to :func:`price_increment`) raise :class:`~domain.errors.NonRetryableError`
  with code ``INVALID_RISK_INPUT``: they are programming or adapter errors, not market
  conditions.

Check codes:

* ``EXIT_LEVELS_VALID``: levels passed sec. 16.4 (same name as the sec. 19 check).
* ``INVALID_EXIT_LEVELS``: at least one coherence condition failed.
* ``PARAM_PENDING``: a required ``OWNER_DECISION`` parameter is still ``None``.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal
from typing import Annotated, Any, Final

from pydantic import Field

from domain.errors import NonRetryableError
from domain.models import CheckResult, DomainModel, Money, PositiveDecimal

__all__ = [
    "EXIT_LEVELS_VALID",
    "INCREMENT_AT_OR_ABOVE_THRESHOLD",
    "INCREMENT_BELOW_THRESHOLD",
    "INVALID_EXIT_LEVELS",
    "INVALID_RISK_INPUT",
    "PARAM_PENDING",
    "SUB_PENNY_THRESHOLD",
    "ExitLevelsResult",
    "ExitParams",
    "check_exit_levels",
    "compute_exit_levels",
    "floor_divide",
    "price_increment",
    "require_decimal",
    "round_down_to_increment",
]

EXIT_LEVELS_VALID: Final = "EXIT_LEVELS_VALID"
INVALID_EXIT_LEVELS: Final = "INVALID_EXIT_LEVELS"
PARAM_PENDING: Final = "PARAM_PENDING"
INVALID_RISK_INPUT: Final = "INVALID_RISK_INPUT"

# Broker price-increment rule (sec. 16.3.1). VERIFICAR against current Alpaca docs.
SUB_PENNY_THRESHOLD: Final = Decimal("1")
"""Prices at or above this value use :data:`INCREMENT_AT_OR_ABOVE_THRESHOLD`."""
INCREMENT_AT_OR_ABOVE_THRESHOLD: Final = Decimal("0.01")
"""Minimum price increment for prices >= $1 (at most 2 decimals)."""
INCREMENT_BELOW_THRESHOLD: Final = Decimal("0.0001")
"""Minimum price increment for prices < $1 (at most 4 decimals)."""


class ExitParams(DomainModel):
    """Exit parameters of ``strategy.exit`` in ``config.yaml`` (sec. 7.3).

    Every field is required and may be ``None`` while the ``OWNER_DECISION`` is
    pending; evaluating with a pending value fails closed with ``PARAM_PENDING``.
    No default is provided on purpose: values come from configuration only.
    """

    stop_atr_multiplier: PositiveDecimal | None
    """Stop distance in ATR units (sec. 16.1)."""
    take_profit_r_multiple: PositiveDecimal | None
    """Take-profit distance in R, where R = ``entry_ref - stop_price`` (sec. 16.2)."""
    min_tp_distance_ticks: Annotated[int, Field(ge=1)] | None
    """Minimum ``take_profit_price - entry_ref`` in price increments (sec. 16.4)."""


class ExitLevelsResult(DomainModel):
    """Inputs, intermediate values, rounded levels and verdict of an exit calculation.

    ``stop_price`` and ``take_profit_price`` are already rounded down to a valid
    increment (sec. 16.3) and are the values the sizer must use (sec. 16.3.4). They are
    ``None`` when they could not be computed (pending parameter or non-positive raw
    stop). They are only safe to use when ``check.passed`` is ``True``.
    """

    entry_ref: Money
    atr: Money
    stop_atr_multiplier: Money | None
    take_profit_r_multiple: Money | None
    min_tp_distance_ticks: int | None
    stop_price_raw: Money | None = None
    stop_price: Money | None = None
    take_profit_price_raw: Money | None = None
    take_profit_price: Money | None = None
    check: CheckResult


def require_decimal(name: str, value: object) -> Decimal:
    """Return ``value`` if it is a finite ``Decimal``; raise otherwise.

    Floats (and anything else) are rejected so binary rounding never reaches risk
    calculations (sec. 57).

    Raises:
        NonRetryableError: code ``INVALID_RISK_INPUT``.
    """
    if not isinstance(value, Decimal) or not value.is_finite():
        raise NonRetryableError(
            f"{name} must be a finite Decimal, got {type(value).__name__}",
            code=INVALID_RISK_INPUT,
        )
    return value


def floor_divide(numerator: Decimal, denominator: Decimal) -> int:
    """Exact ``floor(numerator / denominator)`` for Decimals of any sign.

    ``Decimal.__floordiv__`` truncates toward zero and is exact, so the result never
    suffers from the context precision rounding a true quotient up to the next
    integer.

    Raises:
        NonRetryableError: code ``INVALID_RISK_INPUT`` if ``denominator`` is zero.
    """
    if denominator == 0:
        raise NonRetryableError("division by zero in floor_divide", code=INVALID_RISK_INPUT)
    quotient = numerator // denominator
    if numerator % denominator != 0 and (numerator < 0) != (denominator < 0):
        quotient -= 1
    return int(quotient)


def price_increment(price: Decimal) -> Decimal:
    """Return the minimum valid price increment for ``price`` (sec. 16.3.1).

    Rule: ``price >= $1`` -> ``0.01``; ``price < $1`` -> ``0.0001``.
    VERIFICAR: the current Alpaca/exchange rule before going live (sec. 60).

    Raises:
        NonRetryableError: code ``INVALID_RISK_INPUT`` if ``price`` is not a finite,
            strictly positive Decimal.
    """
    require_decimal("price", price)
    if price <= 0:
        raise NonRetryableError(
            f"price must be > 0 to have an increment, got {price}", code=INVALID_RISK_INPUT
        )
    if price >= SUB_PENNY_THRESHOLD:
        return INCREMENT_AT_OR_ABOVE_THRESHOLD
    return INCREMENT_BELOW_THRESHOLD


def round_down_to_increment(price: Decimal, increment: Decimal | None = None) -> Decimal:
    """Round ``price`` down (toward -inf) to a multiple of ``increment``.

    Args:
        price: Price to round. Must be a finite, strictly positive Decimal when
            ``increment`` is omitted.
        increment: Explicit increment; defaults to :func:`price_increment` of
            ``price``. Rounding down never crosses the $1 threshold upward, so the
            result is valid for its own increment.

    Returns:
        The rounded price, quantized to the exponent of ``increment``.

    Raises:
        NonRetryableError: code ``INVALID_RISK_INPUT`` on invalid input.
    """
    require_decimal("price", price)
    step = price_increment(price) if increment is None else require_decimal("increment", increment)
    if step <= 0:
        raise NonRetryableError(f"increment must be > 0, got {step}", code=INVALID_RISK_INPUT)
    rounded = Decimal(floor_divide(price, step)) * step
    return rounded.quantize(step, rounding=ROUND_FLOOR)


def _is_on_increment(price: Decimal) -> bool:
    """True if ``price`` is an exact multiple of its own valid increment."""
    return price % price_increment(price) == 0


def check_exit_levels(
    entry_ref: Decimal,
    stop_price: Decimal,
    take_profit_price: Decimal,
    min_tp_distance_ticks: int | None,
) -> CheckResult:
    """Coherence checks of a long bracket's exit levels (sec. 16.4).

    Conditions (all evaluated; every failing one is listed in ``detail["failed"]``):

    * ``stop_price > 0``
    * ``stop_price < entry_ref``
    * ``take_profit_price > entry_ref``
    * ``take_profit_price - entry_ref >= min_tp_distance_ticks * increment``, where
      ``increment`` is :func:`price_increment` of ``take_profit_price`` (the larger of
      the entry and TP increments, since TP > entry: the conservative choice).
    * ``stop_price`` and ``take_profit_price`` lie on their valid increment (sec. 16.3;
      stricter than 16.4, the broker would reject them otherwise).

    This is the single implementation of the 16.4 rule; the ``EXIT_LEVELS_VALID``
    check of ``domain/guards/checks.py`` must delegate here (sec. 19).

    Returns:
        ``CheckResult`` with code ``EXIT_LEVELS_VALID`` (pass), ``INVALID_EXIT_LEVELS``
        (fail) or ``PARAM_PENDING`` (``min_tp_distance_ticks`` is ``None``).
    """
    require_decimal("entry_ref", entry_ref)
    require_decimal("stop_price", stop_price)
    require_decimal("take_profit_price", take_profit_price)
    detail: dict[str, Any] = {
        "entry_ref": entry_ref,
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "min_tp_distance_ticks": min_tp_distance_ticks,
    }
    if min_tp_distance_ticks is None:
        return CheckResult(
            passed=False,
            code=PARAM_PENDING,
            detail={**detail, "check": EXIT_LEVELS_VALID, "pending": ["min_tp_distance_ticks"]},
        )

    failed: list[str] = []
    if entry_ref <= 0:
        failed.append("entry_ref_positive")
    if stop_price <= 0:
        failed.append("stop_positive")
    elif not _is_on_increment(stop_price):
        failed.append("stop_on_increment")
    if stop_price >= entry_ref:
        failed.append("stop_below_entry")
    if take_profit_price <= entry_ref:
        failed.append("tp_above_entry")
    if take_profit_price > 0:
        tp_increment = price_increment(take_profit_price)
        min_distance = tp_increment * min_tp_distance_ticks
        detail["min_tp_distance"] = min_distance
        if take_profit_price - entry_ref < min_distance:
            failed.append("tp_min_distance")
        if not _is_on_increment(take_profit_price):
            failed.append("tp_on_increment")

    detail["failed"] = failed
    passed = not failed
    return CheckResult(
        passed=passed, code=EXIT_LEVELS_VALID if passed else INVALID_EXIT_LEVELS, detail=detail
    )


def compute_exit_levels(entry_ref: Decimal, atr: Decimal, params: ExitParams) -> ExitLevelsResult:
    """Compute ATR stop and R-multiple take profit for a long entry (sec. 16.1-16.4).

    Formulas::

        stop_price_raw        = entry_ref - atr * stop_atr_multiplier
        stop_price            = round_down_to_increment(stop_price_raw)
        take_profit_price_raw = entry_ref + (entry_ref - stop_price) * take_profit_r_multiple
        take_profit_price     = round_down_to_increment(take_profit_price_raw)

    The take profit is derived from the *rounded* stop, so R is the distance the
    sizer will actually use (sec. 16.3.4).

    Args:
        entry_ref: Entry reference price (signal bar close or limit price, sec. 15.2).
        atr: Last ATR value as a Decimal (convert float indicators with
            ``Decimal(str(x))`` at the boundary).
        params: Exit parameters; any ``None`` field fails closed with ``PARAM_PENDING``.

    Returns:
        An :class:`ExitLevelsResult`; use it only if ``result.check.passed``.

    Raises:
        NonRetryableError: code ``INVALID_RISK_INPUT`` for non-Decimal/non-finite input.
    """
    require_decimal("entry_ref", entry_ref)
    require_decimal("atr", atr)
    base: dict[str, Any] = {
        "entry_ref": entry_ref,
        "atr": atr,
        "stop_atr_multiplier": params.stop_atr_multiplier,
        "take_profit_r_multiple": params.take_profit_r_multiple,
        "min_tp_distance_ticks": params.min_tp_distance_ticks,
    }
    multiplier = params.stop_atr_multiplier
    r_multiple = params.take_profit_r_multiple
    ticks = params.min_tp_distance_ticks
    if multiplier is None or r_multiple is None or ticks is None:
        pending = [name for name, value in base.items() if value is None]
        check = CheckResult(
            passed=False,
            code=PARAM_PENDING,
            detail={**base, "check": EXIT_LEVELS_VALID, "pending": pending},
        )
        return ExitLevelsResult(**base, check=check)

    stop_raw = entry_ref - atr * multiplier
    if entry_ref <= 0 or stop_raw <= 0:
        reason = "entry_ref_positive" if entry_ref <= 0 else "stop_positive"
        check = CheckResult(
            passed=False,
            code=INVALID_EXIT_LEVELS,
            detail={**base, "stop_price_raw": stop_raw, "failed": [reason]},
        )
        return ExitLevelsResult(**base, stop_price_raw=stop_raw, check=check)

    stop = round_down_to_increment(stop_raw)
    tp_raw = entry_ref + (entry_ref - stop) * r_multiple
    tp = round_down_to_increment(tp_raw) if tp_raw > 0 else tp_raw
    coherence = check_exit_levels(entry_ref, stop, tp, ticks)
    detail = {
        **coherence.detail,
        **base,
        "stop_price_raw": stop_raw,
        "take_profit_price_raw": tp_raw,
    }
    check = CheckResult(passed=coherence.passed, code=coherence.code, detail=detail)
    return ExitLevelsResult(
        **base,
        stop_price_raw=stop_raw,
        stop_price=stop,
        take_profit_price_raw=tp_raw,
        take_profit_price=tp,
        check=check,
    )
