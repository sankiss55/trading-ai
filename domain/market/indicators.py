"""Deterministic technical indicators (sec. 14).

Rules applied by every function in this module:

* Pure: input = a sequence of values taken from closed bars + parameters; output = a
  tuple of values aligned 1:1 with the input. No I/O, no global state, no LLM.
* No lookahead: the value at index ``i`` uses only inputs at indices ``<= i``. Appending
  future values never changes past outputs.
* Insufficient data: positions without enough history are ``None``. Any rule that
  consumes a ``None`` must evaluate to ``False`` (sec. 14.3); that is the caller's job.
* Minimum data: each indicator exposes ``<name>_min_bars(period)``, the number of input
  values needed before the first non-``None`` output.
* Numeric type: indicators are computed in ``float`` (allowed by sec. 6.1). ``Decimal``
  and ``int`` inputs are converted with ``float()`` at the boundary. Indicators never
  feed order prices directly; anything that becomes a price (e.g. an ATR-based stop) is
  converted back to ``Decimal`` and rounded by the risk engine.
* Inputs must be finite numbers. ``None``/NaN/infinity raise ``IndicatorInputError``:
  EMPTY bars carry no prices, so the caller decides how to treat them before calling an
  indicator (the bar helpers below refuse EMPTY bars explicitly).

Variants (sec. 14.7) are documented on each function because libraries differ.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal

from domain.errors import DomainError
from domain.models import Bar, BarStatus

__all__ = [
    "IndicatorInputError",
    "IndicatorSeries",
    "Numeric",
    "atr",
    "atr_min_bars",
    "bar_closes",
    "bar_highs",
    "bar_lows",
    "bar_volumes",
    "ema",
    "ema_min_bars",
    "rsi",
    "rsi_min_bars",
    "sma",
    "sma_min_bars",
    "true_range",
    "true_range_min_bars",
    "volume_average",
    "volume_average_min_bars",
]

Numeric = Decimal | float | int
"""Accepted input value type."""

IndicatorSeries = tuple[float | None, ...]
"""Indicator output aligned with its input; ``None`` where data is insufficient."""

INVALID_PERIOD_CODE = "INDICATOR_INVALID_PERIOD"
INVALID_INPUT_CODE = "INDICATOR_INVALID_INPUT"


class IndicatorInputError(DomainError):
    """Invalid indicator input (bad period, non-finite value, mismatched lengths)."""


# --------------------------------------------------------------------------- helpers


def _check_period(period: int) -> None:
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise IndicatorInputError(
            f"indicator period must be an int >= 1, got {period!r}", code=INVALID_PERIOD_CODE
        )


def _to_floats(values: Sequence[Numeric | None], name: str = "values") -> list[float]:
    out: list[float] = []
    for index, value in enumerate(values):
        if value is None or isinstance(value, bool):
            raise IndicatorInputError(
                f"{name}[{index}] must be a number, got {value!r}", code=INVALID_INPUT_CODE
            )
        number = float(value)
        if not math.isfinite(number):
            raise IndicatorInputError(
                f"{name}[{index}] must be finite, got {value!r}", code=INVALID_INPUT_CODE
            )
        out.append(number)
    return out


def _check_same_length(**series: Sequence[Numeric]) -> None:
    lengths = {name: len(values) for name, values in series.items()}
    if len(set(lengths.values())) > 1:
        raise IndicatorInputError(
            f"input series must have the same length, got {lengths}", code=INVALID_INPUT_CODE
        )


# --------------------------------------------------------------------------- SMA


def sma_min_bars(period: int) -> int:
    """Number of values required before the first non-``None`` SMA value."""
    _check_period(period)
    return period


def sma(values: Sequence[Numeric], period: int) -> IndicatorSeries:
    """Simple moving average: arithmetic mean of the last ``period`` values.

    ``out[i]`` is ``None`` for ``i < period - 1``. Each window is summed directly (no
    running sum) so the result does not accumulate floating-point drift.
    """
    _check_period(period)
    data = _to_floats(values)
    out: list[float | None] = [None] * len(data)
    for i in range(period - 1, len(data)):
        out[i] = math.fsum(data[i - period + 1 : i + 1]) / period
    return tuple(out)


# --------------------------------------------------------------------------- EMA


def ema_min_bars(period: int) -> int:
    """Number of values required before the first non-``None`` EMA value."""
    _check_period(period)
    return period


def ema(values: Sequence[Numeric], period: int) -> IndicatorSeries:
    """Exponential moving average, SMA-seeded.

    Variant: ``alpha = 2 / (period + 1)``. The first value, at index ``period - 1``, is
    the SMA of the first ``period`` inputs; afterwards
    ``ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]``. Earlier indices are ``None``.
    """
    _check_period(period)
    data = _to_floats(values)
    out: list[float | None] = [None] * len(data)
    if len(data) < period:
        return tuple(out)
    alpha = 2.0 / (period + 1)
    current = math.fsum(data[:period]) / period
    out[period - 1] = current
    for i in range(period, len(data)):
        current = alpha * data[i] + (1.0 - alpha) * current
        out[i] = current
    return tuple(out)


# --------------------------------------------------------------------------- RSI


def rsi_min_bars(period: int) -> int:
    """Number of closes required before the first non-``None`` RSI value.

    ``period`` price changes are needed, hence ``period + 1`` closes.
    """
    _check_period(period)
    return period + 1


def rsi(closes: Sequence[Numeric], period: int) -> IndicatorSeries:
    """Relative Strength Index with Wilder smoothing, in ``[0, 100]``.

    Variant (Wilder):

    * ``change[i] = close[i] - close[i - 1]``; gain = max(change, 0), loss = max(-change, 0).
    * Seed at index ``period``: simple mean of the first ``period`` gains and losses
      (changes 1..period).
    * Then ``avg = (avg_prev * (period - 1) + current) / period`` for gains and losses.
    * ``RSI = 100 - 100 / (1 + avg_gain / avg_loss)``.
    * Degenerate cases: ``avg_loss == 0 and avg_gain > 0`` -> ``100.0``;
      ``avg_loss == 0 and avg_gain == 0`` (flat prices) -> ``50.0`` (neutral).

    Indices ``< period`` are ``None``.
    """
    _check_period(period)
    data = _to_floats(closes, "closes")
    out: list[float | None] = [None] * len(data)
    if len(data) < period + 1:
        return tuple(out)
    gains = [max(data[i] - data[i - 1], 0.0) for i in range(1, len(data))]
    losses = [max(data[i - 1] - data[i], 0.0) for i in range(1, len(data))]
    avg_gain = math.fsum(gains[:period]) / period
    avg_loss = math.fsum(losses[:period]) / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(data)):
        change_index = i - 1  # gains[k] is the change from close[k] to close[k + 1]
        avg_gain = (avg_gain * (period - 1) + gains[change_index]) / period
        avg_loss = (avg_loss * (period - 1) + losses[change_index]) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return tuple(out)


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


# --------------------------------------------------------------------------- True range / ATR


def true_range_min_bars() -> int:
    """Number of bars required before the first true-range value (the first bar uses H-L)."""
    return 1


def true_range(
    highs: Sequence[Numeric], lows: Sequence[Numeric], closes: Sequence[Numeric]
) -> tuple[float, ...]:
    """True range per bar.

    ``tr[0] = high[0] - low[0]`` (no previous close);
    ``tr[i] = max(high[i] - low[i], |high[i] - close[i-1]|, |low[i] - close[i-1]|)``.
    """
    _check_same_length(highs=highs, lows=lows, closes=closes)
    h = _to_floats(highs, "highs")
    lo = _to_floats(lows, "lows")
    c = _to_floats(closes, "closes")
    out: list[float] = []
    for i in range(len(h)):
        if lo[i] > h[i]:
            raise IndicatorInputError(
                f"low[{i}]={lo[i]} is above high[{i}]={h[i]}", code=INVALID_INPUT_CODE
            )
        if i == 0:
            out.append(h[0] - lo[0])
        else:
            prev_close = c[i - 1]
            out.append(max(h[i] - lo[i], abs(h[i] - prev_close), abs(lo[i] - prev_close)))
    return tuple(out)


def atr_min_bars(period: int) -> int:
    """Number of bars required before the first non-``None`` ATR value."""
    _check_period(period)
    return period


def atr(
    highs: Sequence[Numeric],
    lows: Sequence[Numeric],
    closes: Sequence[Numeric],
    period: int,
) -> IndicatorSeries:
    """Average True Range with Wilder smoothing.

    Variant (Wilder's original): true range as in :func:`true_range` (first bar uses
    ``high - low``). Seed at index ``period - 1`` = simple mean of ``tr[0..period-1]``;
    then ``atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period``. Earlier indices are
    ``None``. Note: TA-Lib instead skips ``tr[0]`` and starts one bar later.
    """
    _check_period(period)
    tr = true_range(highs, lows, closes)
    out: list[float | None] = [None] * len(tr)
    if len(tr) < period:
        return tuple(out)
    current = math.fsum(tr[:period]) / period
    out[period - 1] = current
    for i in range(period, len(tr)):
        current = (current * (period - 1) + tr[i]) / period
        out[i] = current
    return tuple(out)


# --------------------------------------------------------------------------- volume average


def volume_average_min_bars(period: int) -> int:
    """Number of bars required before the first non-``None`` volume average."""
    return sma_min_bars(period)


def volume_average(volumes: Sequence[Numeric], period: int) -> IndicatorSeries:
    """Average volume: SMA of the last ``period`` volumes (see :func:`sma`).

    Whether the current bar is included in its own comparison baseline (e.g.
    ``volume > volume_avg`` of the previous bars) is a strategy decision: pass the
    appropriate slice or compare against ``out[i - 1]``.
    """
    return sma(volumes, period)


# --------------------------------------------------------------------------- bar extraction


def _require_priced(bars: Sequence[Bar]) -> None:
    for index, bar in enumerate(bars):
        if bar.status is BarStatus.EMPTY:
            raise IndicatorInputError(
                f"bars[{index}] ({bar.symbol} {bar.bar_start_utc.isoformat()}) is EMPTY; "
                "filter EMPTY bars before computing indicators",
                code=INVALID_INPUT_CODE,
            )


def bar_closes(bars: Sequence[Bar]) -> tuple[Decimal, ...]:
    """Close prices of closed, non-EMPTY bars (raises on EMPTY bars)."""
    _require_priced(bars)
    return tuple(bar.close for bar in bars if bar.close is not None)


def bar_highs(bars: Sequence[Bar]) -> tuple[Decimal, ...]:
    """High prices of closed, non-EMPTY bars (raises on EMPTY bars)."""
    _require_priced(bars)
    return tuple(bar.high for bar in bars if bar.high is not None)


def bar_lows(bars: Sequence[Bar]) -> tuple[Decimal, ...]:
    """Low prices of closed, non-EMPTY bars (raises on EMPTY bars)."""
    _require_priced(bars)
    return tuple(bar.low for bar in bars if bar.low is not None)


def bar_volumes(bars: Sequence[Bar]) -> tuple[int, ...]:
    """Volumes of closed bars (EMPTY bars are rejected for consistency with prices)."""
    _require_priced(bars)
    return tuple(bar.volume for bar in bars)
