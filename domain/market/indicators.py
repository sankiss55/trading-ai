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

import itertools
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from domain.errors import DomainError
from domain.models import Bar, BarStatus

__all__ = [
    "IndicatorInputError",
    "IndicatorInputs",
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
    "shared_run",
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

_EXACT_INT = 2**53
"""Largest magnitude below which every ``int`` converts to ``float`` exactly."""


class IndicatorInputError(DomainError):
    """Invalid indicator input (bad period, non-finite value, mismatched lengths)."""


# --------------------------------------------------------------------------- helpers


def _check_period(period: int) -> None:
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise IndicatorInputError(
            f"indicator period must be an int >= 1, got {period!r}", code=INVALID_PERIOD_CODE
        )


def _to_floats(values: Sequence[Numeric | None], name: str = "values") -> list[float]:
    # Fast path: one conversion pass plus one finiteness check of the whole list. Any
    # anomaly (None, bool, non-numeric, non-finite) falls back to the checking loop
    # below, which raises exactly as before; valid inputs give the same floats.
    try:
        fast = [float(value) for value in values if type(value) is not bool]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        fast = None
    if fast is not None and len(fast) == len(values) and math.isfinite(sum(fast)):
        return fast
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

    ``out[i]`` is ``None`` for ``i < period - 1``. Each window sum is the correctly
    rounded exact sum (``math.fsum``), so the result does not accumulate floating-point
    drift. For ``int`` inputs exactly representable as floats (``|v| <= 2**53``, e.g.
    volumes) the exact window sum is kept as a running integer and rounded once with
    ``float(int)``; both are the correctly rounded (half-even) value of the same exact
    sum, so the output is bit-identical to the ``fsum`` path, in O(n).
    """
    _check_period(period)
    data = _to_floats(values)
    ints = _exact_ints(values)
    if ints is not None:
        return _sma_exact_ints(ints, period)
    out: list[float | None] = [None] * len(data)
    for i in range(period - 1, len(data)):
        out[i] = math.fsum(data[i - period + 1 : i + 1]) / period
    return tuple(out)


def _exact_ints(values: Sequence[object]) -> list[int] | None:
    """``values`` as ints if every one is an ``int`` (not bool) with ``|v| <= 2**53``."""
    ints = [value for value in values if type(value) is int]
    if len(ints) != len(values) or any(not -_EXACT_INT <= value <= _EXACT_INT for value in ints):
        return None
    return ints


def _sma_exact_ints(values: Sequence[int], period: int) -> IndicatorSeries:
    """SMA core for ints with ``|v| <= 2**53`` (running exact sum, see :func:`sma`)."""
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return tuple(out)
    total = sum(values[:period])
    out[period - 1] = float(total) / period
    for i in range(period, len(values)):
        total += values[i] - values[i - period]
        out[i] = float(total) / period
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
    return _ema_core(_to_floats(values), period)


def _ema_core(data: Sequence[float], period: int) -> IndicatorSeries:
    out: list[float | None] = [None] * len(data)
    if len(data) < period:
        return tuple(out)
    alpha = 2.0 / (period + 1)
    keep = 1.0 - alpha
    current = math.fsum(data[:period]) / period
    out[period - 1] = current
    for i in range(period, len(data)):
        current = alpha * data[i] + keep * current
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
    if len(data) < period + 1:
        return (None,) * len(data)
    gains, losses = _price_changes(data)
    return _rsi_core(len(data), gains, losses, period)


def _price_changes(data: Sequence[float]) -> tuple[list[float], list[float]]:
    """``(gains, losses)``: ``[k]`` is the change from ``data[k]`` to ``data[k + 1]``."""
    pairs = list(itertools.pairwise(data))
    return (
        [max(cur - prev, 0.0) for prev, cur in pairs],
        [max(prev - cur, 0.0) for prev, cur in pairs],
    )


def _rsi_core(
    count: int, gains: Sequence[float], losses: Sequence[float], period: int
) -> IndicatorSeries:
    """RSI of ``count`` closes from their ``count - 1`` price changes."""
    out: list[float | None] = [None] * count
    if count < period + 1:
        return tuple(out)
    avg_gain = math.fsum(gains[:period]) / period
    avg_loss = math.fsum(losses[:period]) / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    weight = period - 1
    for i in range(period + 1, count):
        avg_gain = (avg_gain * weight + gains[i - 1]) / period
        avg_loss = (avg_loss * weight + losses[i - 1]) / period
        # _rsi_value(avg_gain, avg_loss), inlined (hot loop): same operations.
        if avg_loss == 0.0:
            out[i] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
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
    for i, (high, low) in enumerate(zip(h, lo, strict=True)):
        if low > high:
            raise IndicatorInputError(
                f"low[{i}]={low} is above high[{i}]={high}", code=INVALID_INPUT_CODE
            )
    if not h:
        return ()
    return (h[0] - lo[0], *_linked_true_ranges(h, lo, c))


def _linked_true_ranges(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[float]:
    """True range of bars ``1..n-1``, each against the previous bar's close."""
    return [
        max(high - low, abs(high - prev_close), abs(low - prev_close))
        for high, low, prev_close in zip(highs[1:], lows[1:], closes, strict=False)
    ]


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
    return _atr_core(true_range(highs, lows, closes), period)


def _atr_core(tr: Sequence[float], period: int) -> IndicatorSeries:
    out: list[float | None] = [None] * len(tr)
    if len(tr) < period:
        return tuple(out)
    current = math.fsum(tr[:period]) / period
    out[period - 1] = current
    weight = period - 1
    for i in range(period, len(tr)):
        current = (current * weight + tr[i]) / period
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


# --------------------------------------------------------------------------- incremental inputs


@dataclass(frozen=True, slots=True)
class IndicatorInputs:
    """Validated indicator inputs of a run of priced bars, reusable across windows.

    A sliding window shares all but its newest bars with the previous one. The per-bar
    inputs (prices as float, the true range and the price change of each bar against
    the previous one) do not depend on where the window starts, so :meth:`from_bars`
    carries them over for the bars shared with ``previous`` and converts only the new
    ones. The recursive indicators (EMA, RSI, ATR) do depend on the window start (their
    seed) and are recomputed by the **same** cores as the batch functions above: each
    method returns exactly what the batch function returns on the same bars (sec. 14.5:
    one implementation, see ``tests/unit/market/test_indicators.py``).

    Attributes:
        bars: The priced bars, oldest first.
        opens: ``bar.open`` of each bar (``Decimal``); likewise ``highs``, ``lows`` and
            ``closes``.
        volumes: ``bar.volume`` of each bar (``0 <= v <= 2**53``).
        high_floats: ``float(high)`` of each bar, finite; likewise ``low_floats`` and
            ``close_floats``.
        linked_true_ranges: ``[k]`` = true range of ``bars[k + 1]`` against the close of
            ``bars[k]``.
        gains: ``[k]`` = gain from ``bars[k]`` to ``bars[k + 1]``; likewise ``losses``.
    """

    bars: tuple[Bar, ...]
    opens: tuple[Decimal, ...]
    highs: tuple[Decimal, ...]
    lows: tuple[Decimal, ...]
    closes: tuple[Decimal, ...]
    volumes: tuple[int, ...]
    high_floats: tuple[float, ...]
    low_floats: tuple[float, ...]
    close_floats: tuple[float, ...]
    linked_true_ranges: tuple[float, ...]
    gains: tuple[float, ...]
    losses: tuple[float, ...]

    @classmethod
    def from_bars(
        cls, bars: Sequence[Bar], previous: IndicatorInputs | None = None
    ) -> IndicatorInputs | None:
        """Inputs of ``bars`` (priced bars, oldest first), reusing ``previous``.

        The bars shared with ``previous`` (the same objects in the same order, with
        ``bars[0]`` inside ``previous.bars``) are not converted again. Returns ``None``
        when a bar cannot be used as finite floats (EMPTY or missing price, non-finite
        float, ``low > high`` once converted, volume that is not an ``int`` within
        ``2**53``): callers then use the batch functions, which report such input
        exactly as documented.
        """
        current = tuple(bars)
        shared, offset = _shared_prefix(previous, current)
        tail = current[shared:]
        opens: list[Decimal] = []
        highs: list[Decimal] = []
        lows: list[Decimal] = []
        closes: list[Decimal] = []
        raw_volumes: list[int] = []
        for bar in tail:
            if (
                bar.status is BarStatus.EMPTY
                or bar.open is None
                or bar.high is None
                or bar.low is None
                or bar.close is None
            ):
                return None
            opens.append(bar.open)
            highs.append(bar.high)
            lows.append(bar.low)
            closes.append(bar.close)
            raw_volumes.append(bar.volume)
        volumes = _exact_ints(raw_volumes)
        high_f = [float(value) for value in highs]
        low_f = [float(value) for value in lows]
        close_f = [float(value) for value in closes]
        if (
            volumes is None
            or not math.isfinite(sum(high_f) + sum(low_f) + sum(close_f))
            or any(low > high for high, low in zip(high_f, low_f, strict=True))
        ):
            return None
        if previous is None or not shared:
            gains, losses = _price_changes(close_f)
            return cls(
                bars=current,
                opens=tuple(opens),
                highs=tuple(highs),
                lows=tuple(lows),
                closes=tuple(closes),
                volumes=tuple(volumes),
                high_floats=tuple(high_f),
                low_floats=tuple(low_f),
                close_floats=tuple(close_f),
                linked_true_ranges=tuple(_linked_true_ranges(high_f, low_f, close_f)),
                gains=tuple(gains),
                losses=tuple(losses),
            )
        keep = slice(offset, offset + shared)
        pairs = slice(offset, offset + shared - 1)
        last = offset + shared - 1
        # The new pairs start at the last shared bar (its close is the previous close).
        link_h = [previous.high_floats[last], *high_f]
        link_l = [previous.low_floats[last], *low_f]
        link_c = [previous.close_floats[last], *close_f]
        gains, losses = _price_changes(link_c)
        return cls(
            bars=current,
            opens=previous.opens[keep] + tuple(opens),
            highs=previous.highs[keep] + tuple(highs),
            lows=previous.lows[keep] + tuple(lows),
            closes=previous.closes[keep] + tuple(closes),
            volumes=previous.volumes[keep] + tuple(volumes),
            high_floats=previous.high_floats[keep] + tuple(high_f),
            low_floats=previous.low_floats[keep] + tuple(low_f),
            close_floats=previous.close_floats[keep] + tuple(close_f),
            linked_true_ranges=previous.linked_true_ranges[pairs]
            + tuple(_linked_true_ranges(link_h, link_l, link_c)),
            gains=previous.gains[pairs] + tuple(gains),
            losses=previous.losses[pairs] + tuple(losses),
        )

    def ema(self, period: int) -> IndicatorSeries:
        """:func:`ema` of the closes."""
        _check_period(period)
        return _ema_core(self.close_floats, period)

    def rsi(self, period: int) -> IndicatorSeries:
        """:func:`rsi` of the closes."""
        _check_period(period)
        return _rsi_core(len(self.close_floats), self.gains, self.losses, period)

    def true_range(self) -> tuple[float, ...]:
        """:func:`true_range` of the bars."""
        if not self.bars:
            return ()
        return (self.high_floats[0] - self.low_floats[0], *self.linked_true_ranges)

    def atr(self, period: int) -> IndicatorSeries:
        """:func:`atr` of the bars."""
        _check_period(period)
        return _atr_core(self.true_range(), period)

    def volume_average(self, period: int) -> IndicatorSeries:
        """:func:`volume_average` of the volumes."""
        _check_period(period)
        return _sma_exact_ints(self.volumes, period)


def _shared_prefix(previous: IndicatorInputs | None, bars: tuple[Bar, ...]) -> tuple[int, int]:
    return (0, 0) if previous is None else shared_run(previous.bars, bars)


def shared_run(previous: Sequence[object], current: Sequence[object]) -> tuple[int, int]:
    """``(count, offset)``: ``current[:count]`` are the very objects (``is``) of
    ``previous[offset:offset + count]``, where ``offset`` is the first position of
    ``current[0]`` in ``previous``; ``(0, 0)`` when there is no such run."""
    if not current or not previous:
        return 0, 0
    first = current[0]
    offset = next((i for i, item in enumerate(previous) if item is first), None)
    if offset is None:
        return 0, 0
    count = min(len(previous) - offset, len(current))
    if all(map(operator.is_, previous[offset : offset + count], current[:count])):
        return count, offset
    return 0, 0


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
