"""Feature frames: the series rules are evaluated on (sec. 13.2, 13.3, 14).

A :class:`FeatureFrame` holds, for one symbol and one timeframe, every series of
:data:`~domain.strategy.rules.SERIES_BASE_NAMES` aligned to the bars it was built from:

========== ============================================================================
Series     Definition
========== ============================================================================
open..     ``open``/``high``/``low``/``close`` (Decimal) and ``volume`` (int) of the bar
ema_fast   ``ema(closes, indicators.ema_fast)`` (SMA-seeded, see domain.market)
ema_slow   ``ema(closes, indicators.ema_slow)``
rsi        ``rsi(closes, indicators.rsi_period)`` (Wilder)
atr        ``atr(highs, lows, closes, indicators.atr_period)`` (Wilder)
volume_avg ``volume_average(volumes, indicators.volume_avg_period)``; **includes** the
           bar itself. Use ``{series: volume_avg, offset: -2}`` to compare a bar's volume
           with the average of the bars before it.
atr_pct    ``atr / close`` of the same bar (float)
gap_pct    ``abs(open_today - close_yesterday) / close_yesterday`` (Decimal), constant
           within a session: ``open_today`` is the open of the first priced bar of the
           bar's session and ``close_yesterday`` the close of the last priced bar of the
           previous session in ``sessions``. ``None`` when the bar is outside every given
           session, when the previous session has no priced bar in the input, or when no
           sessions are passed. On daily bars (one bar per session) it is the overnight
           gap of the day.
ibs        ``ibs(highs, lows, closes)``: internal bar strength
           ``(close - low) / (high - low)`` of the same bar (float, ``None`` when
           ``high == low``)
sma_short  ``sma(closes, indicators.sma_short_period)``; optional period
sma_long   ``sma(closes, indicators.sma_long_period)``; optional period
========== ============================================================================

Indicators come exclusively from :mod:`domain.market.indicators` (sec. 14.5, never
reimplemented). They are computed in float; prices stay Decimal.

EMPTY bars (sec. 10.3.4) carry no prices and are **excluded** before any series is
computed: offset ``-1`` is the last *priced* bar. INCOMPLETE bars carry prices and are
included. An indicator whose period is ``None`` yields an all-``None`` series; the
strategy refuses to start when a rule needs such a series (OWNER_DECISION pending).
``sma_short_period`` / ``sma_long_period`` are optional (default ``None`` = not used);
a rule referencing their series while the period is ``None`` is a configuration error.
"""

from __future__ import annotations

import itertools
import operator
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from functools import partial
from types import MappingProxyType

from pydantic import Field

from domain.market.indicators import (
    IndicatorInputs,
    IndicatorSeries,
    atr,
    atr_min_bars,
    bar_closes,
    bar_highs,
    bar_lows,
    bar_volumes,
    ema,
    ema_min_bars,
    ibs,
    rsi,
    rsi_min_bars,
    shared_run,
    sma,
    sma_min_bars,
    volume_average,
    volume_average_min_bars,
)
from domain.models import Bar, BarStatus, DomainModel, RuleValue, SessionDay, Timeframe
from domain.strategy.rules import (
    SERIES_BASE_NAMES,
    RuleTimeframe,
    StrategyConfigError,
    StrategyInputError,
    resolve_series_name,
)

__all__ = [
    "SERIES_PERIOD_PARAM",
    "FeatureFrame",
    "FeatureFrameMemo",
    "IndicatorContext",
    "IndicatorParams",
    "build_feature_frame",
    "build_indicator_context",
]

SERIES_PERIOD_PARAM: Mapping[str, str] = MappingProxyType(
    {
        "ema_fast": "ema_fast",
        "ema_slow": "ema_slow",
        "rsi": "rsi_period",
        "atr": "atr_period",
        "atr_pct": "atr_period",
        "volume_avg": "volume_avg_period",
        "sma_short": "sma_short_period",
        "sma_long": "sma_long_period",
    }
)
"""Base series -> ``strategy.indicators`` parameter it depends on."""


class IndicatorParams(DomainModel):
    """``strategy.indicators`` of config.yaml. ``None`` = OWNER_DECISION pending.

    ``sma_short_period`` and ``sma_long_period`` are optional keys (default ``None`` =
    the indicator is not used), so configurations written before they existed load
    unchanged.
    """

    ema_fast: int | None = Field(ge=1)
    ema_slow: int | None = Field(ge=1)
    rsi_period: int | None = Field(ge=1)
    atr_period: int | None = Field(ge=1)
    volume_avg_period: int | None = Field(ge=1)
    sma_short_period: int | None = Field(default=None, ge=1)
    sma_long_period: int | None = Field(default=None, ge=1)

    def min_bars(self, param: str) -> int:
        """Bars needed before the first non-``None`` value of the indicator ``param``.

        Returns ``0`` when the period is ``None`` (indicator not computed).
        """
        period: int | None = getattr(self, param)
        if period is None:
            return 0
        if param == "rsi_period":
            return rsi_min_bars(period)
        if param == "atr_period":
            return atr_min_bars(period)
        if param == "volume_avg_period":
            return volume_average_min_bars(period)
        if param in ("sma_short_period", "sma_long_period"):
            return sma_min_bars(period)
        return ema_min_bars(period)


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """Immutable series of one symbol/timeframe, aligned with :attr:`bars`.

    Attributes:
        symbol: Symbol of every bar.
        timeframe: Timeframe of every bar.
        bars: Priced bars (EMPTY excluded) in chronological order.
        series: Base series name -> tuple aligned with ``bars``.
    """

    symbol: str
    timeframe: Timeframe
    bars: tuple[Bar, ...]
    series: Mapping[str, tuple[RuleValue, ...]]

    def __len__(self) -> int:
        return len(self.bars)

    def value(self, name: str, offset: int) -> RuleValue:
        """Value of base series ``name`` at ``offset`` (``-1`` = last priced bar).

        Returns ``None`` when the offset is out of range or the value is not available.

        Raises:
            StrategyInputError: unknown series or non-negative offset (lookahead).
        """
        if name not in self.series:
            raise StrategyInputError(f"unknown series {name!r}")
        if offset > -1:
            raise StrategyInputError(f"offset must be <= -1 (no lookahead), got {offset}")
        values = self.series[name]
        if -offset > len(values):
            return None
        return values[offset]


_START = operator.attrgetter("bar_start_utc")
_END = operator.attrgetter("bar_end_utc")
_STATUS = operator.attrgetter("status")
_SYMBOL = operator.attrgetter("symbol")
_TIMEFRAME = operator.attrgetter("timeframe")


def _validate_bars(bars: Sequence[Bar], timeframe: Timeframe, *, valid_prefix: int = 0) -> None:
    """Raise on mixed symbols, another timeframe or non-increasing starts.

    ``valid_prefix``: ``bars[:valid_prefix]`` are known to pass (the same objects were
    validated together for this timeframe), so only the rest and its boundary are
    checked; any problem is then reported by the full check, exactly as without it.
    """
    if valid_prefix and _valid_tail(bars, timeframe, valid_prefix):
        return
    symbols = set(map(_SYMBOL, bars))
    if len(symbols) > 1:
        raise StrategyInputError(f"bars of several symbols in one frame: {sorted(symbols)}")
    starts = list(map(_START, bars))
    if all(map(operator.is_, map(_TIMEFRAME, bars), itertools.repeat(timeframe))) and all(
        map(operator.lt, starts, starts[1:])
    ):
        return
    for index, bar in enumerate(bars):  # report the first problem, in index order
        if bar.timeframe is not timeframe:
            raise StrategyInputError(
                f"bars[{index}] has timeframe {bar.timeframe}, expected {timeframe}"
            )
        if index and bar.bar_start_utc <= bars[index - 1].bar_start_utc:
            raise StrategyInputError(
                f"bars must be strictly increasing by bar_start_utc (index {index})"
            )


def _valid_tail(bars: Sequence[Bar], timeframe: Timeframe, valid_prefix: int) -> bool:
    tail = bars[valid_prefix - 1 :]  # includes the last valid bar: checks the boundary
    symbol = bars[0].symbol
    starts = list(map(_START, tail))
    return (
        all(map(operator.eq, map(_SYMBOL, tail), itertools.repeat(symbol)))
        and all(map(operator.is_, map(_TIMEFRAME, tail), itertools.repeat(timeframe)))
        and all(map(operator.lt, starts, starts[1:]))
    )


def _session_index(start: datetime, sessions: Sequence[SessionDay]) -> int | None:
    for index, session in enumerate(sessions):
        if session.open_utc <= start < session.close_utc:
            return index
    return None


def _gap_series(priced: Sequence[Bar], sessions: Sequence[SessionDay]) -> tuple[RuleValue, ...]:
    ordered = sorted(sessions, key=lambda s: s.open_utc)
    if all(a.close_utc <= b.open_utc for a, b in itertools.pairwise(ordered)):
        return _disjoint_gap_series(priced, ordered)
    session_of = [_session_index(bar.bar_start_utc, ordered) for bar in priced]
    first_open: dict[int, Decimal] = {}
    last_close: dict[int, Decimal] = {}
    for bar, index in zip(priced, session_of, strict=True):
        if index is None or bar.open is None or bar.close is None:
            continue
        first_open.setdefault(index, bar.open)
        last_close[index] = bar.close
    # The gap is a per-session constant: computed once per session index.
    gap_of: dict[int, RuleValue] = {}
    for index, open_today in first_open.items():
        previous_close = last_close.get(index - 1) if index != 0 else None
        if previous_close is not None:
            gap_of[index] = abs(open_today - previous_close) / previous_close
    return tuple(None if index is None else gap_of.get(index) for index in session_of)


def _priced_value(bars: Sequence[Bar], *, last: bool) -> tuple[Decimal, Decimal] | None:
    """``(open, close)`` of the first (or last) bar carrying both, else ``None``."""
    for bar in reversed(bars) if last else bars:
        if bar.open is not None and bar.close is not None:
            return bar.open, bar.close
    return None


def _disjoint_gap_series(
    priced: Sequence[Bar], ordered: Sequence[SessionDay]
) -> tuple[RuleValue, ...]:
    """:func:`_gap_series` for disjoint sessions sorted by open (the calendar case).

    Bars are strictly increasing, so the bars of each session are the contiguous run
    ``open <= start < close`` (found by bisection): the same unique session as the
    per-bar scan, and the same first open / last close per session.
    """
    starts = list(map(_START, priced))
    out: list[RuleValue] = [None] * len(priced)
    runs = [(bisect_left(starts, s.open_utc), bisect_left(starts, s.close_utc)) for s in ordered]
    previous_close: Decimal | None = None
    for index, (first, last) in enumerate(runs):
        run = priced[first:last]
        opening = _priced_value(run, last=False)
        closing = _priced_value(run, last=True)
        if opening is not None and index != 0 and previous_close is not None:
            gap = abs(opening[0] - previous_close) / previous_close
            out[first:last] = [gap] * (last - first)
        previous_close = None if closing is None else closing[1]
    return tuple(out)


def _ratio_series(values: Sequence[RuleValue], divisors: Sequence[float]) -> tuple[RuleValue, ...]:
    """``value / divisor`` where ``value`` is a float, else ``None`` (aligned).

    Indicator series are a run of ``None`` followed by floats: that suffix is divided
    in one pass; any other shape takes the element-by-element definition.
    """
    if len(values) != len(divisors):
        raise ValueError("values and divisors must have the same length")
    lead = next((i for i, value in enumerate(values) if value is not None), len(values))
    tail = values[lead:]
    if all(map(isinstance, tail, itertools.repeat(float))):
        return (None,) * lead + tuple(map(operator.truediv, tail, divisors[lead:]))
    return tuple(
        value / divisor if isinstance(value, float) else None
        for value, divisor in zip(values, divisors, strict=True)
    )


def build_feature_frame(
    bars: Sequence[Bar],
    *,
    timeframe: Timeframe,
    periods: IndicatorParams,
    sessions: Sequence[SessionDay] = (),
) -> FeatureFrame:
    """Build the :class:`FeatureFrame` of closed ``bars`` (one symbol, one timeframe).

    Args:
        bars: Closed bars in chronological order; EMPTY bars are allowed and excluded.
        timeframe: Expected timeframe of every bar.
        periods: Indicator periods; a ``None`` period yields an all-``None`` series.
        sessions: Consecutive calendar sessions covering the bars (for ``gap_pct``).

    Raises:
        StrategyInputError: mixed symbols/timeframes or unordered/duplicate bars.
    """
    return _build_feature_frame(bars, timeframe, periods, sessions, previous=None)[0]


def _build_feature_frame(
    bars: Sequence[Bar],
    timeframe: Timeframe,
    periods: IndicatorParams,
    sessions: Sequence[SessionDay],
    *,
    previous: IndicatorInputs | None,
    valid_prefix: int = 0,
) -> tuple[FeatureFrame, IndicatorInputs | None]:
    """:func:`build_feature_frame`, reusing the indicator inputs of ``previous``.

    Indicators come from :class:`IndicatorInputs` (same cores as the batch functions);
    when the bars cannot be used as finite floats the batch functions are called on the
    raw values instead, so such input is reported exactly as they document.
    """
    _validate_bars(bars, timeframe, valid_prefix=valid_prefix)
    priced = (
        tuple(bar for bar in bars if bar.status is not BarStatus.EMPTY)
        if BarStatus.EMPTY in map(_STATUS, bars)
        else tuple(bars)
    )
    size = len(priced)
    inputs = IndicatorInputs.from_bars(priced, previous)
    opens: tuple[RuleValue, ...]
    close_floats: Sequence[float] | None
    atr_of: Callable[[int], IndicatorSeries]
    ema_of: Callable[[int], IndicatorSeries]
    rsi_of: Callable[[int], IndicatorSeries]
    sma_of: Callable[[int], IndicatorSeries]
    volume_avg_of: Callable[[int], IndicatorSeries]
    ibs_values: IndicatorSeries
    if inputs is None:
        closes = bar_closes(priced)
        highs = bar_highs(priced)
        lows = bar_lows(priced)
        volumes = bar_volumes(priced)
        opens = tuple(bar.open for bar in priced)
        close_floats = None
        atr_of = partial(atr, highs, lows, closes)
        ema_of, rsi_of = partial(ema, closes), partial(rsi, closes)
        sma_of = partial(sma, closes)
        volume_avg_of = partial(volume_average, volumes)
        ibs_values = ibs(highs, lows, closes)
    else:
        closes, highs, lows, volumes = inputs.closes, inputs.highs, inputs.lows, inputs.volumes
        opens = inputs.opens
        close_floats = inputs.close_floats
        atr_of, ema_of, rsi_of = inputs.atr, inputs.ema, inputs.rsi
        sma_of = inputs.sma
        volume_avg_of = inputs.volume_average
        ibs_values = inputs.ibs()

    def maybe(
        period: int | None, compute: Callable[[int], IndicatorSeries]
    ) -> tuple[RuleValue, ...]:
        return (None,) * size if period is None else compute(period)

    atr_values = maybe(periods.atr_period, atr_of)
    divisors = close_floats if close_floats is not None else [float(c) for c in closes]
    atr_pct = _ratio_series(atr_values, divisors)

    series: dict[str, tuple[RuleValue, ...]] = {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "ema_fast": maybe(periods.ema_fast, ema_of),
        "ema_slow": maybe(periods.ema_slow, ema_of),
        "rsi": maybe(periods.rsi_period, rsi_of),
        "atr": atr_values,
        "volume_avg": maybe(periods.volume_avg_period, volume_avg_of),
        "atr_pct": atr_pct,
        "gap_pct": _gap_series(priced, sessions) if sessions else (None,) * size,
        "ibs": ibs_values,
        "sma_short": maybe(periods.sma_short_period, sma_of),
        "sma_long": maybe(periods.sma_long_period, sma_of),
    }
    if set(series) != set(SERIES_BASE_NAMES):  # internal invariant, keeps the two in sync
        raise StrategyConfigError(f"feature series out of sync: {sorted(series)}")
    symbol = bars[0].symbol if bars else ""
    frame = FeatureFrame(
        symbol=symbol,
        timeframe=timeframe,
        bars=priced,
        series=MappingProxyType(series),
    )
    return frame, inputs


def _same_objects(left: Sequence[object], right: Sequence[object]) -> bool:
    return len(left) == len(right) and all(map(operator.is_, left, right))


@dataclass(frozen=True, slots=True)
class _MemoEntry:
    bars: tuple[Bar, ...]
    sessions: tuple[SessionDay, ...]
    periods: IndicatorParams
    frame: FeatureFrame
    inputs: IndicatorInputs | None


class FeatureFrameMemo:
    """Incremental :func:`build_feature_frame` for a sliding window of one symbol.

    :func:`build_feature_frame` is a pure function of its immutable arguments, so:

    * a call that receives the same bar and session *objects* (identity, in the same
      order) and the same ``periods`` object as the previous call for that
      ``(symbol, timeframe)`` returns the previous frame (the confirmation window only
      changes when a confirmation bar closes, but is evaluated on every primary bar);
    * otherwise the frame is rebuilt, carrying over the per-bar indicator inputs of the
      bars shared with the previous call (:meth:`IndicatorInputs.from_bars`).

    Either way the frame equals ``build_feature_frame`` of the same arguments. One entry
    is kept per key.
    """

    def __init__(self) -> None:
        self._last: dict[tuple[str, Timeframe], _MemoEntry] = {}

    def build(
        self,
        bars: Sequence[Bar],
        *,
        timeframe: Timeframe,
        periods: IndicatorParams,
        sessions: Sequence[SessionDay] = (),
    ) -> FeatureFrame:
        """:func:`build_feature_frame` of the arguments (see the class)."""
        bars_key = tuple(bars)
        sessions_key = tuple(sessions)
        key = (bars_key[0].symbol if bars_key else "", timeframe)
        last = self._last.get(key)
        if (
            last is not None
            and last.periods is periods
            and _same_objects(last.bars, bars_key)
            and _same_objects(last.sessions, sessions_key)
        ):
            return last.frame
        shared = 0 if last is None else shared_run(last.bars, bars_key)[0]
        frame, inputs = _build_feature_frame(
            bars_key,
            timeframe,
            periods,
            sessions_key,
            previous=None if last is None else last.inputs,
            valid_prefix=shared,
        )
        self._last[key] = _MemoEntry(bars_key, sessions_key, periods, frame, inputs)
        return frame


@dataclass(frozen=True, slots=True)
class IndicatorContext:
    """Everything a strategy decision is evaluated on.

    Attributes:
        signal_bar: The last closed primary bar as received (it may be EMPTY or
            INCOMPLETE); the decision refers to this bar.
        primary: Primary-timeframe feature frame.
        confirmation: Confirmation-timeframe frame, or ``None`` if unused.
    """

    signal_bar: Bar | None
    primary: FeatureFrame
    confirmation: FeatureFrame | None

    def series_value(self, name: str, offset: int, rule_timeframe: RuleTimeframe) -> RuleValue:
        """Resolve ``name`` (see :func:`resolve_series_name`) and return its value.

        Raises:
            StrategyConfigError: a confirmation series is requested but the context has
                no confirmation frame.
        """
        timeframe, base = resolve_series_name(name, rule_timeframe)
        if timeframe is RuleTimeframe.PRIMARY:
            return self.primary.value(base, offset)
        if self.confirmation is None:
            raise StrategyConfigError(
                f"series {name!r} needs the confirmation timeframe, which is not configured"
            )
        return self.confirmation.value(base, offset)

    def latest_atr(self) -> Decimal | None:
        """ATR of the last priced primary bar as Decimal (input of the ATR stop, 16.1)."""
        value = self.primary.value("atr", -1)
        return Decimal(str(value)) if isinstance(value, float) else None


def build_indicator_context(
    primary_bars: Sequence[Bar],
    confirmation_bars: Sequence[Bar] | None,
    *,
    primary_timeframe: Timeframe,
    confirmation_timeframe: Timeframe | None,
    periods: IndicatorParams,
    sessions: Sequence[SessionDay] = (),
    memo: FeatureFrameMemo | None = None,
) -> IndicatorContext:
    """Build the :class:`IndicatorContext` for one symbol.

    Confirmation bars whose ``bar_end_utc`` is after the signal bar's ``bar_end_utc``
    are dropped (no lookahead across timeframes). With a ``memo`` (one per stream of
    windows) both frames are built incrementally with the same result, see
    :class:`FeatureFrameMemo`.

    Raises:
        StrategyInputError: invalid bars, symbols that differ between timeframes, or
            confirmation bars passed without a confirmation timeframe.
    """
    build = build_feature_frame if memo is None else memo.build
    primary = build(primary_bars, timeframe=primary_timeframe, periods=periods, sessions=sessions)
    signal_bar = primary_bars[-1] if primary_bars else None
    confirmation: FeatureFrame | None = None
    if confirmation_timeframe is not None:
        visible = tuple(confirmation_bars or ())
        if signal_bar is not None and visible and max(map(_END, visible)) > signal_bar.bar_end_utc:
            visible = tuple(b for b in visible if b.bar_end_utc <= signal_bar.bar_end_utc)
        confirmation = build(
            visible, timeframe=confirmation_timeframe, periods=periods, sessions=sessions
        )
        if signal_bar is not None and visible and visible[0].symbol != signal_bar.symbol:
            raise StrategyInputError(
                f"confirmation symbol {visible[0].symbol} != primary {signal_bar.symbol}"
            )
    elif confirmation_bars:
        raise StrategyInputError("confirmation bars given but no confirmation timeframe")
    return IndicatorContext(signal_bar=signal_bar, primary=primary, confirmation=confirmation)
