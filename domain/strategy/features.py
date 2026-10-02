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
           sessions are passed. Meaningful for intraday timeframes only.
========== ============================================================================

Indicators come exclusively from :mod:`domain.market.indicators` (sec. 14.5, never
reimplemented). They are computed in float; prices stay Decimal.

EMPTY bars (sec. 10.3.4) carry no prices and are **excluded** before any series is
computed: offset ``-1`` is the last *priced* bar. INCOMPLETE bars carry prices and are
included. An indicator whose period is ``None`` yields an all-``None`` series; the
strategy refuses to start when a rule needs such a series (OWNER_DECISION pending).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType

from pydantic import Field

from domain.market.indicators import (
    IndicatorSeries,
    atr,
    atr_min_bars,
    bar_closes,
    bar_highs,
    bar_lows,
    bar_volumes,
    ema,
    ema_min_bars,
    rsi,
    rsi_min_bars,
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
    }
)
"""Base series -> ``strategy.indicators`` parameter it depends on."""


class IndicatorParams(DomainModel):
    """``strategy.indicators`` of config.yaml. ``None`` = OWNER_DECISION pending."""

    ema_fast: int | None = Field(ge=1)
    ema_slow: int | None = Field(ge=1)
    rsi_period: int | None = Field(ge=1)
    atr_period: int | None = Field(ge=1)
    volume_avg_period: int | None = Field(ge=1)

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


def _validate_bars(bars: Sequence[Bar], timeframe: Timeframe) -> None:
    symbols = {bar.symbol for bar in bars}
    if len(symbols) > 1:
        raise StrategyInputError(f"bars of several symbols in one frame: {sorted(symbols)}")
    for index, bar in enumerate(bars):
        if bar.timeframe is not timeframe:
            raise StrategyInputError(
                f"bars[{index}] has timeframe {bar.timeframe}, expected {timeframe}"
            )
        if index and bar.bar_start_utc <= bars[index - 1].bar_start_utc:
            raise StrategyInputError(
                f"bars must be strictly increasing by bar_start_utc (index {index})"
            )


def _session_index(start: datetime, sessions: Sequence[SessionDay]) -> int | None:
    for index, session in enumerate(sessions):
        if session.open_utc <= start < session.close_utc:
            return index
    return None


def _gap_series(priced: Sequence[Bar], sessions: Sequence[SessionDay]) -> tuple[RuleValue, ...]:
    ordered = sorted(sessions, key=lambda s: s.open_utc)
    session_of = [_session_index(bar.bar_start_utc, ordered) for bar in priced]
    first_open: dict[int, Decimal] = {}
    last_close: dict[int, Decimal] = {}
    for bar, index in zip(priced, session_of, strict=True):
        if index is None or bar.open is None or bar.close is None:
            continue
        first_open.setdefault(index, bar.open)
        last_close[index] = bar.close
    out: list[RuleValue] = []
    for index in session_of:
        if index is None or index == 0 or index not in first_open:
            out.append(None)
            continue
        previous_close = last_close.get(index - 1)
        if previous_close is None:
            out.append(None)
            continue
        out.append(abs(first_open[index] - previous_close) / previous_close)
    return tuple(out)


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
    _validate_bars(bars, timeframe)
    priced = tuple(bar for bar in bars if bar.status is not BarStatus.EMPTY)
    size = len(priced)
    closes = bar_closes(priced)
    highs = bar_highs(priced)
    lows = bar_lows(priced)
    volumes = bar_volumes(priced)

    def maybe(
        period: int | None, compute: Callable[[int], IndicatorSeries]
    ) -> tuple[RuleValue, ...]:
        return (None,) * size if period is None else compute(period)

    atr_values = maybe(periods.atr_period, lambda p: atr(highs, lows, closes, p))
    atr_pct: list[RuleValue] = []
    for value, close in zip(atr_values, closes, strict=True):
        atr_pct.append(value / float(close) if isinstance(value, float) else None)

    series: dict[str, tuple[RuleValue, ...]] = {
        "open": tuple(bar.open for bar in priced),
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "ema_fast": maybe(periods.ema_fast, lambda p: ema(closes, p)),
        "ema_slow": maybe(periods.ema_slow, lambda p: ema(closes, p)),
        "rsi": maybe(periods.rsi_period, lambda p: rsi(closes, p)),
        "atr": atr_values,
        "volume_avg": maybe(periods.volume_avg_period, lambda p: volume_average(volumes, p)),
        "atr_pct": tuple(atr_pct),
        "gap_pct": _gap_series(priced, sessions) if sessions else (None,) * size,
    }
    if set(series) != set(SERIES_BASE_NAMES):  # internal invariant, keeps the two in sync
        raise StrategyConfigError(f"feature series out of sync: {sorted(series)}")
    symbol = bars[0].symbol if bars else ""
    return FeatureFrame(
        symbol=symbol,
        timeframe=timeframe,
        bars=priced,
        series=MappingProxyType(series),
    )


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
) -> IndicatorContext:
    """Build the :class:`IndicatorContext` for one symbol.

    Confirmation bars whose ``bar_end_utc`` is after the signal bar's ``bar_end_utc``
    are dropped (no lookahead across timeframes).

    Raises:
        StrategyInputError: invalid bars, symbols that differ between timeframes, or
            confirmation bars passed without a confirmation timeframe.
    """
    primary = build_feature_frame(
        primary_bars, timeframe=primary_timeframe, periods=periods, sessions=sessions
    )
    signal_bar = primary_bars[-1] if primary_bars else None
    confirmation: FeatureFrame | None = None
    if confirmation_timeframe is not None:
        visible = tuple(confirmation_bars or ())
        if signal_bar is not None:
            visible = tuple(b for b in visible if b.bar_end_utc <= signal_bar.bar_end_utc)
        confirmation = build_feature_frame(
            visible, timeframe=confirmation_timeframe, periods=periods, sessions=sessions
        )
        if signal_bar is not None and visible and visible[0].symbol != signal_bar.symbol:
            raise StrategyInputError(
                f"confirmation symbol {visible[0].symbol} != primary {signal_bar.symbol}"
            )
    elif confirmation_bars:
        raise StrategyInputError("confirmation bars given but no confirmation timeframe")
    return IndicatorContext(signal_bar=signal_bar, primary=primary, confirmation=confirmation)
