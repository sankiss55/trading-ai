"""Feature frames built from closed bars (sec. 10.3.4, 13.2, 14)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from domain.market.indicators import atr, bar_closes, ema, rsi, volume_average
from domain.models import BarStatus, Timeframe
from domain.strategy.features import (
    IndicatorParams,
    build_feature_frame,
    build_indicator_context,
)
from domain.strategy.rules import (
    SERIES_BASE_NAMES,
    RuleTimeframe,
    StrategyConfigError,
    StrategyInputError,
)
from tests.unit.strategy.factories import (
    DAY_1,
    DAY_2,
    UPTREND_CLOSES,
    bar,
    empty_bar,
    series_bars,
    uptrend_bars,
)

# TEST FIXTURE periods (not owner values).
PERIODS = IndicatorParams(ema_fast=3, ema_slow=5, rsi_period=3, atr_period=3, volume_avg_period=3)


def test_series_use_the_shared_indicators() -> None:
    bars = uptrend_bars()
    frame = build_feature_frame(bars, timeframe=Timeframe.MIN_5, periods=PERIODS)
    closes = bar_closes(bars)
    highs = tuple(b.high for b in bars if b.high is not None)
    lows = tuple(b.low for b in bars if b.low is not None)
    volumes = tuple(b.volume for b in bars)
    assert set(frame.series) == set(SERIES_BASE_NAMES)
    assert frame.series["ema_fast"] == ema(closes, 3)
    assert frame.series["ema_slow"] == ema(closes, 5)
    assert frame.series["rsi"] == rsi(closes, 3)
    assert frame.series["atr"] == atr(highs, lows, closes, 3)
    assert frame.series["volume_avg"] == volume_average(volumes, 3)
    assert frame.series["close"] == closes
    assert frame.value("close", -1) == Decimal(UPTREND_CLOSES[-1])
    assert frame.value("volume", -1) == 3000


def test_atr_pct_is_atr_over_close() -> None:
    frame = build_feature_frame(uptrend_bars(), timeframe=Timeframe.MIN_5, periods=PERIODS)
    atr_last = frame.value("atr", -1)
    close_last = frame.value("close", -1)
    assert isinstance(atr_last, float)
    assert isinstance(close_last, Decimal)
    assert frame.value("atr_pct", -1) == pytest.approx(atr_last / float(close_last))
    assert frame.value("atr_pct", -12) is None  # ATR not available yet


def test_empty_bars_are_excluded() -> None:
    bars = series_bars(["10", "11", "12"])
    with_empty = [
        *bars[:2],
        empty_bar(bars[1].bar_end_utc),
        bars[2].model_copy(
            update={
                "bar_start_utc": bars[2].bar_start_utc + timedelta(minutes=5),
                "bar_end_utc": bars[2].bar_end_utc + timedelta(minutes=5),
            }
        ),
    ]
    frame = build_feature_frame(with_empty, timeframe=Timeframe.MIN_5, periods=PERIODS)
    assert len(frame) == 3
    assert all(b.status is not BarStatus.EMPTY for b in frame.bars)
    assert frame.series["close"] == (Decimal("10"), Decimal("11"), Decimal("12"))


def test_incomplete_bars_are_included() -> None:
    bars = series_bars(["10", "11"])
    incomplete = bar(bars[1].bar_end_utc, "12", status=BarStatus.INCOMPLETE, minutes_present=2)
    frame = build_feature_frame([*bars, incomplete], timeframe=Timeframe.MIN_5, periods=PERIODS)
    assert frame.value("close", -1) == Decimal("12")


def test_unset_period_gives_none_series() -> None:
    periods = PERIODS.model_copy(update={"rsi_period": None})
    frame = build_feature_frame(uptrend_bars(), timeframe=Timeframe.MIN_5, periods=periods)
    assert all(value is None for value in frame.series["rsi"])
    assert periods.min_bars("rsi_period") == 0
    assert PERIODS.min_bars("rsi_period") == 4
    assert PERIODS.min_bars("ema_slow") == 5


def test_value_offsets_and_bounds() -> None:
    frame = build_feature_frame(series_bars(["1", "2"]), timeframe=Timeframe.MIN_5, periods=PERIODS)
    assert frame.value("close", -2) == Decimal("1")
    assert frame.value("close", -3) is None
    with pytest.raises(StrategyInputError):
        frame.value("close", 0)
    with pytest.raises(StrategyInputError):
        frame.value("macd", -1)


def test_empty_input_gives_empty_frame() -> None:
    frame = build_feature_frame([], timeframe=Timeframe.MIN_5, periods=PERIODS)
    assert len(frame) == 0
    assert frame.value("close", -1) is None


def test_rejects_mixed_symbols_timeframes_and_order() -> None:
    bars = series_bars(["1", "2"])
    other = bar(bars[1].bar_end_utc, "3", symbol="QQQ")
    with pytest.raises(StrategyInputError):
        build_feature_frame([*bars, other], timeframe=Timeframe.MIN_5, periods=PERIODS)
    with pytest.raises(StrategyInputError):
        build_feature_frame(bars, timeframe=Timeframe.MIN_15, periods=PERIODS)
    with pytest.raises(StrategyInputError):
        build_feature_frame([bars[1], bars[0]], timeframe=Timeframe.MIN_5, periods=PERIODS)
    with pytest.raises(StrategyInputError):
        build_feature_frame([bars[0], bars[0]], timeframe=Timeframe.MIN_5, periods=PERIODS)


def test_gap_pct_across_sessions() -> None:
    day1 = series_bars(["100", "101", "100"], start=DAY_1.open_utc)
    day2_first = bar(DAY_2.open_utc, "104", open_="103")
    day2_second = bar(DAY_2.open_utc + timedelta(minutes=5), "105", open_="104")
    bars = [*day1, day2_first, day2_second]
    frame = build_feature_frame(
        bars, timeframe=Timeframe.MIN_5, periods=PERIODS, sessions=[DAY_2, DAY_1]
    )
    expected = abs(Decimal("103") - Decimal("100")) / Decimal("100")
    assert frame.value("gap_pct", -1) == expected
    assert frame.value("gap_pct", -2) == expected
    assert frame.value("gap_pct", -3) is None  # first session in the window: no yesterday


def test_gap_pct_none_without_sessions_or_previous_session() -> None:
    bars = series_bars(["100", "101"], start=DAY_2.open_utc)
    no_sessions = build_feature_frame(bars, timeframe=Timeframe.MIN_5, periods=PERIODS)
    no_previous = build_feature_frame(
        bars, timeframe=Timeframe.MIN_5, periods=PERIODS, sessions=[DAY_1, DAY_2]
    )
    assert no_sessions.value("gap_pct", -1) is None
    assert no_previous.value("gap_pct", -1) is None


# --------------------------------------------------------------------------- context


def test_context_resolves_confirmation_and_drops_future_bars() -> None:
    primary = uptrend_bars()
    confirmation = series_bars(
        ["100", "102", "104", "106"], timeframe=Timeframe.MIN_15, start=DAY_1.open_utc
    )
    context = build_indicator_context(
        primary,
        confirmation,
        primary_timeframe=Timeframe.MIN_5,
        confirmation_timeframe=Timeframe.MIN_15,
        periods=PERIODS,
    )
    assert context.signal_bar == primary[-1]
    assert context.confirmation is not None
    # primary ends 14:30; the 15m bar [14:15, 14:30) is visible, nothing later.
    assert all(b.bar_end_utc <= primary[-1].bar_end_utc for b in context.confirmation.bars)
    assert len(context.confirmation) == 4
    assert context.series_value("close_confirm", -1, RuleTimeframe.PRIMARY) == Decimal("106")
    assert context.series_value("close", -1, RuleTimeframe.CONFIRMATION) == Decimal("106")
    assert context.series_value("close", -1, RuleTimeframe.PRIMARY) == Decimal("106")

    shorter = build_indicator_context(
        primary[:6],
        confirmation,
        primary_timeframe=Timeframe.MIN_5,
        confirmation_timeframe=Timeframe.MIN_15,
        periods=PERIODS,
    )
    assert shorter.confirmation is not None
    assert len(shorter.confirmation) == 2  # 13:30-13:45 and 13:45-14:00 only


def test_context_without_confirmation() -> None:
    context = build_indicator_context(
        uptrend_bars(),
        None,
        primary_timeframe=Timeframe.MIN_5,
        confirmation_timeframe=None,
        periods=PERIODS,
    )
    with pytest.raises(StrategyConfigError):
        context.series_value("ema_fast_confirm", -1, RuleTimeframe.PRIMARY)
    with pytest.raises(StrategyInputError):
        build_indicator_context(
            uptrend_bars(),
            series_bars(["1"], timeframe=Timeframe.MIN_15),
            primary_timeframe=Timeframe.MIN_5,
            confirmation_timeframe=None,
            periods=PERIODS,
        )


def test_context_rejects_confirmation_symbol_mismatch() -> None:
    confirmation = [bar(DAY_1.open_utc, "1", timeframe=Timeframe.MIN_15, symbol="QQQ")]
    with pytest.raises(StrategyInputError):
        build_indicator_context(
            uptrend_bars(),
            confirmation,
            primary_timeframe=Timeframe.MIN_5,
            confirmation_timeframe=Timeframe.MIN_15,
            periods=PERIODS,
        )


def test_latest_atr_is_decimal_of_float_repr() -> None:
    context = build_indicator_context(
        uptrend_bars(),
        None,
        primary_timeframe=Timeframe.MIN_5,
        confirmation_timeframe=None,
        periods=PERIODS,
    )
    raw = context.primary.value("atr", -1)
    assert context.latest_atr() == Decimal(str(raw))
    short = build_indicator_context(
        series_bars(["1"]),
        None,
        primary_timeframe=Timeframe.MIN_5,
        confirmation_timeframe=None,
        periods=PERIODS,
    )
    assert short.latest_atr() is None


def test_signal_bar_is_last_input_bar_even_if_empty() -> None:
    bars = series_bars(["1", "2"])
    trailing_empty = empty_bar(bars[-1].bar_end_utc)
    context = build_indicator_context(
        [*bars, trailing_empty],
        None,
        primary_timeframe=Timeframe.MIN_5,
        confirmation_timeframe=None,
        periods=PERIODS,
    )
    assert context.signal_bar == trailing_empty
    assert context.primary.value("close", -1) == Decimal("2")
