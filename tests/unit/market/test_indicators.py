"""Indicator tests (sec. 14): hand-computed reference values, None handling, no lookahead.

Every reference value below is derived by hand in the comment next to it; floats are
compared with ``pytest.approx``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from domain.market.indicators import (
    IndicatorInputError,
    IndicatorSeries,
    Numeric,
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
    sma,
    sma_min_bars,
    true_range,
    true_range_min_bars,
    volume_average,
    volume_average_min_bars,
)
from domain.models import Bar, BarStatus, DataFeed, Timeframe
from tests.unit.market.factories import minute_bar

N = None


def _assert_series(actual: IndicatorSeries, expected: Sequence[float | None]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------- SMA


@pytest.mark.parametrize(
    ("values", "period", "expected"),
    [
        # (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        ([1, 2, 3, 4, 5], 3, [N, N, 2.0, 3.0, 4.0]),
        # (10+11+13)/3=34/3, (11+13+12)/3=12, (13+12+14)/3=13, (12+14+15)/3=41/3
        ([10, 11, 13, 12, 14, 15], 3, [N, N, 34 / 3, 12.0, 13.0, 41 / 3]),
        # Decimal inputs, period 2: (100.5+101)/2=100.75, (101+99.5)/2=100.25,
        # (99.5+102)/2=100.75
        (
            [Decimal("100.5"), Decimal("101.0"), Decimal("99.5"), Decimal("102.0")],
            2,
            [N, 100.75, 100.25, 100.75],
        ),
        # period 1 is the identity
        ([7, 8], 1, [7.0, 8.0]),
    ],
)
def test_sma_reference_values(values: list[Numeric], period: int, expected: list[float]) -> None:
    _assert_series(sma(values, period), expected)


# --------------------------------------------------------------------------- EMA


@pytest.mark.parametrize(
    ("values", "period", "expected"),
    [
        # period 3 -> alpha = 2/4 = 0.5. Seed at i=2: SMA(1,2,3)=2.
        # i=3: 0.5*4 + 0.5*2 = 3; i=4: 0.5*5 + 0.5*3 = 4.
        ([1, 2, 3, 4, 5], 3, [N, N, 2.0, 3.0, 4.0]),
        # alpha = 0.5. Seed: (10+11+13)/3 = 34/3.
        # i=3: (12 + 34/3)/2 = 35/3; i=4: (14 + 35/3)/2 = 77/6; i=5: (15 + 77/6)/2 = 167/12.
        ([10, 11, 13, 12, 14, 15], 3, [N, N, 34 / 3, 35 / 3, 77 / 6, 167 / 12]),
        # period 4 -> alpha = 2/5 = 0.4. Seed at i=3: (2+4+6+8)/4 = 5.
        # i=4: 0.4*10 + 0.6*5 = 7; i=5: 0.4*4 + 0.6*7 = 5.8.
        ([2, 4, 6, 8, 10, 4], 4, [N, N, N, 5.0, 7.0, 5.8]),
    ],
)
def test_ema_reference_values(values: list[Numeric], period: int, expected: list[float]) -> None:
    _assert_series(ema(values, period), expected)


def test_ema_is_seeded_with_sma() -> None:
    values = [3, 9, 6, 12]
    assert ema(values, 3)[2] == sma(values, 3)[2]


# --------------------------------------------------------------------------- RSI


@pytest.mark.parametrize(
    ("closes", "period", "expected"),
    [
        # Only gains: avg_loss = 0, avg_gain > 0 -> 100.
        ([1, 2, 3, 4, 5], 3, [N, N, N, 100.0, 100.0]),
        # changes: +1, -1, +2, -1, +2 (period 3)
        # i=3 seed: avg_gain=(1+0+2)/3=1, avg_loss=(0+1+0)/3=1/3, RS=3 -> 100-100/4 = 75
        # i=4 (-1): avg_gain=(1*2+0)/3=2/3, avg_loss=(1/3*2+1)/3=5/9, RS=6/5 -> 100-100/2.2
        # i=5 (+2): avg_gain=(2/3*2+2)/3=10/9, avg_loss=(5/9*2+0)/3=10/27, RS=3 -> 75
        ([10, 11, 10, 12, 11, 13], 3, [N, N, N, 75.0, 100 - 100 / 2.2, 75.0]),
        # changes: +0.5, -1, +1, +0.5 (period 2)
        # i=2 seed: gain=(0.5+0)/2=0.25, loss=(0+1)/2=0.5, RS=0.5 -> 100-100/1.5
        # i=3 (+1): gain=(0.25+1)/2=0.625, loss=(0.5+0)/2=0.25, RS=2.5 -> 100-100/3.5
        # i=4 (+0.5): gain=(0.625+0.5)/2=0.5625, loss=0.125, RS=4.5 -> 100-100/5.5
        (
            [44, 44.5, 43.5, 44.5, 45],
            2,
            [N, N, 100 - 100 / 1.5, 100 - 100 / 3.5, 100 - 100 / 5.5],
        ),
        # Only losses: avg_gain = 0 -> RS = 0 -> 0.
        ([5, 4, 3, 2], 3, [N, N, N, 0.0]),
        # Flat prices: avg_gain = avg_loss = 0 -> neutral 50 (documented convention).
        ([5, 5, 5, 5], 3, [N, N, N, 50.0]),
    ],
)
def test_rsi_wilder_reference_values(
    closes: list[Numeric], period: int, expected: list[float]
) -> None:
    _assert_series(rsi(closes, period), expected)


# --------------------------------------------------------------------------- True range / ATR


def test_true_range_reference_values() -> None:
    # tr0 = 10-9 = 1; tr1 = max(2, |12-9.5|, |10-9.5|) = 2.5; tr2 = max(2, 0, |9-11|) = 2;
    # tr3 = max(3, |15-10|, |12-10|) = 5; tr4 = max(3, 0, |11-14|) = 3.
    highs = [10, 12, 11, 15, 14]
    lows = [9, 10, 9, 12, 11]
    closes = [9.5, 11, 10, 14, 12]
    assert true_range(highs, lows, closes) == pytest.approx((1.0, 2.5, 2.0, 5.0, 3.0))


@pytest.mark.parametrize(
    ("highs", "lows", "closes", "period", "expected"),
    [
        # Constant range 2 and no gaps: every TR = 2 -> ATR = 2.
        ([10, 11, 12, 13], [8, 9, 10, 11], [9, 10, 11, 12], 3, [N, N, 2.0, 2.0]),
        # TR = 1, 2.5, 2, 5, 3 (see test_true_range_reference_values).
        # i=2 seed: (1+2.5+2)/3 = 11/6; i=3: (11/6*2+5)/3 = 26/9; i=4: (26/9*2+3)/3 = 79/27.
        (
            [10, 12, 11, 15, 14],
            [9, 10, 9, 12, 11],
            [9.5, 11, 10, 14, 12],
            3,
            [N, N, 11 / 6, 26 / 9, 79 / 27],
        ),
        # Decimal inputs, period 2. tr0 = 1; tr1 = max(1, |21-19.5|, |20-19.5|) = 1.5;
        # tr2 = max(2, |19-20.5|, |17-20.5|) = 3.5. i=1 seed: 1.25; i=2: (1.25+3.5)/2 = 2.375.
        (
            [Decimal("20"), Decimal("21"), Decimal("19")],
            [Decimal("19"), Decimal("20"), Decimal("17")],
            [Decimal("19.5"), Decimal("20.5"), Decimal("18")],
            2,
            [N, 1.25, 2.375],
        ),
    ],
)
def test_atr_wilder_reference_values(
    highs: list[Numeric],
    lows: list[Numeric],
    closes: list[Numeric],
    period: int,
    expected: list[float],
) -> None:
    _assert_series(atr(highs, lows, closes, period), expected)


def test_true_range_rejects_low_above_high() -> None:
    with pytest.raises(IndicatorInputError):
        true_range([10], [11], [10])


def test_atr_rejects_mismatched_lengths() -> None:
    with pytest.raises(IndicatorInputError):
        atr([10, 11], [9], [9.5, 10], 1)


# --------------------------------------------------------------------------- volume average


@pytest.mark.parametrize(
    ("volumes", "period", "expected"),
    [
        # (100+200)/2=150, (200+300)/2=250, (300+400)/2=350
        ([100, 200, 300, 400], 2, [N, 150.0, 250.0, 350.0]),
        # (1000+0+500)/3 = 500
        ([1000, 0, 500], 3, [N, N, 500.0]),
        # (10+20+30+40+50)/5 = 30
        ([10, 20, 30, 40, 50], 5, [N, N, N, N, 30.0]),
    ],
)
def test_volume_average_reference_values(
    volumes: list[int], period: int, expected: list[float]
) -> None:
    _assert_series(volume_average(volumes, period), expected)


# --------------------------------------------------------------------------- min bars / None


def test_min_bars_declared() -> None:
    assert sma_min_bars(5) == 5
    assert ema_min_bars(5) == 5
    assert rsi_min_bars(14) == 15
    assert atr_min_bars(14) == 14
    assert volume_average_min_bars(20) == 20
    assert true_range_min_bars() == 1


_SingleSeries = Callable[[Sequence[Numeric], int], IndicatorSeries]


@pytest.mark.parametrize(
    ("func", "min_bars"),
    [(sma, sma_min_bars), (ema, ema_min_bars), (rsi, rsi_min_bars)],
)
@pytest.mark.parametrize("period", [1, 3, 7])
def test_first_value_appears_exactly_at_min_bars(
    func: _SingleSeries, min_bars: Callable[[int], int], period: int
) -> None:
    values = [float(v % 5 + v) for v in range(20)]
    out = func(values, period)
    first = next(i for i, v in enumerate(out) if v is not None)
    assert first == min_bars(period) - 1
    assert all(v is None for v in out[:first])
    assert all(v is not None for v in out[first:])


def test_atr_first_value_appears_at_min_bars() -> None:
    highs = [10 + i for i in range(10)]
    lows = [8 + i for i in range(10)]
    closes = [9 + i for i in range(10)]
    out = atr(highs, lows, closes, 4)
    assert out[: atr_min_bars(4) - 1] == (None, None, None)
    assert all(v is not None for v in out[atr_min_bars(4) - 1 :])


@pytest.mark.parametrize("func", [sma, ema, rsi, volume_average])
def test_insufficient_data_is_all_none(func: _SingleSeries) -> None:
    assert func([1, 2], 3) == (None, None)
    assert func([], 3) == ()


def test_atr_insufficient_data_is_all_none() -> None:
    assert atr([10, 11], [9, 10], [9.5, 10.5], 3) == (None, None)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True])
def test_invalid_input_values_raise(bad: object) -> None:
    with pytest.raises(IndicatorInputError):
        sma([1, bad, 3], 2)  # type: ignore[list-item]


@pytest.mark.parametrize("period", [0, -1, True, 2.0])
def test_invalid_period_raises(period: object) -> None:
    with pytest.raises(IndicatorInputError):
        ema([1, 2, 3], period)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- no lookahead


_SERIES = [
    100.0, 101.5, 100.25, 102.0, 103.75, 103.0, 101.0, 104.5, 105.0, 104.25,
    106.0, 105.5, 107.25, 106.75, 108.0, 107.0, 109.5, 108.25, 110.0, 109.0,
]  # fmt: skip


@pytest.mark.parametrize(
    "compute",
    [
        lambda xs: sma(xs, 5),
        lambda xs: ema(xs, 5),
        lambda xs: rsi(xs, 5),
        lambda xs: volume_average(xs, 5),
        lambda xs: atr([x + 1 for x in xs], [x - 1 for x in xs], xs, 5),
    ],
)
def test_appending_future_values_never_changes_past_outputs(
    compute: Callable[[list[float]], IndicatorSeries],
) -> None:
    full = compute(_SERIES)
    for cut in range(1, len(_SERIES)):
        prefix = compute(_SERIES[:cut])
        assert prefix == full[:cut]


def test_changing_a_future_value_never_changes_past_outputs() -> None:
    altered = [*_SERIES[:-1], 1_000_000.0]
    assert ema(altered, 5)[:-1] == ema(_SERIES, 5)[:-1]
    assert rsi(altered, 5)[:-1] == rsi(_SERIES, 5)[:-1]


# --------------------------------------------------------------------------- bar helpers


def test_bar_helpers_extract_values() -> None:
    start = datetime(2026, 6, 15, 13, 30, tzinfo=UTC)
    bars = [
        minute_bar(start, high="101", low="99", close="100.5", volume=10),
        minute_bar(start + timedelta(minutes=1), high="102", low="100", close="101", volume=20),
    ]
    assert bar_closes(bars) == (Decimal("100.5"), Decimal("101"))
    assert bar_highs(bars) == (Decimal("101"), Decimal("102"))
    assert bar_lows(bars) == (Decimal("99"), Decimal("100"))
    assert bar_volumes(bars) == (10, 20)
    # Decimal series feed indicators directly.
    averaged = sma(bar_closes(bars), 2)
    assert averaged[0] is None
    assert averaged[1] == pytest.approx(100.75)


def test_bar_helpers_refuse_empty_bars() -> None:
    start = datetime(2026, 6, 15, 13, 30, tzinfo=UTC)
    empty = Bar(
        symbol="SPY",
        timeframe=Timeframe.MIN_5,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(minutes=5),
        open=None,
        high=None,
        low=None,
        close=None,
        volume=0,
        feed=DataFeed.IEX,
        status=BarStatus.EMPTY,
        minutes_present=0,
    )
    for helper in (bar_closes, bar_highs, bar_lows, bar_volumes):
        with pytest.raises(IndicatorInputError):
            helper([empty])
