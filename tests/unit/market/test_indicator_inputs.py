"""Exactness of the fast indicator paths (sec. 14.5: one implementation, same floats).

* ``sma`` on ints uses a running exact integer sum: bit-identical to the ``fsum`` path.
* ``IndicatorInputs`` (inputs carried over between sliding windows) returns exactly the
  batch functions' output on the same bars, whatever the reuse pattern.

Floats are compared with ``==`` (bit-exact), never with a tolerance.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from domain.market.indicators import (
    IndicatorInputError,
    IndicatorInputs,
    atr,
    bar_closes,
    bar_highs,
    bar_lows,
    bar_volumes,
    ema,
    rsi,
    sma,
    true_range,
    volume_average,
)
from domain.models import Bar, BarStatus, DataFeed, Timeframe

START = datetime(2026, 6, 15, 13, 30, tzinfo=UTC)
PERIODS = (1, 2, 3, 9, 14, 21)


def _fsum_sma(values: Sequence[float], period: int) -> tuple[float | None, ...]:
    """The documented definition: ``fsum`` of each window divided by ``period``."""
    return tuple(
        math.fsum(values[i - period + 1 : i + 1]) / period if i >= period - 1 else None
        for i in range(len(values))
    )


def _random_bars(rng: random.Random, count: int) -> list[Bar]:
    bars: list[Bar] = []
    price = Decimal("100")
    for index in range(count):
        step = Decimal(rng.randint(-300, 300)) / 100
        open_ = max(price, Decimal("1.00"))
        close = max(open_ + step, Decimal("1.00"))
        spread = Decimal(rng.randint(0, 80)) / 100
        bars.append(
            Bar(
                symbol="SPY",
                timeframe=Timeframe.MIN_5,
                bar_start_utc=START + timedelta(minutes=5 * index),
                bar_end_utc=START + timedelta(minutes=5 * index + 5),
                open=open_,
                high=max(open_, close) + spread,
                low=max(min(open_, close) - spread, Decimal("0.01")),
                close=close,
                volume=rng.randint(0, 10**7),
                feed=DataFeed.IEX,
                status=BarStatus.COMPLETE,
            )
        )
        price = close
    return bars


def _assert_inputs_match_batch(inputs: IndicatorInputs, bars: Sequence[Bar]) -> None:
    closes, highs, lows = bar_closes(bars), bar_highs(bars), bar_lows(bars)
    volumes = bar_volumes(bars)
    assert inputs.bars == tuple(bars)
    assert inputs.closes == closes
    assert inputs.highs == highs
    assert inputs.lows == lows
    assert inputs.opens == tuple(b.open for b in bars)
    assert inputs.volumes == volumes
    assert inputs.true_range() == true_range(highs, lows, closes)
    for period in PERIODS:
        assert inputs.ema(period) == ema(closes, period)
        assert inputs.rsi(period) == rsi(closes, period)
        assert inputs.atr(period) == atr(highs, lows, closes, period)
        assert inputs.volume_average(period) == volume_average(volumes, period)


# --------------------------------------------------------------------------- SMA int path


@pytest.mark.parametrize("seed", range(5))
def test_integer_sma_is_bit_identical_to_the_fsum_definition(seed: int) -> None:
    rng = random.Random(seed)
    # Values near 2**53 make window sums exceed the exact float range: the rounding of
    # the running integer sum must still match fsum's (both correctly rounded).
    big = [rng.randint(2**52, 2**53) for _ in range(200)]
    small = [rng.randint(0, 10**7) for _ in range(200)]
    signed = [rng.randint(-(2**53), 2**53) for _ in range(200)]
    for values in (big, small, signed):
        for period in (1, 2, 3, 7, 20, 50):
            expected = _fsum_sma([float(v) for v in values], period)
            assert sma(values, period) == expected
            assert volume_average(values, period) == expected


def test_non_integer_or_huge_inputs_keep_the_fsum_path() -> None:
    mixed: list[Decimal | int] = [Decimal("1.5"), 2, 3, Decimal("4.25")]
    assert sma(mixed, 2) == _fsum_sma([1.5, 2.0, 3.0, 4.25], 2)
    huge = [2**60 + 1, 3, 2**60 + 7]
    assert sma(huge, 2) == _fsum_sma([float(v) for v in huge], 2)


def test_invalid_inputs_still_raise_like_before() -> None:
    with pytest.raises(IndicatorInputError, match=r"values\[1\] must be a number"):
        sma([1, None, 3], 2)  # type: ignore[list-item]
    with pytest.raises(IndicatorInputError, match=r"values\[0\] must be a number"):
        ema([True, 2, 3], 2)
    with pytest.raises(IndicatorInputError, match=r"closes\[2\] must be finite"):
        rsi([1.0, 2.0, math.inf, 3.0], 2)
    with pytest.raises(IndicatorInputError, match=r"values\[1\] must be finite"):
        ema([1.0, math.nan], 1)


# --------------------------------------------------------------------------- IndicatorInputs


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("maxlen", [1, 5, 30, 120])
def test_sliding_window_inputs_equal_the_batch_functions(seed: int, maxlen: int) -> None:
    rng = random.Random(seed)
    window: deque[Bar] = deque(maxlen=maxlen)
    previous: IndicatorInputs | None = None
    for bar in _random_bars(rng, 250):
        window.append(bar)
        current = tuple(window)
        inputs = IndicatorInputs.from_bars(current, previous)
        assert inputs is not None
        _assert_inputs_match_batch(inputs, current)
        fresh = IndicatorInputs.from_bars(current)
        assert fresh == inputs
        previous = inputs


def test_reuse_patterns_shrink_skip_and_unrelated_windows() -> None:
    bars = _random_bars(random.Random(7), 80)
    previous = IndicatorInputs.from_bars(bars[10:60])
    for window in (
        bars[10:60],  # identical
        bars[15:60],  # dropped several old bars, nothing new
        bars[12:70],  # dropped and appended
        bars[30:40],  # strictly inside
        bars[0:20],  # starts before the previous window: nothing reused
        bars[60:80],  # disjoint
        bars[20:21],  # single bar
        [],
    ):
        inputs = IndicatorInputs.from_bars(window, previous)
        assert inputs is not None
        _assert_inputs_match_batch(inputs, window)


def test_same_values_but_other_objects_are_not_reused() -> None:
    bars = _random_bars(random.Random(3), 30)
    previous = IndicatorInputs.from_bars(bars)
    clones = [bar.model_copy() for bar in bars]
    inputs = IndicatorInputs.from_bars(clones, previous)
    assert inputs is not None
    assert all(a is b for a, b in zip(inputs.bars, clones, strict=True))
    _assert_inputs_match_batch(inputs, clones)


def test_unusable_bars_return_none() -> None:
    bars = _random_bars(random.Random(1), 3)
    empty = Bar(
        symbol="SPY",
        timeframe=Timeframe.MIN_5,
        bar_start_utc=START,
        bar_end_utc=START + timedelta(minutes=5),
        open=None,
        high=None,
        low=None,
        close=None,
        volume=0,
        feed=DataFeed.IEX,
        status=BarStatus.EMPTY,
    )
    assert IndicatorInputs.from_bars([empty]) is None
    huge_price = bars[0].model_copy(
        update={"open": Decimal("1e400"), "high": Decimal("1e400"), "close": Decimal("1e400")}
    )
    assert IndicatorInputs.from_bars([huge_price]) is None
    huge_volume = bars[1].model_copy(update={"volume": 2**60})
    assert IndicatorInputs.from_bars([bars[0], huge_volume]) is None
