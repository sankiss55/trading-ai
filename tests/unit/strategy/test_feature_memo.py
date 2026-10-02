"""``FeatureFrameMemo`` and the fast ``gap_pct`` path give exactly the batch frames."""

from __future__ import annotations

import random
from collections import deque
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from domain.models import Bar, BarStatus, SessionDay, Timeframe
from domain.strategy.features import (
    FeatureFrame,
    FeatureFrameMemo,
    IndicatorParams,
    _gap_series,
    _session_index,
    build_feature_frame,
    build_indicator_context,
)
from domain.strategy.rules import StrategyInputError
from tests.unit.strategy.factories import bar, empty_bar

# TEST FIXTURE periods (not owner values).
PERIODS = IndicatorParams(ema_fast=3, ema_slow=8, rsi_period=5, atr_period=4, volume_avg_period=6)


def _sessions(count: int) -> list[SessionDay]:
    first = date(2026, 6, 15)
    out: list[SessionDay] = []
    for index in range(count):
        day = first + timedelta(days=index)
        out.append(
            SessionDay(
                session_date=day,
                open_utc=datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC),
                close_utc=datetime(day.year, day.month, day.day, 20, 0, tzinfo=UTC),
                is_early_close=False,
            )
        )
    return out


def _session_bars(rng: random.Random, sessions: list[SessionDay]) -> list[Bar]:
    """5Min bars of every session with random closes, EMPTY and INCOMPLETE bars."""
    bars: list[Bar] = []
    price = Decimal("50")
    for session in sessions:
        start = session.open_utc
        while start < session.close_utc:
            roll = rng.random()
            if roll < 0.08:
                bars.append(empty_bar(start))
            else:
                price = max(price + Decimal(rng.randint(-40, 40)) / 100, Decimal("1"))
                incomplete = roll < 0.15
                bars.append(
                    bar(
                        start,
                        str(price),
                        open_=str(max(price + Decimal(rng.randint(-20, 20)) / 100, Decimal(1))),
                        volume=rng.randint(0, 50000),
                        status=BarStatus.INCOMPLETE if incomplete else BarStatus.COMPLETE,
                        minutes_present=rng.randint(1, 4) if incomplete else None,
                    )
                )
            start += timedelta(minutes=5)
    return bars


def _assert_same_frame(actual: FeatureFrame, expected: FeatureFrame) -> None:
    assert actual.symbol == expected.symbol
    assert actual.timeframe is expected.timeframe
    assert actual.bars == expected.bars
    assert dict(actual.series) == dict(expected.series)  # floats compared bit-exactly


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("maxlen", [1, 20, 100])
def test_memo_frames_equal_batch_frames_over_a_sliding_window(seed: int, maxlen: int) -> None:
    rng = random.Random(seed)
    sessions = _sessions(4)
    memo = FeatureFrameMemo()
    window: deque[Bar] = deque(maxlen=maxlen)
    for item in _session_bars(rng, sessions):
        window.append(item)
        current = tuple(window)
        covering = tuple(s for s in sessions if s.close_utc > current[0].bar_start_utc)
        frame = memo.build(current, timeframe=Timeframe.MIN_5, periods=PERIODS, sessions=covering)
        expected = build_feature_frame(
            current, timeframe=Timeframe.MIN_5, periods=PERIODS, sessions=covering
        )
        _assert_same_frame(frame, expected)


def test_memo_returns_the_previous_frame_for_identical_inputs_only() -> None:
    bars = _session_bars(random.Random(5), _sessions(1))[:30]
    sessions = tuple(_sessions(1))
    memo = FeatureFrameMemo()
    first = memo.build(bars, timeframe=Timeframe.MIN_5, periods=PERIODS, sessions=sessions)
    again = memo.build(list(bars), timeframe=Timeframe.MIN_5, periods=PERIODS, sessions=sessions)
    assert again is first
    other_periods = PERIODS.model_copy(update={"ema_fast": 4})
    changed = memo.build(bars, timeframe=Timeframe.MIN_5, periods=other_periods, sessions=sessions)
    assert changed is not first
    _assert_same_frame(
        changed,
        build_feature_frame(
            bars, timeframe=Timeframe.MIN_5, periods=other_periods, sessions=sessions
        ),
    )
    no_sessions = memo.build(bars, timeframe=Timeframe.MIN_5, periods=other_periods)
    assert no_sessions.series["gap_pct"] == (None,) * len(no_sessions)


def test_context_with_memo_equals_context_without() -> None:
    rng = random.Random(11)
    sessions = _sessions(3)
    primary = _session_bars(rng, sessions)
    confirmation = [
        bar(b.bar_start_utc, str(b.close), timeframe=Timeframe.MIN_15)
        for b in primary[::3]
        if b.close is not None
    ]
    memo = FeatureFrameMemo()
    for end in range(1, len(primary), 7):
        window = primary[max(0, end - 60) : end]
        visible = [c for c in confirmation if c.bar_start_utc <= window[-1].bar_start_utc]
        with_memo, plain = (
            build_indicator_context(
                window,
                visible,
                primary_timeframe=Timeframe.MIN_5,
                confirmation_timeframe=Timeframe.MIN_15,
                periods=PERIODS,
                sessions=sessions,
                memo=used,
            )
            for used in (memo, None)
        )
        _assert_same_frame(with_memo.primary, plain.primary)
        assert with_memo.confirmation is not None
        assert plain.confirmation is not None
        _assert_same_frame(with_memo.confirmation, plain.confirmation)


def test_gap_series_sweep_equals_the_per_bar_scan() -> None:
    sessions = _sessions(3)
    bars = [b for b in _session_bars(random.Random(2), sessions) if b.status is not BarStatus.EMPTY]
    # Disjoint calendar sessions (fast path) and overlapping ones (scan kept).
    overlapping = [
        *sessions,
        SessionDay(
            session_date=sessions[1].session_date,
            open_utc=sessions[1].open_utc + timedelta(hours=1),
            close_utc=sessions[2].open_utc + timedelta(hours=1),
            is_early_close=False,
        ),
    ]
    for given in (sessions, overlapping, sessions[1:], sessions[:1]):
        ordered = sorted(given, key=lambda s: s.open_utc)
        session_of = [_session_index(b.bar_start_utc, ordered) for b in bars]
        first_open: dict[int, Decimal] = {}
        last_close: dict[int, Decimal] = {}
        for item, index in zip(bars, session_of, strict=True):
            if index is not None and item.open is not None and item.close is not None:
                first_open.setdefault(index, item.open)
                last_close[index] = item.close
        expected = tuple(
            None
            if index is None or index == 0 or index not in first_open
            else (
                None
                if last_close.get(index - 1) is None
                else abs(first_open[index] - last_close[index - 1]) / last_close[index - 1]
            )
            for index in session_of
        )
        assert _gap_series(bars, given) == expected


def test_memo_reports_an_invalid_new_bar_like_the_batch_builder() -> None:
    bars = _session_bars(random.Random(9), _sessions(1))[:20]
    memo = FeatureFrameMemo()
    memo.build(bars, timeframe=Timeframe.MIN_5, periods=PERIODS)
    late = bar(bars[-1].bar_start_utc, "50")  # not after the last bar
    other_tf = bar(bars[-1].bar_start_utc + timedelta(minutes=5), "50", timeframe=Timeframe.MIN_15)
    other_symbol = bar(bars[-1].bar_start_utc + timedelta(minutes=5), "50", symbol="QQQ")
    for invalid in (late, other_tf, other_symbol):
        window = [*bars[1:], invalid]
        with pytest.raises(StrategyInputError) as batch:
            build_feature_frame(window, timeframe=Timeframe.MIN_5, periods=PERIODS)
        with pytest.raises(StrategyInputError) as incremental:
            memo.build(window, timeframe=Timeframe.MIN_5, periods=PERIODS)
        assert str(incremental.value) == str(batch.value)
