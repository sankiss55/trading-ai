"""Data quality tests (sec. 10.4, 10.6)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from domain.market.quality import (
    QualityCode,
    TimestampIssue,
    assess_staleness,
    bar_key,
    check_duplicate,
    check_staleness,
    deduplicate_bars,
    detect_gaps,
    refill_window,
    validate_bar_timestamp,
)
from domain.models import Timeframe
from tests.unit.market.factories import SUMMER_DAY, daily_bar, minute_bar

OPEN = SUMMER_DAY.open_utc


def _at(minutes: float) -> datetime:
    return OPEN + timedelta(minutes=minutes)


# --------------------------------------------------------------------------- timestamps


def test_valid_finished_minute_bar() -> None:
    result = validate_bar_timestamp(minute_bar(_at(0)), now_utc=_at(1), max_future_seconds=2)
    assert result.passed
    assert result.code == QualityCode.TIMESTAMP_VALID


@pytest.mark.parametrize(
    ("now_offset_seconds", "passed"),
    [(60, True), (58, True), (57.9, False), (0, False)],
)
def test_future_bar_beyond_tolerance_is_invalid(now_offset_seconds: float, passed: bool) -> None:
    # The bar ends at OPEN+60s; with 2s tolerance it is valid from now = OPEN+58s.
    now = OPEN + timedelta(seconds=now_offset_seconds)
    result = validate_bar_timestamp(minute_bar(OPEN), now_utc=now, max_future_seconds=2)
    assert result.passed is passed
    if not passed:
        assert result.code == QualityCode.INVALID_TIMESTAMP
        assert result.detail["issue"] == TimestampIssue.FUTURE


def test_misaligned_bar_is_invalid() -> None:
    bar = minute_bar(OPEN + timedelta(seconds=15))
    result = validate_bar_timestamp(bar, now_utc=_at(10), max_future_seconds=0)
    assert result.code == QualityCode.INVALID_TIMESTAMP
    assert result.detail["issue"] == TimestampIssue.MISALIGNED


def test_wrong_duration_is_invalid() -> None:
    bar = minute_bar(OPEN).model_copy(update={"bar_end_utc": _at(2)})
    result = validate_bar_timestamp(bar, now_utc=_at(10), max_future_seconds=0)
    assert result.detail["issue"] == TimestampIssue.DURATION_MISMATCH


def test_daily_bar_only_checks_future_start() -> None:
    bar = daily_bar(SUMMER_DAY.session_date, 1000)
    assert validate_bar_timestamp(bar, now_utc=_at(5), max_future_seconds=0).passed
    early = bar.bar_start_utc - timedelta(seconds=1)
    result = validate_bar_timestamp(bar, now_utc=early, max_future_seconds=0)
    assert result.detail["issue"] == TimestampIssue.FUTURE


# --------------------------------------------------------------------------- duplicates


def test_bar_key_and_duplicate_check() -> None:
    bar = minute_bar(OPEN)
    assert bar_key(bar) == ("SPY", Timeframe.MIN_1, OPEN)
    assert check_duplicate(bar, set()).code == QualityCode.UNIQUE_BAR
    dup = check_duplicate(bar, {bar_key(bar)})
    assert not dup.passed
    assert dup.code == QualityCode.DUPLICATE_BAR


def test_deduplicate_keeps_first_occurrence_in_order() -> None:
    first = minute_bar(_at(1), volume=1)
    other = minute_bar(_at(0), volume=2)
    repeat = minute_bar(_at(1), volume=3)
    other_symbol = minute_bar(_at(1), symbol="QQQ", volume=4)
    result = deduplicate_bars([first, other, repeat, other_symbol])
    assert result.unique == (first, other, other_symbol)
    assert result.duplicates == (repeat,)


# --------------------------------------------------------------------------- staleness


def test_staleness_per_symbol() -> None:
    last_end = _at(10)
    fresh = check_staleness(
        "SPY", last_end, now_utc=last_end + timedelta(seconds=90), max_bar_age_seconds=90
    )
    assert fresh.passed
    assert fresh.code == QualityCode.DATA_FRESH
    stale = check_staleness(
        "SPY", last_end, now_utc=last_end + timedelta(seconds=90.5), max_bar_age_seconds=90
    )
    assert not stale.passed
    assert stale.code == QualityCode.STALE
    never = check_staleness("SPY", None, now_utc=last_end, max_bar_age_seconds=90)
    assert never.code == QualityCode.STALE
    assert never.detail["reason"] == "NO_DATA"


@pytest.mark.parametrize(
    ("stale_count", "whitelist_size", "warning"),
    [(2, 4, False), (3, 4, True), (1, 3, False), (2, 3, True), (0, 1, False), (1, 1, True)],
)
def test_global_stale_ratio_warning_more_than_half(
    stale_count: int, whitelist_size: int, warning: bool
) -> None:
    now = _at(30)
    symbols = [f"S{i}" for i in range(whitelist_size)]
    last = {
        symbol: (now - timedelta(minutes=10) if i < stale_count else now - timedelta(seconds=5))
        for i, symbol in enumerate(symbols)
    }
    report = assess_staleness(last, symbols, now_utc=now, max_bar_age_seconds=60)
    assert report.stale_count == stale_count
    assert report.whitelist_size == whitelist_size
    assert report.circuit_breaker_warning is warning


def test_missing_symbol_counts_as_stale_and_custom_ratio() -> None:
    now = _at(30)
    report = assess_staleness(
        {"SPY": now},
        ["SPY", "QQQ", "IWM"],
        now_utc=now,
        max_bar_age_seconds=60,
        warning_ratio=Decimal("0.75"),
    )
    assert report.stale_symbols == frozenset({"QQQ", "IWM"})
    assert report.circuit_breaker_warning is False  # 2 > 0.75 * 3 is False


def test_empty_whitelist_never_warns() -> None:
    report = assess_staleness({}, [], now_utc=_at(0), max_bar_age_seconds=60)
    assert report.circuit_breaker_warning is False


# --------------------------------------------------------------------------- gaps


def test_detect_gaps_between_consecutive_bars() -> None:
    bars = [minute_bar(_at(i)) for i in (5, 0, 1, 2, 9, 1)]  # unsorted + duplicate
    gaps = detect_gaps(bars)
    assert [(g.start_utc, g.end_utc, g.missing_minutes) for g in gaps] == [
        (_at(3), _at(5), 2),
        (_at(6), _at(9), 3),
    ]


def test_detect_gaps_with_session_reports_leading_gap_and_ignores_outside() -> None:
    bars = [minute_bar(OPEN - timedelta(minutes=5)), minute_bar(_at(2)), minute_bar(_at(3))]
    gaps = detect_gaps(bars, session=SUMMER_DAY)
    assert [(g.start_utc, g.missing_minutes) for g in gaps] == [(OPEN, 2)]


def test_detect_gaps_session_without_bars_is_one_gap() -> None:
    (gap,) = detect_gaps([minute_bar(OPEN - timedelta(minutes=5))], session=SUMMER_DAY)
    assert gap.missing_minutes == 390


def test_detect_gaps_none_and_errors() -> None:
    assert detect_gaps([]) == ()
    assert detect_gaps([minute_bar(_at(i)) for i in range(5)]) == ()
    with pytest.raises(ValueError, match="single symbol"):
        detect_gaps([minute_bar(OPEN), minute_bar(OPEN, symbol="QQQ")])
    with pytest.raises(ValueError, match="1Min"):
        detect_gaps([daily_bar(SUMMER_DAY.session_date, 1)])


def test_refill_window_after_reconnect() -> None:
    gap = refill_window("SPY", last_bar_end_utc=_at(10), now_utc=_at(15.5), session=SUMMER_DAY)
    assert gap is not None
    assert (gap.start_utc, gap.end_utc, gap.missing_minutes) == (_at(10), _at(15), 5)


def test_refill_window_edge_cases() -> None:
    # Up to date: nothing to refill.
    up_to_date = refill_window(
        "SPY", last_bar_end_utc=_at(15), now_utc=_at(15.9), session=SUMMER_DAY
    )
    assert up_to_date is None
    # No data yet: from the session open.
    gap = refill_window("SPY", last_bar_end_utc=None, now_utc=_at(3), session=SUMMER_DAY)
    assert gap is not None
    assert gap.start_utc == OPEN
    assert gap.missing_minutes == 3
    # Clipped to the session close.
    late = SUMMER_DAY.close_utc + timedelta(hours=2)
    gap = refill_window("SPY", last_bar_end_utc=_at(380), now_utc=late, session=SUMMER_DAY)
    assert gap is not None
    assert gap.end_utc == SUMMER_DAY.close_utc
    # Before the open: nothing.
    before = OPEN - timedelta(minutes=10)
    assert refill_window("SPY", last_bar_end_utc=None, now_utc=before, session=SUMMER_DAY) is None
