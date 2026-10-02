"""Aggregator tests (sec. 10.3, 10.4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from domain.market.aggregator import (
    AggregatorConfigError,
    BarAggregator,
    IngestStatus,
    RejectReason,
)
from domain.models import Bar, BarStatus, DataFeed, SessionDay, Timeframe
from tests.unit.market.factories import (
    EARLY_CLOSE_DAY,
    SUMMER_DAY,
    SUMMER_NEXT_DAY,
    WINTER_DAY,
    minute_bar,
)

OPEN = SUMMER_DAY.open_utc  # 13:30 UTC = 09:30 ET
GRACE = 10.0


def _agg(
    timeframe: Timeframe = Timeframe.MIN_5,
    *,
    symbols: tuple[str, ...] = ("SPY",),
    sessions: tuple[SessionDay, ...] = (SUMMER_DAY,),
) -> BarAggregator:
    agg = BarAggregator(
        timeframe=timeframe, symbols=symbols, feed=DataFeed.IEX, bar_close_grace_seconds=GRACE
    )
    for session in sessions:
        agg.add_session(session)
    return agg


def _at(minutes: int, base: datetime = OPEN) -> datetime:
    return base + timedelta(minutes=minutes)


# --------------------------------------------------------------------------- closing on last minute


def test_complete_bucket_closes_on_its_last_minute_with_proper_ohlcv() -> None:
    agg = _agg()
    prices = [  # (open, high, low, close, volume)
        ("100", "101", "99.5", "100.5", 10),
        ("100.5", "102", "100", "101.5", 20),
        ("101.5", "101.75", "98", "99", 30),
        ("99", "100", "98.5", "99.5", 40),
        ("99.5", "100.25", "99", "100.1", 50),
    ]
    results = []
    for i, (o, h, lo, c, v) in enumerate(prices):
        bar = minute_bar(_at(i), open_=o, high=h, low=lo, close=c, volume=v)
        results.append(agg.ingest(bar))
    assert all(r.status is IngestStatus.ACCEPTED for r in results)
    assert all(r.closed_bars == () for r in results[:-1])
    (closed,) = results[-1].closed_bars
    assert closed.timeframe is Timeframe.MIN_5
    assert closed.bar_start_utc == OPEN
    assert closed.bar_end_utc == _at(5)
    assert closed.status is BarStatus.COMPLETE
    assert closed.minutes_present == 5
    assert closed.open == Decimal("100")
    assert closed.high == Decimal("102")
    assert closed.low == Decimal("98")
    assert closed.close == Decimal("100.1")
    assert closed.volume == 150
    assert closed.feed is DataFeed.IEX


@pytest.mark.parametrize(
    ("session", "timeframe", "minute_offset", "expected_bucket_offset"),
    [
        (SUMMER_DAY, Timeframe.MIN_5, 7, 5),  # 09:37 ET -> 09:35 bucket
        (SUMMER_DAY, Timeframe.MIN_15, 22, 15),  # 09:52 ET -> 09:45 bucket
        (WINTER_DAY, Timeframe.MIN_5, 4, 0),  # 09:34 EST -> 09:30 bucket (14:30 UTC)
        (SUMMER_DAY, Timeframe.HOUR_1, 75, 60),  # 10:45 ET -> 10:30 bucket
    ],
)
def test_buckets_align_to_session_open(
    session: SessionDay, timeframe: Timeframe, minute_offset: int, expected_bucket_offset: int
) -> None:
    agg = _agg(timeframe, sessions=(session,))
    closed = agg.ingest(minute_bar(_at(minute_offset, session.open_utc))).closed_bars
    closed += agg.on_time(_at(24 * 60, session.open_utc))
    priced = [bar for bar in closed if bar.status is not BarStatus.EMPTY]
    assert [bar.bar_start_utc for bar in priced] == [_at(expected_bucket_offset, session.open_utc)]


def test_out_of_order_minute_is_inserted_in_place_while_bucket_open() -> None:
    agg = _agg()
    agg.ingest(minute_bar(_at(1), open_="101", close="101"))
    result = agg.ingest(minute_bar(_at(0), open_="100", close="100"))
    assert result.status is IngestStatus.ACCEPTED
    for i in (2, 3):
        agg.ingest(minute_bar(_at(i), open_="100.25", close="100.25"))
    (closed,) = agg.ingest(minute_bar(_at(4), open_="100.75", close="100.5")).closed_bars
    assert closed.open == Decimal("100")  # from 09:30 even though it arrived second
    assert closed.close == Decimal("100.5")
    assert closed.status is BarStatus.COMPLETE


def test_missing_middle_minute_gives_incomplete_bar() -> None:
    agg = _agg()
    for i in (0, 1, 3):
        agg.ingest(minute_bar(_at(i)))
    (closed,) = agg.ingest(minute_bar(_at(4))).closed_bars
    assert closed.status is BarStatus.INCOMPLETE
    assert closed.minutes_present == 4
    assert closed.volume == 4000


# --------------------------------------------------------------------------- closing by time


def test_bucket_without_last_minute_closes_only_after_grace() -> None:
    agg = _agg()
    for i in range(4):
        agg.ingest(minute_bar(_at(i)))
    deadline = _at(5) + timedelta(seconds=GRACE)
    assert agg.on_time(_at(5)) == ()
    assert agg.on_time(deadline) == ()  # must strictly exceed end + grace
    (closed,) = agg.on_time(deadline + timedelta(microseconds=1))
    assert closed.status is BarStatus.INCOMPLETE
    assert closed.minutes_present == 4
    assert agg.on_time(deadline + timedelta(seconds=1)) == ()  # never emitted twice


def test_bucket_without_minutes_is_empty() -> None:
    agg = _agg(symbols=("QQQ", "SPY"))
    agg.ingest(minute_bar(_at(0), symbol="SPY"))
    closed = agg.on_time(_at(5) + timedelta(seconds=GRACE + 1))
    assert [(b.symbol, b.status) for b in closed] == [
        ("QQQ", BarStatus.EMPTY),
        ("SPY", BarStatus.INCOMPLETE),
    ]
    empty = closed[0]
    assert empty.open is None
    assert empty.close is None
    assert empty.volume == 0
    assert empty.minutes_present == 0
    assert empty.feed is DataFeed.IEX


def test_on_time_emits_in_time_then_symbol_order() -> None:
    agg = _agg(symbols=("SPY", "AAPL"))
    closed = agg.on_time(_at(10) + timedelta(seconds=GRACE + 1))
    assert [(b.bar_start_utc, b.symbol) for b in closed] == [
        (_at(0), "AAPL"),
        (_at(0), "SPY"),
        (_at(5), "AAPL"),
        (_at(5), "SPY"),
    ]


def test_closing_a_bucket_first_closes_earlier_open_buckets() -> None:
    agg = _agg()
    agg.ingest(minute_bar(_at(0)))
    result = agg.ingest(minute_bar(_at(9)))  # last minute of the 09:35 bucket
    assert [(b.bar_start_utc, b.minutes_present) for b in result.closed_bars] == [
        (_at(0), 1),
        (_at(5), 1),
    ]


# --------------------------------------------------------------------------- duplicates / late


def test_duplicate_in_open_bucket_is_discarded_first_wins() -> None:
    agg = _agg()
    agg.ingest(minute_bar(_at(0), high="101", volume=10))
    dup = agg.ingest(minute_bar(_at(0), high="200", volume=999))
    assert dup.status is IngestStatus.DUPLICATE
    for i in range(1, 4):
        agg.ingest(minute_bar(_at(i), volume=10))
    (closed,) = agg.ingest(minute_bar(_at(4), volume=10)).closed_bars
    assert closed.high == Decimal("101")
    assert closed.volume == 50


def test_minute_for_closed_bucket_is_late_or_duplicate_and_never_reopens() -> None:
    agg = _agg()
    for i in (0, 1, 2, 4):
        agg.ingest(minute_bar(_at(i)))
    late = agg.ingest(minute_bar(_at(3)))
    assert late.status is IngestStatus.LATE_BAR
    assert late.closed_bars == ()
    assert agg.ingest(minute_bar(_at(1))).status is IngestStatus.DUPLICATE
    # The bucket is not re-emitted by time either.
    assert all(b.bar_start_utc != OPEN for b in agg.on_time(_at(60)))


def test_last_minute_after_time_close_is_late() -> None:
    agg = _agg()
    agg.ingest(minute_bar(_at(0)))
    (closed,) = agg.on_time(_at(5) + timedelta(seconds=GRACE + 1))
    assert closed.minutes_present == 1
    assert agg.ingest(minute_bar(_at(4))).status is IngestStatus.LATE_BAR


def test_previous_session_minute_resent_is_duplicate() -> None:
    agg = _agg(sessions=(SUMMER_DAY, SUMMER_NEXT_DAY))
    agg.ingest(minute_bar(_at(0)))
    agg.on_time(SUMMER_NEXT_DAY.open_utc + timedelta(minutes=30))
    assert agg.ingest(minute_bar(_at(0))).status is IngestStatus.DUPLICATE
    assert agg.ingest(minute_bar(_at(1))).status is IngestStatus.LATE_BAR


# --------------------------------------------------------------------------- session boundaries


@pytest.mark.parametrize(
    "start",
    [
        OPEN - timedelta(minutes=1),  # premarket
        SUMMER_DAY.close_utc,  # first after-hours minute
        datetime(2026, 6, 14, 15, 0, tzinfo=UTC),  # no registered session
    ],
)
def test_minutes_outside_registered_sessions(start: datetime) -> None:
    assert _agg().ingest(minute_bar(start)).status is IngestStatus.OUT_OF_SESSION


def test_no_session_registered_means_out_of_session() -> None:
    agg = _agg(sessions=())
    assert agg.ingest(minute_bar(OPEN)).status is IngestStatus.OUT_OF_SESSION
    assert agg.on_time(_at(600)) == ()
    assert agg.next_bucket_start("SPY") is None


def test_last_hour_bucket_is_truncated_at_close() -> None:
    agg = _agg(Timeframe.HOUR_1)
    last_bucket_start = SUMMER_DAY.close_utc - timedelta(minutes=30)  # 15:30 ET
    for i in range(30):
        result = agg.ingest(minute_bar(last_bucket_start + timedelta(minutes=i)))
    closed = result.closed_bars[-1]
    assert closed.bar_start_utc == last_bucket_start
    assert closed.bar_end_utc == SUMMER_DAY.close_utc
    assert closed.status is BarStatus.COMPLETE
    assert closed.minutes_present == 30
    # 6 earlier hourly buckets (09:30..14:30) were closed first, all EMPTY.
    assert [b.status for b in result.closed_bars[:-1]] == [BarStatus.EMPTY] * 6


def test_early_close_session() -> None:
    agg = _agg(sessions=(EARLY_CLOSE_DAY,))
    close = EARLY_CLOSE_DAY.close_utc
    after = agg.ingest(minute_bar(close))
    assert after.status is IngestStatus.OUT_OF_SESSION
    closed = agg.on_time(close + timedelta(hours=1))
    assert len(closed) == 42  # 3.5h / 5min
    assert closed[-1].bar_end_utc == close


def test_full_session_and_rollover_to_next_session() -> None:
    agg = _agg(sessions=(SUMMER_DAY,))
    closed = agg.on_time(SUMMER_DAY.close_utc + timedelta(hours=1))
    assert len(closed) == 78  # 6.5h / 5min
    assert agg.next_bucket_start("SPY") is None
    agg.add_session(SUMMER_NEXT_DAY)
    assert agg.next_bucket_start("SPY") == SUMMER_NEXT_DAY.open_utc
    (first,) = agg.ingest(minute_bar(SUMMER_NEXT_DAY.open_utc + timedelta(minutes=4))).closed_bars
    assert first.bar_start_utc == SUMMER_NEXT_DAY.open_utc
    assert first.minutes_present == 1


def test_one_minute_timeframe_passes_bars_through() -> None:
    agg = _agg(Timeframe.MIN_1)
    (closed,) = agg.ingest(minute_bar(OPEN, volume=7)).closed_bars
    assert closed.status is BarStatus.COMPLETE
    assert closed.minutes_present == 1
    assert closed.volume == 7


# --------------------------------------------------------------------------- rejections / config


def test_unknown_symbol() -> None:
    result = _agg().ingest(minute_bar(OPEN, symbol="TSLA"))
    assert result.status is IngestStatus.UNKNOWN_SYMBOL


def _reject_cases() -> list[tuple[Bar, RejectReason]]:
    five = minute_bar(OPEN).model_copy(update={"timeframe": Timeframe.MIN_5, "bar_end_utc": _at(5)})
    sip = minute_bar(OPEN, feed=DataFeed.SIP)
    shifted = minute_bar(OPEN + timedelta(seconds=30))
    long = minute_bar(OPEN).model_copy(update={"bar_end_utc": _at(2)})
    empty = Bar(
        symbol="SPY",
        timeframe=Timeframe.MIN_1,
        bar_start_utc=OPEN,
        bar_end_utc=_at(1),
        open=None,
        high=None,
        low=None,
        close=None,
        volume=0,
        feed=DataFeed.IEX,
        status=BarStatus.EMPTY,
    )
    return [
        (five, RejectReason.NOT_MINUTE_BAR),
        (sip, RejectReason.FEED_MISMATCH),
        (shifted, RejectReason.MISALIGNED),
        (long, RejectReason.MISALIGNED),
        (empty, RejectReason.EMPTY_MINUTE_BAR),
    ]


@pytest.mark.parametrize(("bar", "reason"), _reject_cases())
def test_invalid_minute_bars_are_rejected(bar: Bar, reason: RejectReason) -> None:
    result = _agg().ingest(bar)
    assert result.status is IngestStatus.REJECTED
    assert result.reason is reason


def test_sessions_must_be_added_in_order() -> None:
    agg = _agg(sessions=(SUMMER_NEXT_DAY,))
    with pytest.raises(AggregatorConfigError):
        agg.add_session(SUMMER_DAY)
    with pytest.raises(AggregatorConfigError):
        agg.add_session(SUMMER_NEXT_DAY)


def test_invalid_configuration() -> None:
    with pytest.raises(AggregatorConfigError):
        BarAggregator(
            timeframe=Timeframe.DAY_1, symbols=["SPY"], feed=DataFeed.IEX, bar_close_grace_seconds=1
        )
    with pytest.raises(AggregatorConfigError):
        BarAggregator(
            timeframe=Timeframe.MIN_5,
            symbols=["SPY"],
            feed=DataFeed.IEX,
            bar_close_grace_seconds=-1,
        )


def test_aggregation_is_deterministic() -> None:
    minutes = [minute_bar(_at(i), volume=i + 1) for i in (3, 0, 1, 7, 9, 12, 14)]

    def run() -> list[Bar]:
        agg = _agg(symbols=("QQQ", "SPY"))
        out: list[Bar] = []
        for bar in minutes:
            out.extend(agg.ingest(bar).closed_bars)
        out.extend(agg.on_time(_at(30)))
        return out

    assert run() == run()


def test_exposed_properties() -> None:
    agg = _agg(Timeframe.MIN_15, symbols=("SPY", "QQQ"))
    assert agg.timeframe is Timeframe.MIN_15
    assert agg.symbols == frozenset({"SPY", "QQQ"})
    assert agg.next_bucket_start("SPY") == OPEN
    assert agg.next_bucket_start("TSLA") is None


# --------------------------------------------------------------------------- on_time shortcut


def test_extra_clock_steps_never_change_what_closes_or_when() -> None:
    # on_time skips its scan while no bucket can be due: polling the clock every few
    # seconds must emit exactly what sparse polling emits, at the first step past due.
    dense = _agg(symbols=("SPY", "QQQ"), sessions=(SUMMER_DAY, SUMMER_NEXT_DAY))
    sparse = _agg(symbols=("SPY", "QQQ"), sessions=(SUMMER_DAY, SUMMER_NEXT_DAY))
    minutes = [m for m in range(0, 200) if m % 7 not in (3, 4)]  # gaps: some close by time
    dense_out: list[Bar] = []
    sparse_out: list[Bar] = []
    for minute in minutes:
        bar = minute_bar(_at(minute), symbol="SPY" if minute % 2 else "QQQ")
        dense_out.extend(dense.ingest(bar).closed_bars)
        sparse_out.extend(sparse.ingest(bar).closed_bars)
        for second in range(0, 60, 5):
            dense_out.extend(dense.on_time(_at(minute) + timedelta(seconds=second)))
        sparse_out.extend(sparse.on_time(_at(minute + 1)))
    end = SUMMER_NEXT_DAY.close_utc + timedelta(minutes=1)
    dense_out.extend(dense.on_time(end))
    sparse_out.extend(sparse.on_time(end))
    assert sorted(dense_out, key=lambda b: (b.bar_start_utc, b.symbol)) == sorted(
        sparse_out, key=lambda b: (b.bar_start_utc, b.symbol)
    )
    assert len(dense_out) == 2 * 2 * 78


def test_on_time_resumes_after_a_session_is_added() -> None:
    agg = _agg(sessions=(SUMMER_DAY,))
    assert len(agg.on_time(SUMMER_DAY.close_utc + timedelta(hours=1))) == 78
    assert agg.on_time(SUMMER_DAY.close_utc + timedelta(hours=2)) == ()
    agg.add_session(SUMMER_NEXT_DAY)
    closed = agg.on_time(SUMMER_NEXT_DAY.open_utc + timedelta(minutes=5, seconds=11))
    assert [b.bar_start_utc for b in closed] == [SUMMER_NEXT_DAY.open_utc]
    assert agg.on_time(SUMMER_NEXT_DAY.open_utc + timedelta(minutes=10, seconds=10)) == ()
