"""Session window tests (sec. 11). Parameters are explicit fixture values."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone

import pytest

from domain.market.session import (
    OWNER_DECISION_PENDING_CODE,
    SESSION_BAR_MISMATCH_CODE,
    OwnerDecisionPendingError,
    SessionBarMismatchError,
    SessionWindowParams,
    compute_session_windows,
    empty_session_bar,
    session_bar,
    session_day_from_market_times,
)
from domain.models import Bar, BarStatus, DataFeed, HoldingMode, SessionDay, Timeframe
from tests.unit.market.factories import (
    EARLY_CLOSE_DAY,
    SUMMER_DAY,
    SUMMER_NEXT_DAY,
    WINTER_DAY,
    daily_bar,
    minute_bar,
)

INTRADAY = SessionWindowParams(
    no_entry_first_minutes=15,
    no_entry_last_minutes=30,
    holding_mode=HoldingMode.INTRADAY,
    flatten_minutes_before_close=10,
)


def test_windows_for_regular_intraday_session() -> None:
    w = compute_session_windows(SUMMER_DAY, INTRADAY)
    assert w.session_open_utc == datetime(2026, 6, 15, 13, 30, tzinfo=UTC)
    assert w.session_close_utc == datetime(2026, 6, 15, 20, 0, tzinfo=UTC)
    assert w.entries_allowed_from_utc == datetime(2026, 6, 15, 13, 45, tzinfo=UTC)
    assert w.entries_allowed_until_utc == datetime(2026, 6, 15, 19, 30, tzinfo=UTC)
    assert w.flatten_at_utc == datetime(2026, 6, 15, 19, 50, tzinfo=UTC)
    assert w.session_date == SUMMER_DAY.session_date
    assert w.is_early_close is False


def test_early_close_uses_calendar_close() -> None:
    w = compute_session_windows(EARLY_CLOSE_DAY, INTRADAY)
    assert w.entries_allowed_until_utc == datetime(2026, 11, 27, 17, 30, tzinfo=UTC)
    assert w.flatten_at_utc == datetime(2026, 11, 27, 17, 50, tzinfo=UTC)
    assert w.is_early_close is True


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 6, 15, 13, 44, 59, tzinfo=UTC), False),
        (datetime(2026, 6, 15, 13, 45, tzinfo=UTC), True),  # inclusive start
        (datetime(2026, 6, 15, 19, 29, 59, tzinfo=UTC), True),
        (datetime(2026, 6, 15, 19, 30, tzinfo=UTC), False),  # exclusive end
    ],
)
def test_is_entry_window_half_open(now: datetime, expected: bool) -> None:
    assert compute_session_windows(SUMMER_DAY, INTRADAY).is_entry_window(now) is expected


def test_is_open_and_flatten_time() -> None:
    w = compute_session_windows(SUMMER_DAY, INTRADAY)
    assert not w.is_open(SUMMER_DAY.open_utc - timedelta(seconds=1))
    assert w.is_open(SUMMER_DAY.open_utc)
    assert not w.is_open(SUMMER_DAY.close_utc)
    assert not w.is_flatten_time(datetime(2026, 6, 15, 19, 49, tzinfo=UTC))
    assert w.is_flatten_time(datetime(2026, 6, 15, 19, 50, tzinfo=UTC))


def test_minutes_since_open_and_to_close() -> None:
    w = compute_session_windows(SUMMER_DAY, INTRADAY)
    now = datetime(2026, 6, 15, 14, 0, 30, tzinfo=UTC)
    assert w.minutes_since_open(now) == pytest.approx(30.5)
    assert w.minutes_to_close(now) == pytest.approx(359.5)
    assert w.minutes_since_open(SUMMER_DAY.open_utc - timedelta(minutes=5)) == pytest.approx(-5)


def test_swing_has_no_flatten_and_ignores_flatten_param() -> None:
    params = SessionWindowParams(
        no_entry_first_minutes=0,
        no_entry_last_minutes=0,
        holding_mode=HoldingMode.SWING,
        flatten_minutes_before_close=None,
    )
    w = compute_session_windows(SUMMER_DAY, params)
    assert w.flatten_at_utc is None
    assert not w.is_flatten_time(SUMMER_DAY.close_utc)
    assert w.is_entry_window(SUMMER_DAY.open_utc)


def test_window_can_be_empty_without_error() -> None:
    params = INTRADAY.model_copy(
        update={"no_entry_first_minutes": 120, "no_entry_last_minutes": 120}
    )
    w = compute_session_windows(EARLY_CLOSE_DAY, params)  # 3.5h session
    assert not any(
        w.is_entry_window(EARLY_CLOSE_DAY.open_utc + timedelta(minutes=m)) for m in range(211)
    )


@pytest.mark.parametrize(
    ("update", "parameter"),
    [
        ({"no_entry_first_minutes": None}, "session.no_entry_first_minutes"),
        ({"no_entry_last_minutes": None}, "session.no_entry_last_minutes"),
        ({"holding_mode": None}, "strategy.holding_mode"),
        ({"flatten_minutes_before_close": None}, "strategy.flatten_minutes_before_close"),
    ],
)
def test_pending_owner_decision_raises(update: dict[str, object], parameter: str) -> None:
    params = INTRADAY.model_copy(update=update)
    with pytest.raises(OwnerDecisionPendingError) as excinfo:
        compute_session_windows(SUMMER_DAY, params)
    assert excinfo.value.code == OWNER_DECISION_PENDING_CODE
    assert excinfo.value.parameter == parameter
    assert "OWNER_DECISION" in str(excinfo.value)


def test_negative_minutes_rejected() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        SessionWindowParams(
            no_entry_first_minutes=-1,
            no_entry_last_minutes=0,
            holding_mode=HoldingMode.SWING,
        )


def test_session_day_from_market_times_with_fixed_offset() -> None:
    edt = timezone(timedelta(hours=-4))
    session = session_day_from_market_times(
        date(2026, 6, 15),
        open_local=time(9, 30),
        close_local=time(16, 0),
        market_tz=edt,
        is_early_close=False,
    )
    assert session == SUMMER_DAY


def test_session_day_from_market_times_with_zoneinfo_handles_dst() -> None:
    zoneinfo = pytest.importorskip("zoneinfo")
    try:
        new_york = zoneinfo.ZoneInfo("America/New_York")
    except zoneinfo.ZoneInfoNotFoundError:
        pytest.skip("tz database not available (install tzdata on Windows)")
    summer, winter = (
        session_day_from_market_times(
            day,
            open_local=time(9, 30),
            close_local=time(16, 0),
            market_tz=new_york,
            is_early_close=False,
        )
        for day in (date(2026, 6, 15), date(2026, 1, 12))
    )
    assert summer.open_utc == datetime(2026, 6, 15, 13, 30, tzinfo=UTC)
    assert winter.open_utc == datetime(2026, 1, 12, 14, 30, tzinfo=UTC)


# --------------------------------------------------------------------------- daily bars


def _labelled(day: date, hour: int) -> Bar:
    bar = daily_bar(day, 1000)
    start = datetime(day.year, day.month, day.day, hour, 0, tzinfo=UTC)
    return bar.model_copy(update={"bar_start_utc": start, "bar_end_utc": start + timedelta(days=1)})


@pytest.mark.parametrize(
    ("session", "hour"),
    [
        (SUMMER_DAY, 4),  # SIP label: midnight New York in EDT
        (WINTER_DAY, 5),  # SIP label: midnight New York in EST
        (SUMMER_DAY, 0),  # synthetic label: 00:00Z of the session date
        (EARLY_CLOSE_DAY, 5),
    ],
)
def test_session_bar_restamps_a_daily_bar_to_its_session(session: SessionDay, hour: int) -> None:
    raw = _labelled(session.session_date, hour)
    stamped = session_bar(raw, session)
    assert stamped.bar_start_utc == session.open_utc
    assert stamped.bar_end_utc == session.close_utc
    keep = ("symbol", "timeframe", "open", "high", "low", "close", "volume", "feed", "status")
    assert {k: getattr(stamped, k) for k in keep} == {k: getattr(raw, k) for k in keep}
    assert session_bar(stamped, session) == stamped  # already stamped: unchanged


@pytest.mark.parametrize(
    "label_day",
    [SUMMER_DAY.session_date - timedelta(days=1), SUMMER_NEXT_DAY.session_date],
)
def test_session_bar_refuses_a_bar_of_another_date(label_day: date) -> None:
    with pytest.raises(SessionBarMismatchError) as info:
        session_bar(daily_bar(label_day, 1000), SUMMER_DAY)
    assert info.value.code == SESSION_BAR_MISMATCH_CODE
    assert SUMMER_DAY.session_date.isoformat() in str(info.value)


def test_session_bar_refuses_a_label_after_the_open_and_non_daily_bars() -> None:
    late = _labelled(SUMMER_DAY.session_date, 20)  # same UTC date, but after the open
    with pytest.raises(SessionBarMismatchError):
        session_bar(late, SUMMER_DAY)
    with pytest.raises(SessionBarMismatchError, match="expected 1Day"):
        session_bar(minute_bar(SUMMER_DAY.open_utc), SUMMER_DAY)


def test_empty_session_bar() -> None:
    bar = empty_session_bar("SPY", SUMMER_DAY, feed=DataFeed.SIP)
    assert bar.timeframe is Timeframe.DAY_1
    assert bar.status is BarStatus.EMPTY
    assert (bar.bar_start_utc, bar.bar_end_utc) == (SUMMER_DAY.open_utc, SUMMER_DAY.close_utc)
    assert (bar.open, bar.close, bar.volume, bar.feed) == (None, None, 0, DataFeed.SIP)
