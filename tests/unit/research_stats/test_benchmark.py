"""Buy-and-hold benchmarks (:mod:`backtest.benchmark`) on tiny hand-made series."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from backtest.benchmark import (
    DailyBar,
    buy_and_hold,
    daily_bars_from_bars,
    equal_weight_basket,
    window_bars,
)
from domain.models import Bar, BarStatus, DataFeed, Timeframe

D = Decimal
DAYS = [date(2021, 3, 1), date(2021, 3, 2), date(2021, 3, 3)]


def _bars(prices: list[tuple[str, str]]) -> list[DailyBar]:
    return [
        DailyBar(session_date=day, open=D(o), close=D(c))
        for day, (o, c) in zip(DAYS, prices, strict=True)
    ]


A = _bars([("100", "101"), ("102", "104"), ("103", "110")])
B = _bars([("50", "50"), ("51", "52"), ("54", "55")])


def test_buy_and_hold_arithmetic() -> None:
    # buy 100 * 1.001 = 100.1 -> 9 shares (900.9), cash 99.1; marks 99.1 + 9 * close;
    # sale at 110 * 0.999 = 109.89.
    result = buy_and_hold("AAA", A, starting_cash=D(1000), cost_bps=D(10))
    assert result.shares == (("AAA", 9),)
    assert result.equity == (D(1000), D("1008.1"), D("1035.1"), D("1088.11"))
    assert result.session_dates == tuple(DAYS)
    summary = result.summary
    assert summary.n_periods == 3
    assert summary.total_return == pytest.approx(0.08811)
    assert summary.max_drawdown == 0.0
    assert summary.exposure == 1.0


def test_buy_and_hold_drawdown_and_window() -> None:
    bars = _bars([("100", "101"), ("102", "95"), ("103", "110")])
    result = buy_and_hold("AAA", bars, starting_cash=D(1000), cost_bps=D(10))
    assert result.equity[2] == D("954.1")
    assert result.summary.max_drawdown == pytest.approx((1008.1 - 954.1) / 1008.1)
    # A one-session window buys at the open and sells at the same close.
    single = buy_and_hold(
        "AAA", A, starting_cash=D(1000), cost_bps=D(0), start=DAYS[1], end=DAYS[1]
    )
    assert single.equity == (D(1000), D(1000) - 9 * D(102) + 9 * D(104))


def test_buy_and_hold_refusals() -> None:
    with pytest.raises(ValueError, match="no bars in the window"):
        buy_and_hold("AAA", A, starting_cash=D(1000), cost_bps=D(5), start=date(2022, 1, 1))
    with pytest.raises(ValueError, match="starting_cash"):
        buy_and_hold("AAA", A, starting_cash=D(0), cost_bps=D(5))
    with pytest.raises(ValueError, match="cost_bps"):
        buy_and_hold("AAA", A, starting_cash=D(1000), cost_bps=D(-1))


def test_equal_weight_basket_without_rebalancing() -> None:
    # 500 per leg. A: 4 shares at 100.1, cash 99.6. B: 9 shares at 50.05, cash 49.55.
    result = equal_weight_basket({"BBB": B, "AAA": A}, starting_cash=D(1000), cost_bps=D(10))
    assert result.label == "equal_weight(AAA,BBB)"
    assert result.shares == (("AAA", 4), ("BBB", 9))
    assert result.equity == (D(1000), D("1003.15"), D("1033.15"), D("1083.215"))


def test_equal_weight_basket_keeps_the_rounding_remainder_as_cash() -> None:
    result = equal_weight_basket({"AAA": A, "BBB": B}, starting_cash=D("1000.01"), cost_bps=D(10))
    assert result.equity[1] == D("1003.16")


def test_equal_weight_basket_requires_the_same_sessions() -> None:
    with pytest.raises(ValueError, match="sessions differ"):
        equal_weight_basket({"AAA": A, "BBB": B[:2]}, starting_cash=D(1000), cost_bps=D(5))
    with pytest.raises(ValueError, match="at least one symbol"):
        equal_weight_basket({}, starting_cash=D(1000), cost_bps=D(5))


def test_window_bars_sorts_filters_and_rejects_duplicates() -> None:
    assert window_bars(list(reversed(A)), DAYS[1]) == A[1:]
    with pytest.raises(ValueError, match="two bars for"):
        window_bars([A[0], A[0]])
    with pytest.raises(ValueError, match="prices must be positive"):
        DailyBar(session_date=DAYS[0], open=D(0), close=D(1))


def _bar(start: datetime, status: BarStatus = BarStatus.COMPLETE) -> Bar:
    prices = None if status is BarStatus.EMPTY else D("10")
    return Bar(
        symbol="SPY",
        timeframe=Timeframe.DAY_1,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(days=1),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=0 if status is BarStatus.EMPTY else 100,
        feed=DataFeed.SIP,
        status=status,
    )


def test_daily_bars_from_domain_bars_uses_the_new_york_date() -> None:
    bars = [
        _bar(datetime(2021, 3, 2, 14, 30, tzinfo=UTC)),  # stamped at the session open
        _bar(datetime(2021, 3, 1, 5, 0, tzinfo=UTC)),  # SIP daily bar at NY midnight
        _bar(datetime(2021, 3, 3, 5, 0, tzinfo=UTC), BarStatus.EMPTY),
    ]
    converted = daily_bars_from_bars(bars)
    assert [b.session_date for b in converted] == [date(2021, 3, 1), date(2021, 3, 2)]
    with pytest.raises(ValueError, match="two daily bars"):
        daily_bars_from_bars([bars[1], _bar(datetime(2021, 3, 1, 14, 30, tzinfo=UTC))])
