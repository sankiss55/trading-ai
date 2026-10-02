"""Universe / asset-quality check tests (sec. 12, 44). Thresholds are fixture values."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from domain.market.universe import (
    UniverseCode,
    average_daily_volume,
    check_avg_daily_volume,
    check_corporate_action,
    check_data_fresh,
    check_price_range,
    check_quote_freshness,
    check_spread,
    check_symbol_listed,
    check_tradable,
    spread_bps,
)
from domain.models import Bar, BarStatus, CheckResult, DataFeed, Quote, Timeframe
from tests.unit.market.factories import daily_bar

NOW = datetime(2026, 6, 15, 14, 0, tzinfo=UTC)


def _quote(bid: str, ask: str, *, at: datetime = NOW) -> Quote:
    return Quote(
        symbol="SPY",
        bid_price=Decimal(bid),
        ask_price=Decimal(ask),
        bid_size=Decimal("100"),
        ask_size=Decimal("100"),
        timestamp_utc=at,
    )


def _assert(result: CheckResult, passed: bool, code: UniverseCode) -> None:
    assert result.passed is passed
    assert result.code == code.value


# --------------------------------------------------------------------------- listing / flags


@pytest.mark.parametrize(
    ("symbol", "passed", "code"),
    [
        ("SPY", True, UniverseCode.SYMBOL_LISTED),
        ("TSLA", False, UniverseCode.SYMBOL_NOT_WHITELISTED),
        ("QQQ", False, UniverseCode.SYMBOL_BLACKLISTED),  # blacklist wins over whitelist
    ],
)
def test_check_symbol_listed(symbol: str, passed: bool, code: UniverseCode) -> None:
    result = check_symbol_listed(symbol, whitelist={"SPY", "QQQ"}, blacklist={"QQQ"})
    _assert(result, passed, code)


def test_flag_checks() -> None:
    _assert(check_tradable("SPY", tradable=True), True, UniverseCode.ASSET_TRADABLE)
    _assert(check_tradable("SPY", tradable=False), False, UniverseCode.ASSET_NOT_TRADABLE)
    _assert(check_data_fresh("SPY", is_stale=False), True, UniverseCode.DATA_FRESH)
    _assert(check_data_fresh("SPY", is_stale=True), False, UniverseCode.DATA_STALE)
    _assert(
        check_corporate_action("SPY", has_corporate_action_today=False),
        True,
        UniverseCode.NO_CORPORATE_ACTION,
    )
    _assert(
        check_corporate_action("SPY", has_corporate_action_today=True),
        False,
        UniverseCode.CORPORATE_ACTION_TODAY,
    )


# --------------------------------------------------------------------------- price


@pytest.mark.parametrize(
    ("price", "passed", "code"),
    [
        (Decimal("10"), True, UniverseCode.PRICE_IN_RANGE),  # inclusive min
        (Decimal("500"), True, UniverseCode.PRICE_IN_RANGE),  # inclusive max
        (Decimal("9.99"), False, UniverseCode.PRICE_BELOW_MIN),
        (Decimal("500.01"), False, UniverseCode.PRICE_ABOVE_MAX),
        (None, False, UniverseCode.PRICE_UNAVAILABLE),
    ],
)
def test_check_price_range(price: Decimal | None, passed: bool, code: UniverseCode) -> None:
    result = check_price_range("SPY", price, min_price=Decimal("10"), max_price=Decimal("500"))
    _assert(result, passed, code)


# --------------------------------------------------------------------------- volume


def _daily_series(volumes: list[int]) -> list[Bar]:
    first = date(2026, 6, 1)
    return [daily_bar(first + timedelta(days=i), v) for i, v in enumerate(volumes)]


def test_average_daily_volume_uses_last_completed_days_only() -> None:
    bars = _daily_series([100, 200, 300, 400, 999_999])  # June 1..5
    # as_of = June 5 14:00 UTC: the June 5 bar (ends June 6 04:00) is still in progress.
    as_of = datetime(2026, 6, 5, 14, 0, tzinfo=UTC)
    # Last 3 completed: (200 + 300 + 400) / 3 = 300
    assert average_daily_volume(bars, lookback_days=3, as_of_utc=as_of) == Decimal(300)
    # Unsorted input gives the same result.
    shuffled = list(reversed(bars))
    assert average_daily_volume(shuffled, lookback_days=3, as_of_utc=as_of) == Decimal(300)


def test_average_daily_volume_ignores_empty_and_intraday_bars() -> None:
    bars = _daily_series([100, 200])
    empty = Bar(
        symbol="SPY",
        timeframe=Timeframe.DAY_1,
        bar_start_utc=datetime(2026, 6, 3, 4, tzinfo=UTC),
        bar_end_utc=datetime(2026, 6, 4, 4, tzinfo=UTC),
        open=None,
        high=None,
        low=None,
        close=None,
        volume=0,
        feed=DataFeed.IEX,
        status=BarStatus.EMPTY,
    )
    intraday = bars[0].model_copy(update={"timeframe": Timeframe.MIN_5})
    as_of = datetime(2026, 6, 10, tzinfo=UTC)
    assert average_daily_volume(
        [*bars, empty, intraday], lookback_days=2, as_of_utc=as_of
    ) == Decimal(150)
    assert average_daily_volume([*bars, empty], lookback_days=3, as_of_utc=as_of) is None


def test_average_daily_volume_rejects_bad_lookback() -> None:
    with pytest.raises(ValueError, match="lookback_days"):
        average_daily_volume([], lookback_days=0, as_of_utc=NOW)


@pytest.mark.parametrize(
    ("minimum", "passed", "code"),
    [
        (Decimal(250), True, UniverseCode.AVG_VOLUME_OK),  # avg 250 >= 250 (inclusive)
        (251, False, UniverseCode.AVG_VOLUME_TOO_LOW),
    ],
)
def test_check_avg_daily_volume(minimum: Decimal | int, passed: bool, code: UniverseCode) -> None:
    bars = _daily_series([100, 200, 300])  # last 2: (200+300)/2 = 250
    result = check_avg_daily_volume(
        "SPY",
        bars,
        lookback_days=2,
        min_avg_daily_volume=minimum,
        as_of_utc=datetime(2026, 6, 10, tzinfo=UTC),
    )
    _assert(result, passed, code)


def test_check_avg_daily_volume_has_no_feed_assumption() -> None:
    # Owner decision 2026-10-02: the liquidity filter runs on SIP daily bars while the
    # strategy uses IEX minute bars; the check must accept daily bars of any feed.
    bars = [b.model_copy(update={"feed": DataFeed.SIP}) for b in _daily_series([6_000_000] * 3)]
    result = check_avg_daily_volume(
        "SPY",
        bars,
        lookback_days=2,
        min_avg_daily_volume=5_000_000,
        as_of_utc=datetime(2026, 6, 10, tzinfo=UTC),
    )
    _assert(result, True, UniverseCode.AVG_VOLUME_OK)


def test_check_avg_daily_volume_insufficient_data_fails_closed() -> None:
    result = check_avg_daily_volume(
        "SPY",
        _daily_series([1_000_000]),
        lookback_days=20,
        min_avg_daily_volume=1,
        as_of_utc=datetime(2026, 6, 10, tzinfo=UTC),
    )
    _assert(result, False, UniverseCode.AVG_VOLUME_INSUFFICIENT_DATA)


# --------------------------------------------------------------------------- spread / quote


@pytest.mark.parametrize(
    ("bid", "ask", "expected"),
    [
        # mid = 100.05, spread = 0.1 -> 0.1 / 100.05 * 10000 = 9.995002...
        ("100.00", "100.10", Decimal("0.1") / Decimal("100.05") * 10000),
        # mid = 50, spread = 0 -> 0
        ("50", "50", Decimal(0)),
        # mid = 10, spread = 0.2 -> 0.2 / 10 * 10000 = 200
        ("9.9", "10.1", Decimal(200)),
        ("0", "10", None),  # no bid
        ("10", "0", None),  # no ask
        ("10.1", "10", None),  # crossed
    ],
)
def test_spread_bps(bid: str, ask: str, expected: Decimal | None) -> None:
    assert spread_bps(_quote(bid, ask)) == expected


@pytest.mark.parametrize(
    ("quote", "passed", "code"),
    [
        (_quote("9.9", "10.1"), True, UniverseCode.SPREAD_OK),  # 200 bps <= 200 (inclusive)
        (_quote("9.89", "10.11"), False, UniverseCode.SPREAD_TOO_WIDE),  # 220 bps
        (_quote("10.1", "10"), False, UniverseCode.QUOTE_INVALID),
        (None, False, UniverseCode.QUOTE_UNAVAILABLE),
    ],
)
def test_check_spread(quote: Quote | None, passed: bool, code: UniverseCode) -> None:
    _assert(check_spread("SPY", quote, max_spread_bps=Decimal(200)), passed, code)


@pytest.mark.parametrize(
    ("age_seconds", "passed", "code"),
    [(5, True, UniverseCode.QUOTE_FRESH), (5.5, False, UniverseCode.QUOTE_STALE)],
)
def test_check_quote_freshness(age_seconds: float, passed: bool, code: UniverseCode) -> None:
    quote = _quote("100", "100.1", at=NOW - timedelta(seconds=age_seconds))
    result = check_quote_freshness("SPY", quote, now_utc=NOW, max_quote_age_seconds=5)
    _assert(result, passed, code)


def test_check_quote_freshness_without_quote() -> None:
    result = check_quote_freshness("SPY", None, now_utc=NOW, max_quote_age_seconds=5)
    _assert(result, False, UniverseCode.QUOTE_UNAVAILABLE)
