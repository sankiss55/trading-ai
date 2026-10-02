"""Asset-quality checks of the trading universe (sec. 12, 44).

Each check is a small pure function returning a ``CheckResult`` with a stable code
from :class:`UniverseCode`. Thresholds are always parameters (no trading constants).
The single check library (sec. 19) composes these; the market-open / entry-window
check of sec. 12 lives in ``domain.market.session``.

Fail closed: missing or unusable data (no quote, not enough daily bars, no price)
makes the check fail with an explicit code.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from domain.models import Bar, BarStatus, CheckResult, Quote, Symbol, Timeframe

__all__ = [
    "BPS_PER_UNIT",
    "UniverseCode",
    "average_daily_volume",
    "check_avg_daily_volume",
    "check_corporate_action",
    "check_data_fresh",
    "check_price_range",
    "check_quote_freshness",
    "check_spread",
    "check_symbol_listed",
    "check_tradable",
    "spread_bps",
]

BPS_PER_UNIT = Decimal(10_000)
"""Basis points per unit (unit conversion, not a trading threshold)."""


class UniverseCode(StrEnum):
    """Stable codes of the universe checks."""

    SYMBOL_LISTED = "SYMBOL_LISTED"
    SYMBOL_NOT_WHITELISTED = "SYMBOL_NOT_WHITELISTED"
    SYMBOL_BLACKLISTED = "SYMBOL_BLACKLISTED"
    ASSET_TRADABLE = "ASSET_TRADABLE"
    ASSET_NOT_TRADABLE = "ASSET_NOT_TRADABLE"
    PRICE_IN_RANGE = "PRICE_IN_RANGE"
    PRICE_BELOW_MIN = "PRICE_BELOW_MIN"
    PRICE_ABOVE_MAX = "PRICE_ABOVE_MAX"
    PRICE_UNAVAILABLE = "PRICE_UNAVAILABLE"
    AVG_VOLUME_OK = "AVG_VOLUME_OK"
    AVG_VOLUME_TOO_LOW = "AVG_VOLUME_TOO_LOW"
    AVG_VOLUME_INSUFFICIENT_DATA = "AVG_VOLUME_INSUFFICIENT_DATA"
    SPREAD_OK = "SPREAD_OK"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    QUOTE_UNAVAILABLE = "QUOTE_UNAVAILABLE"
    QUOTE_INVALID = "QUOTE_INVALID"
    QUOTE_FRESH = "QUOTE_FRESH"
    QUOTE_STALE = "QUOTE_STALE"
    DATA_FRESH = "DATA_FRESH"
    DATA_STALE = "DATA_STALE"
    NO_CORPORATE_ACTION = "NO_CORPORATE_ACTION"
    CORPORATE_ACTION_TODAY = "CORPORATE_ACTION_TODAY"


def _result(passed: bool, code: UniverseCode, **detail: object) -> CheckResult:
    return CheckResult(passed=passed, code=code.value, detail=detail)


# --------------------------------------------------------------------------- listing / asset


def check_symbol_listed(
    symbol: Symbol, *, whitelist: Collection[str], blacklist: Collection[str]
) -> CheckResult:
    """Symbol must be in the whitelist and not in the blacklist (blacklist wins)."""
    if symbol in blacklist:
        return _result(False, UniverseCode.SYMBOL_BLACKLISTED, symbol=symbol)
    if symbol not in whitelist:
        return _result(False, UniverseCode.SYMBOL_NOT_WHITELISTED, symbol=symbol)
    return _result(True, UniverseCode.SYMBOL_LISTED, symbol=symbol)


def check_tradable(symbol: Symbol, *, tradable: bool) -> CheckResult:
    """Broker ``tradable`` flag of the asset (refreshed at startup and daily)."""
    if not tradable:
        return _result(False, UniverseCode.ASSET_NOT_TRADABLE, symbol=symbol)
    return _result(True, UniverseCode.ASSET_TRADABLE, symbol=symbol)


# --------------------------------------------------------------------------- price


def check_price_range(
    symbol: Symbol, last_price: Decimal | None, *, min_price: Decimal, max_price: Decimal
) -> CheckResult:
    """``min_price <= last_price <= max_price`` (last closed bar's close, sec. 12)."""
    detail = {
        "symbol": symbol,
        "price": None if last_price is None else str(last_price),
        "min_price": str(min_price),
        "max_price": str(max_price),
    }
    if last_price is None:
        return _result(False, UniverseCode.PRICE_UNAVAILABLE, **detail)
    if last_price < min_price:
        return _result(False, UniverseCode.PRICE_BELOW_MIN, **detail)
    if last_price > max_price:
        return _result(False, UniverseCode.PRICE_ABOVE_MAX, **detail)
    return _result(True, UniverseCode.PRICE_IN_RANGE, **detail)


# --------------------------------------------------------------------------- volume


def average_daily_volume(
    daily_bars: Sequence[Bar], *, lookback_days: int, as_of_utc: datetime
) -> Decimal | None:
    """Mean volume of the last ``lookback_days`` completed daily bars.

    Only ``1Day`` bars that ended at or before ``as_of_utc`` count (previous days, never
    today's partial bar; sec. 10.2.1). EMPTY bars are ignored. ``None`` when fewer than
    ``lookback_days`` bars are available.

    Raises:
        ValueError: ``lookback_days < 1``.
    """
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
    eligible = sorted(
        (
            bar
            for bar in daily_bars
            if bar.timeframe is Timeframe.DAY_1
            and bar.status is not BarStatus.EMPTY
            and bar.bar_end_utc <= as_of_utc
        ),
        key=lambda bar: bar.bar_start_utc,
    )
    if len(eligible) < lookback_days:
        return None
    window = eligible[-lookback_days:]
    return Decimal(sum(bar.volume for bar in window)) / Decimal(lookback_days)


def check_avg_daily_volume(
    symbol: Symbol,
    daily_bars: Sequence[Bar],
    *,
    lookback_days: int,
    min_avg_daily_volume: Decimal | int,
    as_of_utc: datetime,
) -> CheckResult:
    """Average daily volume over ``lookback_days`` must be ``>= min_avg_daily_volume``."""
    average = average_daily_volume(daily_bars, lookback_days=lookback_days, as_of_utc=as_of_utc)
    detail = {
        "symbol": symbol,
        "avg_daily_volume": None if average is None else str(average),
        "min_avg_daily_volume": str(min_avg_daily_volume),
        "lookback_days": lookback_days,
    }
    if average is None:
        return _result(False, UniverseCode.AVG_VOLUME_INSUFFICIENT_DATA, **detail)
    if average < Decimal(min_avg_daily_volume):
        return _result(False, UniverseCode.AVG_VOLUME_TOO_LOW, **detail)
    return _result(True, UniverseCode.AVG_VOLUME_OK, **detail)


# --------------------------------------------------------------------------- spread / quote


def spread_bps(quote: Quote) -> Decimal | None:
    """Spread in basis points of the mid price: ``(ask - bid) / mid * 10000``.

    ``None`` for an unusable quote (bid or ask not positive, or crossed: ask < bid).
    """
    if quote.bid_price <= 0 or quote.ask_price <= 0 or quote.ask_price < quote.bid_price:
        return None
    mid = (quote.bid_price + quote.ask_price) / 2
    return (quote.ask_price - quote.bid_price) / mid * BPS_PER_UNIT


def check_spread(symbol: Symbol, quote: Quote | None, *, max_spread_bps: Decimal) -> CheckResult:
    """Spread must be ``<= max_spread_bps`` (sec. 10.5; only when the filter applies)."""
    if quote is None:
        return _result(False, UniverseCode.QUOTE_UNAVAILABLE, symbol=symbol)
    spread = spread_bps(quote)
    detail = {
        "symbol": symbol,
        "bid": str(quote.bid_price),
        "ask": str(quote.ask_price),
        "spread_bps": None if spread is None else str(spread),
        "max_spread_bps": str(max_spread_bps),
    }
    if spread is None:
        return _result(False, UniverseCode.QUOTE_INVALID, **detail)
    if spread > max_spread_bps:
        return _result(False, UniverseCode.SPREAD_TOO_WIDE, **detail)
    return _result(True, UniverseCode.SPREAD_OK, **detail)


def check_quote_freshness(
    symbol: Symbol, quote: Quote | None, *, now_utc: datetime, max_quote_age_seconds: float
) -> CheckResult:
    """Quote age ``now_utc - quote.timestamp_utc`` must be ``<= max_quote_age_seconds``."""
    if quote is None:
        return _result(False, UniverseCode.QUOTE_UNAVAILABLE, symbol=symbol)
    age = (now_utc - quote.timestamp_utc).total_seconds()
    detail = {"symbol": symbol, "age_seconds": age, "max_quote_age_seconds": max_quote_age_seconds}
    if age > max_quote_age_seconds:
        return _result(False, UniverseCode.QUOTE_STALE, **detail)
    return _result(True, UniverseCode.QUOTE_FRESH, **detail)


# --------------------------------------------------------------------------- flags


def check_data_fresh(symbol: Symbol, *, is_stale: bool) -> CheckResult:
    """Market Data Engine freshness flag (see ``domain.market.quality.check_staleness``)."""
    if is_stale:
        return _result(False, UniverseCode.DATA_STALE, symbol=symbol)
    return _result(True, UniverseCode.DATA_FRESH, symbol=symbol)


def check_corporate_action(symbol: Symbol, *, has_corporate_action_today: bool) -> CheckResult:
    """No entries on a day with a corporate action on the symbol (sec. 44)."""
    if has_corporate_action_today:
        return _result(False, UniverseCode.CORPORATE_ACTION_TODAY, symbol=symbol)
    return _result(True, UniverseCode.NO_CORPORATE_ACTION, symbol=symbol)
