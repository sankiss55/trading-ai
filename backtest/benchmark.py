"""Buy-and-hold benchmarks for the family v2 research gate (plan WU3).

Built from the same daily bars and per-side cost (bps) as the strategy:

* **Per symbol**: buy whole shares at the open of the first session of the window at
  ``open * (1 + cost_bps / 10_000)``, mark the position at every session close, sell
  everything at the last close at ``close * (1 - cost_bps / 10_000)``. Unspent cash
  stays as cash (no interest).
* **Equal-weight basket**: the starting cash is split equally between the symbols
  (each leg gets ``cash / n`` rounded down to the cent; the remainder stays as cash)
  and every leg is a per-symbol buy-and-hold. **No rebalancing**: the weights drift
  with prices. All symbols must have bars on exactly the same sessions of the window
  (fail closed on a missing bar).

Equity series are ``[starting_cash, close_1, ..., close_n]`` (the last value is after
the sale), so :func:`backtest.stats.daily_returns` gives one return per session.
Dividends are not credited (``split``-adjusted bars, DECISIONS.md).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_DOWN, Decimal
from zoneinfo import ZoneInfo

from adapters.simulation.static_calendar import MARKET_TIMEZONE
from backtest.stats import PerformanceSummary, daily_returns, performance_summary
from domain.models import Bar, BarStatus

__all__ = [
    "BenchmarkResult",
    "DailyBar",
    "buy_and_hold",
    "daily_bars_from_bars",
    "equal_weight_basket",
    "window_bars",
]

_BPS = Decimal(10_000)
_CENT = Decimal("0.01")


@dataclass(frozen=True)
class DailyBar:
    """The two prices of one session that the benchmarks trade at."""

    session_date: date
    open: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        if self.open <= 0 or self.close <= 0:
            raise ValueError(f"{self.session_date}: prices must be positive")


@dataclass(frozen=True)
class BenchmarkResult:
    """Equity series and performance of one buy-and-hold benchmark."""

    label: str
    session_dates: tuple[date, ...]
    equity: tuple[Decimal, ...]
    """``[starting_cash, equity at close_1, ..., final equity after the sale]``."""
    shares: tuple[tuple[str, int], ...]
    summary: PerformanceSummary


def daily_bars_from_bars(bars: Sequence[Bar]) -> list[DailyBar]:
    """Convert domain daily bars to :class:`DailyBar` (session date in New York time of
    ``bar_start_utc``, valid for SIP bars stamped at midnight or at the session open).
    ``EMPTY`` bars are skipped; duplicated session dates are rejected."""
    zone = ZoneInfo(MARKET_TIMEZONE)
    out: dict[date, DailyBar] = {}
    for bar in bars:
        if bar.status is BarStatus.EMPTY or bar.open is None or bar.close is None:
            continue
        day = bar.bar_start_utc.astimezone(zone).date()
        if day in out:
            raise ValueError(f"{bar.symbol}: two daily bars for {day}")
        out[day] = DailyBar(session_date=day, open=bar.open, close=bar.close)
    return [out[day] for day in sorted(out)]


def window_bars(
    bars: Sequence[DailyBar], start: date | None = None, end: date | None = None
) -> list[DailyBar]:
    """Bars with ``start <= session_date <= end`` (inclusive, ``None`` = open), sorted;
    duplicated dates are rejected."""
    selected = sorted(
        (
            b
            for b in bars
            if (start is None or b.session_date >= start) and (end is None or b.session_date <= end)
        ),
        key=lambda b: b.session_date,
    )
    for previous, current in itertools.pairwise(selected):
        if previous.session_date == current.session_date:
            raise ValueError(f"two bars for {current.session_date}")
    return selected


def _leg_equity(
    bars: Sequence[DailyBar], cash: Decimal, cost_bps: Decimal
) -> tuple[list[Decimal], int]:
    fee = cost_bps / _BPS
    buy_price = bars[0].open * (1 + fee)
    shares = int(cash // buy_price)
    left = cash - shares * buy_price
    equity = [left + shares * bar.close for bar in bars[:-1]]
    equity.append(left + shares * bars[-1].close * (1 - fee))
    return equity, shares


def _check_inputs(starting_cash: Decimal, cost_bps: Decimal) -> None:
    if starting_cash <= 0:
        raise ValueError("starting_cash must be positive")
    if cost_bps < 0:
        raise ValueError("cost_bps must be >= 0")


def _result(
    label: str,
    dates: Sequence[date],
    equity: list[Decimal],
    shares: dict[str, int],
) -> BenchmarkResult:
    invested = any(count > 0 for count in shares.values())
    summary = performance_summary(daily_returns(equity), exposure_flags=[invested] * len(dates))
    return BenchmarkResult(
        label=label,
        session_dates=tuple(dates),
        equity=tuple(equity),
        shares=tuple(sorted(shares.items())),
        summary=summary,
    )


def buy_and_hold(
    symbol: str,
    bars: Sequence[DailyBar],
    *,
    starting_cash: Decimal,
    cost_bps: Decimal,
    start: date | None = None,
    end: date | None = None,
) -> BenchmarkResult:
    """Buy-and-hold of one symbol over ``[start, end]`` (see the module docstring)."""
    _check_inputs(starting_cash, cost_bps)
    window = window_bars(bars, start, end)
    if not window:
        raise ValueError(f"{symbol}: no bars in the window {start}..{end}")
    equity, shares = _leg_equity(window, starting_cash, cost_bps)
    dates = [b.session_date for b in window]
    return _result(symbol, dates, [starting_cash, *equity], {symbol: shares})


def equal_weight_basket(
    bars_by_symbol: Mapping[str, Sequence[DailyBar]],
    *,
    starting_cash: Decimal,
    cost_bps: Decimal,
    start: date | None = None,
    end: date | None = None,
) -> BenchmarkResult:
    """Equal-weight buy-and-hold basket, no rebalancing (see the module docstring)."""
    _check_inputs(starting_cash, cost_bps)
    if not bars_by_symbol:
        raise ValueError("the basket needs at least one symbol")
    symbols = sorted(bars_by_symbol)
    windows = {s: window_bars(bars_by_symbol[s], start, end) for s in symbols}
    dates = [b.session_date for b in windows[symbols[0]]]
    if not dates:
        raise ValueError(f"no bars in the window {start}..{end}")
    for symbol in symbols:
        if [b.session_date for b in windows[symbol]] != dates:
            raise ValueError(f"{symbol}: sessions differ from {symbols[0]} in the window")
    leg_cash = (starting_cash / len(symbols)).quantize(_CENT, rounding=ROUND_DOWN)
    total = [starting_cash - leg_cash * len(symbols)] * len(dates)
    shares: dict[str, int] = {}
    for symbol in symbols:
        leg, shares[symbol] = _leg_equity(windows[symbol], leg_cash, cost_bps)
        total = [a + b for a, b in zip(total, leg, strict=True)]
    label = "equal_weight(" + ",".join(symbols) + ")"
    return _result(label, dates, [starting_cash, *total], shares)
