"""Random-entry ("monkey") benchmark for the family v2 research gate (plan WU3).

Question answered: is the strategy's per-trade result better than entering at random
times with the same trading footprint? Each of ``n_sims`` seeded simulations keeps,
per symbol, the real number of trades and their holding lengths (in sessions), and
places them at random non-overlapping session opens inside the window (uniform over all
non-overlapping arrangements: random order + random gaps).

Simulated trade: buy at the open of session ``i`` at ``open_i * (1 + c)``; sell at the
open of session ``i + h`` at ``open_(i+h) * (1 - c)`` with ``c = cost_bps / 10_000`` per
side, or at the close of session ``i`` when ``h == 0`` (a real same-session exit).
Return = ``exit / entry - 1``. Approximation (documented): no stops, no take profit, no
sizing; every trade has the same notional, so the "total return" is the SUM of the
per-trade returns (not compounded) and the expectancy is their mean. The real strategy
is measured with the same definitions on its own ``return_pct`` values.

A trade occupies sessions ``i .. i + max(h, 1) - 1``; the next trade of the same symbol
may start at the open where the previous one exits. Entries lie in the first
``n - 1`` sessions of the window so that every exit open is inside it.

Percentile of the real value: ``100 * (#sims below + 0.5 * #sims equal) / n_sims``.
Deterministic: ``random.Random(seed)``; symbols are processed in sorted order.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from backtest.benchmark import DailyBar, window_bars
from backtest.report import TradeRecord

__all__ = [
    "RandomEntryResult",
    "TradeSpec",
    "percentile_of",
    "random_entry_benchmark",
    "trade_specs_from_records",
]

_BPS = 10_000.0


@dataclass(frozen=True)
class TradeSpec:
    """Footprint of one real trade: symbol, entry session, sessions held, real return."""

    symbol: str
    entry_date: date
    holding_sessions: int
    return_pct: float
    """Real net return of the trade as a fraction (``TradeRecord.return_pct``)."""

    def __post_init__(self) -> None:
        if self.holding_sessions < 0:
            raise ValueError("holding_sessions must be >= 0")


@dataclass(frozen=True)
class RandomEntryResult:
    """Distribution of the random-entry simulations and the real strategy's percentile."""

    n_sims: int
    seed: int
    n_trades: int
    cost_bps: float
    real_expectancy: float
    real_total_return: float
    sim_expectancies: tuple[float, ...]
    sim_total_returns: tuple[float, ...]
    expectancy_quantiles: tuple[float, float, float]
    """p5, p50, p95 of the simulated expectancies."""
    total_return_quantiles: tuple[float, float, float]
    expectancy_percentile: float
    total_return_percentile: float


def trade_specs_from_records(
    trades: Sequence[TradeRecord], session_dates: Sequence[date]
) -> list[TradeSpec]:
    """Footprints of simulated trades: holding = sessions between the entry and exit
    fill dates (UTC date = session date for US regular sessions, as in the report)."""
    index = {day: i for i, day in enumerate(sorted(session_dates))}
    specs: list[TradeSpec] = []
    for trade in trades:
        entry_day = trade.entry_filled_at_utc.date()
        exit_day = trade.exit_filled_at_utc.date()
        if entry_day not in index or exit_day not in index:
            raise ValueError(f"{trade.trade_id}: entry or exit date is not a known session")
        specs.append(
            TradeSpec(
                symbol=trade.symbol,
                entry_date=entry_day,
                holding_sessions=index[exit_day] - index[entry_day],
                return_pct=float(trade.return_pct),
            )
        )
    return specs


def percentile_of(value: float, sample: Sequence[float]) -> float:
    """Mid-rank percentile of ``value`` in ``sample`` (0..100)."""
    if not sample:
        raise ValueError("sample must not be empty")
    below = sum(1 for s in sample if s < value)
    equal = sum(1 for s in sample if s == value)
    return 100.0 * (below + 0.5 * equal) / len(sample)


def _quantiles(sample: Sequence[float]) -> tuple[float, float, float]:
    ordered = sorted(sample)
    last = len(ordered) - 1

    def at(q: float) -> float:
        position = q * last
        low = math.floor(position)
        high = min(low + 1, last)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return at(0.05), at(0.5), at(0.95)


@dataclass(frozen=True)
class _SymbolPlan:
    opens: tuple[float, ...]
    closes: tuple[float, ...]
    holdings: tuple[int, ...]
    free_slots: int


def _plan(symbol: str, bars: Sequence[DailyBar], holdings: list[int]) -> _SymbolPlan:
    slots = len(bars) - 1
    occupied = sum(max(h, 1) for h in holdings)
    if occupied > slots:
        raise ValueError(
            f"{symbol}: {len(holdings)} trades need {occupied} sessions, the window has {slots}"
        )
    return _SymbolPlan(
        opens=tuple(float(b.open) for b in bars),
        closes=tuple(float(b.close) for b in bars),
        holdings=tuple(holdings),
        free_slots=slots - occupied,
    )


def _simulate(plan: _SymbolPlan, rng: random.Random, cost: float, out: list[float]) -> None:
    count = len(plan.holdings)
    order = list(range(count))
    rng.shuffle(order)
    positions = sorted(rng.sample(range(plan.free_slots + count), count))
    offset = 0
    for rank, position in enumerate(positions):
        holding = plan.holdings[order[rank]]
        entry = position - rank + offset
        offset += max(holding, 1)
        buy = plan.opens[entry] * (1.0 + cost)
        sell_at = plan.opens[entry + holding] if holding > 0 else plan.closes[entry]
        out.append(sell_at * (1.0 - cost) / buy - 1.0)


def random_entry_benchmark(
    bars_by_symbol: Mapping[str, Sequence[DailyBar]],
    trades: Sequence[TradeSpec],
    *,
    cost_bps: Decimal,
    n_sims: int = 1000,
    seed: int = 0,
    start: date | None = None,
    end: date | None = None,
) -> RandomEntryResult:
    """Run the random-entry simulations and rank the real strategy among them."""
    if not trades:
        raise ValueError("the real strategy has no trades")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    if cost_bps < 0:
        raise ValueError("cost_bps must be >= 0")
    holdings: dict[str, list[int]] = {}
    for trade in trades:
        holdings.setdefault(trade.symbol, []).append(trade.holding_sessions)
    plans: list[_SymbolPlan] = []
    for symbol in sorted(holdings):
        if symbol not in bars_by_symbol:
            raise ValueError(f"{symbol}: no bars for a traded symbol")
        window = window_bars(bars_by_symbol[symbol], start, end)
        plans.append(_plan(symbol, window, sorted(holdings[symbol])))

    cost = float(cost_bps) / _BPS
    rng = random.Random(seed)
    expectancies: list[float] = []
    totals: list[float] = []
    for _ in range(n_sims):
        returns: list[float] = []
        for plan in plans:
            _simulate(plan, rng, cost, returns)
        total = math.fsum(returns)
        totals.append(total)
        expectancies.append(total / len(returns))

    real_total = math.fsum(t.return_pct for t in trades)
    real_expectancy = real_total / len(trades)
    return RandomEntryResult(
        n_sims=n_sims,
        seed=seed,
        n_trades=len(trades),
        cost_bps=float(cost_bps),
        real_expectancy=real_expectancy,
        real_total_return=real_total,
        sim_expectancies=tuple(expectancies),
        sim_total_returns=tuple(totals),
        expectancy_quantiles=_quantiles(expectancies),
        total_return_quantiles=_quantiles(totals),
        expectancy_percentile=percentile_of(real_expectancy, expectancies),
        total_return_percentile=percentile_of(real_total, totals),
    )
