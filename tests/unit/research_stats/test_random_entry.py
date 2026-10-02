"""Random-entry benchmark (:mod:`backtest.random_entry`)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from backtest.benchmark import DailyBar
from backtest.random_entry import (
    TradeSpec,
    percentile_of,
    random_entry_benchmark,
    trade_specs_from_records,
)
from backtest.report import TradeRecord
from domain.models import ExitReason

D = Decimal
START = date(2021, 1, 4)


def _days(n: int) -> list[date]:
    return [START + timedelta(days=i) for i in range(n)]


def _series(opens: list[float], closes: list[float] | None = None) -> list[DailyBar]:
    closes = closes or opens
    return [
        DailyBar(session_date=day, open=D(repr(o)), close=D(repr(c)))
        for day, o, c in zip(_days(len(opens)), opens, closes, strict=True)
    ]


def _spec(symbol: str, holding: int, return_pct: float = 0.0, day: int = 0) -> TradeSpec:
    return TradeSpec(
        symbol=symbol,
        entry_date=START + timedelta(days=day),
        holding_sessions=holding,
        return_pct=return_pct,
    )


def test_percentile_of_uses_mid_ranks() -> None:
    assert percentile_of(3.0, [1.0, 2.0, 3.0, 4.0]) == 62.5
    assert percentile_of(10.0, [1.0, 2.0]) == 100.0
    assert percentile_of(0.0, [1.0, 2.0]) == 0.0
    with pytest.raises(ValueError, match="must not be empty"):
        percentile_of(1.0, [])


def test_simulations_are_deterministic_for_a_seed() -> None:
    opens = [100.0 + ((i * 37) % 11) for i in range(60)]
    bars = {"AAA": _series(opens), "BBB": _series(list(reversed(opens)))}
    trades = [_spec("AAA", 3, 0.01), _spec("AAA", 5, -0.02), _spec("BBB", 2, 0.03)]
    first = random_entry_benchmark(bars, trades, cost_bps=D(5), n_sims=200, seed=1)
    assert first == random_entry_benchmark(bars, trades, cost_bps=D(5), n_sims=200, seed=1)
    other = random_entry_benchmark(bars, trades, cost_bps=D(5), n_sims=200, seed=2)
    assert first.sim_expectancies != other.sim_expectancies
    assert first.n_trades == 3
    assert first.real_total_return == pytest.approx(0.02)
    assert first.real_expectancy == pytest.approx(0.02 / 3)


def test_holding_lengths_are_kept() -> None:
    # Opens grow 1% per session: a trade held h sessions returns 1.01^h - 1 wherever it
    # starts, so every simulation has exactly the real footprint's mean return.
    opens = [100.0 * 1.01**i for i in range(40)]
    holdings = [1, 2, 5]
    trades = [_spec("AAA", h, 1.01**h - 1) for h in holdings]
    result = random_entry_benchmark({"AAA": _series(opens)}, trades, cost_bps=D(0), n_sims=50)
    expected = sum(1.01**h - 1 for h in holdings) / 3
    assert all(e == pytest.approx(expected) for e in result.sim_expectancies)


def test_entries_do_not_overlap() -> None:
    # Holdings 2 and 3 fill the 5 entry slots of a 6-session window exactly: the only
    # arrangements are entries (0, 2) [h=2 first] or (0, 3) [h=3 first].
    opens = [100.0, 103.0, 101.0, 107.0, 104.0, 111.0]
    trades = [_spec("AAA", 2), _spec("AAA", 3)]

    def r(entry: int, holding: int) -> float:
        return opens[entry + holding] / opens[entry] - 1

    allowed = {round(r(0, 2) + r(2, 3), 12), round(r(0, 3) + r(3, 2), 12)}
    result = random_entry_benchmark({"AAA": _series(opens)}, trades, cost_bps=D(0), n_sims=100)
    observed = {round(t, 12) for t in result.sim_total_returns}
    assert observed == allowed


def test_costs_and_same_session_exits() -> None:
    flat = _series([100.0] * 20)
    trades = [_spec("AAA", 2, 0.001), _spec("AAA", 4, 0.001)]
    result = random_entry_benchmark({"AAA": flat}, trades, cost_bps=D(10), n_sims=20)
    per_trade = 0.999 / 1.001 - 1
    assert all(e == pytest.approx(per_trade) for e in result.sim_expectancies)
    assert result.expectancy_percentile == 100.0
    assert result.expectancy_quantiles == pytest.approx((per_trade, per_trade, per_trade))
    # h == 0 exits at the close of the entry session: open 100 -> close 102.
    intraday = _series([100.0] * 10, [102.0] * 10)
    same_day = random_entry_benchmark(
        {"AAA": intraday}, [_spec("AAA", 0, 102.0 / 100.0 - 1)], cost_bps=D(0), n_sims=10
    )
    assert all(e == pytest.approx(0.02) for e in same_day.sim_expectancies)
    assert same_day.expectancy_percentile == pytest.approx(50.0)


def test_window_restricts_the_entries() -> None:
    opens = [100.0] * 10 + [200.0 * 1.01**i for i in range(10)]
    bars = {"AAA": _series(opens)}
    window_start = START + timedelta(days=10)
    result = random_entry_benchmark(
        bars, [_spec("AAA", 1)], cost_bps=D(0), n_sims=30, start=window_start
    )
    assert all(e == pytest.approx(0.01) for e in result.sim_expectancies)


def test_refusals() -> None:
    bars = {"AAA": _series([100.0] * 5)}
    with pytest.raises(ValueError, match="need 6 sessions"):  # only 4 entry slots
        random_entry_benchmark(bars, [_spec("AAA", 3), _spec("AAA", 3)], cost_bps=D(0))
    with pytest.raises(ValueError, match="no bars for a traded symbol"):
        random_entry_benchmark(bars, [_spec("ZZZ", 1)], cost_bps=D(0))
    with pytest.raises(ValueError, match="no trades"):
        random_entry_benchmark(bars, [], cost_bps=D(0))
    with pytest.raises(ValueError, match="n_sims"):
        random_entry_benchmark(bars, [_spec("AAA", 1)], cost_bps=D(0), n_sims=0)
    with pytest.raises(ValueError, match="holding_sessions"):
        _spec("AAA", -1)


def _record(entry_day: date, exit_day: date) -> TradeRecord:
    opened = datetime(entry_day.year, entry_day.month, entry_day.day, 14, 30, tzinfo=UTC)
    closed = datetime(exit_day.year, exit_day.month, exit_day.day, 14, 30, tzinfo=UTC)
    return TradeRecord(
        trade_id="sig_1",
        symbol="SPY",
        qty=10,
        entry_ref=D(100),
        stop_price=D(95),
        take_profit_price=D(150),
        entry_filled_at_utc=opened,
        entry_price=D(100),
        exit_filled_at_utc=closed,
        exit_price=D(102),
        exit_reason=ExitReason.SIGNAL_REVERSAL,
        gross_pnl=D(20),
        commissions=D(0),
        net_pnl=D(20),
        return_pct=D("0.02"),
        result_r=D("0.4"),
        duration_minutes=D(0),
        entry_slippage=D(0),
    )


def test_trade_specs_from_records_count_sessions() -> None:
    sessions = [date(2021, 3, 1), date(2021, 3, 2), date(2021, 3, 3), date(2021, 3, 5)]
    specs = trade_specs_from_records([_record(sessions[0], sessions[3])], sessions)
    assert specs == [TradeSpec("SPY", sessions[0], 3, 0.02)]
    with pytest.raises(ValueError, match="not a known session"):
        trade_specs_from_records([_record(sessions[0], date(2021, 3, 4))], sessions)
