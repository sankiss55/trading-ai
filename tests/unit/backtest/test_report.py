"""Backtest metrics (sec. 45.3), continuation check (45.4) and period helpers."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from app.config import BacktestSection
from backtest.report import (
    EquityPoint,
    PeriodMetrics,
    TradeRecord,
    compute_metrics,
    continuation_check,
)
from backtest.runner import add_months, walk_forward_windows
from domain.models import ExitReason

D = Decimal


def _trade(day: int, entry: str, exit_: str, *, stop: str, qty: int = 10) -> TradeRecord:
    entry_d, exit_d, stop_d = D(entry), D(exit_), D(stop)
    opened = datetime(2025, 1, day, 15, 0, tzinfo=UTC)
    gross = (exit_d - entry_d) * qty
    return TradeRecord(
        trade_id=f"sig_{day}",
        symbol="SYNTH",
        qty=qty,
        entry_ref=entry_d,
        stop_price=stop_d,
        take_profit_price=entry_d + 2 * (entry_d - stop_d),
        entry_filled_at_utc=opened,
        entry_price=entry_d,
        exit_filled_at_utc=opened + timedelta(minutes=30),
        exit_price=exit_d,
        exit_reason=ExitReason.TAKE_PROFIT if exit_d > entry_d else ExitReason.STOP_LOSS,
        gross_pnl=gross,
        commissions=D(0),
        net_pnl=gross,
        return_pct=gross / (entry_d * qty),
        result_r=(exit_d - entry_d) / (entry_d - stop_d),
        duration_minutes=D(30),
        entry_slippage=D(0),
    )


def _point(day: int, equity: str) -> EquityPoint:
    return EquityPoint(
        timestamp_utc=datetime(2025, 1, day, 21, 0, tzinfo=UTC), equity=D(equity), exposure=D("0.5")
    )


TRADES = (
    _trade(2, "100", "102", stop="99"),  # +2R, +20
    _trade(3, "100", "99", stop="99"),  # -1R, -10
    _trade(6, "50", "51", stop="49"),  # +1R, +10
)
CURVE = (_point(2, "1020"), _point(3, "1010"), _point(6, "1020"))


def test_full_period_metrics() -> None:
    m = compute_metrics(
        "full",
        start=date(2025, 1, 2),
        end=date(2025, 1, 6),
        sessions=3,
        trades=TRADES,
        curve=CURVE,
        starting_cash=D(1000),
    )
    assert m.trades == 3
    assert (m.wins, m.losses) == (2, 1)
    assert m.win_rate == D("0.666667")
    assert m.profit_factor == D(3)
    assert m.expectancy_r == D("0.666667")
    assert m.avg_win_r == D("1.5")
    assert m.avg_loss_r == D(-1)
    assert m.avg_win_pct == D("0.02")
    assert m.avg_loss_pct == D("-0.01")
    assert m.total_return == D("0.02")
    assert m.max_drawdown == D("0.009804")  # (1020 - 1010) / 1020
    assert m.avg_duration_minutes == D(30)
    assert m.avg_exposure == D("0.5")
    assert m.annualized_return is not None
    assert m.annualized_return > m.total_return


def test_sub_period_uses_previous_equity_as_start_and_entry_date_for_trades() -> None:
    m = compute_metrics(
        "oos",
        start=date(2025, 1, 3),
        end=date(2025, 1, 6),
        sessions=2,
        trades=TRADES,
        curve=CURVE,
        starting_cash=D(1000),
    )
    assert m.starting_equity == D("1020.00")
    assert m.ending_equity == D("1020.00")
    assert m.total_return == 0
    assert m.trades == 2


def test_empty_period_has_no_ratios() -> None:
    m = compute_metrics(
        "empty",
        start=date(2025, 2, 1),
        end=date(2025, 2, 28),
        sessions=0,
        trades=TRADES,
        curve=CURVE,
        starting_cash=D(1000),
    )
    assert m.trades == 0
    assert m.win_rate is None
    assert m.profit_factor is None
    assert m.expectancy_r is None
    assert m.annualized_return is None
    assert m.max_drawdown == 0


def _section(**values: object) -> BacktestSection:
    base: dict[str, object] = {
        "start_date": date(2025, 1, 2),
        "end_date": date(2025, 3, 31),
        "out_of_sample_start": date(2025, 3, 1),
        "walk_forward_train_months": 1,
        "walk_forward_test_months": 1,
        "min_expectancy_r": D("0.1"),
        "min_profit_factor": D("1.2"),
        "max_drawdown_pct": D("0.2"),
        "min_trades_out_of_sample": 2,
    }
    base.update(values)
    return BacktestSection.model_validate(base)


def _oos() -> PeriodMetrics:
    return compute_metrics(
        "out_of_sample",
        start=date(2025, 1, 2),
        end=date(2025, 1, 6),
        sessions=3,
        trades=TRADES,
        curve=CURVE,
        starting_cash=D(1000),
    )


def test_continuation_pass_and_fail() -> None:
    assert continuation_check(_section(), _oos()).status == "PASS"
    failing = continuation_check(_section(min_trades_out_of_sample=10), _oos())
    assert failing.status == "FAIL"
    assert [c.name for c in failing.criteria if c.passed is False] == ["trades_out_of_sample"]


def test_continuation_pending_when_a_threshold_is_null() -> None:
    check = continuation_check(_section(min_profit_factor=None), _oos())
    assert check.status == "PENDING_OWNER_DECISION"
    assert check.pending == ("backtest.min_profit_factor",)
    by_name = {c.name: c for c in check.criteria}
    assert by_name["profit_factor"].passed is None
    assert by_name["expectancy_r"].passed is True


def test_continuation_without_trades_fails() -> None:
    empty = compute_metrics(
        "out_of_sample",
        start=date(2025, 2, 1),
        end=date(2025, 2, 28),
        sessions=20,
        trades=(),
        curve=(),
        starting_cash=D(1000),
    )
    check = continuation_check(_section(min_trades_out_of_sample=0), empty)
    assert check.status == "FAIL"


@pytest.mark.parametrize(
    ("day", "months", "expected"),
    [
        (date(2025, 1, 31), 1, date(2025, 2, 28)),
        (date(2024, 1, 31), 1, date(2024, 2, 29)),
        (date(2025, 11, 15), 3, date(2026, 2, 15)),
        (date(2025, 1, 2), 0, date(2025, 1, 2)),
    ],
)
def test_add_months(day: date, months: int, expected: date) -> None:
    assert add_months(day, months) == expected


def test_walk_forward_windows_are_sequential_and_truncated() -> None:
    windows = walk_forward_windows(
        date(2025, 1, 1), date(2025, 4, 15), train_months=2, test_months=1
    )
    assert windows == [
        (date(2025, 1, 1), date(2025, 2, 28), date(2025, 3, 1), date(2025, 3, 31)),
        (date(2025, 2, 1), date(2025, 3, 31), date(2025, 4, 1), date(2025, 4, 15)),
    ]
    assert (
        walk_forward_windows(date(2025, 1, 1), date(2025, 1, 31), train_months=1, test_months=1)
        == []
    )
