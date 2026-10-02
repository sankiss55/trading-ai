"""Backtest results (sec. 45.3), continuation check (sec. 45.4) and text rendering.

Definitions (all P&L from simulated fills, sec. 39):

* ``gross_pnl = (exit_price - entry_price) * qty``; ``net_pnl = gross_pnl - commissions``.
* ``result_r = (exit_price - entry_price) / (entry_price - stop_price)`` (sec. 39.3);
  ``None`` when ``entry_price <= stop_price`` (gap below the stop at entry), and such
  trades are left out of the R statistics.
* ``return_pct = net_pnl / (entry_price * qty)`` (fraction).
* A trade belongs to the period containing its entry fill (UTC date = session date for
  US regular sessions).
* Period return uses the equity curve: ``ending / starting - 1`` where ``starting`` is
  the last equity sample before the period (or the starting cash) and ``ending`` the
  last sample inside it. Annualised with ``TRADING_DAYS_PER_YEAR`` sessions per year.
* Max drawdown: largest ``(peak - equity) / peak`` over the samples of the period.
* Win: ``net_pnl > 0``; loss: ``net_pnl < 0``. Profit factor = sum of winning
  ``net_pnl`` / ``|sum of losing net_pnl|``; ``None`` when there is no losing trade.
* Exposure: market value of open positions / equity at each equity sample (one sample
  per closed primary bar and one per session close); ``avg_exposure`` is their mean.

Ratios are fractions (``0.05`` = 5 %), quantized for stable, readable output.
Backtest results do not imply future profitability (sec. 45.4).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from app.config import BacktestSection
from domain.models import ExitReason, UtcDatetime

__all__ = [
    "DISCLAIMER",
    "TRADING_DAYS_PER_YEAR",
    "BacktestReport",
    "ContinuationCheck",
    "ContinuationStatus",
    "Criterion",
    "EquityPoint",
    "PeriodMetrics",
    "RunCounters",
    "SlippageScenario",
    "TradeRecord",
    "WalkForwardWindow",
    "compute_metrics",
    "continuation_check",
    "render_text",
]

TRADING_DAYS_PER_YEAR: Final = 252
"""Annualisation convention (sessions per year); not a trading parameter."""

DISCLAIMER: Final = (
    "Backtest results do not imply future profitability (sec. 45.4). "
    "Only the deterministic strategy is evaluated; the AI filter is not (sec. 45.1)."
)

_RATIO = Decimal("0.000001")
_MONEY = Decimal("0.01")

ContinuationStatus = Literal["PASS", "FAIL", "PENDING_OWNER_DECISION"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TradeRecord(_Model):
    """One closed simulated trade (sec. 39)."""

    trade_id: str
    symbol: str
    qty: int
    entry_ref: Decimal
    stop_price: Decimal
    take_profit_price: Decimal
    entry_filled_at_utc: UtcDatetime
    entry_price: Decimal
    exit_filled_at_utc: UtcDatetime
    exit_price: Decimal
    exit_reason: ExitReason
    gross_pnl: Decimal
    commissions: Decimal
    net_pnl: Decimal
    return_pct: Decimal
    result_r: Decimal | None
    duration_minutes: Decimal
    entry_slippage: Decimal
    """``entry_price - entry_ref`` (measured entry slippage per share)."""


class EquityPoint(_Model):
    """Equity sample (after a closed primary bar or at a session close)."""

    timestamp_utc: UtcDatetime
    equity: Decimal
    exposure: Decimal


class PeriodMetrics(_Model):
    """The sec. 45.3 results of one period."""

    label: str
    start_date: date
    end_date: date
    sessions: int
    starting_equity: Decimal
    ending_equity: Decimal
    total_return: Decimal
    annualized_return: Decimal | None
    max_drawdown: Decimal
    trades: int
    wins: int
    losses: int
    win_rate: Decimal | None
    profit_factor: Decimal | None
    avg_win_r: Decimal | None
    avg_loss_r: Decimal | None
    avg_win_pct: Decimal | None
    avg_loss_pct: Decimal | None
    expectancy_r: Decimal | None
    avg_duration_minutes: Decimal | None
    avg_exposure: Decimal | None


class WalkForwardWindow(_Model):
    """One walk-forward step: the train window (reported only) and its test results."""

    index: int
    train_start: date
    train_end: date
    test: PeriodMetrics


class SlippageScenario(_Model):
    """Full-period results with the simulated slippage scaled by ``multiplier``."""

    multiplier: Decimal
    slippage_bps: Decimal
    metrics: PeriodMetrics


class Criterion(_Model):
    """One sec. 45.4 criterion; ``passed`` is ``None`` while its threshold is pending."""

    name: str
    comparison: Literal[">=", "<="]
    threshold: Decimal | None
    value: Decimal | None
    passed: bool | None


class ContinuationCheck(_Model):
    """Sec. 45.4 continuation check, evaluated on the out-of-sample period."""

    status: ContinuationStatus
    basis: str
    criteria: tuple[Criterion, ...]
    pending: tuple[str, ...]


class RunCounters(_Model):
    """Event counters of one simulation (audit of what happened besides trades)."""

    minute_bars: int
    closed_bars: int
    signals: int
    entries_submitted: int
    entries_filled: int
    entries_canceled: int
    entries_expired: int
    rejections: dict[str, int]
    broker_rejections: dict[str, int]
    exits_by_reason: dict[str, int]
    deferred_exits: int
    sessions_with_position_after_close: int
    legs_expired_with_position: int
    open_trades_at_end: int


class BacktestReport(_Model):
    """Complete backtest report (JSON via ``model_dump_json``)."""

    disclaimer: str
    config_version: str
    strategy_version: str
    risk_version: str
    config_hash: str
    data_fingerprint: str
    symbols: tuple[str, ...]
    holding_mode: str
    primary_timeframe: str
    start_date: date
    end_date: date
    out_of_sample_start: date
    starting_cash: Decimal
    commission_per_fill: Decimal
    slippage_bps: Decimal
    full: PeriodMetrics
    in_sample: PeriodMetrics
    out_of_sample: PeriodMetrics
    walk_forward: tuple[WalkForwardWindow, ...]
    slippage_sensitivity: tuple[SlippageScenario, ...]
    continuation: ContinuationCheck
    counters: RunCounters
    trades: tuple[TradeRecord, ...]
    notes: tuple[str, ...]


# --------------------------------------------------------------------------- metrics


def _q(value: Decimal, step: Decimal = _RATIO) -> Decimal:
    return value.quantize(step)


def _mean(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    return _q(sum(values, Decimal(0)) / len(values))


def _in_period(day: date, start: date, end: date) -> bool:
    return start <= day <= end


def compute_metrics(
    label: str,
    *,
    start: date,
    end: date,
    sessions: int,
    trades: Sequence[TradeRecord],
    curve: Sequence[EquityPoint],
    starting_cash: Decimal,
) -> PeriodMetrics:
    """Compute the sec. 45.3 metrics of the period ``[start, end]`` (inclusive dates)."""
    period_trades = [t for t in trades if _in_period(t.entry_filled_at_utc.date(), start, end)]
    before = [p for p in curve if p.timestamp_utc.date() < start]
    inside = [p for p in curve if _in_period(p.timestamp_utc.date(), start, end)]
    starting = before[-1].equity if before else starting_cash
    ending = inside[-1].equity if inside else starting
    total_return = ending / starting - 1 if starting > 0 else Decimal(0)
    annualized: Decimal | None = None
    if sessions > 0 and starting > 0 and ending > 0:
        growth = float(ending / starting)
        try:
            raw = growth ** (TRADING_DAYS_PER_YEAR / sessions) - 1
        except OverflowError:
            raw = math.inf
        annualized = Decimal(repr(round(raw, 6))) if math.isfinite(raw) else None

    peak = starting
    max_dd = Decimal(0)
    for point in inside:
        peak = max(peak, point.equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - point.equity) / peak)

    wins = [t for t in period_trades if t.net_pnl > 0]
    losses = [t for t in period_trades if t.net_pnl < 0]
    gross_win = sum((t.net_pnl for t in wins), Decimal(0))
    gross_loss = -sum((t.net_pnl for t in losses), Decimal(0))
    r_values = [t.result_r for t in period_trades if t.result_r is not None]
    win_r = [t.result_r for t in wins if t.result_r is not None]
    loss_r = [t.result_r for t in losses if t.result_r is not None]
    count = len(period_trades)
    return PeriodMetrics(
        label=label,
        start_date=start,
        end_date=end,
        sessions=sessions,
        starting_equity=_q(starting, _MONEY),
        ending_equity=_q(ending, _MONEY),
        total_return=_q(total_return),
        annualized_return=annualized,
        max_drawdown=_q(max_dd),
        trades=count,
        wins=len(wins),
        losses=len(losses),
        win_rate=_q(Decimal(len(wins)) / count) if count else None,
        profit_factor=_q(gross_win / gross_loss) if gross_loss > 0 else None,
        avg_win_r=_mean(win_r),
        avg_loss_r=_mean(loss_r),
        avg_win_pct=_mean([t.return_pct for t in wins]),
        avg_loss_pct=_mean([t.return_pct for t in losses]),
        expectancy_r=_mean(r_values),
        avg_duration_minutes=_mean([t.duration_minutes for t in period_trades]),
        avg_exposure=_mean([p.exposure for p in inside]),
    )


# --------------------------------------------------------------------------- 45.4


def continuation_check(thresholds: BacktestSection, oos: PeriodMetrics) -> ContinuationCheck:
    """Compare out-of-sample results with the owner thresholds (sec. 45.4).

    ``PENDING_OWNER_DECISION`` if any threshold is ``null`` (the criteria that can be
    evaluated are still shown); otherwise ``PASS`` only if every criterion passes. A
    profit factor without losing trades counts as infinite (passes) when there are
    trades; a missing value (no trades) fails.
    """
    no_losses = oos.trades > 0 and oos.losses == 0
    rows: list[tuple[str, Literal[">=", "<="], Decimal | None, Decimal | None, str]] = [
        (
            "expectancy_r",
            ">=",
            thresholds.min_expectancy_r,
            oos.expectancy_r,
            "backtest.min_expectancy_r",
        ),
        (
            "profit_factor",
            ">=",
            thresholds.min_profit_factor,
            oos.profit_factor,
            "backtest.min_profit_factor",
        ),
        (
            "max_drawdown",
            "<=",
            thresholds.max_drawdown_pct,
            oos.max_drawdown,
            "backtest.max_drawdown_pct",
        ),
        (
            "trades_out_of_sample",
            ">=",
            None
            if thresholds.min_trades_out_of_sample is None
            else Decimal(thresholds.min_trades_out_of_sample),
            Decimal(oos.trades),
            "backtest.min_trades_out_of_sample",
        ),
    ]
    criteria: list[Criterion] = []
    pending: list[str] = []
    for name, comparison, threshold, value, path in rows:
        passed: bool | None
        if threshold is None:
            passed = None
            pending.append(path)
        elif value is None:
            passed = name == "profit_factor" and no_losses
        elif comparison == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        criteria.append(
            Criterion(
                name=name, comparison=comparison, threshold=threshold, value=value, passed=passed
            )
        )
    status: ContinuationStatus
    if pending:
        status = "PENDING_OWNER_DECISION"
    elif all(c.passed for c in criteria):
        status = "PASS"
    else:
        status = "FAIL"
    return ContinuationCheck(
        status=status, basis="out_of_sample", criteria=tuple(criteria), pending=tuple(pending)
    )


# --------------------------------------------------------------------------- text


def _fmt(value: Decimal | int | None, *, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{Decimal(value) * 100:.2f}%"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _period_lines(metrics: PeriodMetrics) -> list[str]:
    return [
        f"[{metrics.label}] {metrics.start_date} .. {metrics.end_date} "
        f"({metrics.sessions} sessions)",
        f"  return {_fmt(metrics.total_return, pct=True)} "
        f"(annualised {_fmt(metrics.annualized_return, pct=True)}), "
        f"max drawdown {_fmt(metrics.max_drawdown, pct=True)}, "
        f"equity {metrics.starting_equity} -> {metrics.ending_equity}",
        f"  trades {metrics.trades} (wins {metrics.wins}, losses {metrics.losses}), "
        f"win rate {_fmt(metrics.win_rate, pct=True)}, "
        f"profit factor {_fmt(metrics.profit_factor)}",
        f"  avg win {_fmt(metrics.avg_win_r)} R / {_fmt(metrics.avg_win_pct, pct=True)}, "
        f"avg loss {_fmt(metrics.avg_loss_r)} R / {_fmt(metrics.avg_loss_pct, pct=True)}, "
        f"expectancy {_fmt(metrics.expectancy_r)} R",
        f"  avg duration {_fmt(metrics.avg_duration_minutes)} min, "
        f"avg exposure {_fmt(metrics.avg_exposure, pct=True)}",
    ]


def render_text(report: BacktestReport) -> str:
    """Readable multi-line summary of ``report``."""
    lines = [
        "BACKTEST REPORT (deterministic strategy only)",
        f"strategy {report.strategy_version} / risk {report.risk_version} / "
        f"config {report.config_version} (sha256 {report.config_hash[:12]})",
        f"data {report.data_fingerprint[:12]}  symbols {', '.join(report.symbols)}  "
        f"{report.holding_mode} {report.primary_timeframe}",
        f"cash {report.starting_cash}  slippage {report.slippage_bps} bps  "
        f"commission/fill {report.commission_per_fill}",
        "",
        *_period_lines(report.full),
        *_period_lines(report.in_sample),
        *_period_lines(report.out_of_sample),
        "",
        "Walk-forward (sequential test windows; no parameters are optimised):",
    ]
    for window in report.walk_forward:
        test = window.test
        lines.append(
            f"  #{window.index} train {window.train_start}..{window.train_end} "
            f"test {test.start_date}..{test.end_date}: trades {test.trades}, "
            f"return {_fmt(test.total_return, pct=True)}, "
            f"expectancy {_fmt(test.expectancy_r)} R, PF {_fmt(test.profit_factor)}, "
            f"max DD {_fmt(test.max_drawdown, pct=True)}"
        )
    if not report.walk_forward:
        lines.append("  (no complete test window in the period)")
    lines.append("")
    lines.append("Slippage sensitivity (full period):")
    for scenario in report.slippage_sensitivity:
        m = scenario.metrics
        lines.append(
            f"  x{scenario.multiplier} ({scenario.slippage_bps} bps): trades {m.trades}, "
            f"return {_fmt(m.total_return, pct=True)}, expectancy {_fmt(m.expectancy_r)} R, "
            f"PF {_fmt(m.profit_factor)}, max DD {_fmt(m.max_drawdown, pct=True)}"
        )
    c = report.counters
    lines += [
        "",
        f"Signals {c.signals}, entries submitted {c.entries_submitted}, filled "
        f"{c.entries_filled}, canceled {c.entries_canceled}, expired {c.entries_expired}",
        f"Rejections {dict(sorted(c.rejections.items()))}; broker "
        f"{dict(sorted(c.broker_rejections.items()))}",
        f"Exits {dict(sorted(c.exits_by_reason.items()))}; deferred {c.deferred_exits}; "
        f"sessions with a position after close {c.sessions_with_position_after_close}",
        "",
        f"Continuation check (sec. 45.4, {report.continuation.basis}): "
        f"{report.continuation.status}",
    ]
    for criterion in report.continuation.criteria:
        verdict = "pending" if criterion.passed is None else ("ok" if criterion.passed else "FAIL")
        lines.append(
            f"  {criterion.name} {criterion.comparison} {_fmt(criterion.threshold)}: "
            f"value {_fmt(criterion.value)} -> {verdict}"
        )
    if report.continuation.pending:
        lines.append(f"  pending OWNER_DECISIONs: {', '.join(report.continuation.pending)}")
    lines += ["", *report.notes, report.disclaimer]
    return "\n".join(lines)
