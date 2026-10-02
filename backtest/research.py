"""RESEARCH mode: segmented backtest, one fresh account per segment.

This is NOT the official sec. 45.4 run (:func:`backtest.runner.run_backtest` is). In
the official run a risk halt is permanent (e.g. ``MAX_DRAWDOWN``: the peak never falls),
so after an early halt the rest of the period produces no trade and the later windows
are empty. The research mode measures the strategy in every market regime instead:

* Every segment is simulated by :func:`backtest.runner.simulate` with
  ``window = (start, end)``: an independent account (starting cash, empty book, fresh
  risk state: peak equity, week start, halts) whose indicators are warmed up on the
  stored sessions before ``start`` only, exactly as the official run warms up before
  ``start_date``. No simulation logic lives here.
* Segments: each walk-forward test window, ``in_sample`` ``[start_date,
  out_of_sample_start)``, ``out_of_sample`` ``[out_of_sample_start, end_date]`` and one
  segment per calendar year (clipped to the period). Slippage sensitivity (x1.5, x2.0)
  is simulated for the out-of-sample segment and for every calendar-year segment.
* Aggregates pool the trades of one family of segments (walk-forward test windows;
  calendar years per slippage multiplier). Families overlap in time, so they are never
  pooled together.
* The sec. 45.4 criteria are evaluated on the fresh-account out-of-sample segment and
  labelled as a research reading: the official decision requires the official run.

All segment simulations are independent and may run in parallel (``workers``); results
are merged in the fixed segment order, so the report does not depend on ``workers``.

:func:`run_config_jobs` runs simulations of SEVERAL configurations in one process pool
(the research CLI of family v2, :mod:`backtest.research_cli`); like
:func:`backtest.runner.run_simulations` it returns the results in job order.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from app.config import AppConfig, LoadedConfig, pending_backtest_decisions, risk_params
from backtest.data import CALENDAR_FILE, BacktestData, load_backtest_data, load_calendar
from backtest.report import (
    DISCLAIMER,
    ContinuationCheck,
    EquityPoint,
    PeriodMetrics,
    RunCounters,
    TradeRecord,
    compute_metrics,
    continuation_check,
    format_value,
    period_lines,
)
from backtest.runner import (
    SLIPPAGE_MULTIPLIERS,
    BacktestRefusedError,
    SimulationJob,
    SimulationResult,
    data_fingerprint,
    run_simulations,
    simulate,
    walk_forward_windows,
)
from domain.errors import NonRetryableError
from domain.models import DataFeed, ExitReason
from domain.risk.risk_engine import DAILY_LOSS_LIMIT, MAX_DRAWDOWN, WEEKLY_LOSS_LIMIT

__all__ = [
    "CONTINUATION_LABEL",
    "HALT_CODES",
    "RESEARCH_BANNER",
    "ConfigJob",
    "ResearchReport",
    "Segment",
    "SegmentAggregate",
    "SegmentKind",
    "SegmentReport",
    "render_research_text",
    "research_segments",
    "run_config_jobs",
    "run_research",
    "run_research_async",
]

RESEARCH_BANNER: Final = (
    "RESEARCH MODE (segmented, fresh account per segment) — NOT the official sec. 45.4 "
    "run; halts reset per segment"
)
CONTINUATION_LABEL: Final = "research reading — official decision requires the official run"
HALT_CODES: Final[tuple[str, ...]] = (DAILY_LOSS_LIMIT, WEEKLY_LOSS_LIMIT, MAX_DRAWDOWN)
"""Risk rejections that halt new entries (loss limits and max drawdown)."""

SegmentKind = Literal["walk_forward", "in_sample", "out_of_sample", "year"]

_BASE: Final = Decimal(1)
_RATIO = Decimal("0.000001")
_MONEY = Decimal("0.01")


# --------------------------------------------------------------------------- segments


@dataclass(frozen=True, slots=True)
class Segment:
    """One independent fresh-account simulation of the research mode."""

    label: str
    kind: SegmentKind
    start: date
    end: date
    slippage_multiplier: Decimal = _BASE
    train_start: date | None = None
    train_end: date | None = None


def _suffix(multiplier: Decimal) -> str:
    return "" if multiplier == _BASE else f"_x{multiplier}"


def research_segments(
    start: date,
    end: date,
    out_of_sample_start: date,
    *,
    train_months: int,
    test_months: int,
) -> tuple[Segment, ...]:
    """Segments in report order: walk-forward tests, in/out-of-sample, years, sensitivity."""
    if not start <= out_of_sample_start <= end:
        raise ValueError("out_of_sample_start must lie in [start, end]")
    segments = [
        Segment(
            label=f"walk_forward_{index}",
            kind="walk_forward",
            start=test_start,
            end=test_end,
            train_start=train_start,
            train_end=train_end,
        )
        for index, (train_start, train_end, test_start, test_end) in enumerate(
            walk_forward_windows(start, end, train_months=train_months, test_months=test_months)
        )
    ]
    if out_of_sample_start > start:
        segments.append(
            Segment("in_sample", "in_sample", start, out_of_sample_start - timedelta(days=1))
        )
    years = [
        (year, max(start, date(year, 1, 1)), min(end, date(year, 12, 31)))
        for year in range(start.year, end.year + 1)
    ]
    for multiplier in (_BASE, *SLIPPAGE_MULTIPLIERS):
        suffix = _suffix(multiplier)
        segments.append(
            Segment(
                f"out_of_sample{suffix}",
                "out_of_sample",
                out_of_sample_start,
                end,
                slippage_multiplier=multiplier,
            )
        )
        segments.extend(
            Segment(f"year_{year}{suffix}", "year", first, last, slippage_multiplier=multiplier)
            for year, first, last in years
        )
    base = [s for s in segments if s.slippage_multiplier == _BASE]
    sensitivity = [s for s in segments if s.slippage_multiplier != _BASE]
    return (*base, *sensitivity)


# --------------------------------------------------------------------------- report models


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SegmentReport(_Model):
    """Results of one fresh-account segment."""

    label: str
    kind: SegmentKind
    slippage_multiplier: Decimal
    slippage_bps: Decimal
    train_start: date | None
    train_end: date | None
    metrics: PeriodMetrics
    halt_rejections: dict[str, int]
    """Rejected signals per halt code (:data:`HALT_CODES`; zero counts included)."""
    drawdown_limit_reached_on: date | None
    """First equity sample at or beyond ``risk.max_drawdown_pct`` from the peak."""
    exits_by_reason: dict[str, int]
    take_profit_rate: Decimal | None
    """``TAKE_PROFIT`` exits / trades."""
    counters: RunCounters
    trades: tuple[TradeRecord, ...]


class SegmentAggregate(_Model):
    """Pooled results over one family of segments (trades pooled, not averaged)."""

    label: str
    slippage_multiplier: Decimal
    segments: tuple[str, ...]
    trades: int
    wins: int
    losses: int
    win_rate: Decimal | None
    profit_factor: Decimal | None
    expectancy_r: Decimal | None
    take_profit_rate: Decimal | None
    net_pnl: Decimal
    positive_expectancy_segments: int
    positive_expectancy_share: Decimal | None
    """Segments with ``expectancy_r > 0`` / all segments (no trade = not positive)."""
    halted_segments: int
    """Segments with at least one halt rejection."""
    exits_by_reason: dict[str, int]
    halt_rejections: dict[str, int]


class ResearchReport(_Model):
    """Segmented research report (JSON via ``model_dump_json``)."""

    mode: Literal["segmented"]
    banner: str
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
    segments: tuple[SegmentReport, ...]
    aggregates: tuple[SegmentAggregate, ...]
    continuation: ContinuationCheck
    continuation_label: str
    notes: tuple[str, ...]


# --------------------------------------------------------------------------- building


def _drawdown_limit_date(
    curve: Sequence[EquityPoint], starting_cash: Decimal, limit: Decimal | None
) -> date | None:
    if limit is None:
        return None
    peak = starting_cash
    for point in curve:
        peak = max(peak, point.equity)
        if peak > 0 and (peak - point.equity) / peak >= limit:
            return point.timestamp_utc.date()
    return None


def _rate(part: int, whole: int) -> Decimal | None:
    return (Decimal(part) / whole).quantize(_RATIO) if whole else None


def _exits(trades: Sequence[TradeRecord]) -> dict[str, int]:
    return dict(sorted(Counter(t.exit_reason.value for t in trades).items()))


def _segment_report(
    segment: Segment,
    result: SimulationResult,
    *,
    starting_cash: Decimal,
    drawdown_limit: Decimal | None,
) -> SegmentReport:
    metrics = compute_metrics(
        segment.label,
        start=segment.start,
        end=segment.end,
        sessions=len(result.sessions),
        trades=result.trades,
        curve=result.equity_curve,
        starting_cash=starting_cash,
    )
    take_profits = sum(1 for t in result.trades if t.exit_reason is ExitReason.TAKE_PROFIT)
    return SegmentReport(
        label=segment.label,
        kind=segment.kind,
        slippage_multiplier=segment.slippage_multiplier,
        slippage_bps=result.slippage_bps,
        train_start=segment.train_start,
        train_end=segment.train_end,
        metrics=metrics,
        halt_rejections={code: result.counters.rejections.get(code, 0) for code in HALT_CODES},
        drawdown_limit_reached_on=_drawdown_limit_date(
            result.equity_curve, starting_cash, drawdown_limit
        ),
        exits_by_reason=_exits(result.trades),
        take_profit_rate=_rate(take_profits, len(result.trades)),
        counters=result.counters,
        trades=result.trades,
    )


def _aggregate(
    label: str, multiplier: Decimal, segments: Sequence[SegmentReport]
) -> SegmentAggregate:
    trades = [t for s in segments for t in s.trades]
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    gross_win = sum((t.net_pnl for t in wins), Decimal(0))
    gross_loss = -sum((t.net_pnl for t in losses), Decimal(0))
    r_values = [t.result_r for t in trades if t.result_r is not None]
    positive = sum(
        1 for s in segments if s.metrics.expectancy_r is not None and s.metrics.expectancy_r > 0
    )
    halts: Counter[str] = Counter()
    for segment in segments:
        halts.update(segment.halt_rejections)
    take_profits = sum(1 for t in trades if t.exit_reason is ExitReason.TAKE_PROFIT)
    return SegmentAggregate(
        label=label,
        slippage_multiplier=multiplier,
        segments=tuple(s.label for s in segments),
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=_rate(len(wins), len(trades)),
        profit_factor=(gross_win / gross_loss).quantize(_RATIO) if gross_loss > 0 else None,
        expectancy_r=(
            (sum(r_values, Decimal(0)) / len(r_values)).quantize(_RATIO) if r_values else None
        ),
        take_profit_rate=_rate(take_profits, len(trades)),
        net_pnl=sum((t.net_pnl for t in trades), Decimal(0)).quantize(_MONEY),
        positive_expectancy_segments=positive,
        positive_expectancy_share=_rate(positive, len(segments)),
        halted_segments=sum(1 for s in segments if any(s.halt_rejections.values())),
        exits_by_reason=_exits(trades),
        halt_rejections={code: halts.get(code, 0) for code in HALT_CODES},
    )


def _aggregates(segments: Sequence[SegmentReport]) -> tuple[SegmentAggregate, ...]:
    walk_forward = [s for s in segments if s.kind == "walk_forward"]
    result = [_aggregate("walk_forward_tests", _BASE, walk_forward)] if walk_forward else []
    for multiplier in (_BASE, *SLIPPAGE_MULTIPLIERS):
        years = [s for s in segments if s.kind == "year" and s.slippage_multiplier == multiplier]
        if years:
            result.append(_aggregate(f"years{_suffix(multiplier)}", multiplier, years))
    return tuple(result)


_NOTES: Final[tuple[str, ...]] = (
    "Research mode: every segment is an independent simulation (starting cash, empty "
    "book, fresh risk state: peak equity, week start, halts) on the same code path as "
    "the official run; indicators are warmed up on the stored sessions before the "
    "segment start only (no lookahead).",
    "Segment families overlap in time (walk-forward tests, in/out-of-sample, calendar "
    "years); aggregates pool the trades of one family only.",
    "Walk-forward: no optimisable parameters exist yet, so each test window is a fresh "
    "account evaluated on its own; train windows are informative.",
    "Slippage sensitivity scales only the simulated fills (x"
    + ", x".join(map(str, SLIPPAGE_MULTIPLIERS))
    + "); sizing keeps risk.slippage_buffer_bps.",
    "drawdown_limit_reached_on is the first equity sample at or beyond "
    "risk.max_drawdown_pct from the segment peak (after it the segment takes no entry).",
    "Phase 1 runner: the shared check library (Phase 4) and the execution guard / "
    "state machine (Phase 5) are not wired yet (AC-22 pending).",
)


async def run_research_async(
    loaded: LoadedConfig,
    data_dir: Path,
    *,
    starting_cash: Decimal,
    commission_per_fill: Decimal = Decimal(0),
    workers: int = 1,
) -> ResearchReport:
    """Segmented research backtest (see module docstring); NOT the official run.

    Args:
        loaded: Validated configuration.
        data_dir: Dataset directory.
        starting_cash: Initial capital of every segment.
        commission_per_fill: Flat commission per fill.
        workers: Processes for the segment simulations (``1`` = all in this process).
            The report does not depend on it.

    Raises:
        BacktestRefusedError: a required OWNER_DECISION is ``null`` (nothing is run).
    """
    config = loaded.config
    pending = pending_backtest_decisions(config)
    if pending:
        raise BacktestRefusedError(pending)
    backtest = config.backtest
    start, end, oos = backtest.start_date, backtest.end_date, backtest.out_of_sample_start
    train, test = backtest.walk_forward_train_months, backtest.walk_forward_test_months
    feed, holding, primary = (
        config.market_data.feed,
        config.strategy.holding_mode,
        config.strategy.primary_timeframe,
    )
    if (
        start is None
        or end is None
        or oos is None
        or train is None
        or test is None
        or feed is None
        or holding is None
        or primary is None
    ):  # pragma: no cover - guarded by pending_backtest_decisions
        raise BacktestRefusedError(pending_backtest_decisions(config))
    if not data_dir.is_dir():
        raise NonRetryableError(f"data directory {data_dir} not found", code="INVALID_BAR_DATA")
    stored = [s.session_date for s in load_calendar(data_dir / CALENDAR_FILE)]
    candidates = research_segments(start, end, oos, train_months=train, test_months=test)
    segments = tuple(s for s in candidates if any(s.start <= d <= s.end for d in stored))
    skipped = tuple(s.label for s in candidates if s not in segments)
    if not any(s.kind == "out_of_sample" for s in segments):
        raise NonRetryableError(f"no stored session between {oos} and {end}", code="NO_SESSIONS")
    # Longest segments first keeps the pool busy; results keep the segment order.
    submit_order = sorted(range(len(segments)), key=lambda i: segments[i].start - segments[i].end)
    results = await run_simulations(
        config,
        data_dir,
        DataFeed(feed),
        starting_cash=starting_cash,
        commission_per_fill=commission_per_fill,
        jobs=[SimulationJob(s.slippage_multiplier, (s.start, s.end)) for s in segments],
        workers=workers,
        submit_order=submit_order,
    )
    drawdown_limit = risk_params(config).max_drawdown_pct
    reports = tuple(
        _segment_report(segment, result, starting_cash=starting_cash, drawdown_limit=drawdown_limit)
        for segment, result in zip(segments, results, strict=True)
    )
    oos_report = next(
        s for s in reports if s.kind == "out_of_sample" and s.slippage_multiplier == _BASE
    )
    continuation = continuation_check(backtest, oos_report.metrics).model_copy(
        update={"basis": "out_of_sample fresh-account segment"}
    )
    return ResearchReport(
        mode="segmented",
        banner=RESEARCH_BANNER,
        disclaimer=DISCLAIMER,
        config_version=config.config_version,
        strategy_version=config.strategy_version,
        risk_version=config.risk_version,
        config_hash=loaded.config_hash,
        data_fingerprint=data_fingerprint(data_dir),
        symbols=tuple(config.universe.whitelist or ()),
        holding_mode=holding.value,
        primary_timeframe=primary.value,
        start_date=start,
        end_date=end,
        out_of_sample_start=oos,
        starting_cash=starting_cash,
        commission_per_fill=commission_per_fill,
        slippage_bps=oos_report.slippage_bps,
        segments=reports,
        aggregates=_aggregates(reports),
        continuation=continuation,
        continuation_label=CONTINUATION_LABEL,
        notes=(
            *_NOTES,
            *(
                (f"Skipped segments without stored sessions: {', '.join(skipped)}.",)
                if skipped
                else ()
            ),
        ),
    )


def run_research(
    loaded: LoadedConfig,
    data_dir: Path,
    *,
    starting_cash: Decimal,
    commission_per_fill: Decimal = Decimal(0),
    workers: int = 1,
) -> ResearchReport:
    """Synchronous wrapper of :func:`run_research_async`."""
    return asyncio.run(
        run_research_async(
            loaded,
            data_dir,
            starting_cash=starting_cash,
            commission_per_fill=commission_per_fill,
            workers=workers,
        )
    )


# --------------------------------------------------------------------------- text


def _halts_text(halts: dict[str, int]) -> str:
    return ", ".join(f"{code} {count}" for code, count in halts.items())


def _segment_lines(segment: SegmentReport) -> list[str]:
    lines = period_lines(segment.metrics)
    if segment.train_start is not None and segment.train_end is not None:
        lines.insert(1, f"  train window (informative) {segment.train_start}..{segment.train_end}")
    reached = segment.drawdown_limit_reached_on
    c = segment.counters
    lines += [
        f"  halt rejections: {_halts_text(segment.halt_rejections)}; drawdown limit reached "
        f"{reached if reached is not None else 'never'}",
        f"  exits {segment.exits_by_reason}; take-profit hit rate "
        f"{format_value(segment.take_profit_rate, pct=True)}",
        f"  signals {c.signals}, entries filled {c.entries_filled}, rejections "
        f"{dict(sorted(c.rejections.items()))}, open at end {c.open_trades_at_end}",
    ]
    return lines


def _segment_line(segment: SegmentReport) -> str:
    m = segment.metrics
    reached = segment.drawdown_limit_reached_on
    return (
        f"  {segment.label} {m.start_date}..{m.end_date} ({segment.slippage_bps} bps): "
        f"trades {m.trades}, return {format_value(m.total_return, pct=True)}, "
        f"expectancy {format_value(m.expectancy_r)} R, PF {format_value(m.profit_factor)}, "
        f"win rate {format_value(m.win_rate, pct=True)}, "
        f"TP rate {format_value(segment.take_profit_rate, pct=True)}, "
        f"max DD {format_value(m.max_drawdown, pct=True)}, "
        f"MAX_DRAWDOWN rejections {segment.halt_rejections.get(MAX_DRAWDOWN, 0)}"
        + (f" (limit reached {reached})" if reached is not None else "")
    )


def _aggregate_lines(aggregate: SegmentAggregate) -> list[str]:
    a = aggregate
    return [
        f"[{a.label}] x{a.slippage_multiplier} over {len(a.segments)} segments: "
        f"trades {a.trades} (wins {a.wins}, losses {a.losses}), "
        f"win rate {format_value(a.win_rate, pct=True)}",
        f"  pooled expectancy {format_value(a.expectancy_r)} R, "
        f"pooled profit factor {format_value(a.profit_factor)}, "
        f"take-profit hit rate {format_value(a.take_profit_rate, pct=True)}, "
        f"net P&L {a.net_pnl}",
        f"  segments with positive expectancy {a.positive_expectancy_segments}/"
        f"{len(a.segments)} ({format_value(a.positive_expectancy_share, pct=True)}), "
        f"halted segments {a.halted_segments}/{len(a.segments)}",
        f"  exits {a.exits_by_reason}; halt rejections {_halts_text(a.halt_rejections)}",
    ]


def render_research_text(report: ResearchReport) -> str:
    """Readable multi-line summary of a research ``report`` (banner first)."""
    base = [s for s in report.segments if s.slippage_multiplier == _BASE]
    sensitivity = [s for s in report.segments if s.slippage_multiplier != _BASE]
    lines = [
        report.banner,
        "",
        "BACKTEST RESEARCH REPORT (deterministic strategy only)",
        f"strategy {report.strategy_version} / risk {report.risk_version} / "
        f"config {report.config_version} (sha256 {report.config_hash[:12]})",
        f"data {report.data_fingerprint[:12]}  symbols {', '.join(report.symbols)}  "
        f"{report.holding_mode} {report.primary_timeframe}",
        f"cash {report.starting_cash} per segment  slippage {report.slippage_bps} bps  "
        f"commission/fill {report.commission_per_fill}",
        f"period {report.start_date}..{report.end_date}, out-of-sample from "
        f"{report.out_of_sample_start}; {len(report.segments)} fresh-account segments",
    ]
    sections: tuple[tuple[str, SegmentKind], ...] = (
        ("Walk-forward test windows (fresh account per window):", "walk_forward"),
        ("In-sample (fresh account):", "in_sample"),
        ("Out-of-sample (fresh account):", "out_of_sample"),
        ("Calendar years (fresh account per year):", "year"),
    )
    for title, kind in sections:
        chosen = [s for s in base if s.kind == kind]
        if not chosen:
            continue
        lines += ["", title]
        for segment in chosen:
            lines += _segment_lines(segment)
    if sensitivity:
        lines += ["", "Slippage sensitivity (fresh accounts):"]
        lines += [_segment_line(s) for s in sensitivity]
    lines += ["", "Aggregates (pooled trades within one segment family):"]
    for aggregate in report.aggregates:
        lines += _aggregate_lines(aggregate)
    check = report.continuation
    lines += [
        "",
        f"Continuation check (sec. 45.4 criteria, {check.basis}) — "
        f"{report.continuation_label}: {check.status}",
    ]
    for criterion in check.criteria:
        verdict = "pending" if criterion.passed is None else ("ok" if criterion.passed else "FAIL")
        lines.append(
            f"  {criterion.name} {criterion.comparison} {format_value(criterion.threshold)}: "
            f"value {format_value(criterion.value)} -> {verdict}"
        )
    if check.pending:
        lines.append(f"  pending OWNER_DECISIONs: {', '.join(check.pending)}")
    lines += ["", *report.notes, report.disclaimer, "", report.banner]
    return "\n".join(lines)


# --------------------------------------------------------------------------- multi-config jobs


@dataclass(frozen=True, slots=True)
class ConfigJob:
    """One independent simulation of ``config`` (research CLI: several configs per pool)."""

    config: AppConfig
    job: SimulationJob


_WORKER_DATA: dict[tuple[Path, DataFeed], BacktestData] = {}
"""Dataset loaded by a worker process, reused for every job the pool gives it."""


def _simulate_config_job(
    data_dir: Path,
    feed: DataFeed,
    starting_cash: Decimal,
    commission_per_fill: Decimal,
    job: ConfigJob,
) -> SimulationResult:
    """Worker-process entry point: load the dataset (once per process) and simulate."""
    key = (data_dir, feed)
    data = _WORKER_DATA.get(key)
    if data is None:
        data = _WORKER_DATA[key] = load_backtest_data(data_dir, feed=feed)
    return asyncio.run(
        simulate(
            job.config,
            data,
            starting_cash=starting_cash,
            commission_per_fill=commission_per_fill,
            slippage_multiplier=job.job.slippage_multiplier,
            window=job.job.window,
        )
    )


async def run_config_jobs(
    jobs: Sequence[ConfigJob],
    data: BacktestData,
    feed: DataFeed,
    *,
    starting_cash: Decimal,
    commission_per_fill: Decimal = Decimal(0),
    workers: int = 1,
    submit_order: Sequence[int] | None = None,
) -> list[SimulationResult]:
    """Run ``jobs`` (each with its own configuration) and return results in job order.

    Args:
        data: Dataset already loaded by the caller; ``workers <= 1`` simulates on it in
            this process, otherwise every worker process loads ``data.directory`` once.
        submit_order: Optional permutation of ``range(len(jobs))`` handed to the pool
            first-to-last (scheduling only: results keep ``jobs`` order, so they depend
            neither on it nor on ``workers``).
    """
    order = list(range(len(jobs))) if submit_order is None else list(submit_order)
    if sorted(order) != list(range(len(jobs))):
        raise ValueError("submit_order must be a permutation of the job indices")
    if workers <= 1 or len(jobs) <= 1:
        return [
            await simulate(
                job.config,
                data,
                starting_cash=starting_cash,
                commission_per_fill=commission_per_fill,
                slippage_multiplier=job.job.slippage_multiplier,
                window=job.job.window,
            )
            for job in jobs
        ]
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {
            index: loop.run_in_executor(
                pool,
                _simulate_config_job,
                data.directory,
                feed,
                starting_cash,
                commission_per_fill,
                jobs[index],
            )
            for index in order
        }
        return list(await asyncio.gather(*(futures[index] for index in range(len(jobs)))))
