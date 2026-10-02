"""Historical data for the backtest: stored bars, stored calendar, synthetic generator.

Data directory layout (one directory per dataset)::

    calendar.csv              stored trading calendar: session_date,open_local,close_local
                              (America/New_York wall times, e.g. 2025-11-28,09:30,13:00)
    bars/{SYMBOL}_1Min.csv    minute bars (t,o,h,l,c,v; t = bar START, UTC ISO 8601 "Z")
    bars/{SYMBOL}_1Day.csv    daily bars, same format (liquidity filters, Phase 4)

The bar format is the one of ``adapters.simulation.historical_feed`` (loaded with
``HistoricalFeed.from_csv_dir``). The calendar is required: the backtest uses a stored
historical calendar (sec. 8.6), never one guessed from the bars.

:func:`generate_synthetic_data` writes a deterministic dataset (seeded random walk on
regular weekday sessions) for tests and demos. It is **not** market data and carries
no information about any real instrument.

Real data: :func:`download_alpaca_bars` downloads historical bars and the trading
calendar from Alpaca into the same layout (declared early pull-forward of part of
Phase 2: historical market data and calendar only, no orders, no streaming). See its
docstring for the resumable chunk cache and the regular-session filter.

CLI::

    python -m backtest.data download --symbols SPY QQQ --start 2025-06-02
        --end 2025-06-03 --feed iex --adjustment split --daily-warmup-days 20
        --out data/alpaca [--env-file .env]
    python -m backtest.data synthetic --out DIR --symbols SYNTH --start ... --end ... --seed N

The legacy invocation without a subcommand (``python -m backtest.data --out ...``) still
runs the synthetic generator.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from zoneinfo import ZoneInfo

from adapters.simulation import HistoricalFeed, build_regular_sessions
from adapters.simulation.static_calendar import MARKET_TIMEZONE, REGULAR_CLOSE, REGULAR_OPEN
from domain.errors import DomainError, NonRetryableError
from domain.models import Bar, DataFeed, SessionDay
from domain.ports import IMarketData

if TYPE_CHECKING:
    # Imported lazily at runtime: alpaca-py (and pandas) is only needed to download.
    from adapters.alpaca import BarAdjustment

__all__ = [
    "BARS_DIR",
    "CALENDAR_COLUMNS",
    "CALENDAR_FILE",
    "DOWNLOAD_CACHE_DIR",
    "BacktestData",
    "DownloadResult",
    "DownloadSpec",
    "SessionSource",
    "SyntheticSpec",
    "business_days",
    "download_alpaca_bars",
    "generate_synthetic_data",
    "load_backtest_data",
    "load_calendar",
    "main",
    "write_calendar",
]

CALENDAR_FILE = "calendar.csv"
BARS_DIR = "bars"
CALENDAR_COLUMNS = ("session_date", "open_local", "close_local")
DOWNLOAD_CACHE_DIR = ".download"
"""Resumable chunk cache of :func:`download_alpaca_bars` (outside ``bars/``)."""
_MANIFEST_FILE = "manifest.json"
_MANIFEST_VERSION = 1
_BAR_HEADER = "t,o,h,l,c,v"
_CENT = Decimal("0.01")
_MINUTES_PER_DAY_CAP = 24 * 60


@dataclass(frozen=True, slots=True)
class BacktestData:
    """A loaded dataset.

    Attributes:
        feed: ``IMarketData`` over the stored bars (no clock: the runner pulls one
            session at a time and processes it strictly in time order).
        sessions: Stored trading calendar, ordered by date.
        directory: Source directory.
    """

    feed: HistoricalFeed
    sessions: tuple[SessionDay, ...]
    directory: Path


def _parse_hhmm(text: str, *, where: str) -> time:
    try:
        parsed = datetime.strptime(text.strip(), "%H:%M")  # noqa: DTZ007 - wall time only
    except ValueError as exc:
        raise NonRetryableError(f"{where}: invalid time {text!r}", code="INVALID_CALENDAR") from exc
    return parsed.time()


def load_calendar(path: Path) -> tuple[SessionDay, ...]:
    """Load ``calendar.csv`` into UTC sessions (``build_regular_sessions``).

    Every row must use the same open time; a close different from the regular close
    is an early close.

    Raises:
        NonRetryableError: code ``INVALID_CALENDAR`` (missing file, bad header or row).
    """
    if not path.is_file():
        raise NonRetryableError(
            f"missing stored calendar {path} (required, sec. 8.6)", code="INVALID_CALENDAR"
        )
    days: list[date] = []
    opens: set[time] = set()
    closes: dict[date, time] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CALENDAR_COLUMNS:
            raise NonRetryableError(
                f"{path.name}: header must be {','.join(CALENDAR_COLUMNS)}",
                code="INVALID_CALENDAR",
            )
        for line, row in enumerate(reader, start=2):
            where = f"{path.name}:{line}"
            try:
                day = date.fromisoformat(row["session_date"].strip())
            except ValueError as exc:
                raise NonRetryableError(f"{where}: invalid date", code="INVALID_CALENDAR") from exc
            days.append(day)
            opens.add(_parse_hhmm(row["open_local"], where=where))
            close = _parse_hhmm(row["close_local"], where=where)
            if close != REGULAR_CLOSE:
                closes[day] = close
    if len(opens) > 1:
        raise NonRetryableError(
            f"{path.name}: sessions with different open times are not supported",
            code="INVALID_CALENDAR",
        )
    if len(set(days)) != len(days):
        raise NonRetryableError(f"{path.name}: duplicate session dates", code="INVALID_CALENDAR")
    open_time = opens.pop() if opens else REGULAR_OPEN
    sessions = build_regular_sessions(days, early_closes=closes, open_time=open_time)
    return tuple(sessions.values())


def load_backtest_data(directory: Path, *, feed: DataFeed) -> BacktestData:
    """Load the bars (``HistoricalFeed.from_csv_dir``) and the calendar of ``directory``.

    Raises:
        NonRetryableError: ``INVALID_BAR_DATA``, ``DUPLICATE_BAR`` or ``INVALID_CALENDAR``.
    """
    if not directory.is_dir():
        raise NonRetryableError(f"data directory {directory} not found", code="INVALID_BAR_DATA")
    sessions = load_calendar(directory / CALENDAR_FILE)
    bars_dir = directory / BARS_DIR
    if not bars_dir.is_dir() or not any(bars_dir.glob("*.csv")):
        raise NonRetryableError(f"no bar files in {bars_dir}", code="INVALID_BAR_DATA")
    return BacktestData(
        feed=HistoricalFeed.from_csv_dir(bars_dir, feed=feed),
        sessions=sessions,
        directory=directory,
    )


# --------------------------------------------------------------------------- synthetic data


def business_days(start: date, end: date) -> list[date]:
    """Weekdays in ``[start, end]`` (no holiday calendar: synthetic data only)."""
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """Parameters of the synthetic random walk (fixture values, not market estimates).

    Attributes:
        symbols: Symbols to generate (independent walks, one seed stream each).
        start: First calendar day.
        end: Last calendar day.
        seed: Random seed (same seed -> byte-identical files).
        start_price: First open price.
        drift_bps_per_minute: Mean log return per minute, in basis points.
        volatility_bps_per_minute: Standard deviation of the minute log return, in bps.
        overnight_gap_bps: Standard deviation of the overnight gap, in bps.
        base_volume: Mean minute volume.
        missing_minute_rate: Probability of dropping a minute bar (IEX-like gaps).
        early_closes: Early-close wall times by date.
    """

    symbols: tuple[str, ...]
    start: date
    end: date
    seed: int
    start_price: Decimal = Decimal("100")
    drift_bps_per_minute: float = 0.0
    volatility_bps_per_minute: float = 8.0
    overnight_gap_bps: float = 30.0
    base_volume: int = 1000
    missing_minute_rate: float = 0.0
    early_closes: Mapping[date, time] | None = None


def _cents(value: float) -> Decimal:
    return Decimal(repr(value)).quantize(_CENT, rounding=ROUND_HALF_EVEN)


_Row = tuple[datetime, Decimal, Decimal, Decimal, Decimal, int]
"""``(start, open, high, low, close, volume)``."""


def _write_bars(path: Path, rows: Sequence[_Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("t", "o", "h", "l", "c", "v"))
        for start, o, h, low, c, v in rows:
            writer.writerow((start.strftime("%Y-%m-%dT%H:%M:%SZ"), o, h, low, c, v))


def _write_calendar_rows(path: Path, rows: Iterable[tuple[date, time, time]]) -> None:
    """Write ``calendar.csv`` from ``(session_date, open_local, close_local)`` rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CALENDAR_COLUMNS)
        for day, open_local, close_local in rows:
            writer.writerow(
                (day.isoformat(), open_local.strftime("%H:%M"), close_local.strftime("%H:%M"))
            )


def write_calendar(path: Path, sessions: Iterable[SessionDay]) -> None:
    """Write UTC ``sessions`` as ``calendar.csv`` (America/New_York wall times), by date.

    The inverse of :func:`load_calendar` for sessions in the market time zone.
    """
    zone = ZoneInfo(MARKET_TIMEZONE)
    ordered = sorted(sessions, key=lambda s: s.session_date)
    _write_calendar_rows(
        path,
        (
            (
                s.session_date,
                s.open_utc.astimezone(zone).time(),
                s.close_utc.astimezone(zone).time(),
            )
            for s in ordered
        ),
    )


def generate_synthetic_data(directory: Path, spec: SyntheticSpec) -> tuple[SessionDay, ...]:
    """Write a deterministic synthetic dataset (bars + calendar) into ``directory``.

    Each symbol follows a seeded geometric random walk minute by minute during regular
    sessions (09:30-16:00 America/New_York, early closes honored), with an overnight
    gap at each open. High/low extend the open/close range by a random fraction of the
    volatility. Prices are rounded to cents. Daily bars aggregate the minute bars.

    Returns:
        The generated sessions.
    """
    directory.mkdir(parents=True, exist_ok=True)
    days = business_days(spec.start, spec.end)
    early = dict(spec.early_closes or {})
    sessions = tuple(build_regular_sessions(days, early_closes=early).values())
    _write_calendar_rows(
        directory / CALENDAR_FILE,
        (
            (s.session_date, REGULAR_OPEN, early.get(s.session_date, REGULAR_CLOSE))
            for s in sessions
        ),
    )
    vol = spec.volatility_bps_per_minute / 10_000
    drift = spec.drift_bps_per_minute / 10_000
    gap = spec.overnight_gap_bps / 10_000
    for index, symbol in enumerate(spec.symbols):
        rng = random.Random(f"{spec.seed}:{index}:{symbol}")
        price = float(spec.start_price)
        minute_rows: list[_Row] = []
        daily_rows: list[_Row] = []
        for session_index, session in enumerate(sessions):
            if session_index:
                price *= math.exp(rng.gauss(0.0, gap))
            minutes = int((session.close_utc - session.open_utc) / timedelta(minutes=1))
            day_rows: list[_Row] = []
            for minute in range(min(minutes, _MINUTES_PER_DAY_CAP)):
                open_ = price
                close = open_ * math.exp(drift + rng.gauss(0.0, vol))
                high = max(open_, close) * (1 + abs(rng.gauss(0.0, vol / 2)))
                low = min(open_, close) * (1 - abs(rng.gauss(0.0, vol / 2)))
                volume = max(1, int(rng.expovariate(1.0) * spec.base_volume))
                price = close
                if spec.missing_minute_rate and rng.random() < spec.missing_minute_rate:
                    continue
                o, c = _cents(open_), _cents(close)
                h = max(_cents(high), o, c)
                low_c = min(_cents(low), o, c)
                if low_c <= 0:
                    raise NonRetryableError(
                        "synthetic walk reached a non-positive price; lower the volatility",
                        code="INVALID_SYNTHETIC_SPEC",
                    )
                start = session.open_utc + timedelta(minutes=minute)
                day_rows.append((start, o, h, low_c, c, volume))
            minute_rows.extend(day_rows)
            if day_rows:
                daily_rows.append(
                    (
                        datetime.combine(session.session_date, time(0, 0), tzinfo=UTC),
                        day_rows[0][1],
                        max(r[2] for r in day_rows),
                        min(r[3] for r in day_rows),
                        day_rows[-1][4],
                        sum(r[5] for r in day_rows),
                    )
                )
        bars_dir = directory / BARS_DIR
        bars_dir.mkdir(exist_ok=True)
        _write_bars(bars_dir / f"{symbol}_1Min.csv", minute_rows)
        _write_bars(bars_dir / f"{symbol}_1Day.csv", daily_rows)
    return sessions


# --------------------------------------------------------------------------- Alpaca download


class SessionSource(Protocol):
    """Trading calendar over a date range (``AlpacaCalendar.get_sessions``)."""

    async def get_sessions(self, start: date, end: date) -> list[SessionDay]:
        """Sessions with ``start <= session_date <= end``, ordered by date."""
        ...


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """What :func:`download_alpaca_bars` downloads.

    Attributes:
        symbols: Symbols (upper case, unique).
        start: First session of the dataset (minute bars and ``calendar.csv``). To feed
            the backtest indicator warm-up, start some sessions before
            ``backtest.start_date`` (the runner warms up on stored sessions before it).
        end: Last session of the dataset (inclusive).
        feed: Data feed; must be the one the strategy runs on live (sec. 10.2).
        adjustment: Corporate-action adjustment; OWNER_DECISION, same in live and
            backtest (sec. 44). Recorded in the manifest, applied by the market data.
        daily_warmup_days: Trading sessions of daily bars to download BEFORE ``start``
            so the volume lookback (``universe.avg_volume_lookback_days``) is covered
            from the first session.
    """

    symbols: tuple[str, ...]
    start: date
    end: date
    feed: DataFeed
    adjustment: BarAdjustment
    daily_warmup_days: int

    def __post_init__(self) -> None:
        if not self.symbols:
            raise NonRetryableError("no symbols to download", code="INVALID_DOWNLOAD_SPEC")
        if len(set(self.symbols)) != len(self.symbols) or any(
            not s or s != s.upper() or not s.replace(".", "").isalnum() for s in self.symbols
        ):
            raise NonRetryableError(
                "symbols must be unique upper-case tickers", code="INVALID_DOWNLOAD_SPEC"
            )
        if self.end < self.start:
            raise NonRetryableError("end is before start", code="INVALID_DOWNLOAD_SPEC")
        if self.daily_warmup_days < 0:
            raise NonRetryableError("daily_warmup_days must be >= 0", code="INVALID_DOWNLOAD_SPEC")

    def manifest(self) -> dict[str, object]:
        """Parameters that identify the dataset (resume refuses a different one)."""
        return {
            "version": _MANIFEST_VERSION,
            "symbols": sorted(self.symbols),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "feed": self.feed.value,
            "adjustment": str(self.adjustment.value),
            "daily_warmup_days": self.daily_warmup_days,
            "minute_bars": "regular_session_only",
        }


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of a download.

    Attributes:
        directory: Dataset directory.
        sessions: Sessions written to ``calendar.csv``.
        daily_start: First date of the daily bars.
        rows: Data rows per written bar file name (e.g. ``"SPY_1Min.csv"``).
        downloaded_chunks: Chunks fetched from the API in this run.
        cached_chunks: Chunks reused from a previous (interrupted) run.
    """

    directory: Path
    sessions: tuple[SessionDay, ...]
    daily_start: date
    rows: dict[str, int] = field(default_factory=dict)
    downloaded_chunks: int = 0
    cached_chunks: int = 0


def _bar_row(bar: Bar) -> _Row:
    if bar.open is None or bar.high is None or bar.low is None or bar.close is None:
        raise NonRetryableError(f"bar without prices: {bar.symbol}", code="INVALID_BAR_DATA")
    return (bar.bar_start_utc, bar.open, bar.high, bar.low, bar.close, bar.volume)


def _replace_atomically(tmp: Path, target: Path) -> None:
    os.replace(tmp, target)


def _write_chunk(path: Path, bars: Sequence[Bar]) -> None:
    """Write one chunk atomically: a chunk file exists only when complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    _write_bars(tmp, [_bar_row(b) for b in sorted(bars, key=lambda b: b.bar_start_utc)])
    _replace_atomically(tmp, path)


def _concat_chunks(chunks: Sequence[Path], target: Path) -> int:
    """Concatenate chunk files (already sorted, disjoint) into ``target``; data rows."""
    lines: list[str] = []
    for chunk in chunks:
        content = chunk.read_text(encoding="utf-8").splitlines()
        if not content or content[0] != _BAR_HEADER:
            raise NonRetryableError(f"corrupt chunk {chunk}", code="INVALID_BAR_DATA")
        lines.extend(content[1:])
    stamps = [line.split(",", 1)[0] for line in lines]
    if stamps != sorted(set(stamps)):
        raise NonRetryableError(f"chunks of {target.name} overlap", code="DUPLICATE_BAR")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text("\n".join([_BAR_HEADER, *lines]) + "\n", encoding="utf-8", newline="\n")
    _replace_atomically(tmp, target)
    return len(lines)


def _check_manifest(cache: Path, spec: DownloadSpec) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / _MANIFEST_FILE
    expected = spec.manifest()
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise NonRetryableError(
                f"unreadable {path}; delete {cache} to restart", code="DOWNLOAD_MANIFEST_MISMATCH"
            ) from exc
        if stored != expected:
            raise NonRetryableError(
                f"{path} was created with different parameters ({stored}); use another "
                f"--out directory or delete {cache} to restart",
                code="DOWNLOAD_MANIFEST_MISMATCH",
            )
        return
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _in_session(bar: Bar, by_date: Mapping[date, SessionDay]) -> bool:
    session = by_date.get(bar.bar_start_utc.date())
    return (
        session is not None
        and session.open_utc <= bar.bar_start_utc
        and bar.bar_end_utc <= session.close_utc
    )


def _year_ranges(start: date, end: date) -> list[tuple[int, date, date]]:
    return [
        (year, max(start, date(year, 1, 1)), min(end, date(year, 12, 31)))
        for year in range(start.year, end.year + 1)
    ]


async def download_alpaca_bars(
    directory: Path,
    spec: DownloadSpec,
    *,
    market_data: IMarketData,
    calendar: SessionSource,
    progress: Callable[[str], None] = print,
) -> DownloadResult:
    """Download real bars + calendar into ``directory`` in the backtest layout.

    Writes exactly what :func:`load_backtest_data` reads: ``calendar.csv`` (sessions in
    ``[spec.start, spec.end]`` from the broker calendar) and ``bars/{SYMBOL}_1Min.csv``
    / ``bars/{SYMBOL}_1Day.csv`` (writer of the synthetic generator, ``t`` = bar START).

    * Minute bars: **regular session only** (``open_utc <= start`` and ``end <=
      close_utc`` of the session of that date, early closes honored). Pre/post-market
      bars are dropped: the strategy only trades the regular session (sec. 5.5).
    * Daily bars: from ``spec.daily_warmup_days`` sessions before ``spec.start`` to
      ``spec.end``.
    * Resumable: each symbol/month (minute) and symbol/year (daily) chunk is written
      atomically under ``directory/.download/chunks``; a re-run with the same
      parameters (checked against ``.download/manifest.json``) skips completed chunks,
      so a crash never restarts from zero. Final files are rebuilt from the chunks.
    * Deterministic: chunks and final files are sorted by time.

    Args:
        directory: Dataset directory (created if needed).
        spec: What to download.
        market_data: Historical bars (``AlpacaMarketData`` configured with
            ``spec.feed`` and ``spec.adjustment``).
        calendar: Trading calendar (``AlpacaCalendar``).
        progress: Receives one line per symbol/chunk.

    Raises:
        NonRetryableError: invalid spec, manifest mismatch, no sessions, bad data.
        RetryableError: API retries exhausted (re-run to resume).
    """
    cache = directory / DOWNLOAD_CACHE_DIR
    _check_manifest(cache, spec)
    lookback = timedelta(days=spec.daily_warmup_days * 2 + 10)
    all_sessions = await calendar.get_sessions(spec.start - lookback, spec.end)
    sessions = tuple(s for s in all_sessions if spec.start <= s.session_date <= spec.end)
    if not sessions:
        raise NonRetryableError(
            f"no trading sessions between {spec.start} and {spec.end}", code="NO_SESSIONS"
        )
    before = [s for s in all_sessions if s.session_date < spec.start]
    if len(before) < spec.daily_warmup_days:
        raise NonRetryableError(
            f"calendar has only {len(before)} sessions before {spec.start}",
            code="NO_SESSIONS",
        )
    daily_start = (
        before[-spec.daily_warmup_days].session_date if spec.daily_warmup_days else spec.start
    )
    by_date = {s.session_date: s for s in sessions}
    months: dict[tuple[int, int], list[SessionDay]] = {}
    for session in sessions:
        months.setdefault((session.session_date.year, session.session_date.month), []).append(
            session
        )
    downloaded = cached = 0
    rows: dict[str, int] = {}
    for symbol in spec.symbols:
        minute_chunks: list[Path] = []
        for (year, month), month_sessions in sorted(months.items()):
            path = cache / "chunks" / symbol / "1Min" / f"{year:04d}-{month:02d}.csv"
            minute_chunks.append(path)
            label = f"{symbol} 1Min {year:04d}-{month:02d}"
            if path.is_file():
                cached += 1
                progress(f"{label}: cached")
                continue
            bars = await market_data.get_minute_bars(
                symbol, month_sessions[0].open_utc, month_sessions[-1].close_utc
            )
            kept = [b for b in bars if _in_session(b, by_date)]
            _write_chunk(path, kept)
            downloaded += 1
            progress(f"{label}: {len(kept)} regular-session bars ({len(bars) - len(kept)} dropped)")
        daily_chunks: list[Path] = []
        for year, first, last in _year_ranges(daily_start, spec.end):
            path = cache / "chunks" / symbol / "1Day" / f"{year:04d}.csv"
            daily_chunks.append(path)
            label = f"{symbol} 1Day {year:04d}"
            if path.is_file():
                cached += 1
                progress(f"{label}: cached")
                continue
            daily = await market_data.get_daily_bars(symbol, first, last)
            _write_chunk(path, daily)
            downloaded += 1
            progress(f"{label}: {len(daily)} daily bars")
        for name, chunks in (
            (f"{symbol}_1Min.csv", minute_chunks),
            (f"{symbol}_1Day.csv", daily_chunks),
        ):
            rows[name] = _concat_chunks(chunks, directory / BARS_DIR / name)
        progress(
            f"{symbol}: {rows[f'{symbol}_1Min.csv']} minute bars, "
            f"{rows[f'{symbol}_1Day.csv']} daily bars"
        )
    calendar_tmp = directory / (CALENDAR_FILE + ".tmp")
    write_calendar(calendar_tmp, sessions)
    _replace_atomically(calendar_tmp, directory / CALENDAR_FILE)
    return DownloadResult(
        directory=directory,
        sessions=sessions,
        daily_start=daily_start,
        rows=rows,
        downloaded_chunks=downloaded,
        cached_chunks=cached,
    )


# --------------------------------------------------------------------------- CLI

_ADJUSTMENT_HELP = (
    "corporate-action adjustment of the bars (REQUIRED, no default). The dividend/split "
    "convention is an OWNER_DECISION (sec. 44, 59; config market_data.adjustment): live "
    "trading and the backtest MUST use the same convention."
)


def _synthetic_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.data synthetic",
        description="Generate a deterministic SYNTHETIC dataset (not market data).",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start-price", type=Decimal, default=Decimal("100"))
    parser.add_argument("--drift-bps", type=float, default=0.0)
    parser.add_argument("--volatility-bps", type=float, default=8.0)
    return parser


def _download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.data download",
        description=(
            "Download REAL historical bars and the trading calendar from Alpaca "
            "(historical market data only: no orders, no streaming) into the backtest "
            "dataset layout. Resumable: re-run the same command after an interruption."
        ),
        epilog=(
            "NOTE: --adjustment has no default. The dividend/split adjustment convention "
            "is an OWNER_DECISION (sec. 44/59) that must equal config "
            "market_data.adjustment; live and backtest MUST use the same convention. "
            "--feed must match config market_data.feed (sec. 10.2)."
        ),
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    parser.add_argument("--feed", choices=[f.value for f in DataFeed], required=True)
    parser.add_argument(
        "--adjustment",
        choices=["raw", "split", "dividend", "all"],
        required=True,
        help=_ADJUSTMENT_HELP,
    )
    parser.add_argument(
        "--daily-warmup-days",
        type=int,
        required=True,
        help="trading sessions of daily bars before --start (>= avg_volume_lookback_days)",
    )
    parser.add_argument("--out", type=Path, required=True, help="dataset directory")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="secrets file")
    parser.add_argument(
        "--max-attempts", type=int, default=5, help="attempts per request (bounded retry)"
    )
    return parser


def _run_synthetic(argv: Sequence[str]) -> int:
    args = _synthetic_parser().parse_args(argv)
    spec = SyntheticSpec(
        symbols=tuple(args.symbols),
        start=args.start,
        end=args.end,
        seed=args.seed,
        start_price=args.start_price,
        drift_bps_per_minute=args.drift_bps,
        volatility_bps_per_minute=args.volatility_bps,
    )
    sessions = generate_synthetic_data(args.out, spec)
    print(f"wrote {len(sessions)} synthetic sessions for {', '.join(spec.symbols)} to {args.out}")
    return 0


def _run_download(argv: Sequence[str]) -> int:
    args = _download_parser().parse_args(argv)
    # Composition root of this CLI: the only place that builds the Alpaca adapters.
    from adapters.alpaca import AlpacaCalendar, AlpacaMarketData, BarAdjustment, RetryPolicy
    from app.secrets import load_secrets

    try:
        spec = DownloadSpec(
            symbols=tuple(args.symbols),
            start=args.start,
            end=args.end,
            feed=DataFeed(args.feed),
            adjustment=BarAdjustment(args.adjustment),
            daily_warmup_days=args.daily_warmup_days,
        )
        credentials = load_secrets(args.env_file).alpaca_credentials()
        policy = RetryPolicy(max_attempts=args.max_attempts)
        market_data = AlpacaMarketData.from_credentials(
            credentials, feed=spec.feed, adjustment=spec.adjustment, retry_policy=policy
        )
        calendar = AlpacaCalendar.from_credentials(credentials, retry_policy=policy)
        result = asyncio.run(
            download_alpaca_bars(
                args.out,
                spec,
                market_data=market_data,
                calendar=calendar,
                progress=lambda line: print(line, flush=True),
            )
        )
    except (DomainError, ValueError) as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {len(result.sessions)} sessions ({result.sessions[0].session_date}.."
        f"{result.sessions[-1].session_date}), daily bars from {result.daily_start}, "
        f"{result.downloaded_chunks} chunks downloaded, {result.cached_chunks} cached, "
        f"to {args.out}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    * ``python -m backtest.data download ...``: real data from Alpaca.
    * ``python -m backtest.data synthetic ...``: synthetic generator.
    * ``python -m backtest.data --out ...`` (no subcommand): legacy synthetic invocation.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "download":
        return _run_download(args[1:])
    if args and args[0] == "synthetic":
        return _run_synthetic(args[1:])
    if args and args[0] in {"-h", "--help"}:
        print(
            "usage: python -m backtest.data {download,synthetic} [options]\n\n"
            "  download   real historical bars + calendar from Alpaca (see download -h)\n"
            "  synthetic  deterministic synthetic dataset (see synthetic -h)\n\n"
            "Without a subcommand the arguments are passed to 'synthetic' (legacy)."
        )
        return 0
    return _run_synthetic(args)


if __name__ == "__main__":
    raise SystemExit(main())
