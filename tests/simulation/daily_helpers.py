"""Helpers for the daily (``1Day``) backtest simulations.

Every parameter comes from ``tests/fixtures/config.daily.yaml`` (TEST FIXTURE values,
not owner decisions). Daily bars are built in memory and labelled like SIP daily bars:
at midnight America/New_York (04:00Z in summer, 05:00Z in winter), ``bar_end = start +
1 day``; the runner re-stamps them to their session.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from adapters.simulation import HistoricalFeed
from adapters.simulation.static_calendar import build_regular_sessions
from app.config import AppConfig, LoadedConfig, load_config
from backtest.data import BARS_DIR, CALENDAR_FILE, BacktestData, write_calendar
from domain.models import Bar, BarStatus, DataFeed, SessionDay, Timeframe

ROOT = Path(__file__).resolve().parents[2]
DAILY_CONFIG = ROOT / "tests" / "fixtures" / "config.daily.yaml"
NEW_YORK = ZoneInfo("America/New_York")
CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class Ohlc:
    """Prices of one synthetic daily bar (text, parsed as Decimal)."""

    open: str
    high: str
    low: str
    close: str
    volume: int = 1_000_000


def weekdays(start: date, count: int) -> list[date]:
    """The first ``count`` weekdays from ``start`` (no holidays: synthetic calendar)."""
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def sessions_for(days: Sequence[date]) -> tuple[SessionDay, ...]:
    """Regular 09:30-16:00 New York sessions of ``days``."""
    return tuple(build_regular_sessions(days).values())


def daily_bar(symbol: str, day: date, ohlc: Ohlc, *, feed: DataFeed = DataFeed.SIP) -> Bar:
    """A COMPLETE 1Day bar labelled at midnight New York of ``day`` (SIP convention)."""
    start = datetime.combine(day, time(0, 0), tzinfo=NEW_YORK).astimezone(UTC)
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.DAY_1,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(days=1),
        open=Decimal(ohlc.open),
        high=Decimal(ohlc.high),
        low=Decimal(ohlc.low),
        close=Decimal(ohlc.close),
        volume=ohlc.volume,
        feed=feed,
        status=BarStatus.COMPLETE,
    )


def flat_day(price: str = "100") -> Ohlc:
    """A neutral day: IBS 0.5, range 2."""
    mid = Decimal(price)
    return Ohlc(str(mid), str(mid + 1), str(mid - 1), str(mid))


def dataset(
    bars_by_symbol: Mapping[str, Sequence[Ohlc | None]], days: Sequence[date], directory: Path
) -> BacktestData:
    """In-memory dataset: one bar per symbol and day (``None`` entries are skipped)."""
    bars = [
        daily_bar(symbol, day, ohlc)
        for symbol, series in bars_by_symbol.items()
        for day, ohlc in zip(days, series, strict=True)
        if ohlc is not None
    ]
    return BacktestData(feed=HistoricalFeed(bars), sessions=sessions_for(days), directory=directory)


def random_walk(seed: int, count: int, *, start: str = "100") -> list[Ohlc]:
    """Seeded daily random walk with an overnight gap (fixture data, not market data)."""
    rng = random.Random(seed)
    price = float(start)
    out: list[Ohlc] = []
    for _ in range(count):
        open_ = price * (1 + rng.gauss(0.0, 0.004))
        close = open_ * (1 + rng.gauss(0.0003, 0.012))
        high = max(open_, close) * (1 + abs(rng.gauss(0.0, 0.004)))
        low = min(open_, close) * (1 - abs(rng.gauss(0.0, 0.004)))
        o, c = _cents(open_), _cents(close)
        out.append(
            Ohlc(
                str(o),
                str(max(_cents(high), o, c)),
                str(min(_cents(low), o, c)),
                str(c),
                volume=rng.randint(500_000, 2_000_000),
            )
        )
        price = close
    return out


def _cents(value: float) -> Decimal:
    return Decimal(repr(value)).quantize(CENT, rounding=ROUND_HALF_EVEN)


def write_dataset(
    directory: Path, bars_by_symbol: Mapping[str, Sequence[Ohlc]], days: Sequence[date]
) -> Path:
    """Write the dataset as ``calendar.csv`` + ``bars/{SYMBOL}_1Day.csv`` (no manifest)."""
    directory.mkdir(parents=True, exist_ok=True)
    write_calendar(directory / CALENDAR_FILE, sessions_for(days))
    bars_dir = directory / BARS_DIR
    bars_dir.mkdir(exist_ok=True)
    for symbol, series in bars_by_symbol.items():
        with (bars_dir / f"{symbol}_1Day.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(("t", "o", "h", "l", "c", "v"))
            for day, ohlc in zip(days, series, strict=True):
                bar = daily_bar(symbol, day, ohlc)
                writer.writerow(
                    (
                        bar.bar_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        ohlc.open,
                        ohlc.high,
                        ohlc.low,
                        ohlc.close,
                        ohlc.volume,
                    )
                )
    return directory


def daily_loaded(
    *,
    whitelist: Iterable[str] | None = None,
    backtest: dict[str, Any] | None = None,
    strategy: dict[str, Any] | None = None,
    exit_: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> LoadedConfig:
    """The daily fixture config with optional section overrides (values must stay valid)."""
    loaded = load_config(DAILY_CONFIG)
    config: AppConfig = loaded.config
    strategy_section = config.strategy
    if exit_:
        strategy_section = strategy_section.model_copy(
            update={"exit": strategy_section.exit.model_copy(update=exit_)}
        )
    if strategy:
        strategy_section = strategy_section.model_copy(update=strategy)
    universe = config.universe
    if whitelist is not None:
        universe = universe.model_copy(update={"whitelist": tuple(whitelist)})
    config = config.model_copy(
        update={
            "universe": universe,
            "strategy": strategy_section,
            "backtest": config.backtest.model_copy(update=backtest or {}),
            "execution": config.execution.model_copy(update=execution or {}),
        }
    )
    return LoadedConfig(config=config, config_hash=loaded.config_hash, path=loaded.path)
