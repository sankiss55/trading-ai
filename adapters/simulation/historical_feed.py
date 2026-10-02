"""Stored-bar market data for backtest and simulation (sec. 8.6, 10.3, 45.2).

``HistoricalFeed`` (backtest) serves clean stored bars; ``ScriptedFeed`` (tests and
deterministic simulation, sec. 49.5) replays an exact scripted sequence, including
duplicates, gaps and out-of-order bars, so failure cases can be reproduced.

CSV format (one file per symbol and timeframe, named ``{SYMBOL}_{TIMEFRAME}.csv``,
e.g. ``SPY_1Min.csv`` or ``SPY_1Day.csv``)::

    t,o,h,l,c,v
    2026-10-01T13:30:00Z,500.10,501.00,499.90,500.50,1000

* ``t`` is the bar START in UTC, ISO 8601 (``Z`` or ``+00:00`` suffix required), which
  is the Alpaca timestamp convention (sec. 10.3.2, VERIFICAR).
* ``bar_end_utc`` is ``t + timeframe`` for intraday timeframes and ``t + 1 day`` for
  daily bars.
* Prices are parsed as ``Decimal`` from their text; volume as ``int``.
* Every stored bar is ``COMPLETE``.

Determinism: no network, no wall clock, no randomness. Ordering ties are broken by
symbol.
"""

from __future__ import annotations

import csv
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from adapters.simulation.sim_clock import AdvanceableClock
from domain.errors import NonRetryableError
from domain.models import Bar, BarStatus, DataFeed, Quote, Timeframe

__all__ = ["CSV_COLUMNS", "HistoricalFeed", "ScriptedFeed", "bar_duration", "load_bars_csv"]

CSV_COLUMNS = ("t", "o", "h", "l", "c", "v")


def bar_duration(timeframe: Timeframe) -> timedelta:
    """Duration covered by one bar of ``timeframe`` (daily bars cover 24 hours)."""
    minutes = timeframe.minutes
    return timedelta(days=1) if minutes is None else timedelta(minutes=minutes)


def _parse_utc(text: str) -> datetime:
    value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"timestamp must be UTC: {text!r}")
    return value


def _parse_decimal(text: str) -> Decimal:
    try:
        return Decimal(text.strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal {text!r}") from exc


def load_bars_csv(path: Path, *, symbol: str, timeframe: Timeframe, feed: DataFeed) -> list[Bar]:
    """Load one ``t,o,h,l,c,v`` CSV file into validated ``Bar`` models.

    Raises:
        NonRetryableError: code ``INVALID_BAR_DATA`` for a bad header or row.
    """
    duration = bar_duration(timeframe)
    bars: list[Bar] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise NonRetryableError(
                f"{path.name}: header must be {','.join(CSV_COLUMNS)}", code="INVALID_BAR_DATA"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                start = _parse_utc(row["t"])
                bars.append(
                    Bar(
                        symbol=symbol,
                        timeframe=timeframe,
                        bar_start_utc=start,
                        bar_end_utc=start + duration,
                        open=_parse_decimal(row["o"]),
                        high=_parse_decimal(row["h"]),
                        low=_parse_decimal(row["l"]),
                        close=_parse_decimal(row["c"]),
                        volume=int(row["v"]),
                        feed=feed,
                        status=BarStatus.COMPLETE,
                    )
                )
            except (ValueError, TypeError) as exc:
                raise NonRetryableError(
                    f"{path.name}:{line_number}: {exc}", code="INVALID_BAR_DATA"
                ) from exc
    return bars


def _stream_key(bar: Bar) -> tuple[datetime, str]:
    return (bar.bar_start_utc, bar.symbol)


class HistoricalFeed:
    """``IMarketData`` over clean stored bars.

    Args:
        bars: Stored bars (any timeframe; only ``1Min`` and ``1Day`` are queried).
            Duplicate ``(symbol, timeframe, bar_start_utc)`` keys are rejected.
        clock: Optional simulated clock. When given, ``stream_minute_bars`` advances it
            to each bar's END before yielding it, and historical queries and quotes
            never return data that is not yet known at ``clock.now_utc()`` (bars with
            ``bar_end_utc > now``, quotes with ``timestamp_utc > now``): no lookahead
            (sec. 45.2.8).
        quotes: Optional scripted quotes per symbol (any order).
    """

    def __init__(
        self,
        bars: Iterable[Bar] = (),
        *,
        clock: AdvanceableClock | None = None,
        quotes: Mapping[str, Sequence[Quote]] | None = None,
    ) -> None:
        self._by_key: dict[tuple[str, Timeframe], list[Bar]] = {}
        seen: set[tuple[str, Timeframe, datetime]] = set()
        for bar in bars:
            key = (bar.symbol, bar.timeframe, bar.bar_start_utc)
            if key in seen:
                raise NonRetryableError(
                    f"duplicate bar {bar.symbol} {bar.timeframe} {bar.bar_start_utc.isoformat()}",
                    code="DUPLICATE_BAR",
                )
            seen.add(key)
            self._by_key.setdefault((bar.symbol, bar.timeframe), []).append(bar)
        for series in self._by_key.values():
            series.sort(key=lambda b: b.bar_start_utc)
        self._clock = clock
        self._quotes = {
            symbol: sorted(items, key=lambda q: q.timestamp_utc)
            for symbol, items in (quotes or {}).items()
        }

    @classmethod
    def from_csv_dir(
        cls,
        directory: Path,
        *,
        feed: DataFeed,
        clock: AdvanceableClock | None = None,
        quotes: Mapping[str, Sequence[Quote]] | None = None,
    ) -> HistoricalFeed:
        """Load every ``{SYMBOL}_{TIMEFRAME}.csv`` file in ``directory`` (sorted by name).

        Raises:
            NonRetryableError: code ``INVALID_BAR_DATA`` for a bad file name or content.
        """
        bars: list[Bar] = []
        for path in sorted(directory.glob("*.csv")):
            symbol, sep, timeframe_text = path.stem.rpartition("_")
            try:
                timeframe = Timeframe(timeframe_text)
            except ValueError as exc:
                raise NonRetryableError(
                    f"{path.name}: unknown timeframe {timeframe_text!r}", code="INVALID_BAR_DATA"
                ) from exc
            if not sep or not symbol:
                raise NonRetryableError(
                    f"{path.name}: expected SYMBOL_TIMEFRAME.csv", code="INVALID_BAR_DATA"
                )
            bars.extend(load_bars_csv(path, symbol=symbol, timeframe=timeframe, feed=feed))
        return cls(bars, clock=clock, quotes=quotes)

    def _known(self, bar: Bar) -> bool:
        return self._clock is None or bar.bar_end_utc <= self._clock.now_utc()

    async def get_minute_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        """``1Min`` bars fully inside ``[start, end]``: ``start <= bar_start_utc`` and
        ``bar_end_utc <= end``, sorted by time."""
        return [
            bar
            for bar in self._by_key.get((symbol, Timeframe.MIN_1), [])
            if start <= bar.bar_start_utc and bar.bar_end_utc <= end and self._known(bar)
        ]

    async def get_daily_bars(self, symbol: str, start: date, end: date) -> list[Bar]:
        """``1Day`` bars whose UTC start date is in ``[start, end]`` (inclusive)."""
        return [
            bar
            for bar in self._by_key.get((symbol, Timeframe.DAY_1), [])
            if start <= bar.bar_start_utc.date() <= end and self._known(bar)
        ]

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        """Latest scripted quote known at the clock time, or ``None``."""
        candidates = self._quotes.get(symbol, [])
        if self._clock is not None:
            now = self._clock.now_utc()
            candidates = [q for q in candidates if q.timestamp_utc <= now]
        return candidates[-1] if candidates else None

    async def stream_minute_bars(self, symbols: Sequence[str]) -> AsyncIterator[Bar]:
        """Replay stored ``1Min`` bars of ``symbols`` in time order (ties by symbol).

        If a clock was given, it is advanced to ``bar_end_utc`` before each yield, so
        consumers observe the bar exactly when it closes. Bars ending before the
        current clock time are skipped (already in the past).
        """
        wanted = sorted(set(symbols))
        merged = sorted(
            (bar for symbol in wanted for bar in self._by_key.get((symbol, Timeframe.MIN_1), [])),
            key=_stream_key,
        )
        for bar in merged:
            if self._clock is not None:
                if bar.bar_end_utc < self._clock.now_utc():
                    continue
                self._clock.advance_to(bar.bar_end_utc)
            yield bar


class ScriptedFeed:
    """``IMarketData`` that replays an exact script (sec. 49.5).

    Unlike ``HistoricalFeed`` the stream is NOT sorted or de-duplicated: bars are
    yielded exactly in script order (filtered by symbol), which lets tests inject
    duplicates, gaps and out-of-order bars (sec. 49.3).

    Args:
        stream: Bars yielded by ``stream_minute_bars`` in this order.
        history: Bars served by ``get_minute_bars`` / ``get_daily_bars``.
        quotes: Latest quote per symbol (``None`` or missing means no quote).
        clock: Optional clock advanced to ``bar_end_utc`` before each yield, only when
            that moves it forward (an out-of-order bar never moves time backwards).
    """

    def __init__(
        self,
        stream: Sequence[Bar] = (),
        *,
        history: Sequence[Bar] = (),
        quotes: Mapping[str, Quote | None] | None = None,
        clock: AdvanceableClock | None = None,
    ) -> None:
        self._stream = tuple(stream)
        self._history = tuple(history)
        self._quotes = dict(quotes or {})
        self._clock = clock

    async def get_minute_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        """Scripted ``1Min`` history bars of ``symbol`` with ``start <= bar_start_utc``
        and ``bar_end_utc <= end``, in script order."""
        return [
            bar
            for bar in self._history
            if bar.symbol == symbol
            and bar.timeframe is Timeframe.MIN_1
            and start <= bar.bar_start_utc
            and bar.bar_end_utc <= end
        ]

    async def get_daily_bars(self, symbol: str, start: date, end: date) -> list[Bar]:
        """Scripted ``1Day`` history bars of ``symbol`` with UTC start date in range."""
        return [
            bar
            for bar in self._history
            if bar.symbol == symbol
            and bar.timeframe is Timeframe.DAY_1
            and start <= bar.bar_start_utc.date() <= end
        ]

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        """Scripted quote of ``symbol`` or ``None``."""
        return self._quotes.get(symbol)

    async def stream_minute_bars(self, symbols: Sequence[str]) -> AsyncIterator[Bar]:
        """Yield scripted bars of ``symbols`` in script order."""
        wanted = set(symbols)
        for bar in self._stream:
            if bar.symbol not in wanted:
                continue
            if self._clock is not None and bar.bar_end_utc > self._clock.now_utc():
                self._clock.advance_to(bar.bar_end_utc)
            yield bar
