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
from bisect import bisect_left
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
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
    return list(_iter_bars_csv(path, symbol=symbol, timeframe=timeframe, feed=feed))


def _iter_bars_csv(
    path: Path, *, symbol: str, timeframe: Timeframe, feed: DataFeed
) -> Iterator[Bar]:
    """Validated ``Bar`` of each row of ``path``, in file order (see :func:`load_bars_csv`).

    Equal price texts share one ``Decimal`` object (immutable, parsed from the same
    text, so the values and their exponents are identical): this keeps a large stored
    dataset small without changing any value.
    """
    duration = bar_duration(timeframe)
    decimals: dict[str, Decimal] = {}

    def parse_decimal(text: str) -> Decimal:
        value = decimals.get(text)
        if value is None:
            value = decimals[text] = _parse_decimal(text)
        return value

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise NonRetryableError(
                f"{path.name}: header must be {','.join(CSV_COLUMNS)}", code="INVALID_BAR_DATA"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                start = _parse_utc(row["t"])
                bar = Bar(
                    symbol=symbol,
                    timeframe=timeframe,
                    bar_start_utc=start,
                    bar_end_utc=start + duration,
                    open=parse_decimal(row["o"]),
                    high=parse_decimal(row["h"]),
                    low=parse_decimal(row["l"]),
                    close=parse_decimal(row["c"]),
                    volume=int(row["v"]),
                    feed=feed,
                    status=BarStatus.COMPLETE,
                )
            except (ValueError, TypeError) as exc:
                raise NonRetryableError(
                    f"{path.name}:{line_number}: {exc}", code="INVALID_BAR_DATA"
                ) from exc
            yield bar


def _stream_key(bar: Bar) -> tuple[datetime, str]:
    return (bar.bar_start_utc, bar.symbol)


def _duplicate_error(symbol: str, timeframe: Timeframe, start: datetime) -> NonRetryableError:
    return NonRetryableError(
        f"duplicate bar {symbol} {timeframe} {start.isoformat()}", code="DUPLICATE_BAR"
    )


class _BarSeries:
    """Stored bars of one ``(symbol, timeframe)``, sorted by start, kept as ``Bar``."""

    __slots__ = ("_bars", "starts")

    def __init__(self, bars: list[Bar]) -> None:
        bars.sort(key=lambda b: b.bar_start_utc)
        self._bars = bars
        self.starts = [bar.bar_start_utc for bar in bars]

    def bars(self, first: int = 0, last: int | None = None) -> list[Bar]:
        """Stored bars ``[first:last]``."""
        return self._bars[first:last]


_Row = tuple[datetime, Decimal, Decimal, Decimal, Decimal, int]
"""``(bar_start_utc, open, high, low, close, volume)`` of a validated stored bar."""


class _CompactSeries:
    """Validated CSV bars of one ``(symbol, timeframe)``, sorted by start, kept compact.

    Every row was validated once as a ``Bar`` when its file was loaded; only the
    validated values are kept (a few hundred bytes less per bar than a model, which
    matters for millions of minute bars). :meth:`bars` rebuilds the models from them
    with ``Bar.model_construct`` (no second validation): each rebuilt bar equals the
    validated one field by field, ``bar_end_utc`` being ``bar_start_utc + duration`` as
    at load time.
    """

    __slots__ = ("_duration", "_feed", "_rows", "_symbol", "_timeframe", "starts")

    def __init__(self, symbol: str, timeframe: Timeframe, feed: DataFeed, rows: list[_Row]) -> None:
        rows.sort(key=lambda r: r[0])
        self._symbol = symbol
        self._timeframe = timeframe
        self._feed = feed
        self._duration = bar_duration(timeframe)
        self._rows = rows
        self.starts = [row[0] for row in rows]

    def bars(self, first: int = 0, last: int | None = None) -> list[Bar]:
        """Stored bars ``[first:last]`` (rebuilt from validated values)."""
        construct = Bar.model_construct
        symbol, timeframe, feed = self._symbol, self._timeframe, self._feed
        duration = self._duration
        return [
            construct(
                symbol=symbol,
                timeframe=timeframe,
                bar_start_utc=start,
                bar_end_utc=start + duration,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                feed=feed,
                status=BarStatus.COMPLETE,
            )
            for start, open_, high, low, close, volume in self._rows[first:last]
        ]


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
        grouped: dict[tuple[str, Timeframe], list[Bar]] = {}
        seen: set[tuple[str, Timeframe, datetime]] = set()
        for bar in bars:
            key = (bar.symbol, bar.timeframe, bar.bar_start_utc)
            if key in seen:
                raise _duplicate_error(bar.symbol, bar.timeframe, bar.bar_start_utc)
            seen.add(key)
            grouped.setdefault((bar.symbol, bar.timeframe), []).append(bar)
        self._series: dict[tuple[str, Timeframe], _BarSeries | _CompactSeries] = {
            key: _BarSeries(series) for key, series in grouped.items()
        }
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
        feed_by_timeframe: Mapping[Timeframe, DataFeed] | None = None,
        clock: AdvanceableClock | None = None,
        quotes: Mapping[str, Sequence[Quote]] | None = None,
    ) -> HistoricalFeed:
        """Load every ``{SYMBOL}_{TIMEFRAME}.csv`` file in ``directory`` (sorted by name).

        Every row is validated as a ``Bar`` while loading (errors are raised here, file
        by file, then duplicates); the validated values are stored compactly and served
        without a second validation (see :class:`_CompactSeries`).

        Args:
            directory: Directory of the bar files.
            feed: Feed the bars are tagged with.
            feed_by_timeframe: Per-timeframe override of ``feed`` (e.g. daily bars of the
                liquidity filter downloaded from ``sip`` while minute bars are ``iex``).
            clock: See the class.
            quotes: See the class.

        Raises:
            NonRetryableError: code ``INVALID_BAR_DATA`` for a bad file name or content,
                ``DUPLICATE_BAR`` for a repeated ``(symbol, timeframe, t)``.
        """
        grouped: dict[tuple[str, Timeframe], tuple[DataFeed, list[_Row]]] = {}
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
            file_feed = (feed_by_timeframe or {}).get(timeframe, feed)
            _, rows = grouped.setdefault((symbol, timeframe), (file_feed, []))
            for bar in _iter_bars_csv(path, symbol=symbol, timeframe=timeframe, feed=file_feed):
                if bar.open is None or bar.high is None or bar.low is None or bar.close is None:
                    raise NonRetryableError(  # pragma: no cover - COMPLETE bars carry prices
                        f"{path.name}: bar without prices", code="INVALID_BAR_DATA"
                    )
                rows.append((bar.bar_start_utc, bar.open, bar.high, bar.low, bar.close, bar.volume))
        # Same duplicate check, in the same order, as the constructor (file, then row).
        seen: set[tuple[str, Timeframe, datetime]] = set()
        for (symbol, timeframe), (_, rows) in grouped.items():
            for row in rows:
                key = (symbol, timeframe, row[0])
                if key in seen:
                    raise _duplicate_error(symbol, timeframe, row[0])
                seen.add(key)
        loaded = cls(clock=clock, quotes=quotes)
        for (symbol, timeframe), (file_feed, rows) in grouped.items():
            if rows:
                loaded._series[(symbol, timeframe)] = _CompactSeries(
                    symbol, timeframe, file_feed, rows
                )
        return loaded

    def _known(self, bar: Bar) -> bool:
        return self._clock is None or bar.bar_end_utc <= self._clock.now_utc()

    def _all(self, symbol: str, timeframe: Timeframe) -> list[Bar]:
        series = self._series.get((symbol, timeframe))
        return [] if series is None else series.bars()

    async def get_minute_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        """``1Min`` bars fully inside ``[start, end]``: ``start <= bar_start_utc`` and
        ``bar_end_utc <= end``, sorted by time.

        The series is sorted by start, so the candidates ``start <= bar_start_utc < end``
        are found by bisection (a bar starting at or after ``end`` ends after it).
        """
        series = self._series.get((symbol, Timeframe.MIN_1))
        if series is None:
            return []
        first, last = bisect_left(series.starts, start), bisect_left(series.starts, end)
        return [
            bar for bar in series.bars(first, last) if bar.bar_end_utc <= end and self._known(bar)
        ]

    async def get_daily_bars(self, symbol: str, start: date, end: date) -> list[Bar]:
        """``1Day`` bars whose UTC start date is in ``[start, end]`` (inclusive)."""
        return [
            bar
            for bar in self._all(symbol, Timeframe.DAY_1)
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
            (bar for symbol in wanted for bar in self._all(symbol, Timeframe.MIN_1)),
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
