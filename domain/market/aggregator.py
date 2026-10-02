"""Aggregation of 1-minute bars into N-minute bars (sec. 10.3, 10.4).

Alignment: buckets are aligned to the *session open* of each calendar session, so for
5-minute bars on a regular day they start at 09:30, 09:35, 09:40 ET. The session open
comes from the broker calendar as a UTC ``SessionDay`` (sec. 11.1), which already
accounts for DST and early closes; no timezone database is needed here. The last bucket
of a session is truncated at the session close when the session length is not a
multiple of N (e.g. 1Hour bars on a 6.5h session end with a 30-minute bucket).

Closing rules for a bucket ``[t, t+N)`` (sec. 10.3.3), whichever happens first:

* the 1-minute bar starting at ``t+N-1`` (the last minute of the bucket) is ingested;
* :meth:`BarAggregator.on_time` is called with ``now_utc > t+N + grace``.

Emission order: buckets of one symbol are always emitted in time order. When a bucket
closes because its last minute arrived, every earlier bucket of that symbol that is
still open is closed first (they ended before the triggering minute started).

Bucket status: all minutes present -> ``COMPLETE``; some -> ``INCOMPLETE`` with
``minutes_present``; none -> ``EMPTY`` (sec. 10.3.4/5). Only closed bars are emitted.

Out-of-order / duplicates (sec. 10.4): a minute whose bucket is still open is inserted
in place; a minute whose bucket is already closed is reported as ``LATE_BAR`` and never
reopens it; a minute already seen is ``DUPLICATE``. Results are returned, never logged.

The aggregator is stateful but deterministic: time is only received as data.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from domain.errors import DomainError
from domain.models import (
    Bar,
    BarStatus,
    DataFeed,
    DomainModel,
    SessionDay,
    Symbol,
    Timeframe,
    UtcDatetime,
)

__all__ = [
    "AggregatorConfigError",
    "BarAggregator",
    "IngestResult",
    "IngestStatus",
    "RejectReason",
]

_ONE_MINUTE = timedelta(minutes=1)

_BucketId = tuple[int, int]
"""``(session_index, bucket_index)``; ordered like time."""


class AggregatorConfigError(DomainError):
    """Invalid aggregator configuration or session registration."""


class IngestStatus(StrEnum):
    """Outcome of ingesting one 1-minute bar."""

    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"
    LATE_BAR = "LATE_BAR"
    OUT_OF_SESSION = "OUT_OF_SESSION"
    UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"
    REJECTED = "REJECTED"


class RejectReason(StrEnum):
    """Why a bar was ``REJECTED``."""

    NOT_MINUTE_BAR = "NOT_MINUTE_BAR"
    FEED_MISMATCH = "FEED_MISMATCH"
    MISALIGNED = "MISALIGNED"
    EMPTY_MINUTE_BAR = "EMPTY_MINUTE_BAR"


class IngestResult(DomainModel):
    """Result of :meth:`BarAggregator.ingest`.

    ``closed_bars`` holds the aggregated bars closed by this minute (time order, one
    symbol); it is empty unless the minute was the last one of its bucket.
    """

    status: IngestStatus
    symbol: str
    minute_start_utc: UtcDatetime
    reason: RejectReason | None = None
    closed_bars: tuple[Bar, ...] = ()


@dataclass
class _SymbolState:
    cursor: _BucketId = (0, 0)
    pending: dict[_BucketId, dict[datetime, Bar]] = field(default_factory=dict)
    closed_minutes: dict[int, set[datetime]] = field(default_factory=dict)


class BarAggregator:
    """Aggregates 1-minute bars of a fixed symbol set into one target timeframe.

    Args:
        timeframe: Target intraday timeframe (``5Min``, ``15Min``...).
        symbols: Symbols to aggregate (the whitelist). EMPTY bars are emitted for every
            one of them, even if no minute ever arrives.
        feed: Expected feed of the input bars; also stamped on EMPTY bars.
        bar_close_grace_seconds: Grace after a bucket's end before the clock closes it.

    Sessions must be registered with :meth:`add_session`, in chronological order,
    before their minutes are ingested; minutes outside every registered session are
    ``OUT_OF_SESSION``.
    """

    def __init__(
        self,
        *,
        timeframe: Timeframe,
        symbols: Iterable[Symbol],
        feed: DataFeed,
        bar_close_grace_seconds: float,
    ) -> None:
        minutes = timeframe.minutes
        if minutes is None:
            raise AggregatorConfigError(
                f"timeframe {timeframe.value} is not intraday", code="INVALID_TIMEFRAME"
            )
        if bar_close_grace_seconds < 0:
            raise AggregatorConfigError(
                "bar_close_grace_seconds must be >= 0", code="INVALID_GRACE"
            )
        self._timeframe = timeframe
        self._step = timedelta(minutes=minutes)
        self._feed = feed
        self._grace = timedelta(seconds=bar_close_grace_seconds)
        self._sessions: list[SessionDay] = []
        self._opens: list[datetime] = []
        self._states: dict[str, _SymbolState] = {symbol: _SymbolState() for symbol in symbols}

    # ------------------------------------------------------------------ public API

    @property
    def timeframe(self) -> Timeframe:
        """Target timeframe."""
        return self._timeframe

    @property
    def symbols(self) -> frozenset[str]:
        """Symbols handled by this aggregator."""
        return frozenset(self._states)

    def add_session(self, session: SessionDay) -> None:
        """Register the next calendar session (strictly after the previous one).

        Raises:
            AggregatorConfigError: the session overlaps or precedes the last one.
        """
        if self._sessions and session.open_utc < self._sessions[-1].close_utc:
            raise AggregatorConfigError(
                f"session {session.session_date} must start after the previous session "
                f"{self._sessions[-1].session_date} closes",
                code="SESSION_OUT_OF_ORDER",
            )
        self._sessions.append(session)
        self._opens.append(session.open_utc)

    def next_bucket_start(self, symbol: Symbol) -> datetime | None:
        """Start of the earliest bucket of ``symbol`` not yet emitted.

        ``None`` if the symbol is unknown or every registered session is fully emitted.
        """
        state = self._states.get(symbol)
        if state is None or state.cursor[0] >= len(self._sessions):
            return None
        return self._bounds(state.cursor)[0]

    def ingest(self, bar: Bar) -> IngestResult:
        """Insert one 1-minute bar; may close its bucket (see module docstring)."""
        symbol = bar.symbol
        start = bar.bar_start_utc
        state = self._states.get(symbol)
        if state is None:
            return IngestResult(
                status=IngestStatus.UNKNOWN_SYMBOL, symbol=symbol, minute_start_utc=start
            )
        reason = self._reject_reason(bar)
        if reason is not None:
            return IngestResult(
                status=IngestStatus.REJECTED,
                symbol=symbol,
                minute_start_utc=start,
                reason=reason,
            )
        bucket_id = self._locate(start)
        if bucket_id is None:
            return IngestResult(
                status=IngestStatus.OUT_OF_SESSION, symbol=symbol, minute_start_utc=start
            )
        if bucket_id < state.cursor:
            closed = state.closed_minutes.get(bucket_id[0], set())
            status = IngestStatus.DUPLICATE if start in closed else IngestStatus.LATE_BAR
            return IngestResult(status=status, symbol=symbol, minute_start_utc=start)
        minutes = state.pending.setdefault(bucket_id, {})
        if start in minutes:
            return IngestResult(
                status=IngestStatus.DUPLICATE, symbol=symbol, minute_start_utc=start
            )
        minutes[start] = bar
        closed_bars: tuple[Bar, ...] = ()
        if start == self._bounds(bucket_id)[1] - _ONE_MINUTE:
            closed_bars = tuple(self._close_through(symbol, state, bucket_id))
        return IngestResult(
            status=IngestStatus.ACCEPTED,
            symbol=symbol,
            minute_start_utc=start,
            closed_bars=closed_bars,
        )

    def on_time(self, now_utc: datetime) -> tuple[Bar, ...]:
        """Close every bucket whose ``end + grace`` is strictly before ``now_utc``.

        Returns the newly closed bars sorted by ``(bar_start_utc, symbol)``.
        """
        emitted: list[Bar] = []
        for symbol in sorted(self._states):
            state = self._states[symbol]
            while state.cursor[0] < len(self._sessions):
                _, end = self._bounds(state.cursor)
                if not now_utc > end + self._grace:
                    break
                emitted.append(self._emit(symbol, state))
        emitted.sort(key=lambda b: (b.bar_start_utc, b.symbol))
        return tuple(emitted)

    # ------------------------------------------------------------------ internals

    def _reject_reason(self, bar: Bar) -> RejectReason | None:
        if bar.timeframe is not Timeframe.MIN_1:
            return RejectReason.NOT_MINUTE_BAR
        if bar.feed is not self._feed:
            return RejectReason.FEED_MISMATCH
        if bar.status is BarStatus.EMPTY:
            return RejectReason.EMPTY_MINUTE_BAR
        start = bar.bar_start_utc
        if start.second != 0 or start.microsecond != 0 or bar.bar_end_utc - start != _ONE_MINUTE:
            return RejectReason.MISALIGNED
        return None

    def _bucket_count(self, session: SessionDay) -> int:
        length = session.close_utc - session.open_utc
        return -(-length // self._step)

    def _bounds(self, bucket_id: _BucketId) -> tuple[datetime, datetime]:
        session = self._sessions[bucket_id[0]]
        start = session.open_utc + self._step * bucket_id[1]
        return start, min(start + self._step, session.close_utc)

    def _locate(self, minute_start: datetime) -> _BucketId | None:
        index = bisect_right(self._opens, minute_start) - 1
        if index < 0:
            return None
        session = self._sessions[index]
        if minute_start >= session.close_utc:
            return None
        return index, (minute_start - session.open_utc) // self._step

    def _advance(self, bucket_id: _BucketId) -> _BucketId:
        session_index, bucket_index = bucket_id
        if bucket_index + 1 < self._bucket_count(self._sessions[session_index]):
            return session_index, bucket_index + 1
        return session_index + 1, 0

    def _close_through(self, symbol: str, state: _SymbolState, bucket_id: _BucketId) -> list[Bar]:
        closed: list[Bar] = []
        while state.cursor <= bucket_id:
            closed.append(self._emit(symbol, state))
        return closed

    def _emit(self, symbol: str, state: _SymbolState) -> Bar:
        bucket_id = state.cursor
        session_index = bucket_id[0]
        minutes = state.pending.pop(bucket_id, {})
        state.closed_minutes.setdefault(session_index, set()).update(minutes)
        for old in [k for k in state.closed_minutes if k < session_index - 1]:
            del state.closed_minutes[old]
        state.cursor = self._advance(bucket_id)
        start, end = self._bounds(bucket_id)
        return self._build_bar(symbol, start, end, minutes)

    def _build_bar(
        self, symbol: str, start: datetime, end: datetime, minutes: dict[datetime, Bar]
    ) -> Bar:
        if not minutes:
            return Bar(
                symbol=symbol,
                timeframe=self._timeframe,
                bar_start_utc=start,
                bar_end_utc=end,
                open=None,
                high=None,
                low=None,
                close=None,
                volume=0,
                feed=self._feed,
                status=BarStatus.EMPTY,
                minutes_present=0,
            )
        ordered = [minutes[key] for key in sorted(minutes)]
        expected = (end - start) // _ONE_MINUTE
        present = len(ordered)
        return Bar(
            symbol=symbol,
            timeframe=self._timeframe,
            bar_start_utc=start,
            bar_end_utc=end,
            open=_price(ordered[0].open),
            high=max(_price(b.high) for b in ordered),
            low=min(_price(b.low) for b in ordered),
            close=_price(ordered[-1].close),
            volume=sum(b.volume for b in ordered),
            feed=self._feed,
            status=BarStatus.COMPLETE if present == expected else BarStatus.INCOMPLETE,
            minutes_present=present,
        )


def _price(value: Decimal | None) -> Decimal:
    if value is None:  # pragma: no cover - EMPTY minute bars are rejected on ingest
        raise DomainError("priced minute bar expected", code="INTERNAL_INVARIANT")
    return value
