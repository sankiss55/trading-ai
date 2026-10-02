"""Session windows derived from the broker calendar (sec. 11).

The broker calendar (``SessionDay``) is the source of truth for open/close, including
early closes (sec. 11.1, 11.3). This module derives the per-day windows of sec. 11.4:

```text
session_open            = calendar open
session_close           = calendar close (may be early)
entries_allowed_from    = session_open  + no_entry_first_minutes
entries_allowed_until   = session_close - no_entry_last_minutes
flatten_at              = session_close - flatten_minutes_before_close   (intraday only)
```

Every parameter is passed explicitly. A parameter that is still ``None`` (an
OWNER_DECISION not taken yet) raises :class:`OwnerDecisionPendingError`; the module
never substitutes a default. Time is always passed in as data (``now_utc``).

Daily bars (strategy family v2, ``primary_timeframe = 1Day``): the data feed labels a
daily bar with a calendar instant (SIP: midnight America/New_York, i.e. 04:00Z/05:00Z;
synthetic data: 00:00Z) and ``bar_end = start + 1 day``. :func:`session_bar` re-stamps it
to the session it summarises, ``[open_utc, close_utc)``, so fills at the open, intrabar
stops and the decision at the close get the real session instants (no lookahead: the
bar is known only at the close). :func:`empty_session_bar` is the EMPTY bar of a session
without a daily bar (sec. 10.3.4), so elapsed sessions are still counted.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, tzinfo

from pydantic import Field

from domain.errors import DomainError, NonRetryableError
from domain.models import (
    Bar,
    BarStatus,
    DataFeed,
    DomainModel,
    HoldingMode,
    SessionDay,
    Symbol,
    Timeframe,
    UtcDatetime,
)

__all__ = [
    "OWNER_DECISION_PENDING_CODE",
    "SESSION_BAR_MISMATCH_CODE",
    "OwnerDecisionPendingError",
    "SessionBarMismatchError",
    "SessionWindowParams",
    "SessionWindows",
    "compute_session_windows",
    "empty_session_bar",
    "session_bar",
    "session_day_from_market_times",
]

OWNER_DECISION_PENDING_CODE = "OWNER_DECISION_PENDING"
SESSION_BAR_MISMATCH_CODE = "SESSION_BAR_MISMATCH"


class OwnerDecisionPendingError(DomainError):
    """A required OWNER_DECISION parameter is still unset (``None``)."""

    def __init__(self, parameter: str) -> None:
        super().__init__(
            f"parameter {parameter!r} is an OWNER_DECISION that is still pending (None); "
            "it must be set in config before session windows can be computed",
            code=OWNER_DECISION_PENDING_CODE,
        )
        self.parameter = parameter


class SessionWindowParams(DomainModel):
    """Session-window parameters (config ``session.*`` and ``strategy.*``).

    ``None`` means "OWNER_DECISION pending". ``flatten_minutes_before_close`` is only
    required when ``holding_mode`` is ``intraday``.
    """

    no_entry_first_minutes: int | None = Field(ge=0)
    no_entry_last_minutes: int | None = Field(ge=0)
    holding_mode: HoldingMode | None
    flatten_minutes_before_close: int | None = Field(default=None, ge=0)


class SessionWindows(DomainModel):
    """Derived windows of one trading session, all in UTC (sec. 11.4).

    Intervals are half-open: entries are allowed in
    ``[entries_allowed_from_utc, entries_allowed_until_utc)``. If the parameters leave no
    room (e.g. on an early-close day), the entry window is empty and
    :meth:`is_entry_window` is always ``False``.
    """

    session_date: date
    session_open_utc: UtcDatetime
    session_close_utc: UtcDatetime
    entries_allowed_from_utc: UtcDatetime
    entries_allowed_until_utc: UtcDatetime
    flatten_at_utc: UtcDatetime | None
    is_early_close: bool

    def is_open(self, now_utc: datetime) -> bool:
        """``True`` when ``now_utc`` is inside ``[session_open, session_close)``."""
        return self.session_open_utc <= now_utc < self.session_close_utc

    def is_entry_window(self, now_utc: datetime) -> bool:
        """``True`` when new entries are allowed at ``now_utc`` (sec. 11.4)."""
        return self.entries_allowed_from_utc <= now_utc < self.entries_allowed_until_utc

    def is_flatten_time(self, now_utc: datetime) -> bool:
        """``True`` from ``flatten_at`` on (intraday only; always ``False`` for swing)."""
        return self.flatten_at_utc is not None and now_utc >= self.flatten_at_utc

    def minutes_since_open(self, now_utc: datetime) -> float:
        """Minutes elapsed since the session open (negative before the open)."""
        return (now_utc - self.session_open_utc).total_seconds() / 60.0

    def minutes_to_close(self, now_utc: datetime) -> float:
        """Minutes remaining until the session close (negative after the close)."""
        return (self.session_close_utc - now_utc).total_seconds() / 60.0


def _require(value: int | None, name: str) -> int:
    if value is None:
        raise OwnerDecisionPendingError(name)
    return value


def compute_session_windows(session: SessionDay, params: SessionWindowParams) -> SessionWindows:
    """Derive the windows of sec. 11.4 for one calendar session.

    Raises:
        OwnerDecisionPendingError: a required parameter is ``None``.
    """
    first = _require(params.no_entry_first_minutes, "session.no_entry_first_minutes")
    last = _require(params.no_entry_last_minutes, "session.no_entry_last_minutes")
    if params.holding_mode is None:
        raise OwnerDecisionPendingError("strategy.holding_mode")
    flatten_at: datetime | None = None
    if params.holding_mode is HoldingMode.INTRADAY:
        flatten_minutes = _require(
            params.flatten_minutes_before_close, "strategy.flatten_minutes_before_close"
        )
        flatten_at = session.close_utc - timedelta(minutes=flatten_minutes)
    return SessionWindows(
        session_date=session.session_date,
        session_open_utc=session.open_utc,
        session_close_utc=session.close_utc,
        entries_allowed_from_utc=session.open_utc + timedelta(minutes=first),
        entries_allowed_until_utc=session.close_utc - timedelta(minutes=last),
        flatten_at_utc=flatten_at,
        is_early_close=session.is_early_close,
    )


def session_day_from_market_times(
    session_date: date,
    *,
    open_local: time,
    close_local: time,
    market_tz: tzinfo,
    is_early_close: bool,
) -> SessionDay:
    """Build a ``SessionDay`` from local market times (simulation/backtest helper).

    Live code takes ``SessionDay`` from the broker calendar (sec. 11.1); this helper is
    for fakes and backtests whose calendar is expressed in market time. ``market_tz`` is
    passed in (e.g. ``zoneinfo.ZoneInfo("America/New_York")`` built by the caller) so the
    domain never assumes a timezone database is present.
    """
    open_utc = datetime.combine(session_date, open_local, tzinfo=market_tz).astimezone(UTC)
    close_utc = datetime.combine(session_date, close_local, tzinfo=market_tz).astimezone(UTC)
    return SessionDay(
        session_date=session_date,
        open_utc=open_utc,
        close_utc=close_utc,
        is_early_close=is_early_close,
    )


# --------------------------------------------------------------------------- daily bars


class SessionBarMismatchError(NonRetryableError):
    """A daily bar was paired with a session it does not belong to."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=SESSION_BAR_MISMATCH_CODE)


def session_bar(daily_bar: Bar, session: SessionDay) -> Bar:
    """``daily_bar`` re-stamped to ``[session.open_utc, session.close_utc)``.

    OHLCV, feed, status and ``minutes_present`` are kept. The bar's label date (the UTC
    calendar date of ``bar_start_utc``; for a label at midnight America/New_York this is
    the New York date) must be ``session.session_date`` and the label must not be after
    the session open. A bar already stamped to ``session`` is returned unchanged in value.

    Raises:
        SessionBarMismatchError: not a ``1Day`` bar, or a bar of another date.
    """
    if daily_bar.timeframe is not Timeframe.DAY_1:
        raise SessionBarMismatchError(
            f"{daily_bar.symbol} bar has timeframe {daily_bar.timeframe}, expected 1Day"
        )
    label = daily_bar.bar_start_utc
    if label.date() != session.session_date or label > session.open_utc:
        raise SessionBarMismatchError(
            f"{daily_bar.symbol} daily bar labelled {label.isoformat()} does not belong to "
            f"the session of {session.session_date.isoformat()}"
        )
    return daily_bar.model_copy(
        update={"bar_start_utc": session.open_utc, "bar_end_utc": session.close_utc}
    )


def empty_session_bar(symbol: Symbol, session: SessionDay, *, feed: DataFeed) -> Bar:
    """EMPTY ``1Day`` bar of a session that has no daily bar for ``symbol``."""
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.DAY_1,
        bar_start_utc=session.open_utc,
        bar_end_utc=session.close_utc,
        open=None,
        high=None,
        low=None,
        close=None,
        volume=0,
        feed=feed,
        status=BarStatus.EMPTY,
    )
