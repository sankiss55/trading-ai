"""Market data quality checks (sec. 10.4, 10.6).

Pure validators: they return result objects with stable codes and never log, sleep or
read the wall clock. The application layer decides what to record (``LATE_BAR``,
``INVALID_TIMESTAMP``...) and which actions follow (NO NEW TRADE, circuit breaker
WARNING, REST refill).
"""

from __future__ import annotations

from collections.abc import Container, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise

from pydantic import Field

from domain.models import Bar, CheckResult, DomainModel, SessionDay, Symbol, Timeframe, UtcDatetime

__all__ = [
    "SPEC_STALE_WARNING_RATIO",
    "BarGap",
    "BarKey",
    "DedupResult",
    "QualityCode",
    "StaleReport",
    "TimestampIssue",
    "assess_staleness",
    "bar_key",
    "check_duplicate",
    "check_staleness",
    "deduplicate_bars",
    "detect_gaps",
    "refill_window",
    "validate_bar_timestamp",
]

_ONE_MINUTE = timedelta(minutes=1)

SPEC_STALE_WARNING_RATIO = Decimal("0.5")
"""Sec. 10.6: more than half of the whitelist STALE -> circuit breaker WARNING."""


class QualityCode(StrEnum):
    """Stable codes of the data quality checks."""

    TIMESTAMP_VALID = "TIMESTAMP_VALID"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    UNIQUE_BAR = "UNIQUE_BAR"
    DUPLICATE_BAR = "DUPLICATE_BAR"
    DATA_FRESH = "DATA_FRESH"
    STALE = "STALE"


class TimestampIssue(StrEnum):
    """Why a bar timestamp is invalid (``detail["issue"]`` of ``INVALID_TIMESTAMP``)."""

    FUTURE = "FUTURE"
    MISALIGNED = "MISALIGNED"
    DURATION_MISMATCH = "DURATION_MISMATCH"


BarKey = tuple[str, Timeframe, datetime]
"""Uniqueness key of a bar: ``(symbol, timeframe, bar_start_utc)`` (sec. 10.4)."""


def bar_key(bar: Bar) -> BarKey:
    """Return the uniqueness key of ``bar``."""
    return (bar.symbol, bar.timeframe, bar.bar_start_utc)


# --------------------------------------------------------------------------- timestamps


def validate_bar_timestamp(
    bar: Bar, *, now_utc: datetime, max_future_seconds: float
) -> CheckResult:
    """Validate the timestamps of a raw feed bar.

    Intraday bars (fixed-length timeframes):

    * ``bar_start_utc`` must be on a whole minute (no seconds/microseconds) -> else
      ``MISALIGNED``;
    * ``bar_end_utc - bar_start_utc`` must equal the timeframe length -> else
      ``DURATION_MISMATCH`` (intended for raw feed bars; aggregated bars may be truncated
      by the session close);
    * the bar must be finished: ``bar_end_utc <= now_utc + max_future_seconds`` -> else
      ``FUTURE``.

    Daily bars only get the future check on ``bar_start_utc``.

    Returns ``TIMESTAMP_VALID`` or ``INVALID_TIMESTAMP`` with ``detail["issue"]``.
    """
    tolerance = timedelta(seconds=max_future_seconds)
    minutes = bar.timeframe.minutes
    detail: dict[str, object] = {
        "symbol": bar.symbol,
        "timeframe": bar.timeframe.value,
        "bar_start_utc": bar.bar_start_utc.isoformat(),
    }
    if minutes is None:
        if bar.bar_start_utc > now_utc + tolerance:
            return _invalid(TimestampIssue.FUTURE, detail)
        return CheckResult(passed=True, code=QualityCode.TIMESTAMP_VALID, detail=detail)
    if bar.bar_start_utc.second != 0 or bar.bar_start_utc.microsecond != 0:
        return _invalid(TimestampIssue.MISALIGNED, detail)
    if bar.bar_end_utc - bar.bar_start_utc != timedelta(minutes=minutes):
        return _invalid(TimestampIssue.DURATION_MISMATCH, detail)
    if bar.bar_end_utc > now_utc + tolerance:
        return _invalid(TimestampIssue.FUTURE, detail)
    return CheckResult(passed=True, code=QualityCode.TIMESTAMP_VALID, detail=detail)


def _invalid(issue: TimestampIssue, detail: dict[str, object]) -> CheckResult:
    return CheckResult(
        passed=False,
        code=QualityCode.INVALID_TIMESTAMP,
        detail={**detail, "issue": issue.value},
    )


# --------------------------------------------------------------------------- duplicates


class DedupResult(DomainModel):
    """Bars split into first occurrences and discarded duplicates (input order kept)."""

    unique: tuple[Bar, ...]
    duplicates: tuple[Bar, ...]


def check_duplicate(bar: Bar, seen: Container[BarKey]) -> CheckResult:
    """``DUPLICATE_BAR`` if ``bar_key(bar)`` is in ``seen``, else ``UNIQUE_BAR``."""
    key = bar_key(bar)
    detail = {"symbol": bar.symbol, "bar_start_utc": bar.bar_start_utc.isoformat()}
    if key in seen:
        return CheckResult(passed=False, code=QualityCode.DUPLICATE_BAR, detail=detail)
    return CheckResult(passed=True, code=QualityCode.UNIQUE_BAR, detail=detail)


def deduplicate_bars(bars: Iterable[Bar]) -> DedupResult:
    """Keep the first bar of each key; later ones with the same key are duplicates."""
    seen: set[BarKey] = set()
    unique: list[Bar] = []
    duplicates: list[Bar] = []
    for bar in bars:
        key = bar_key(bar)
        if key in seen:
            duplicates.append(bar)
        else:
            seen.add(key)
            unique.append(bar)
    return DedupResult(unique=tuple(unique), duplicates=tuple(duplicates))


# --------------------------------------------------------------------------- staleness


def check_staleness(
    symbol: Symbol,
    last_bar_end_utc: datetime | None,
    *,
    now_utc: datetime,
    max_bar_age_seconds: float,
) -> CheckResult:
    """Per-symbol freshness check (sec. 10.6).

    The age of the data is ``now_utc - last_bar_end_utc`` (the end of the last 1-minute
    bar received). ``STALE`` when it exceeds ``max_bar_age_seconds`` or when no bar was
    ever received (fail closed). Meaningful only during the regular session; the caller
    decides when to apply it.
    """
    if last_bar_end_utc is None:
        return CheckResult(
            passed=False,
            code=QualityCode.STALE,
            detail={"symbol": symbol, "age_seconds": None, "reason": "NO_DATA"},
        )
    age = (now_utc - last_bar_end_utc).total_seconds()
    detail = {"symbol": symbol, "age_seconds": age, "max_bar_age_seconds": max_bar_age_seconds}
    if age > max_bar_age_seconds:
        return CheckResult(passed=False, code=QualityCode.STALE, detail=detail)
    return CheckResult(passed=True, code=QualityCode.DATA_FRESH, detail=detail)


class StaleReport(DomainModel):
    """Whitelist-wide staleness assessment (sec. 10.6)."""

    stale_symbols: frozenset[str]
    whitelist_size: int = Field(ge=0)
    stale_count: int = Field(ge=0)
    circuit_breaker_warning: bool
    """``True`` when ``stale_count > warning_ratio * whitelist_size``."""


def assess_staleness(
    last_bar_end_by_symbol: Mapping[str, datetime | None],
    whitelist: Iterable[Symbol],
    *,
    now_utc: datetime,
    max_bar_age_seconds: float,
    warning_ratio: Decimal = SPEC_STALE_WARNING_RATIO,
) -> StaleReport:
    """Evaluate every whitelisted symbol and flag the global WARNING condition.

    Symbols missing from ``last_bar_end_by_symbol`` count as STALE (no data). The
    default ``warning_ratio`` is the spec rule "more than half of the whitelist".
    """
    symbols = sorted(set(whitelist))
    stale = frozenset(
        symbol
        for symbol in symbols
        if not check_staleness(
            symbol,
            last_bar_end_by_symbol.get(symbol),
            now_utc=now_utc,
            max_bar_age_seconds=max_bar_age_seconds,
        ).passed
    )
    warning = len(symbols) > 0 and Decimal(len(stale)) > warning_ratio * len(symbols)
    return StaleReport(
        stale_symbols=stale,
        whitelist_size=len(symbols),
        stale_count=len(stale),
        circuit_breaker_warning=warning,
    )


# --------------------------------------------------------------------------- gaps


class BarGap(DomainModel):
    """Missing interval ``[start_utc, end_utc)`` in a symbol's 1-minute series."""

    symbol: Symbol
    start_utc: UtcDatetime
    end_utc: UtcDatetime
    missing_minutes: int = Field(ge=1)


def _gap(symbol: str, start: datetime, end: datetime) -> BarGap | None:
    missing = int((end - start) // _ONE_MINUTE)
    if missing < 1:
        return None
    return BarGap(symbol=symbol, start_utc=start, end_utc=end, missing_minutes=missing)


def detect_gaps(bars: Sequence[Bar], *, session: SessionDay | None = None) -> tuple[BarGap, ...]:
    """Find holes between consecutive 1-minute bars of one symbol.

    Bars are sorted by start (duplicates ignored). A gap exists when a bar starts after
    the previous one ended. With ``session``, only bars inside it are considered and a
    leading hole between the session open and the first bar is reported too.

    Note: with the IEX feed a missing minute may simply have had no trades (sec. 10.2);
    gaps are refill *requests* and REST may legitimately return nothing for them.

    Raises:
        ValueError: bars of more than one symbol or a non-1Min timeframe.
    """
    if not bars:
        return ()
    symbols = {bar.symbol for bar in bars}
    if len(symbols) != 1:
        raise ValueError(f"detect_gaps expects a single symbol, got {sorted(symbols)}")
    if any(bar.timeframe is not Timeframe.MIN_1 for bar in bars):
        raise ValueError("detect_gaps expects 1Min bars only")
    symbol = next(iter(symbols))
    ordered = sorted(deduplicate_bars(bars).unique, key=lambda b: b.bar_start_utc)
    if session is not None:
        ordered = [b for b in ordered if session.open_utc <= b.bar_start_utc < session.close_utc]
        if not ordered:
            gap = _gap(symbol, session.open_utc, session.close_utc)
            return (gap,) if gap is not None else ()
    gaps: list[BarGap] = []
    if session is not None:
        leading = _gap(symbol, session.open_utc, ordered[0].bar_start_utc)
        if leading is not None:
            gaps.append(leading)
    for previous, current in pairwise(ordered):
        gap = _gap(symbol, previous.bar_end_utc, current.bar_start_utc)
        if gap is not None:
            gaps.append(gap)
    return tuple(gaps)


def refill_window(
    symbol: Symbol,
    *,
    last_bar_end_utc: datetime | None,
    now_utc: datetime,
    session: SessionDay,
) -> BarGap | None:
    """Interval to request from REST after a reconnect (sec. 10.4).

    From the end of the last bar received (or the session open if none / earlier) to
    the last *completed* minute (``now_utc`` floored to the minute), clipped to the
    session. ``None`` when nothing is missing.
    """
    start = session.open_utc
    if last_bar_end_utc is not None and last_bar_end_utc > start:
        start = last_bar_end_utc
    end = min(now_utc.replace(second=0, microsecond=0), session.close_utc)
    if end <= start:
        return None
    return _gap(symbol, start, end)
