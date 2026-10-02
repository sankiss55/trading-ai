"""Deterministic BUY signals (sec. 13.7).

``signal_id = "sig_" + sha256(canonical).hexdigest()`` with the canonical string::

    "{strategy_version}|{symbol}|{timeframe}|{bar_start_utc ISO-8601 UTC}|BUY"

for example ``"0.1.0|SPY|5Min|2026-06-15T13:30:00+00:00|BUY"``. The same bar therefore
always produces the same id, even after a restart or when the bar arrives twice; a new
``strategy_version`` produces a new id. ``created_at_utc`` is passed in (no clock read).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Final

from domain.models import RuleResult, Signal, Timeframe
from domain.strategy.rules import StrategyInputError

__all__ = [
    "SIGNAL_ACTION",
    "SIGNAL_ID_PREFIX",
    "build_signal",
    "canonical_signal_key",
    "signal_id_for",
]

SIGNAL_ID_PREFIX: Final = "sig_"
SIGNAL_ACTION: Final = "BUY"


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise StrategyInputError(f"{name} must be timezone-aware (UTC)")
    return value.astimezone(UTC)


def canonical_signal_key(
    *, strategy_version: str, symbol: str, timeframe: Timeframe, bar_start_utc: datetime
) -> str:
    """Canonical string hashed into the signal id (any aware datetime is normalized to UTC)."""
    if not strategy_version:
        raise StrategyInputError("strategy_version must not be empty")
    start = _utc(bar_start_utc, "bar_start_utc").isoformat()
    return f"{strategy_version}|{symbol}|{timeframe.value}|{start}|{SIGNAL_ACTION}"


def signal_id_for(
    *, strategy_version: str, symbol: str, timeframe: Timeframe, bar_start_utc: datetime
) -> str:
    """Deterministic ``signal_id`` of a BUY signal on the bar starting at ``bar_start_utc``."""
    key = canonical_signal_key(
        strategy_version=strategy_version,
        symbol=symbol,
        timeframe=timeframe,
        bar_start_utc=bar_start_utc,
    )
    return SIGNAL_ID_PREFIX + hashlib.sha256(key.encode("utf-8")).hexdigest()


def build_signal(
    *,
    strategy_version: str,
    symbol: str,
    timeframe: Timeframe,
    bar_start_utc: datetime,
    bar_end_utc: datetime,
    created_at_utc: datetime,
    signal_ttl_seconds: int,
    rule_results: Sequence[RuleResult],
    expires_at_utc: datetime | None = None,
) -> Signal:
    """Build the immutable :class:`~domain.models.Signal` of a BUY decision.

    ``expires_at_utc = bar_end_utc + signal_ttl_seconds`` (sec. 13.7) unless the caller
    passes ``expires_at_utc`` (daily bars: the next session open plus the TTL; it may
    not be before ``bar_end_utc``).

    Raises:
        StrategyInputError: non-positive TTL, empty version, naive datetimes or an
            explicit expiry before ``bar_end_utc``.
    """
    if isinstance(signal_ttl_seconds, bool) or signal_ttl_seconds <= 0:
        raise StrategyInputError(f"signal_ttl_seconds must be > 0, got {signal_ttl_seconds!r}")
    end = _utc(bar_end_utc, "bar_end_utc")
    expires = end + timedelta(seconds=signal_ttl_seconds)
    if expires_at_utc is not None:
        expires = _utc(expires_at_utc, "expires_at_utc")
        if expires < end:
            raise StrategyInputError("expires_at_utc must not be before bar_end_utc")
    return Signal(
        signal_id=signal_id_for(
            strategy_version=strategy_version,
            symbol=symbol,
            timeframe=timeframe,
            bar_start_utc=bar_start_utc,
        ),
        symbol=symbol,
        timeframe=timeframe,
        bar_start_utc=_utc(bar_start_utc, "bar_start_utc"),
        bar_end_utc=end,
        created_at_utc=_utc(created_at_utc, "created_at_utc"),
        expires_at_utc=expires,
        rule_results=tuple(rule_results),
        strategy_version=strategy_version,
    )
