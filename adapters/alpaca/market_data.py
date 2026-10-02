"""``IMarketData`` over the Alpaca historical REST API (sec. 8.6, 10.1).

Scope (declared early pull-forward of part of Phase 2): **historical** 1-minute and
daily bars only, so the Phase 1 backtest can run on real data. The real-time stream is
Phase 3 (``stream_minute_bars`` raises ``NotImplementedError``) and quotes are not
fetched yet (``get_latest_quote`` returns ``None``: no quote, which the universe checks
treat as fail-closed).

SDK facts (verified against alpaca-py 0.44.0 source and the Alpaca market-data FAQ):

* ``StockHistoricalDataClient.get_stock_bars(StockBarsRequest)`` returns a ``BarSet``
  whose ``data`` maps symbol -> ``list[alpaca.data.models.Bar]``; it follows
  ``next_page_token`` internally (pages of up to 10 000 bars) until the range is done.
* ``StockBarsRequest`` takes ``symbol_or_symbols``, ``timeframe`` (``TimeFrame(1,
  TimeFrameUnit.Minute | Day)``), ``start``/``end`` (aware datetimes are converted to
  naive UTC; the API treats both as inclusive), ``feed`` (``DataFeed.IEX | SIP``) and
  ``adjustment`` (``Adjustment.RAW | SPLIT | DIVIDEND | ALL``).
* SDK bars carry ``timestamp`` (aware UTC datetime), float ``open/high/low/close`` and
  float ``volume``.

Timestamp convention (sec. 10.3.2): Alpaca labels each bar with the START of its
interval, which is inclusive, the end being exclusive (market-data FAQ, "How are bars
aggregated?"). Minute bars truncate the trade timestamp to the minute; daily bars
truncate to the day in New York time (so a daily bar starts at 04:00Z or 05:00Z).
Hence ``bar_start_utc = timestamp`` and ``bar_end_utc = timestamp + timeframe``
(``+ 1 day`` for daily bars, the convention of the stored-bar loader). Covered by
``tests/unit/alpaca/test_market_data.py``; confirmed live by the smoke download
(first regular-session minute bar at 13:30Z in June).

Conversion: prices ``Decimal(str(float))`` (shortest repr, never binary noise), volume
must be integral, status ``COMPLETE``. No SDK type leaves this module (sec. 8.3.5).

Long ranges are split into calendar-month chunks (UTC) so each request stays bounded;
every chunk goes through :func:`call_with_retry` (bounded backoff, sec. 29) and an
optional local request pacer (Alpaca Basic plan: 200 requests/minute, VERIFICAR).

Callers must only request intervals that have already closed: a bar still forming at
request time would be returned as ``COMPLETE``.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from alpaca.data.enums import Adjustment
from alpaca.data.enums import DataFeed as SdkDataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.models import Bar as SdkBar
from alpaca.data.models import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from pydantic import ValidationError

from adapters.alpaca._http import AlpacaCredentials, HttpTimeouts, install_timeouts
from adapters.alpaca._retry import RequestPacer, RetryPolicy, Sleep, call_with_retry
from domain.errors import NonRetryableError
from domain.models import Bar, BarStatus, DataFeed, Quote, Timeframe

__all__ = [
    "DEFAULT_MIN_REQUEST_INTERVAL_SECONDS",
    "AlpacaMarketData",
    "BarAdjustment",
    "StockBarsClient",
    "convert_bar",
    "month_chunks",
]

DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.3
"""Pacing for the Basic plan limit of 200 requests/minute (SUGERIDO, VERIFICAR)."""


class BarAdjustment(StrEnum):
    """Corporate-action adjustment of historical bars (Alpaca ``adjustment`` parameter).

    The dividend convention is an OWNER_DECISION (sec. 44, 59): there is no default and
    live and backtest MUST use the same value.
    """

    RAW = "raw"
    SPLIT = "split"
    DIVIDEND = "dividend"
    ALL = "all"


class StockBarsClient(Protocol):
    """The subset of ``StockHistoricalDataClient`` used here (injectable in tests)."""

    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet | dict[str, Any]:
        """Bars for the request, all pages followed."""
        ...


_SDK_TIMEFRAMES: dict[Timeframe, TimeFrame] = {
    Timeframe.MIN_1: TimeFrame(1, TimeFrameUnit.Minute),
    Timeframe.DAY_1: TimeFrame(1, TimeFrameUnit.Day),
}
_DURATIONS: dict[Timeframe, timedelta] = {
    Timeframe.MIN_1: timedelta(minutes=1),
    Timeframe.DAY_1: timedelta(days=1),
}


def _to_decimal(value: float, *, field: str) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def convert_bar(sdk_bar: SdkBar, *, symbol: str, timeframe: Timeframe, feed: DataFeed) -> Bar:
    """Convert one alpaca-py bar into a validated, ``COMPLETE`` domain ``Bar``.

    ``bar_start_utc`` is the SDK timestamp (bar START, see module docstring) and
    ``bar_end_utc = start + timeframe``.

    Raises:
        NonRetryableError: code ``INVALID_BAR_DATA`` (naive timestamp, non-integral
            volume, inconsistent OHLC, symbol mismatch...).
    """
    try:
        stamp = sdk_bar.timestamp
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("timestamp is not timezone-aware")
        if sdk_bar.symbol != symbol:
            raise ValueError(f"symbol {sdk_bar.symbol!r} != requested {symbol!r}")
        start = stamp.astimezone(UTC)
        volume = _to_decimal(sdk_bar.volume, field="volume")
        if volume != volume.to_integral_value():
            raise ValueError(f"volume is not integral: {sdk_bar.volume!r}")
        return Bar(
            symbol=symbol,
            timeframe=timeframe,
            bar_start_utc=start,
            bar_end_utc=start + _DURATIONS[timeframe],
            open=_to_decimal(sdk_bar.open, field="open"),
            high=_to_decimal(sdk_bar.high, field="high"),
            low=_to_decimal(sdk_bar.low, field="low"),
            close=_to_decimal(sdk_bar.close, field="close"),
            volume=int(volume),
            feed=feed,
            status=BarStatus.COMPLETE,
        )
    except (ValueError, ValidationError) as exc:
        raise NonRetryableError(
            f"{symbol} {timeframe.value} bar at {sdk_bar.timestamp!s}: {exc}",
            code="INVALID_BAR_DATA",
        ) from exc


def _month_start(moment: datetime) -> datetime:
    return datetime(moment.year, moment.month, 1, tzinfo=UTC)


def _next_month(moment: datetime) -> datetime:
    if moment.month == 12:
        return datetime(moment.year + 1, 1, 1, tzinfo=UTC)
    return datetime(moment.year, moment.month + 1, 1, tzinfo=UTC)


def month_chunks(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    """Split ``[start, end)`` at UTC calendar-month boundaries."""
    cursor = start
    while cursor < end:
        boundary = min(_next_month(_month_start(cursor)), end)
        yield cursor, boundary
        cursor = boundary


def _require_utc(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


class AlpacaMarketData:
    """Historical bars from Alpaca as validated domain ``Bar`` models.

    Args:
        client: alpaca-py ``StockHistoricalDataClient`` or a test double.
        feed: Data feed (``iex`` / ``sip``, OWNER_DECISION sec. 10.2).
        adjustment: Corporate-action adjustment (OWNER_DECISION sec. 44; no default).
        retry_policy: Bounded retry policy for every request.
        sleep: Async sleep used by retries and the pacer (inject a recorder in tests).
        pacer: Optional local request pacer.
        unit_random: ``U[0, 1)`` source of the retry jitter.
    """

    def __init__(
        self,
        client: StockBarsClient,
        *,
        feed: DataFeed,
        adjustment: BarAdjustment,
        retry_policy: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
        pacer: RequestPacer | None = None,
        unit_random: Callable[[], float] = random.random,
    ) -> None:
        self._client = client
        self._feed = feed
        self._adjustment = adjustment
        self._policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._pacer = pacer
        self._unit_random = unit_random

    @classmethod
    def from_credentials(
        cls,
        credentials: AlpacaCredentials,
        *,
        feed: DataFeed,
        adjustment: BarAdjustment,
        retry_policy: RetryPolicy | None = None,
        timeouts: HttpTimeouts | None = None,
        min_request_interval_seconds: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
    ) -> AlpacaMarketData:
        """Build the adapter over a real ``StockHistoricalDataClient`` (network).

        Only the composition root may call this (sec. 8.3.4).
        """
        client = StockHistoricalDataClient(
            api_key=credentials.api_key.get_secret_value(),
            secret_key=credentials.secret_key.get_secret_value(),
        )
        install_timeouts(client, timeouts or HttpTimeouts())
        return cls(
            client,
            feed=feed,
            adjustment=adjustment,
            retry_policy=retry_policy,
            pacer=RequestPacer(min_request_interval_seconds),
        )

    @property
    def feed(self) -> DataFeed:
        """Configured data feed."""
        return self._feed

    @property
    def adjustment(self) -> BarAdjustment:
        """Configured corporate-action adjustment."""
        return self._adjustment

    async def _fetch(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
    ) -> list[Bar]:
        """Bars of one bounded chunk with ``start <= bar_start_utc < end``."""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_SDK_TIMEFRAMES[timeframe],
            start=start,
            end=end,
            feed=SdkDataFeed(self._feed.value),
            adjustment=Adjustment(self._adjustment.value),
        )
        operation = f"get_stock_bars {symbol} {timeframe.value} {start.isoformat()}"
        result = await call_with_retry(
            lambda: self._client.get_stock_bars(request),
            operation=operation,
            policy=self._policy,
            sleep=self._sleep,
            pacer=self._pacer,
            unit_random=self._unit_random,
        )
        if not isinstance(result, BarSet):
            raise NonRetryableError(
                f"{operation}: expected a BarSet, got {type(result).__name__}",
                code="INVALID_SCHEMA",
            )
        bars = [
            convert_bar(item, symbol=symbol, timeframe=timeframe, feed=self._feed)
            for item in result.data.get(symbol, [])
        ]
        # The API end bound is inclusive: keep [start, end) so chunks never overlap.
        return [bar for bar in bars if start <= bar.bar_start_utc < end]

    async def _bars(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
    ) -> list[Bar]:
        by_start: dict[datetime, Bar] = {}
        for chunk_start, chunk_end in month_chunks(start, end):
            for bar in await self._fetch(symbol, timeframe, chunk_start, chunk_end):
                if bar.bar_start_utc in by_start:
                    raise NonRetryableError(
                        f"duplicate bar {symbol} {timeframe.value} {bar.bar_start_utc}",
                        code="DUPLICATE_BAR",
                    )
                by_start[bar.bar_start_utc] = bar
        return [by_start[key] for key in sorted(by_start)]

    async def get_minute_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        """``1Min`` bars fully inside ``[start, end]`` (``start <= bar_start_utc`` and
        ``bar_end_utc <= end``), sorted, any session (pre/post-market included).

        Raises:
            RetryableError: retries exhausted (rate limit, 5xx, network).
            NonRetryableError: auth, rejected request, invalid data.
        """
        start_utc, end_utc = _require_utc("start", start), _require_utc("end", end)
        if end_utc <= start_utc:
            return []
        bars = await self._bars(symbol, Timeframe.MIN_1, start_utc, end_utc)
        return [bar for bar in bars if bar.bar_end_utc <= end_utc]

    async def get_daily_bars(self, symbol: str, start: date, end: date) -> list[Bar]:
        """``1Day`` bars whose UTC start date is in ``[start, end]`` (inclusive), sorted.

        Raises:
            RetryableError: retries exhausted (rate limit, 5xx, network).
            NonRetryableError: auth, rejected request, invalid data.
        """
        if end < start:
            return []
        start_utc = datetime.combine(start, time(0, 0), tzinfo=UTC)
        end_utc = datetime.combine(end + timedelta(days=1), time(0, 0), tzinfo=UTC)
        bars = await self._bars(symbol, Timeframe.DAY_1, start_utc, end_utc)
        return [bar for bar in bars if start <= bar.bar_start_utc.date() <= end]

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        """Quotes are not fetched before Phase 3: always ``None`` (no quote available)."""
        return None

    def stream_minute_bars(self, symbols: Sequence[str]) -> AsyncIterator[Bar]:
        """Real-time stream (``StockDataStream``) is Phase 3.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError("Phase 3")
