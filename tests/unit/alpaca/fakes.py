"""Test doubles for the alpaca-py clients: no network, no credentials (sec. 58.4).

The doubles return REAL alpaca-py model objects (``BarSet``, ``Calendar``) built from
API-shaped payloads, so the SDK parsing is exercised too, and raise REAL SDK exceptions
(``APIError`` wrapping a ``requests.HTTPError``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import requests
from alpaca.common.exceptions import APIError
from alpaca.data.models import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.models import Calendar, Clock
from alpaca.trading.requests import GetCalendarRequest

FAKE_KEY = "PKTESTFAKEKEY000000"
FAKE_SECRET = "fake-secret-value-0000000000000000000000"


def api_error(status: int, message: str = "simulated") -> APIError:
    """A real ``APIError`` as alpaca-py raises it for an HTTP ``status`` response."""
    response = requests.Response()
    response.status_code = status
    return APIError(  # type: ignore[no-untyped-call]  # untyped SDK constructor
        json.dumps({"code": status * 100000, "message": message}),
        requests.HTTPError(response=response),
    )


def bar_payload(
    start: datetime, price: float, *, volume: float = 100.0, spread: float = 0.05
) -> dict[str, Any]:
    """An API-shaped bar (``t,o,h,l,c,v,n,vw``) starting at ``start`` (UTC)."""
    return {
        "t": start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "o": price,
        "h": round(price + spread, 4),
        "l": round(price - spread, 4),
        "c": price,
        "v": volume,
        "n": 3,
        "vw": price,
    }


def minute_payloads(
    start: datetime, minutes: int, *, price: float = 500.0, step: float = 0.01
) -> list[dict[str, Any]]:
    """``minutes`` consecutive 1-minute bar payloads from ``start``."""
    return [
        bar_payload(start + timedelta(minutes=i), round(price + i * step, 4))
        for i in range(minutes)
    ]


def _naive_utc(payload: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(payload["t"]).replace("Z", "+00:00")).replace(tzinfo=None)


class FakeBarsClient:
    """``StockBarsClient`` double over API-shaped payloads keyed by (symbol, timeframe).

    Like the API, ``start`` and ``end`` are inclusive. ``failures`` are raised, in order,
    by the first calls (``None`` entries mean "succeed").
    """

    def __init__(
        self,
        payloads: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
        *,
        failures: Iterable[BaseException | None] = (),
        result: object | None = None,
    ) -> None:
        self._payloads = payloads
        self._failures = list(failures)
        self._result = result
        self.requests: list[StockBarsRequest] = []

    def get_stock_bars(self, request_params: StockBarsRequest) -> Any:
        self.requests.append(request_params)
        if self._failures:
            failure = self._failures.pop(0)
            if failure is not None:
                raise failure
        if self._result is not None:
            return self._result
        symbol = str(request_params.symbol_or_symbols)
        timeframe = request_params.timeframe.value
        start, end = request_params.start, request_params.end
        assert start is not None
        assert end is not None
        assert start.tzinfo is None
        assert end.tzinfo is None
        rows = [
            p for p in self._payloads.get((symbol, timeframe), []) if start <= _naive_utc(p) <= end
        ]
        return BarSet({symbol: rows} if rows else {})


def calendar_payload(
    day: date, open_hhmm: str = "09:30", close_hhmm: str = "16:00"
) -> dict[str, str]:
    """An API-shaped calendar day (New York wall times)."""
    return {"date": day.isoformat(), "open": open_hhmm, "close": close_hhmm}


def clock_payload(
    timestamp: str = "2026-10-03T10:00:00.123456789-04:00",
    *,
    is_open: bool = False,
    next_open: str = "2026-10-05T09:30:00-04:00",
    next_close: str = "2026-10-05T16:00:00-04:00",
) -> dict[str, Any]:
    """An API-shaped ``GET /v2/clock`` body (New York offsets, nanosecond timestamp)."""
    return {
        "timestamp": timestamp,
        "is_open": is_open,
        "next_open": next_open,
        "next_close": next_close,
    }


class FakeCalendarClient:
    """``CalendarClient`` double returning real ``Calendar`` / ``Clock`` models (inclusive
    range). ``failures`` are raised, in order, by the first calls of either method."""

    def __init__(
        self,
        days: Sequence[dict[str, str]],
        *,
        failures: Iterable[BaseException | None] = (),
        clock: dict[str, Any] | None = None,
    ) -> None:
        self._days = list(days)
        self._failures = list(failures)
        self._clock = clock if clock is not None else clock_payload()
        self.requests: list[GetCalendarRequest | None] = []
        self.clock_calls = 0

    def _maybe_fail(self) -> None:
        if self._failures:
            failure = self._failures.pop(0)
            if failure is not None:
                raise failure

    def get_clock(self) -> Any:
        self.clock_calls += 1
        self._maybe_fail()
        return Clock(**self._clock)

    def get_calendar(self, filters: GetCalendarRequest | None = None) -> Any:
        self.requests.append(filters)
        self._maybe_fail()
        result = [Calendar(**dict(d)) for d in self._days]
        if filters is not None:
            result = [
                c
                for c in result
                if (filters.start is None or c.date >= filters.start)
                and (filters.end is None or c.date <= filters.end)
            ]
        return result


class SleepRecorder:
    """Async ``sleep`` double that records the requested delays and never waits."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def weekdays(start: date, end: date, *, holidays: Iterable[date] = ()) -> list[date]:
    """Weekdays in ``[start, end]`` minus ``holidays``."""
    skip = set(holidays)
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in skip:
            days.append(current)
        current += timedelta(days=1)
    return days
