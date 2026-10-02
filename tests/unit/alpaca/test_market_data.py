"""AlpacaMarketData: SDK -> domain conversion, chunking, retries, error translation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
import requests
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.models import Bar as SdkBar
from pydantic import SecretStr

from adapters.alpaca import (
    AlpacaCredentials,
    AlpacaMarketData,
    BarAdjustment,
    RequestPacer,
    RetryPolicy,
)
from adapters.alpaca._http import HttpTimeouts, _TimeoutAdapter, install_timeouts
from adapters.alpaca.market_data import convert_bar, month_chunks
from domain.errors import NonRetryableError, RetryableError
from domain.models import BarStatus, DataFeed, Timeframe
from tests.unit.alpaca.fakes import (
    FAKE_KEY,
    FAKE_SECRET,
    FakeBarsClient,
    SleepRecorder,
    api_error,
    bar_payload,
    minute_payloads,
)

OPEN = datetime(2025, 6, 2, 13, 30, tzinfo=UTC)  # 09:30 America/New_York (EDT)
NO_JITTER = RetryPolicy(max_attempts=4, base_delay_seconds=1.0, max_delay_seconds=3.0)


def _adapter(
    client: FakeBarsClient,
    *,
    policy: RetryPolicy = NO_JITTER,
    sleep: SleepRecorder | None = None,
    adjustment: BarAdjustment = BarAdjustment.SPLIT,
) -> AlpacaMarketData:
    return AlpacaMarketData(
        client,
        feed=DataFeed.IEX,
        adjustment=adjustment,
        retry_policy=policy,
        sleep=sleep or SleepRecorder(),
        unit_random=lambda: 0.0,
    )


# --------------------------------------------------------------------------- conversion


def test_convert_minute_bar_uses_sdk_timestamp_as_bar_start() -> None:
    sdk = SdkBar("SPY", bar_payload(OPEN, 590.015, volume=1234.0))
    bar = convert_bar(sdk, symbol="SPY", timeframe=Timeframe.MIN_1, feed=DataFeed.IEX)
    # Alpaca labels a bar with the inclusive START of its interval (sec. 10.3.2).
    assert bar.bar_start_utc == OPEN
    assert bar.bar_start_utc.tzinfo is not None
    assert bar.bar_start_utc.utcoffset() == timedelta(0)
    assert bar.bar_end_utc == OPEN + timedelta(minutes=1)
    assert bar.open == Decimal("590.015")  # Decimal(str(float)): no binary noise
    assert bar.high == Decimal("590.065")
    assert bar.low == Decimal("589.965")
    assert isinstance(bar.close, Decimal)
    assert bar.volume == 1234
    assert isinstance(bar.volume, int)
    assert bar.status is BarStatus.COMPLETE
    assert bar.feed is DataFeed.IEX
    assert bar.timeframe is Timeframe.MIN_1


def test_convert_daily_bar_covers_one_day_from_new_york_midnight() -> None:
    start = datetime(2025, 6, 2, 4, 0, tzinfo=UTC)  # 00:00 America/New_York
    sdk = SdkBar("SPY", bar_payload(start, 590.0, volume=2_000_000.0))
    bar = convert_bar(sdk, symbol="SPY", timeframe=Timeframe.DAY_1, feed=DataFeed.SIP)
    assert bar.bar_start_utc == start
    assert bar.bar_end_utc == start + timedelta(days=1)
    assert bar.feed is DataFeed.SIP


@pytest.mark.parametrize(
    ("payload_update", "match"),
    [
        ({"t": "2025-06-02T13:30:00"}, "timezone-aware"),
        ({"v": 10.5}, "volume is not integral"),
        ({"h": 100.0, "l": 101.0}, "OHLC inconsistent"),
    ],
)
def test_convert_bar_rejects_invalid_data(payload_update: dict[str, object], match: str) -> None:
    payload = {**bar_payload(OPEN, 100.5), **payload_update}
    sdk = SdkBar("SPY", payload)
    with pytest.raises(NonRetryableError, match=match) as info:
        convert_bar(sdk, symbol="SPY", timeframe=Timeframe.MIN_1, feed=DataFeed.IEX)
    assert info.value.code == "INVALID_BAR_DATA"


def test_convert_bar_rejects_symbol_mismatch() -> None:
    sdk = SdkBar("QQQ", bar_payload(OPEN, 100.0))
    with pytest.raises(NonRetryableError, match="symbol"):
        convert_bar(sdk, symbol="SPY", timeframe=Timeframe.MIN_1, feed=DataFeed.IEX)


# --------------------------------------------------------------------------- requests, chunks


def test_month_chunks_split_at_utc_month_boundaries() -> None:
    start = datetime(2025, 5, 30, 13, 30, tzinfo=UTC)
    end = datetime(2025, 7, 2, 20, 0, tzinfo=UTC)
    assert list(month_chunks(start, end)) == [
        (start, datetime(2025, 6, 1, tzinfo=UTC)),
        (datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 7, 1, tzinfo=UTC)),
        (datetime(2025, 7, 1, tzinfo=UTC), end),
    ]
    assert list(month_chunks(end, end)) == []


async def test_minute_bars_request_one_bounded_chunk_per_month() -> None:
    days = [
        datetime(2025, 5, 30, 13, 30, tzinfo=UTC),
        OPEN,
        datetime(2025, 7, 1, 13, 30, tzinfo=UTC),
    ]
    payloads = [p for day in days for p in minute_payloads(day, 3)]
    client = FakeBarsClient({("SPY", "1Min"): payloads})
    adapter = _adapter(client, adjustment=BarAdjustment.ALL)
    bars = await adapter.get_minute_bars("SPY", days[0], days[-1] + timedelta(minutes=3))
    assert [b.bar_start_utc for b in bars] == [
        d + timedelta(minutes=i) for d in days for i in range(3)
    ]
    assert len(client.requests) == 3
    first = client.requests[0]
    assert first.timeframe.value == "1Min"
    assert first.feed is not None
    assert first.feed.value == "iex"
    assert first.adjustment is not None
    assert first.adjustment.value == "all"
    assert first.start == datetime(2025, 5, 30, 13, 30, tzinfo=UTC).replace(
        tzinfo=None
    )  # naive UTC in SDK
    assert first.end == datetime(2025, 6, 1, tzinfo=UTC).replace(tzinfo=None)


async def test_minute_bars_are_fully_inside_the_range() -> None:
    client = FakeBarsClient({("SPY", "1Min"): minute_payloads(OPEN, 10)})
    bars = await _adapter(client).get_minute_bars("SPY", OPEN, OPEN + timedelta(minutes=5))
    # The API end is inclusive: the bar starting AT ``end`` must not be returned.
    assert len(bars) == 5
    assert bars[-1].bar_end_utc == OPEN + timedelta(minutes=5)


async def test_minute_bars_require_aware_datetimes() -> None:
    adapter = _adapter(FakeBarsClient({}))
    with pytest.raises(ValueError, match="timezone-aware"):
        await adapter.get_minute_bars("SPY", datetime(2025, 6, 2), OPEN)  # noqa: DTZ001


async def test_empty_range_makes_no_request() -> None:
    client = FakeBarsClient({})
    assert await _adapter(client).get_minute_bars("SPY", OPEN, OPEN) == []
    assert await _adapter(client).get_daily_bars("SPY", date(2025, 6, 3), date(2025, 6, 2)) == []
    assert client.requests == []


async def test_daily_bars_filter_by_inclusive_dates() -> None:
    payloads = [
        bar_payload(datetime(2025, 6, d, 4, 0, tzinfo=UTC), 500.0 + d, volume=1e6)
        for d in (2, 3, 4)
    ]
    client = FakeBarsClient({("SPY", "1Day"): payloads})
    bars = await _adapter(client).get_daily_bars("SPY", date(2025, 6, 2), date(2025, 6, 3))
    assert [b.bar_start_utc.date() for b in bars] == [date(2025, 6, 2), date(2025, 6, 3)]
    assert all(b.timeframe is Timeframe.DAY_1 and b.volume == 1_000_000 for b in bars)
    assert client.requests[0].timeframe.value == "1Day"


async def test_unexpected_result_type_is_schema_error() -> None:
    client = FakeBarsClient({}, result={"bars": []})
    with pytest.raises(NonRetryableError) as info:
        await _adapter(client).get_minute_bars("SPY", OPEN, OPEN + timedelta(minutes=1))
    assert info.value.code == "INVALID_SCHEMA"


async def test_quote_and_stream_are_not_available_before_phase_3() -> None:
    adapter = _adapter(FakeBarsClient({}))
    assert await adapter.get_latest_quote("SPY") is None
    with pytest.raises(NotImplementedError, match="Phase 3"):
        adapter.stream_minute_bars(["SPY"])


# --------------------------------------------------------------------------- retries


async def test_rate_limit_is_retried_with_backoff_then_succeeds() -> None:
    sleep = SleepRecorder()
    client = FakeBarsClient(
        {("SPY", "1Min"): minute_payloads(OPEN, 2)}, failures=[api_error(429), api_error(429)]
    )
    bars = await _adapter(client, sleep=sleep).get_minute_bars(
        "SPY", OPEN, OPEN + timedelta(minutes=2)
    )
    assert len(bars) == 2
    assert len(client.requests) == 3
    assert sleep.delays == [1.0, 2.0]


async def test_server_errors_give_up_after_bounded_attempts() -> None:
    sleep = SleepRecorder()
    client = FakeBarsClient({}, failures=[api_error(503)] * 10)
    with pytest.raises(RetryableError, match="gave up after 4 attempts") as info:
        await _adapter(client, sleep=sleep).get_minute_bars(
            "SPY", OPEN, OPEN + timedelta(minutes=1)
        )
    assert info.value.code == "ALPACA_SERVER_ERROR"
    assert len(client.requests) == 4  # max_attempts, never more
    assert sleep.delays == [1.0, 2.0, 3.0]  # exponential, capped at max_delay_seconds


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "ALPACA_AUTH"), (403, "ALPACA_AUTH"), (422, "ALPACA_REQUEST_REJECTED")],
)
async def test_client_errors_are_not_retried(status: int, code: str) -> None:
    sleep = SleepRecorder()
    client = FakeBarsClient({}, failures=[api_error(status, "forbidden")])
    with pytest.raises(NonRetryableError) as info:
        await _adapter(client, sleep=sleep).get_minute_bars(
            "SPY", OPEN, OPEN + timedelta(minutes=1)
        )
    assert info.value.code == code
    assert f"HTTP {status}" in str(info.value)
    assert len(client.requests) == 1
    assert sleep.delays == []


@pytest.mark.parametrize(
    "failure",
    [requests.ConnectionError("boom"), requests.Timeout("slow")],
    ids=["connection", "timeout"],
)
async def test_network_errors_are_retryable(failure: Exception) -> None:
    client = FakeBarsClient({("SPY", "1Min"): minute_payloads(OPEN, 1)}, failures=[failure])
    bars = await _adapter(client).get_minute_bars("SPY", OPEN, OPEN + timedelta(minutes=1))
    assert len(bars) == 1
    assert len(client.requests) == 2


async def test_network_error_exhausted_is_translated() -> None:
    client = FakeBarsClient({}, failures=[requests.ConnectionError("https://x?y")] * 4)
    with pytest.raises(RetryableError) as info:
        await _adapter(client).get_minute_bars("SPY", OPEN, OPEN + timedelta(minutes=1))
    assert info.value.code == "NETWORK_ERROR"
    assert "https://x" not in str(info.value)  # only the exception type is reported


def test_retry_policy_validation_and_jitter() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    policy = RetryPolicy(base_delay_seconds=2.0, max_delay_seconds=60.0, jitter_ratio=0.5)
    assert policy.delay(1, 0.0) == 2.0
    assert policy.delay(3, 1.0) == 12.0  # 8 * (1 + 0.5)
    assert policy.delay(10, 0.0) == 60.0


# --------------------------------------------------------------------------- real client wiring


def test_from_credentials_builds_sdk_client_with_timeouts_and_masked_keys() -> None:
    credentials = AlpacaCredentials(api_key=SecretStr(FAKE_KEY), secret_key=SecretStr(FAKE_SECRET))
    adapter = AlpacaMarketData.from_credentials(
        credentials, feed=DataFeed.IEX, adjustment=BarAdjustment.RAW
    )
    client = adapter._client
    assert isinstance(client, StockHistoricalDataClient)
    session = client._session
    assert isinstance(session.get_adapter("https://data.alpaca.markets"), _TimeoutAdapter)
    assert FAKE_SECRET not in repr(credentials)
    assert FAKE_KEY not in repr(credentials)
    assert adapter.feed is DataFeed.IEX
    assert adapter.adjustment is BarAdjustment.RAW


def test_install_timeouts_refuses_unknown_sdk_layout() -> None:
    with pytest.raises(NonRetryableError) as info:
        install_timeouts(object(), HttpTimeouts())
    assert info.value.code == "SDK_INCOMPATIBLE"


def test_timeout_adapter_applies_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def fake_send(self: object, request: object, **kwargs: object) -> requests.Response:
        seen.append(kwargs["timeout"])
        response = requests.Response()
        response.status_code = 200
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    client = StockHistoricalDataClient(api_key=FAKE_KEY, secret_key=FAKE_SECRET)
    install_timeouts(client, HttpTimeouts(connect_seconds=3.0, read_seconds=7.0))
    client._session.get("https://data.alpaca.markets/v2/ping")  # no network: send is faked
    client._session.get("https://data.alpaca.markets/v2/ping", timeout=1.0)
    assert seen == [(3.0, 7.0), 1.0]


async def test_request_pacer_spaces_requests() -> None:
    now = [100.0]
    sleep = SleepRecorder()
    pacer = RequestPacer(0.3, monotonic=lambda: now[0], sleep=sleep)
    await pacer.wait()
    now[0] += 0.1
    await pacer.wait()
    now[0] += 1.0
    await pacer.wait()
    assert sleep.delays == [pytest.approx(0.2)]
