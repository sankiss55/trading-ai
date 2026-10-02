"""backtest.data.download_alpaca_bars over fake alpaca-py clients (no network)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from adapters.alpaca import AlpacaCalendar, AlpacaMarketData, BarAdjustment, RetryPolicy
from backtest.data import (
    BARS_DIR,
    CALENDAR_FILE,
    DOWNLOAD_CACHE_DIR,
    DownloadSpec,
    download_alpaca_bars,
    load_backtest_data,
    main,
)
from domain.errors import NonRetryableError
from domain.models import DataFeed, Timeframe
from tests.unit.alpaca.fakes import (
    FAKE_KEY,
    FAKE_SECRET,
    FakeBarsClient,
    FakeCalendarClient,
    SleepRecorder,
    api_error,
    bar_payload,
    calendar_payload,
    minute_payloads,
    weekdays,
)

HOLIDAYS = {date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4)}
TRADING_DAYS = weekdays(date(2025, 4, 1), date(2025, 7, 31), holidays=HOLIDAYS)
EARLY_CLOSE = date(2025, 7, 3)
START, END = date(2025, 6, 27), date(2025, 7, 3)  # spans two months, ends on an early close
SYMBOLS = ("SPY", "QQQ")


def _calendar_days() -> list[dict[str, str]]:
    return [
        calendar_payload(d, close_hhmm="13:00" if d == EARLY_CLOSE else "16:00")
        for d in TRADING_DAYS
    ]


def _payloads() -> dict[tuple[str, str], list[dict[str, Any]]]:
    payloads: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, symbol in enumerate(SYMBOLS):
        minutes: list[dict[str, Any]] = []
        daily: list[dict[str, Any]] = []
        base = 500.0 + 100 * index
        for day in TRADING_DAYS:
            open_utc = datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC)  # EDT
            length = 210 if day == EARLY_CLOSE else 390
            minutes.append(bar_payload(open_utc - timedelta(minutes=90), base))  # pre-market
            minutes.extend(minute_payloads(open_utc, length, price=base))
            minutes.append(bar_payload(open_utc + timedelta(minutes=length + 5), base))  # post
            daily.append(
                bar_payload(
                    datetime(day.year, day.month, day.day, 4, 0, tzinfo=UTC), base, volume=1e6
                )
            )
        payloads[(symbol, "1Min")] = minutes
        payloads[(symbol, "1Day")] = daily
    return payloads


def _spec(**overrides: Any) -> DownloadSpec:
    values: dict[str, Any] = {
        "symbols": SYMBOLS,
        "start": START,
        "end": END,
        "feed": DataFeed.IEX,
        "daily_feed": DataFeed.SIP,
        "adjustment": BarAdjustment.SPLIT,
        "daily_warmup_days": 20,
    }
    values.update(overrides)
    return DownloadSpec(**values)


def _sources(
    bars_client: FakeBarsClient, *, daily_feed: DataFeed = DataFeed.SIP
) -> tuple[AlpacaMarketData, AlpacaMarketData, AlpacaCalendar]:
    """Minute (IEX) and daily (``daily_feed``) market data over ONE recording client."""
    policy = RetryPolicy(max_attempts=2, base_delay_seconds=0.0)

    def market_data(feed: DataFeed) -> AlpacaMarketData:
        return AlpacaMarketData(
            bars_client,
            feed=feed,
            adjustment=BarAdjustment.SPLIT,
            retry_policy=policy,
            sleep=SleepRecorder(),
        )

    calendar = AlpacaCalendar(
        FakeCalendarClient(_calendar_days()), retry_policy=policy, sleep=SleepRecorder()
    )
    return market_data(DataFeed.IEX), market_data(daily_feed), calendar


async def _download(
    directory: Path,
    bars_client: FakeBarsClient,
    progress: list[str] | None = None,
    **spec_overrides: Any,
) -> Any:
    market_data, daily_market_data, calendar = _sources(bars_client)
    lines = progress if progress is not None else []
    return await download_alpaca_bars(
        directory,
        _spec(**spec_overrides),
        market_data=market_data,
        daily_market_data=daily_market_data,
        calendar=calendar,
        progress=lines.append,
    )


def _files(directory: Path) -> dict[str, bytes]:
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in [directory / CALENDAR_FILE, *sorted((directory / BARS_DIR).glob("*.csv"))]
    }


async def test_download_writes_a_dataset_the_backtest_loads(tmp_path: Path) -> None:
    progress: list[str] = []
    result = await _download(tmp_path, FakeBarsClient(_payloads()), progress)
    assert set(_files(tmp_path)) == {
        CALENDAR_FILE,
        f"{BARS_DIR}/SPY_1Min.csv",
        f"{BARS_DIR}/SPY_1Day.csv",
        f"{BARS_DIR}/QQQ_1Min.csv",
        f"{BARS_DIR}/QQQ_1Day.csv",
    }
    assert (tmp_path / "manifest.json").is_file()
    data = load_backtest_data(tmp_path, feed=DataFeed.IEX)
    expected_days = [d for d in TRADING_DAYS if START <= d <= END]
    assert [s.session_date for s in data.sessions] == expected_days
    assert data.sessions == result.sessions
    early = data.sessions[-1]
    assert early.is_early_close
    assert early.close_utc == datetime(2025, 7, 3, 17, 0, tzinfo=UTC)
    for session in data.sessions:
        bars = await data.feed.get_minute_bars("SPY", session.open_utc, session.close_utc)
        assert len(bars) == (210 if session.is_early_close else 390)
        assert bars[0].bar_start_utc == session.open_utc  # 13:30Z in summer
    all_minutes = await data.feed.get_minute_bars(
        "SPY", data.sessions[0].open_utc - timedelta(days=1), early.close_utc + timedelta(days=1)
    )
    assert len(all_minutes) == 4 * 390 + 210  # pre/post-market bars were dropped
    assert result.rows["SPY_1Min.csv"] == len(all_minutes)
    # Daily bars cover the 20-session volume lookback before the first session.
    daily = await data.feed.get_daily_bars("SPY", date(2025, 1, 1), END)
    assert all(b.timeframe is Timeframe.DAY_1 for b in daily)
    assert len(daily) == 20 + len(expected_days)
    assert result.daily_start == daily[0].bar_start_utc.date()
    assert [d for d in TRADING_DAYS if result.daily_start <= d < START][-1] == date(2025, 6, 26)
    assert result.downloaded_chunks == len(SYMBOLS) * (2 + 1)  # 2 months + 1 year
    assert any(line.startswith("SPY 1Min 2025-06:") for line in progress)


async def test_download_is_deterministic(tmp_path: Path) -> None:
    await _download(tmp_path / "a", FakeBarsClient(_payloads()))
    await _download(tmp_path / "b", FakeBarsClient(_payloads()))
    assert _files(tmp_path / "a") == _files(tmp_path / "b")


async def test_interrupted_download_resumes_from_completed_chunks(tmp_path: Path) -> None:
    # SPY June succeeds, SPY July fails permanently: the run aborts.
    failing = FakeBarsClient(_payloads(), failures=[None, api_error(403)])
    with pytest.raises(NonRetryableError):
        await _download(tmp_path, failing)
    june = tmp_path / DOWNLOAD_CACHE_DIR / "chunks" / "SPY" / "1Min" / "2025-06.csv"
    assert june.is_file()
    assert not (tmp_path / CALENDAR_FILE).exists()
    assert not list((tmp_path / DOWNLOAD_CACHE_DIR).rglob("*.tmp"))

    resumed = FakeBarsClient(_payloads())
    progress: list[str] = []
    result = await _download(tmp_path, resumed, progress)
    assert "SPY 1Min 2025-06: cached" in progress
    assert result.cached_chunks == 1
    assert result.downloaded_chunks == len(SYMBOLS) * 3 - 1
    requested = {(str(r.symbol_or_symbols), r.timeframe.value, r.start) for r in resumed.requests}
    assert (
        "SPY",
        "1Min",
        datetime(2025, 6, 27, 13, 30, tzinfo=UTC).replace(tzinfo=None),
    ) not in requested

    fresh = tmp_path / "fresh"
    await _download(fresh, FakeBarsClient(_payloads()))
    assert _files(tmp_path) == _files(fresh)


async def test_resume_with_different_parameters_is_refused(tmp_path: Path) -> None:
    await _download(tmp_path, FakeBarsClient(_payloads()))
    with pytest.raises(NonRetryableError) as info:
        await _download(tmp_path, FakeBarsClient(_payloads()), adjustment=BarAdjustment.ALL)
    assert info.value.code == "DOWNLOAD_MANIFEST_MISMATCH"


async def test_no_sessions_in_range_fails(tmp_path: Path) -> None:
    with pytest.raises(NonRetryableError) as info:
        await _download(tmp_path, FakeBarsClient({}), start=date(2025, 7, 4), end=date(2025, 7, 4))
    assert info.value.code == "NO_SESSIONS"


# --------------------------------------------------------------------------- two feeds


async def test_minute_bars_use_the_strategy_feed_and_daily_bars_the_liquidity_feed(
    tmp_path: Path,
) -> None:
    client = FakeBarsClient(_payloads())
    await _download(tmp_path, client)
    feeds = {(r.timeframe.value, str(r.feed.value) if r.feed else None) for r in client.requests}
    assert feeds == {("1Min", "iex"), ("1Day", "sip")}


async def test_manifest_records_both_feeds(tmp_path: Path) -> None:
    await _download(tmp_path, FakeBarsClient(_payloads()))
    cache = json.loads((tmp_path / DOWNLOAD_CACHE_DIR / "manifest.json").read_text("utf-8"))
    dataset = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert cache == dataset
    assert dataset["feeds"] == {"1Min": "iex", "1Day": "sip"}
    assert dataset["version"] == 2
    assert "feed" not in dataset


async def test_resume_with_a_different_daily_feed_is_refused(tmp_path: Path) -> None:
    await _download(tmp_path, FakeBarsClient(_payloads()))
    with pytest.raises(NonRetryableError) as info:
        await _download(tmp_path, FakeBarsClient(_payloads()), daily_feed=DataFeed.IEX)
    assert info.value.code == "DOWNLOAD_MANIFEST_MISMATCH"


async def test_daily_source_with_the_wrong_feed_is_refused(tmp_path: Path) -> None:
    market_data, wrong_daily, calendar = _sources(
        FakeBarsClient(_payloads()), daily_feed=DataFeed.IEX
    )
    with pytest.raises(NonRetryableError) as info:
        await download_alpaca_bars(
            tmp_path,
            _spec(),
            market_data=market_data,
            daily_market_data=wrong_daily,
            calendar=calendar,
            progress=lambda _line: None,
        )
    assert info.value.code == "FEED_MISMATCH"
    assert not (tmp_path / "manifest.json").exists()


async def test_loader_tags_daily_bars_sip_and_minute_bars_iex(tmp_path: Path) -> None:
    await _download(tmp_path, FakeBarsClient(_payloads()))
    data = load_backtest_data(tmp_path, feed=DataFeed.IEX)
    session = data.sessions[0]
    minutes = await data.feed.get_minute_bars("SPY", session.open_utc, session.close_utc)
    daily = await data.feed.get_daily_bars("SPY", date(2025, 1, 1), END)
    assert minutes
    assert daily
    assert {b.feed for b in minutes} == {DataFeed.IEX}
    assert {b.feed for b in daily} == {DataFeed.SIP}
    streamed = [b async for b in data.feed.stream_minute_bars(["SPY"])]
    assert {(b.timeframe, b.feed) for b in streamed} == {(Timeframe.MIN_1, DataFeed.IEX)}


async def test_loader_refuses_a_strategy_feed_other_than_the_minute_feed(
    tmp_path: Path,
) -> None:
    await _download(tmp_path, FakeBarsClient(_payloads()))
    with pytest.raises(NonRetryableError) as info:
        load_backtest_data(tmp_path, feed=DataFeed.SIP)
    assert info.value.code == "DATASET_FEED_MISMATCH"


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"version": 1, "feed": "iex"}',
        '{"version": 2, "feeds": {"1Min": "iex"}}',
        '{"version": 2, "feeds": {"1Min": "iex", "1Day": "nasdaq"}}',
    ],
)
async def test_loader_rejects_an_invalid_dataset_manifest(tmp_path: Path, content: str) -> None:
    await _download(tmp_path, FakeBarsClient(_payloads()))
    (tmp_path / "manifest.json").write_text(content, encoding="utf-8")
    with pytest.raises(NonRetryableError) as info:
        load_backtest_data(tmp_path, feed=DataFeed.IEX)
    assert info.value.code == "INVALID_BAR_DATA"


async def test_legacy_dataset_without_manifest_uses_one_feed(tmp_path: Path) -> None:
    await _download(tmp_path, FakeBarsClient(_payloads()))
    (tmp_path / "manifest.json").unlink()
    data = load_backtest_data(tmp_path, feed=DataFeed.IEX)
    session = data.sessions[0]
    minutes = await data.feed.get_minute_bars("SPY", session.open_utc, session.close_utc)
    daily = await data.feed.get_daily_bars("SPY", date(2025, 1, 1), END)
    assert {b.feed for b in [*minutes, *daily]} == {DataFeed.IEX}


@pytest.mark.parametrize(
    "overrides",
    [
        {"symbols": ()},
        {"symbols": ("SPY", "SPY")},
        {"symbols": ("spy",)},
        {"end": date(2025, 6, 1)},
        {"daily_warmup_days": -1},
    ],
)
def test_invalid_spec_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(NonRetryableError) as info:
        _spec(**overrides)
    assert info.value.code == "INVALID_DOWNLOAD_SPEC"


# --------------------------------------------------------------------------- CLI


def test_download_cli_requires_adjustment(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["download", "--symbols", "SPY", "--start", "2025-06-02", "--end", "2025-06-03"]
    argv += ["--feed", "iex", "--daily-feed", "sip", "--daily-warmup-days", "20", "--out", "x"]
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    assert "--adjustment" in capsys.readouterr().err


def test_download_cli_requires_daily_feed(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["download", "--symbols", "SPY", "--start", "2025-06-02", "--end", "2025-06-03"]
    argv += ["--feed", "iex", "--adjustment", "split", "--daily-warmup-days", "20", "--out", "x"]
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    assert "--daily-feed" in capsys.readouterr().err


def test_download_cli_help_states_the_owner_decision(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["download", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "OWNER_DECISION" in text
    assert "live and backtest MUST use the same convention" in text


def test_download_cli_refuses_live_before_any_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("APP_ENV", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        f"APP_ENV=live\nALPACA_API_KEY={FAKE_KEY}\nALPACA_SECRET_KEY={FAKE_SECRET}\n",
        encoding="utf-8",
    )
    argv = ["download", "--symbols", "SPY", "--start", "2025-06-02", "--end", "2025-06-03"]
    argv += ["--feed", "iex", "--daily-feed", "sip", "--adjustment", "split"]
    argv += ["--daily-warmup-days", "20", "--out", str(tmp_path / "out"), "--env-file", str(env)]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "LIVE_BLOCKED" in err
    assert FAKE_KEY not in err
    assert FAKE_SECRET not in err


def test_synthetic_cli_keeps_the_legacy_invocation(tmp_path: Path) -> None:
    common = ["--symbols", "SYNTH", "--start", "2025-01-02", "--end", "2025-01-03", "--seed", "3"]
    assert main(["--out", str(tmp_path / "legacy"), *common]) == 0
    assert main(["synthetic", "--out", str(tmp_path / "sub"), *common]) == 0
    legacy = {p.name: p.read_bytes() for p in (tmp_path / "legacy").rglob("*.csv")}
    sub = {p.name: p.read_bytes() for p in (tmp_path / "sub").rglob("*.csv")}
    assert legacy == sub
    assert len(legacy) == 3
