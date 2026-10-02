"""Deterministic signal ids and signal construction (sec. 13.7)."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta, timezone

import pytest

from domain.models import RuleResult, Timeframe
from domain.strategy.rules import StrategyInputError
from domain.strategy.signals import build_signal, canonical_signal_key, signal_id_for

START = datetime(2026, 6, 15, 13, 30, tzinfo=UTC)
END = START + timedelta(minutes=5)


def _id(version: str = "1.0.0", symbol: str = "SPY", start: datetime = START) -> str:
    return signal_id_for(
        strategy_version=version, symbol=symbol, timeframe=Timeframe.MIN_5, bar_start_utc=start
    )


def test_canonical_key_format() -> None:
    key = canonical_signal_key(
        strategy_version="1.0.0", symbol="SPY", timeframe=Timeframe.MIN_5, bar_start_utc=START
    )
    assert key == "1.0.0|SPY|5Min|2026-06-15T13:30:00+00:00|BUY"


def test_signal_id_is_prefixed_sha256_of_canonical_key() -> None:
    expected = hashlib.sha256(b"1.0.0|SPY|5Min|2026-06-15T13:30:00+00:00|BUY").hexdigest()
    assert _id() == "sig_" + expected
    assert re.fullmatch(r"sig_[0-9a-f]{64}", _id())


def test_signal_id_is_deterministic_and_discriminating() -> None:
    assert _id() == _id()
    assert _id(version="1.0.1") != _id()
    assert _id(symbol="QQQ") != _id()
    assert _id(start=START + timedelta(minutes=5)) != _id()
    other_tf = signal_id_for(
        strategy_version="1.0.0", symbol="SPY", timeframe=Timeframe.MIN_15, bar_start_utc=START
    )
    assert other_tf != _id()


def test_same_instant_in_other_offset_gives_same_id() -> None:
    eastern = START.astimezone(timezone(timedelta(hours=-4)))
    assert _id(start=eastern) == _id()


def test_rejects_naive_datetime_and_empty_version() -> None:
    with pytest.raises(StrategyInputError):
        _id(start=datetime(2026, 6, 15, 13, 30))  # noqa: DTZ001 - naive on purpose
    with pytest.raises(StrategyInputError):
        _id(version="")


def test_build_signal_fields() -> None:
    created = END + timedelta(seconds=2)
    results = (RuleResult(rule_id="ENTRY_TREND_01", result=True, values={"x": 1}),)
    signal = build_signal(
        strategy_version="1.0.0",
        symbol="SPY",
        timeframe=Timeframe.MIN_5,
        bar_start_utc=START,
        bar_end_utc=END,
        created_at_utc=created,
        signal_ttl_seconds=120,
        rule_results=results,
    )
    assert signal.signal_id == _id()
    assert signal.expires_at_utc == END + timedelta(seconds=120)
    assert signal.created_at_utc == created
    assert signal.rule_results == results
    assert signal.strategy_version == "1.0.0"
    assert (signal.bar_start_utc, signal.bar_end_utc) == (START, END)


@pytest.mark.parametrize("ttl", [0, -1])
def test_build_signal_rejects_non_positive_ttl(ttl: int) -> None:
    with pytest.raises(StrategyInputError):
        build_signal(
            strategy_version="1.0.0",
            symbol="SPY",
            timeframe=Timeframe.MIN_5,
            bar_start_utc=START,
            bar_end_utc=END,
            created_at_utc=END,
            signal_ttl_seconds=ttl,
            rule_results=(),
        )


def test_build_signal_explicit_expiry_for_daily_bars() -> None:
    """Daily bars expire at the next session open + TTL (passed in by the flow)."""
    next_open = END + timedelta(hours=17, minutes=30)
    signal = build_signal(
        strategy_version="1.0.0",
        symbol="SPY",
        timeframe=Timeframe.DAY_1,
        bar_start_utc=START,
        bar_end_utc=END,
        created_at_utc=END + timedelta(seconds=11),
        signal_ttl_seconds=120,
        rule_results=(),
        expires_at_utc=next_open + timedelta(seconds=120),
    )
    assert signal.expires_at_utc == next_open + timedelta(seconds=120)
    assert signal.signal_id == signal_id_for(
        strategy_version="1.0.0", symbol="SPY", timeframe=Timeframe.DAY_1, bar_start_utc=START
    )


def test_build_signal_rejects_an_expiry_before_the_bar_end() -> None:
    with pytest.raises(StrategyInputError, match="expires_at_utc"):
        build_signal(
            strategy_version="1.0.0",
            symbol="SPY",
            timeframe=Timeframe.DAY_1,
            bar_start_utc=START,
            bar_end_utc=END,
            created_at_utc=END,
            signal_ttl_seconds=120,
            rule_results=(),
            expires_at_utc=END - timedelta(seconds=1),
        )
