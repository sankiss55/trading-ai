"""Unit tests for ``ScriptedAIFilter``: scripted, in order, never raising."""

from __future__ import annotations

from decimal import Decimal

from adapters.simulation.scripted_ai_filter import (
    SCRIPT_EXHAUSTED,
    ScriptedAIFilter,
    always_approve,
    approve_result,
    unavailable_result,
    veto_result,
)
from domain.models import (
    AIValidity,
    AIVerdictKind,
    AIVerdictResult,
    Snapshot,
    SnapshotMarketContext,
    SnapshotPayload,
    SnapshotPortfolioContext,
    SnapshotProposedTrade,
    SnapshotSession,
    SnapshotTimeframe,
    Timeframe,
)
from domain.ports import IAIFilter
from tests.unit.simulation_adapters.builders import T0


def snapshot(signal_id: str = "sig-1") -> Snapshot:
    payload = SnapshotPayload(
        snapshot_version="1",
        signal_id=signal_id,
        symbol="SPY",
        timestamp_utc=T0,
        session=SnapshotSession(
            minutes_since_open=30, minutes_to_close=360, is_early_close_day=False
        ),
        primary_timeframe=SnapshotTimeframe(timeframe=Timeframe.MIN_5, bars=()),
        confirmation_timeframe=None,
        rule_results=(),
        proposed_trade=SnapshotProposedTrade(
            entry_ref=Decimal(100),
            stop_price=Decimal(98),
            take_profit_price=Decimal(104),
            r_multiple=Decimal(2),
            risk_pct_of_equity=Decimal("0.5"),
        ),
        market_context=SnapshotMarketContext(
            benchmark_symbol="SPY", benchmark_change_today_pct=None, benchmark_above_ema_fast=None
        ),
        portfolio_context=SnapshotPortfolioContext(
            open_positions=0, daily_pnl_pct=Decimal(0), trades_today=0
        ),
    )
    return Snapshot(payload=payload, snapshot_hash="0" * 64)


async def test_sequence_is_returned_in_order_then_unavailable() -> None:
    ai: IAIFilter = ScriptedAIFilter([approve_result("sig-1"), veto_result("sig-2")])
    first = await ai.evaluate(snapshot("sig-1"))
    second = await ai.evaluate(snapshot("sig-2"))
    third = await ai.evaluate(snapshot("sig-3"))
    assert first.verdict is not None
    assert first.verdict.verdict is AIVerdictKind.APPROVE
    assert second.verdict is not None
    assert second.verdict.verdict is AIVerdictKind.VETO
    assert third.validity is AIValidity.UNAVAILABLE
    assert third.invalid_reason == SCRIPT_EXHAUSTED


async def test_custom_exhausted_result() -> None:
    fallback = unavailable_result("TIMEOUT")
    ai = ScriptedAIFilter([], exhausted_result=fallback)
    assert await ai.evaluate(snapshot()) == fallback


async def test_callable_script_and_call_recording() -> None:
    ai = ScriptedAIFilter(always_approve)
    snap = snapshot("sig-42")
    result = await ai.evaluate(snap)
    assert result.validity is AIValidity.VALID
    assert result.verdict is not None
    assert result.verdict.signal_id == "sig-42"
    assert ai.calls == (snap,)


async def test_failing_callable_is_reported_as_unavailable() -> None:
    def broken(_: Snapshot) -> AIVerdictResult:
        raise RuntimeError("boom")

    result = await ScriptedAIFilter(broken).evaluate(snapshot())
    assert result.validity is AIValidity.UNAVAILABLE
    assert result.invalid_reason == "SCRIPT_FAILED: RuntimeError"
