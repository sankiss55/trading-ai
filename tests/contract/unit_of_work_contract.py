"""Reusable ``IUnitOfWork`` contract (sec. 8.5, 20, 21.1, 22, 31.1, 38, 49.2).

Subclass ``UnitOfWorkContract`` in a ``test_*.py`` module and provide a ``uow_harness``
fixture returning a ``UnitOfWorkHarness``. Every unit of work the harness creates must
work on the SAME empty database (like new connections to one file).

The builders below (``make_signal`` ...) are shared with the adapter unit tests.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

import pytest

from domain.errors import NonRetryableError, RetryableError, StateCriticalError
from domain.models import (
    AIDecisionRecord,
    AIMode,
    AIReasonCode,
    AIRiskFlag,
    AIUsage,
    AIValidity,
    AIVerdict,
    AIVerdictKind,
    AIVerdictResult,
    BarStatus,
    BracketOrderRequest,
    BrokerOrderStatus,
    CheckResult,
    EquitySnapshot,
    EquitySnapshotKind,
    ExitReason,
    Fill,
    OrderEvent,
    OrderRecord,
    OrderRole,
    OrderSide,
    OrderType,
    ReconciliationOutcome,
    ReconciliationRecord,
    RiskEvent,
    RuleResult,
    ShadowOutcome,
    Signal,
    SimpleOrderRequest,
    Snapshot,
    SnapshotBar,
    SnapshotMarketContext,
    SnapshotPayload,
    SnapshotPortfolioContext,
    SnapshotProposedTrade,
    SnapshotSession,
    SnapshotTimeframe,
    SystemControl,
    SystemEvent,
    Timeframe,
    TimeInForce,
    Trade,
    TradeState,
    VersionInfo,
)
from domain.ports import IUnitOfWork

T0 = datetime(2026, 10, 1, 13, 35, tzinfo=UTC)
"""09:35 New York; microsecond 0 on purpose (see the ordering tests)."""

SPEC_FINAL_STATES = frozenset(
    {
        TradeState.AI_VETOED,
        TradeState.CLOSED,
        TradeState.REJECTED,
        TradeState.EXPIRED,
        TradeState.CANCELLED,
        TradeState.FAILED,
    }
)
"""Final states written in sec. 21.1, kept here as an independent oracle."""

SHORT_LOCK_TIMEOUT = 0.2


class UnitOfWorkHarness(Protocol):
    """What ``UnitOfWorkContract`` needs from an implementation under test."""

    def new_uow(self, *, lock_timeout_seconds: float = 5.0) -> IUnitOfWork:
        """A new unit of work on the shared database of this test."""
        ...


# --------------------------------------------------------------------------- builders


def make_signal(signal_id: str = "sig-1", *, created_at: datetime = T0) -> Signal:
    """BUY signal with typed rule values (Decimal, float, int, bool, str, None)."""
    return Signal(
        signal_id=signal_id,
        symbol="SPY",
        timeframe=Timeframe.MIN_5,
        bar_start_utc=created_at - timedelta(minutes=5),
        bar_end_utc=created_at,
        created_at_utc=created_at,
        expires_at_utc=created_at + timedelta(minutes=2),
        rule_results=(
            RuleResult(
                rule_id="ENTRY_TREND_01",
                result=True,
                values={
                    "ema_fast": Decimal("501.2500"),
                    "ema_slow": 500.125,
                    "bars": 21,
                    "confirmed": True,
                    "label": "trend-up",
                    "missing": None,
                },
            ),
        ),
        strategy_version="1.0.0",
    )


def make_versions() -> VersionInfo:
    return VersionInfo(
        strategy_version="1.0.0",
        risk_version="1.0.0",
        config_version="2.3.0",
        config_hash="abc123",
        app_version="0.1.0",
    )


def make_trade(
    signal_id: str = "sig-1", *, trade_id: str | None = None, state: TradeState = TradeState.AI_DONE
) -> Trade:
    """Trade with exact decimals that a float would alter."""
    return Trade(
        trade_id=trade_id or f"trade-{signal_id}",
        signal_id=signal_id,
        symbol="SPY",
        qty=7,
        entry_ref=Decimal("501.2500"),
        stop_price=Decimal("499.10"),
        take_profit_price=Decimal("505.55"),
        versions=make_versions(),
        state=state,
    )


def make_bracket(client_order_id: str) -> BracketOrderRequest:
    return BracketOrderRequest(
        symbol="SPY",
        qty=7,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("501.25"),
        time_in_force=TimeInForce.DAY,
        take_profit_limit_price=Decimal("505.55"),
        stop_loss_stop_price=Decimal("499.10"),
        client_order_id=client_order_id,
    )


def make_order(
    client_order_id: str = "paper-sig1-entry",
    *,
    trade_id: str = "trade-sig-1",
    state: TradeState = TradeState.SUBMITTING,
    role: OrderRole = OrderRole.ENTRY,
    order_id: str | None = None,
    simple: bool = False,
) -> OrderRecord:
    request: BracketOrderRequest | SimpleOrderRequest
    if simple:
        request = SimpleOrderRequest(
            symbol="SPY",
            qty=7,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=Decimal("499.10"),
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )
    else:
        request = make_bracket(client_order_id)
    return OrderRecord(
        client_order_id=client_order_id,
        trade_id=trade_id,
        role=role,
        state=state,
        request=request,
        order_id=order_id,
    )


def make_fill(activity_id: str = "act-1", *, order_id: str = "broker-1") -> Fill:
    return Fill(
        order_id=order_id,
        activity_id=activity_id,
        symbol="SPY",
        side=OrderSide.BUY,
        qty=Decimal("7"),
        price=Decimal("501.2731"),
        timestamp_utc=T0 + timedelta(seconds=1, microseconds=250_000),
    )


def make_snapshot(signal_id: str = "sig-1") -> Snapshot:
    bar = SnapshotBar(
        t=T0 - timedelta(minutes=5),
        o=Decimal("500.10"),
        h=Decimal("501.50"),
        l=Decimal("499.90"),
        c=Decimal("501.25"),
        v=12_345,
        status=BarStatus.COMPLETE,
    )
    payload = SnapshotPayload(
        snapshot_version="1",
        signal_id=signal_id,
        symbol="SPY",
        timestamp_utc=T0,
        session=SnapshotSession(
            minutes_since_open=5, minutes_to_close=385, is_early_close_day=False
        ),
        primary_timeframe=SnapshotTimeframe(
            timeframe=Timeframe.MIN_5, bars=(bar,), indicators={"ema_fast": Decimal("501.25")}
        ),
        confirmation_timeframe=None,
        rule_results=make_signal(signal_id).rule_results,
        proposed_trade=SnapshotProposedTrade(
            entry_ref=Decimal("501.25"),
            stop_price=Decimal("499.10"),
            take_profit_price=Decimal("505.55"),
            r_multiple=Decimal("2.0"),
            risk_pct_of_equity=Decimal("0.005"),
        ),
        market_context=SnapshotMarketContext(
            benchmark_symbol="SPY", benchmark_change_today_pct=None, benchmark_above_ema_fast=True
        ),
        portfolio_context=SnapshotPortfolioContext(
            open_positions=0, daily_pnl_pct=Decimal("0"), trades_today=0
        ),
    )
    return Snapshot(payload=payload, snapshot_hash="0" * 64)


def make_ai_decision(
    created_at: datetime, cost: str, *, signal_id: str = "sig-1", valid: bool = True
) -> AIDecisionRecord:
    if valid:
        result = AIVerdictResult(
            validity=AIValidity.VALID,
            verdict=AIVerdict(
                signal_id=signal_id,
                verdict=AIVerdictKind.APPROVE,
                reason_code=AIReasonCode.SIGNAL_CONFIRMED,
                risk_flags=(AIRiskFlag.LATE_IN_SESSION,),
                confidence=0.75,
                rationale="trend and volume confirm",
            ),
            raw_response='{"verdict":"APPROVE"}',
            usage=AIUsage(input_tokens=900, output_tokens=60),
            latency_ms=1_850,
            stop_reason="end_turn",
            model_id="model-x",
        )
    else:
        result = AIVerdictResult(validity=AIValidity.UNAVAILABLE, invalid_reason="timeout")
    return AIDecisionRecord(
        signal_id=signal_id,
        ai_mode=AIMode.SHADOW,
        snapshot=make_snapshot(signal_id),
        result=result,
        model_id="model-x",
        prompt_version="p1",
        estimated_cost_usd=Decimal(cost),
        created_at_utc=created_at,
    )


def make_equity(taken_at: datetime, equity: str) -> EquitySnapshot:
    return EquitySnapshot(
        taken_at_utc=taken_at,
        kind=EquitySnapshotKind.PERIODIC,
        equity=Decimal(equity),
        last_equity=Decimal("100000.00"),
        buying_power=Decimal("200000.00"),
    )


def make_shadow(signal_id: str, recorded_at: datetime, result_r: str | None) -> ShadowOutcome:
    return ShadowOutcome(
        signal_id=signal_id,
        recorded_at_utc=recorded_at,
        validity=AIValidity.VALID,
        verdict=AIVerdictKind.VETO,
        reason_code=AIReasonCode.EXTENDED_MOVE,
        confidence=0.6,
        executed=False,
        simulated=True,
        result_r=None if result_r is None else Decimal(result_r),
    )


def make_order_event(
    client_order_id: str = "paper-sig1-entry", *, broker_event: dict[str, Any] | None = None
) -> OrderEvent:
    return OrderEvent(
        client_order_id=client_order_id,
        occurred_at_utc=T0,
        from_state=None,
        to_state=TradeState.SUBMITTING,
        cause="EXECUTION_GUARD",
        broker_event=broker_event,
    )


def make_risk_event(signal_id: str = "sig-1") -> RiskEvent:
    return RiskEvent(
        signal_id=signal_id,
        occurred_at_utc=T0,
        checks=(CheckResult(passed=True, code="MAX_POSITIONS", detail={"open": 0, "max": 3}),),
        sizing={"qty": 7, "risk_amount": "15.05"},
        decision="APPROVED",
    )


def make_reconciliation() -> ReconciliationRecord:
    return ReconciliationRecord(
        occurred_at_utc=T0,
        outcome=ReconciliationOutcome.RECONCILED_APPLIED,
        differences=({"client_order_id": "paper-sig1-entry", "field": "state"},),
        actions=("APPLIED_BROKER_STATE",),
    )


def make_system_event(event_type: str = "STARTUP") -> SystemEvent:
    return SystemEvent(occurred_at_utc=T0, event_type=event_type, detail={"mode": "BOOTING"})


async def seed_trade(harness: UnitOfWorkHarness, signal_id: str = "sig-1") -> Trade:
    """Commit a signal and its trade; return the trade."""
    trade = make_trade(signal_id)
    async with harness.new_uow() as uow:
        await uow.signals.add(make_signal(signal_id))
        await uow.trades.add(trade)
        await uow.commit()
    return trade


def _code(excinfo: pytest.ExceptionInfo[Any]) -> str | None:
    error = excinfo.value
    assert isinstance(error, NonRetryableError | RetryableError | StateCriticalError)
    return error.code


# --------------------------------------------------------------------------- contract


class UnitOfWorkContract:
    """Behaviors every ``IUnitOfWork`` implementation must have."""

    # ------------------------------------------------------------------ transactions

    async def test_commit_makes_writes_visible_to_a_new_unit_of_work(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.signals.add(make_signal())
            await uow.commit()
        async with uow_harness.new_uow() as other:
            assert await other.signals.exists("sig-1")

    async def test_leaving_without_commit_discards_writes(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.signals.add(make_signal())
            assert await uow.signals.exists("sig-1")  # own writes are visible
        async with uow_harness.new_uow() as other:
            assert not await other.signals.exists("sig-1")

    async def test_rollback_discards_and_the_unit_of_work_stays_usable(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.signals.add(make_signal("sig-1"))
            await uow.rollback()
            assert not await uow.signals.exists("sig-1")
            await uow.signals.add(make_signal("sig-2"))
            await uow.commit()
        async with uow_harness.new_uow() as other:
            assert not await other.signals.exists("sig-1")
            assert await other.signals.exists("sig-2")

    async def test_exception_inside_the_context_rolls_back_and_propagates(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        uow = uow_harness.new_uow()

        async def write_then_fail() -> None:
            async with uow:
                await uow.signals.add(make_signal())
                raise LookupError("boom")

        with pytest.raises(LookupError, match="boom"):
            await write_then_fail()
        async with uow_harness.new_uow() as other:
            assert not await other.signals.exists("sig-1")

    async def test_several_commits_in_one_context(self, uow_harness: UnitOfWorkHarness) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.signals.add(make_signal("sig-1"))
            await uow.commit()
            await uow.signals.add(make_signal("sig-2"))
            await uow.commit()
            await uow.signals.add(make_signal("sig-3"))  # never committed
        async with uow_harness.new_uow() as other:
            assert await other.signals.exists("sig-1")
            assert await other.signals.exists("sig-2")
            assert not await other.signals.exists("sig-3")

    async def test_commit_and_rollback_without_writes_are_noops(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.commit()
            await uow.rollback()
            assert await uow.signals.list_pending() == []

    async def test_operations_outside_the_context_are_refused(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        uow = uow_harness.new_uow()
        with pytest.raises(NonRetryableError) as excinfo:
            await uow.signals.exists("sig-1")
        assert _code(excinfo) == "UOW_NOT_ACTIVE"
        with pytest.raises(NonRetryableError) as excinfo:
            await uow.commit()
        assert _code(excinfo) == "UOW_NOT_ACTIVE"
        async with uow:
            pass
        with pytest.raises(NonRetryableError) as excinfo:
            await uow.control.get()
        assert _code(excinfo) == "UOW_NOT_ACTIVE"

    async def test_nested_entry_of_the_same_instance_is_refused(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        uow = uow_harness.new_uow()
        async with uow:
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.__aenter__()
            assert _code(excinfo) == "UOW_ALREADY_ACTIVE"

    async def test_an_instance_can_be_reused_sequentially(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        uow = uow_harness.new_uow()
        async with uow:
            await uow.signals.add(make_signal("sig-1"))
            await uow.commit()
        async with uow:
            await uow.signals.add(make_signal("sig-2"))
        async with uow:
            assert await uow.signals.exists("sig-1")
            assert not await uow.signals.exists("sig-2")

    async def test_a_failed_operation_keeps_earlier_writes_of_the_transaction(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.signals.add(make_signal("sig-1"))
            with pytest.raises(NonRetryableError):
                await uow.signals.add(make_signal("sig-1"))
            await uow.signals.add(make_signal("sig-2"))
            await uow.commit()
        async with uow_harness.new_uow() as other:
            assert [s.signal_id for s in await other.signals.list_pending()] == ["sig-1", "sig-2"]

    async def test_submitting_is_durable_and_unlocked_before_the_broker_call(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        """Sec. 20.7: SUBMITTING committed, then (while the broker is called) any other
        unit of work, e.g. a restart or reconciliation, sees it without waiting."""
        trade = await seed_trade(uow_harness)
        async with uow_harness.new_uow() as guard:
            await guard.orders.add(make_order(trade_id=trade.trade_id))
            await guard.commit()
            # --- the broker call would happen here; no lock may be held ---
            async with uow_harness.new_uow(lock_timeout_seconds=SHORT_LOCK_TIMEOUT) as observer:
                seen = await observer.orders.get_by_client_id("paper-sig1-entry")
                assert seen is not None
                assert seen.state is TradeState.SUBMITTING
                assert [o.client_order_id for o in await observer.orders.list_non_final()] == [
                    "paper-sig1-entry"
                ]
            await guard.orders.update_status(
                "paper-sig1-entry",
                TradeState.SUBMITTED,
                broker_status=BrokerOrderStatus.ACCEPTED,
                order_id="broker-1",
            )
            await guard.commit()
        async with uow_harness.new_uow() as after:
            stored = await after.orders.get_by_client_id("paper-sig1-entry")
            assert stored is not None
            assert stored.state is TradeState.SUBMITTED
            assert stored.order_id == "broker-1"

    async def test_an_open_write_transaction_blocks_others_until_timeout(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as writer:
            await writer.signals.add(make_signal())
            async with uow_harness.new_uow(lock_timeout_seconds=SHORT_LOCK_TIMEOUT) as other:
                with pytest.raises(RetryableError) as excinfo:
                    await other.signals.exists("sig-1")
                assert _code(excinfo) == "DB_BUSY"
        async with uow_harness.new_uow() as later:
            assert not await later.signals.exists("sig-1")  # uncommitted write is gone

    async def test_concurrent_writers_are_serialized(self, uow_harness: UnitOfWorkHarness) -> None:
        first_wrote = asyncio.Event()

        async def first() -> None:
            async with uow_harness.new_uow() as uow:
                await uow.signals.add(make_signal("sig-1"))
                first_wrote.set()
                await asyncio.sleep(0.05)
                await uow.commit()

        async def second() -> bool:
            await first_wrote.wait()
            async with uow_harness.new_uow() as uow:
                saw_first = await uow.signals.exists("sig-1")  # waits for first's commit
                await uow.signals.add(make_signal("sig-2"))
                await uow.commit()
                return saw_first

        _, saw_first = await asyncio.gather(first(), second())
        assert saw_first
        async with uow_harness.new_uow() as uow:
            assert [s.signal_id for s in await uow.signals.list_pending()] == ["sig-1", "sig-2"]

    # ------------------------------------------------------------------ signals

    async def test_signal_round_trip(self, uow_harness: UnitOfWorkHarness) -> None:
        signal = make_signal()
        async with uow_harness.new_uow() as uow:
            assert await uow.signals.get("sig-1") is None
            assert not await uow.signals.exists("sig-1")
            await uow.signals.add(signal)
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            stored = await uow.signals.get("sig-1")
        assert stored == signal
        assert stored.rule_results[0].values["ema_fast"] == Decimal("501.2500")
        assert str(stored.rule_results[0].values["ema_fast"]) == "501.2500"
        assert stored.created_at_utc.tzinfo is UTC

    async def test_duplicate_signal_is_refused(self, uow_harness: UnitOfWorkHarness) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.signals.add(make_signal())
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.signals.add(make_signal())
        assert _code(excinfo) == "DUPLICATE_KEY"

    async def test_list_pending_excludes_exactly_the_final_states(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.signals.add(make_signal("fresh"))
            for state in TradeState:
                await uow.signals.add(make_signal(f"sig-{state.value}"))
                await uow.signals.mark_status(f"sig-{state.value}", state, reason="TEST")
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            pending = [s.signal_id for s in await uow.signals.list_pending()]
        expected = ["fresh"] + [f"sig-{s.value}" for s in TradeState if s not in SPEC_FINAL_STATES]
        assert pending == expected
        assert "sig-UNKNOWN_SUBMISSION" in pending
        assert "sig-RECOVERY_REQUIRED" in pending

    async def test_mark_status_of_an_unknown_signal_is_refused(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.signals.mark_status("nope", TradeState.REJECTED, "X")
        assert _code(excinfo) == "SIGNAL_NOT_FOUND"

    async def test_free_form_values_are_normalized_like_json(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        """Typed fields are exact; ambiguous ``RuleValue`` items come back as their JSON
        value would validate (a numeric string becomes a Decimal) in every adapter."""
        signal = make_signal().model_copy(
            update={"rule_results": (RuleResult(rule_id="R", result=False, values={"s": "1.5"}),)}
        )
        async with uow_harness.new_uow() as uow:
            await uow.signals.add(signal)
            stored = await uow.signals.get(signal.signal_id)
        assert stored == Signal.model_validate_json(signal.model_dump_json())

    # ------------------------------------------------------------------ trades

    async def test_trade_round_trip_and_queries(self, uow_harness: UnitOfWorkHarness) -> None:
        trade = await seed_trade(uow_harness)
        async with uow_harness.new_uow() as uow:
            assert await uow.trades.get(trade.trade_id) == trade
            assert await uow.trades.get_by_signal("sig-1") == trade
            assert await uow.trades.get("nope") is None
            assert await uow.trades.get_by_signal("nope") is None
            assert await uow.trades.list_open() == [trade]
            stored = await uow.trades.get(trade.trade_id)
        assert stored is not None
        assert str(stored.entry_ref) == "501.2500"
        assert str(stored.stop_price) == "499.10"

    async def test_trade_update_replaces_the_stored_version(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        trade = await seed_trade(uow_harness)
        closed = trade.model_copy(
            update={
                "state": TradeState.CLOSED,
                "filled_qty": Decimal("7"),
                "entry_avg_price": Decimal("501.2731"),
                "entry_filled_at_utc": T0 + timedelta(microseconds=123_456),
                "exit_avg_price": Decimal("505.5500"),
                "exit_filled_at_utc": T0 + timedelta(hours=2),
                "exit_reason": ExitReason.TAKE_PROFIT,
                "gross_pnl": Decimal("29.9383"),
                "net_pnl": Decimal("29.90"),
                "result_r": Decimal("1.98"),
            }
        )
        async with uow_harness.new_uow() as uow:
            await uow.trades.update(closed)
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            stored = await uow.trades.get(trade.trade_id)
            assert await uow.trades.list_open() == []
        assert stored == closed
        assert stored.entry_filled_at_utc == T0 + timedelta(microseconds=123_456)
        assert str(stored.exit_avg_price) == "505.5500"

    async def test_trade_update_errors(self, uow_harness: UnitOfWorkHarness) -> None:
        trade = await seed_trade(uow_harness)
        await seed_trade(uow_harness, "sig-2")
        async with uow_harness.new_uow() as uow:
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.trades.update(make_trade("sig-9"))
            assert _code(excinfo) == "TRADE_NOT_FOUND"
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.trades.update(trade.model_copy(update={"signal_id": "sig-2"}))
            assert _code(excinfo) == "TRADE_IDENTITY_CHANGED"

    async def test_trade_unique_keys_and_signal_reference(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        await seed_trade(uow_harness)
        async with uow_harness.new_uow() as uow:
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.trades.add(make_trade("sig-1", trade_id="other-trade"))
            assert _code(excinfo) == "DUPLICATE_KEY"  # one trade per signal (sec. 22.5)
            await uow.signals.add(make_signal("sig-2"))
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.trades.add(make_trade("sig-2", trade_id="trade-sig-1"))
            assert _code(excinfo) == "DUPLICATE_KEY"
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.trades.add(make_trade("unknown-signal"))
            assert _code(excinfo) == "FOREIGN_KEY_VIOLATION"

    async def test_list_open_excludes_final_trades(self, uow_harness: UnitOfWorkHarness) -> None:
        async with uow_harness.new_uow() as uow:
            for state in TradeState:
                await uow.signals.add(make_signal(state.value))
                await uow.trades.add(make_trade(state.value, state=state))
            await uow.commit()
            open_ids = [t.signal_id for t in await uow.trades.list_open()]
        assert open_ids == [s.value for s in TradeState if s not in SPEC_FINAL_STATES]

    # ------------------------------------------------------------------ orders

    async def test_order_round_trip_for_both_request_types(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        await seed_trade(uow_harness)
        entry = make_order()
        stop = make_order("paper-sig1-protective", role=OrderRole.PROTECTIVE, simple=True)
        async with uow_harness.new_uow() as uow:
            await uow.orders.add(entry)
            await uow.orders.add(stop)
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            assert await uow.orders.get_by_client_id("paper-sig1-entry") == entry
            assert await uow.orders.get_by_client_id("paper-sig1-protective") == stop
            assert await uow.orders.get_by_client_id("nope") is None
            stored = await uow.orders.get_by_client_id("paper-sig1-protective")
        assert stored is not None
        assert isinstance(stored.request, SimpleOrderRequest)

    async def test_duplicate_client_order_id_is_refused(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        await seed_trade(uow_harness)
        async with uow_harness.new_uow() as uow:
            await uow.orders.add(make_order())
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.orders.add(make_order())
            assert _code(excinfo) == "DUPLICATE_KEY"
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.orders.add(make_order(state=TradeState.SIGNAL_CREATED))
        assert _code(excinfo) == "DUPLICATE_KEY"

    async def test_order_requires_a_known_trade(self, uow_harness: UnitOfWorkHarness) -> None:
        async with uow_harness.new_uow() as uow:
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.orders.add(make_order(trade_id="no-such-trade"))
        assert _code(excinfo) == "FOREIGN_KEY_VIOLATION"

    async def test_update_status_sets_only_what_is_given(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        await seed_trade(uow_harness)
        async with uow_harness.new_uow() as uow:
            await uow.orders.add(make_order())
            await uow.orders.update_status(
                "paper-sig1-entry",
                TradeState.SUBMITTED,
                broker_status=BrokerOrderStatus.ACCEPTED,
                order_id="broker-1",
            )
            await uow.orders.update_status("paper-sig1-entry", TradeState.PARTIALLY_FILLED)
            await uow.orders.update_status(
                "paper-sig1-entry",
                TradeState.FILLED,
                broker_status=BrokerOrderStatus.FILLED,
                order_id="broker-1",
            )
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            stored = await uow.orders.get_by_client_id("paper-sig1-entry")
            assert await uow.orders.list_non_final() == [stored]
        assert stored is not None
        assert stored.state is TradeState.FILLED
        assert stored.broker_status is BrokerOrderStatus.FILLED
        assert stored.order_id == "broker-1"
        assert stored.request == make_order().request

    async def test_update_status_keeps_unspecified_fields(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        await seed_trade(uow_harness)
        async with uow_harness.new_uow() as uow:
            await uow.orders.add(make_order(order_id="broker-1"))
            await uow.orders.update_status(
                "paper-sig1-entry", TradeState.SUBMITTED, broker_status=BrokerOrderStatus.NEW
            )
            await uow.orders.update_status("paper-sig1-entry", TradeState.CANCELLED)
            stored = await uow.orders.get_by_client_id("paper-sig1-entry")
            assert await uow.orders.list_non_final() == []
        assert stored is not None
        assert stored.broker_status is BrokerOrderStatus.NEW
        assert stored.order_id == "broker-1"

    async def test_update_status_errors(self, uow_harness: UnitOfWorkHarness) -> None:
        await seed_trade(uow_harness)
        async with uow_harness.new_uow() as uow:
            await uow.orders.add(make_order(order_id="broker-1"))
            await uow.orders.add(make_order("paper-sig1-exit", role=OrderRole.EXIT, simple=True))
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.orders.update_status("nope", TradeState.FILLED)
            assert _code(excinfo) == "ORDER_NOT_FOUND"
            with pytest.raises(StateCriticalError) as mismatch:
                await uow.orders.update_status(
                    "paper-sig1-entry", TradeState.FILLED, order_id="broker-2"
                )
            assert _code(mismatch) == "ORDER_ID_MISMATCH"
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.orders.update_status(
                    "paper-sig1-exit", TradeState.SUBMITTED, order_id="broker-1"
                )
            assert _code(excinfo) == "DUPLICATE_KEY"
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.orders.add(
                    make_order("paper-sig1-tp", role=OrderRole.TAKE_PROFIT, order_id="broker-1")
                )
            assert _code(excinfo) == "DUPLICATE_KEY"
            stored = await uow.orders.get_by_client_id("paper-sig1-entry")
        assert stored is not None
        assert stored.state is TradeState.SUBMITTING  # failed updates changed nothing

    # ------------------------------------------------------------------ append-only logs

    async def test_append_only_records_are_accepted(self, uow_harness: UnitOfWorkHarness) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.order_events.append(make_order_event())
            await uow.order_events.append(
                make_order_event(broker_event={"event": "new", "qty": "7", "nested": [1, None]})
            )
            await uow.risk_events.append(make_risk_event())
            await uow.reconciliations.add(make_reconciliation())
            await uow.system_events.append(make_system_event())
            await uow.ai_decisions.add(make_ai_decision(T0, "0.01", valid=False))
            await uow.commit()

    # ------------------------------------------------------------------ fills

    async def test_fills_are_idempotent_by_activity_id(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            assert await uow.fills.add_if_new(make_fill("act-1"))
            assert not await uow.fills.add_if_new(make_fill("act-1"))
            assert await uow.fills.add_if_new(make_fill("act-2"))
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            assert not await uow.fills.add_if_new(make_fill("act-1"))
            assert not await uow.fills.add_if_new(make_fill("act-2", order_id="other"))
            assert await uow.fills.add_if_new(make_fill("act-3"))
            await uow.rollback()
            assert await uow.fills.add_if_new(make_fill("act-3"))

    # ------------------------------------------------------------------ AI decisions

    async def test_ai_counts_and_costs_use_utc_day_and_month_windows(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        last_instant_of_sep = datetime(2026, 9, 30, 23, 59, 59, 999_999, tzinfo=UTC)
        first_instant_of_oct = datetime(2026, 10, 1, tzinfo=UTC)
        async with uow_harness.new_uow() as uow:
            await uow.ai_decisions.add(make_ai_decision(last_instant_of_sep, "0.5"))
            await uow.ai_decisions.add(make_ai_decision(first_instant_of_oct, "0.1"))
            await uow.ai_decisions.add(make_ai_decision(T0, "0.2", valid=False))
            await uow.ai_decisions.add(make_ai_decision(T0 + timedelta(days=1), "0.000001"))
            await uow.ai_decisions.add(make_ai_decision(datetime(2026, 11, 1, tzinfo=UTC), "7.00"))
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            assert await uow.ai_decisions.count_today(date(2026, 10, 1)) == 2
            assert await uow.ai_decisions.count_today(date(2026, 9, 30)) == 1
            assert await uow.ai_decisions.count_today(date(2026, 10, 3)) == 0
            today = await uow.ai_decisions.cost_today(date(2026, 10, 1))
            assert today == Decimal("0.3")  # exact: 0.1 + 0.2 as decimals
            assert isinstance(today, Decimal)
            assert await uow.ai_decisions.cost_today(date(2026, 10, 3)) == Decimal(0)
            assert await uow.ai_decisions.cost_month(2026, 10) == Decimal("0.300001")
            assert await uow.ai_decisions.cost_month(2026, 9) == Decimal("0.5")
            assert await uow.ai_decisions.cost_month(2026, 11) == Decimal("7.00")
            assert await uow.ai_decisions.cost_month(2026, 12) == Decimal(0)

    async def test_ai_cost_month_rejects_an_invalid_month(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            with pytest.raises(NonRetryableError) as excinfo:
                await uow.ai_decisions.cost_month(2026, 13)
        assert _code(excinfo) == "INVALID_ARGUMENT"

    async def test_uncommitted_ai_decisions_do_not_count(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.ai_decisions.add(make_ai_decision(T0, "1"))
            assert await uow.ai_decisions.count_today(T0.date()) == 1
        async with uow_harness.new_uow() as uow:
            assert await uow.ai_decisions.count_today(T0.date()) == 0

    # ------------------------------------------------------------------ equity

    async def test_peak_equity_is_numeric_not_lexicographic(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            assert await uow.equity_snapshots.peak_equity() is None
            for minutes, equity in enumerate(("99999.99", "100000.10", "-5", "9999999")):
                await uow.equity_snapshots.add(make_equity(T0 + timedelta(minutes=minutes), equity))
            await uow.equity_snapshots.add(make_equity(T0, "100000.10"))
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            peak = await uow.equity_snapshots.peak_equity()
        assert peak == Decimal("9999999")

    async def test_week_start_equity_is_the_first_snapshot_on_or_after_the_day(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        monday = date(2026, 9, 28)
        monday_start = datetime(2026, 9, 28, tzinfo=UTC)
        async with uow_harness.new_uow() as uow:
            await uow.equity_snapshots.add(
                make_equity(monday_start + timedelta(hours=14), "101.50")
            )
            await uow.equity_snapshots.add(
                make_equity(monday_start - timedelta(microseconds=1), "1")
            )
            await uow.equity_snapshots.add(make_equity(monday_start + timedelta(days=1), "102"))
            await uow.equity_snapshots.add(make_equity(monday_start, "100.00"))
            await uow.equity_snapshots.add(make_equity(monday_start, "100.99"))
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            at_monday = await uow.equity_snapshots.week_start_equity(monday)
            at_tuesday = await uow.equity_snapshots.week_start_equity(date(2026, 9, 29))
            later = await uow.equity_snapshots.week_start_equity(date(2026, 10, 5))
        assert at_monday is not None
        assert str(at_monday) == "100.00"  # at 00:00 exactly; first inserted of the tie
        assert at_tuesday == Decimal("102")
        assert later is None

    # ------------------------------------------------------------------ shadow outcomes

    async def test_shadow_outcomes_report_order_and_since_filter(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        late = make_shadow("sig-3", T0 + timedelta(seconds=1), "-1.00")
        sub_second = make_shadow("sig-2", T0 + timedelta(microseconds=500_000), None)
        early = make_shadow("sig-1", T0, "2.0000")
        async with uow_harness.new_uow() as uow:
            for outcome in (late, sub_second, early):
                await uow.shadow_outcomes.add(outcome)
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            everything = await uow.shadow_outcomes.list_for_report()
            since = await uow.shadow_outcomes.list_for_report(T0 + timedelta(microseconds=500_000))
            none = await uow.shadow_outcomes.list_for_report(T0 + timedelta(days=1))
        assert everything == [early, sub_second, late]
        assert str(everything[0].result_r) == "2.0000"
        assert since == [sub_second, late]  # inclusive bound
        assert none == []

    # ------------------------------------------------------------------ control

    async def test_control_is_fail_closed_when_never_written(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            control = await uow.control.get()
        assert control == SystemControl()
        assert control.trading_enabled is False
        assert control.emergency_close is False
        assert control.ai_mode is AIMode.DISABLED

    async def test_control_set_replaces_the_row(self, uow_harness: UnitOfWorkHarness) -> None:
        enabled = SystemControl(
            trading_enabled=True,
            ai_mode=AIMode.SHADOW,
            updated_at_utc=T0 + timedelta(microseconds=7),
            updated_by="owner",
            reason="paper start",
        )
        async with uow_harness.new_uow() as uow:
            await uow.control.set(enabled)
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            assert await uow.control.get() == enabled
            await uow.control.set(SystemControl(emergency_close=True))
            await uow.commit()
        async with uow_harness.new_uow() as uow:
            assert await uow.control.get() == SystemControl(emergency_close=True)

    async def test_uncommitted_control_change_is_discarded(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        async with uow_harness.new_uow() as uow:
            await uow.control.set(SystemControl(trading_enabled=True))
            assert (await uow.control.get()).trading_enabled is True
        async with uow_harness.new_uow() as uow:
            assert await uow.control.get() == SystemControl()

    # ------------------------------------------------------------------ isolation

    async def test_returned_models_do_not_alias_stored_state(
        self, uow_harness: UnitOfWorkHarness
    ) -> None:
        signal = make_signal()
        async with uow_harness.new_uow() as uow:
            await uow.signals.add(signal)
            signal.rule_results[0].values["ema_fast"] = "tampered"  # caller mutates its input
            first = await uow.signals.get("sig-1")
            assert first is not None
            first.rule_results[0].values["bars"] = -1  # caller mutates a result
            second = await uow.signals.get("sig-1")
        assert second == make_signal()
