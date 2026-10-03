"""Storage format of the SQLite adapter: exact TEXT decimals, fixed-width UTC text,
JSON payloads, append-only logs, audit stamps and corrupt-row handling (sec. 38)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from adapters.simulation.sim_clock import FixedClock
from adapters.sqlite.unit_of_work import SqliteUnitOfWork
from domain.errors import StateCriticalError
from domain.models import (
    AIVerdictResult,
    BrokerOrderStatus,
    OrderEvent,
    RiskEvent,
    Snapshot,
    TradeState,
)
from tests.contract.unit_of_work_contract import (
    T0,
    make_ai_decision,
    make_equity,
    make_fill,
    make_order,
    make_order_event,
    make_reconciliation,
    make_risk_event,
    make_signal,
    make_system_event,
    make_trade,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "operational.sqlite3"
    SqliteUnitOfWork.initialize(path)
    return path


def _query(path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


async def _seed(db_path: Path, *, clock: FixedClock | None = None) -> None:
    async with SqliteUnitOfWork(db_path, clock=clock) as uow:
        await uow.signals.add(make_signal())
        await uow.trades.add(make_trade())
        await uow.orders.add(make_order())
        await uow.order_events.append(make_order_event(broker_event={"qty": "7", "n": 1.5}))
        await uow.fills.add_if_new(make_fill())
        await uow.ai_decisions.add(make_ai_decision(T0, "0.012300"))
        await uow.risk_events.append(make_risk_event())
        await uow.equity_snapshots.add(make_equity(T0, "100000.10"))
        await uow.reconciliations.add(make_reconciliation())
        await uow.system_events.append(make_system_event())
        await uow.commit()


async def test_money_is_exact_text_and_times_are_fixed_width_utc(db_path: Path) -> None:
    await _seed(db_path)
    trade = _query(
        db_path,
        "SELECT entry_ref, typeof(entry_ref) AS t, stop_price, qty, typeof(qty) AS q, "
        "state, versions_json FROM trades",
    )[0]
    assert (trade["entry_ref"], trade["t"], trade["stop_price"]) == ("501.2500", "text", "499.10")
    assert (trade["qty"], trade["q"], trade["state"]) == (7, "integer", "AI_DONE")
    assert json.loads(trade["versions_json"])["config_version"] == "2.3.0"
    fill = _query(db_path, "SELECT price, timestamp_utc FROM fills")[0]
    assert fill["price"] == "501.2731"
    assert fill["timestamp_utc"] == "2026-10-01T13:35:01.250000Z"
    signal = _query(db_path, "SELECT created_at_utc, status FROM signals")[0]
    assert signal["created_at_utc"] == "2026-10-01T13:35:00.000000Z"
    assert signal["status"] == "SIGNAL_CREATED"
    equity = _query(db_path, "SELECT equity, typeof(equity) AS t FROM equity_snapshots")[0]
    assert (equity["equity"], equity["t"]) == ("100000.10", "text")


async def test_structured_payloads_are_pydantic_json(db_path: Path) -> None:
    await _seed(db_path)
    event_row = _query(db_path, "SELECT * FROM order_events")[0]
    event = OrderEvent.model_validate(
        {
            **{k: event_row[k] for k in ("client_order_id", "occurred_at_utc", "cause")},
            "from_state": event_row["from_state"],
            "to_state": event_row["to_state"],
            "broker_event": json.loads(event_row["broker_event_json"]),
        }
    )
    assert event == make_order_event(broker_event={"qty": "7", "n": 1.5})
    risk_row = _query(db_path, "SELECT checks_json, sizing_json FROM risk_events")[0]
    risk = RiskEvent(
        signal_id="sig-1",
        occurred_at_utc=T0,
        checks=json.loads(risk_row["checks_json"]),
        sizing=json.loads(risk_row["sizing_json"]),
        decision="APPROVED",
    )
    assert risk == make_risk_event()
    order_row = _query(db_path, "SELECT request_json FROM orders")[0]
    assert json.loads(order_row["request_json"])["order_class"] == "BRACKET"


async def test_ai_decision_audit_columns(db_path: Path) -> None:
    await _seed(db_path)
    row = _query(db_path, "SELECT * FROM ai_decisions")[0]
    record = make_ai_decision(T0, "0.012300")
    assert Snapshot.model_validate_json(row["snapshot_json"]) == record.snapshot
    assert AIVerdictResult.model_validate_json(row["result_json"]) == record.result
    assert row["snapshot_hash"] == "0" * 64
    assert (row["validity"], row["verdict"], row["reason_code"]) == (
        "VALID",
        "APPROVE",
        "SIGNAL_CONFIRMED",
    )
    assert json.loads(row["risk_flags_json"]) == ["LATE_IN_SESSION"]
    assert row["confidence"] == 0.75
    assert json.loads(row["usage_json"])["input_tokens"] == 900
    assert (row["latency_ms"], row["stop_reason"]) == (1850, "end_turn")
    assert row["estimated_cost_usd"] == "0.012300"


@pytest.mark.parametrize("table", ["order_events", "risk_events", "system_events"])
async def test_append_only_tables_refuse_update_and_delete(db_path: Path, table: str) -> None:
    await _seed(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(f"UPDATE {table} SET occurred_at_utc = 'x'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(f"DELETE FROM {table}")
    finally:
        conn.close()


async def test_audit_stamps_come_from_the_injected_clock(db_path: Path) -> None:
    clock = FixedClock(T0 + timedelta(hours=1))
    await _seed(db_path, clock=clock)
    clock.set(T0 + timedelta(hours=2))
    async with SqliteUnitOfWork(db_path, clock=clock) as uow:
        await uow.signals.mark_status("sig-1", TradeState.CHECKS_PASSED, "OK")
        await uow.orders.update_status(
            "paper-sig1-entry", TradeState.SUBMITTED, broker_status=BrokerOrderStatus.NEW
        )
        await uow.commit()
    signal = _query(db_path, "SELECT status, status_reason, status_updated_at_utc FROM signals")[0]
    assert tuple(signal) == ("CHECKS_PASSED", "OK", "2026-10-01T15:35:00.000000Z")
    trade = _query(db_path, "SELECT updated_at_utc FROM trades")[0]
    assert trade["updated_at_utc"] == "2026-10-01T14:35:00.000000Z"
    order = _query(db_path, "SELECT updated_at_utc FROM orders")[0]
    assert order["updated_at_utc"] == "2026-10-01T15:35:00.000000Z"


async def test_audit_stamps_are_null_without_a_clock(db_path: Path) -> None:
    await _seed(db_path)
    assert _query(db_path, "SELECT status_updated_at_utc FROM signals")[0][0] is None
    assert _query(db_path, "SELECT updated_at_utc FROM orders")[0][0] is None


@pytest.mark.parametrize(
    ("sql", "read"),
    [
        ("UPDATE trades SET state = 'BOGUS'", "trade"),
        ("UPDATE trades SET entry_ref = 'abc'", "trade"),
        ("UPDATE trades SET versions_json = '{not json'", "trade"),
        ("UPDATE equity_snapshots SET equity = 'NaN'", "peak"),
        (
            "INSERT INTO system_control VALUES (1, 1, 0, 'ACTIVE', 'yesterday', NULL, NULL)",
            "control",
        ),
    ],
)
async def test_corrupt_rows_raise_state_critical(db_path: Path, sql: str, read: str) -> None:
    await _seed(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute(sql)
    conn.close()
    async with SqliteUnitOfWork(db_path) as uow:
        readers: dict[str, Callable[[], Awaitable[object]]] = {
            "trade": lambda: uow.trades.get("trade-sig-1"),
            "peak": uow.equity_snapshots.peak_equity,
            "control": uow.control.get,
        }
        with pytest.raises(StateCriticalError) as excinfo:
            await readers[read]()
    assert excinfo.value.code == "CORRUPT_RECORD"


async def test_system_control_check_constraints_refuse_invalid_manual_edits(
    db_path: Path,
) -> None:
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO system_control VALUES (1, 1, 0, 'ON', NULL, NULL, NULL)")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO system_control VALUES (2, 1, 0, 'ACTIVE', NULL, NULL, NULL)")
    finally:
        conn.close()
