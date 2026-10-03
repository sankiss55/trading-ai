"""Phase 2 exit (sec. 55): observation-mode startup in APP_ENV=test (no network).

* AC-01: a clean account (no positions) reaches ``READY``; trading stays disabled and
  no order is ever submitted.
* Every step leaves its ``system_events`` trail (sec. 25, 32): STARTUP, mode transitions,
  sync report, health, shutdown report.
* Failures: broker unreachable -> exit 1 with a critical notification; ``APP_ENV=live``
  -> exit 2 before anything is built.
* The loop honors the STOP file and ``emergency_close`` without restarting.
* The same run against a real SQLite file, and ``python -m app.main --once`` as a
  subprocess.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

from adapters.clock.system_clock import SystemClock
from adapters.simulation.in_memory_uow import InMemoryDatabase, InMemoryUnitOfWork
from adapters.simulation.recording_notifier import RecordingNotifier
from app.container import AdapterOverrides, ExchangeAdapters, initialize_sqlite, sqlite_uow_factory
from app.logging_setup import LOG_FILE_NAME
from app.main import (
    EVENT_HEALTH_CHECK,
    EVENT_SHUTDOWN_REPORT,
    EVENT_STARTUP,
    EVENT_STARTUP_FAILED,
    EVENT_STOP_FILE_DETECTED,
    EVENT_SYNC_REPORT,
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    EXIT_STARTUP_FAILED,
    run,
)
from domain.models import (
    NotificationEventType,
    Position,
    ReconciliationOutcome,
    SystemControl,
    SystemEvent,
)
from tests.unit.app_runtime.fakes import FakeBroker, FakeCalendar, unreachable, write_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAKE_ENV = {"APP_ENV": "test", "ALPACA_API_KEY": "FAKEKEY12345", "ALPACA_SECRET_KEY": "FAKESECRET1"}


def _argv(config: Path, tmp_path: Path, *extra: str) -> list[str]:
    return ["--config", str(config), "--env-file", str(tmp_path / "absent.env"), *extra]


def _transitions(events: list[SystemEvent]) -> list[tuple[str, str]]:
    return [
        (str(e.detail["from"]), str(e.detail["to"]))
        for e in events
        if e.event_type == "MODE_TRANSITION"
    ]


def _types(events: list[SystemEvent]) -> list[str]:
    return [event.event_type for event in events]


def _log_lines(tmp_path: Path) -> list[dict[str, object]]:
    text = (tmp_path / "logs" / "paper" / LOG_FILE_NAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class Harness:
    """In-process run with a shared in-memory database and a recording transport."""

    def __init__(self, tmp_path: Path, *, pending: bool = False, poll: str | None = None) -> None:
        self.tmp_path = tmp_path
        self.config = write_config(tmp_path, pending=pending, poll_seconds=poll)
        self.database = InMemoryDatabase()
        self.transport = RecordingNotifier()
        self.clock = SystemClock()
        self.broker = FakeBroker()
        self.calendar = FakeCalendar(self.clock)

    def uow(self) -> InMemoryUnitOfWork:
        return InMemoryUnitOfWork(self.database)

    def overrides(self, *, fake_exchange: bool = True) -> AdapterOverrides:
        exchange = (
            ExchangeAdapters(broker=self.broker, calendar=self.calendar, market_data=None)
            if fake_exchange
            else None
        )
        return AdapterOverrides(exchange=exchange, uow_factory=self.uow, transport=self.transport)

    async def run(self, *extra: str, **kwargs: object) -> int:
        return await run(
            _argv(self.config, self.tmp_path, *extra),
            environ=dict(FAKE_ENV),
            overrides=self.overrides(),
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def events(self) -> list[SystemEvent]:
        return list(self.database.system_events)


async def test_once_reaches_ready_with_trading_disabled_ac01(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    assert await harness.run("--once") == EXIT_OK
    events = harness.events
    assert _transitions(events) == [
        ("BOOTING", "SYNCING"),
        ("SYNCING", "READY"),
        ("READY", "SHUTTING_DOWN"),
    ]
    types = _types(events)
    assert types[0] == EVENT_STARTUP
    for expected in (EVENT_SYNC_REPORT, EVENT_HEALTH_CHECK, EVENT_SHUTDOWN_REPORT):
        assert expected in types
    ready = next(
        e for e in events if e.event_type == "MODE_TRANSITION" and e.detail["to"] == "READY"
    )
    assert ready.detail["observation_only"] is True
    assert ready.detail["trading_effective"] is False
    assert ready.detail["trading_enabled_control"] is False
    assert "PHASE_2_OBSERVATION_ONLY" in ready.detail["blocked_by"]
    assert ready.detail["health"] == "OK"
    startup = events[0]
    assert startup.detail["config_version"] == "2.3.0"
    assert len(str(startup.detail["config_hash"])) == 64
    assert startup.detail["pending_owner_decisions"] == []
    # AC-01: no positions, no orders; nothing was ever submitted
    assert harness.broker.order_calls == 0
    (reconciliation,) = harness.database.reconciliations
    assert reconciliation.outcome is ReconciliationOutcome.RECONCILED_OK
    assert reconciliation.actions == ("REPORT_ONLY_PHASE_2",)
    (snapshot,) = harness.database.equity_snapshots
    assert snapshot.equity == Decimal("100000")
    assert harness.database.control is None or not harness.database.control.trading_enabled
    assert harness.transport.events == []  # nothing critical happened
    assert harness.broker.closed
    lines = _log_lines(tmp_path)
    assert any(line.get("event") == "CONFIG_LOADED" for line in lines)
    assert any(
        line.get("event") == "MODE_TRANSITION" and line.get("to_mode") == "READY" for line in lines
    )
    raw = (tmp_path / "logs" / "paper" / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "FAKESECRET1" not in raw


async def test_default_test_env_adapters_reach_ready(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    code = await run(
        _argv(harness.config, tmp_path, "--once"),
        environ=dict(FAKE_ENV),
        overrides=harness.overrides(fake_exchange=False),
    )
    assert code == EXIT_OK
    assert ("SYNCING", "READY") in _transitions(harness.events)


async def test_trading_enabled_in_control_is_still_not_effective(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    async with harness.uow() as uow:
        await uow.control.set(SystemControl(trading_enabled=True))
        await uow.commit()
    assert await harness.run("--once") == EXIT_OK
    ready = next(e for e in harness.events if e.detail.get("to") == "READY")
    assert ready.detail["trading_enabled_control"] is True
    assert ready.detail["trading_effective"] is False


async def test_pending_decisions_still_start_in_observation(tmp_path: Path) -> None:
    harness = Harness(tmp_path, pending=True)
    assert await harness.run("--once") == EXIT_OK
    startup = harness.events[0]
    assert "universe.whitelist" in startup.detail["pending_owner_decisions"]
    ready = next(e for e in harness.events if e.detail.get("to") == "READY")
    assert ready.detail["health"] == "DEGRADED"


async def test_unknown_broker_position_is_reported_but_never_acted_on(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.broker.positions = [
        Position(
            symbol="SPY",
            qty=Decimal(3),
            avg_entry_price=Decimal("500"),
            market_value=Decimal("1500"),
        )
    ]
    assert await harness.run("--once") == EXIT_OK
    (reconciliation,) = harness.database.reconciliations
    assert reconciliation.outcome is ReconciliationOutcome.ORPHANED
    assert reconciliation.differences[0]["symbol"] == "SPY"
    assert [e.event_type for e in harness.transport.events] == [
        NotificationEventType.ORPHANED_POSITION
    ]
    assert harness.broker.order_calls == 0
    shutdown = next(e for e in harness.events if e.event_type == EVENT_SHUTDOWN_REPORT)
    assert shutdown.detail["positions"] == [{"symbol": "SPY", "qty": "3"}]


async def test_broker_unreachable_fails_startup_with_a_critical_alert(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.broker.error = unreachable()
    assert await harness.run("--once") == EXIT_STARTUP_FAILED
    assert _transitions(harness.events) == [("BOOTING", "SYNCING"), ("SYNCING", "SHUTTING_DOWN")]
    failed = next(e for e in harness.events if e.event_type == EVENT_STARTUP_FAILED)
    assert "NETWORK" in str(failed.detail["error"])
    assert [e.event_type for e in harness.transport.events] == [NotificationEventType.BROKER_ERROR]


async def test_live_env_is_refused_before_anything_is_built(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    code = await run(
        _argv(harness.config, tmp_path, "--once"),
        environ={**FAKE_ENV, "APP_ENV": "live"},
        overrides=harness.overrides(),
    )
    assert code == EXIT_CONFIG_ERROR
    assert harness.events == []
    assert any(line.get("code") == "LIVE_BLOCKED" for line in _log_lines(tmp_path))


async def test_invalid_config_exits_2(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("system: {}\n", encoding="utf-8")
    assert await run(_argv(config, tmp_path, "--once"), environ=dict(FAKE_ENV)) == EXIT_CONFIG_ERROR


async def _wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


async def test_loop_honors_stop_file_and_emergency_close_then_stops(tmp_path: Path) -> None:
    harness = Harness(tmp_path, poll="0.05")
    stop = asyncio.Event()
    task = asyncio.create_task(harness.run(stop_event=stop))

    def reached(target: str) -> Callable[[], bool]:
        return lambda: any(t == target for _, t in _transitions(harness.events))

    await _wait_until(reached("READY"))
    (tmp_path / "control").mkdir(exist_ok=True)
    (tmp_path / "control" / "STOP").write_text("manual\n", encoding="utf-8")
    await _wait_until(lambda: EVENT_STOP_FILE_DETECTED in _types(harness.events))
    async with harness.uow() as uow:
        await uow.control.set(SystemControl(emergency_close=True, reason="drill"))
        await uow.commit()
    await _wait_until(reached("EMERGENCY"))
    stop.set()
    assert await asyncio.wait_for(task, 10) == EXIT_OK
    assert _transitions(harness.events)[-2:] == [
        ("READY", "EMERGENCY"),
        ("EMERGENCY", "SHUTTING_DOWN"),
    ]
    assert NotificationEventType.EMERGENCY_CLOSE in [e.event_type for e in harness.transport.events]
    assert harness.broker.order_calls == 0


async def test_once_against_a_real_sqlite_database(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    db_path = tmp_path / "data" / "paper" / "trading.db"
    await initialize_sqlite(db_path)
    code = await run(
        _argv(harness.config, tmp_path, "--once"),
        environ=dict(FAKE_ENV),
        overrides=AdapterOverrides(
            exchange=ExchangeAdapters(
                broker=harness.broker, calendar=harness.calendar, market_data=None
            ),
            uow_factory=sqlite_uow_factory(db_path, harness.clock),
            transport=harness.transport,
        ),
    )
    assert code == EXIT_OK
    with sqlite3.connect(db_path) as connection:
        types = [
            row[0]
            for row in connection.execute("SELECT event_type FROM system_events ORDER BY rowid")
        ]
        snapshots = connection.execute("SELECT COUNT(*) FROM equity_snapshots").fetchone()[0]
    assert types[0] == EVENT_STARTUP
    assert types.count("MODE_TRANSITION") == 3
    assert EVENT_SHUTDOWN_REPORT in types
    assert snapshots == 1


def test_python_m_app_main_once_subprocess(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("ALPACA_", "GMAIL_"))
    }
    env.update(FAKE_ENV)
    env.pop("NOTIFICATION_EMAIL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.main",
            "--config",
            str(config),
            "--env-file",
            str(tmp_path / "absent.env"),
            "--once",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == EXIT_OK, result.stderr[-2000:]
    lines = _log_lines(tmp_path)
    assert any(
        line.get("event") == "MODE_TRANSITION" and line.get("to_mode") == "READY" for line in lines
    )
    assert any(line.get("event") == "PROCESS_STOPPED" for line in lines)
