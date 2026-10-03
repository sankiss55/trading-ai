"""Control CLI (sec. 31.4, AC-20) over a fake unit of work, and over SQLite when present."""

from __future__ import annotations

import io
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from adapters.simulation.sim_clock import FixedClock
from app.config import load_config, pending_owner_decisions
from app.control import (
    EVENT_CONTROL_CHANGED,
    EVENT_CONTROL_REFUSED,
    EVENT_STOP_FILE_CREATED,
    EVENT_STOP_FILE_REMOVED,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_REFUSED,
    HealthProvider,
    main,
)
from application.health import HealthCheck, HealthReport, HealthStatus
from domain.models import AIMode, SystemControl
from tests.unit.app_runtime.fakes import FakeUnitOfWork, at, write_config

FAKE_ENV = {"APP_ENV": "test", "ALPACA_API_KEY": "fake-key", "ALPACA_SECRET_KEY": "fake-secret"}


async def healthy() -> HealthReport:
    return HealthReport(
        checked_at_utc=at(14),
        checks=(HealthCheck(name="broker", status=HealthStatus.OK, detail="reachable"),),
    )


async def broken_health() -> HealthReport:
    raise RuntimeError("no network")


@dataclass
class Cli:
    tmp_path: Path
    config: Path
    uow: FakeUnitOfWork
    environ: dict[str, str]
    health: HealthProvider | None = healthy

    @property
    def stop_file(self) -> Path:
        return self.tmp_path / "control" / "STOP"

    def run(self, *command: str, use_fake_db: bool = True) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        db = self.tmp_path / "data" / "paper" / "trading.db"
        argv: Sequence[str] = [
            "--config",
            str(self.config),
            "--db",
            str(db),
            "--env-file",
            str(self.tmp_path / "absent.env"),
            *command,
        ]
        code = main(
            argv,
            environ=self.environ,
            uow_factory=self.uow.factory if use_fake_db else None,
            clock=FixedClock(at(14, 30)),
            health_provider=self.health,
            out=out,
            err=err,
        )
        return code, out.getvalue(), err.getvalue()

    @property
    def control(self) -> SystemControl:
        return self.uow.committed.control or SystemControl()


@pytest.fixture
def cli(tmp_path: Path) -> Cli:
    return Cli(tmp_path, write_config(tmp_path), FakeUnitOfWork(), {"APP_ENV": "paper"})


@pytest.fixture
def pending_cli(tmp_path: Path) -> Cli:
    return Cli(
        tmp_path, write_config(tmp_path, pending=True), FakeUnitOfWork(), {"APP_ENV": "paper"}
    )


def test_status_shows_fail_closed_defaults(cli: Cli) -> None:
    code, out, _ = cli.run("status")
    assert code == EXIT_OK
    assert "trading_enabled: false" in out
    assert "emergency_close: false" in out
    assert "ai_mode: DISABLED" in out
    assert "updated_at_utc: never" in out
    assert "(absent)" in out
    assert "effective_trading: disabled (TRADING_DISABLED)" in out
    assert "pending_owner_decisions: 0" in out
    assert "app_env: paper" in out
    assert "observation mode only" in out


def test_options_may_follow_the_command(cli: Cli) -> None:
    out = io.StringIO()
    code = main(
        ["status", "--config", str(cli.config), "--db", str(cli.tmp_path / "x.db")],
        environ=cli.environ,
        uow_factory=cli.uow.factory,
        out=out,
        err=io.StringIO(),
    )
    assert code == EXIT_OK
    assert "trading_enabled: false" in out.getvalue()


def test_enable_trading_with_complete_config_and_health(cli: Cli) -> None:
    code, out, _ = cli.run("enable-trading", "--reason", "start paper")
    assert code == EXIT_OK, out
    control = cli.control
    assert control.trading_enabled
    assert control.reason == "start paper"
    assert control.updated_at_utc == at(14, 30)
    assert control.updated_by is not None
    assert control.updated_by.startswith("cli:")
    (event,) = cli.uow.events(EVENT_CONTROL_CHANGED)
    assert event.detail["command"] == "enable-trading"
    assert event.detail["before"]["trading_enabled"] is False
    assert event.detail["after"]["trading_enabled"] is True
    assert event.detail["reason"] == "start paper"
    assert "observation mode only" in out


def test_enable_trading_refused_lists_exactly_the_missing_parameters(pending_cli: Cli) -> None:
    pending = pending_owner_decisions(load_config(pending_cli.config).config)
    code, out, _ = pending_cli.run("enable-trading", "--reason", "try")
    assert code == EXIT_REFUSED
    assert out.startswith("REFUSED: trading cannot be enabled")
    assert "PENDING_OWNER_DECISIONS" in out
    header = f"Missing OWNER_DECISION parameters ({len(pending)}):"
    listed = out.split(header, 1)[1].split()
    assert tuple(listed) == pending
    assert not pending_cli.control.trading_enabled
    assert pending_cli.uow.events(EVENT_CONTROL_CHANGED) == []
    (refused,) = pending_cli.uow.events(EVENT_CONTROL_REFUSED)
    assert refused.detail["command"] == "enable-trading"
    assert refused.detail["refusals"][0]["details"] == list(pending)


def test_enable_trading_refused_in_live(cli: Cli) -> None:
    cli.environ = {"APP_ENV": "live"}
    code, out, _ = cli.run("enable-trading", "--reason", "go live")
    assert code == EXIT_REFUSED
    assert "LIVE_ENVIRONMENT_BLOCKED" in out
    assert not cli.control.trading_enabled


def test_enable_trading_refused_when_health_cannot_run(cli: Cli) -> None:
    cli.health = broken_health
    code, out, _ = cli.run("enable-trading", "--reason", "x")
    assert code == EXIT_REFUSED
    assert "health check could not run: RuntimeError" in out
    assert "HEALTH_UNKNOWN" in out


def test_enable_trading_refused_with_stop_file(cli: Cli) -> None:
    assert cli.run("stop-file", "create", "--reason", "maintenance")[0] == EXIT_OK
    code, out, _ = cli.run("enable-trading", "--reason", "x")
    assert code == EXIT_REFUSED
    assert "STOP_FILE_PRESENT" in out


def test_enable_trading_with_default_health_in_test_env(cli: Cli) -> None:
    cli.environ = dict(FAKE_ENV)
    cli.health = None  # the real provider: test-env adapters, no network
    code, out, _ = cli.run("enable-trading", "--reason", "smoke")
    assert code == EXIT_OK, out
    assert cli.control.trading_enabled


def test_disable_trading(cli: Cli) -> None:
    cli.run("enable-trading", "--reason", "on")
    code, out, _ = cli.run("disable-trading", "--reason", "off")
    assert code == EXIT_OK
    assert "trading_enabled set to false" in out
    assert not cli.control.trading_enabled
    commands = [e.detail["command"] for e in cli.uow.events(EVENT_CONTROL_CHANGED)]
    assert commands == ["enable-trading", "disable-trading"]


def test_set_ai_mode_shadow_and_active_refused(cli: Cli) -> None:
    code, out, _ = cli.run("set-ai-mode", "SHADOW", "--reason", "evaluate")
    assert code == EXIT_OK
    assert cli.control.ai_mode is AIMode.SHADOW
    assert "Phase 8" in out
    code, out, _ = cli.run("set-ai-mode", "ACTIVE", "--reason", "trust it")
    assert code == EXIT_REFUSED
    assert "SHADOW_APPROVAL_MISSING" in out
    assert "sec. 46.4" in out
    assert cli.control.ai_mode is AIMode.SHADOW
    assert cli.uow.events(EVENT_CONTROL_REFUSED)[0].detail["command"] == "set-ai-mode"


def test_emergency_close_sets_flag_disables_trading_and_needs_manual_clear(cli: Cli) -> None:
    cli.run("enable-trading", "--reason", "on")
    code, out, _ = cli.run("emergency-close", "--reason", "panic")
    assert code == EXIT_OK
    assert "Phase 2" in out
    assert cli.control.emergency_close
    assert not cli.control.trading_enabled
    code, out, _ = cli.run("enable-trading", "--reason", "again")
    assert code == EXIT_REFUSED
    assert "EMERGENCY_CLOSE_ACTIVE" in out
    code, _, _ = cli.run("emergency-close", "--clear", "--reason", "verified flat")
    assert code == EXIT_OK
    cleared = cli.uow.committed.control
    assert cleared is not None
    assert (cleared.emergency_close, cleared.trading_enabled) == (False, False)
    commands = [e.detail["command"] for e in cli.uow.events(EVENT_CONTROL_CHANGED)]
    assert commands == ["enable-trading", "emergency-close", "emergency-close --clear"]


def test_stop_file_create_and_remove_are_audited(cli: Cli) -> None:
    code, out, _ = cli.run("stop-file", "create", "--reason", "maintenance")
    assert code == EXIT_OK
    assert cli.stop_file.is_file()
    assert "reason: maintenance" in cli.stop_file.read_text(encoding="utf-8")
    assert "created" in out
    code, out, _ = cli.run("stop-file", "create", "--reason", "again")
    assert "already present" in out
    _, status, _ = cli.run("status")
    assert "(PRESENT)" in status
    assert "STOP_FILE_PRESENT" in status
    code, out, _ = cli.run("stop-file", "remove", "--reason", "done")
    assert code == EXIT_OK
    assert not cli.stop_file.exists()
    _, out, _ = cli.run("stop-file", "remove", "--reason", "twice")
    assert "was not present" in out
    created = cli.uow.events(EVENT_STOP_FILE_CREATED)
    removed = cli.uow.events(EVENT_STOP_FILE_REMOVED)
    assert [e.detail["already_present"] for e in created] == [False, True]
    assert [e.detail["was_present"] for e in removed] == [True, False]


def test_stop_file_works_when_the_database_is_down(cli: Cli) -> None:
    cli.uow.fail_writes = True
    code, _, err = cli.run("stop-file", "create", "--reason", "db down")
    assert code == EXIT_OK
    assert cli.stop_file.is_file()
    assert "audit event STOP_FILE_CREATED not written" in err


def test_stop_file_works_without_any_database(cli: Cli) -> None:
    code, _, err = cli.run("stop-file", "create", "--reason", "no db", use_fake_db=False)
    assert code == EXIT_OK
    assert cli.stop_file.is_file()
    assert "no database" in err


def test_status_without_database_is_an_error(cli: Cli) -> None:
    code, _, err = cli.run("status", use_fake_db=False)
    assert code == EXIT_ERROR
    assert "database not found" in err


def test_write_failure_suggests_the_stop_file(cli: Cli) -> None:
    cli.uow.fail_writes = True
    code, _, err = cli.run("disable-trading", "--reason", "x")
    assert code == EXIT_ERROR
    assert "system_control not written" in err
    assert "stop-file create" in err


def test_blank_reason_is_rejected(cli: Cli) -> None:
    code, _, err = cli.run("disable-trading", "--reason", "   ")
    assert code == EXIT_ERROR
    assert "--reason" in err


def test_invalid_config_is_an_error(cli: Cli) -> None:
    cli.config.write_text("config_version: 1\n", encoding="utf-8")
    code, _, err = cli.run("status")
    assert code == EXIT_ERROR
    assert "CONFIG_INVALID" in err


def test_config_and_db_are_required(cli: Cli) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["status"], out=io.StringIO(), err=io.StringIO())
    assert caught.value.code == 2


# --------------------------------------------------------------------------- real SQLite


def _sqlite_ready() -> Callable[[Path], object] | None:
    try:
        from adapters.sqlite.unit_of_work import SqliteUnitOfWork
    except ImportError:
        return None
    return SqliteUnitOfWork.initialize


@pytest.mark.skipif(_sqlite_ready() is None, reason="adapters.sqlite.unit_of_work not available")
def test_cli_against_a_real_sqlite_database(tmp_path: Path) -> None:
    import asyncio

    from app.container import initialize_sqlite

    config = write_config(tmp_path)
    db = tmp_path / "data" / "paper" / "trading.db"
    asyncio.run(initialize_sqlite(db))
    assert db.is_file()

    def run(*command: str) -> tuple[int, str]:
        out = io.StringIO()
        code = main(
            [
                "--config",
                str(config),
                "--db",
                str(db),
                "--env-file",
                str(tmp_path / "x.env"),
                *command,
            ],
            environ=dict(FAKE_ENV),
            out=out,
            err=io.StringIO(),
        )
        return code, out.getvalue()

    assert run("enable-trading", "--reason", "sqlite")[0] == EXIT_OK
    assert run("set-ai-mode", "ACTIVE", "--reason", "no")[0] == EXIT_REFUSED
    code, status = run("status")
    assert code == EXIT_OK
    assert "trading_enabled: true" in status
    assert "reason: sqlite" in status
    with sqlite3.connect(db) as connection:
        types = [
            row[0]
            for row in connection.execute("SELECT event_type FROM system_events ORDER BY rowid")
        ]
    assert EVENT_CONTROL_CHANGED in types
    assert EVENT_CONTROL_REFUSED in types
