"""Composition root (sec. 8.6): adapters per APP_ENV, no network in any of these tests.

The paper-env test only CONSTRUCTS the Alpaca clients with fake keys (no request is
made); the test-env container never imports the Alpaca trading client path at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import SecretStr

from adapters.alpaca import AlpacaCalendar, AlpacaMarketData
from adapters.alpaca.broker import AlpacaBroker
from adapters.clock.system_clock import SystemClock
from adapters.simulation.historical_feed import ScriptedFeed
from adapters.simulation.in_memory_uow import InMemoryUnitOfWork
from adapters.simulation.recording_notifier import RecordingNotifier
from adapters.simulation.simulated_broker import SimulatedBroker
from adapters.simulation.static_calendar import StaticCalendar
from adapters.smtp.notifier import REDACTED, ConsoleNotifier, NotificationPolicy, SmtpNotifier
from app.config import load_config
from app.container import (
    LIVE_BLOCKED,
    TEST_STARTING_CASH,
    AdapterOverrides,
    ContainerError,
    build_clock,
    build_container,
    resolve_runtime_paths,
)
from app.secrets import AppEnv, Secrets
from domain.models import SystemEvent
from tests.unit.app_runtime.fakes import FakeUnitOfWork, at, write_config


def _secrets(app_env: AppEnv) -> Secrets:
    return Secrets(
        app_env=app_env,
        alpaca_api_key=SecretStr("FAKEKEY12345"),
        alpaca_secret_key=SecretStr("FAKESECRET12345"),
    )


SMTP_ENV = {
    "GMAIL_USER": "bot@example.com",
    "GMAIL_APP_PASSWORD": "fake app password",
    "NOTIFICATION_EMAIL": "owner@example.com",
}


async def test_test_env_builds_fakes_without_network(tmp_path: Path) -> None:
    loaded = load_config(write_config(tmp_path))
    container = build_container(loaded, _secrets(AppEnv.TEST), environ={})
    assert isinstance(container.clock, SystemClock)
    assert isinstance(container.broker, SimulatedBroker)
    assert isinstance(container.calendar, StaticCalendar)
    assert isinstance(container.market_data, ScriptedFeed)
    assert isinstance(container.transport, RecordingNotifier)
    assert not container.uses_sqlite
    assert container.notifier.policy is NotificationPolicy.CRITICAL_PLUS_DAILY
    account = await container.broker.get_account()
    assert account.equity == TEST_STARTING_CASH
    assert await container.broker.get_positions() == []
    assert await container.broker.get_open_orders() == []
    market_clock = await container.calendar.get_clock()
    assert market_clock.next_open_utc > market_clock.now_utc
    await container.initialize_database()  # no-op for the in-memory database
    await container.aclose()


async def test_test_env_units_of_work_share_one_database(tmp_path: Path) -> None:
    container = build_container(
        load_config(write_config(tmp_path)), _secrets(AppEnv.TEST), environ={}
    )
    first, second = container.uow_factory(), container.uow_factory()
    assert isinstance(first, InMemoryUnitOfWork)
    assert first is not second
    async with first:
        await first.system_events.append(SystemEvent(occurred_at_utc=at(14), event_type="X"))
        await first.commit()
    assert isinstance(second, InMemoryUnitOfWork)
    assert [event.event_type for event in second.database.system_events] == ["X"]
    await container.aclose()


def test_live_is_refused_before_anything_is_built(tmp_path: Path) -> None:
    loaded = load_config(write_config(tmp_path))
    with pytest.raises(ContainerError) as caught:
        build_container(loaded, _secrets(AppEnv.LIVE), environ={})
    assert caught.value.code == LIVE_BLOCKED
    with pytest.raises(ContainerError):
        build_clock(AppEnv.LIVE)


async def test_paper_env_builds_alpaca_adapters_and_console_without_smtp(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    loaded = load_config(write_config(tmp_path))
    with caplog.at_level(logging.WARNING):
        container = build_container(
            loaded,
            _secrets(AppEnv.PAPER),
            env_file=tmp_path / "absent.env",
            environ={},
            overrides=AdapterOverrides(uow_factory=FakeUnitOfWork().factory),
        )
    assert isinstance(container.broker, AlpacaBroker)
    assert isinstance(container.calendar, AlpacaCalendar)
    assert isinstance(container.market_data, AlpacaMarketData)
    assert isinstance(container.transport, ConsoleNotifier)
    assert any(
        getattr(record, "fields", {}).get("event") == "SMTP_NOT_CONFIGURED"
        for record in caplog.records
    )
    assert container.redactor("FAKESECRET12345") == REDACTED
    await container.aclose(notifier_timeout=1)


async def test_paper_env_uses_smtp_when_its_secrets_exist(tmp_path: Path) -> None:
    container = build_container(
        load_config(write_config(tmp_path)),
        _secrets(AppEnv.DEV),
        environ=dict(SMTP_ENV),
        overrides=AdapterOverrides(uow_factory=FakeUnitOfWork().factory),
    )
    assert isinstance(container.transport, SmtpNotifier)
    assert container.redactor("fake app password") == REDACTED
    assert container.uses_sqlite is False  # overridden
    await container.aclose(notifier_timeout=1)


async def test_paper_env_defaults_to_sqlite_at_system_db_path(tmp_path: Path) -> None:
    loaded = load_config(write_config(tmp_path))
    container = build_container(loaded, _secrets(AppEnv.PAPER), env_file=None, environ={})
    assert container.uses_sqlite
    assert container.paths.db_path == tmp_path.resolve() / "data" / "paper" / "trading.db"
    assert not container.paths.db_path.exists()  # created only by initialize_database
    await container.initialize_database()
    assert container.paths.db_path.is_file()
    async with container.uow_factory() as uow:
        control = await uow.control.get()
    assert not control.trading_enabled
    await container.aclose(notifier_timeout=1)


async def test_pending_market_data_decisions_skip_the_market_data_adapter(tmp_path: Path) -> None:
    container = build_container(
        load_config(write_config(tmp_path, pending=True)),
        _secrets(AppEnv.PAPER),
        environ={},
        overrides=AdapterOverrides(uow_factory=FakeUnitOfWork().factory),
    )
    assert container.market_data is None
    assert container.notifier.policy is NotificationPolicy.CRITICAL_PLUS_DAILY
    await container.aclose(notifier_timeout=1)


def test_runtime_paths_are_relative_to_the_config_file(tmp_path: Path) -> None:
    paths = resolve_runtime_paths(load_config(write_config(tmp_path)))
    base = tmp_path.resolve()
    assert paths.config_dir == base
    assert paths.db_path == base / "data" / "paper" / "trading.db"
    assert paths.log_dir == base / "logs" / "paper"
    assert paths.stop_file == base / "control" / "STOP"


async def test_overrides_replace_every_piece(tmp_path: Path) -> None:
    transport = RecordingNotifier()
    uow = FakeUnitOfWork()
    container = build_container(
        load_config(write_config(tmp_path)),
        _secrets(AppEnv.TEST),
        environ={},
        overrides=AdapterOverrides(uow_factory=uow.factory, transport=transport),
    )
    assert container.transport is transport
    assert container.uow_factory() is uow
    await container.aclose()
