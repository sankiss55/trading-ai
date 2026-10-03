"""Composition root (sec. 8.3.4, 8.6, 9): the ONLY place that instantiates adapters.

Adapters per ``APP_ENV`` for the live process (``python -m app.main``):

==================  =============================================  ==========================
port                ``paper`` / ``dev``                            ``test`` (no network)
==================  =============================================  ==========================
``IClock``          ``SystemClock``                                ``SystemClock``
``IBroker``         ``AlpacaBroker`` (paper forced)                ``SimulatedBroker`` (empty
                                                                   account)
``IMarketCalendar`` ``AlpacaCalendar``                             ``StaticCalendar`` (weekday
                                                                   sessions around today)
``IMarketData``     ``AlpacaMarketData`` (``None`` while           ``ScriptedFeed`` (empty)
                    ``market_data.feed``/``adjustment`` pending)
``INotifier``       ``PolicyNotifier`` over ``SmtpNotifier``       ``PolicyNotifier`` over
                    (``ConsoleNotifier`` if SMTP secrets missing)  ``RecordingNotifier``
``IUnitOfWork``     ``SqliteUnitOfWork`` at ``system.db_path``     ``InMemoryUnitOfWork``
==================  =============================================  ==========================

``APP_ENV=live`` is refused (``LIVE_BLOCKED``) before anything is built (sec. 7.2, 48);
``app/secrets.py`` refuses it first, this is a second barrier. Tests replace any piece
with :class:`AdapterOverrides`. The backtest still builds its simulation adapters in
``backtest/runner.py`` (declared Phase 1 deviation, ``docs/DECISIONS.md``).

Relative paths of ``system.db_path``, ``system.log_dir`` and ``system.control_stop_file``
are resolved against the directory of the config file, so the process behaves the same
whatever its working directory (e.g. under a process supervisor).

Adapters owned by other work units are imported lazily inside their factory functions
(``adapters.sqlite.unit_of_work``, ``adapters.simulation.in_memory_uow``,
``adapters.alpaca.broker``), so a test-env container never imports the Alpaca trading
client and a missing optional adapter fails only where it is needed.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final

from adapters.alpaca import AlpacaCalendar, AlpacaMarketData, BarAdjustment
from adapters.clock.system_clock import SystemClock
from adapters.simulation.historical_feed import ScriptedFeed
from adapters.simulation.recording_notifier import RecordingNotifier
from adapters.simulation.simulated_broker import SimulatedBroker
from adapters.simulation.static_calendar import StaticCalendar, build_regular_sessions
from adapters.smtp.notifier import (
    ConsoleNotifier,
    DeliveryTransport,
    PolicyNotifier,
    Redactor,
    SmtpNotifier,
    SmtpSettings,
)
from app.config import AppConfig, LoadedConfig
from app.lifecycle import UnitOfWorkFactory
from app.logging_setup import fields
from app.notifier_secrets import load_notifier_secrets
from app.secrets import AppEnv, Secrets
from domain.errors import NonRetryableError
from domain.ports import IBroker, IClock, IMarketCalendar, IMarketData, IUnitOfWork

__all__ = [
    "LIVE_BLOCKED",
    "TEST_STARTING_CASH",
    "AdapterOverrides",
    "Container",
    "ContainerError",
    "ExchangeAdapters",
    "RuntimePaths",
    "build_clock",
    "build_container",
    "build_exchange_adapters",
    "build_transport",
    "build_uow_factory",
    "close_quietly",
    "initialize_sqlite",
    "resolve_runtime_paths",
    "sqlite_uow_factory",
]

LIVE_BLOCKED: Final = "LIVE_BLOCKED"
TEST_STARTING_CASH: Final = Decimal("100000")
"""Cash of the simulated test-env account (no position, no order: AC-01 start)."""
_TEST_CALENDAR_DAYS_BEFORE: Final = 14
_TEST_CALENDAR_DAYS_AFTER: Final = 60

_LOGGER = logging.getLogger(__name__)


class ContainerError(NonRetryableError):
    """The adapter set cannot be built (never carries a secret)."""


def _refuse_live(app_env: AppEnv) -> None:
    if app_env is AppEnv.LIVE:
        raise ContainerError(
            "APP_ENV=live is blocked in the MVP (sec. 7.2, 48): no adapter is built",
            code=LIVE_BLOCKED,
        )


# --------------------------------------------------------------------------- paths


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """Absolute runtime paths derived from ``config.system``."""

    config_dir: Path
    db_path: Path
    log_dir: Path
    stop_file: Path


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def resolve_runtime_paths(loaded: LoadedConfig) -> RuntimePaths:
    """``system.db_path``, ``log_dir`` and ``control_stop_file`` relative to the config."""
    base = loaded.path.resolve().parent
    system = loaded.config.system
    return RuntimePaths(
        config_dir=base,
        db_path=_resolve(base, system.db_path),
        log_dir=_resolve(base, system.log_dir),
        stop_file=_resolve(base, system.control_stop_file),
    )


# --------------------------------------------------------------------------- factories


@dataclass(frozen=True, slots=True)
class ExchangeAdapters:
    """Broker-side adapters: trading API, market calendar/clock and market data."""

    broker: IBroker
    calendar: IMarketCalendar
    market_data: IMarketData | None


@dataclass(frozen=True, slots=True)
class AdapterOverrides:
    """Replacements for tests (``None`` keeps the environment's default)."""

    clock: IClock | None = None
    exchange: ExchangeAdapters | None = None
    uow_factory: UnitOfWorkFactory | None = None
    transport: DeliveryTransport | None = None


def build_clock(app_env: AppEnv) -> IClock:
    """``SystemClock`` for every runtime environment (sec. 8.6)."""
    _refuse_live(app_env)
    return SystemClock()


def _test_calendar(clock: IClock) -> StaticCalendar:
    today = clock.now_utc().date()
    first = today - timedelta(days=_TEST_CALENDAR_DAYS_BEFORE)
    days = (
        first + timedelta(days=offset)
        for offset in range(_TEST_CALENDAR_DAYS_BEFORE + _TEST_CALENDAR_DAYS_AFTER + 1)
    )
    weekdays: list[date] = [day for day in days if day.weekday() < 5]
    return StaticCalendar(build_regular_sessions(weekdays), clock)


def build_exchange_adapters(
    app_env: AppEnv, secrets: Secrets, config: AppConfig, clock: IClock
) -> ExchangeAdapters:
    """Broker, calendar and market data for ``app_env`` (paper/dev: network clients)."""
    _refuse_live(app_env)
    if app_env is AppEnv.TEST:
        broker = SimulatedBroker(
            clock=clock,
            starting_cash=TEST_STARTING_CASH,
            slippage_bps=config.risk.slippage_buffer_bps or Decimal(0),
        )
        return ExchangeAdapters(
            broker=broker, calendar=_test_calendar(clock), market_data=ScriptedFeed()
        )
    from adapters.alpaca.broker import AlpacaBroker  # lazy: optional at import time

    credentials = secrets.alpaca_credentials()
    broker_adapter = AlpacaBroker.from_credentials(credentials, app_env=app_env.value)
    calendar = AlpacaCalendar.from_credentials(credentials)
    market_data: IMarketData | None = None
    feed, adjustment = config.market_data.feed, config.market_data.adjustment
    if feed is None or adjustment is None:
        _LOGGER.warning(
            "market data adapter not built: market_data.feed/adjustment pending (OWNER_DECISION)",
            extra=fields(event="MARKET_DATA_NOT_CONFIGURED"),
        )
    else:
        market_data = AlpacaMarketData.from_credentials(
            credentials, feed=feed, adjustment=BarAdjustment(adjustment)
        )
    return ExchangeAdapters(broker=broker_adapter, calendar=calendar, market_data=market_data)


def sqlite_uow_factory(db_path: Path, clock: IClock) -> UnitOfWorkFactory:
    """A fresh ``SqliteUnitOfWork`` per transaction over ``db_path``."""
    from adapters.sqlite.unit_of_work import SqliteUnitOfWork

    def factory() -> IUnitOfWork:
        return SqliteUnitOfWork(db_path, clock=clock)

    return factory


async def initialize_sqlite(db_path: Path) -> None:
    """Create the database directory and apply the pending migrations."""
    from adapters.sqlite.unit_of_work import SqliteUnitOfWork

    await asyncio.to_thread(SqliteUnitOfWork.initialize, db_path)


def build_uow_factory(
    app_env: AppEnv, db_path: Path, clock: IClock
) -> tuple[UnitOfWorkFactory, bool]:
    """``(factory, uses_sqlite)``: SQLite for paper/dev, in-memory for test.

    Both factories return a fresh unit of work per transaction ("a new connection") over
    one shared database, so concurrent tasks never enter the same instance twice.
    """
    _refuse_live(app_env)
    if app_env is AppEnv.TEST:
        from adapters.simulation.in_memory_uow import InMemoryDatabase, InMemoryUnitOfWork

        database = InMemoryDatabase()

        def in_memory() -> IUnitOfWork:
            return InMemoryUnitOfWork(database)

        return in_memory, False
    return sqlite_uow_factory(db_path, clock), True


def build_transport(
    app_env: AppEnv,
    *,
    env_file: Path | None,
    environ: Mapping[str, str] | None,
    redactor: Redactor,
) -> DeliveryTransport:
    """SMTP for paper/dev when its secrets exist (console otherwise); recording for test."""
    _refuse_live(app_env)
    if app_env is AppEnv.TEST:
        return RecordingNotifier()
    smtp_secrets, problems = load_notifier_secrets(env_file, environ=environ)
    if smtp_secrets is None:
        _LOGGER.warning(
            "SMTP notifications disabled: notifications go to the log only",
            extra=fields(event="SMTP_NOT_CONFIGURED", problems=list(problems)),
        )
        return ConsoleNotifier(redactor=redactor)
    redactor.add(smtp_secrets.gmail_app_password)
    settings = SmtpSettings(
        username=smtp_secrets.gmail_user,
        password=smtp_secrets.gmail_app_password,
        recipient=smtp_secrets.notification_email,
        subject_prefix=f"[trading-agent {app_env.value}]",
    )
    return SmtpNotifier(settings, redactor=redactor)


# --------------------------------------------------------------------------- container


async def close_quietly(component: object) -> None:
    """Close an adapter through its ``aclose``/``close`` if it has one; never raises."""
    for name in ("aclose", "close"):
        closer = getattr(component, name, None)
        if callable(closer):
            try:
                result = closer()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - shutdown continues (sec. 54.2)
                _LOGGER.warning(
                    "adapter close failed",
                    extra=fields(
                        event="ADAPTER_CLOSE_FAILED",
                        adapter=type(component).__name__,
                        error_type=type(exc).__name__,
                    ),
                )
            return


@dataclass(slots=True)
class Container:
    """The adapter set of one process."""

    app_env: AppEnv
    loaded: LoadedConfig
    paths: RuntimePaths
    clock: IClock
    exchange: ExchangeAdapters
    transport: DeliveryTransport
    notifier: PolicyNotifier
    uow_factory: UnitOfWorkFactory
    uses_sqlite: bool
    redactor: Redactor

    @property
    def config(self) -> AppConfig:
        """Validated configuration."""
        return self.loaded.config

    @property
    def broker(self) -> IBroker:
        """Trading API adapter."""
        return self.exchange.broker

    @property
    def calendar(self) -> IMarketCalendar:
        """Market calendar and clock adapter."""
        return self.exchange.calendar

    @property
    def market_data(self) -> IMarketData | None:
        """Market data adapter (``None`` while its owner decisions are pending)."""
        return self.exchange.market_data

    async def initialize_database(self) -> None:
        """Open the DB and apply pending migrations (no-op for the in-memory UoW)."""
        if self.uses_sqlite:
            await initialize_sqlite(self.paths.db_path)

    async def aclose(self, *, notifier_timeout: float = 30.0) -> None:
        """Deliver queued notifications (bounded) and close the adapters."""
        await self.notifier.aclose(timeout=notifier_timeout)
        for component in (self.broker, self.calendar, self.market_data):
            if component is not None:
                await close_quietly(component)


def build_container(
    loaded: LoadedConfig,
    secrets: Secrets,
    *,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: AdapterOverrides | None = None,
    redactor: Redactor | None = None,
) -> Container:
    """Build every adapter for ``secrets.app_env`` (sec. 8.6).

    Raises:
        ContainerError: ``APP_ENV=live`` (code ``LIVE_BLOCKED``).
        domain.errors.DomainError: an adapter refused its configuration.
    """
    app_env = secrets.app_env
    _refuse_live(app_env)
    chosen = overrides or AdapterOverrides()
    redact = redactor or Redactor()
    redact.add(secrets.alpaca_api_key, secrets.alpaca_secret_key)
    config = loaded.config
    paths = resolve_runtime_paths(loaded)
    clock = chosen.clock or build_clock(app_env)
    exchange = chosen.exchange or build_exchange_adapters(app_env, secrets, config, clock)
    if chosen.uow_factory is not None:
        uow_factory, uses_sqlite = chosen.uow_factory, False
    else:
        uow_factory, uses_sqlite = build_uow_factory(app_env, paths.db_path, clock)
    transport = chosen.transport or build_transport(
        app_env, env_file=env_file, environ=environ, redactor=redact
    )
    notifier = PolicyNotifier(
        transport,
        policy=config.notifications.policy,
        clock=clock,
        error_digest_minutes=config.notifications.error_digest_minutes,
        queue_max_size=config.system.queue_max_size,
        redactor=redact,
    )
    _LOGGER.info(
        "adapters built",
        extra=fields(
            event="CONTAINER_BUILT",
            app_env=app_env.value,
            broker=type(exchange.broker).__name__,
            calendar=type(exchange.calendar).__name__,
            market_data=type(exchange.market_data).__name__ if exchange.market_data else None,
            notifier_transport=type(transport).__name__,
            database="sqlite" if uses_sqlite else "in_memory",
        ),
    )
    return Container(
        app_env=app_env,
        loaded=loaded,
        paths=paths,
        clock=clock,
        exchange=exchange,
        transport=transport,
        notifier=notifier,
        uow_factory=uow_factory,
        uses_sqlite=uses_sqlite,
        redactor=redact,
    )
