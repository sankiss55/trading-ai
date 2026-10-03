"""Process entry point (sec. 9, 25, 54): Phase 2 OBSERVATION mode.

Usage::

    python -m app.main --config config.yaml [--env-file .env] [--once]

Phase 2 exit criterion (sec. 55): the process starts in observation mode only. It never
submits, cancels or modifies an order, and never reaches ``RUNNING``: whatever
``system_control.trading_enabled`` says, trading is effectively disabled (AC-01 is
"no positions -> READY"). ``--once`` runs the startup and one health cycle, then shuts
down (tests and smoke runs).

Startup (sec. 25, Phase 2 subset)::

    BOOTING   load + validate config.yaml (sha256 logged with config_version)
              JSON logging to system.log_dir; secrets (APP_ENV=live refused)
              adapters for APP_ENV (sec. 8.6); open the DB and apply migrations
              read system_control and the STOP file; record STARTUP
    SYNCING   broker account, market clock and today's session
              positions and open orders; equity snapshot
              reconciliation, REPORT ONLY (Phase 5 applies actions)
              ai_mode check (the AI filter is Phase 8); full health check (sec. 42)
    READY     trading disabled (observation)

Loop, every ``system.control_poll_seconds``: STOP file and ``system_control`` (a manual STOP
file is recorded; ``emergency_close`` moves the system to ``EMERGENCY``), a warning if
``config.yaml`` changed on disk (ignored until restart, sec. 33), the notification error
digest, the health checks every :data:`HEALTH_INTERVAL_SECONDS` (recorded when they
change, sec. 42) and the daily summary at ``notifications.daily_summary_time_et``.

Shutdown on SIGINT / SIGTERM (sec. 54.2 subset): ``SHUTTING_DOWN``, a final report-only
snapshot of broker positions and orders, bounded delivery of queued notifications,
adapters closed. Shutdown never closes positions.

Exit codes: ``0`` clean run (``READY`` reached), ``1`` startup failure, ``2`` configuration,
secrets, environment or adapter construction error.

Declared deviation: the startup sync lives here until ``application/startup.py`` and
``app/orchestrator.py`` exist (Phase 5).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from types import FrameType
from typing import Any, Final
from zoneinfo import ZoneInfo

from adapters.smtp.notifier import Redactor
from app.config import (
    ConfigError,
    LoadedConfig,
    config_hash,
    load_config,
    pending_owner_decisions,
)
from app.container import AdapterOverrides, Container, build_container, resolve_runtime_paths
from app.lifecycle import SystemLifecycle, record_system_event
from app.logging_setup import close_logging, configure_logging, fields
from app.secrets import DEFAULT_ENV_FILE, SecretsError, load_secrets
from application.control_rules import trading_block_reasons
from application.health import (
    CHECK_BROKER,
    CHECK_CLOCK_SKEW,
    HealthReport,
    HealthStatus,
    run_health_checks,
)
from domain.errors import DomainError
from domain.models import (
    AIMode,
    EquitySnapshot,
    EquitySnapshotKind,
    NotificationEvent,
    NotificationEventType,
    ReconciliationOutcome,
    ReconciliationRecord,
    Severity,
    SystemControl,
    SystemEvent,
    SystemMode,
)

__all__ = [
    "EVENT_CONFIG_CHANGED_ON_DISK",
    "EVENT_HEALTH_CHECK",
    "EVENT_SHUTDOWN_REPORT",
    "EVENT_STARTUP",
    "EVENT_STARTUP_FAILED",
    "EVENT_STOP_FILE_CLEARED",
    "EVENT_STOP_FILE_DETECTED",
    "EVENT_SYNC_REPORT",
    "EXIT_CONFIG_ERROR",
    "EXIT_OK",
    "EXIT_STARTUP_FAILED",
    "HEALTH_INTERVAL_SECONDS",
    "ObservationRuntime",
    "build_parser",
    "main",
    "next_daily_summary_at",
    "run",
]

EXIT_OK: Final = 0
EXIT_STARTUP_FAILED: Final = 1
EXIT_CONFIG_ERROR: Final = 2

HEALTH_INTERVAL_SECONDS: Final = 60
"""Health cycle period (sec. 42 SUGERIDO: broker and DB every 60 s)."""
NOTIFIER_SHUTDOWN_TIMEOUT_SECONDS: Final = 30.0

EVENT_STARTUP: Final = "STARTUP"
EVENT_STARTUP_FAILED: Final = "STARTUP_FAILED"
EVENT_SYNC_REPORT: Final = "SYNC_REPORT"
EVENT_HEALTH_CHECK: Final = "HEALTH_CHECK"
EVENT_STOP_FILE_DETECTED: Final = "STOP_FILE_DETECTED"
EVENT_STOP_FILE_CLEARED: Final = "STOP_FILE_CLEARED"
EVENT_CONFIG_CHANGED_ON_DISK: Final = "CONFIG_CHANGED_ON_DISK"
EVENT_AI_MODE_UNAVAILABLE: Final = "AI_MODE_UNAVAILABLE"
EVENT_SHUTDOWN_REPORT: Final = "SHUTDOWN_REPORT"
REPORT_ONLY: Final = "REPORT_ONLY_PHASE_2"

_LOGGER = logging.getLogger("app.main")


def next_daily_summary_at(after_utc: datetime, hhmm: str, tz: ZoneInfo) -> datetime:
    """First instant strictly after ``after_utc`` at local wall time ``hhmm`` in ``tz``."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = after_utc.astimezone(tz)
    candidate = datetime.combine(local.date(), time(hour, minute), tzinfo=tz)
    if candidate <= local:
        candidate = datetime.combine(
            local.date() + timedelta(days=1), time(hour, minute), tzinfo=tz
        )
    return candidate.astimezone(UTC)


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, DomainError):
        return f"{type(exc).__name__}: {exc}"
    return type(exc).__name__


class ObservationRuntime:
    """Phase 2 runtime: startup sync, observation loop and safe shutdown.

    Args:
        container: Adapter set of this process.
    """

    def __init__(self, container: Container) -> None:
        self._c = container
        config = container.config
        self.lifecycle = SystemLifecycle(uow_factory=container.uow_factory, clock=container.clock)
        self._pending = pending_owner_decisions(config)
        self._market_tz = ZoneInfo(config.system.timezone_market)
        self._poll_seconds = float(config.system.control_poll_seconds)
        self._control: SystemControl | None = None
        self._stop_present: bool | None = None
        self._health: HealthReport | None = None
        self._next_health_at: datetime | None = None
        self._next_summary_at: datetime | None = None
        self._config_change_reported = False
        self._control_error_reported = False

    @property
    def last_health(self) -> HealthReport | None:
        """Most recent health report."""
        return self._health

    # ------------------------------------------------------------------ helpers

    def _event(self, event_type: str, detail: Mapping[str, Any]) -> SystemEvent:
        return SystemEvent(
            occurred_at_utc=self._c.clock.now_utc(), event_type=event_type, detail=dict(detail)
        )

    async def _record(self, event_type: str, detail: Mapping[str, Any]) -> bool:
        """Best-effort audit event (the loop never dies because the DB is down)."""
        try:
            await record_system_event(self._c.uow_factory, self._event(event_type, detail))
        except Exception as exc:  # noqa: BLE001 - logged; DB health is reported separately
            _LOGGER.error(
                "system event not written",
                extra=fields(
                    event="SYSTEM_EVENT_WRITE_FAILED",
                    event_type=event_type,
                    error_type=type(exc).__name__,
                ),
            )
            return False
        return True

    async def _notify(
        self, event_type: NotificationEventType, severity: Severity, subject: str, body: str
    ) -> None:
        await self._c.notifier.send(
            NotificationEvent(event_type=event_type, severity=severity, subject=subject, body=body)
        )

    async def _read_control(self) -> SystemControl:
        async with self._c.uow_factory() as uow:
            return await uow.control.get()

    def _stop_file_present(self) -> bool:
        return self._c.paths.stop_file.exists()

    # ------------------------------------------------------------------ startup

    async def startup(self) -> None:
        """Run the Phase 2 startup sequence up to ``READY``.

        Raises:
            Exception: any step failed (the caller shuts down with exit code 1).
        """
        c = self._c
        config = c.config
        await c.initialize_database()
        control = await self._read_control()
        stop_present = self._stop_file_present()
        self._control, self._stop_present = control, stop_present
        await record_system_event(
            c.uow_factory,
            self._event(
                EVENT_STARTUP,
                {
                    "app_env": c.app_env.value,
                    "pid": os.getpid(),
                    "config_path": str(c.loaded.path),
                    "config_version": config.config_version,
                    "config_hash": c.loaded.config_hash,
                    "strategy_version": config.strategy_version,
                    "risk_version": config.risk_version,
                    "prompt_version": config.prompt_version,
                    "pending_owner_decisions": list(self._pending),
                    "control": control.model_dump(mode="json"),
                    "stop_file_present": stop_present,
                    "observation_only": True,
                },
            ),
        )
        await self.lifecycle.transition(
            SystemMode.SYNCING, reason="configuration, secrets, database and controls loaded"
        )
        await self._sync()
        health = await self.health_cycle(force_record=True)
        await self.lifecycle.transition(
            SystemMode.READY,
            reason="startup sync complete; observation mode (Phase 2): trading disabled",
            detail={
                "observation_only": True,
                "trading_effective": False,
                "trading_enabled_control": control.trading_enabled,
                "blocked_by": [
                    *trading_block_reasons(control, stop_file_present=stop_present),
                    "PHASE_2_OBSERVATION_ONLY",
                ],
                "health": str(health.status),
            },
        )
        if control.emergency_close:
            await self._enter_emergency(control)

    async def _sync(self) -> None:
        c = self._c
        account = await c.broker.get_account()
        market_clock = await c.calendar.get_clock()
        market_day = market_clock.now_utc.astimezone(self._market_tz).date()
        session = await c.calendar.get_session(market_day)
        positions = await c.broker.get_positions()
        orders = await c.broker.get_open_orders()
        now = c.clock.now_utc()
        differences: list[dict[str, Any]] = [
            {"kind": "broker_position_without_trade", "symbol": p.symbol, "qty": str(p.qty)}
            for p in positions
        ] + [
            {
                "kind": "broker_order_unknown",
                "symbol": o.symbol,
                "client_order_id": o.client_order_id,
                "status": str(o.status),
            }
            for o in orders
        ]
        if positions:
            outcome = ReconciliationOutcome.ORPHANED
        elif orders:
            outcome = ReconciliationOutcome.STATE_MISMATCH
        else:
            outcome = ReconciliationOutcome.RECONCILED_OK
        async with c.uow_factory() as uow:
            await uow.equity_snapshots.add(
                EquitySnapshot(
                    taken_at_utc=now,
                    kind=EquitySnapshotKind.PERIODIC,
                    equity=account.equity,
                    last_equity=account.last_equity,
                    buying_power=account.buying_power,
                )
            )
            await uow.reconciliations.add(
                ReconciliationRecord(
                    occurred_at_utc=now,
                    outcome=outcome,
                    differences=tuple(differences),
                    actions=(REPORT_ONLY,),
                )
            )
            await uow.system_events.append(
                self._event(
                    EVENT_SYNC_REPORT,
                    {
                        "account_status": account.status,
                        "market_open": market_clock.is_open,
                        "market_now_utc": market_clock.now_utc.isoformat(),
                        "next_open_utc": market_clock.next_open_utc.isoformat(),
                        "next_close_utc": market_clock.next_close_utc.isoformat(),
                        "trading_day": session is not None,
                        "positions": len(positions),
                        "open_orders": len(orders),
                        "reconciliation": str(outcome),
                    },
                )
            )
            await uow.commit()
        _LOGGER.info(
            "startup sync",
            extra=fields(
                event=EVENT_SYNC_REPORT,
                account_status=account.status,
                market_open=market_clock.is_open,
                trading_day=session is not None,
                positions=len(positions),
                open_orders=len(orders),
                reconciliation=str(outcome),
            ),
        )
        if differences:
            _LOGGER.warning(
                "broker holds positions/orders unknown to this system (report only)",
                extra=fields(event="RECONCILIATION_DIFFERENCES", differences=differences),
            )
            lines = "\n".join(
                f"- {item['kind']}: {item['symbol']}"
                + (f" qty {item['qty']}" if "qty" in item else f" {item['status']}")
                for item in differences
            )
            await self._notify(
                NotificationEventType.ORPHANED_POSITION
                if positions
                else NotificationEventType.STATE_MISMATCH,
                Severity.CRITICAL,
                "Broker positions/orders not known to the system",
                "Startup reconciliation (report only, Phase 2: no action taken):\n"
                f"{lines}\nVerify the broker account manually.",
            )
        if self._control is not None and self._control.ai_mode is not AIMode.DISABLED:
            _LOGGER.warning(
                "ai_mode is not DISABLED but the AI filter does not exist before Phase 8",
                extra=fields(event=EVENT_AI_MODE_UNAVAILABLE, ai_mode=str(self._control.ai_mode)),
            )
            await self._record(EVENT_AI_MODE_UNAVAILABLE, {"ai_mode": str(self._control.ai_mode)})

    # ------------------------------------------------------------------ health

    async def health_cycle(self, *, force_record: bool = False) -> HealthReport:
        """Run the health checks; record and notify when the result changes."""
        c = self._c
        report = await run_health_checks(
            broker=c.broker,
            calendar=c.calendar,
            clock=c.clock,
            uow=c.uow_factory(),
            max_clock_skew_seconds=c.config.system.max_clock_skew_seconds,
            pending_decisions=self._pending,
            stop_file_present=self._stop_file_present(),
        )
        previous = self._health
        self._health = report
        self._next_health_at = report.checked_at_utc + timedelta(seconds=HEALTH_INTERVAL_SECONDS)
        changed = previous is None or previous.signature() != report.signature()
        if changed or force_record:
            await self._record(EVENT_HEALTH_CHECK, report.model_dump(mode="json"))
            level = logging.INFO if report.status is HealthStatus.OK else logging.WARNING
            _LOGGER.log(
                level,
                "health",
                extra=fields(
                    event=EVENT_HEALTH_CHECK,
                    status=str(report.status),
                    not_ok=[f"{check.name}: {check.detail}" for check in report.not_ok()],
                ),
            )
        if changed:
            await self._notify_new_failures(previous, report)
        return report

    async def _notify_new_failures(
        self, previous: HealthReport | None, report: HealthReport
    ) -> None:
        for check in report.checks:
            before = previous.get(check.name) if previous is not None else None
            newly_failed = check.status is HealthStatus.FAILED and (
                before is None or before.status is not HealthStatus.FAILED
            )
            if newly_failed:
                event_type = (
                    NotificationEventType.BROKER_ERROR
                    if check.name in (CHECK_BROKER, CHECK_CLOCK_SKEW)
                    else NotificationEventType.SYSTEM_ERROR
                )
                await self._notify(
                    event_type,
                    Severity.CRITICAL,
                    f"Health check failed: {check.name}",
                    f"{check.detail}\nMode: {self.lifecycle.mode}. Trading is disabled "
                    "(observation mode).",
                )

    # ------------------------------------------------------------------ loop

    async def _enter_emergency(self, control: SystemControl) -> None:
        if self.lifecycle.mode in (SystemMode.EMERGENCY, SystemMode.SHUTTING_DOWN):
            return
        await self.lifecycle.transition(
            SystemMode.EMERGENCY,
            reason="emergency_close set in system_control",
            detail={"control_reason": control.reason, "control_updated_by": control.updated_by},
        )
        await self._notify(
            NotificationEventType.EMERGENCY_CLOSE,
            Severity.CRITICAL,
            "Emergency close requested",
            "emergency_close is set. Phase 2 has no execution path: no order was sent and no "
            "position was closed by the system (sec. 31.3 procedure arrives in Phase 5). "
            "Verify the broker account manually.",
        )

    async def poll_controls(self) -> None:
        """Read the STOP file and ``system_control``; record and react to changes."""
        stop_present = self._stop_file_present()
        if stop_present != self._stop_present:
            event_type = EVENT_STOP_FILE_DETECTED if stop_present else EVENT_STOP_FILE_CLEARED
            _LOGGER.warning(
                "STOP file " + ("detected: trading disabled" if stop_present else "removed"),
                extra=fields(event=event_type, path=str(self._c.paths.stop_file)),
            )
            await self._record(event_type, {"path": str(self._c.paths.stop_file)})
            self._stop_present = stop_present
        try:
            control = await self._read_control()
        except Exception as exc:  # noqa: BLE001 - the STOP file keeps working (sec. 31.2)
            if not self._control_error_reported:
                self._control_error_reported = True
                _LOGGER.error(
                    "system_control unreadable; the STOP file is still honored",
                    extra=fields(event="CONTROL_READ_FAILED", error_type=type(exc).__name__),
                )
                await self._notify(
                    NotificationEventType.SYSTEM_ERROR,
                    Severity.CRITICAL,
                    "system_control unreadable",
                    f"{_describe_error(exc)}. Use the STOP file if needed.",
                )
            return
        self._control_error_reported = False
        if control != self._control:
            _LOGGER.info(
                "system_control changed",
                extra=fields(event="CONTROL_OBSERVED", control=control.model_dump(mode="json")),
            )
            if control.trading_enabled:
                _LOGGER.warning(
                    "trading_enabled is true but this build is observation-only (Phase 2)",
                    extra=fields(event="TRADING_NOT_AVAILABLE"),
                )
            self._control = control
        if control.emergency_close:
            await self._enter_emergency(control)

    async def _check_config_on_disk(self) -> None:
        if self._config_change_reported:
            return
        loaded = self._c.loaded
        try:
            changed = config_hash(loaded.path.read_bytes()) != loaded.config_hash
        except OSError:
            changed = True
        if changed:
            self._config_change_reported = True
            _LOGGER.warning(
                "config.yaml changed on disk: ignored until the next start (sec. 33)",
                extra=fields(event=EVENT_CONFIG_CHANGED_ON_DISK, path=str(loaded.path)),
            )
            await self._record(EVENT_CONFIG_CHANGED_ON_DISK, {"path": str(loaded.path)})

    def _summary_body(self) -> str:
        health = self._health
        control = self._control or SystemControl()
        lines = [
            f"Mode: {self.lifecycle.mode} (observation only, Phase 2: no trades).",
            f"Health: {health.status if health else 'unknown'}",
        ]
        if health is not None:
            lines += [f"  {check.name}: {check.status} ({check.detail})" for check in health.checks]
        lines += [
            f"trading_enabled: {control.trading_enabled}, emergency_close: "
            f"{control.emergency_close}, ai_mode: {control.ai_mode}",
            f"STOP file present: {bool(self._stop_present)}",
            f"Pending owner decisions: {len(self._pending)}",
        ]
        return "\n".join(lines)

    async def _daily_summary_if_due(self) -> None:
        now = self._c.clock.now_utc()
        hhmm = self._c.config.notifications.daily_summary_time_et
        if self._next_summary_at is None:
            self._next_summary_at = next_daily_summary_at(now, hhmm, self._market_tz)
            return
        if now >= self._next_summary_at:
            await self._notify(
                NotificationEventType.DAILY_SUMMARY,
                Severity.INFO,
                f"Daily summary {now.astimezone(self._market_tz).date().isoformat()}",
                self._summary_body(),
            )
            self._next_summary_at = next_daily_summary_at(now, hhmm, self._market_tz)

    async def tick(self, *, force_health: bool = False) -> None:
        """One loop iteration. Each step is guarded: the loop never dies (sec. 27)."""
        steps = (
            ("controls", self.poll_controls),
            ("config", self._check_config_on_disk),
            ("notifications", self._c.notifier.flush_due),
            ("daily_summary", self._daily_summary_if_due),
        )
        for name, step in steps:
            try:
                await step()
            except Exception as exc:  # noqa: BLE001 - supervised step, logged and retried next tick
                _LOGGER.error(
                    "loop step failed",
                    extra=fields(
                        event="LOOP_STEP_FAILED", step=name, error_type=type(exc).__name__
                    ),
                )
        due = self._next_health_at is None or self._c.clock.now_utc() >= self._next_health_at
        if force_health or due:
            try:
                await self.health_cycle()
            except Exception as exc:  # noqa: BLE001 - supervised step
                _LOGGER.error(
                    "health cycle failed",
                    extra=fields(
                        event="LOOP_STEP_FAILED", step="health", error_type=type(exc).__name__
                    ),
                )

    async def run(self, stop_event: asyncio.Event, *, once: bool = False) -> int:
        """Startup, observation loop until ``stop_event`` (or one cycle), shutdown."""
        try:
            await self.startup()
        except Exception as exc:  # noqa: BLE001 - startup failure: report and shut down
            await self._startup_failed(exc)
            await self.shutdown(reason=f"startup failed: {type(exc).__name__}")
            return EXIT_STARTUP_FAILED
        if once:
            await self.tick(force_health=True)
        else:
            while not stop_event.is_set():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=self._poll_seconds)
                if stop_event.is_set():
                    break
                await self.tick()
        await self.shutdown(reason="--once completed" if once else "stop requested")
        return EXIT_OK

    async def _startup_failed(self, exc: Exception) -> None:
        detail = _describe_error(exc)
        _LOGGER.error(
            "startup failed",
            extra=fields(event=EVENT_STARTUP_FAILED, mode=str(self.lifecycle.mode), error=detail),
        )
        await self._record(
            EVENT_STARTUP_FAILED, {"mode": str(self.lifecycle.mode), "error": detail}
        )
        broker_phase = self.lifecycle.mode is SystemMode.SYNCING
        await self._notify(
            NotificationEventType.BROKER_ERROR
            if broker_phase
            else NotificationEventType.SYSTEM_ERROR,
            Severity.CRITICAL,
            f"Startup failed in {self.lifecycle.mode}",
            f"{detail}\nThe process is shutting down; no order was sent.",
        )

    # ------------------------------------------------------------------ shutdown

    async def shutdown(self, *, reason: str) -> None:
        """Safe shutdown (sec. 54.2 subset). Never closes positions."""
        if self.lifecycle.mode is not SystemMode.SHUTTING_DOWN:
            await self.lifecycle.transition(SystemMode.SHUTTING_DOWN, reason=reason)
        try:
            positions = await self._c.broker.get_positions()
            orders = await self._c.broker.get_open_orders()
            report: dict[str, Any] = {
                "positions": [{"symbol": p.symbol, "qty": str(p.qty)} for p in positions],
                "open_orders": [
                    {
                        "symbol": o.symbol,
                        "client_order_id": o.client_order_id,
                        "status": str(o.status),
                    }
                    for o in orders
                ],
            }
        except Exception as exc:  # noqa: BLE001 - shutdown continues (sec. 54.2)
            report = {"error": _describe_error(exc)}
        report["reason"] = reason
        _LOGGER.info("shutdown report", extra=fields(event=EVENT_SHUTDOWN_REPORT, **report))
        await self._record(EVENT_SHUTDOWN_REPORT, report)
        await self._c.aclose(notifier_timeout=NOTIFIER_SHUTDOWN_TIMEOUT_SECONDS)
        _LOGGER.info("process stopped", extra=fields(event="PROCESS_STOPPED"))


# --------------------------------------------------------------------------- entry point


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``python -m app.main``."""
    parser = argparse.ArgumentParser(
        prog="python -m app.main", description="Trading agent runtime (Phase 2: observation)."
    )
    parser.add_argument("--config", type=Path, required=True, help="config.yaml path")
    parser.add_argument(
        "--env-file", type=Path, default=DEFAULT_ENV_FILE, help="secrets file (default: .env)"
    )
    parser.add_argument(
        "--once", action="store_true", help="startup and one health cycle, then shut down"
    )
    return parser


@contextlib.contextmanager
def _stop_on_signals(stop_event: asyncio.Event, loop: asyncio.AbstractEventLoop) -> Iterator[None]:
    """SIGINT/SIGTERM (and SIGBREAK on Windows) set ``stop_event`` instead of killing."""

    def handler(signum: int, frame: FrameType | None) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    installed: dict[int, Any] = {}
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            installed[number] = signal.signal(number, handler)
        except (ValueError, OSError) as exc:  # not the main thread / unsupported
            _LOGGER.debug("signal handler not installed", extra=fields(signal=name, error=str(exc)))
    try:
        yield
    finally:
        for number, previous in installed.items():
            signal.signal(number, previous)


async def run(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    overrides: AdapterOverrides | None = None,
    stop_event: asyncio.Event | None = None,
) -> int:
    """Run the process; returns the exit code. Keyword arguments exist for tests."""
    args = build_parser().parse_args(argv)
    try:
        loaded: LoadedConfig = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    config = loaded.config
    paths = resolve_runtime_paths(loaded)
    redactor = Redactor()
    try:
        log_file = configure_logging(
            paths.log_dir, redactor=redactor, retention_days=config.retention.logs_days
        )
    except OSError as exc:
        print(f"ERROR: cannot create log directory {paths.log_dir}: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    try:
        _LOGGER.info(
            "configuration loaded",
            extra=fields(
                event="CONFIG_LOADED",
                config_path=str(loaded.path),
                config_version=config.config_version,
                config_hash=loaded.config_hash,
                strategy_version=config.strategy_version,
                risk_version=config.risk_version,
                prompt_version=config.prompt_version,
                pending_owner_decisions=len(pending_owner_decisions(config)),
                log_file=str(log_file),
            ),
        )
        try:
            secrets = load_secrets(args.env_file, environ=environ)
        except SecretsError as exc:
            _LOGGER.error(
                "secrets refused",
                extra=fields(event="SECRETS_INVALID", code=exc.code, error=str(exc)),
            )
            return EXIT_CONFIG_ERROR
        redactor.add(secrets.alpaca_api_key, secrets.alpaca_secret_key)
        try:
            container = build_container(
                loaded,
                secrets,
                env_file=args.env_file,
                environ=environ,
                overrides=overrides,
                redactor=redactor,
            )
        except Exception as exc:  # noqa: BLE001 - reported; the process must not start
            _LOGGER.error(
                "adapters could not be built",
                extra=fields(event="CONTAINER_FAILED", error=_describe_error(exc)),
            )
            return EXIT_CONFIG_ERROR
        runtime = ObservationRuntime(container)
        event = stop_event or asyncio.Event()
        with _stop_on_signals(event, asyncio.get_running_loop()):
            return await runtime.run(event, once=args.once)
    finally:
        close_logging()


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point."""
    return asyncio.run(run(argv))


if __name__ == "__main__":
    sys.exit(main())
