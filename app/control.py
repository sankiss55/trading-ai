"""Control CLI (sec. 31.4): the human's only write path to ``system_control``.

Usage (the common options may also follow the command)::

    python -m app.control --config config.yaml --db data/paper/trading.db [--env-file .env] status
    python -m app.control ... enable-trading --reason "..."
    python -m app.control ... disable-trading --reason "..."
    python -m app.control ... set-ai-mode DISABLED|SHADOW|ACTIVE --reason "..."
    python -m app.control ... emergency-close --reason "..." [--clear]
    python -m app.control ... stop-file create|remove --reason "..."

Exit codes: ``0`` done, ``1`` refused by a control rule, ``2`` usage, configuration or
database error.

Rules (``application/control_rules.py``):

* ``enable-trading`` refuses, printing every reason and EXACTLY which ``OWNER_DECISION``
  parameters are missing (sec. 7.4.3, AC-20); it also refuses in ``APP_ENV=live``, with
  the STOP file present, with ``emergency_close`` set, or when the broker / broker clock /
  database health check is not OK (the check runs now, against the ``APP_ENV`` adapters).
* ``set-ai-mode ACTIVE`` refuses until an approved shadow evaluation (sec. 46.4) is
  recorded; that record does not exist before Phase 9, so it always refuses for now.
* ``emergency-close`` sets ``emergency_close`` and clears ``trading_enabled``; the
  procedure of sec. 31.3 is Phase 5. ``--clear`` resets the flag (manual action only,
  sec. 31.3.7); trading stays disabled until ``enable-trading``.
* ``stop-file`` creates or removes ``system.control_stop_file`` (relative to the config
  file). It works even when the database is unavailable (sec. 31.2, last resort).

Every change is written together with a ``CONTROL_CHANGED`` row of ``system_events``
in one transaction, and every refusal is recorded as ``CONTROL_REFUSED`` (sec. 31.1).
The rule is re-evaluated inside the write transaction against the row being replaced.
Claude never has access to this CLI (sec. 31.4).
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TextIO

from adapters.clock.system_clock import SystemClock
from app.config import ConfigError, LoadedConfig, load_config, pending_owner_decisions
from app.container import (
    build_exchange_adapters,
    close_quietly,
    initialize_sqlite,
    resolve_runtime_paths,
    sqlite_uow_factory,
)
from app.lifecycle import UnitOfWorkFactory, record_system_event
from app.notifier_secrets import read_env_value
from app.secrets import DEFAULT_ENV_FILE, load_secrets
from application.control_rules import (
    ControlDecision,
    can_enable_trading,
    can_set_ai_mode,
    trading_block_reasons,
)
from application.health import HealthReport, run_health_checks
from domain.errors import DomainError
from domain.models import AIMode, SystemControl, SystemEvent
from domain.ports import IClock

__all__ = [
    "EVENT_CONTROL_CHANGED",
    "EVENT_CONTROL_REFUSED",
    "EVENT_STOP_FILE_CREATED",
    "EVENT_STOP_FILE_REMOVED",
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_REFUSED",
    "HealthProvider",
    "build_parser",
    "main",
]

EVENT_CONTROL_CHANGED: Final = "CONTROL_CHANGED"
EVENT_CONTROL_REFUSED: Final = "CONTROL_REFUSED"
EVENT_STOP_FILE_CREATED: Final = "STOP_FILE_CREATED"
EVENT_STOP_FILE_REMOVED: Final = "STOP_FILE_REMOVED"

EXIT_OK: Final = 0
EXIT_REFUSED: Final = 1
EXIT_ERROR: Final = 2

OBSERVATION_NOTE: Final = (
    "note: this build runs in observation mode only (Phase 2): no order is ever submitted"
)

HealthProvider = Callable[[], Awaitable[HealthReport]]
"""Runs the health checks used by ``enable-trading`` (raises if they cannot run)."""


class _RefusedError(Exception):
    """A control rule refused the change inside the write transaction."""

    def __init__(self, decision: ControlDecision) -> None:
        super().__init__("refused")
        self.decision = decision


# --------------------------------------------------------------------------- parser


def _add_common(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    default: Any = argparse.SUPPRESS if suppress else None
    parser.add_argument("--config", type=Path, default=default, help="config.yaml path")
    parser.add_argument("--db", type=Path, default=default, help="operational SQLite DB path")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=argparse.SUPPRESS if suppress else DEFAULT_ENV_FILE,
        help="secrets file read for APP_ENV and the health check (default: .env)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the control CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m app.control", description="Runtime controls (sec. 31.4)."
    )
    _add_common(parser, suppress=False)
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common, suppress=True)
    reason = argparse.ArgumentParser(add_help=False)
    reason.add_argument("--reason", required=True, help="why (recorded in the audit trail)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", parents=[common], help="show the runtime controls")
    commands.add_parser("enable-trading", parents=[common, reason], help="allow new entries")
    commands.add_parser("disable-trading", parents=[common, reason], help="block new entries")
    ai_mode = commands.add_parser("set-ai-mode", parents=[common, reason], help="set ai_mode")
    ai_mode.add_argument("mode", choices=[mode.value for mode in AIMode])
    emergency = commands.add_parser(
        "emergency-close", parents=[common, reason], help="set (or --clear) emergency_close"
    )
    emergency.add_argument("--clear", action="store_true", help="reset emergency_close")
    stop = commands.add_parser("stop-file", parents=[common, reason], help="STOP file")
    stop.add_argument("action", choices=["create", "remove"])
    return parser


# --------------------------------------------------------------------------- context


@dataclass(frozen=True, slots=True)
class _Context:
    loaded: LoadedConfig
    db_path: Path
    stop_file: Path
    env_file: Path
    app_env: str | None
    uow_factory: UnitOfWorkFactory | None
    clock: IClock
    health_provider: HealthProvider
    operator: str
    out: TextIO
    err: TextIO


def _operator() -> str:
    try:
        return f"cli:{getpass.getuser()}"
    except (OSError, KeyError, ImportError):
        return "cli:unknown"


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, DomainError):
        return f"{type(exc).__name__}: {exc}"
    return type(exc).__name__


def _default_health_provider(
    loaded: LoadedConfig,
    env_file: Path,
    environ: Mapping[str, str] | None,
    clock: IClock,
    uow_factory: UnitOfWorkFactory,
    stop_file: Path,
) -> HealthProvider:
    async def provide() -> HealthReport:
        secrets = load_secrets(env_file, environ=environ)
        exchange = build_exchange_adapters(secrets.app_env, secrets, loaded.config, clock)
        try:
            return await run_health_checks(
                broker=exchange.broker,
                calendar=exchange.calendar,
                clock=clock,
                uow=uow_factory(),
                max_clock_skew_seconds=loaded.config.system.max_clock_skew_seconds,
                pending_decisions=pending_owner_decisions(loaded.config),
                stop_file_present=stop_file.exists(),
            )
        finally:
            for component in (exchange.broker, exchange.calendar, exchange.market_data):
                if component is not None:
                    await close_quietly(component)

    return provide


# --------------------------------------------------------------------------- helpers


def _print_decision(ctx: _Context, action: str, decision: ControlDecision) -> None:
    print(f"REFUSED: {action}", file=ctx.out)
    for refusal in decision.refusals:
        print(f"- {refusal.code}: {refusal.message}", file=ctx.out)
        for item in refusal.details:
            print(f"    - {item}", file=ctx.out)
    missing = decision.missing_parameters
    if missing:
        print(f"Missing OWNER_DECISION parameters ({len(missing)}):", file=ctx.out)
        for path in missing:
            print(f"  {path}", file=ctx.out)


def _event(ctx: _Context, event_type: str, detail: dict[str, Any]) -> SystemEvent:
    return SystemEvent(
        occurred_at_utc=ctx.clock.now_utc(),
        event_type=event_type,
        detail={"by": ctx.operator, **detail},
    )


async def _record_best_effort(ctx: _Context, event: SystemEvent) -> bool:
    if ctx.uow_factory is None:
        print(f"warning: no database: audit event {event.event_type} not written", file=ctx.err)
        return False
    try:
        await record_system_event(ctx.uow_factory, event)
    except Exception as exc:  # noqa: BLE001 - the audit write is best effort here
        print(
            f"warning: audit event {event.event_type} not written: {_describe_error(exc)}",
            file=ctx.err,
        )
        return False
    return True


def _control_view(control: SystemControl) -> dict[str, Any]:
    return control.model_dump(mode="json")


async def _change_control(
    ctx: _Context,
    *,
    command: str,
    reason: str,
    changes: Mapping[str, Any],
    check: Callable[[SystemControl], ControlDecision] | None = None,
) -> SystemControl:
    """Replace ``system_control`` and append ``CONTROL_CHANGED`` in one transaction.

    Raises:
        _RefusedError: ``check`` refused against the row being replaced (nothing written).
    """
    if ctx.uow_factory is None:
        raise FileNotFoundError(ctx.db_path)
    async with ctx.uow_factory() as uow:
        before = await uow.control.get()
        if check is not None:
            decision = check(before)
            if not decision.allowed:
                raise _RefusedError(decision)
        after = SystemControl.model_validate(
            {
                **before.model_dump(),
                **changes,
                "updated_at_utc": ctx.clock.now_utc(),
                "updated_by": ctx.operator,
                "reason": reason,
            }
        )
        await uow.control.set(after)
        await uow.system_events.append(
            _event(
                ctx,
                EVENT_CONTROL_CHANGED,
                {
                    "command": command,
                    "reason": reason,
                    "before": _control_view(before),
                    "after": _control_view(after),
                },
            )
        )
        await uow.commit()
    return after


async def _refuse(
    ctx: _Context, *, command: str, reason: str, action: str, decision: ControlDecision
) -> int:
    _print_decision(ctx, action, decision)
    await _record_best_effort(
        ctx,
        _event(
            ctx,
            EVENT_CONTROL_REFUSED,
            {
                "command": command,
                "reason": reason,
                "refusals": [refusal.model_dump(mode="json") for refusal in decision.refusals],
            },
        ),
    )
    return EXIT_REFUSED


async def _apply(
    ctx: _Context,
    *,
    command: str,
    reason: str,
    action: str,
    changes: Mapping[str, Any],
    check: Callable[[SystemControl], ControlDecision] | None = None,
) -> SystemControl | int:
    try:
        return await _change_control(
            ctx, command=command, reason=reason, changes=changes, check=check
        )
    except _RefusedError as refused:
        return await _refuse(
            ctx, command=command, reason=reason, action=action, decision=refused.decision
        )
    except Exception as exc:  # noqa: BLE001 - reported to the operator with a fallback
        print(f"ERROR: system_control not written: {_describe_error(exc)}", file=ctx.err)
        print(
            "If the database is unavailable, block trading with the STOP file: "
            'python -m app.control ... stop-file create --reason "..."',
            file=ctx.err,
        )
        return EXIT_ERROR


def _print_control(ctx: _Context, control: SystemControl) -> None:
    print(f"trading_enabled: {str(control.trading_enabled).lower()}", file=ctx.out)
    print(f"emergency_close: {str(control.emergency_close).lower()}", file=ctx.out)
    print(f"ai_mode: {control.ai_mode}", file=ctx.out)


# --------------------------------------------------------------------------- commands


async def _status(ctx: _Context) -> int:
    if ctx.uow_factory is None:
        print(f"ERROR: database not found: {ctx.db_path}", file=ctx.err)
        return EXIT_ERROR
    async with ctx.uow_factory() as uow:
        control = await uow.control.get()
    config = ctx.loaded.config
    stop_present = ctx.stop_file.exists()
    pending = pending_owner_decisions(config)
    blocks = trading_block_reasons(control, stop_file_present=stop_present)
    print(
        f"config: {ctx.loaded.path} (config_version {config.config_version}, "
        f"sha256 {ctx.loaded.config_hash[:12]})",
        file=ctx.out,
    )
    print(f"database: {ctx.db_path}", file=ctx.out)
    print(f"app_env: {ctx.app_env or 'unset'}", file=ctx.out)
    _print_control(ctx, control)
    updated = control.updated_at_utc.isoformat() if control.updated_at_utc else "never"
    print(f"updated_at_utc: {updated}", file=ctx.out)
    print(f"updated_by: {control.updated_by or '-'}", file=ctx.out)
    print(f"reason: {control.reason or '-'}", file=ctx.out)
    print(f"stop_file: {ctx.stop_file} ({'PRESENT' if stop_present else 'absent'})", file=ctx.out)
    effective = "enabled" if not blocks else f"disabled ({', '.join(blocks)})"
    print(f"effective_trading: {effective}", file=ctx.out)
    print(f"pending_owner_decisions: {len(pending)}", file=ctx.out)
    for path in pending:
        print(f"  {path}", file=ctx.out)
    print(OBSERVATION_NOTE, file=ctx.out)
    return EXIT_OK


async def _enable_trading(ctx: _Context, reason: str) -> int:
    pending = pending_owner_decisions(ctx.loaded.config)
    health: HealthReport | None = None
    try:
        health = await ctx.health_provider()
    except Exception as exc:  # noqa: BLE001 - reported as HEALTH_UNKNOWN
        print(f"health check could not run: {_describe_error(exc)}", file=ctx.out)
    stop_present = ctx.stop_file.exists()

    def check(control: SystemControl) -> ControlDecision:
        return can_enable_trading(
            pending_decisions=pending,
            environment=ctx.app_env or "",
            control=control,
            stop_file_present=stop_present,
            health=health,
        )

    result = await _apply(
        ctx,
        command="enable-trading",
        reason=reason,
        action="trading cannot be enabled",
        changes={"trading_enabled": True},
        check=check,
    )
    if isinstance(result, int):
        return result
    print("trading_enabled set to true", file=ctx.out)
    print(OBSERVATION_NOTE, file=ctx.out)
    return EXIT_OK


async def _disable_trading(ctx: _Context, reason: str) -> int:
    result = await _apply(
        ctx,
        command="disable-trading",
        reason=reason,
        action="trading cannot be disabled",
        changes={"trading_enabled": False},
    )
    if isinstance(result, int):
        return result
    print("trading_enabled set to false", file=ctx.out)
    return EXIT_OK


async def _set_ai_mode(ctx: _Context, mode: AIMode, reason: str) -> int:
    # No query port for approval records exists before Phase 9 (shadow report): fail closed.
    decision = can_set_ai_mode(mode, shadow_approval_recorded=False)
    if not decision.allowed:
        return await _refuse(
            ctx,
            command="set-ai-mode",
            reason=reason,
            action=f"ai_mode cannot be set to {mode}",
            decision=decision,
        )
    result = await _apply(
        ctx,
        command="set-ai-mode",
        reason=reason,
        action=f"ai_mode cannot be set to {mode}",
        changes={"ai_mode": mode},
    )
    if isinstance(result, int):
        return result
    print(f"ai_mode set to {mode}", file=ctx.out)
    if mode is not AIMode.DISABLED:
        print(
            "note: the AI filter is built in Phase 8; ai_mode has no effect until then",
            file=ctx.out,
        )
    return EXIT_OK


async def _emergency_close(ctx: _Context, reason: str, *, clear: bool) -> int:
    changes: dict[str, Any] = (
        {"emergency_close": False} if clear else {"emergency_close": True, "trading_enabled": False}
    )
    result = await _apply(
        ctx,
        command="emergency-close --clear" if clear else "emergency-close",
        reason=reason,
        action="emergency_close cannot be changed",
        changes=changes,
    )
    if isinstance(result, int):
        return result
    if clear:
        print("emergency_close cleared; trading stays disabled until enable-trading", file=ctx.out)
        return EXIT_OK
    print("emergency_close set to true and trading_enabled set to false", file=ctx.out)
    print(
        "Phase 2: the emergency procedure (cancel entries, close positions, verify; "
        "sec. 31.3) is not implemented yet. The running process enters EMERGENCY mode. "
        "Check the broker account manually.",
        file=ctx.out,
    )
    return EXIT_OK


async def _stop_file(ctx: _Context, action: str, reason: str) -> int:
    path = ctx.stop_file
    now = ctx.clock.now_utc()
    if action == "create":
        already = path.exists()
        if not already:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"STOP created_at_utc={now.isoformat()} by={ctx.operator}\nreason: {reason}\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                print(f"ERROR: STOP file not created: {_describe_error(exc)}", file=ctx.err)
                return EXIT_ERROR
        await _record_best_effort(
            ctx,
            _event(
                ctx,
                EVENT_STOP_FILE_CREATED,
                {"path": str(path), "reason": reason, "already_present": already},
            ),
        )
        state = "already present" if already else "created"
        print(f"STOP file {state}: {path}", file=ctx.out)
        print("trading is disabled while it exists (sec. 31.2)", file=ctx.out)
        return EXIT_OK
    existed = path.exists()
    if existed:
        try:
            path.unlink()
        except OSError as exc:
            print(f"ERROR: STOP file not removed: {_describe_error(exc)}", file=ctx.err)
            return EXIT_ERROR
    await _record_best_effort(
        ctx,
        _event(
            ctx,
            EVENT_STOP_FILE_REMOVED,
            {"path": str(path), "reason": reason, "was_present": existed},
        ),
    )
    print(f"STOP file {'removed' if existed else 'was not present'}: {path}", file=ctx.out)
    return EXIT_OK


async def _dispatch(ctx: _Context, args: argparse.Namespace) -> int:
    command: str = args.command
    if command == "status":
        return await _status(ctx)
    reason = str(args.reason).strip()
    if not reason:
        print("ERROR: --reason must not be blank", file=ctx.err)
        return EXIT_ERROR
    if command == "enable-trading":
        return await _enable_trading(ctx, reason)
    if command == "disable-trading":
        return await _disable_trading(ctx, reason)
    if command == "set-ai-mode":
        return await _set_ai_mode(ctx, AIMode(args.mode), reason)
    if command == "emergency-close":
        return await _emergency_close(ctx, reason, clear=bool(args.clear))
    return await _stop_file(ctx, str(args.action), reason)


# --------------------------------------------------------------------------- entry point


async def _run(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None,
    uow_factory: UnitOfWorkFactory | None,
    clock: IClock | None,
    health_provider: HealthProvider | None,
    out: TextIO,
    err: TextIO,
) -> int:
    try:
        loaded = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=err)
        return EXIT_ERROR
    paths = resolve_runtime_paths(loaded)
    db_path: Path = args.db
    if db_path.resolve() != paths.db_path.resolve():
        print(
            f"warning: --db {db_path} differs from system.db_path ({paths.db_path}) "
            "used by the runtime",
            file=err,
        )
    the_clock = clock or SystemClock()
    factory = uow_factory
    if factory is None and db_path.is_file():
        try:
            await initialize_sqlite(db_path)
        except Exception as exc:  # noqa: BLE001 - reported to the operator
            print(f"ERROR: cannot open database {db_path}: {_describe_error(exc)}", file=err)
            return EXIT_ERROR
        factory = sqlite_uow_factory(db_path, the_clock)
    if factory is None and args.command != "stop-file":
        print(
            f"ERROR: database not found: {db_path} (start the runtime once to create it, "
            "or check --db)",
            file=err,
        )
        return EXIT_ERROR
    env_file: Path = args.env_file
    provider = health_provider
    if provider is None and factory is not None:
        provider = _default_health_provider(
            loaded, env_file, environ, the_clock, factory, paths.stop_file
        )

    async def no_health() -> HealthReport:
        raise FileNotFoundError(db_path)

    ctx = _Context(
        loaded=loaded,
        db_path=db_path,
        stop_file=paths.stop_file,
        env_file=env_file,
        app_env=read_env_value("APP_ENV", env_file, environ=environ),
        uow_factory=factory,
        clock=the_clock,
        health_provider=provider or no_health,
        operator=_operator(),
        out=out,
        err=err,
    )
    return await _dispatch(ctx, args)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    uow_factory: UnitOfWorkFactory | None = None,
    clock: IClock | None = None,
    health_provider: HealthProvider | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Run one control command; returns the process exit code.

    The keyword arguments exist for tests (fake unit of work, clock, health and streams).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "config", None) is None or getattr(args, "db", None) is None:
        parser.error("--config and --db are required")
    return asyncio.run(
        _run(
            args,
            environ=environ,
            uow_factory=uow_factory,
            clock=clock,
            health_provider=health_provider,
            out=out or sys.stdout,
            err=err or sys.stderr,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
