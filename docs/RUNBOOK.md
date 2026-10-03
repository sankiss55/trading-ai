# Runbook

Operational procedures (master spec v2.1, sec. 58.3). Each section is completed in
the phase that delivers the corresponding capability.

**Phase 2 status: observation mode only.** `python -m app.main` starts, synchronizes with
the broker, reaches `READY` and watches. It never submits, cancels or modifies an order and
never reaches `RUNNING`, whatever `system_control.trading_enabled` says. Live trading is
blocked (`APP_ENV=live` refuses to start).

Conventions used below:

- Commands run from the project root with the virtual environment's Python
  (`.venv/Scripts/python` on Windows, `.venv/bin/python` elsewhere), shown as `python`.
- Relative paths in `config.yaml` (`system.db_path`, `system.log_dir`,
  `system.control_stop_file`) are resolved against the directory of the config file,
  not the working directory. With the repository `config.yaml` (paper):
  DB `data/paper/trading.db`, logs `logs/paper/trading-agent.jsonl`, STOP file `control/STOP`.
- `CTL` stands for `python -m app.control --config config.yaml --db data/paper/trading.db`.
  `--env-file` defaults to `.env`. The control CLI exits `0` (done), `1` (refused by a
  rule) or `2` (usage, configuration or database error).
- Every control change is recorded in `system_events` (`CONTROL_CHANGED`, refusals as
  `CONTROL_REFUSED`, STOP file actions as `STOP_FILE_CREATED` / `STOP_FILE_REMOVED`).

## Normal startup

Observation mode (Phase 2):

1. Check `.env` exists with `APP_ENV=paper` (or `dev`), the paper `ALPACA_API_KEY` /
   `ALPACA_SECRET_KEY`, and optionally `GMAIL_USER`, `GMAIL_APP_PASSWORD`
   (Google App Password) and `NOTIFICATION_EMAIL`. Without the three SMTP values the
   process starts anyway and writes notifications to the log (`SMTP_NOT_CONFIGURED`
   warning). Never commit `.env`.
2. Optional smoke run: `python -m app.main --config config.yaml --once` runs the full
   startup and one health cycle, then shuts down. Exit code `0` means `READY` was reached.
3. Start: `python -m app.main --config config.yaml`.
4. Verify in `logs/paper/trading-agent.jsonl` (one JSON object per line):
   - `CONFIG_LOADED` with `config_version` and `config_hash` (sha256 of `config.yaml`);
   - `MODE_TRANSITION` `BOOTING -> SYNCING`, then `SYNCING -> READY`;
   - `startup sync` (`SYNC_REPORT`): account status, market open, positions and open
     orders (both `0` for a clean account, AC-01), reconciliation `RECONCILED_OK`;
   - `health` with `status` `OK` (or the checks that are not OK).
5. `CTL status` shows the controls, the STOP file and the pending owner decisions.

Startup sequence (sec. 25, Phase 2 subset): load and validate config, logging, secrets
(`APP_ENV=live` refused), adapters, DB open and migrations, read `system_control` and
the STOP file (`STARTUP` event) -> `SYNCING`: broker account, market clock, today's
session, positions and open orders, equity snapshot, reconciliation (report only),
health check -> `READY`.

Exit codes: `0` clean, `1` startup failure (e.g. broker unreachable; a critical
`BROKER_ERROR`/`SYSTEM_ERROR` notification is sent and `STARTUP_FAILED` is recorded),
`2` configuration, secrets or environment error (nothing is started).

If the broker holds positions or open orders at startup, the system still reaches `READY`
(observation), records the reconciliation as `ORPHANED` / `STATE_MISMATCH` with the
differences, and sends a critical notification. It takes no action: check the paper
account manually.

`config.yaml` edited while running is ignored until the next start; a
`CONFIG_CHANGED_ON_DISK` warning is logged once.

## Normal shutdown

1. Press `Ctrl+C` in the console, or send `SIGTERM` (`SIGBREAK` / `Ctrl+Break` on Windows).
2. The process moves to `SHUTTING_DOWN`, records a `SHUTDOWN_REPORT` (broker positions and
   open orders, report only), delivers queued notifications for at most 30 s, closes the
   adapters and logs `PROCESS_STOPPED`.
3. Shutdown never closes positions; in later phases the protection stays at the broker.

## Kill switch: disable trading

Preferred (database):

```text
CTL disable-trading --reason "why"
```

Last resort (the database or the CLI do not respond): the STOP file. While it exists the
system behaves as `trading_enabled = false`, whatever the database says (sec. 31.2). The
runtime checks it every `system.control_poll_seconds` and records `STOP_FILE_DETECTED` /
`STOP_FILE_CLEARED`.

```text
CTL stop-file create --reason "why"      # also works when the database is unavailable
CTL stop-file remove --reason "why"
```

Without any working Python, create the file by hand (any content), e.g.
`New-Item control/STOP` (PowerShell) or `touch control/STOP`.

## Enable trading

```text
CTL enable-trading --reason "why"
```

It refuses (exit `1`) and prints every reason when: any `OWNER_DECISION` in
`config.yaml` is `null` (the exact dotted paths are listed under
`Missing OWNER_DECISION parameters`, AC-20), `APP_ENV=live`, the STOP file exists,
`emergency_close` is set, or the health check (broker reachable and account `ACTIVE`,
broker clock skew within `system.max_clock_skew_seconds`, database writable) is not OK
or cannot run. The health check runs at that moment with the `APP_ENV` adapters, so it
needs the `.env` keys and network access to the paper API.

In Phase 2 a successful `enable-trading` only sets the flag: the runtime logs
`TRADING_NOT_AVAILABLE` and stays in observation mode.

## Claude outage

TBD (Phase 8)

## Alpaca outage

TBD (Phase 5). Phase 2: if the broker is unreachable at startup the process exits with
code `1` after a critical notification; while running, the health check reports the
`broker` check `FAILED` (one critical `BROKER_ERROR` notification per change).

## Market data outage

TBD (Phase 3)

## Database failure

Phase 2 behavior:

- At startup, a database that cannot be opened or migrated stops the process (exit `1`).
- While running, the `database` health check becomes `FAILED` (critical `SYSTEM_ERROR`
  notification); an unreadable `system_control` is reported once and the STOP file keeps
  working.
- The control CLI prints `ERROR: system_control not written` (exit `2`) and suggests the
  STOP file.

Procedure: create the STOP file, stop the process, check disk space and file permissions
of `system.db_path`, then follow "Restore from backup" if the file is damaged. Full HALTED
handling arrives with the circuit breaker (Phase 4).

## State mismatch

TBD (Phase 5)

## Unprotected position

TBD (Phase 5)

## Risk halt

TBD (Phase 4)

## Emergency close

Phase 2 (flag only; the procedure of sec. 31.3 arrives in Phase 5):

```text
CTL emergency-close --reason "why"
```

Sets `emergency_close = true` and `trading_enabled = false`. The running process moves to
`EMERGENCY` within `control_poll_seconds` and sends a critical `EMERGENCY_CLOSE`
notification. In Phase 2 the system holds no positions of its own and sends no orders:
check the paper account at Alpaca and close anything there manually if needed.
`EMERGENCY` is left only by restarting the process.

Clearing is a manual action (sec. 31.3.7):

```text
CTL emergency-close --clear --reason "verified flat"
```

Trading stays disabled until `enable-trading`.

## Restore from backup

TBD. The database copy and restore functions exist (`adapters/sqlite/backup.py`:
`backup_database`, `restore_database`); their scheduling (`backups.frequency =
daily_after_close`, `backups.retention_days = 30`) and the operator command are not wired
yet. Restore always runs with the process stopped, followed by a normal startup (which
applies pending migrations).

## AI mode change

Phase 2: the AI filter does not exist yet (Phase 8).

```text
CTL set-ai-mode DISABLED --reason "why"
CTL set-ai-mode SHADOW --reason "why"     # accepted; no effect before Phase 8
CTL set-ai-mode ACTIVE --reason "why"     # always refused for now
```

`ACTIVE` requires an approved shadow-mode evaluation (sec. 46.4) recorded in
`system_events`; until Phase 9 produces it the CLI refuses with
`SHADOW_APPROVAL_MISSING`. At startup a non-`DISABLED` mode is logged as
`AI_MODE_UNAVAILABLE`.

## Model change

TBD (Phase 8)
