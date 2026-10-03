# Handoff — state of the project and how to continue

Last updated: 2026-10-03. Written for the next Claude Code session (or any engineer) taking over.
Read this file first, then `docs/DECISIONS.md` (all owner decisions and deviations) and the master
spec `D:\Descargas\DOCUMENTO_MAESTRO_TRADING_AGENT_v2.1.md` (Spanish; single source of truth).

## 1. Working agreement with the owner (mandatory)

- **Owner decisions are the owner's.** Never invent a value marked `OWNER_DECISION`; propose with
  pros/cons, ask ONE question at a time, wait. Record every decision with date in `docs/DECISIONS.md`.
- **Phase gates (sec. 55 / 56.4):** finish a phase, report evidence, STOP, wait for explicit approval,
  record it in the "Phase approvals" table. Never start the next phase without approval.
- **Git:** push every *verified* change directly to `main` on `https://github.com/sankiss55/trading-ai.git`
  (owner instruction). No PRs. Conventional commits, **no AI attribution / no Co-Authored-By**.
  Repo-local identity `sankiss55 <sanchezvera490@gmail.com>` (already configured).
- **Verification before every push** — use `set -o pipefail` (a `pytest | tail && git commit` chain once
  pushed failing tests):
  ```bash
  set -o pipefail
  .venv/Scripts/python -m pytest -q -p no:warnings | tail -1
  .venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
  .venv/Scripts/python -m mypy domain application adapters backtest app
  .venv/Scripts/lint-imports
  .venv/Scripts/python -m backtest.parity --config research/configs/mr_a1_ibs.yaml --data data/alpaca_sip_daily_split
  ```
  Then scan staged content for secrets and make sure no `.env`, `data/`, `logs/`, `control/`, `*.db` is staged.
- **Secrets:** `.env` holds Alpaca PAPER keys and Gmail app password. The owner's Claude settings deny
  Read/Edit of `.env` — do not work around it. App code loads it at runtime via `app/secrets.py` and
  `app/notifier_secrets.py`. Live trading is blocked in code (`APP_ENV=live` refused) and must stay so.
- **Conversation language with the owner:** Spanish (Rioplatense, voseo). All artifacts in English.
- **Delegation:** the owner wants work delegated to subagents with disjoint file ownership; the
  coordinator verifies, records decisions and pushes. Subagents hit usage limits often — resume them
  with their context (SendMessage) and have them re-check files for truncation.

## 2. Environment

- Project root: `C:\Users\opc\Desktop\trading-agent` (Windows Server, Git Bash + PowerShell).
- Python 3.12 embedded build (no `venv` module): `.venv` was created with `virtualenv`. Install:
  `.venv/Scripts/python -m pip install -e ".[dev]"`.
- Datasets (git-ignored, local only):
  - `data/alpaca_iex_split/` — IEX 1Min + SIP 1Day, 8 symbols, 2021-12..2026-06 (strategy v1, intraday).
  - `data/alpaca_sip_daily_split/` — SIP 1Day, SPY/QQQ/IWM, 2016-01-04..2026-06-30 (family v2, daily).
  - Paper DB: `data/paper/trading.db` (SQLite, WAL). Logs: `logs/paper/trading-agent.jsonl`.
- Alpaca paper account: equity 100,000, buying power 4x (margin) — sizing stays cash-only.

## 3. Architecture (hexagonal, enforced by tests)

`domain/` (pure: models, ports, market, strategy, guards, risk) → `application/` (use cases, ports only,
time via `IClock`) → `adapters/` (alpaca, sqlite, smtp, clock, simulation) → `app/` (composition root,
the only place that builds adapters). `tests/architecture` + import-linter forbid SDKs, I/O, wall clock and
`subprocess` in domain/application. One implementation per indicator (`domain/market/indicators.py`) and
one check library (`domain/guards/checks.py`).

## 4. Status by phase

| Phase | Status | Evidence / notes |
|---|---|---|
| 0 Spec | Done | All sec. 59 decisions taken (see DECISIONS.md) |
| 1 Strategy & backtest | **Exit criterion NOT met** | v1 (5-min EMA intraday) rejected; family v2 (daily ETFs): 7 pre-registered hypotheses all FAIL dev gates; `mr_a1_ibs` failed the lockbox and is discarded. Owner decided (2026-10-02) to build the paper system anyway |
| 2 Infrastructure | **Approved 2026-10-03** | `python -m app.main --once` reaches READY on paper, trading disabled |
| 3 Live market data | **Approved 2026-10-03** | Daily-only (no minute stream). `python -m backtest.parity --live` PASS |
| 4 Risk & safety | **Approved 2026-10-03** | Check library + circuit breaker; AC-05/AC-15 logic tests |
| **5 Execution** | **NEXT — approved to start; may submit TEST orders to the PAPER account** | See §6 |
| 6 Failure tests & simulation | Pending | sec. 49.3, 49.4, 49.5 |
| 7 Paper rules-only | Pending — **strategy to run is an open owner decision** | see §7 |
| 8 AI shadow | Pending | Claude Code CLI session, no API key (DECISIONS "AI provider") |
| 9-10 | Pending | Shadow evaluation, AI active |
| 11 Live gate | Out of MVP | Blocked in code |

Test suite at handoff: **1851 passed, 4 skipped** (skips = Alpaca trade-update stream, Phase 5).

## 5. Research protocol (Phase 1, family v2)

- Protocol: `research/protocol.yaml` (dev 2016-11..2022-12, lockbox 2023-01..2026-06, embargo 2026-07+,
  budget 12 trials, gate sample `continuous_walk_forward_span`). Hypotheses: `research/hypotheses/`;
  configs: `research/configs/`; registry: `research/trials.jsonl` (8 used incl. v1; 4 left);
  results: `research/results/`.
- CLI: `python -m backtest.research_cli list|dev|lockbox|report` (lockbox needs a clean committed tree and
  opens once per hypothesis). Official/segmented backtests: `python -m backtest --config ... --data ... --starting-cash 100000 [--mode segmented] [--workers N]`.
- Lesson: never change acceptance criteria after seeing results; "nothing passes" is a valid outcome.

## 6. Phase 5 — execution (what to build next)

Spec: sec. 20 (execution guard), 21 (state machine), 22 (idempotency), 23 (brackets, exits, 23.6 system
exit, 23.7 unprotected position), 24 (reconciliation), 25 (startup), 26 (crash recovery), 27-28 (tasks,
queues, locks), 29, 31.3 (emergency close), 38.2 (audit trace), 39 (P&L), 54 (shutdown), 55 (exit: AC-02,
AC-03, AC-08..AC-12, AC-16 in paper; AC-22).

Work items (from the Phase 2-4 reports):
1. `domain/execution/`: `state_machine.py` (sec. 21 transitions; move `FINAL_STATES` here from
   `adapters/sqlite/semantics.py`), `idempotency.py` (deterministic trade_id / client_order_id ≤ 128 chars),
   `reconciliation.py` (pure DB vs broker diff), `pnl.py` (sec. 39).
2. Port gaps (`domain/ports/persistence.py` + both UoW adapters + contract suite): read a signal's status;
   read fills (P&L); list orders by trade (TP/SL legs); `shadow_outcomes.exists` (no double count).
3. `application/execution_guard.py` (`execute_signal`): symbol lock → refresh broker state → re-read
   `system_control` + STOP file → `RiskGate.context` → `run_execution_checks` with `ExecutionFacts`
   (client order id unused in DB and broker, lock held, AI result) → persist all results in `risk_events` →
   commit `SUBMITTING` BEFORE calling the broker → submit bracket (GTC for swing) → record order.
4. `application/trade_updates.py` (`on_trade_update`), `exit_procedures.py` (`run_system_exit`: cancel legs
   then market sell; `protect_position`; `emergency_close`; `flatten_end_of_day` for intraday),
   `reconcile.py`, `startup.py` (sec. 25 + crash recovery 26; `UNKNOWN_SUBMISSION` resolution by
   client_order_id lookup).
5. `adapters/alpaca/broker.py`: implement `stream_trade_updates` (alpaca `TradingStream`), plus polling
   reconciliation fallback. VERIFICAR on paper: duplicate `client_order_id` scope after an order is final;
   whether POST / by-client-id responses include legs; filled parent with open legs in `status=open&nested`;
   rejection message texts; `pending_review`; rate limit 200/min.
6. `app/orchestrator.py` + real `GuardEnvironment` in `app/`: daily schedule — after close + 20 min run
   `application/daily_cycle.py` `run_session_close`; store pending signals (expire next open + TTL);
   at next open run `execute_signal`; control watcher every `control_poll_seconds` (AC-15 timing);
   breaker runtime (persist `BreakerStatus`, transitions → `system_events` + notification, procedures
   23.7/31.3, breaker → system mode); periodic reconciliation; `clear-halt` CLI on `can_clear/clear_halt`;
   prefetch `AlpacaCalendar.get_sessions` before warm-up; per-symbol `BookSource`;
   `TradeBook.executed_signal_ids` from the DB; move `guard_settings` into `app/container.py`;
   wire live quotes and switch the spread filter to ENFORCED; config key for the daily data delay.
7. AC-22: run execution guard + breaker inside the backtest runner (same code path as paper); re-run the
   backtests and explain any change.
8. Simulator parity: Alpaca cancels the whole bracket group when one leg is cancelled; `SimulatedBroker`
   cancels only the leg — align (sec. 49.2).
9. Paper verification with TEST orders (owner approved): small whole-share orders on SPY during regular
   hours, bracket accept → fill → legs → cancel/exit, duplicate client_order_id, restart mid-order
   (crash recovery), reconciliation. Never leave positions unprotected; clean up after tests.

## 7. Open owner decisions

- **Which strategy runs in Phase 7 paper trading.** None is validated. Options to present: run a rejected
  strategy purely as a system load test (labelled as such), use remaining research budget (4 trials) for a
  pre-registered new idea (e.g. MR-IBS + trend portfolio combination), or paper-test only the machinery.
- Pending confirmations listed in DECISIONS.md "Technical decisions pending owner confirmation" and the
  Phase 4 rows (WARNING blocks entries, manual-only resets, strict expiry).
- Real test email from the SMTP notifier has never been sent; ask before doing it.

## 8. Ideas the owner asked about (later phases)

- AI veto in SHADOW mode via the Claude Code CLI (`claude -p --output-format json`, like
  `C:\Users\opc\Desktop\prospectos_whatsapp\src\adapters\ai\claude-cli.ts`), news headlines from Alpaca
  (VERIFICAR), measured per sec. 46 before any ACTIVE use. Claude never originates trades (sec. 4).
- Read-only MCP server so the owner can ask Claude about positions/P&L (sec. 37, after Phase 7).
- Be honest with the owner: AI does not create an edge by itself; buy-and-hold is the benchmark to beat.
