# trading-agent

An algorithmic trading system for US equities and ETFs, **paper trading only**.
The master spec (v2.1) is the single source of truth.

- A **deterministic strategy** generates BUY signals from explicit, backtestable rules.
- An **AI veto filter** (Anthropic model) can only APPROVE or VETO a signal. It never
  originates trades and never sets prices, quantities or symbols.
- A **deterministic layer** handles validation, risk, execution, persistence,
  reconciliation and recovery. **Alpaca** is the broker and the source of truth for
  execution.

## Architecture

Hexagonal (sec. 8.2). Dependencies point one way only:

```text
app/          composition root: builds adapters, injects them, starts tasks
application/  use cases: orchestrate domain logic through ports
domain/       immutable models, pure logic and ports (typing.Protocol); no infrastructure
adapters/     Alpaca, Claude Code CLI, SQLite, SMTP, clock and simulation (implement ports)
backtest/     backtest runner and reports
shadow/       shadow-mode counterfactual outcomes and report
```

The dependency rules (sec. 8.3) are enforced by `tests/architecture/` (AC-21) and by
import-linter contracts in `pyproject.toml`.

## Setup

Requires Python 3.11+. The AI filter runs through the Claude Code CLI with the machine's
logged-in Claude session: no Anthropic API key (see `docs/DECISIONS.md`).

```bash
python -m venv .venv            # or: python -m virtualenv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows
# .venv/bin/python -m pip install -e ".[dev]"     # Linux/macOS
cp .env.example .env            # fill in secrets; .env is never committed
```

Configuration: secrets and `APP_ENV` live in `.env`; everything else lives in the
versioned `config.yaml`. Every `OWNER_DECISION` value stays `null` until the owner
sets it in `docs/DECISIONS.md`; a `null` blocks enabling trading.

## Checks

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m mypy domain
.venv/Scripts/lint-imports
```

## Phase status (sec. 55)

| Phase | Status |
| --- | --- |
| 0. Specification (owner decisions) | In progress: owner decisions of 2026-10-02 recorded in `config.yaml` 2.2.0; still open: `ai.model_id` (VERIFICAR) and the notification-policy item of sec. 59 |
| 1. Strategy and backtest | In progress: models, ports, errors and architecture test done |
| 2-11 | Not started |

Each phase ends with an explicit owner approval recorded in `docs/DECISIONS.md`.
No backtest or paper result implies future profitability.
