"""Migration 1: operational tables of sec. 38.1 that have a repository port (sec. 8.5).

Conventions (see ``adapters/sqlite/_codec.py``): money/prices/quantities as exact TEXT
decimals, datetimes as fixed-width ISO-8601 UTC TEXT, enums as TEXT, structured fields
as ``<field>_json`` TEXT. Tables are ``STRICT``.

Unique constraints of sec. 22.5: ``signals.signal_id``, ``trades.signal_id``,
``orders.client_order_id``; plus ``fills.activity_id`` (idempotent fills) and broker
``orders.order_id`` when known. ``order_events``, ``risk_events`` and ``system_events``
are append-only (triggers refuse UPDATE and DELETE).

Foreign keys: ``trades.signal_id -> signals`` and ``orders.trade_id -> trades``. Audit
logs (events, fills, AI decisions) deliberately have none: an audit row is never
refused for referential reasons (e.g. a fill of an orphaned order, sec. 24).
"""

from __future__ import annotations

from typing import Final

__all__ = ["NAME", "STATEMENTS", "VERSION"]

VERSION: Final = 1
NAME: Final = "initial"


def _append_only(table: str) -> tuple[str, str]:
    return (
        f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
        f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END",
        f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} "
        f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END",
    )


STATEMENTS: Final[tuple[str, ...]] = (
    # ------------------------------------------------------------------ signals
    """
    CREATE TABLE signals (
        signal_id TEXT NOT NULL PRIMARY KEY,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        bar_start_utc TEXT NOT NULL,
        bar_end_utc TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        expires_at_utc TEXT NOT NULL,
        rule_results_json TEXT NOT NULL,
        strategy_version TEXT NOT NULL,
        status TEXT NOT NULL,
        status_reason TEXT,
        status_updated_at_utc TEXT
    ) STRICT
    """,
    "CREATE INDEX ix_signals_status ON signals (status)",
    # ------------------------------------------------------------------ trades
    """
    CREATE TABLE trades (
        trade_id TEXT NOT NULL PRIMARY KEY,
        signal_id TEXT NOT NULL UNIQUE REFERENCES signals (signal_id),
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty INTEGER NOT NULL,
        filled_qty TEXT NOT NULL,
        entry_ref TEXT NOT NULL,
        stop_price TEXT NOT NULL,
        take_profit_price TEXT NOT NULL,
        entry_avg_price TEXT,
        entry_filled_at_utc TEXT,
        exit_avg_price TEXT,
        exit_filled_at_utc TEXT,
        exit_reason TEXT,
        gross_pnl TEXT,
        net_pnl TEXT,
        result_r TEXT,
        versions_json TEXT NOT NULL,
        state TEXT NOT NULL,
        updated_at_utc TEXT
    ) STRICT
    """,
    "CREATE INDEX ix_trades_state ON trades (state)",
    # ------------------------------------------------------------------ orders
    """
    CREATE TABLE orders (
        client_order_id TEXT NOT NULL PRIMARY KEY,
        trade_id TEXT NOT NULL REFERENCES trades (trade_id),
        role TEXT NOT NULL,
        state TEXT NOT NULL,
        request_json TEXT NOT NULL,
        order_id TEXT,
        broker_status TEXT,
        updated_at_utc TEXT
    ) STRICT
    """,
    "CREATE UNIQUE INDEX ux_orders_order_id ON orders (order_id) WHERE order_id IS NOT NULL",
    "CREATE INDEX ix_orders_trade_id ON orders (trade_id)",
    "CREATE INDEX ix_orders_state ON orders (state)",
    # ------------------------------------------------------------------ order_events
    """
    CREATE TABLE order_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_order_id TEXT NOT NULL,
        occurred_at_utc TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        cause TEXT NOT NULL,
        broker_event_json TEXT
    ) STRICT
    """,
    "CREATE INDEX ix_order_events_client_order_id ON order_events (client_order_id)",
    *_append_only("order_events"),
    # ------------------------------------------------------------------ fills
    """
    CREATE TABLE fills (
        activity_id TEXT NOT NULL PRIMARY KEY,
        order_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty TEXT NOT NULL,
        price TEXT NOT NULL,
        timestamp_utc TEXT NOT NULL
    ) STRICT
    """,
    "CREATE INDEX ix_fills_order_id ON fills (order_id)",
    # ------------------------------------------------------------------ ai_decisions
    # snapshot_json / result_json are the source of the record; the columns between
    # them are derived at insert time for audit queries (sec. 38.1, 38.2).
    """
    CREATE TABLE ai_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id TEXT NOT NULL,
        ai_mode TEXT NOT NULL,
        snapshot_json TEXT NOT NULL,
        snapshot_hash TEXT NOT NULL,
        result_json TEXT NOT NULL,
        validity TEXT NOT NULL,
        verdict TEXT,
        reason_code TEXT,
        risk_flags_json TEXT,
        confidence REAL,
        rationale TEXT,
        raw_response TEXT,
        usage_json TEXT,
        latency_ms INTEGER,
        stop_reason TEXT,
        model_id TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        estimated_cost_usd TEXT NOT NULL,
        created_at_utc TEXT NOT NULL
    ) STRICT
    """,
    "CREATE INDEX ix_ai_decisions_created_at ON ai_decisions (created_at_utc)",
    "CREATE INDEX ix_ai_decisions_signal_id ON ai_decisions (signal_id)",
    # ------------------------------------------------------------------ risk_events
    """
    CREATE TABLE risk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id TEXT NOT NULL,
        occurred_at_utc TEXT NOT NULL,
        checks_json TEXT NOT NULL,
        sizing_json TEXT NOT NULL,
        decision TEXT NOT NULL
    ) STRICT
    """,
    "CREATE INDEX ix_risk_events_signal_id ON risk_events (signal_id)",
    *_append_only("risk_events"),
    # ------------------------------------------------------------------ equity_snapshots
    """
    CREATE TABLE equity_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        taken_at_utc TEXT NOT NULL,
        kind TEXT NOT NULL,
        equity TEXT NOT NULL,
        last_equity TEXT NOT NULL,
        buying_power TEXT NOT NULL
    ) STRICT
    """,
    "CREATE INDEX ix_equity_snapshots_taken_at ON equity_snapshots (taken_at_utc)",
    # ------------------------------------------------------------------ reconciliations
    """
    CREATE TABLE reconciliations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        occurred_at_utc TEXT NOT NULL,
        outcome TEXT NOT NULL,
        differences_json TEXT NOT NULL,
        actions_json TEXT NOT NULL
    ) STRICT
    """,
    # ------------------------------------------------------------------ shadow_outcomes
    """
    CREATE TABLE shadow_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id TEXT NOT NULL,
        recorded_at_utc TEXT NOT NULL,
        validity TEXT NOT NULL,
        verdict TEXT,
        reason_code TEXT,
        confidence REAL,
        executed INTEGER NOT NULL CHECK (executed IN (0, 1)),
        simulated INTEGER NOT NULL CHECK (simulated IN (0, 1)),
        result_r TEXT
    ) STRICT
    """,
    "CREATE INDEX ix_shadow_outcomes_recorded_at ON shadow_outcomes (recorded_at_utc)",
    "CREATE INDEX ix_shadow_outcomes_signal_id ON shadow_outcomes (signal_id)",
    # ------------------------------------------------------------------ system_control
    # One row (id = 1). No row means "never written": the repository returns the
    # fail-closed defaults of sec. 31.1.
    """
    CREATE TABLE system_control (
        id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
        trading_enabled INTEGER NOT NULL CHECK (trading_enabled IN (0, 1)),
        emergency_close INTEGER NOT NULL CHECK (emergency_close IN (0, 1)),
        ai_mode TEXT NOT NULL CHECK (ai_mode IN ('DISABLED', 'SHADOW', 'ACTIVE')),
        updated_at_utc TEXT,
        updated_by TEXT,
        reason TEXT
    ) STRICT
    """,
    # ------------------------------------------------------------------ system_events
    """
    CREATE TABLE system_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        occurred_at_utc TEXT NOT NULL,
        event_type TEXT NOT NULL,
        detail_json TEXT NOT NULL
    ) STRICT
    """,
    "CREATE INDEX ix_system_events_occurred_at ON system_events (occurred_at_utc)",
    *_append_only("system_events"),
)
