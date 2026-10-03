"""Audit tables (sec. 38.1, 38.2): ``ai_decisions``, ``risk_events``, ``reconciliations``,
``shadow_outcomes`` and ``system_events``."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from decimal import Decimal

from adapters.sqlite._codec import SqlValue, format_utc, from_row, parse_decimal, to_row
from adapters.sqlite.repositories._base import SqliteRepository, fetch_all, fetch_one, insert_sql
from adapters.sqlite.semantics import day_window, month_window
from domain.models import (
    AIDecisionRecord,
    ReconciliationRecord,
    RiskEvent,
    ShadowOutcome,
    SystemEvent,
)

__all__ = [
    "SqliteAIDecisionRepository",
    "SqliteReconciliationRepository",
    "SqliteRiskEventRepository",
    "SqliteShadowOutcomeRepository",
    "SqliteSystemEventRepository",
]

_AI_JSON = frozenset({"snapshot", "result"})
_SHADOW = "shadow_outcomes"


def _ai_audit_columns(record: AIDecisionRecord) -> dict[str, SqlValue]:
    """Columns derived from the (immutable) record for audit queries (sec. 38.1)."""
    result = record.result
    verdict = result.verdict
    return {
        "snapshot_hash": record.snapshot.snapshot_hash,
        "validity": result.validity.value,
        "verdict": None if verdict is None else verdict.verdict.value,
        "reason_code": None if verdict is None else verdict.reason_code.value,
        "risk_flags_json": (
            None if verdict is None else json.dumps([flag.value for flag in verdict.risk_flags])
        ),
        "confidence": None if verdict is None else verdict.confidence,
        "rationale": None if verdict is None else verdict.rationale,
        "raw_response": result.raw_response,
        "usage_json": None if result.usage is None else result.usage.model_dump_json(),
        "latency_ms": result.latency_ms,
        "stop_reason": result.stop_reason,
    }


class SqliteAIDecisionRepository(SqliteRepository):
    """Implements ``IAIDecisionRepository``. Day/month windows use ``created_at_utc``."""

    async def add(self, record: AIDecisionRecord) -> None:
        """Insert one AI decision."""
        row = to_row(record, json_fields=_AI_JSON) | _ai_audit_columns(record)
        sql = insert_sql("ai_decisions", row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("ai_decisions.add", job)

    async def count_today(self, day: date) -> int:
        """Number of AI calls recorded on ``day`` (UTC date)."""
        start, end = day_window(day)
        params = (format_utc(start), format_utc(end))

        def job(conn: sqlite3.Connection) -> int:
            row = fetch_one(
                conn,
                "SELECT COUNT(*) AS n FROM ai_decisions "
                "WHERE created_at_utc >= ? AND created_at_utc < ?",
                params,
            )
            return 0 if row is None or row["n"] is None else int(row["n"])

        return await self._session.run("ai_decisions.count_today", job)

    async def cost_today(self, day: date) -> Decimal:
        """Exact sum of ``estimated_cost_usd`` recorded on ``day`` (UTC date)."""
        start, end = day_window(day)
        return await self._cost_between(start, end, "ai_decisions.cost_today")

    async def cost_month(self, year: int, month: int) -> Decimal:
        """Exact sum of ``estimated_cost_usd`` recorded in the month (UTC)."""
        start, end = month_window(year, month)
        return await self._cost_between(start, end, "ai_decisions.cost_month")

    async def _cost_between(self, start: datetime, end: datetime, operation: str) -> Decimal:
        params = (format_utc(start), format_utc(end))

        def job(conn: sqlite3.Connection) -> Decimal:
            # Summed in Python: SQL SUM() over TEXT would go through binary floats.
            rows = fetch_all(
                conn,
                "SELECT estimated_cost_usd FROM ai_decisions "
                "WHERE created_at_utc >= ? AND created_at_utc < ?",
                params,
            )
            return sum(
                (
                    parse_decimal(
                        row["estimated_cost_usd"], table="ai_decisions", column="estimated_cost_usd"
                    )
                    for row in rows
                ),
                Decimal(0),
            )

        return await self._session.run(operation, job)


class SqliteRiskEventRepository(SqliteRepository):
    """Implements ``IRiskEventRepository`` (append-only, enforced by triggers)."""

    async def append(self, event: RiskEvent) -> None:
        """Append one check/sizing evaluation."""
        row = to_row(event, json_fields=frozenset({"checks", "sizing"}))
        sql = insert_sql("risk_events", row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("risk_events.append", job)


class SqliteReconciliationRepository(SqliteRepository):
    """Implements ``IReconciliationRepository``."""

    async def add(self, record: ReconciliationRecord) -> None:
        """Insert one reconciliation result."""
        row = to_row(record, json_fields=frozenset({"differences", "actions"}))
        sql = insert_sql("reconciliations", row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("reconciliations.add", job)


class SqliteShadowOutcomeRepository(SqliteRepository):
    """Implements ``IShadowOutcomeRepository`` (sec. 46.2)."""

    async def add(self, outcome: ShadowOutcome) -> None:
        """Insert one counterfactual outcome."""
        row = to_row(outcome)
        sql = insert_sql(_SHADOW, row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("shadow_outcomes.add", job)

    async def list_for_report(self, since_utc: datetime | None = None) -> list[ShadowOutcome]:
        """Outcomes ordered by ``recorded_at_utc`` (then insertion); only those recorded
        at or after ``since_utc`` when given."""
        where = "" if since_utc is None else "WHERE recorded_at_utc >= ? "
        params: tuple[SqlValue, ...] = () if since_utc is None else (format_utc(since_utc),)
        sql = f"SELECT * FROM shadow_outcomes {where}ORDER BY recorded_at_utc, id"

        def job(conn: sqlite3.Connection) -> list[ShadowOutcome]:
            return [
                from_row(ShadowOutcome, row, table=_SHADOW) for row in fetch_all(conn, sql, params)
            ]

        return await self._session.run("shadow_outcomes.list_for_report", job)


class SqliteSystemEventRepository(SqliteRepository):
    """Implements ``ISystemEventRepository`` (append-only, enforced by triggers)."""

    async def append(self, event: SystemEvent) -> None:
        """Append one system event."""
        row = to_row(event, json_fields=frozenset({"detail"}))
        sql = insert_sql("system_events", row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("system_events.append", job)
