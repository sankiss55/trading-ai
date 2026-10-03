"""``signals`` table (sec. 13.7, 38.1). ``signal_id`` is the primary key (sec. 22.5)."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

from adapters.sqlite._codec import SqlValue, from_row, to_row
from adapters.sqlite.repositories._base import (
    FINAL_STATE_PARAMS,
    FINAL_STATE_PLACEHOLDERS,
    SqliteRepository,
    fetch_all,
    fetch_one,
    insert_sql,
)
from adapters.sqlite.semantics import SIGNAL_NOT_FOUND
from domain.errors import NonRetryableError
from domain.models import Signal, TradeState

__all__ = ["SqliteSignalRepository"]

_TABLE = "signals"
_JSON = frozenset({"rule_results"})


def _signal(row: Mapping[str, SqlValue]) -> Signal:
    return from_row(Signal, row, table=_TABLE, json_fields=_JSON)


class SqliteSignalRepository(SqliteRepository):
    """Implements ``ISignalRepository``. The lifecycle state lives in ``status``."""

    async def add(self, signal: Signal) -> None:
        """Insert a new signal with state ``SIGNAL_CREATED``.

        Raises:
            NonRetryableError: ``DUPLICATE_KEY`` if the ``signal_id`` already exists.
        """
        row = to_row(signal, json_fields=_JSON)
        row["status"] = TradeState.SIGNAL_CREATED.value
        row["status_reason"] = None
        row["status_updated_at_utc"] = self._session.stamp()
        sql = insert_sql(_TABLE, row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("signals.add", job)

    async def get(self, signal_id: str) -> Signal | None:
        """Signal by id, or ``None``."""

        def job(conn: sqlite3.Connection) -> Signal | None:
            row = fetch_one(conn, "SELECT * FROM signals WHERE signal_id = ?", (signal_id,))
            return None if row is None else _signal(row)

        return await self._session.run("signals.get", job)

    async def exists(self, signal_id: str) -> bool:
        """Whether a signal with this id was already recorded."""

        def job(conn: sqlite3.Connection) -> bool:
            sql = "SELECT 1 FROM signals WHERE signal_id = ?"
            return fetch_one(conn, sql, (signal_id,)) is not None

        return await self._session.run("signals.exists", job)

    async def mark_status(
        self, signal_id: str, status: TradeState, reason: str | None = None
    ) -> None:
        """Set the lifecycle state and reason code (transitions are validated upstream).

        Raises:
            NonRetryableError: ``SIGNAL_NOT_FOUND`` for an unknown ``signal_id``.
        """
        params = (status.value, reason, self._session.stamp(), signal_id)

        def job(conn: sqlite3.Connection) -> None:
            cursor = conn.execute(
                "UPDATE signals SET status = ?, status_reason = ?, status_updated_at_utc = ? "
                "WHERE signal_id = ?",
                params,
            )
            if cursor.rowcount == 0:
                raise NonRetryableError(f"signal {signal_id!r} not found", code=SIGNAL_NOT_FOUND)

        await self._session.run("signals.mark_status", job)

    async def list_pending(self) -> list[Signal]:
        """Signals not in a final state, in insertion order."""

        def job(conn: sqlite3.Connection) -> list[Signal]:
            rows = fetch_all(
                conn,
                f"SELECT * FROM signals WHERE status NOT IN ({FINAL_STATE_PLACEHOLDERS}) "
                "ORDER BY rowid",
                FINAL_STATE_PARAMS,
            )
            return [_signal(row) for row in rows]

        return await self._session.run("signals.list_pending", job)
