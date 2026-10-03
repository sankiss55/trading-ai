"""``system_control`` table (sec. 31.1): a single row, fail-closed when absent."""

from __future__ import annotations

import sqlite3

from adapters.sqlite._codec import from_row, to_row
from adapters.sqlite.repositories._base import SqliteRepository, fetch_one, insert_sql
from domain.models import SystemControl

__all__ = ["SqliteControlRepository"]

_TABLE = "system_control"
_ROW_ID = 1


class SqliteControlRepository(SqliteRepository):
    """Implements ``IControlRepository``.

    A database where the controls were never written returns ``SystemControl()``:
    trading disabled, no emergency close, AI ``DISABLED`` (sec. 31.1). A stored row
    that fails validation raises ``StateCriticalError`` (never a permissive default).
    Recording the change in ``system_events`` (sec. 31.1) is the caller's job.
    """

    async def get(self) -> SystemControl:
        """Current runtime controls (fail-closed defaults if never written)."""

        def job(conn: sqlite3.Connection) -> SystemControl:
            row = fetch_one(conn, "SELECT * FROM system_control WHERE id = ?", (_ROW_ID,))
            return SystemControl() if row is None else from_row(SystemControl, row, table=_TABLE)

        return await self._session.run("control.get", job)

    async def set(self, control: SystemControl) -> None:
        """Replace the runtime controls."""
        row = {"id": _ROW_ID, **to_row(control)}
        updates = ", ".join(f"{name} = excluded.{name}" for name in row if name != "id")
        sql = insert_sql(_TABLE, row, suffix=f" ON CONFLICT (id) DO UPDATE SET {updates}")

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("control.set", job)
