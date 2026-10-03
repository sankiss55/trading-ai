"""``equity_snapshots`` table (sec. 38.1); inputs of the drawdown and weekly-loss limits."""

from __future__ import annotations

import sqlite3
from datetime import date
from decimal import Decimal

from adapters.sqlite._codec import format_utc, parse_decimal, to_row
from adapters.sqlite.repositories._base import SqliteRepository, fetch_all, fetch_one, insert_sql
from adapters.sqlite.semantics import day_window
from domain.models import EquitySnapshot

__all__ = ["SqliteEquitySnapshotRepository"]

_TABLE = "equity_snapshots"


class SqliteEquitySnapshotRepository(SqliteRepository):
    """Implements ``IEquitySnapshotRepository``. Comparisons are exact ``Decimal``
    comparisons done in Python (TEXT ordering is not numeric ordering)."""

    async def add(self, snapshot: EquitySnapshot) -> None:
        """Insert one equity snapshot."""
        row = to_row(snapshot)
        sql = insert_sql(_TABLE, row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("equity_snapshots.add", job)

    async def peak_equity(self) -> Decimal | None:
        """Highest recorded equity, or ``None`` without snapshots."""

        def job(conn: sqlite3.Connection) -> Decimal | None:
            rows = fetch_all(conn, "SELECT equity FROM equity_snapshots")
            values = [parse_decimal(row["equity"], table=_TABLE, column="equity") for row in rows]
            return max(values) if values else None

        return await self._session.run("equity_snapshots.peak_equity", job)

    async def week_start_equity(self, week_start: date) -> Decimal | None:
        """Equity of the first snapshot taken on or after ``week_start`` 00:00 UTC."""
        start, _ = day_window(week_start)
        params = (format_utc(start),)

        def job(conn: sqlite3.Connection) -> Decimal | None:
            row = fetch_one(
                conn,
                "SELECT equity FROM equity_snapshots WHERE taken_at_utc >= ? "
                "ORDER BY taken_at_utc, id LIMIT 1",
                params,
            )
            if row is None:
                return None
            return parse_decimal(row["equity"], table=_TABLE, column="equity")

        return await self._session.run("equity_snapshots.week_start_equity", job)
