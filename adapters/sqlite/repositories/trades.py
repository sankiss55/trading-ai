"""``trades`` table (sec. 38.1). One trade per signal: ``signal_id`` is unique (sec. 22.5)."""

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
from adapters.sqlite.semantics import TRADE_IDENTITY_CHANGED, TRADE_NOT_FOUND
from domain.errors import NonRetryableError
from domain.models import Trade

__all__ = ["SqliteTradeRepository"]

_TABLE = "trades"
_JSON = frozenset({"versions"})
_IDENTITY = ("trade_id", "signal_id")


def _trade(row: Mapping[str, SqlValue]) -> Trade:
    return from_row(Trade, row, table=_TABLE, json_fields=_JSON)


class SqliteTradeRepository(SqliteRepository):
    """Implements ``ITradeRepository``."""

    async def add(self, trade: Trade) -> None:
        """Insert a new trade.

        Raises:
            NonRetryableError: ``DUPLICATE_KEY`` (trade or signal already has a trade),
                ``FOREIGN_KEY_VIOLATION`` (unknown ``signal_id``).
        """
        row = to_row(trade, json_fields=_JSON)
        row["updated_at_utc"] = self._session.stamp()
        sql = insert_sql(_TABLE, row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("trades.add", job)

    async def get(self, trade_id: str) -> Trade | None:
        """Trade by id, or ``None``."""

        def job(conn: sqlite3.Connection) -> Trade | None:
            row = fetch_one(conn, "SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            return None if row is None else _trade(row)

        return await self._session.run("trades.get", job)

    async def get_by_signal(self, signal_id: str) -> Trade | None:
        """Trade originated by a signal, or ``None``."""

        def job(conn: sqlite3.Connection) -> Trade | None:
            row = fetch_one(conn, "SELECT * FROM trades WHERE signal_id = ?", (signal_id,))
            return None if row is None else _trade(row)

        return await self._session.run("trades.get_by_signal", job)

    async def list_open(self) -> list[Trade]:
        """Trades not in a final state, in insertion order."""

        def job(conn: sqlite3.Connection) -> list[Trade]:
            rows = fetch_all(
                conn,
                f"SELECT * FROM trades WHERE state NOT IN ({FINAL_STATE_PLACEHOLDERS}) "
                "ORDER BY rowid",
                FINAL_STATE_PARAMS,
            )
            return [_trade(row) for row in rows]

        return await self._session.run("trades.list_open", job)

    async def update(self, trade: Trade) -> None:
        """Replace the stored trade with this version. ``trade_id``/``signal_id`` are fixed.

        Raises:
            NonRetryableError: ``TRADE_NOT_FOUND`` or ``TRADE_IDENTITY_CHANGED``.
        """
        row = to_row(trade, json_fields=_JSON)
        row["updated_at_utc"] = self._session.stamp()
        assignments = ", ".join(f"{name} = :{name}" for name in row if name not in _IDENTITY)
        sql = f"UPDATE trades SET {assignments} WHERE trade_id = :trade_id"

        def job(conn: sqlite3.Connection) -> None:
            stored = fetch_one(
                conn, "SELECT signal_id FROM trades WHERE trade_id = ?", (trade.trade_id,)
            )
            if stored is None:
                raise NonRetryableError(f"trade {trade.trade_id!r} not found", code=TRADE_NOT_FOUND)
            if stored["signal_id"] != trade.signal_id:
                raise NonRetryableError(
                    f"trade {trade.trade_id!r} belongs to another signal; "
                    "trade_id and signal_id cannot change",
                    code=TRADE_IDENTITY_CHANGED,
                )
            conn.execute(sql, row)

        await self._session.run("trades.update", job)
