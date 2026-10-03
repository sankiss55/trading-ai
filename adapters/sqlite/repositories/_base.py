"""Shared helpers of the SQLite repositories."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Final

from adapters.sqlite._codec import SqlValue
from adapters.sqlite._connection import SqliteSession
from adapters.sqlite.semantics import FINAL_STATES

__all__ = [
    "FINAL_STATE_PARAMS",
    "FINAL_STATE_PLACEHOLDERS",
    "SqliteRepository",
    "fetch_all",
    "fetch_one",
    "insert_sql",
]

FINAL_STATE_PARAMS: Final[tuple[str, ...]] = tuple(sorted(state.value for state in FINAL_STATES))
"""Parameters for ``... NOT IN (FINAL_STATE_PLACEHOLDERS)``."""
FINAL_STATE_PLACEHOLDERS: Final = ", ".join("?" for _ in FINAL_STATE_PARAMS)


class SqliteRepository:
    """Base class: every repository runs its statements through the UoW session."""

    def __init__(self, session: SqliteSession) -> None:
        self._session = session


def insert_sql(table: str, columns: Iterable[str], *, suffix: str = "") -> str:
    """``INSERT INTO table (a, b) VALUES (:a, :b)`` plus an optional suffix.

    Column names always come from domain model field names, never from input data.
    """
    names = list(columns)
    placeholders = ", ".join(f":{name}" for name in names)
    return f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders}){suffix}"


def fetch_one(
    conn: sqlite3.Connection, sql: str, params: Iterable[SqlValue] | Mapping[str, SqlValue] = ()
) -> dict[str, SqlValue] | None:
    """First row as a plain ``dict`` (``sqlite3.Row`` never leaves the adapter)."""
    row = conn.execute(sql, params if isinstance(params, Mapping) else tuple(params)).fetchone()
    return None if row is None else dict(row)


def fetch_all(
    conn: sqlite3.Connection, sql: str, params: Iterable[SqlValue] | Mapping[str, SqlValue] = ()
) -> list[dict[str, SqlValue]]:
    """All rows as plain ``dict`` objects."""
    rows = conn.execute(sql, params if isinstance(params, Mapping) else tuple(params)).fetchall()
    return [dict(row) for row in rows]
