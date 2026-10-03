"""Versioned SQL migrations of the operational database (sec. 38).

Each migration is a module ``vNNNN_<name>.py`` exposing ``VERSION``, ``NAME`` and
``STATEMENTS`` (one SQL statement per item, so triggers need no script splitting).
Migrations are Python modules rather than ``.sql`` files so they ship with the package
without any package-data configuration. A migration is never edited once released: a
schema change is a new module appended to ``MIGRATIONS``.

``apply_migrations`` runs every pending migration in ONE ``BEGIN IMMEDIATE`` transaction
and records each version in ``schema_version``; it is idempotent and safe to call from
several processes at once (the write lock serializes them).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final

from adapters.sqlite.migrations import v0001_initial
from adapters.sqlite.semantics import DB_NOT_INITIALIZED, SCHEMA_VERSION_MISMATCH
from domain.errors import NonRetryableError

__all__ = [
    "LATEST_VERSION",
    "MIGRATIONS",
    "Migration",
    "apply_migrations",
    "current_version",
    "verify_schema",
]


@dataclass(frozen=True)
class Migration:
    """One schema version."""

    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: Final[tuple[Migration, ...]] = (
    Migration(v0001_initial.VERSION, v0001_initial.NAME, v0001_initial.STATEMENTS),
)
LATEST_VERSION: Final = MIGRATIONS[-1].version

_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at_utc TEXT NOT NULL
) STRICT
"""
_NOW_UTC_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


def _has_schema_version_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    return row is not None


def current_version(conn: sqlite3.Connection) -> int:
    """Highest applied version, 0 for a database without migrations."""
    if not _has_schema_version_table(conn):
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    """Apply the pending migrations atomically; return the versions applied (maybe none).

    ``conn`` must be in autocommit mode (``isolation_level=None``) with no open
    transaction.

    Raises:
        NonRetryableError: ``SCHEMA_VERSION_MISMATCH`` if the database is newer than
            this code (never downgrade).
        sqlite3.Error: on SQL failure (the caller translates it); nothing is applied.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(_CREATE_SCHEMA_VERSION)
        current = current_version(conn)
        if current > LATEST_VERSION:
            raise NonRetryableError(
                f"database schema version {current} is newer than this code "
                f"({LATEST_VERSION}); refusing to run",
                code=SCHEMA_VERSION_MISMATCH,
            )
        applied: list[int] = []
        for migration in MIGRATIONS:
            if migration.version <= current:
                continue
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(
                f"INSERT INTO schema_version (version, name, applied_at_utc) "
                f"VALUES (?, ?, {_NOW_UTC_SQL})",
                (migration.version, migration.name),
            )
            applied.append(migration.version)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return applied


def verify_schema(conn: sqlite3.Connection) -> None:
    """Refuse to work on a database that is not initialized at exactly ``LATEST_VERSION``
    in WAL mode.

    Raises:
        NonRetryableError: ``DB_NOT_INITIALIZED`` or ``SCHEMA_VERSION_MISMATCH``.
    """
    journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    if journal_mode != "wal":
        raise NonRetryableError(
            f"database journal mode is {journal_mode!r}, expected 'wal'; "
            "run SqliteUnitOfWork.initialize first",
            code=DB_NOT_INITIALIZED,
        )
    version = current_version(conn)
    if version == 0:
        raise NonRetryableError(
            "database has no schema; run SqliteUnitOfWork.initialize first",
            code=DB_NOT_INITIALIZED,
        )
    if version != LATEST_VERSION:
        raise NonRetryableError(
            f"database schema version {version} != expected {LATEST_VERSION}; "
            "run SqliteUnitOfWork.initialize (never downgrade)",
            code=SCHEMA_VERSION_MISMATCH,
        )
