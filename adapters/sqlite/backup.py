"""Online backup and restore of the operational database (sec. 58.2).

``backup_database`` uses the SQLite backup API from a read-only connection, so it can
run while the system is live: the copy is a consistent snapshot of COMMITTED data
(an open write transaction of another connection is not included). The copy is written
to a temporary file, checked with ``PRAGMA integrity_check``, switched to the
self-contained ``DELETE`` journal mode and only then renamed to its final name, so an
interrupted backup never leaves a truncated file that looks valid.

``restore_database`` validates the backup and copies it into ``db_path`` with the backup
API (the target is switched back to WAL). Run it with the system stopped (runbook), then
call ``SqliteUnitOfWork.initialize(db_path)`` to apply any newer migrations.

Frequency and retention are owner decisions (``backups.frequency``,
``backups.retention_days``); scheduling and pruning belong to the caller.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from adapters.sqlite._connection import connect
from adapters.sqlite._errors import translate_sqlite_error
from adapters.sqlite.migrations import current_version
from adapters.sqlite.semantics import DB_CORRUPT, DB_NOT_INITIALIZED, INVALID_ARGUMENT
from domain.errors import NonRetryableError, StateCriticalError

__all__ = [
    "BACKUP_EXISTS",
    "INVALID_BACKUP",
    "backup_database",
    "backup_file_name",
    "restore_database",
]

BACKUP_EXISTS = "BACKUP_EXISTS"
INVALID_BACKUP = "INVALID_BACKUP"
_TIMEOUT_SECONDS = 30.0


def backup_file_name(db_path: Path, timestamp: datetime) -> str:
    """``<db stem>-<YYYYMMDDTHHMMSSZ>.sqlite3``.

    Raises:
        NonRetryableError: ``INVALID_ARGUMENT`` if ``timestamp`` is not UTC-aware.
    """
    offset = timestamp.utcoffset()
    if offset is None or offset != timedelta(0):
        raise NonRetryableError(
            "backup timestamp must be timezone-aware UTC", code=INVALID_ARGUMENT
        )
    return f"{Path(db_path).stem}-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.sqlite3"


def _open_read_only(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=_TIMEOUT_SECONDS, isolation_level=None)


def _integrity_ok(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("PRAGMA integrity_check").fetchall()
    return [str(row[0]) for row in rows] == ["ok"]


def backup_database(db_path: Path, dest_dir: Path, *, timestamp: datetime) -> Path:
    """Write a consistent, verified copy of ``db_path`` into ``dest_dir``.

    Args:
        db_path: Live database (WAL mode is fine).
        dest_dir: Backup directory (created if missing).
        timestamp: UTC instant used in the file name (from ``IClock``).

    Returns:
        Path of the new backup file.

    Raises:
        NonRetryableError: ``DB_NOT_INITIALIZED`` (missing source), ``BACKUP_EXISTS``
            (never overwrites), ``INVALID_ARGUMENT`` (naive timestamp).
        StateCriticalError: ``DB_CORRUPT`` if the copy fails the integrity check, or a
            translated I/O error.
    """
    source_path = Path(db_path)
    if not source_path.is_file():
        raise NonRetryableError("database file to back up does not exist", code=DB_NOT_INITIALIZED)
    target = Path(dest_dir) / backup_file_name(source_path, timestamp)
    if target.exists():
        raise NonRetryableError(f"backup {target.name} already exists", code=BACKUP_EXISTS)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NonRetryableError(
            f"cannot create the backup directory: {type(exc).__name__}", code=INVALID_ARGUMENT
        ) from exc
    partial = target.with_name(target.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        source = _open_read_only(source_path)
        try:
            copy = sqlite3.connect(partial, timeout=_TIMEOUT_SECONDS, isolation_level=None)
            try:
                source.backup(copy)
                copy.execute("PRAGMA journal_mode = DELETE")
                if not _integrity_ok(copy):
                    raise StateCriticalError(
                        "backup copy failed the integrity check", code=DB_CORRUPT
                    )
            finally:
                copy.close()
        finally:
            source.close()
        os.replace(partial, target)
    except sqlite3.Error as exc:
        partial.unlink(missing_ok=True)
        raise translate_sqlite_error(exc, operation="backup database") from exc
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        raise
    return target


def restore_database(backup_path: Path, db_path: Path) -> None:
    """Replace the content of ``db_path`` with a validated backup.

    Raises:
        NonRetryableError: ``INVALID_BACKUP`` if the file is missing, not a database,
            fails the integrity check or has no schema.
        StateCriticalError: translated I/O errors on the target.
    """
    source_path = Path(backup_path)
    if not source_path.is_file():
        raise NonRetryableError("backup file does not exist", code=INVALID_BACKUP)
    try:
        source = _open_read_only(source_path)
    except sqlite3.Error as exc:
        raise NonRetryableError(f"cannot open backup: {exc}", code=INVALID_BACKUP) from exc
    try:
        try:
            valid = _integrity_ok(source) and current_version(source) > 0
        except sqlite3.Error as exc:
            raise NonRetryableError(
                f"backup is not a valid database: {exc}", code=INVALID_BACKUP
            ) from exc
        if not valid:
            raise NonRetryableError(
                "backup failed the integrity check or has no schema", code=INVALID_BACKUP
            )
        target_path = Path(db_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target = connect(target_path, create=True, timeout_seconds=_TIMEOUT_SECONDS)
        try:
            source.backup(target)
            target.execute("PRAGMA journal_mode = WAL")
            if not _integrity_ok(target):
                raise StateCriticalError(
                    "restored database failed the integrity check", code=DB_CORRUPT
                )
        finally:
            target.close()
    except sqlite3.Error as exc:
        raise translate_sqlite_error(exc, operation="restore database") from exc
    finally:
        source.close()
