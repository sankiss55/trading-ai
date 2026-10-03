"""Translation of ``sqlite3`` errors into domain errors (sec. 8.3.7, 29, 30.2).

Classification by the SQLite (extended) result code name:

* ``SQLITE_BUSY*`` / ``SQLITE_LOCKED*`` / ``SQLITE_PROTOCOL``: ``RetryableError``
  ``DB_BUSY`` (another connection holds the write lock past the busy timeout).
* ``SQLITE_CONSTRAINT_PRIMARYKEY`` / ``_UNIQUE``: ``NonRetryableError`` ``DUPLICATE_KEY``.
* ``SQLITE_CONSTRAINT_FOREIGNKEY``: ``NonRetryableError`` ``FOREIGN_KEY_VIOLATION``.
* Any other ``SQLITE_CONSTRAINT*`` (CHECK, NOT NULL, append-only triggers):
  ``NonRetryableError`` ``CONSTRAINT_VIOLATION``.
* ``SQLITE_CORRUPT*`` / ``SQLITE_NOTADB``: ``StateCriticalError`` ``DB_CORRUPT``.
* ``SQLITE_FULL`` / ``SQLITE_IOERR*`` / ``SQLITE_READONLY*`` / ``SQLITE_CANTOPEN*``:
  ``StateCriticalError`` ``DB_NOT_WRITABLE`` (sec. 30.2: a non-writable DB halts).
* Anything else (SQL/schema errors, misuse): ``NonRetryableError`` ``DB_ERROR``.

SQLite messages name constraints and tables, never stored values, so they are safe to
include (sec. 43.1).
"""

from __future__ import annotations

import sqlite3

from adapters.sqlite.semantics import (
    CONSTRAINT_VIOLATION,
    DB_BUSY,
    DB_CORRUPT,
    DB_ERROR,
    DB_NOT_WRITABLE,
    DUPLICATE_KEY,
    FOREIGN_KEY_VIOLATION,
)
from domain.errors import DomainError, NonRetryableError, RetryableError, StateCriticalError

__all__ = ["translate_sqlite_error"]

_MAX_DETAIL = 200
_RETRYABLE_PREFIXES = ("SQLITE_BUSY", "SQLITE_LOCKED", "SQLITE_PROTOCOL")
_DUPLICATE_NAMES = frozenset({"SQLITE_CONSTRAINT_PRIMARYKEY", "SQLITE_CONSTRAINT_UNIQUE"})
_CORRUPT_PREFIXES = ("SQLITE_CORRUPT", "SQLITE_NOTADB")
_NOT_WRITABLE_PREFIXES = ("SQLITE_FULL", "SQLITE_IOERR", "SQLITE_READONLY", "SQLITE_CANTOPEN")


def translate_sqlite_error(exc: sqlite3.Error, *, operation: str) -> DomainError:
    """Map a ``sqlite3`` exception to ``RetryableError``, ``NonRetryableError`` or
    ``StateCriticalError``.

    Args:
        exc: Exception raised by ``sqlite3``.
        operation: Short description of the call, e.g. ``"orders.add"``.
    """
    name: str = getattr(exc, "sqlite_errorname", "") or type(exc).__name__
    where = f"{operation}: {name}: {str(exc)[:_MAX_DETAIL]}"
    if name.startswith(_RETRYABLE_PREFIXES):
        return RetryableError(where, code=DB_BUSY)
    if name in _DUPLICATE_NAMES:
        return NonRetryableError(where, code=DUPLICATE_KEY)
    if name == "SQLITE_CONSTRAINT_FOREIGNKEY":
        return NonRetryableError(where, code=FOREIGN_KEY_VIOLATION)
    if name.startswith("SQLITE_CONSTRAINT"):
        return NonRetryableError(where, code=CONSTRAINT_VIOLATION)
    if name.startswith(_CORRUPT_PREFIXES):
        return StateCriticalError(where, code=DB_CORRUPT)
    if name.startswith(_NOT_WRITABLE_PREFIXES):
        return StateCriticalError(where, code=DB_NOT_WRITABLE)
    return NonRetryableError(where, code=DB_ERROR)
