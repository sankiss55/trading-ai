"""One SQLite connection served by a dedicated worker thread (sec. 8.5, 38).

Threading: each active ``SqliteUnitOfWork`` context owns ONE connection and ONE
single-thread executor. Every statement runs in that thread, in submission order, so

* the event loop never blocks on SQLite I/O or lock waits (``busy_timeout``);
* the connection is only ever used from the thread that created it
  (``check_same_thread`` stays on) and statements are serialized;
* a cancelled ``await`` never leaves the connection in concurrent use: the running job
  finishes and later jobs (rollback, close) queue behind it.

Transactions: started lazily with ``BEGIN IMMEDIATE`` by the first statement after
entering, ``commit()`` or ``rollback()``. The write lock is taken up front (no
``SQLITE_BUSY_SNAPSHOT`` on a read-to-write upgrade in WAL mode) and is never held
between transactions, e.g. while the caller awaits the broker after committing
``SUBMITTING`` (sec. 20). A failed statement only undoes itself, as in SQLite; if
SQLite aborts the whole transaction (I/O error, disk full), the connection refuses
further statements until ``rollback()`` so partial work can never be committed.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TypeVar

from adapters.sqlite._codec import format_utc
from adapters.sqlite._errors import translate_sqlite_error
from adapters.sqlite.semantics import TRANSACTION_ABORTED, UOW_NOT_ACTIVE
from domain.errors import NonRetryableError
from domain.ports import IClock

__all__ = ["SqliteConnection", "SqliteSession", "connect"]

T = TypeVar("T")


def connect(
    db_path: Path, *, create: bool, timeout_seconds: float, durable: bool = True
) -> sqlite3.Connection:
    """Open ``db_path`` in autocommit mode with foreign keys on.

    ``create=False`` opens an existing file only (``mode=rw``): a missing database is
    an error instead of a silently created empty file. ``durable=True`` uses
    ``synchronous=FULL`` (a commit survives a power loss); ``False`` uses ``NORMAL``
    (WAL: survives a process crash, may lose the last commits on power loss), only for
    throw-away databases such as a backtest's temporary DB.
    """
    uri = f"{db_path.resolve().as_uri()}?mode={'rwc' if create else 'rw'}"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        # FULL: a committed SUBMITTING survives a power loss, not only a crash (sec. 20).
        conn.execute(f"PRAGMA synchronous = {'FULL' if durable else 'NORMAL'}")
    except BaseException:
        conn.close()
        raise
    return conn


class SqliteConnection:
    """A connection plus its worker thread. Created by :meth:`open`."""

    def __init__(self, conn: sqlite3.Connection, executor: ThreadPoolExecutor) -> None:
        self._conn = conn
        self._executor = executor
        self._aborted = False  # only touched from the worker thread

    @classmethod
    async def open(
        cls,
        db_path: Path,
        *,
        timeout_seconds: float,
        verify: Callable[[sqlite3.Connection], None],
        durable: bool = True,
    ) -> SqliteConnection:
        """Connect in a new worker thread and run ``verify`` (schema checks) there.

        Raises:
            DomainError: translated ``sqlite3`` errors, or whatever ``verify`` raises.
        """
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite-uow")

        def job() -> sqlite3.Connection:
            conn = connect(db_path, create=False, timeout_seconds=timeout_seconds, durable=durable)
            try:
                verify(conn)
            except BaseException:
                conn.close()
                raise
            return conn

        loop = asyncio.get_running_loop()
        try:
            conn = await loop.run_in_executor(executor, job)
        except sqlite3.Error as exc:
            executor.shutdown(wait=False)
            raise translate_sqlite_error(exc, operation="open database") from exc
        except BaseException:
            executor.shutdown(wait=False)
            raise
        return cls(conn, executor)

    async def _submit(self, operation: str, fn: Callable[[], T]) -> T:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, fn)
        except sqlite3.Error as exc:
            raise translate_sqlite_error(exc, operation=operation) from exc

    async def run(self, operation: str, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn(conn)`` in the worker thread inside the current transaction."""
        return await self._submit(operation, partial(self._run_in_transaction, fn))

    async def commit(self) -> None:
        """Commit the open transaction, if any. On failure everything is rolled back."""
        await self._submit("commit", self._commit)

    async def rollback(self) -> None:
        """Roll back the open transaction, if any."""
        await self._submit("rollback", self._rollback)

    async def close(self) -> None:
        """Roll back anything uncommitted, close the connection and stop the thread."""
        try:
            await self._submit("close", self._close)
        finally:
            self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ worker thread

    def _run_in_transaction(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        if self._aborted:
            raise NonRetryableError(
                "the transaction was aborted by SQLite after an error; roll back first",
                code=TRANSACTION_ABORTED,
            )
        if not self._conn.in_transaction:
            self._conn.execute("BEGIN IMMEDIATE")
        try:
            return fn(self._conn)
        except BaseException:
            if not self._conn.in_transaction:
                self._aborted = True
            raise

    def _commit(self) -> None:
        if self._aborted:
            self._aborted = False
            raise NonRetryableError(
                "nothing was committed: the transaction was aborted by SQLite after an error",
                code=TRANSACTION_ABORTED,
            )
        if not self._conn.in_transaction:
            return
        try:
            self._conn.execute("COMMIT")
        except sqlite3.Error:
            self._rollback_quietly()
            raise

    def _rollback(self) -> None:
        self._aborted = False
        if self._conn.in_transaction:
            self._conn.execute("ROLLBACK")

    def _rollback_quietly(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")

    def _close(self) -> None:
        self._rollback_quietly()
        self._conn.close()  # closing also discards any transaction left open


class SqliteSession:
    """Active connection of one unit of work, shared by its repositories."""

    def __init__(self, clock: IClock | None) -> None:
        self._clock = clock
        self._connection: SqliteConnection | None = None

    @property
    def active(self) -> bool:
        """Whether a connection is bound (inside ``async with``)."""
        return self._connection is not None

    def bind(self, connection: SqliteConnection) -> None:
        """Attach the connection opened by ``__aenter__``."""
        self._connection = connection

    def unbind(self) -> SqliteConnection | None:
        """Detach and return the connection (``__aexit__``)."""
        connection, self._connection = self._connection, None
        return connection

    def connection(self) -> SqliteConnection:
        """The bound connection.

        Raises:
            NonRetryableError: ``UOW_NOT_ACTIVE`` outside ``async with uow:``.
        """
        if self._connection is None:
            raise NonRetryableError(
                "unit of work is not active; use 'async with uow:'", code=UOW_NOT_ACTIVE
            )
        return self._connection

    async def run(self, operation: str, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn`` on the bound connection (see :meth:`SqliteConnection.run`)."""
        return await self.connection().run(operation, fn)

    def stamp(self) -> str | None:
        """Audit timestamp for ``*_updated_at_utc`` columns; ``None`` without a clock."""
        return None if self._clock is None else format_utc(self._clock.now_utc())
