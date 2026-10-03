"""``SqliteUnitOfWork``: the ``IUnitOfWork`` adapter over SQLite in WAL mode (sec. 8.5, 38).

Usage::

    SqliteUnitOfWork.initialize(db_path)          # once at startup: schema + WAL
    uow = SqliteUnitOfWork(db_path, clock=clock)
    async with uow:
        await uow.orders.add(record)              # state SUBMITTING
        await uow.commit()                        # durable BEFORE calling the broker
        ...                                       # broker call: no lock is held here
        await uow.orders.update_status(cid, TradeState.SUBMITTED, order_id=oid)
        await uow.commit()

Semantics (shared with ``InMemoryUnitOfWork``, pinned by
``tests/contract/unit_of_work_contract.py``):

* Each ``async with`` opens one connection served by one worker thread (see
  ``_connection``); leaving it rolls back anything not committed and closes it.
* Transactions start lazily (``BEGIN IMMEDIATE``) on the first repository call, so
  writers are serialized: a second unit of work waits up to ``lock_timeout_seconds``
  and then raises ``RetryableError(code="DB_BUSY")``. Keep transactions short and never
  await external I/O inside one; ``commit()`` releases the lock.
* Several ``commit()`` calls per context are allowed. A failed statement only undoes
  itself; the rest of the transaction is kept until commit or rollback.
* An instance can be reused sequentially but not entered twice at the same time: use
  one instance per concurrent task (e.g. a factory in the composition root).
* Every ``sqlite3`` error is translated (``_errors``); rows leave as domain models.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Final, Self

from adapters.sqlite._connection import SqliteConnection, SqliteSession, connect
from adapters.sqlite._errors import translate_sqlite_error
from adapters.sqlite.migrations import apply_migrations, verify_schema
from adapters.sqlite.repositories import (
    SqliteAIDecisionRepository,
    SqliteControlRepository,
    SqliteEquitySnapshotRepository,
    SqliteFillRepository,
    SqliteOrderEventRepository,
    SqliteOrderRepository,
    SqliteReconciliationRepository,
    SqliteRiskEventRepository,
    SqliteShadowOutcomeRepository,
    SqliteSignalRepository,
    SqliteSystemEventRepository,
    SqliteTradeRepository,
)
from adapters.sqlite.semantics import (
    DB_ERROR,
    DB_NOT_INITIALIZED,
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    UOW_ALREADY_ACTIVE,
)
from domain.errors import NonRetryableError
from domain.ports import IClock

__all__ = ["MIN_SQLITE_VERSION", "SqliteUnitOfWork"]

MIN_SQLITE_VERSION: Final = (3, 37, 0)
"""``STRICT`` tables need SQLite 3.37."""


def _check_sqlite_version() -> None:
    if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        required = ".".join(str(part) for part in MIN_SQLITE_VERSION)
        raise NonRetryableError(
            f"SQLite {sqlite3.sqlite_version} is too old; {required} or newer is required",
            code=DB_ERROR,
        )


class SqliteUnitOfWork:
    """``IUnitOfWork`` over the SQLite file ``db_path``.

    Args:
        db_path: Database file created by :meth:`initialize`.
        clock: Source of the ``*_updated_at_utc`` audit columns; they stay ``NULL``
            without a clock. Never used for record timestamps (those come from models).
        lock_timeout_seconds: SQLite busy timeout while another connection holds the
            write lock.
        durable: ``True`` (default, paper/live): ``synchronous=FULL``, every commit is
            fsynced. ``False`` only for throw-away databases (backtest temp DB): faster
            commits that may be lost on power loss.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        clock: IClock | None = None,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        durable: bool = True,
    ) -> None:
        self._db_path = Path(db_path)
        self._timeout = lock_timeout_seconds
        self._durable = durable
        self._session = SqliteSession(clock)
        self._entering = False
        self._signals = SqliteSignalRepository(self._session)
        self._trades = SqliteTradeRepository(self._session)
        self._orders = SqliteOrderRepository(self._session)
        self._order_events = SqliteOrderEventRepository(self._session)
        self._fills = SqliteFillRepository(self._session)
        self._ai_decisions = SqliteAIDecisionRepository(self._session)
        self._risk_events = SqliteRiskEventRepository(self._session)
        self._equity_snapshots = SqliteEquitySnapshotRepository(self._session)
        self._reconciliations = SqliteReconciliationRepository(self._session)
        self._shadow_outcomes = SqliteShadowOutcomeRepository(self._session)
        self._control = SqliteControlRepository(self._session)
        self._system_events = SqliteSystemEventRepository(self._session)

    # ------------------------------------------------------------------ schema

    @staticmethod
    def initialize(
        db_path: Path, *, lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS
    ) -> None:
        """Create or migrate the database: WAL mode, then pending migrations.

        Idempotent (a second call changes nothing) and blocking: call it once at
        startup, e.g. ``await asyncio.to_thread(SqliteUnitOfWork.initialize, path)``.
        Parent directories are created.

        Raises:
            NonRetryableError: SQLite too old, directory not creatable, database newer
                than this code (``SCHEMA_VERSION_MISMATCH``) or not switchable to WAL.
            StateCriticalError: corrupt or non-writable database file.
        """
        _check_sqlite_version()
        path = Path(db_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise NonRetryableError(
                f"cannot create the database directory: {type(exc).__name__}",
                code=DB_NOT_INITIALIZED,
            ) from exc
        try:
            conn = connect(path, create=True, timeout_seconds=lock_timeout_seconds)
            try:
                mode = str(conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
                if mode != "wal":
                    raise NonRetryableError(
                        f"cannot enable WAL mode (journal mode is {mode!r})",
                        code=DB_NOT_INITIALIZED,
                    )
                apply_migrations(conn)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise translate_sqlite_error(exc, operation="initialize database") from exc

    # ------------------------------------------------------------------ repositories

    @property
    def signals(self) -> SqliteSignalRepository:
        """Signals repository."""
        return self._signals

    @property
    def trades(self) -> SqliteTradeRepository:
        """Trades repository."""
        return self._trades

    @property
    def orders(self) -> SqliteOrderRepository:
        """Orders repository."""
        return self._orders

    @property
    def order_events(self) -> SqliteOrderEventRepository:
        """Order events repository."""
        return self._order_events

    @property
    def fills(self) -> SqliteFillRepository:
        """Fills repository."""
        return self._fills

    @property
    def ai_decisions(self) -> SqliteAIDecisionRepository:
        """AI decisions repository."""
        return self._ai_decisions

    @property
    def risk_events(self) -> SqliteRiskEventRepository:
        """Risk events repository."""
        return self._risk_events

    @property
    def equity_snapshots(self) -> SqliteEquitySnapshotRepository:
        """Equity snapshots repository."""
        return self._equity_snapshots

    @property
    def reconciliations(self) -> SqliteReconciliationRepository:
        """Reconciliations repository."""
        return self._reconciliations

    @property
    def shadow_outcomes(self) -> SqliteShadowOutcomeRepository:
        """Shadow outcomes repository."""
        return self._shadow_outcomes

    @property
    def control(self) -> SqliteControlRepository:
        """System control repository."""
        return self._control

    @property
    def system_events(self) -> SqliteSystemEventRepository:
        """System events repository."""
        return self._system_events

    # ------------------------------------------------------------------ transaction

    async def __aenter__(self) -> Self:
        """Open the connection (no transaction yet: it starts on the first statement).

        Raises:
            NonRetryableError: ``UOW_ALREADY_ACTIVE``, ``DB_NOT_INITIALIZED`` (missing
                file, no schema, not WAL) or ``SCHEMA_VERSION_MISMATCH``.
            StateCriticalError: corrupt or unreadable database.
        """
        if self._session.active or self._entering:
            raise NonRetryableError(
                "this unit of work is already active; use one instance per task",
                code=UOW_ALREADY_ACTIVE,
            )
        if not self._db_path.is_file():
            raise NonRetryableError(
                "database file does not exist; run SqliteUnitOfWork.initialize first",
                code=DB_NOT_INITIALIZED,
            )
        self._entering = True
        try:
            connection = await SqliteConnection.open(
                self._db_path,
                timeout_seconds=self._timeout,
                verify=verify_schema,
                durable=self._durable,
            )
        finally:
            self._entering = False
        self._session.bind(connection)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Roll back anything not committed and close the connection."""
        connection = self._session.unbind()
        if connection is not None:
            await connection.close()

    async def commit(self) -> None:
        """Commit the current transaction (no-op without one).

        Raises:
            NonRetryableError: ``UOW_NOT_ACTIVE``; ``TRANSACTION_ABORTED`` if SQLite
                aborted the transaction earlier (nothing was committed).
        """
        await self._session.connection().commit()

    async def rollback(self) -> None:
        """Roll back the current transaction (no-op without one)."""
        await self._session.connection().rollback()
