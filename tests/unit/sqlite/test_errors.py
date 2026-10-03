"""Translation of real ``sqlite3`` errors into domain errors (sec. 8.3.7, 29)."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from adapters.sqlite._connection import SqliteConnection
from adapters.sqlite._errors import translate_sqlite_error
from adapters.sqlite.migrations import verify_schema
from adapters.sqlite.unit_of_work import SqliteUnitOfWork
from domain.errors import DomainError, NonRetryableError, RetryableError, StateCriticalError
from tests.contract.unit_of_work_contract import make_signal


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(tmp_path / "e.sqlite3", isolation_level=None, timeout=0)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("CREATE TABLE p (id TEXT NOT NULL PRIMARY KEY, n INTEGER CHECK (n > 0))")
    connection.execute("CREATE TABLE c (id TEXT PRIMARY KEY, p_id TEXT REFERENCES p (id))")
    connection.execute("CREATE UNIQUE INDEX ux ON c (p_id)")
    connection.execute("INSERT INTO p VALUES ('a', 1)")
    yield connection
    connection.close()


def _translated(action: Callable[[], object]) -> DomainError:
    with pytest.raises(sqlite3.Error) as excinfo:
        action()
    return translate_sqlite_error(excinfo.value, operation="test.op")


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("INSERT INTO p VALUES ('a', 2)", "DUPLICATE_KEY"),
        ("INSERT INTO c VALUES ('x', 'missing')", "FOREIGN_KEY_VIOLATION"),
        ("INSERT INTO p VALUES ('b', 0)", "CONSTRAINT_VIOLATION"),
        ("INSERT INTO p VALUES (NULL, 1)", "CONSTRAINT_VIOLATION"),
        ("SELECT * FROM no_such_table", "DB_ERROR"),
        ("THIS IS NOT SQL", "DB_ERROR"),
    ],
)
def test_constraint_and_sql_errors_are_non_retryable(
    conn: sqlite3.Connection, sql: str, code: str
) -> None:
    error = _translated(lambda: conn.execute(sql))
    assert type(error) is NonRetryableError
    assert error.code == code
    assert error.message.startswith("test.op: SQLITE_")


def test_unique_index_violation_is_a_duplicate(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO c VALUES ('x', 'a')")
    error = _translated(lambda: conn.execute("INSERT INTO c VALUES ('y', 'a')"))
    assert error.code == "DUPLICATE_KEY"


def test_busy_database_is_retryable(conn: sqlite3.Connection, tmp_path: Path) -> None:
    conn.execute("BEGIN IMMEDIATE")
    other = sqlite3.connect(tmp_path / "e.sqlite3", isolation_level=None, timeout=0)
    try:
        error = _translated(lambda: other.execute("BEGIN IMMEDIATE"))
    finally:
        other.close()
        conn.execute("ROLLBACK")
    assert type(error) is RetryableError
    assert error.code == "DB_BUSY"


def test_corrupt_file_is_state_critical(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"garbage" * 1000)
    bad = sqlite3.connect(path)
    try:
        error = _translated(lambda: bad.execute("SELECT * FROM sqlite_master"))
    finally:
        bad.close()
    assert type(error) is StateCriticalError
    assert error.code == "DB_CORRUPT"


def test_read_only_database_is_state_critical(conn: sqlite3.Connection, tmp_path: Path) -> None:
    read_only = sqlite3.connect(f"{(tmp_path / 'e.sqlite3').as_uri()}?mode=ro", uri=True)
    try:
        error = _translated(lambda: read_only.execute("INSERT INTO p VALUES ('z', 1)"))
    finally:
        read_only.close()
    assert type(error) is StateCriticalError
    assert error.code == "DB_NOT_WRITABLE"


def test_python_level_misuse_is_non_retryable(tmp_path: Path) -> None:
    closed = sqlite3.connect(tmp_path / "closed.sqlite3")
    closed.close()
    error = _translated(lambda: closed.execute("SELECT 1"))
    assert type(error) is NonRetryableError
    assert error.code == "DB_ERROR"


async def test_no_sqlite_exception_leaves_the_unit_of_work(tmp_path: Path) -> None:
    path = tmp_path / "op.sqlite3"
    SqliteUnitOfWork.initialize(path)
    async with SqliteUnitOfWork(path) as uow:
        await uow.signals.add(make_signal())
        with pytest.raises(NonRetryableError) as excinfo:
            await uow.signals.add(make_signal())
    assert isinstance(excinfo.value.__cause__, sqlite3.IntegrityError)  # chained, not raised


async def test_transaction_aborted_by_sqlite_cannot_be_committed(tmp_path: Path) -> None:
    """If SQLite rolls the whole transaction back after an error, the partial work of
    the caller must never be committed by a later implicit BEGIN."""
    path = tmp_path / "op.sqlite3"
    SqliteUnitOfWork.initialize(path)
    connection = await SqliteConnection.open(path, timeout_seconds=1, verify=verify_schema)

    def insert(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO system_events VALUES (NULL, 'a', 'T', '{}')")

    def fail_after_sqlite_rollback(conn: sqlite3.Connection) -> None:
        conn.execute("ROLLBACK")  # what SQLite does itself on e.g. SQLITE_FULL
        raise sqlite3.OperationalError("simulated abort")

    try:
        await connection.run("t.insert", insert)
        with pytest.raises(NonRetryableError):
            await connection.run("t.fail", fail_after_sqlite_rollback)
        with pytest.raises(NonRetryableError) as excinfo:
            await connection.run("t.insert", insert)
        assert excinfo.value.code == "TRANSACTION_ABORTED"
        with pytest.raises(NonRetryableError) as excinfo:
            await connection.commit()
        assert excinfo.value.code == "TRANSACTION_ABORTED"
        await connection.run("t.insert", insert)  # usable again after the refusal
        await connection.commit()
    finally:
        await connection.close()
    raw = sqlite3.connect(path)
    try:
        assert raw.execute("SELECT COUNT(*) FROM system_events").fetchone()[0] == 1
    finally:
        raw.close()
