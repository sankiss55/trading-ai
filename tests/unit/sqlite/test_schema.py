"""Schema, migrations and connection settings of the SQLite adapter (sec. 38)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from adapters.sqlite import unit_of_work as uow_module
from adapters.sqlite._connection import connect
from adapters.sqlite.migrations import LATEST_VERSION, MIGRATIONS
from adapters.sqlite.unit_of_work import SqliteUnitOfWork
from domain.errors import NonRetryableError, StateCriticalError
from tests.contract.unit_of_work_contract import make_signal

EXPECTED_TABLES = {
    "schema_version",
    "signals",
    "trades",
    "orders",
    "order_events",
    "fills",
    "ai_decisions",
    "risk_events",
    "equity_snapshots",
    "reconciliations",
    "shadow_outcomes",
    "system_control",
    "system_events",
}


def _raw(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path, isolation_level=None)


def _tables(path: Path) -> set[str]:
    conn = _raw(path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "nested" / "dir" / "operational.sqlite3"
    SqliteUnitOfWork.initialize(path)
    return path


def test_migration_registry_is_contiguous() -> None:
    assert [m.version for m in MIGRATIONS] == list(range(1, len(MIGRATIONS) + 1))
    assert MIGRATIONS[-1].version == LATEST_VERSION
    assert len({m.name for m in MIGRATIONS}) == len(MIGRATIONS)


def test_initialize_creates_parent_dirs_tables_and_wal(db_path: Path) -> None:
    assert db_path.is_file()
    assert _tables(db_path) == EXPECTED_TABLES
    conn = _raw(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        versions = conn.execute(
            "SELECT version, name, applied_at_utc FROM schema_version"
        ).fetchall()
        strict = conn.execute(
            "SELECT name, strict FROM pragma_table_list WHERE schema = 'main'"
        ).fetchall()
    finally:
        conn.close()
    assert [(v, n) for v, n, _ in versions] == [(m.version, m.name) for m in MIGRATIONS]
    assert versions[0][2].endswith("Z")
    assert all(flag == 1 for name, flag in strict if name in EXPECTED_TABLES)


async def test_initialize_is_idempotent_and_keeps_data(db_path: Path) -> None:
    async with SqliteUnitOfWork(db_path) as uow:
        await uow.signals.add(make_signal())
        await uow.commit()
    SqliteUnitOfWork.initialize(db_path)
    SqliteUnitOfWork.initialize(db_path)
    conn = _raw(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == len(MIGRATIONS)
    finally:
        conn.close()
    async with SqliteUnitOfWork(db_path) as uow:
        assert await uow.signals.exists("sig-1")


def test_connections_enforce_foreign_keys_and_full_sync(db_path: Path) -> None:
    conn = connect(db_path, create=False, timeout_seconds=1)
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
        assert conn.isolation_level is None
    finally:
        conn.close()


def test_connect_without_create_never_creates_a_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        connect(missing, create=False, timeout_seconds=1)
    assert not missing.exists()


async def test_missing_database_is_refused_and_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(NonRetryableError) as excinfo:
        async with SqliteUnitOfWork(missing):
            pass
    assert excinfo.value.code == "DB_NOT_INITIALIZED"
    assert not missing.exists()


async def test_database_without_schema_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.sqlite3"
    conn = _raw(empty)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.close()
    with pytest.raises(NonRetryableError) as excinfo:
        async with SqliteUnitOfWork(empty):
            pass
    assert excinfo.value.code == "DB_NOT_INITIALIZED"


async def test_database_not_in_wal_mode_is_refused(db_path: Path) -> None:
    conn = _raw(db_path)
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.close()
    with pytest.raises(NonRetryableError) as excinfo:
        async with SqliteUnitOfWork(db_path):
            pass
    assert excinfo.value.code == "DB_NOT_INITIALIZED"
    SqliteUnitOfWork.initialize(db_path)  # restores WAL
    async with SqliteUnitOfWork(db_path) as uow:
        assert await uow.signals.list_pending() == []


async def test_newer_schema_is_refused_by_initialize_and_open(db_path: Path) -> None:
    conn = _raw(db_path)
    conn.execute(
        "INSERT INTO schema_version (version, name, applied_at_utc) VALUES (?, 'future', 'x')",
        (LATEST_VERSION + 1,),
    )
    conn.close()
    with pytest.raises(NonRetryableError) as excinfo:
        SqliteUnitOfWork.initialize(db_path)
    assert excinfo.value.code == "SCHEMA_VERSION_MISMATCH"
    with pytest.raises(NonRetryableError) as excinfo:
        async with SqliteUnitOfWork(db_path):
            pass
    assert excinfo.value.code == "SCHEMA_VERSION_MISMATCH"


async def test_corrupt_file_is_state_critical(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"this is not a database" * 500)
    with pytest.raises(StateCriticalError) as excinfo:
        SqliteUnitOfWork.initialize(corrupt)
    assert excinfo.value.code == "DB_CORRUPT"
    with pytest.raises(StateCriticalError) as excinfo:
        async with SqliteUnitOfWork(corrupt):
            pass
    assert excinfo.value.code == "DB_CORRUPT"


def test_too_old_sqlite_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uow_module, "MIN_SQLITE_VERSION", (99, 0, 0))
    with pytest.raises(NonRetryableError, match="too old"):
        SqliteUnitOfWork.initialize(tmp_path / "x.sqlite3")
    assert not (tmp_path / "x.sqlite3").exists()


def test_non_durable_connections_use_normal_sync(db_path: Path) -> None:
    conn = connect(db_path, create=False, timeout_seconds=1, durable=False)
    try:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    finally:
        conn.close()


async def test_non_durable_unit_of_work_commits(db_path: Path) -> None:
    async with SqliteUnitOfWork(db_path, durable=False) as uow:
        await uow.signals.add(make_signal())
        await uow.commit()
    async with SqliteUnitOfWork(db_path) as uow:
        assert await uow.signals.exists("sig-1")
