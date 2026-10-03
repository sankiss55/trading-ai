"""Backup and restore of the operational database (sec. 58.2)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from adapters.sqlite.backup import backup_database, backup_file_name, restore_database
from adapters.sqlite.unit_of_work import SqliteUnitOfWork
from domain.errors import NonRetryableError
from domain.models import SystemControl
from tests.contract.unit_of_work_contract import make_signal

STAMP = datetime(2026, 10, 2, 20, 15, 30, tzinfo=UTC)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "live" / "trading.sqlite3"
    SqliteUnitOfWork.initialize(path)
    return path


async def _add_signal(db_path: Path, signal_id: str) -> None:
    async with SqliteUnitOfWork(db_path) as uow:
        await uow.signals.add(make_signal(signal_id))
        await uow.commit()


def _journal_mode(path: Path) -> str:
    conn = sqlite3.connect(path)
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0])
    finally:
        conn.close()


def test_backup_file_name_is_utc_stamped() -> None:
    assert (
        backup_file_name(Path("data/trading.sqlite3"), STAMP) == "trading-20261002T201530Z.sqlite3"
    )
    with pytest.raises(NonRetryableError):
        backup_file_name(
            Path("t.sqlite3"), datetime(2026, 10, 2, tzinfo=timezone(timedelta(hours=2)))
        )
    with pytest.raises(NonRetryableError):
        backup_file_name(Path("t.sqlite3"), STAMP.replace(tzinfo=None))


async def test_backup_and_restore_round_trip(db_path: Path, tmp_path: Path) -> None:
    await _add_signal(db_path, "sig-1")
    async with SqliteUnitOfWork(db_path) as uow:
        await uow.control.set(SystemControl(trading_enabled=True, updated_at_utc=STAMP))
        await uow.commit()

    backup = backup_database(db_path, tmp_path / "backups", timestamp=STAMP)

    assert backup == tmp_path / "backups" / "trading-20261002T201530Z.sqlite3"
    assert backup.is_file()
    assert not backup.with_name(backup.name + ".partial").exists()
    assert _journal_mode(backup) == "delete"  # self-contained file, no -wal needed

    restored = tmp_path / "restored" / "trading.sqlite3"
    restore_database(backup, restored)
    assert _journal_mode(restored) == "wal"
    SqliteUnitOfWork.initialize(restored)
    async with SqliteUnitOfWork(restored) as uow:
        assert await uow.signals.exists("sig-1")
        assert await uow.control.get() == SystemControl(trading_enabled=True, updated_at_utc=STAMP)


async def test_restore_replaces_the_current_content(db_path: Path, tmp_path: Path) -> None:
    await _add_signal(db_path, "sig-1")
    backup = backup_database(db_path, tmp_path / "backups", timestamp=STAMP)
    await _add_signal(db_path, "sig-after-backup")

    restore_database(backup, db_path)

    async with SqliteUnitOfWork(db_path) as uow:
        assert await uow.signals.exists("sig-1")
        assert not await uow.signals.exists("sig-after-backup")


async def test_backup_contains_only_committed_data(db_path: Path, tmp_path: Path) -> None:
    await _add_signal(db_path, "sig-1")
    async with SqliteUnitOfWork(db_path) as writer:
        await writer.signals.add(make_signal("uncommitted"))  # holds the write lock
        backup = backup_database(db_path, tmp_path / "backups", timestamp=STAMP)
    restored = tmp_path / "restored.sqlite3"
    restore_database(backup, restored)
    async with SqliteUnitOfWork(restored) as uow:
        assert await uow.signals.exists("sig-1")
        assert not await uow.signals.exists("uncommitted")


async def test_backup_never_overwrites(db_path: Path, tmp_path: Path) -> None:
    backup_database(db_path, tmp_path / "backups", timestamp=STAMP)
    with pytest.raises(NonRetryableError) as excinfo:
        backup_database(db_path, tmp_path / "backups", timestamp=STAMP)
    assert excinfo.value.code == "BACKUP_EXISTS"


def test_backup_of_a_missing_database_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NonRetryableError) as excinfo:
        backup_database(tmp_path / "missing.sqlite3", tmp_path / "b", timestamp=STAMP)
    assert excinfo.value.code == "DB_NOT_INITIALIZED"
    assert not (tmp_path / "b").exists()


@pytest.mark.parametrize("content", [None, b"not a database" * 100, b""])
def test_restore_refuses_invalid_backups(
    db_path: Path, tmp_path: Path, content: bytes | None
) -> None:
    candidate = tmp_path / "candidate.sqlite3"
    if content is not None:
        candidate.write_bytes(content)
    with pytest.raises(NonRetryableError) as excinfo:
        restore_database(candidate, db_path)
    assert excinfo.value.code == "INVALID_BACKUP"


def test_restore_refuses_a_database_without_schema(db_path: Path, tmp_path: Path) -> None:
    plain = tmp_path / "plain.sqlite3"
    conn = sqlite3.connect(plain)
    conn.execute("CREATE TABLE t (x)")
    conn.close()
    with pytest.raises(NonRetryableError) as excinfo:
        restore_database(plain, db_path)
    assert excinfo.value.code == "INVALID_BACKUP"
