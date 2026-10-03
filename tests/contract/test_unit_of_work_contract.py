"""``IUnitOfWork`` contract run against ``SqliteUnitOfWork`` and ``InMemoryUnitOfWork``."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.simulation.in_memory_uow import InMemoryDatabase, InMemoryUnitOfWork
from adapters.sqlite.unit_of_work import SqliteUnitOfWork
from domain.ports import IUnitOfWork
from tests.contract.unit_of_work_contract import (
    UnitOfWorkContract,
    make_fill,
    make_order_event,
    make_system_event,
)


class SqliteHarness:
    """Units of work on one temporary SQLite file."""

    def __init__(self, db_path: Path) -> None:
        SqliteUnitOfWork.initialize(db_path)
        self.db_path = db_path

    def new_uow(self, *, lock_timeout_seconds: float = 5.0) -> IUnitOfWork:
        return SqliteUnitOfWork(self.db_path, lock_timeout_seconds=lock_timeout_seconds)


class InMemoryHarness:
    """Units of work on one shared in-memory database."""

    def __init__(self) -> None:
        self.database = InMemoryDatabase()

    def new_uow(self, *, lock_timeout_seconds: float = 5.0) -> IUnitOfWork:
        return InMemoryUnitOfWork(self.database, lock_timeout_seconds=lock_timeout_seconds)


class TestSqliteUnitOfWorkContract(UnitOfWorkContract):
    @pytest.fixture
    def uow_harness(self, tmp_path: Path) -> SqliteHarness:
        return SqliteHarness(tmp_path / "operational.sqlite3")


class TestInMemoryUnitOfWorkContract(UnitOfWorkContract):
    @pytest.fixture
    def uow_harness(self) -> InMemoryHarness:
        return InMemoryHarness()


async def test_in_memory_database_exposes_committed_records_only() -> None:
    database = InMemoryDatabase()
    async with InMemoryUnitOfWork(database) as uow:
        await uow.order_events.append(make_order_event())
        await uow.fills.add_if_new(make_fill())
        await uow.system_events.append(make_system_event())
        assert database.committed_order_events() == ()
        await uow.commit()
        await uow.system_events.append(make_system_event("NOT_COMMITTED"))
    assert database.committed_order_events() == (make_order_event(),)
    assert database.committed_fills() == (make_fill(),)
    assert [e.event_type for e in database.committed_system_events()] == ["STARTUP"]
    assert database.committed_risk_events() == ()
    assert database.committed_ai_decisions() == ()
    assert database.committed_reconciliations() == ()
