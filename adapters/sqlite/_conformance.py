"""Static (mypy-only) proof that both unit-of-work adapters satisfy ``IUnitOfWork``.

Nothing here runs: the assignments are type-checked by ``mypy adapters``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.simulation.in_memory_uow import InMemoryUnitOfWork
    from adapters.sqlite.unit_of_work import SqliteUnitOfWork
    from domain.ports import IUnitOfWork

    def _check_conformance(sqlite_uow: SqliteUnitOfWork, memory_uow: InMemoryUnitOfWork) -> None:
        _sqlite: IUnitOfWork = sqlite_uow
        _memory: IUnitOfWork = memory_uow
