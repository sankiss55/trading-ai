"""SQLite implementations of the repository ports (sec. 8.5, 38.1).

Repositories never commit: they run inside the transaction of their
``SqliteUnitOfWork`` and only exchange validated domain models.
"""

from adapters.sqlite.repositories.audit import (
    SqliteAIDecisionRepository,
    SqliteReconciliationRepository,
    SqliteRiskEventRepository,
    SqliteShadowOutcomeRepository,
    SqliteSystemEventRepository,
)
from adapters.sqlite.repositories.control import SqliteControlRepository
from adapters.sqlite.repositories.equity import SqliteEquitySnapshotRepository
from adapters.sqlite.repositories.orders import (
    SqliteFillRepository,
    SqliteOrderEventRepository,
    SqliteOrderRepository,
)
from adapters.sqlite.repositories.signals import SqliteSignalRepository
from adapters.sqlite.repositories.trades import SqliteTradeRepository

__all__ = [
    "SqliteAIDecisionRepository",
    "SqliteControlRepository",
    "SqliteEquitySnapshotRepository",
    "SqliteFillRepository",
    "SqliteOrderEventRepository",
    "SqliteOrderRepository",
    "SqliteReconciliationRepository",
    "SqliteRiskEventRepository",
    "SqliteShadowOutcomeRepository",
    "SqliteSignalRepository",
    "SqliteSystemEventRepository",
    "SqliteTradeRepository",
]
