"""Ports (``typing.Protocol``) through which the domain talks to the outside (sec. 8.5)."""

from domain.ports.ai_filter import IAIFilter
from domain.ports.broker import IBroker
from domain.ports.clock import IClock
from domain.ports.market_data import IMarketCalendar, IMarketData
from domain.ports.notifier import INotifier
from domain.ports.persistence import (
    IAIDecisionRepository,
    IControlRepository,
    IEquitySnapshotRepository,
    IFillRepository,
    IOrderEventRepository,
    IOrderRepository,
    IReconciliationRepository,
    IRiskEventRepository,
    IShadowOutcomeRepository,
    ISignalRepository,
    ISystemEventRepository,
    ITradeRepository,
    IUnitOfWork,
)

__all__ = [
    "IAIDecisionRepository",
    "IAIFilter",
    "IBroker",
    "IClock",
    "IControlRepository",
    "IEquitySnapshotRepository",
    "IFillRepository",
    "IMarketCalendar",
    "IMarketData",
    "INotifier",
    "IOrderEventRepository",
    "IOrderRepository",
    "IReconciliationRepository",
    "IRiskEventRepository",
    "IShadowOutcomeRepository",
    "ISignalRepository",
    "ISystemEventRepository",
    "ITradeRepository",
    "IUnitOfWork",
]
