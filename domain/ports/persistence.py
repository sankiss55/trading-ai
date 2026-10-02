"""Persistence ports: repositories grouped by a unit of work (sec. 8.5, 38).

All operations are ``async`` so that adapters can offload blocking I/O (SQLite) without
blocking the event loop. Repositories never commit on their own: ``IUnitOfWork``
delimits transactions. The ``SUBMITTING`` state (sec. 20) MUST be committed with
``commit()`` before calling ``IBroker``.

Day/month arguments refer to the calendar date in UTC of the record timestamp. The
caller obtains "today" from ``IClock``; repositories never read system time.
"""

from datetime import date, datetime
from decimal import Decimal
from types import TracebackType
from typing import Protocol, Self

from domain.models import (
    AIDecisionRecord,
    BrokerOrderStatus,
    EquitySnapshot,
    Fill,
    OrderEvent,
    OrderRecord,
    ReconciliationRecord,
    RiskEvent,
    ShadowOutcome,
    Signal,
    SystemControl,
    SystemEvent,
    Trade,
    TradeState,
)

__all__ = [
    "IAIDecisionRepository",
    "IControlRepository",
    "IEquitySnapshotRepository",
    "IFillRepository",
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


class ISignalRepository(Protocol):
    """``signals`` table. ``signal_id`` is unique."""

    async def add(self, signal: Signal) -> None:
        """Insert a new signal with state ``SIGNAL_CREATED``."""
        ...

    async def get(self, signal_id: str) -> Signal | None:
        """Signal by id, or ``None``."""
        ...

    async def exists(self, signal_id: str) -> bool:
        """Whether a signal with this id was already recorded."""
        ...

    async def mark_status(
        self, signal_id: str, status: TradeState, reason: str | None = None
    ) -> None:
        """Set the signal's lifecycle state, with an optional reason code."""
        ...

    async def list_pending(self) -> list[Signal]:
        """Signals not yet in a final state."""
        ...


class ITradeRepository(Protocol):
    """``trades`` table. ``signal_id`` is unique (one trade per signal)."""

    async def add(self, trade: Trade) -> None:
        """Insert a new trade."""
        ...

    async def get(self, trade_id: str) -> Trade | None:
        """Trade by id, or ``None``."""
        ...

    async def get_by_signal(self, signal_id: str) -> Trade | None:
        """Trade originated by a signal, or ``None``."""
        ...

    async def list_open(self) -> list[Trade]:
        """Trades not in a final state."""
        ...

    async def update(self, trade: Trade) -> None:
        """Replace the stored trade with this (new, immutable) version."""
        ...


class IOrderRepository(Protocol):
    """``orders`` table. ``client_order_id`` is unique."""

    async def add(self, order: OrderRecord) -> None:
        """Insert a new order record."""
        ...

    async def get_by_client_id(self, client_order_id: str) -> OrderRecord | None:
        """Order record by client id, or ``None``."""
        ...

    async def list_non_final(self) -> list[OrderRecord]:
        """Orders whose state is not final (input of reconciliation, sec. 24.2)."""
        ...

    async def update_status(
        self,
        client_order_id: str,
        state: TradeState,
        *,
        broker_status: BrokerOrderStatus | None = None,
        order_id: str | None = None,
    ) -> None:
        """Update the local state and, when known, the broker status and order id."""
        ...


class IOrderEventRepository(Protocol):
    """``order_events`` table (append-only)."""

    async def append(self, event: OrderEvent) -> None:
        """Append one state transition."""
        ...


class IFillRepository(Protocol):
    """``fills`` table, unique by broker activity id."""

    async def add_if_new(self, fill: Fill) -> bool:
        """Insert the fill if its ``activity_id`` is unknown. Returns ``True`` if inserted."""
        ...


class IAIDecisionRepository(Protocol):
    """``ai_decisions`` table and AI budget queries (sec. 34)."""

    async def add(self, record: AIDecisionRecord) -> None:
        """Insert one AI decision."""
        ...

    async def count_today(self, day: date) -> int:
        """Number of AI calls recorded on ``day`` (UTC date)."""
        ...

    async def cost_today(self, day: date) -> Decimal:
        """Estimated AI cost in USD recorded on ``day`` (UTC date)."""
        ...

    async def cost_month(self, year: int, month: int) -> Decimal:
        """Estimated AI cost in USD recorded in the given month (UTC)."""
        ...


class IRiskEventRepository(Protocol):
    """``risk_events`` table (append-only)."""

    async def append(self, event: RiskEvent) -> None:
        """Append one check/sizing evaluation."""
        ...


class IEquitySnapshotRepository(Protocol):
    """``equity_snapshots`` table."""

    async def add(self, snapshot: EquitySnapshot) -> None:
        """Insert one equity snapshot."""
        ...

    async def peak_equity(self) -> Decimal | None:
        """Highest recorded equity, or ``None`` if there are no snapshots."""
        ...

    async def week_start_equity(self, week_start: date) -> Decimal | None:
        """Equity of the first snapshot on or after ``week_start``, or ``None``."""
        ...


class IReconciliationRepository(Protocol):
    """``reconciliations`` table."""

    async def add(self, record: ReconciliationRecord) -> None:
        """Insert one reconciliation result."""
        ...


class IShadowOutcomeRepository(Protocol):
    """``shadow_outcomes`` table (sec. 46.2)."""

    async def add(self, outcome: ShadowOutcome) -> None:
        """Insert one counterfactual outcome."""
        ...

    async def list_for_report(self, since_utc: datetime | None = None) -> list[ShadowOutcome]:
        """Outcomes for the shadow report, optionally only those recorded since a time."""
        ...


class IControlRepository(Protocol):
    """``system_control`` table (sec. 31.1). Written only by the control CLI."""

    async def get(self) -> SystemControl:
        """Current runtime controls (fail-closed defaults if never written)."""
        ...

    async def set(self, control: SystemControl) -> None:
        """Replace the runtime controls."""
        ...


class ISystemEventRepository(Protocol):
    """``system_events`` table (append-only)."""

    async def append(self, event: SystemEvent) -> None:
        """Append one system event."""
        ...


class IUnitOfWork(Protocol):
    """Groups the repositories and delimits one transaction.

    Usage::

        async with uow:
            await uow.orders.add(record)
            await uow.commit()

    Leaving the context without ``commit()`` (or with an exception) rolls back.
    """

    @property
    def signals(self) -> ISignalRepository:
        """Signals repository."""
        ...

    @property
    def trades(self) -> ITradeRepository:
        """Trades repository."""
        ...

    @property
    def orders(self) -> IOrderRepository:
        """Orders repository."""
        ...

    @property
    def order_events(self) -> IOrderEventRepository:
        """Order events repository."""
        ...

    @property
    def fills(self) -> IFillRepository:
        """Fills repository."""
        ...

    @property
    def ai_decisions(self) -> IAIDecisionRepository:
        """AI decisions repository."""
        ...

    @property
    def risk_events(self) -> IRiskEventRepository:
        """Risk events repository."""
        ...

    @property
    def equity_snapshots(self) -> IEquitySnapshotRepository:
        """Equity snapshots repository."""
        ...

    @property
    def reconciliations(self) -> IReconciliationRepository:
        """Reconciliations repository."""
        ...

    @property
    def shadow_outcomes(self) -> IShadowOutcomeRepository:
        """Shadow outcomes repository."""
        ...

    @property
    def control(self) -> IControlRepository:
        """System control repository."""
        ...

    @property
    def system_events(self) -> ISystemEventRepository:
        """System events repository."""
        ...

    async def __aenter__(self) -> Self:
        """Begin a transaction."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """End the transaction; roll back anything not committed."""
        ...

    async def commit(self) -> None:
        """Commit the current transaction."""
        ...

    async def rollback(self) -> None:
        """Roll back the current transaction."""
        ...
