"""``InMemoryUnitOfWork``: ``IUnitOfWork`` for tests and deterministic simulation (sec. 8.6).

It reproduces the observable semantics of ``SqliteUnitOfWork`` (both pass
``tests/contract/unit_of_work_contract.py``):

* ``InMemoryDatabase`` plays the role of the database file. Several units of work
  share one database: ``InMemoryUnitOfWork(uow.database)`` is "a new connection".
* Transactions start lazily on the first repository call and take the database write
  lock (like ``BEGIN IMMEDIATE``): a second unit of work waits up to
  ``lock_timeout_seconds`` and then raises ``RetryableError(code="DB_BUSY")``.
* Writes go to a per-transaction overlay: ``commit()`` publishes them, ``rollback()`` or
  leaving the context without committing discards them. A failed operation changes
  nothing; earlier writes of the transaction are kept.
* Same unique keys, foreign keys, final states, UTC windows and error codes as SQLite
  (``adapters.sqlite.semantics``), and the same JSON normalization of stored records
  (``normalized``). Returned models are deep copies: callers can never alter stored state.

No wall clock, no I/O. The lock is an ``asyncio.Lock``: use a database from one event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from types import TracebackType
from typing import Generic, Self, TypeVar

from adapters.sqlite.semantics import (
    DB_BUSY,
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    DUPLICATE_KEY,
    FINAL_STATES,
    FOREIGN_KEY_VIOLATION,
    ORDER_ID_MISMATCH,
    ORDER_NOT_FOUND,
    SIGNAL_NOT_FOUND,
    TRADE_IDENTITY_CHANGED,
    TRADE_NOT_FOUND,
    UOW_ALREADY_ACTIVE,
    UOW_NOT_ACTIVE,
    day_window,
    month_window,
    normalized,
)
from domain.errors import NonRetryableError, RetryableError, StateCriticalError
from domain.models import (
    AIDecisionRecord,
    BrokerOrderStatus,
    DomainModel,
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

__all__ = ["InMemoryDatabase", "InMemoryUnitOfWork"]

V = TypeVar("V")
M = TypeVar("M", bound=DomainModel)


@dataclass(frozen=True)
class _SignalRow:
    signal: Signal
    status: TradeState
    reason: str | None


def _duplicate(constraint: str) -> NonRetryableError:
    return NonRetryableError(f"UNIQUE constraint failed: {constraint}", code=DUPLICATE_KEY)


def _foreign_key() -> NonRetryableError:
    return NonRetryableError("FOREIGN KEY constraint failed", code=FOREIGN_KEY_VIOLATION)


def _copy(model: M) -> M:
    return model.model_copy(deep=True)


class InMemoryDatabase:
    """Committed state shared by every ``InMemoryUnitOfWork`` bound to it.

    The read-only properties expose committed append-only records (copies) so tests
    can assert what was persisted.
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.signals: dict[str, _SignalRow] = {}
        self.trades: dict[str, Trade] = {}
        self.orders: dict[str, OrderRecord] = {}
        self.fills: dict[str, Fill] = {}
        self.order_events: list[OrderEvent] = []
        self.ai_decisions: list[AIDecisionRecord] = []
        self.risk_events: list[RiskEvent] = []
        self.equity_snapshots: list[EquitySnapshot] = []
        self.reconciliations: list[ReconciliationRecord] = []
        self.shadow_outcomes: list[ShadowOutcome] = []
        self.system_events: list[SystemEvent] = []
        self.control: SystemControl | None = None

    def committed_order_events(self) -> tuple[OrderEvent, ...]:
        """Committed order events, in append order."""
        return tuple(_copy(item) for item in self.order_events)

    def committed_fills(self) -> tuple[Fill, ...]:
        """Committed fills, in insertion order."""
        return tuple(_copy(item) for item in self.fills.values())

    def committed_risk_events(self) -> tuple[RiskEvent, ...]:
        """Committed risk events, in append order."""
        return tuple(_copy(item) for item in self.risk_events)

    def committed_ai_decisions(self) -> tuple[AIDecisionRecord, ...]:
        """Committed AI decisions, in insertion order."""
        return tuple(_copy(item) for item in self.ai_decisions)

    def committed_reconciliations(self) -> tuple[ReconciliationRecord, ...]:
        """Committed reconciliation records, in insertion order."""
        return tuple(_copy(item) for item in self.reconciliations)

    def committed_system_events(self) -> tuple[SystemEvent, ...]:
        """Committed system events, in append order."""
        return tuple(_copy(item) for item in self.system_events)


class _Overlay(Generic[V]):
    """Keyed table seen through the uncommitted changes of one transaction."""

    def __init__(self, committed: dict[str, V]) -> None:
        self._committed = committed
        self._changes: dict[str, V] = {}

    def get(self, key: str) -> V | None:
        if key in self._changes:
            return self._changes[key]
        return self._committed.get(key)

    def put(self, key: str, value: V) -> None:
        self._changes[key] = value

    def values(self) -> list[V]:
        """Rows in insertion order (an update keeps the row's position, like a rowid)."""
        merged = dict(self._committed)
        merged.update(self._changes)
        return list(merged.values())

    def publish(self) -> None:
        self._committed.update(self._changes)


class _Log(Generic[V]):
    """Append-only table seen through the uncommitted appends of one transaction."""

    def __init__(self, committed: list[V]) -> None:
        self._committed = committed
        self._pending: list[V] = []

    def append(self, value: V) -> None:
        self._pending.append(value)

    def __iter__(self) -> Iterator[V]:
        yield from self._committed
        yield from self._pending

    def publish(self) -> None:
        self._committed.extend(self._pending)


class _Transaction:
    """Uncommitted view of the database owned by one unit of work."""

    def __init__(self, db: InMemoryDatabase) -> None:
        self._db = db
        self.signals = _Overlay(db.signals)
        self.trades = _Overlay(db.trades)
        self.orders = _Overlay(db.orders)
        self.fills = _Overlay(db.fills)
        self.order_events = _Log(db.order_events)
        self.ai_decisions = _Log(db.ai_decisions)
        self.risk_events = _Log(db.risk_events)
        self.equity_snapshots = _Log(db.equity_snapshots)
        self.reconciliations = _Log(db.reconciliations)
        self.shadow_outcomes = _Log(db.shadow_outcomes)
        self.system_events = _Log(db.system_events)
        self.control = db.control
        self.control_written = False

    def publish(self) -> None:
        for table in (self.signals, self.trades, self.orders, self.fills):
            table.publish()
        for log in (
            self.order_events,
            self.ai_decisions,
            self.risk_events,
            self.equity_snapshots,
            self.reconciliations,
            self.shadow_outcomes,
            self.system_events,
        ):
            log.publish()
        if self.control_written:
            self._db.control = self.control


class _Session:
    """Transaction state of one ``InMemoryUnitOfWork``, shared by its repositories."""

    def __init__(self, db: InMemoryDatabase, lock_timeout_seconds: float) -> None:
        self.db = db
        self._timeout = lock_timeout_seconds
        self.active = False
        self._tx: _Transaction | None = None
        self._begin_lock = asyncio.Lock()

    async def tx(self) -> _Transaction:
        """Current transaction, started (write lock taken) on first use."""
        self.require_active()
        async with self._begin_lock:
            if self._tx is None:
                try:
                    await asyncio.wait_for(self.db.lock.acquire(), self._timeout)
                except TimeoutError as exc:
                    raise RetryableError("database is locked", code=DB_BUSY) from exc
                self._tx = _Transaction(self.db)
            return self._tx

    def require_active(self) -> None:
        if not self.active:
            raise NonRetryableError(
                "unit of work is not active; use 'async with uow:'", code=UOW_NOT_ACTIVE
            )

    def end(self, *, publish: bool) -> None:
        """Publish (commit) or discard the transaction and release the write lock."""
        tx, self._tx = self._tx, None
        if tx is None:
            return
        if publish:
            tx.publish()
        self.db.lock.release()


class _Repository:
    def __init__(self, session: _Session) -> None:
        self._session = session


class _SignalRepository(_Repository):
    async def add(self, signal: Signal) -> None:
        tx = await self._session.tx()
        if tx.signals.get(signal.signal_id) is not None:
            raise _duplicate("signals.signal_id")
        tx.signals.put(
            signal.signal_id, _SignalRow(normalized(signal), TradeState.SIGNAL_CREATED, None)
        )

    async def get(self, signal_id: str) -> Signal | None:
        row = (await self._session.tx()).signals.get(signal_id)
        return None if row is None else _copy(row.signal)

    async def exists(self, signal_id: str) -> bool:
        return (await self._session.tx()).signals.get(signal_id) is not None

    async def mark_status(
        self, signal_id: str, status: TradeState, reason: str | None = None
    ) -> None:
        tx = await self._session.tx()
        row = tx.signals.get(signal_id)
        if row is None:
            raise NonRetryableError(f"signal {signal_id!r} not found", code=SIGNAL_NOT_FOUND)
        tx.signals.put(signal_id, replace(row, status=status, reason=reason))

    async def list_pending(self) -> list[Signal]:
        rows = (await self._session.tx()).signals.values()
        return [_copy(row.signal) for row in rows if row.status not in FINAL_STATES]


class _TradeRepository(_Repository):
    async def add(self, trade: Trade) -> None:
        tx = await self._session.tx()
        if tx.trades.get(trade.trade_id) is not None:
            raise _duplicate("trades.trade_id")
        if any(stored.signal_id == trade.signal_id for stored in tx.trades.values()):
            raise _duplicate("trades.signal_id")
        if tx.signals.get(trade.signal_id) is None:
            raise _foreign_key()
        tx.trades.put(trade.trade_id, normalized(trade))

    async def get(self, trade_id: str) -> Trade | None:
        trade = (await self._session.tx()).trades.get(trade_id)
        return None if trade is None else _copy(trade)

    async def get_by_signal(self, signal_id: str) -> Trade | None:
        trades = (await self._session.tx()).trades.values()
        return next((_copy(t) for t in trades if t.signal_id == signal_id), None)

    async def list_open(self) -> list[Trade]:
        trades = (await self._session.tx()).trades.values()
        return [_copy(t) for t in trades if t.state not in FINAL_STATES]

    async def update(self, trade: Trade) -> None:
        tx = await self._session.tx()
        stored = tx.trades.get(trade.trade_id)
        if stored is None:
            raise NonRetryableError(f"trade {trade.trade_id!r} not found", code=TRADE_NOT_FOUND)
        if stored.signal_id != trade.signal_id:
            raise NonRetryableError(
                f"trade {trade.trade_id!r} belongs to another signal; "
                "trade_id and signal_id cannot change",
                code=TRADE_IDENTITY_CHANGED,
            )
        tx.trades.put(trade.trade_id, normalized(trade))


class _OrderRepository(_Repository):
    async def add(self, order: OrderRecord) -> None:
        tx = await self._session.tx()
        if tx.orders.get(order.client_order_id) is not None:
            raise _duplicate("orders.client_order_id")
        if order.order_id is not None and any(
            stored.order_id == order.order_id for stored in tx.orders.values()
        ):
            raise _duplicate("orders.order_id")
        if tx.trades.get(order.trade_id) is None:
            raise _foreign_key()
        tx.orders.put(order.client_order_id, normalized(order))

    async def get_by_client_id(self, client_order_id: str) -> OrderRecord | None:
        order = (await self._session.tx()).orders.get(client_order_id)
        return None if order is None else _copy(order)

    async def list_non_final(self) -> list[OrderRecord]:
        orders = (await self._session.tx()).orders.values()
        return [_copy(o) for o in orders if o.state not in FINAL_STATES]

    async def update_status(
        self,
        client_order_id: str,
        state: TradeState,
        *,
        broker_status: BrokerOrderStatus | None = None,
        order_id: str | None = None,
    ) -> None:
        tx = await self._session.tx()
        stored = tx.orders.get(client_order_id)
        if stored is None:
            raise NonRetryableError(f"order {client_order_id!r} not found", code=ORDER_NOT_FOUND)
        if order_id is not None and stored.order_id is not None and stored.order_id != order_id:
            raise StateCriticalError(
                f"order {client_order_id!r} already has broker order id {stored.order_id!r}, "
                f"got {order_id!r}",
                code=ORDER_ID_MISMATCH,
            )
        if order_id is not None and any(
            other.order_id == order_id and other.client_order_id != client_order_id
            for other in tx.orders.values()
        ):
            raise _duplicate("orders.order_id")
        updated = stored.model_copy(
            update={
                "state": state,
                "broker_status": broker_status
                if broker_status is not None
                else stored.broker_status,
                "order_id": order_id if order_id is not None else stored.order_id,
            }
        )
        tx.orders.put(client_order_id, normalized(updated))


class _OrderEventRepository(_Repository):
    async def append(self, event: OrderEvent) -> None:
        (await self._session.tx()).order_events.append(normalized(event))


class _FillRepository(_Repository):
    async def add_if_new(self, fill: Fill) -> bool:
        tx = await self._session.tx()
        if tx.fills.get(fill.activity_id) is not None:
            return False
        tx.fills.put(fill.activity_id, normalized(fill))
        return True


def _between(start: datetime, end: datetime) -> Callable[[AIDecisionRecord], bool]:
    return lambda record: start <= record.created_at_utc < end


class _AIDecisionRepository(_Repository):
    async def add(self, record: AIDecisionRecord) -> None:
        (await self._session.tx()).ai_decisions.append(normalized(record))

    async def count_today(self, day: date) -> int:
        start, end = day_window(day)
        inside = _between(start, end)
        return sum(1 for record in (await self._session.tx()).ai_decisions if inside(record))

    async def cost_today(self, day: date) -> Decimal:
        start, end = day_window(day)
        return await self._cost(_between(start, end))

    async def cost_month(self, year: int, month: int) -> Decimal:
        start, end = month_window(year, month)
        return await self._cost(_between(start, end))

    async def _cost(self, inside: Callable[[AIDecisionRecord], bool]) -> Decimal:
        records = (await self._session.tx()).ai_decisions
        return sum((r.estimated_cost_usd for r in records if inside(r)), Decimal(0))


class _RiskEventRepository(_Repository):
    async def append(self, event: RiskEvent) -> None:
        (await self._session.tx()).risk_events.append(normalized(event))


class _EquitySnapshotRepository(_Repository):
    async def add(self, snapshot: EquitySnapshot) -> None:
        (await self._session.tx()).equity_snapshots.append(normalized(snapshot))

    async def peak_equity(self) -> Decimal | None:
        values = [s.equity for s in (await self._session.tx()).equity_snapshots]
        return max(values) if values else None

    async def week_start_equity(self, week_start: date) -> Decimal | None:
        start, _ = day_window(week_start)
        candidates = [
            s for s in (await self._session.tx()).equity_snapshots if s.taken_at_utc >= start
        ]
        if not candidates:
            return None
        # min() keeps the first of equal timestamps: insertion order, like ORDER BY ..., id.
        return min(candidates, key=lambda s: s.taken_at_utc).equity


class _ReconciliationRepository(_Repository):
    async def add(self, record: ReconciliationRecord) -> None:
        (await self._session.tx()).reconciliations.append(normalized(record))


class _ShadowOutcomeRepository(_Repository):
    async def add(self, outcome: ShadowOutcome) -> None:
        (await self._session.tx()).shadow_outcomes.append(normalized(outcome))

    async def list_for_report(self, since_utc: datetime | None = None) -> list[ShadowOutcome]:
        outcomes = [
            o
            for o in (await self._session.tx()).shadow_outcomes
            if since_utc is None or o.recorded_at_utc >= since_utc
        ]
        # sorted() is stable: equal timestamps keep insertion order.
        return [_copy(o) for o in sorted(outcomes, key=lambda o: o.recorded_at_utc)]


class _ControlRepository(_Repository):
    async def get(self) -> SystemControl:
        control = (await self._session.tx()).control
        return SystemControl() if control is None else _copy(control)

    async def set(self, control: SystemControl) -> None:
        tx = await self._session.tx()
        tx.control = normalized(control)
        tx.control_written = True


class _SystemEventRepository(_Repository):
    async def append(self, event: SystemEvent) -> None:
        (await self._session.tx()).system_events.append(normalized(event))


class InMemoryUnitOfWork:
    """``IUnitOfWork`` over an ``InMemoryDatabase``.

    Args:
        database: Shared committed state; a fresh empty database when omitted.
        lock_timeout_seconds: How long to wait for another unit of work's transaction.
    """

    def __init__(
        self,
        database: InMemoryDatabase | None = None,
        *,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        self._database = database if database is not None else InMemoryDatabase()
        self._session = _Session(self._database, lock_timeout_seconds)
        self._signals = _SignalRepository(self._session)
        self._trades = _TradeRepository(self._session)
        self._orders = _OrderRepository(self._session)
        self._order_events = _OrderEventRepository(self._session)
        self._fills = _FillRepository(self._session)
        self._ai_decisions = _AIDecisionRepository(self._session)
        self._risk_events = _RiskEventRepository(self._session)
        self._equity_snapshots = _EquitySnapshotRepository(self._session)
        self._reconciliations = _ReconciliationRepository(self._session)
        self._shadow_outcomes = _ShadowOutcomeRepository(self._session)
        self._control = _ControlRepository(self._session)
        self._system_events = _SystemEventRepository(self._session)

    @property
    def database(self) -> InMemoryDatabase:
        """The shared database (pass it to another ``InMemoryUnitOfWork``)."""
        return self._database

    @property
    def signals(self) -> _SignalRepository:
        """Signals repository."""
        return self._signals

    @property
    def trades(self) -> _TradeRepository:
        """Trades repository."""
        return self._trades

    @property
    def orders(self) -> _OrderRepository:
        """Orders repository."""
        return self._orders

    @property
    def order_events(self) -> _OrderEventRepository:
        """Order events repository."""
        return self._order_events

    @property
    def fills(self) -> _FillRepository:
        """Fills repository."""
        return self._fills

    @property
    def ai_decisions(self) -> _AIDecisionRepository:
        """AI decisions repository."""
        return self._ai_decisions

    @property
    def risk_events(self) -> _RiskEventRepository:
        """Risk events repository."""
        return self._risk_events

    @property
    def equity_snapshots(self) -> _EquitySnapshotRepository:
        """Equity snapshots repository."""
        return self._equity_snapshots

    @property
    def reconciliations(self) -> _ReconciliationRepository:
        """Reconciliations repository."""
        return self._reconciliations

    @property
    def shadow_outcomes(self) -> _ShadowOutcomeRepository:
        """Shadow outcomes repository."""
        return self._shadow_outcomes

    @property
    def control(self) -> _ControlRepository:
        """System control repository."""
        return self._control

    @property
    def system_events(self) -> _SystemEventRepository:
        """System events repository."""
        return self._system_events

    async def __aenter__(self) -> Self:
        """Activate the unit of work (the transaction starts on the first call)."""
        if self._session.active:
            raise NonRetryableError(
                "this unit of work is already active; use one instance per task",
                code=UOW_ALREADY_ACTIVE,
            )
        self._session.active = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Discard anything not committed and deactivate."""
        self._session.end(publish=False)
        self._session.active = False

    async def commit(self) -> None:
        """Publish the current transaction (no-op without one)."""
        self._session.require_active()
        self._session.end(publish=True)

    async def rollback(self) -> None:
        """Discard the current transaction (no-op without one)."""
        self._session.require_active()
        self._session.end(publish=False)
