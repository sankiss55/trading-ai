"""Test doubles for the Phase 2 runtime tests (no network, no SQLite).

* :class:`FakeUnitOfWork`: transactional in-memory ``IUnitOfWork`` limited to the
  repositories the runtime uses (control, system events, equity snapshots,
  reconciliations). Writes are staged and applied on ``commit``; leaving the context
  without commit discards them. ``fail_writes`` / ``fail_reads`` simulate a broken DB.
* :class:`FakeBroker`: ``IBroker`` with a fixed account; any order call fails the test
  (Phase 2 never submits orders).
* :class:`FakeCalendar`: ``IMarketCalendar`` whose broker clock is the test clock plus
  a configurable skew.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

from domain.errors import RetryableError
from domain.models import (
    AccountState,
    BracketOrderRequest,
    BrokerOrder,
    EquitySnapshot,
    MarketClock,
    Position,
    ReconciliationRecord,
    SessionDay,
    SimpleOrderRequest,
    SystemControl,
    SystemEvent,
    TradeUpdate,
)
from domain.ports import (
    IAIDecisionRepository,
    IClock,
    IFillRepository,
    IOrderEventRepository,
    IOrderRepository,
    IRiskEventRepository,
    IShadowOutcomeRepository,
    ISignalRepository,
    ITradeRepository,
)


class DatabaseDownError(Exception):
    """Simulated database failure."""


@dataclass
class _State:
    control: SystemControl | None = None
    system_events: list[SystemEvent] = field(default_factory=list)
    equity_snapshots: list[EquitySnapshot] = field(default_factory=list)
    reconciliations: list[ReconciliationRecord] = field(default_factory=list)


class _Unused:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str) -> Any:
        raise AssertionError(f"repository {self._name!r} must not be used in Phase 2")


class FakeUnitOfWork:
    """In-memory transactional unit of work (one shared instance per test)."""

    def __init__(self) -> None:
        self.committed = _State()
        self._staged: _State | None = None
        self.fail_writes = False
        self.fail_reads = False
        self.commits = 0

    # -------------------------------------------------------------- transaction

    async def __aenter__(self) -> Self:
        if self._staged is not None:
            raise AssertionError("nested transaction")
        self._staged = _State(control=self.committed.control)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._staged = None

    async def commit(self) -> None:
        staged = self._txn()
        if self.fail_writes:
            raise DatabaseDownError("commit failed")
        self.committed.control = staged.control
        self.committed.system_events.extend(staged.system_events)
        self.committed.equity_snapshots.extend(staged.equity_snapshots)
        self.committed.reconciliations.extend(staged.reconciliations)
        self._staged = _State(control=self.committed.control)
        self.commits += 1

    async def rollback(self) -> None:
        self._staged = _State(control=self.committed.control)

    def _txn(self) -> _State:
        if self._staged is None:
            raise AssertionError("repository used outside a transaction")
        return self._staged

    def _write(self) -> _State:
        if self.fail_writes:
            raise DatabaseDownError("database not writable")
        return self._txn()

    def _read(self) -> _State:
        if self.fail_reads:
            raise DatabaseDownError("database not readable")
        return self._txn()

    # -------------------------------------------------------------- repositories

    @property
    def control(self) -> _ControlRepo:
        return _ControlRepo(self)

    @property
    def system_events(self) -> _EventRepo:
        return _EventRepo(self)

    @property
    def equity_snapshots(self) -> _EquityRepo:
        return _EquityRepo(self)

    @property
    def reconciliations(self) -> _ReconciliationRepo:
        return _ReconciliationRepo(self)

    @property
    def signals(self) -> ISignalRepository:
        return cast(ISignalRepository, _Unused("signals"))

    @property
    def trades(self) -> ITradeRepository:
        return cast(ITradeRepository, _Unused("trades"))

    @property
    def orders(self) -> IOrderRepository:
        return cast(IOrderRepository, _Unused("orders"))

    @property
    def order_events(self) -> IOrderEventRepository:
        return cast(IOrderEventRepository, _Unused("order_events"))

    @property
    def fills(self) -> IFillRepository:
        return cast(IFillRepository, _Unused("fills"))

    @property
    def ai_decisions(self) -> IAIDecisionRepository:
        return cast(IAIDecisionRepository, _Unused("ai_decisions"))

    @property
    def risk_events(self) -> IRiskEventRepository:
        return cast(IRiskEventRepository, _Unused("risk_events"))

    @property
    def shadow_outcomes(self) -> IShadowOutcomeRepository:
        return cast(IShadowOutcomeRepository, _Unused("shadow_outcomes"))

    # -------------------------------------------------------------- helpers

    def events(self, event_type: str | None = None) -> list[SystemEvent]:
        """Committed system events, optionally of one type."""
        return [
            event
            for event in self.committed.system_events
            if event_type is None or event.event_type == event_type
        ]

    def factory(self) -> FakeUnitOfWork:
        """``UnitOfWorkFactory`` returning this shared instance."""
        return self


class _ControlRepo:
    def __init__(self, uow: FakeUnitOfWork) -> None:
        self._uow = uow

    async def get(self) -> SystemControl:
        return self._uow._read().control or SystemControl()

    async def set(self, control: SystemControl) -> None:
        self._uow._write().control = control


class _EventRepo:
    def __init__(self, uow: FakeUnitOfWork) -> None:
        self._uow = uow

    async def append(self, event: SystemEvent) -> None:
        self._uow._write().system_events.append(event)


class _EquityRepo:
    def __init__(self, uow: FakeUnitOfWork) -> None:
        self._uow = uow

    async def add(self, snapshot: EquitySnapshot) -> None:
        self._uow._write().equity_snapshots.append(snapshot)

    async def peak_equity(self) -> Decimal | None:
        values = [s.equity for s in self._uow.committed.equity_snapshots]
        return max(values) if values else None

    async def week_start_equity(self, week_start: date) -> Decimal | None:
        return None


class _ReconciliationRepo:
    def __init__(self, uow: FakeUnitOfWork) -> None:
        self._uow = uow

    async def add(self, record: ReconciliationRecord) -> None:
        self._uow._write().reconciliations.append(record)


ACTIVE_ACCOUNT = AccountState(
    equity=Decimal("100000"),
    last_equity=Decimal("100000"),
    buying_power=Decimal("100000"),
    status="ACTIVE",
)


class FakeBroker:
    """``IBroker`` for observation tests; every order call is a test failure."""

    def __init__(
        self,
        *,
        account: AccountState = ACTIVE_ACCOUNT,
        positions: Sequence[Position] = (),
        orders: Sequence[BrokerOrder] = (),
        error: Exception | None = None,
    ) -> None:
        self.account = account
        self.positions = list(positions)
        self.orders = list(orders)
        self.error = error
        self.order_calls = 0
        self.closed = False

    async def get_account(self) -> AccountState:
        if self.error is not None:
            raise self.error
        return self.account

    async def get_positions(self) -> list[Position]:
        if self.error is not None:
            raise self.error
        return list(self.positions)

    async def get_open_orders(self) -> list[BrokerOrder]:
        if self.error is not None:
            raise self.error
        return list(self.orders)

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        return None

    async def submit_bracket(self, request: BracketOrderRequest) -> BrokerOrder:
        self.order_calls += 1
        raise AssertionError("Phase 2 must never submit orders")

    async def submit_simple(self, request: SimpleOrderRequest) -> BrokerOrder:
        self.order_calls += 1
        raise AssertionError("Phase 2 must never submit orders")

    async def cancel_order(self, order_id: str) -> None:
        self.order_calls += 1
        raise AssertionError("Phase 2 must never cancel orders")

    def stream_trade_updates(self) -> AsyncIterator[TradeUpdate]:
        raise AssertionError("Phase 2 does not stream trade updates")

    async def aclose(self) -> None:
        self.closed = True


class FakeCalendar:
    """``IMarketCalendar`` whose broker clock is ``clock + skew``."""

    def __init__(
        self,
        clock: IClock,
        *,
        skew: timedelta = timedelta(0),
        is_open: bool = False,
        error: Exception | None = None,
    ) -> None:
        self._clock = clock
        self.skew = skew
        self.is_open = is_open
        self.error = error

    async def get_clock(self) -> MarketClock:
        if self.error is not None:
            raise self.error
        now = self._clock.now_utc() + self.skew
        return MarketClock(
            is_open=self.is_open,
            now_utc=now,
            next_open_utc=now + timedelta(hours=1),
            next_close_utc=now + timedelta(hours=2),
        )

    async def get_session(self, day: date) -> SessionDay | None:
        return None


def unreachable() -> RetryableError:
    """A broker network failure as the Alpaca adapters raise it."""
    return RetryableError("connection timed out", code="NETWORK")


def at(hour: int, minute: int = 0) -> datetime:
    """2026-10-02 (a Friday) at ``hour:minute`` UTC."""
    return datetime(2026, 10, 2, hour, minute, tzinfo=UTC)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REAL_CONFIG = PROJECT_ROOT / "config.yaml"
PENDING_CONFIG = PROJECT_ROOT / "tests" / "fixtures" / "config.pending.yaml"


def write_config(
    directory: Path, *, pending: bool = False, poll_seconds: str | None = None
) -> Path:
    """Copy ``config.yaml`` (or the all-null pending fixture) into ``directory``.

    Its relative ``system`` paths (DB, logs, STOP file) then resolve inside ``directory``.
    """
    text = (PENDING_CONFIG if pending else REAL_CONFIG).read_text(encoding="utf-8")
    if poll_seconds is not None:
        old = "control_poll_seconds: 5 "
        assert old in text
        text = text.replace(old, f"control_poll_seconds: {poll_seconds} ", 1)
    path = directory / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path
