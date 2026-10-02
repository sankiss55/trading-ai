"""Reusable ``IBroker`` contracts (sec. 22, 23, 49.2).

* ``BrokerContract``: behaviors any broker must have (simulated, fake or a real paper
  account). It only needs a ``BrokerHarness``.
* ``BracketSemanticsContract``: bar-level bracket semantics (stop first, gaps, DAY legs
  expiring at the close) that ``SimulatedBroker`` and ``FakeBroker`` must reproduce.
  It needs a ``BracketSemanticsHarness`` able to script market bars.

Subclass a contract in a ``test_*.py`` module and provide a ``broker_harness`` fixture.
Trade updates are observed through ``UpdateRecorder``, which subscribes to the stream
before acting, so the same tests work against a live stream.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from decimal import Decimal
from types import TracebackType
from typing import Protocol

import pytest

from domain.errors import NonRetryableError
from domain.models import (
    BracketOrderRequest,
    BrokerOrder,
    BrokerOrderStatus,
    OrderClass,
    OrderSide,
    OrderType,
    SimpleOrderRequest,
    TimeInForce,
    TradeUpdate,
    TradeUpdateEvent,
)
from domain.ports import IBroker

ACTIVE_STATUSES = frozenset(
    {
        BrokerOrderStatus.NEW,
        BrokerOrderStatus.ACCEPTED,
        BrokerOrderStatus.PENDING_NEW,
        BrokerOrderStatus.PARTIALLY_FILLED,
    }
)
UPDATE_TIMEOUT_SECONDS = 5.0


class BrokerHarness(Protocol):
    """What ``BrokerContract`` needs from an implementation under test."""

    @property
    def broker(self) -> IBroker:
        """Broker under test, starting flat and without open orders for ``symbol``."""
        ...

    @property
    def symbol(self) -> str:
        """Symbol used by the tests."""
        ...

    def client_order_id(self, tag: str) -> str:
        """A client order id unique for this test run (real brokers keep history)."""
        ...

    def bracket_request(
        self, client_order_id: str, *, qty: int = 1, tif: TimeInForce = TimeInForce.DAY
    ) -> BracketOrderRequest:
        """A valid MARKET bracket entry for ``symbol``."""
        ...

    def resting_bracket_request(self, client_order_id: str) -> BracketOrderRequest:
        """A valid LIMIT bracket entry that will not fill (limit far below the market)."""
        ...

    async def fill_pending_market_orders(self) -> None:
        """Drive (or wait for) the market so pending MARKET orders fill."""
        ...


class BracketSemanticsHarness(BrokerHarness, Protocol):
    """Harness that can also script the market bar by bar."""

    @property
    def stop_price(self) -> Decimal:
        """SL stop price used by ``bracket_request``."""
        ...

    @property
    def take_profit_price(self) -> Decimal:
        """TP limit price used by ``bracket_request``."""
        ...

    async def run_bar(self, open_: Decimal, high: Decimal, low: Decimal, close: Decimal) -> None:
        """Make the market trade one bar of ``symbol`` with these prices."""
        ...

    async def end_session(self) -> None:
        """Reach the end of the regular session."""
        ...


class UpdateRecorder:
    """Background consumer of ``IBroker.stream_trade_updates`` for assertions."""

    def __init__(self, broker: IBroker) -> None:
        self._broker = broker
        self._queue: asyncio.Queue[TradeUpdate] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.seen: list[TradeUpdate] = []

    async def __aenter__(self) -> UpdateRecorder:
        # Live implementations must make the subscription ready before returning from
        # ``stream_trade_updates`` iteration; simulated ones queue every update anyway.
        self._task = asyncio.create_task(self._consume())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _consume(self) -> None:
        async for update in self._broker.stream_trade_updates():
            self._queue.put_nowait(update)

    async def wait_for(self, predicate: Callable[[TradeUpdate], bool]) -> TradeUpdate:
        """Return the first (not yet consumed) update matching ``predicate``."""

        async def _find() -> TradeUpdate:
            while True:
                update = await self._queue.get()
                self.seen.append(update)
                if predicate(update):
                    return update

        return await asyncio.wait_for(_find(), UPDATE_TIMEOUT_SECONDS)


def is_event(event: TradeUpdateEvent, order_id: str) -> Callable[[TradeUpdate], bool]:
    """Predicate: update ``event`` for ``order_id``."""
    return lambda update: update.event is event and update.order.order_id == order_id


def leg(order: BrokerOrder, order_type: OrderType) -> BrokerOrder:
    """The single leg of ``order`` with ``order_type``."""
    matches = [item for item in order.legs if item.order_type is order_type]
    assert len(matches) == 1, f"expected one {order_type} leg, got {order.legs}"
    return matches[0]


async def lookup(broker: IBroker, client_order_id: str) -> BrokerOrder:
    """Broker order by client id; fails the test if unknown."""
    order = await broker.get_order_by_client_id(client_order_id)
    assert order is not None
    return order


async def position_qty(broker: IBroker, symbol: str) -> Decimal:
    """Position quantity of ``symbol`` (0 if flat)."""
    return sum((p.qty for p in await broker.get_positions() if p.symbol == symbol), Decimal(0))


async def open_filled_bracket(
    harness: BrokerHarness, tag: str, *, tif: TimeInForce = TimeInForce.DAY, qty: int = 2
) -> BrokerOrder:
    """Submit a MARKET bracket, fill it and return the refreshed parent order."""
    coid = harness.client_order_id(tag)
    async with UpdateRecorder(harness.broker) as recorder:
        order = await harness.broker.submit_bracket(harness.bracket_request(coid, qty=qty, tif=tif))
        await harness.fill_pending_market_orders()
        await recorder.wait_for(is_event(TradeUpdateEvent.FILL, order.order_id))
    return await lookup(harness.broker, coid)


class BrokerContract:
    """Behaviors every ``IBroker`` implementation must have."""

    async def test_account_is_well_formed(self, broker_harness: BrokerHarness) -> None:
        account = await broker_harness.broker.get_account()
        assert account.status
        assert account.equity >= 0
        symbols = [p.symbol for p in await broker_harness.broker.get_positions()]
        assert len(symbols) == len(set(symbols))

    async def test_unknown_client_order_id_returns_none(
        self, broker_harness: BrokerHarness
    ) -> None:
        unknown = broker_harness.client_order_id("never-submitted")
        assert await broker_harness.broker.get_order_by_client_id(unknown) is None

    async def test_bracket_is_open_with_tp_and_sl_legs(self, broker_harness: BrokerHarness) -> None:
        h = broker_harness
        coid = h.client_order_id("bracket")
        request = h.resting_bracket_request(coid)
        order = await h.broker.submit_bracket(request)
        assert order.client_order_id == coid
        assert order.order_class is OrderClass.BRACKET
        assert order.side is OrderSide.BUY
        take_profit = leg(order, OrderType.LIMIT)
        stop_loss = leg(order, OrderType.STOP)
        assert take_profit.side is stop_loss.side is OrderSide.SELL
        assert take_profit.limit_price == request.take_profit_limit_price
        assert stop_loss.stop_price == request.stop_loss_stop_price
        assert take_profit.qty == stop_loss.qty == Decimal(request.qty)
        assert (await lookup(h.broker, coid)).order_id == order.order_id
        assert order.order_id in {o.order_id for o in await h.broker.get_open_orders()}

    async def test_repeated_client_order_id_never_creates_a_second_order(
        self, broker_harness: BrokerHarness
    ) -> None:
        h = broker_harness
        coid = h.client_order_id("dup")
        first = await h.broker.submit_bracket(h.resting_bracket_request(coid))
        rejection_code: str | None = None
        second: BrokerOrder | None = None
        try:
            second = await h.broker.submit_bracket(h.resting_bracket_request(coid))
        except NonRetryableError as exc:
            rejection_code = exc.code
        if second is None:
            assert rejection_code == "DUPLICATE_CLIENT_ORDER_ID"
        else:
            assert second.order_id == first.order_id
        matching = [o for o in await h.broker.get_open_orders() if o.client_order_id == coid]
        assert len(matching) == 1

    async def test_cancel_unfilled_entry(self, broker_harness: BrokerHarness) -> None:
        h = broker_harness
        coid = h.client_order_id("cancel")
        async with UpdateRecorder(h.broker) as recorder:
            order = await h.broker.submit_bracket(h.resting_bracket_request(coid))
            await h.broker.cancel_order(order.order_id)
            await recorder.wait_for(is_event(TradeUpdateEvent.CANCELED, order.order_id))
        assert (await lookup(h.broker, coid)).status is BrokerOrderStatus.CANCELED
        assert order.order_id not in {o.order_id for o in await h.broker.get_open_orders()}

    async def test_entry_fill_opens_position_and_activates_legs(
        self, broker_harness: BrokerHarness
    ) -> None:
        h = broker_harness
        parent = await open_filled_bracket(h, "fill", qty=2)
        assert parent.status is BrokerOrderStatus.FILLED
        assert parent.filled_qty == Decimal(2)
        assert parent.filled_avg_price is not None
        assert await position_qty(h.broker, h.symbol) == Decimal(2)
        for item in parent.legs:
            assert item.status in ACTIVE_STATUSES
        open_leg_ids = {item.order_id for o in await h.broker.get_open_orders() for item in o.legs}
        assert {item.order_id for item in parent.legs} <= open_leg_ids

    async def test_active_legs_hold_the_position_quantity(
        self, broker_harness: BrokerHarness
    ) -> None:
        h = broker_harness
        await open_filled_bracket(h, "held", qty=2)
        exit_request = SimpleOrderRequest(
            symbol=h.symbol,
            qty=2,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=h.client_order_id("held-exit"),
        )
        with pytest.raises(NonRetryableError):
            await h.broker.submit_simple(exit_request)

    async def test_system_exit_after_canceling_legs(self, broker_harness: BrokerHarness) -> None:
        """Procedure of sec. 23.6: cancel legs, then close with a market order."""
        h = broker_harness
        parent = await open_filled_bracket(h, "exit", qty=2)
        for item in parent.legs:
            await h.broker.cancel_order(item.order_id)
        exit_coid = h.client_order_id("exit-close")
        async with UpdateRecorder(h.broker) as recorder:
            exit_order = await h.broker.submit_simple(
                SimpleOrderRequest(
                    symbol=h.symbol,
                    qty=2,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=exit_coid,
                )
            )
            await h.fill_pending_market_orders()
            update = await recorder.wait_for(is_event(TradeUpdateEvent.FILL, exit_order.order_id))
        assert update.position_qty == Decimal(0)
        assert await position_qty(h.broker, h.symbol) == Decimal(0)


class BracketSemanticsContract:
    """Bar-level bracket semantics shared by simulated brokers (sec. 45.2, 49.2)."""

    async def test_stop_first_when_stop_and_take_profit_touch_the_same_bar(
        self, broker_harness: BracketSemanticsHarness
    ) -> None:
        h = broker_harness
        parent = await open_filled_bracket(h, "both")
        sl, tp = leg(parent, OrderType.STOP), leg(parent, OrderType.LIMIT)
        async with UpdateRecorder(h.broker) as recorder:
            await h.run_bar(
                h.stop_price + 1, h.take_profit_price + 1, h.stop_price - 1, h.stop_price
            )
            fill = await recorder.wait_for(is_event(TradeUpdateEvent.FILL, sl.order_id))
            await recorder.wait_for(is_event(TradeUpdateEvent.CANCELED, tp.order_id))
        assert fill.fill is not None
        assert fill.fill.price <= h.stop_price
        tp_fills = [
            u
            for u in recorder.seen
            if u.order.order_id == tp.order_id and u.event is TradeUpdateEvent.FILL
        ]
        assert not tp_fills
        assert await position_qty(h.broker, h.symbol) == Decimal(0)

    async def test_take_profit_fill_cancels_stop(
        self, broker_harness: BracketSemanticsHarness
    ) -> None:
        h = broker_harness
        parent = await open_filled_bracket(h, "tp")
        sl, tp = leg(parent, OrderType.STOP), leg(parent, OrderType.LIMIT)
        async with UpdateRecorder(h.broker) as recorder:
            await h.run_bar(
                h.take_profit_price - 1,
                h.take_profit_price + 1,
                h.take_profit_price - 2,
                h.take_profit_price,
            )
            fill = await recorder.wait_for(is_event(TradeUpdateEvent.FILL, tp.order_id))
            await recorder.wait_for(is_event(TradeUpdateEvent.CANCELED, sl.order_id))
        assert fill.fill is not None
        assert fill.fill.price >= h.take_profit_price
        assert await position_qty(h.broker, h.symbol) == Decimal(0)

    async def test_gap_through_stop_fills_at_open_or_worse(
        self, broker_harness: BracketSemanticsHarness
    ) -> None:
        h = broker_harness
        parent = await open_filled_bracket(h, "gap")
        sl = leg(parent, OrderType.STOP)
        gap_open = h.stop_price - 2
        async with UpdateRecorder(h.broker) as recorder:
            await h.run_bar(gap_open, gap_open + Decimal("0.5"), gap_open - 1, gap_open)
            fill = await recorder.wait_for(is_event(TradeUpdateEvent.FILL, sl.order_id))
        assert fill.fill is not None
        assert fill.fill.price <= gap_open

    async def test_day_legs_do_not_survive_the_session_close(
        self, broker_harness: BracketSemanticsHarness
    ) -> None:
        h = broker_harness
        parent = await open_filled_bracket(h, "day", tif=TimeInForce.DAY)
        await h.end_session()
        refreshed = await lookup(h.broker, parent.client_order_id)
        assert all(item.status not in ACTIVE_STATUSES for item in refreshed.legs)
        assert await position_qty(h.broker, h.symbol) == Decimal(2)

    async def test_gtc_legs_survive_the_session_close(
        self, broker_harness: BracketSemanticsHarness
    ) -> None:
        h = broker_harness
        parent = await open_filled_bracket(h, "gtc", tif=TimeInForce.GTC)
        await h.end_session()
        refreshed = await lookup(h.broker, parent.client_order_id)
        assert all(item.status in ACTIVE_STATUSES for item in refreshed.legs)
