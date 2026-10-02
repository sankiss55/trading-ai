"""Fill rules of ``SimulatedBroker`` with hand-calculated Decimal expectations."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from adapters.simulation.sim_clock import SimClock
from adapters.simulation.simulated_broker import SimulatedBroker
from domain.errors import NonRetryableError
from domain.models import (
    BrokerOrder,
    BrokerOrderStatus,
    OrderSide,
    OrderType,
    SimpleOrderRequest,
    TimeInForce,
    TradeUpdate,
    TradeUpdateEvent,
)
from domain.ports import IBroker
from tests.unit.simulation_adapters.builders import T0, bracket, empty_bar, minute_bar

MINUTE = timedelta(minutes=1)


class Sim:
    """Broker plus clock; ``bar`` processes the next 1-minute bar and advances time."""

    def __init__(
        self,
        *,
        cash: str = "100000",
        slippage_bps: str = "5",
        commission: str = "0",
        duplicate: str = "return_existing",
    ) -> None:
        self.clock = SimClock(T0)
        self.broker = SimulatedBroker(
            clock=self.clock,
            starting_cash=Decimal(cash),
            slippage_bps=Decimal(slippage_bps),
            commission_per_fill=Decimal(commission),
            duplicate_client_order_id=duplicate,  # type: ignore[arg-type]
        )

    def bar(
        self, open_: str, high: str, low: str, close: str, symbol: str = "SPY"
    ) -> list[TradeUpdate]:
        start = self.clock.now_utc()
        updates = self.broker.process_bar(minute_bar(start, open_, high, low, close, symbol=symbol))
        self.clock.advance_to(start + MINUTE)
        return updates

    async def order(self, client_order_id: str) -> BrokerOrder:
        order = await self.broker.get_order_by_client_id(client_order_id)
        assert order is not None
        return order


def leg(order: BrokerOrder, order_type: OrderType) -> BrokerOrder:
    return next(item for item in order.legs if item.order_type is order_type)


def fills(updates: list[TradeUpdate]) -> list[tuple[str, Decimal]]:
    """``(order_id, price)`` of every FILL update."""
    return [(u.order.order_id, u.fill.price) for u in updates if u.fill is not None]


def sell(
    client_order_id: str, qty: int, order_type: OrderType = OrderType.MARKET, **prices: str
) -> SimpleOrderRequest:
    return SimpleOrderRequest(
        symbol="SPY",
        qty=qty,
        side=OrderSide.SELL,
        order_type=order_type,
        limit_price=Decimal(prices["limit"]) if "limit" in prices else None,
        stop_price=Decimal(prices["stop"]) if "stop" in prices else None,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )


@pytest.fixture
def sim() -> Sim:
    return Sim()


def test_conforms_to_ibroker(sim: Sim) -> None:
    broker: IBroker = sim.broker
    assert broker is sim.broker


def test_rejects_float_and_negative_parameters() -> None:
    clock = SimClock(T0)
    with pytest.raises(TypeError):
        SimulatedBroker(clock=clock, starting_cash=1000.0, slippage_bps=Decimal(0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="slippage_bps"):
        SimulatedBroker(clock=clock, starting_cash=Decimal(1000), slippage_bps=Decimal(-1))


# --------------------------------------------------------------------------- entries


async def test_market_entry_fills_at_next_open_plus_slippage(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    assert order.status is BrokerOrderStatus.NEW
    assert {item.status for item in order.legs} == {BrokerOrderStatus.HELD}
    updates = sim.bar("100.00", "100.50", "99.50", "100.20")
    # 100.00 * (1 + 5/10000) = 100.05
    assert fills(updates) == [(order.order_id, Decimal("100.0500"))]
    entry_fill = updates[0]
    assert entry_fill.timestamp_utc == T0
    assert entry_fill.position_qty == Decimal(10)
    parent = await sim.order("paper-abc-entry")
    assert parent.status is BrokerOrderStatus.FILLED
    assert {item.status for item in parent.legs} == {BrokerOrderStatus.NEW}
    assert [u.event for u in updates] == [
        TradeUpdateEvent.FILL,
        TradeUpdateEvent.NEW,
        TradeUpdateEvent.NEW,
    ]


async def test_market_entry_slippage_rounds_up_to_the_quantum() -> None:
    sim = Sim(slippage_bps="3")
    await sim.broker.submit_bracket(bracket(stop="120", take_profit="130"))
    updates = sim.bar("123.45", "123.50", "123.40", "123.45")
    # 123.45 * 1.0003 = 123.487035 -> ceiling to 0.0001 = 123.4871
    assert fills(updates)[0][1] == Decimal("123.4871")


async def test_entry_never_fills_on_a_bar_that_started_before_submission(sim: Sim) -> None:
    sim.clock.advance_to(T0 + MINUTE)
    await sim.broker.submit_bracket(bracket())
    stale = minute_bar(T0, "100", "101", "99", "100")
    assert sim.broker.process_bar(stale) == []
    assert (await sim.order("paper-abc-entry")).status is BrokerOrderStatus.NEW


async def test_limit_entry_gapping_below_limit_fills_at_open_plus_slippage_capped(
    sim: Sim,
) -> None:
    await sim.broker.submit_bracket(bracket(limit="100.00"))
    updates = sim.bar("99.00", "99.50", "98.50", "99.20")
    # 99.00 * 1.0005 = 99.0495 < limit 100.00
    assert fills(updates)[0][1] == Decimal("99.0495")


async def test_limit_entry_slipped_open_is_capped_at_the_limit(sim: Sim) -> None:
    await sim.broker.submit_bracket(bracket(limit="100.00"))
    updates = sim.bar("99.99", "100.10", "99.90", "100.00")
    # 99.99 * 1.0005 = 100.039995 -> capped at 100.00
    assert fills(updates)[0][1] == Decimal("100.00")


async def test_limit_entry_needs_the_bar_to_trade_through_the_limit(sim: Sim) -> None:
    await sim.broker.submit_bracket(bracket(limit="100.00"))
    assert fills(sim.bar("100.50", "101.00", "100.00", "100.80")) == []  # touch only
    updates = sim.bar("100.50", "101.00", "99.99", "100.80")
    assert fills(updates)[0][1] == Decimal("100.00")
    assert updates[0].timestamp_utc == T0 + 2 * MINUTE  # intrabar fill -> bar end


# --------------------------------------------------------------------------- exits


async def test_stop_first_when_both_legs_are_touchable(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "100.50", "99.50", "100.00")
    sl, tp = leg(order, OrderType.STOP), leg(order, OrderType.LIMIT)
    updates = sim.bar("100.00", "103.00", "97.00", "101.00")
    # stop 98 * (1 - 0.0005) = 97.951
    assert fills(updates) == [(sl.order_id, Decimal("97.9510"))]
    assert [(u.event, u.order.order_id) for u in updates] == [
        (TradeUpdateEvent.FILL, sl.order_id),
        (TradeUpdateEvent.CANCELED, tp.order_id),
    ]
    assert updates[0].position_qty == Decimal(0)
    assert await sim.broker.get_positions() == []
    assert await sim.broker.get_open_orders() == []


async def test_gap_through_stop_fills_at_open_minus_slippage(sim: Sim) -> None:
    await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "100.50", "99.50", "100.00")
    updates = sim.bar("96.00", "96.50", "95.00", "96.00")
    # 96.00 * 0.9995 = 95.952 (worse than the 98 stop)
    assert fills(updates)[0][1] == Decimal("95.9520")
    assert updates[0].timestamp_utc == T0 + MINUTE  # at the open -> bar start


async def test_take_profit_requires_trading_through_and_fills_at_limit(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "100.50", "99.50", "100.00")
    assert fills(sim.bar("101.00", "102.00", "100.50", "101.50")) == []  # touch only
    tp, sl = leg(order, OrderType.LIMIT), leg(order, OrderType.STOP)
    updates = sim.bar("101.00", "102.01", "100.50", "101.50")
    assert fills(updates) == [(tp.order_id, Decimal(102))]
    assert (updates[1].event, updates[1].order.order_id) == (TradeUpdateEvent.CANCELED, sl.order_id)


async def test_take_profit_gap_above_limit_fills_at_open_minus_slippage(sim: Sim) -> None:
    await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "100.50", "99.50", "100.00")
    updates = sim.bar("103.00", "104.00", "102.50", "103.50")
    # max(103.00 * 0.9995 = 102.9485, 102) = 102.9485
    assert fills(updates)[0][1] == Decimal("102.9485")


async def test_legs_can_fill_in_the_same_bar_as_the_entry(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    updates = sim.bar("100.00", "100.50", "97.50", "98.00")
    assert fills(updates) == [
        (order.order_id, Decimal("100.0500")),
        (leg(order, OrderType.STOP).order_id, Decimal("97.9510")),
    ]


async def test_cash_and_equity_with_commission() -> None:
    sim = Sim(cash="10000", commission="1")
    await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "101.50", "99.50", "101.00")
    account = await sim.broker.get_account()
    # cash = 10000 - 10 * 100.05 - 1 = 8998.50; equity = 8998.50 + 10 * 101.00 = 10008.50
    assert sim.broker.cash == Decimal("8998.50")
    assert account.equity == Decimal("10008.50")
    assert account.buying_power == Decimal("8998.50")
    assert account.last_equity == Decimal(10000)
    [position] = await sim.broker.get_positions()
    assert (position.qty, position.avg_entry_price, position.market_value) == (
        Decimal(10),
        Decimal("100.05"),
        Decimal("1010.00"),
    )
    sim.bar("99.00", "99.50", "97.00", "97.50")
    # cash = 8998.50 + 10 * 97.951 - 1 = 9977.01
    assert sim.broker.cash == Decimal("9977.01")
    assert (await sim.broker.get_account()).equity == Decimal("9977.01")


async def test_average_entry_price_of_two_entries(sim: Sim) -> None:
    await sim.broker.submit_bracket(bracket("a", qty=10, take_profit="110"))
    sim.bar("100.00", "100.50", "99.50", "100.00")
    await sim.broker.submit_bracket(bracket("b", qty=30, stop="97", take_profit="110"))
    sim.bar("102.00", "102.50", "101.50", "102.00")
    [position] = await sim.broker.get_positions()
    # (10 * 100.05 + 30 * 102.051) / 40 = 4062.03 / 40 = 101.55075
    assert position.qty == Decimal(40)
    assert position.avg_entry_price == Decimal("101.55075")


# --------------------------------------------------------------------------- simple orders


async def test_active_legs_reserve_the_position_quantity(sim: Sim) -> None:
    await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "100.50", "99.50", "100.00")
    with pytest.raises(NonRetryableError) as excinfo:
        await sim.broker.submit_simple(sell("exit-1", 10))
    assert excinfo.value.code == "INSUFFICIENT_QTY_AVAILABLE"


async def test_system_exit_after_cancelling_legs_fills_at_next_open(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "100.50", "99.50", "100.00")
    for item in order.legs:
        await sim.broker.cancel_order(item.order_id)
    exit_order = await sim.broker.submit_simple(sell("exit-1", 10))
    updates = sim.bar("99.00", "99.50", "98.50", "99.00")
    # 99.00 * 0.9995 = 98.9505
    assert fills(updates) == [(exit_order.order_id, Decimal("98.9505"))]
    assert await sim.broker.get_positions() == []


async def test_protective_stop_fills_when_touched(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "100.50", "99.50", "100.00")
    for item in order.legs:
        await sim.broker.cancel_order(item.order_id)
    stop = await sim.broker.submit_simple(sell("protect-1", 10, OrderType.STOP, stop="99.00"))
    assert fills(sim.bar("99.50", "99.80", "99.01", "99.20")) == []
    updates = sim.bar("99.50", "99.80", "99.00", "99.20")
    # 99.00 * 0.9995 = 98.9505
    assert fills(updates) == [(stop.order_id, Decimal("98.9505"))]


async def test_simple_limit_sell_fills_at_limit(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket(take_profit="110"))
    sim.bar("100.00", "100.50", "99.50", "100.00")
    for item in order.legs:
        await sim.broker.cancel_order(item.order_id)
    limit = await sim.broker.submit_simple(sell("exit-l", 4, OrderType.LIMIT, limit="101.00"))
    updates = sim.bar("100.50", "101.20", "100.40", "101.00")
    assert fills(updates) == [(limit.order_id, Decimal("101.00"))]
    assert updates[0].position_qty == Decimal(6)


@pytest.mark.parametrize(
    ("request_", "code"),
    [
        (sell("x", 1), "NO_POSITION"),
        (
            SimpleOrderRequest(
                symbol="SPY",
                qty=1,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id="buy-1",
            ),
            "UNSUPPORTED_ORDER",
        ),
        (
            sell("y", 1, OrderType.STOP_LIMIT, stop="99", limit="98.50"),
            "UNSUPPORTED_ORDER",
        ),
    ],
)
async def test_simple_order_rejections(sim: Sim, request_: SimpleOrderRequest, code: str) -> None:
    with pytest.raises(NonRetryableError) as excinfo:
        await sim.broker.submit_simple(request_)
    assert excinfo.value.code == code
    assert await sim.broker.get_open_orders() == []


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("request_kwargs", "code"),
    [
        ({"stop": "98.005"}, "INVALID_PRICE_INCREMENT"),
        ({"take_profit": "102.001"}, "INVALID_PRICE_INCREMENT"),
        ({"stop": "102", "take_profit": "98"}, "INVALID_BRACKET_PRICES"),
        ({"limit": "97"}, "INVALID_BRACKET_PRICES"),
        ({"limit": "100", "qty": 2000}, "INSUFFICIENT_BUYING_POWER"),
    ],
)
async def test_invalid_brackets_raise_and_create_nothing(
    sim: Sim, request_kwargs: dict[str, object], code: str
) -> None:
    with pytest.raises(NonRetryableError) as excinfo:
        await sim.broker.submit_bracket(bracket(**request_kwargs))  # type: ignore[arg-type]
    assert excinfo.value.code == code
    assert await sim.broker.get_open_orders() == []
    assert sim.broker.drain_trade_updates() == []


async def test_sub_dollar_prices_allow_four_decimals(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(
        bracket(limit="0.5000", stop="0.4512", take_profit="0.5523")
    )
    assert order.status is BrokerOrderStatus.NEW


async def test_open_entries_reserve_buying_power_until_cancelled(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket(limit="100", qty=600))
    assert (await sim.broker.get_account()).buying_power == Decimal(40_000)
    with pytest.raises(NonRetryableError):
        await sim.broker.submit_bracket(bracket("second", limit="100", qty=600))
    await sim.broker.cancel_order(order.order_id)
    assert (await sim.broker.get_account()).buying_power == Decimal(100_000)


# --------------------------------------------------------------------------- idempotency


async def test_repeated_client_order_id_returns_the_existing_order(sim: Sim) -> None:
    first = await sim.broker.submit_bracket(bracket())
    sim.broker.drain_trade_updates()
    second = await sim.broker.submit_bracket(bracket(qty=99))
    assert second == first
    assert sim.broker.drain_trade_updates() == []
    assert len(await sim.broker.get_open_orders()) == 1


async def test_repeated_client_order_id_can_be_rejected() -> None:
    sim = Sim(duplicate="reject")
    await sim.broker.submit_bracket(bracket())
    with pytest.raises(NonRetryableError) as excinfo:
        await sim.broker.submit_bracket(bracket())
    assert excinfo.value.code == "DUPLICATE_CLIENT_ORDER_ID"


async def test_order_and_fill_ids_are_deterministic(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    assert order.order_id == "sim-000001"
    assert [item.order_id for item in order.legs] == ["sim-000002", "sim-000003"]
    sim.bar("100.00", "100.50", "99.50", "100.00")
    assert [f.activity_id for f in sim.broker.fills] == ["sim-fill-000001"]


# --------------------------------------------------------------------------- cancellation


async def test_cancel_unfilled_parent_cancels_held_legs(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    sim.broker.drain_trade_updates()
    await sim.broker.cancel_order(order.order_id)
    updates = sim.broker.drain_trade_updates()
    assert [u.event for u in updates] == [TradeUpdateEvent.CANCELED] * 3
    assert fills(sim.bar("100.00", "100.50", "99.50", "100.00")) == []
    await sim.broker.cancel_order(order.order_id)  # already canceled: no-op
    assert sim.broker.drain_trade_updates() == []


async def test_cancel_errors(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "100.50", "99.50", "100.00")
    with pytest.raises(NonRetryableError) as not_found:
        await sim.broker.cancel_order("nope")
    assert not_found.value.code == "ORDER_NOT_FOUND"
    with pytest.raises(NonRetryableError) as filled:
        await sim.broker.cancel_order(order.order_id)
    assert filled.value.code == "ORDER_NOT_CANCELABLE"


# --------------------------------------------------------------------------- session close


async def test_close_session_expires_day_legs_and_rolls_last_equity(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    gtc = await sim.broker.submit_bracket(bracket("gtc", tif=TimeInForce.GTC))
    sim.bar("100.00", "100.50", "99.50", "100.00")
    updates = sim.broker.close_session()
    expired = {u.order.order_id for u in updates}
    assert expired == {item.order_id for item in order.legs}
    assert all(u.event is TradeUpdateEvent.EXPIRED for u in updates)
    refreshed = await sim.order(gtc.client_order_id)
    assert {item.status for item in refreshed.legs} == {BrokerOrderStatus.NEW}
    account = await sim.broker.get_account()
    assert account.last_equity == account.equity
    # The DAY position is now unprotected: a market exit for its quantity is accepted.
    await sim.broker.submit_simple(sell("after-close", 10))


async def test_close_session_expires_unfilled_day_entry_and_releases_cash(sim: Sim) -> None:
    await sim.broker.submit_bracket(bracket(limit="50", stop="49", take_profit="60", qty=100))
    assert (await sim.broker.get_account()).buying_power == Decimal(95_000)
    updates = sim.broker.close_session()
    assert len(updates) == 3
    assert (await sim.broker.get_account()).buying_power == Decimal(100_000)


# --------------------------------------------------------------------------- bar handling


async def test_bars_must_be_processed_in_order_per_symbol(sim: Sim) -> None:
    sim.bar("100", "101", "99", "100")
    with pytest.raises(NonRetryableError) as excinfo:
        sim.broker.process_bar(minute_bar(T0, "100", "101", "99", "100"))
    assert excinfo.value.code == "BAR_OUT_OF_ORDER"
    sim.broker.process_bar(minute_bar(T0, "50", "51", "49", "50", symbol="QQQ"))


async def test_empty_bars_and_other_symbols_do_not_fill(sim: Sim) -> None:
    await sim.broker.submit_bracket(bracket())
    assert sim.broker.process_bar(empty_bar(T0)) == []
    sim.clock.advance_to(T0 + MINUTE)
    assert sim.bar("50", "51", "49", "50", symbol="QQQ") == []
    assert (await sim.order("paper-abc-entry")).status is BrokerOrderStatus.NEW


async def test_stream_yields_updates_in_emission_order(sim: Sim) -> None:
    order = await sim.broker.submit_bracket(bracket())
    sim.bar("100.00", "100.50", "99.50", "100.00")
    stream = sim.broker.stream_trade_updates()
    received = [await anext(stream) for _ in range(4)]
    assert [u.event for u in received] == [
        TradeUpdateEvent.NEW,
        TradeUpdateEvent.FILL,
        TradeUpdateEvent.NEW,
        TradeUpdateEvent.NEW,
    ]
    assert received[0].order.order_id == order.order_id
    assert sim.broker.drain_trade_updates() == []
