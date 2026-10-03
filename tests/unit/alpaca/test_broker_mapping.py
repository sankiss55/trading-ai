"""AlpacaBroker conversions: SDK orders (with legs), account and positions -> domain."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from alpaca.trading.enums import OrderStatus as SdkOrderStatus
from alpaca.trading.models import Order, TradeAccount
from alpaca.trading.models import Position as SdkPosition

from adapters.alpaca.broker import convert_account, convert_order, convert_position
from domain.errors import NonRetryableError
from domain.models import BrokerOrderStatus, OrderClass, OrderSide, OrderType
from tests.unit.alpaca.fake_trading_api import account_payload, order_payload, position_payload

SAME_NAME_STATUSES = [s for s in SdkOrderStatus if s is not SdkOrderStatus.PENDING_REVIEW]


@pytest.mark.parametrize("status", SAME_NAME_STATUSES, ids=lambda s: s.value)
def test_every_sdk_status_maps_to_the_same_named_domain_status(status: SdkOrderStatus) -> None:
    order = convert_order(Order(**order_payload(status=status.value)))
    assert order.status is BrokerOrderStatus[status.name]


def test_pending_review_maps_to_pending_new() -> None:
    order = convert_order(Order(**order_payload(status="pending_review")))
    assert order.status is BrokerOrderStatus.PENDING_NEW


@pytest.mark.parametrize(
    ("sdk_type", "prices", "expected"),
    [
        ("market", {}, OrderType.MARKET),
        ("limit", {"limit_price": "10.5"}, OrderType.LIMIT),
        ("stop", {"stop_price": "9.5"}, OrderType.STOP),
        ("stop_limit", {"limit_price": "9.4", "stop_price": "9.5"}, OrderType.STOP_LIMIT),
    ],
)
def test_order_types_are_mapped(sdk_type: str, prices: dict[str, str], expected: OrderType) -> None:
    order = convert_order(Order(**order_payload(type=sdk_type, side="sell", **prices)))
    assert order.order_type is expected
    assert order.side is OrderSide.SELL
    assert order.limit_price == (
        Decimal(prices["limit_price"]) if "limit_price" in prices else None
    )
    assert order.stop_price == (Decimal(prices["stop_price"]) if "stop_price" in prices else None)


def test_deprecated_order_type_field_is_used_when_type_is_missing() -> None:
    payload = order_payload(type=None, order_type="limit", limit_price="5")
    assert convert_order(Order(**payload)).order_type is OrderType.LIMIT


@pytest.mark.parametrize(
    ("sdk_class", "expected"),
    [
        ("simple", OrderClass.SIMPLE),
        ("", OrderClass.SIMPLE),  # the SDK treats "" as simple
        ("bracket", OrderClass.BRACKET),
        ("oco", OrderClass.OCO),
        ("oto", OrderClass.OTO),
    ],
)
def test_order_classes_are_mapped(sdk_class: str, expected: OrderClass) -> None:
    assert convert_order(Order(**order_payload(order_class=sdk_class))).order_class is expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "trailing_stop", "trail_percent": "1"},
        {"order_class": "mleg"},
        {"qty": None, "notional": "100"},
    ],
    ids=["trailing-stop", "mleg", "notional"],
)
def test_orders_the_domain_cannot_represent_are_refused(overrides: dict[str, str | None]) -> None:
    with pytest.raises(NonRetryableError) as info:
        convert_order(Order(**order_payload(**overrides)))
    assert info.value.code == "UNSUPPORTED_BROKER_ORDER"


def test_bracket_parent_and_legs_are_converted_recursively() -> None:
    legs = [
        order_payload(
            order_class="bracket",
            type="limit",
            side="sell",
            qty="3",
            limit_price="190.1",
            status="new",
        ),
        order_payload(
            order_class="bracket",
            type="stop",
            side="sell",
            qty="3",
            stop_price="185.07",
            status="new",
        ),
    ]
    parent = order_payload(
        order_class="bracket",
        type="limit",
        qty="3",
        filled_qty="3",
        filled_avg_price="187.3",
        limit_price="187.35",
        status="filled",
        legs=legs,
        updated_at="2026-10-01T09:31:00.5-04:00",
    )
    order = convert_order(Order(**parent))
    assert order.order_class is OrderClass.BRACKET
    assert (order.status, order.qty, order.filled_qty) == (
        BrokerOrderStatus.FILLED,
        Decimal(3),
        Decimal(3),
    )
    assert order.filled_avg_price == Decimal("187.3")
    assert order.updated_at_utc == datetime(2026, 10, 1, 13, 31, 0, 500000, tzinfo=UTC)
    take_profit, stop_loss = order.legs
    assert (take_profit.order_type, take_profit.side, take_profit.limit_price) == (
        OrderType.LIMIT,
        OrderSide.SELL,
        Decimal("190.1"),
    )
    assert (stop_loss.order_type, stop_loss.stop_price, stop_loss.qty) == (
        OrderType.STOP,
        Decimal("185.07"),
        Decimal(3),
    )
    assert take_profit.status is stop_loss.status is BrokerOrderStatus.NEW
    assert take_profit.order_id == legs[0]["id"]
    assert take_profit.client_order_id == legs[0]["client_order_id"]


def test_numbers_become_decimals_and_zero_fill_price_means_none() -> None:
    order = convert_order(Order(**order_payload(qty=2.0, filled_qty=0.5, filled_avg_price="0")))
    assert order.qty == Decimal("2.0")
    assert order.filled_qty == Decimal("0.5")
    assert order.filled_avg_price is None


@pytest.mark.parametrize(
    "overrides",
    [{"symbol": None}, {"side": None}, {"limit_price": "-1"}, {"filled_qty": "abc"}],
    ids=["no-symbol", "no-side", "negative-price", "garbage-number"],
)
def test_invalid_order_data_is_rejected(overrides: dict[str, str | None]) -> None:
    with pytest.raises(NonRetryableError) as info:
        convert_order(Order(**order_payload(**overrides)))
    assert info.value.code == "INVALID_ORDER_DATA"


def test_account_is_converted_to_decimals_with_upper_case_status() -> None:
    account = convert_account(TradeAccount(**account_payload()))
    assert account.equity == Decimal("100000.5")
    assert account.last_equity == Decimal("99950.25")
    assert account.buying_power == Decimal(200000)
    assert account.status == "ACTIVE"


def test_account_without_equity_is_invalid() -> None:
    with pytest.raises(NonRetryableError) as info:
        convert_account(TradeAccount(**account_payload(equity=None)))
    assert info.value.code == "INVALID_ACCOUNT_DATA"


def test_positions_are_converted_and_shorts_are_negative() -> None:
    long_ = convert_position(SdkPosition(**position_payload("SPY", "3", avg_entry_price="500.25")))
    assert (long_.symbol, long_.qty, long_.avg_entry_price) == (
        "SPY",
        Decimal(3),
        Decimal("500.25"),
    )
    assert long_.market_value == Decimal("1500.75")
    short = convert_position(SdkPosition(**position_payload("QQQ", "2", side="short")))
    assert short.qty == Decimal(-2)


def test_position_without_market_value_is_invalid() -> None:
    with pytest.raises(NonRetryableError) as info:
        convert_position(SdkPosition(**position_payload("SPY", "1", market_value=None)))
    assert info.value.code == "INVALID_POSITION_DATA"
