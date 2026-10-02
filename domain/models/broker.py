"""Broker-facing models: account, positions, order requests, orders, fills, updates.

Requests use whole-share ``int`` quantities (brackets do not allow fractions, sec. 23.2).
Broker-reported quantities (positions, orders, fills) are ``Decimal`` so that any
unexpected fractional or negative value is represented faithfully and detected by
reconciliation instead of being silently truncated (sec. 3.3, 24).
"""

from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, model_validator

from domain.models.base import (
    DomainModel,
    Money,
    NonEmptyStr,
    NonNegativeDecimal,
    PositiveDecimal,
    Price,
    Symbol,
    UtcDatetime,
)
from domain.models.enums import (
    BrokerOrderStatus,
    OrderClass,
    OrderSide,
    OrderType,
    TimeInForce,
    TradeUpdateEvent,
)

__all__ = [
    "AccountState",
    "BracketOrderRequest",
    "BrokerOrder",
    "Fill",
    "Position",
    "SimpleOrderRequest",
    "TradeUpdate",
]


class AccountState(DomainModel):
    """Broker account state (sec. 15.1 uses ``equity`` and ``last_equity``)."""

    equity: Money
    last_equity: Money
    buying_power: Money
    status: NonEmptyStr
    """Broker account status, normalized to upper case by the adapter (e.g. ``ACTIVE``)."""


class Position(DomainModel):
    """Open position as reported by the broker (source of truth, sec. 3.3)."""

    symbol: Symbol
    qty: Money
    avg_entry_price: Price
    market_value: Money


def _check_prices_for_type(
    order_type: OrderType, limit_price: object | None, stop_price: object | None
) -> None:
    """Validate which optional prices an order type requires or forbids."""
    needs_limit = order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)
    needs_stop = order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
    if needs_limit != (limit_price is not None):
        verb = "require" if needs_limit else "forbid"
        raise ValueError(f"{order_type} orders {verb} limit_price")
    if needs_stop != (stop_price is not None):
        verb = "require" if needs_stop else "forbid"
        raise ValueError(f"{order_type} orders {verb} stop_price")


class BracketOrderRequest(DomainModel):
    """Bracket entry order parameters (sec. 23.3). Long-only, regular session only."""

    symbol: Symbol
    qty: int = Field(ge=1)
    side: Literal[OrderSide.BUY] = OrderSide.BUY
    order_type: Literal[OrderType.MARKET, OrderType.LIMIT]
    limit_price: Price | None = None
    time_in_force: TimeInForce
    order_class: Literal[OrderClass.BRACKET] = OrderClass.BRACKET
    take_profit_limit_price: Price
    stop_loss_stop_price: Price
    client_order_id: NonEmptyStr
    extended_hours: Literal[False] = False

    @model_validator(mode="after")
    def _check_prices(self) -> Self:
        _check_prices_for_type(self.order_type, self.limit_price, None)
        return self


class SimpleOrderRequest(DomainModel):
    """Simple order: system-initiated close or protective stop (sec. 23.6, 23.7)."""

    symbol: Symbol
    qty: int = Field(ge=1)
    side: OrderSide
    order_type: OrderType
    limit_price: Price | None = None
    stop_price: Price | None = None
    time_in_force: TimeInForce
    client_order_id: NonEmptyStr

    @model_validator(mode="after")
    def _check_prices(self) -> Self:
        _check_prices_for_type(self.order_type, self.limit_price, self.stop_price)
        return self


class BrokerOrder(DomainModel):
    """Order as known by the broker, with its legs (bracket TP/SL)."""

    order_id: NonEmptyStr
    client_order_id: NonEmptyStr
    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    order_class: OrderClass = OrderClass.SIMPLE
    status: BrokerOrderStatus
    qty: NonNegativeDecimal
    filled_qty: NonNegativeDecimal = Decimal(0)
    filled_avg_price: Price | None = None
    limit_price: Price | None = None
    stop_price: Price | None = None
    legs: tuple["BrokerOrder", ...] = ()
    updated_at_utc: UtcDatetime | None = None


class Fill(DomainModel):
    """One execution reported by the broker. ``activity_id`` is unique (sec. 38.1)."""

    order_id: NonEmptyStr
    activity_id: NonEmptyStr
    symbol: Symbol
    side: OrderSide
    qty: PositiveDecimal
    price: Price
    timestamp_utc: UtcDatetime


class TradeUpdate(DomainModel):
    """Normalized broker trade-update event (type, order, fill)."""

    event: TradeUpdateEvent
    order: BrokerOrder
    timestamp_utc: UtcDatetime
    fill: Fill | None = None
    position_qty: Money | None = None

    @model_validator(mode="after")
    def _check_fill(self) -> Self:
        is_fill_event = self.event in (TradeUpdateEvent.FILL, TradeUpdateEvent.PARTIAL_FILL)
        if is_fill_event and self.fill is None:
            raise ValueError(f"{self.event} updates require a fill")
        if not is_fill_event and self.fill is not None:
            raise ValueError(f"{self.event} updates cannot carry a fill")
        if self.fill is not None and self.fill.order_id != self.order.order_id:
            raise ValueError("fill.order_id must match order.order_id")
        return self
