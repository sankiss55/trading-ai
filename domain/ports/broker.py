"""Broker port (sec. 8.5). The broker is the source of truth for execution (sec. 3.3)."""

from collections.abc import AsyncIterator
from typing import Protocol

from domain.models import (
    AccountState,
    BracketOrderRequest,
    BrokerOrder,
    Position,
    SimpleOrderRequest,
    TradeUpdate,
)

__all__ = ["IBroker"]


class IBroker(Protocol):
    """Trading API. Implementations raise only ``domain.errors`` exceptions."""

    async def get_account(self) -> AccountState:
        """Account state."""
        ...

    async def get_positions(self) -> list[Position]:
        """Open positions."""
        ...

    async def get_open_orders(self) -> list[BrokerOrder]:
        """Open orders, with their legs."""
        ...

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """Look up an order by ``client_order_id``; ``None`` if the broker does not know it."""
        ...

    async def submit_bracket(self, request: BracketOrderRequest) -> BrokerOrder:
        """Submit a bracket entry order."""
        ...

    async def submit_simple(self, request: SimpleOrderRequest) -> BrokerOrder:
        """Submit a simple order: system-initiated close or protective stop."""
        ...

    async def cancel_order(self, order_id: str) -> None:
        """Request cancellation; confirmation arrives via trade updates or a query."""
        ...

    def stream_trade_updates(self) -> AsyncIterator[TradeUpdate]:
        """Real-time order and fill events."""
        ...
