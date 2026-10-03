"""``orders``, ``order_events`` and ``fills`` tables (sec. 21.2, 22, 38.1)."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

from adapters.sqlite._codec import SqlValue, from_row, to_row
from adapters.sqlite.repositories._base import (
    FINAL_STATE_PARAMS,
    FINAL_STATE_PLACEHOLDERS,
    SqliteRepository,
    fetch_all,
    fetch_one,
    insert_sql,
)
from adapters.sqlite.semantics import ORDER_ID_MISMATCH, ORDER_NOT_FOUND
from domain.errors import NonRetryableError, StateCriticalError
from domain.models import BrokerOrderStatus, Fill, OrderEvent, OrderRecord, TradeState

__all__ = ["SqliteFillRepository", "SqliteOrderEventRepository", "SqliteOrderRepository"]

_ORDERS = "orders"
_ORDER_JSON = frozenset({"request"})
_EVENT_JSON = frozenset({"broker_event"})


def _order(row: Mapping[str, SqlValue]) -> OrderRecord:
    return from_row(OrderRecord, row, table=_ORDERS, json_fields=_ORDER_JSON)


class SqliteOrderRepository(SqliteRepository):
    """Implements ``IOrderRepository``. ``client_order_id`` is the primary key and the
    broker ``order_id`` is unique once known."""

    async def add(self, order: OrderRecord) -> None:
        """Insert a new order record.

        Raises:
            NonRetryableError: ``DUPLICATE_KEY`` (``client_order_id`` or ``order_id``
                already used), ``FOREIGN_KEY_VIOLATION`` (unknown ``trade_id``).
        """
        row = to_row(order, json_fields=_ORDER_JSON)
        row["updated_at_utc"] = self._session.stamp()
        sql = insert_sql(_ORDERS, row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("orders.add", job)

    async def get_by_client_id(self, client_order_id: str) -> OrderRecord | None:
        """Order record by client id, or ``None``."""

        def job(conn: sqlite3.Connection) -> OrderRecord | None:
            row = fetch_one(
                conn, "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
            )
            return None if row is None else _order(row)

        return await self._session.run("orders.get_by_client_id", job)

    async def list_non_final(self) -> list[OrderRecord]:
        """Orders whose state is not final, in insertion order (sec. 24.2)."""

        def job(conn: sqlite3.Connection) -> list[OrderRecord]:
            rows = fetch_all(
                conn,
                f"SELECT * FROM orders WHERE state NOT IN ({FINAL_STATE_PLACEHOLDERS}) "
                "ORDER BY rowid",
                FINAL_STATE_PARAMS,
            )
            return [_order(row) for row in rows]

        return await self._session.run("orders.list_non_final", job)

    async def update_status(
        self,
        client_order_id: str,
        state: TradeState,
        *,
        broker_status: BrokerOrderStatus | None = None,
        order_id: str | None = None,
    ) -> None:
        """Set ``state``; set ``broker_status``/``order_id`` only when given (``None``
        keeps the stored value).

        Raises:
            NonRetryableError: ``ORDER_NOT_FOUND``; ``DUPLICATE_KEY`` if ``order_id``
                belongs to another order.
            StateCriticalError: ``ORDER_ID_MISMATCH`` if the order already has a
                different broker ``order_id`` (state mismatch, sec. 29).
        """
        params: dict[str, SqlValue] = {
            "client_order_id": client_order_id,
            "state": state.value,
            "broker_status": None if broker_status is None else broker_status.value,
            "order_id": order_id,
            "updated_at_utc": self._session.stamp(),
        }

        def job(conn: sqlite3.Connection) -> None:
            stored = fetch_one(
                conn, "SELECT order_id FROM orders WHERE client_order_id = ?", (client_order_id,)
            )
            if stored is None:
                raise NonRetryableError(
                    f"order {client_order_id!r} not found", code=ORDER_NOT_FOUND
                )
            known = stored["order_id"]
            if order_id is not None and known is not None and known != order_id:
                raise StateCriticalError(
                    f"order {client_order_id!r} already has broker order id {known!r}, "
                    f"got {order_id!r}",
                    code=ORDER_ID_MISMATCH,
                )
            conn.execute(
                "UPDATE orders SET state = :state, "
                "broker_status = COALESCE(:broker_status, broker_status), "
                "order_id = COALESCE(:order_id, order_id), "
                "updated_at_utc = :updated_at_utc "
                "WHERE client_order_id = :client_order_id",
                params,
            )

        await self._session.run("orders.update_status", job)


class SqliteOrderEventRepository(SqliteRepository):
    """Implements ``IOrderEventRepository`` (append-only, enforced by triggers)."""

    async def append(self, event: OrderEvent) -> None:
        """Append one state transition."""
        row = to_row(event, json_fields=_EVENT_JSON)
        sql = insert_sql("order_events", row)

        def job(conn: sqlite3.Connection) -> None:
            conn.execute(sql, row)

        await self._session.run("order_events.append", job)


class SqliteFillRepository(SqliteRepository):
    """Implements ``IFillRepository``: idempotent by broker ``activity_id``."""

    async def add_if_new(self, fill: Fill) -> bool:
        """Insert the fill unless its ``activity_id`` is known. ``True`` if inserted."""
        row = to_row(fill)
        sql = insert_sql("fills", row, suffix=" ON CONFLICT (activity_id) DO NOTHING")

        def job(conn: sqlite3.Connection) -> bool:
            return conn.execute(sql, row).rowcount == 1

        return await self._session.run("fills.add_if_new", job)
