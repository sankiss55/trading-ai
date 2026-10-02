"""Deterministic simulated broker for backtests (sec. 8.6, 22, 23, 45.2).

``SimulatedBroker`` implements ``IBroker``. Fills are driven explicitly by
``process_bar(bar)``; nothing happens between bars. All state transitions are
synchronous inside each call, so they are atomic for asyncio callers.

Fill rules (all prices ``Decimal``; slippage is ``slippage_bps / 10000`` applied
adversely: buys are priced up and rounded up, sells priced down and rounded down to
``price_quantum``):

Entries (bracket parent, BUY), eligible on the first processed bar whose
``bar_start_utc >= submission time`` (no lookahead, sec. 45.2.8):

* ``MARKET``: ``open * (1 + slip)``, rounded up.
* ``LIMIT``: if ``open < limit`` fill at ``min(open * (1 + slip), limit)``; else if the bar
  trades through the limit (``low < limit``) fill at ``limit``; else keep waiting.

When the entry fills, its TP (sell LIMIT) and SL (sell STOP) legs go from ``HELD`` to
``NEW`` and are evaluated on the same bar (after the open).

Exits (SELL), for legs and simple orders:

* ``STOP``: if ``open <= stop`` (gap through the stop) fill at ``open * (1 - slip)``;
  else if ``low <= stop`` fill at ``stop * (1 - slip)``.
* ``LIMIT``: if ``open > limit`` (gap above) fill at ``max(open * (1 - slip), limit)``;
  else if ``high > limit`` fill at ``limit``.
* ``MARKET``: ``open * (1 - slip)``.
* Bracket legs are OCO: if the stop is touchable in the bar, the STOP fills and the TP
  is canceled, even if the TP was also touchable (stop first, sec. 45.2.6). When a leg
  fills, the other leg is canceled.

Fill timestamps: fills at the open use ``bar_start_utc``; intrabar fills use
``bar_end_utc`` (the earliest instant at which a bar-level simulation knows them).

Other semantics:

* Idempotency (sec. 22): a repeated ``client_order_id`` returns the existing order and
  creates nothing (``duplicate_client_order_id="return_existing"``, default) or raises
  ``NonRetryableError`` code ``DUPLICATE_CLIENT_ORDER_ID`` (``"reject"``). Which one
  matches Alpaca is VERIFICAR (sec. 22.4); both are supported so the simulator can
  mirror the verified behavior.
* Active SELL orders, including active bracket legs, reserve position quantity: a simple
  SELL for more than the unreserved quantity raises ``INSUFFICIENT_QTY_AVAILABLE``
  (sec. 23.6, 49.2). Long only: a SELL without position raises ``NO_POSITION``; simple
  BUY orders raise ``UNSUPPORTED_ORDER`` (entries are always brackets, sec. 23.1).
* Price increments (VERIFICAR): prices >= 1 allow 2 decimals, below 1 allow 4;
  otherwise ``INVALID_PRICE_INCREMENT``. Bracket prices must satisfy ``stop < take
  profit`` (and ``stop < limit < take profit`` for LIMIT): ``INVALID_BRACKET_PRICES``.
  Invalid requests raise and create no order (no events).
* ``close_session()`` expires every open ``DAY`` order, including active legs of DAY
  brackets, which leaves the position unprotected (sec. 23.2, 49.2), and rolls
  ``last_equity``.
* Cash account: ``buying_power = cash - cash reserved by open entries``. Entries whose
  estimated cost (limit price, or last close, plus slippage and commission) exceeds
  buying power raise ``INSUFFICIENT_BUYING_POWER``. A market entry that gaps above the
  estimate still fills (cash may go negative); the shortfall is visible in the account.
* ``equity = cash + sum(qty * mark)`` with ``mark`` = last processed close of the symbol
  (average entry price if none yet).
* Whole shares only, no partial fills (partial fills belong to ``FakeBroker``).
* Every update is returned by the call that produced it and also queued for
  ``stream_trade_updates()`` (single consumer). Callers that do not consume the stream
  should call ``drain_trade_updates()`` to keep the queue bounded.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Literal

from domain.errors import NonRetryableError, StateCriticalError
from domain.models import (
    AccountState,
    Bar,
    BarStatus,
    BracketOrderRequest,
    BrokerOrder,
    BrokerOrderStatus,
    Fill,
    OrderClass,
    OrderRole,
    OrderSide,
    OrderType,
    Position,
    SimpleOrderRequest,
    TimeInForce,
    TradeUpdate,
    TradeUpdateEvent,
)
from domain.ports import IClock

__all__ = ["DuplicatePolicy", "SimulatedBroker"]

DuplicatePolicy = Literal["return_existing", "reject"]

_BPS = Decimal(10_000)
_ZERO = Decimal(0)
_ACTIVE = frozenset(
    {
        BrokerOrderStatus.NEW,
        BrokerOrderStatus.ACCEPTED,
        BrokerOrderStatus.PENDING_NEW,
        BrokerOrderStatus.PARTIALLY_FILLED,
    }
)
"""Statuses in which an order can fill."""
_OPEN = _ACTIVE | {BrokerOrderStatus.HELD}
"""Non-final statuses (held legs wait for their parent)."""


@dataclass
class _SimOrder:
    """Mutable internal order record; exposed only as ``BrokerOrder`` snapshots."""

    seq: int
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    order_class: OrderClass
    role: OrderRole
    qty: int
    time_in_force: TimeInForce
    limit_price: Decimal | None
    stop_price: Decimal | None
    status: BrokerOrderStatus
    eligible_from: datetime
    updated_at: datetime
    parent_id: str | None = None
    leg_ids: tuple[str, ...] = ()
    filled_qty: int = 0
    filled_avg_price: Decimal | None = None
    reserved_cash: Decimal = field(default=_ZERO)


@dataclass
class _SimPosition:
    qty: int
    avg_price: Decimal


def _require_decimal(name: str, value: object, *, minimum: Decimal = _ZERO) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise TypeError(f"{name} must be a finite Decimal")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _has_valid_increment(price: Decimal) -> bool:
    quantum = Decimal("0.01") if price >= 1 else Decimal("0.0001")
    return price == price.quantize(quantum)


class SimulatedBroker:
    """``IBroker`` for backtests with bracket simulation (stop first) and slippage.

    Args:
        clock: Time source for submissions, cancellations and session closes.
        starting_cash: Initial cash (``Decimal``).
        slippage_bps: Adverse slippage in basis points applied to market and stop fills
            and capped by the limit for limit fills. Required: its value is an
            OWNER_DECISION (``risk.slippage_buffer_bps``, sec. 45.2.5).
        commission_per_fill: Flat commission charged on every fill.
        price_quantum: Rounding step of slipped fill prices (rounded adversely).
        duplicate_client_order_id: Behavior on a repeated ``client_order_id``.
        id_prefix: Prefix of the deterministic order ids (``{prefix}-000001``).
    """

    def __init__(
        self,
        *,
        clock: IClock,
        starting_cash: Decimal,
        slippage_bps: Decimal,
        commission_per_fill: Decimal = _ZERO,
        price_quantum: Decimal = Decimal("0.0001"),
        duplicate_client_order_id: DuplicatePolicy = "return_existing",
        id_prefix: str = "sim",
    ) -> None:
        self._clock = clock
        self._cash = _require_decimal("starting_cash", starting_cash)
        self._last_equity = self._cash
        self._slip = _require_decimal("slippage_bps", slippage_bps) / _BPS
        self._commission = _require_decimal("commission_per_fill", commission_per_fill)
        self._quantum = _require_decimal("price_quantum", price_quantum)
        if self._quantum == 0:
            raise ValueError("price_quantum must be > 0")
        if duplicate_client_order_id not in ("return_existing", "reject"):
            raise ValueError(f"unknown duplicate policy {duplicate_client_order_id!r}")
        self._duplicate_policy: DuplicatePolicy = duplicate_client_order_id
        self._id_prefix = id_prefix
        self._seq = 0
        self._fill_seq = 0
        self._orders: dict[str, _SimOrder] = {}
        self._live: dict[str, _SimOrder] = {}
        """Orders of every group (simple order, or bracket parent + legs) with at least
        one non-final member, in creation (``seq``) order. A final status never becomes
        open again, so a group leaves this index for good once all its members are final.
        """
        self._by_client_id: dict[str, str] = {}
        self._positions: dict[str, _SimPosition] = {}
        self._last_close: dict[str, Decimal] = {}
        self._last_bar_start: dict[str, datetime] = {}
        self._fills: list[Fill] = []
        self._queue: asyncio.Queue[TradeUpdate] = asyncio.Queue()

    # ------------------------------------------------------------------ IBroker queries

    async def get_account(self) -> AccountState:
        """Cash account state; equity is marked to the last processed close."""
        return AccountState(
            equity=self.equity(),
            last_equity=self._last_equity,
            buying_power=self._buying_power(),
            status="ACTIVE",
        )

    async def get_positions(self) -> list[Position]:
        """Open positions sorted by symbol."""
        return [
            Position(
                symbol=symbol,
                qty=Decimal(pos.qty),
                avg_entry_price=pos.avg_price,
                market_value=Decimal(pos.qty) * self._mark(symbol),
            )
            for symbol, pos in sorted(self._positions.items())
        ]

    async def get_open_orders(self) -> list[BrokerOrder]:
        """Top-level orders that are open or still have an open leg, with nested legs.

        A FILLED bracket parent whose TP/SL legs are active is included (VERIFICAR
        against Alpaca ``nested=true`` listing).
        """
        result: list[BrokerOrder] = []
        for order in self._ordered():
            if order.parent_id is not None:
                continue
            legs_open = any(self._orders[leg].status in _OPEN for leg in order.leg_ids)
            if order.status in _OPEN or legs_open:
                result.append(self._snapshot(order))
        return result

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """Order (or leg) with this ``client_order_id``; ``None`` if unknown."""
        order_id = self._by_client_id.get(client_order_id)
        return None if order_id is None else self._snapshot(self._orders[order_id])

    # ------------------------------------------------------------------ IBroker commands

    async def submit_bracket(self, request: BracketOrderRequest) -> BrokerOrder:
        """Accept a bracket entry: parent ``NEW``, TP/SL legs ``HELD`` until it fills.

        Raises:
            NonRetryableError: ``DUPLICATE_CLIENT_ORDER_ID`` (reject policy),
                ``INVALID_PRICE_INCREMENT``, ``INVALID_BRACKET_PRICES``,
                ``INSUFFICIENT_BUYING_POWER``.
        """
        existing = self._existing(request.client_order_id)
        if existing is not None:
            return existing
        prices = [request.take_profit_limit_price, request.stop_loss_stop_price]
        if request.limit_price is not None:
            prices.append(request.limit_price)
        self._check_increments(prices)
        stop, take_profit = request.stop_loss_stop_price, request.take_profit_limit_price
        if not stop < take_profit or (
            request.limit_price is not None and not stop < request.limit_price < take_profit
        ):
            raise NonRetryableError(
                "bracket requires stop_loss < (limit <) take_profit",
                code="INVALID_BRACKET_PRICES",
            )
        reserved = self._estimate_entry_cost(request)
        if reserved > self._buying_power():
            raise NonRetryableError(
                f"estimated cost {reserved} exceeds buying power {self._buying_power()}",
                code="INSUFFICIENT_BUYING_POWER",
            )
        now = self._clock.now_utc()
        parent = self._new_order(
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=OrderSide.BUY,
            order_type=request.order_type,
            order_class=OrderClass.BRACKET,
            role=OrderRole.ENTRY,
            qty=request.qty,
            time_in_force=request.time_in_force,
            limit_price=request.limit_price,
            stop_price=None,
            status=BrokerOrderStatus.NEW,
            now=now,
        )
        parent.reserved_cash = reserved
        legs = (
            (OrderRole.TAKE_PROFIT, OrderType.LIMIT, take_profit, None),
            (OrderRole.STOP_LOSS, OrderType.STOP, None, stop),
        )
        leg_ids: list[str] = []
        for role, order_type, limit_price, stop_price in legs:
            leg = self._new_order(
                client_order_id=None,
                symbol=request.symbol,
                side=OrderSide.SELL,
                order_type=order_type,
                order_class=OrderClass.BRACKET,
                role=role,
                qty=request.qty,
                time_in_force=request.time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                status=BrokerOrderStatus.HELD,
                now=now,
            )
            leg.parent_id = parent.order_id
            leg_ids.append(leg.order_id)
        parent.leg_ids = tuple(leg_ids)
        self._emit(TradeUpdateEvent.NEW, parent, now)
        return self._snapshot(parent)

    async def submit_simple(self, request: SimpleOrderRequest) -> BrokerOrder:
        """Accept a simple SELL exit (``MARKET``, ``LIMIT`` or ``STOP``).

        Raises:
            NonRetryableError: ``DUPLICATE_CLIENT_ORDER_ID`` (reject policy),
                ``UNSUPPORTED_ORDER``, ``INVALID_PRICE_INCREMENT``, ``NO_POSITION``,
                ``INSUFFICIENT_QTY_AVAILABLE``.
        """
        existing = self._existing(request.client_order_id)
        if existing is not None:
            return existing
        if request.side is not OrderSide.SELL:
            raise NonRetryableError(
                "simple BUY orders are not supported: entries are brackets (sec. 23.1)",
                code="UNSUPPORTED_ORDER",
            )
        if request.order_type is OrderType.STOP_LIMIT:
            raise NonRetryableError("STOP_LIMIT orders are not simulated", code="UNSUPPORTED_ORDER")
        self._check_increments(
            [p for p in (request.limit_price, request.stop_price) if p is not None]
        )
        position = self._positions.get(request.symbol)
        if position is None:
            raise NonRetryableError(
                f"no position in {request.symbol}: long only", code="NO_POSITION"
            )
        available = position.qty - self._reserved_sell_qty(request.symbol)
        if request.qty > available:
            raise NonRetryableError(
                f"insufficient qty available for {request.symbol}: requested "
                f"{request.qty}, available {available} (held by open orders)",
                code="INSUFFICIENT_QTY_AVAILABLE",
            )
        now = self._clock.now_utc()
        order = self._new_order(
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=OrderSide.SELL,
            order_type=request.order_type,
            order_class=OrderClass.SIMPLE,
            role=OrderRole.EXIT,
            qty=request.qty,
            time_in_force=request.time_in_force,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            status=BrokerOrderStatus.NEW,
            now=now,
        )
        self._emit(TradeUpdateEvent.NEW, order, now)
        return self._snapshot(order)

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an open order immediately and emit ``CANCELED`` updates.

        Canceling an unfilled bracket parent also cancels its held legs. Canceling one
        active leg cancels only that leg (sec. 23.6 cancels each leg explicitly;
        VERIFICAR whether Alpaca cancels the sibling). Canceling an already canceled
        order is a no-op.

        Raises:
            NonRetryableError: ``ORDER_NOT_FOUND`` or ``ORDER_NOT_CANCELABLE`` (the order
                is filled, expired or rejected).
        """
        order = self._orders.get(order_id)
        if order is None:
            raise NonRetryableError(f"unknown order {order_id}", code="ORDER_NOT_FOUND")
        if order.status is BrokerOrderStatus.CANCELED:
            return
        if order.status not in _OPEN:
            raise NonRetryableError(
                f"order {order_id} is {order.status} and cannot be canceled",
                code="ORDER_NOT_CANCELABLE",
            )
        now = self._clock.now_utc()
        self._finish(order, BrokerOrderStatus.CANCELED, TradeUpdateEvent.CANCELED, now)
        for leg_id in order.leg_ids:
            leg = self._orders[leg_id]
            if leg.status in _OPEN:
                self._finish(leg, BrokerOrderStatus.CANCELED, TradeUpdateEvent.CANCELED, now)

    async def stream_trade_updates(self) -> AsyncIterator[TradeUpdate]:
        """Endless stream of every update, in emission order (single consumer)."""
        while True:
            yield await self._queue.get()

    # ------------------------------------------------------------------ simulation driver

    def process_bar(self, bar: Bar) -> list[TradeUpdate]:
        """Simulate the market of ``bar.symbol`` during ``bar`` and return the updates.

        Order: pending entries first (then their legs activate), then exits grouped per
        bracket/simple order in submission order, stop before take profit. Finally the
        bar close becomes the symbol's mark. ``EMPTY`` bars only advance the bar cursor.

        Raises:
            NonRetryableError: ``BAR_OUT_OF_ORDER`` if ``bar.bar_start_utc`` is not after
                the previous bar processed for the symbol.
        """
        previous = self._last_bar_start.get(bar.symbol)
        if previous is not None and bar.bar_start_utc <= previous:
            raise NonRetryableError(
                f"{bar.symbol} bar {bar.bar_start_utc.isoformat()} is not after "
                f"{previous.isoformat()}",
                code="BAR_OUT_OF_ORDER",
            )
        self._last_bar_start[bar.symbol] = bar.bar_start_utc
        if (
            bar.status is BarStatus.EMPTY
            or bar.open is None
            or bar.high is None
            or bar.low is None
            or bar.close is None
        ):
            return []
        updates: list[TradeUpdate] = []
        self._process_entries(bar, bar.open, bar.low, updates)
        self._process_exits(bar, bar.open, bar.high, bar.low, updates)
        self._last_close[bar.symbol] = bar.close
        return updates

    def close_session(self) -> list[TradeUpdate]:
        """End of the regular session: expire every open ``DAY`` order (entries, legs and
        simple orders) and roll ``last_equity`` to the current equity.

        Expired legs leave their position unprotected, as at the broker (sec. 23.2).
        """
        now = self._clock.now_utc()
        updates: list[TradeUpdate] = []
        for order in self._ordered():
            if order.time_in_force is TimeInForce.DAY and order.status in _OPEN:
                updates.append(
                    self._finish(order, BrokerOrderStatus.EXPIRED, TradeUpdateEvent.EXPIRED, now)
                )
        self._last_equity = self.equity()
        return updates

    def drain_trade_updates(self) -> list[TradeUpdate]:
        """Remove and return every queued update without waiting."""
        drained: list[TradeUpdate] = []
        while not self._queue.empty():
            drained.append(self._queue.get_nowait())
        return drained

    def equity(self) -> Decimal:
        """Cash plus positions marked to the last processed close."""
        return self._cash + sum(
            (Decimal(pos.qty) * self._mark(symbol) for symbol, pos in self._positions.items()),
            _ZERO,
        )

    @property
    def cash(self) -> Decimal:
        """Current cash balance."""
        return self._cash

    @property
    def fills(self) -> tuple[Fill, ...]:
        """Every fill so far, in execution order."""
        return tuple(self._fills)

    # ------------------------------------------------------------------ fill engine

    def _process_entries(
        self, bar: Bar, open_: Decimal, low: Decimal, updates: list[TradeUpdate]
    ) -> None:
        for order in self._ordered():
            if (
                order.symbol != bar.symbol
                or order.role is not OrderRole.ENTRY
                or order.status not in _ACTIVE
                or order.eligible_from > bar.bar_start_utc
            ):
                continue
            outcome = self._entry_price(order, open_, low)
            if outcome is None:
                continue
            price, at_open = outcome
            when = bar.bar_start_utc if at_open else bar.bar_end_utc
            order.reserved_cash = _ZERO
            for leg_id in order.leg_ids:
                leg = self._orders[leg_id]
                if leg.status is BrokerOrderStatus.HELD:
                    leg.status = BrokerOrderStatus.NEW
                    leg.eligible_from = bar.bar_start_utc
                    leg.updated_at = when
            updates.append(self._fill(order, price, when))
            for leg_id in order.leg_ids:
                leg = self._orders[leg_id]
                if leg.status is BrokerOrderStatus.NEW:
                    updates.append(self._emit(TradeUpdateEvent.NEW, leg, when))

    def _process_exits(
        self,
        bar: Bar,
        open_: Decimal,
        high: Decimal,
        low: Decimal,
        updates: list[TradeUpdate],
    ) -> None:
        for order in self._ordered():
            if order.symbol != bar.symbol or order.role is OrderRole.ENTRY:
                continue
            if order.role is OrderRole.EXIT:
                group = [order]
            elif order.role is OrderRole.STOP_LOSS:
                # Bracket legs are handled together when the SL (created last) is reached:
                # stop first, then take profit.
                assert order.parent_id is not None
                parent = self._orders[order.parent_id]
                group = sorted(
                    (self._orders[leg] for leg in parent.leg_ids),
                    key=lambda leg: 0 if leg.role is OrderRole.STOP_LOSS else 1,
                )
            else:
                continue
            for candidate in group:
                if candidate.status not in _ACTIVE or candidate.eligible_from > bar.bar_start_utc:
                    continue
                outcome = self._exit_price(candidate, open_, high, low)
                if outcome is None:
                    continue
                price, at_open = outcome
                when = bar.bar_start_utc if at_open else bar.bar_end_utc
                updates.append(self._fill(candidate, price, when))
                for sibling in group:
                    if sibling is not candidate and sibling.status in _OPEN:
                        updates.append(
                            self._finish(
                                sibling,
                                BrokerOrderStatus.CANCELED,
                                TradeUpdateEvent.CANCELED,
                                when,
                            )
                        )
                break

    def _entry_price(
        self, order: _SimOrder, open_: Decimal, low: Decimal
    ) -> tuple[Decimal, bool] | None:
        if order.order_type is OrderType.MARKET:
            return self._buy_price(open_), True
        limit = order.limit_price
        assert limit is not None
        if open_ < limit:
            return min(self._buy_price(open_), limit), True
        if low < limit:
            return limit, False
        return None

    def _exit_price(
        self, order: _SimOrder, open_: Decimal, high: Decimal, low: Decimal
    ) -> tuple[Decimal, bool] | None:
        if order.order_type is OrderType.MARKET:
            return self._sell_price(open_), True
        if order.order_type is OrderType.STOP:
            stop = order.stop_price
            assert stop is not None
            if open_ <= stop:
                return self._sell_price(open_), True
            if low <= stop:
                return self._sell_price(stop), False
            return None
        limit = order.limit_price
        assert limit is not None
        if open_ > limit:
            return max(self._sell_price(open_), limit), True
        if high > limit:
            return limit, False
        return None

    def _buy_price(self, raw: Decimal) -> Decimal:
        return (raw * (1 + self._slip)).quantize(self._quantum, rounding=ROUND_CEILING)

    def _sell_price(self, raw: Decimal) -> Decimal:
        return (raw * (1 - self._slip)).quantize(self._quantum, rounding=ROUND_FLOOR)

    def _fill(self, order: _SimOrder, price: Decimal, when: datetime) -> TradeUpdate:
        qty = order.qty - order.filled_qty
        position = self._positions.get(order.symbol)
        if order.side is OrderSide.BUY:
            self._cash -= Decimal(qty) * price + self._commission
            if position is None:
                self._positions[order.symbol] = _SimPosition(qty=qty, avg_price=price)
            else:
                total = position.qty + qty
                position.avg_price = (
                    Decimal(position.qty) * position.avg_price + Decimal(qty) * price
                ) / Decimal(total)
                position.qty = total
        else:
            if position is None or position.qty < qty:
                raise StateCriticalError(
                    f"sell fill of {qty} {order.symbol} exceeds the simulated position",
                    code="STATE_MISMATCH",
                )
            self._cash += Decimal(qty) * price - self._commission
            position.qty -= qty
            if position.qty == 0:
                del self._positions[order.symbol]
        order.filled_avg_price = price
        order.filled_qty = order.qty
        order.status = BrokerOrderStatus.FILLED
        order.updated_at = when
        self._fill_seq += 1
        fill = Fill(
            order_id=order.order_id,
            activity_id=f"{self._id_prefix}-fill-{self._fill_seq:06d}",
            symbol=order.symbol,
            side=order.side,
            qty=Decimal(qty),
            price=price,
            timestamp_utc=when,
        )
        self._fills.append(fill)
        return self._emit(TradeUpdateEvent.FILL, order, when, fill=fill)

    # ------------------------------------------------------------------ helpers

    def _existing(self, client_order_id: str) -> BrokerOrder | None:
        order_id = self._by_client_id.get(client_order_id)
        if order_id is None:
            return None
        if self._duplicate_policy == "reject":
            raise NonRetryableError(
                f"client_order_id {client_order_id!r} already used",
                code="DUPLICATE_CLIENT_ORDER_ID",
            )
        return self._snapshot(self._orders[order_id])

    def _new_order(
        self,
        *,
        client_order_id: str | None,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_class: OrderClass,
        role: OrderRole,
        qty: int,
        time_in_force: TimeInForce,
        limit_price: Decimal | None,
        stop_price: Decimal | None,
        status: BrokerOrderStatus,
        now: datetime,
    ) -> _SimOrder:
        self._seq += 1
        order_id = f"{self._id_prefix}-{self._seq:06d}"
        order = _SimOrder(
            seq=self._seq,
            order_id=order_id,
            client_order_id=client_order_id or order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_class=order_class,
            role=role,
            qty=qty,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            status=status,
            eligible_from=now,
            updated_at=now,
        )
        self._orders[order_id] = order
        self._live[order_id] = order
        self._by_client_id[order.client_order_id] = order_id
        return order

    def _finish(
        self,
        order: _SimOrder,
        status: BrokerOrderStatus,
        event: TradeUpdateEvent,
        when: datetime,
    ) -> TradeUpdate:
        order.status = status
        order.updated_at = when
        order.reserved_cash = _ZERO
        return self._emit(event, order, when)

    def _emit(
        self,
        event: TradeUpdateEvent,
        order: _SimOrder,
        when: datetime,
        *,
        fill: Fill | None = None,
    ) -> TradeUpdate:
        position_qty: Decimal | None = None
        if fill is not None:
            position = self._positions.get(order.symbol)
            position_qty = Decimal(position.qty) if position is not None else _ZERO
        update = TradeUpdate(
            event=event,
            order=self._snapshot(order),
            timestamp_utc=when,
            fill=fill,
            position_qty=position_qty,
        )
        self._queue.put_nowait(update)
        return update

    def _snapshot(self, order: _SimOrder) -> BrokerOrder:
        return BrokerOrder(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            order_class=order.order_class,
            status=order.status,
            qty=Decimal(order.qty),
            filled_qty=Decimal(order.filled_qty),
            filled_avg_price=order.filled_avg_price,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            legs=tuple(self._snapshot(self._orders[leg]) for leg in order.leg_ids),
            updated_at_utc=order.updated_at,
        )

    def _ordered(self) -> list[_SimOrder]:
        """Orders that may still act, in ``seq`` order.

        Equivalent to every order sorted by ``seq`` for all callers: they skip orders
        whose whole group is final (a bracket's legs are evaluated through its stop
        leg, so a final stop leg stays listed while a sibling is open). Orders are
        created in ``seq`` order, so the insertion order of ``_live`` is ``seq`` order.
        """
        finished = [
            order_id
            for order_id, order in self._live.items()
            if order.status not in _OPEN and not self._group_open(order)
        ]
        for order_id in finished:
            del self._live[order_id]
        return list(self._live.values())

    def _group_open(self, order: _SimOrder) -> bool:
        root = order if order.parent_id is None else self._orders[order.parent_id]
        return root.status in _OPEN or any(
            self._orders[leg].status in _OPEN for leg in root.leg_ids
        )

    def _mark(self, symbol: str) -> Decimal:
        close = self._last_close.get(symbol)
        if close is not None:
            return close
        position = self._positions.get(symbol)
        return position.avg_price if position is not None else _ZERO

    def _buying_power(self) -> Decimal:
        reserved = sum((o.reserved_cash for o in self._ordered() if o.status in _OPEN), _ZERO)
        return self._cash - reserved

    def _reserved_sell_qty(self, symbol: str) -> int:
        """Quantity held by active SELL orders; OCO legs of one bracket hold it once."""
        simple = 0
        per_bracket: dict[str, int] = {}
        for o in self._ordered():
            if o.symbol != symbol or o.side is not OrderSide.SELL or o.status not in _ACTIVE:
                continue
            remaining = o.qty - o.filled_qty
            if o.parent_id is None:
                simple += remaining
            else:
                per_bracket[o.parent_id] = max(per_bracket.get(o.parent_id, 0), remaining)
        return simple + sum(per_bracket.values())

    def _estimate_entry_cost(self, request: BracketOrderRequest) -> Decimal:
        if request.limit_price is not None:
            reference: Decimal | None = request.limit_price
        else:
            last = self._last_close.get(request.symbol)
            reference = None if last is None else self._buy_price(last)
        if reference is None:
            return _ZERO
        return Decimal(request.qty) * reference + self._commission

    @staticmethod
    def _check_increments(prices: list[Decimal]) -> None:
        for price in prices:
            if not _has_valid_increment(price):
                raise NonRetryableError(
                    f"price {price} has an invalid increment (VERIFICAR broker rules)",
                    code="INVALID_PRICE_INCREMENT",
                )
