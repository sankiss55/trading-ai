"""``IBroker`` over the Alpaca trading REST API, always on the paper endpoint (sec. 7.2, 8.5).

Scope (Phase 2): account, positions, open orders with legs, lookup by
``client_order_id``, bracket entries, simple exits and cancellation. The trade-updates
stream (``TradingStream``) is Phase 5: ``stream_trade_updates`` raises
``NotImplementedError``. No SDK type leaves this module (sec. 8.3.5) and every SDK error
is translated (sec. 8.3.7).

SDK facts (verified against the alpaca-py 0.44.0 source, 2026-10-02):

* ``TradingClient(api_key, secret_key, paper=True)`` targets
  ``https://paper-api.alpaca.markets``; alpaca-py reads no environment variable that could
  redirect it. :meth:`AlpacaBroker.from_credentials` refuses ``APP_ENV=live`` before any
  client exists and never passes ``paper=False``.
* ``get_account() -> TradeAccount`` (``equity``, ``last_equity``, ``buying_power`` as
  strings, ``status: AccountStatus`` upper-case); ``get_all_positions() -> list[Position]``
  (``qty``, ``avg_entry_price``, ``market_value`` strings, ``side: long | short``).
* ``get_orders(GetOrdersRequest(status=OPEN, nested=True, limit=500))`` (``limit``
  defaults to 50, max 500) rolls bracket legs up under ``Order.legs``.
* ``get_order_by_client_id(id)`` calls ``GET /v2/orders:by_client_order_id``, which
  answers 404 for an unknown id; the endpoint has no ``nested`` parameter, so a
  multi-leg order returned without ``legs`` is re-read with ``get_order_by_id(id,
  GetOrderByIdRequest(nested=True))``.
* ``submit_order(MarketOrderRequest | LimitOrderRequest | StopOrderRequest)`` posts
  ``to_request_fields()``: prices and ``qty`` are SDK ``float`` fields, serialized with
  the shortest repr (``Decimal("187.35")`` -> ``187.35``, ``3`` -> ``3.0``). Each price is
  checked to survive that round trip exactly. Brackets carry ``order_class=bracket``,
  ``take_profit={limit_price}``, ``stop_loss={stop_price}`` (no stop-limit),
  ``extended_hours=false``, ``time_in_force`` from the request and the caller's
  ``client_order_id`` (Alpaca limit: 128 characters, checked before sending).
* ``cancel_order_by_id(uuid)`` (``DELETE /v2/orders/{id}``): 204 accepted, 404 unknown,
  422 no longer cancelable. A non-UUID id is rejected locally (``ORDER_NOT_FOUND``).
* ``Order`` fields used: ``id``, ``client_order_id``, ``symbol``, ``side``, ``type`` (or
  deprecated ``order_type``), ``order_class``, ``status``, ``qty``, ``filled_qty``,
  ``filled_avg_price`` (may be ``0`` before processing -> ``None``), ``limit_price``,
  ``stop_price``, ``legs``, ``updated_at``. Numbers may arrive as ``str`` or ``float``.

Idempotency and retries (sec. 22, 29):

* The SDK's own 429/504 re-send is disabled on this client (``disable_sdk_retries``).
* Reads and cancellations go through :func:`call_with_retry` (bounded backoff; 429, 5xx
  and network errors retried, 4xx not). A cancel answered 422 whose order is already
  ``CANCELED``/``PENDING_CANCEL`` (e.g. a retried cancel, or a leg canceled with its
  sibling) succeeds as a no-op.
* Submissions: ``client_order_id must be unique`` adopts the existing order (looked up by
  ``client_order_id``; a different symbol/side/qty/class is ``StateCriticalError``
  ``CLIENT_ORDER_ID_CONFLICT``). 429 is retried with backoff (the order was not
  accepted). Any other retryable failure (timeout, connection error, 5xx) is
  **ambiguous**: before re-sending, the order is looked up by ``client_order_id``; if it
  exists it is returned, otherwise the same request (same ``client_order_id``) is
  re-sent. Attempts are bounded. An outcome still unknown at the end, a failed lookup,
  a rejection that follows an ambiguous attempt without the order being visible, or an
  accepted submission whose response cannot be read raise ``StateCriticalError``
  ``UNKNOWN_SUBMISSION`` (sec. 21.1): reconciliation settles it by ``client_order_id``.

Mapping decisions: every SDK ``OrderStatus`` maps to the same-named domain status except
``pending_review``, which has no domain member and maps to ``PENDING_NEW`` (an open,
not-yet-working order). ``trailing_stop`` orders, ``mleg`` orders and notional orders
cannot be represented and raise ``UNSUPPORTED_BROKER_ORDER`` (fail closed: the account
is expected to hold only system orders). Short positions are reported with negative
``qty``. ``AccountState.buying_power`` is Alpaca ``buying_power``.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, TypeVar
from uuid import UUID

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass as SdkOrderClass
from alpaca.trading.enums import OrderSide as SdkOrderSide
from alpaca.trading.enums import OrderStatus as SdkOrderStatus
from alpaca.trading.enums import OrderType as SdkOrderType
from alpaca.trading.enums import PositionSide, QueryOrderStatus
from alpaca.trading.enums import TimeInForce as SdkTimeInForce
from alpaca.trading.models import Order as SdkOrder
from alpaca.trading.models import Position as SdkPosition
from alpaca.trading.models import TradeAccount
from alpaca.trading.requests import (
    GetOrderByIdRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    OrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
)

from adapters.alpaca._errors import (
    translate_cancel_error,
    translate_error,
    translate_lookup_error,
    translate_submit_error,
)
from adapters.alpaca._http import (
    AlpacaCredentials,
    HttpTimeouts,
    disable_sdk_retries,
    install_timeouts,
    require_paper_environment,
)
from adapters.alpaca._retry import (
    ErrorTranslator,
    RequestPacer,
    RetryPolicy,
    Sleep,
    call_with_retry,
)
from domain.errors import DomainError, NonRetryableError, RetryableError, StateCriticalError
from domain.models import (
    AccountState,
    BracketOrderRequest,
    BrokerOrder,
    BrokerOrderStatus,
    OrderClass,
    OrderSide,
    OrderType,
    Position,
    SimpleOrderRequest,
    TimeInForce,
    TradeUpdate,
)

__all__ = [
    "DEFAULT_MIN_REQUEST_INTERVAL_SECONDS",
    "DEFAULT_SUBMIT_RETRY_POLICY",
    "MAX_CLIENT_ORDER_ID_LENGTH",
    "OPEN_ORDERS_LIMIT",
    "AlpacaBroker",
    "TradingApiClient",
    "convert_account",
    "convert_order",
    "convert_position",
]

_LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.3
"""Pacing for the trading API limit of 200 requests/minute per account (VERIFICAR)."""

DEFAULT_SUBMIT_RETRY_POLICY = RetryPolicy(
    max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=4.0, jitter_ratio=0.2
)
"""Submission attempts are fewer and faster than reads: a stale entry is worse than none
(SUGERIDO technical default, sec. 29)."""

MAX_CLIENT_ORDER_ID_LENGTH = 128
"""Alpaca ``client_order_id`` limit (docs ``postorder``: "<= 128 characters")."""

OPEN_ORDERS_LIMIT = 500
"""``GET /v2/orders`` maximum page size; a full page may be truncated and is refused."""

_SDK_MULTI_LEG = frozenset({SdkOrderClass.BRACKET, SdkOrderClass.OCO, SdkOrderClass.OTO})

_STATUSES: dict[SdkOrderStatus, BrokerOrderStatus] = {
    SdkOrderStatus.NEW: BrokerOrderStatus.NEW,
    SdkOrderStatus.PARTIALLY_FILLED: BrokerOrderStatus.PARTIALLY_FILLED,
    SdkOrderStatus.FILLED: BrokerOrderStatus.FILLED,
    SdkOrderStatus.DONE_FOR_DAY: BrokerOrderStatus.DONE_FOR_DAY,
    SdkOrderStatus.CANCELED: BrokerOrderStatus.CANCELED,
    SdkOrderStatus.EXPIRED: BrokerOrderStatus.EXPIRED,
    SdkOrderStatus.REPLACED: BrokerOrderStatus.REPLACED,
    SdkOrderStatus.PENDING_CANCEL: BrokerOrderStatus.PENDING_CANCEL,
    SdkOrderStatus.PENDING_REPLACE: BrokerOrderStatus.PENDING_REPLACE,
    SdkOrderStatus.PENDING_REVIEW: BrokerOrderStatus.PENDING_NEW,
    SdkOrderStatus.ACCEPTED: BrokerOrderStatus.ACCEPTED,
    SdkOrderStatus.PENDING_NEW: BrokerOrderStatus.PENDING_NEW,
    SdkOrderStatus.ACCEPTED_FOR_BIDDING: BrokerOrderStatus.ACCEPTED_FOR_BIDDING,
    SdkOrderStatus.STOPPED: BrokerOrderStatus.STOPPED,
    SdkOrderStatus.REJECTED: BrokerOrderStatus.REJECTED,
    SdkOrderStatus.SUSPENDED: BrokerOrderStatus.SUSPENDED,
    SdkOrderStatus.CALCULATED: BrokerOrderStatus.CALCULATED,
    SdkOrderStatus.HELD: BrokerOrderStatus.HELD,
}
_TYPES: dict[SdkOrderType, OrderType] = {
    SdkOrderType.MARKET: OrderType.MARKET,
    SdkOrderType.LIMIT: OrderType.LIMIT,
    SdkOrderType.STOP: OrderType.STOP,
    SdkOrderType.STOP_LIMIT: OrderType.STOP_LIMIT,
}
_CLASSES: dict[SdkOrderClass, OrderClass] = {
    SdkOrderClass.SIMPLE: OrderClass.SIMPLE,
    SdkOrderClass.BRACKET: OrderClass.BRACKET,
    SdkOrderClass.OCO: OrderClass.OCO,
    SdkOrderClass.OTO: OrderClass.OTO,
}
_SIDES: dict[SdkOrderSide, OrderSide] = {
    SdkOrderSide.BUY: OrderSide.BUY,
    SdkOrderSide.SELL: OrderSide.SELL,
}
_SDK_SIDES = {domain: sdk for sdk, domain in _SIDES.items()}
_SDK_TIME_IN_FORCE: dict[TimeInForce, SdkTimeInForce] = {
    TimeInForce.DAY: SdkTimeInForce.DAY,
    TimeInForce.GTC: SdkTimeInForce.GTC,
}
_ALREADY_CANCELING = frozenset({BrokerOrderStatus.CANCELED, BrokerOrderStatus.PENDING_CANCEL})
_SINGLE_ATTEMPT = RetryPolicy(max_attempts=1)


class TradingApiClient(Protocol):
    """The subset of ``TradingClient`` used here (injectable in tests)."""

    def get_account(self) -> TradeAccount | dict[str, Any]:
        """Account details."""
        ...

    def get_all_positions(self) -> list[SdkPosition] | dict[str, Any]:
        """Open positions."""
        ...

    def get_orders(self, filter: GetOrdersRequest | None = None) -> list[SdkOrder] | dict[str, Any]:
        """Orders matching the filter."""
        ...

    def get_order_by_id(
        self, order_id: UUID | str, filter: GetOrderByIdRequest | None = None
    ) -> SdkOrder | dict[str, Any]:
        """One order by broker id."""
        ...

    def get_order_by_client_id(self, client_id: str) -> SdkOrder | dict[str, Any]:
        """One order by ``client_order_id``."""
        ...

    def submit_order(self, order_data: OrderRequest) -> SdkOrder | dict[str, Any]:
        """Create an order."""
        ...

    def cancel_order_by_id(self, order_id: UUID | str) -> None:
        """Request the cancellation of an order."""
        ...


# --------------------------------------------------------------------------- conversion


class _UnsupportedOrderError(ValueError):
    """An SDK order the domain model cannot represent."""


def _decimal(value: str | float | None, *, field: str) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} is not a number: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def _required_decimal(value: str | float | None, *, field: str) -> Decimal:
    result = _decimal(value, field=field)
    if result is None:
        raise ValueError(f"{field} is missing")
    return result


def _utc(moment: datetime, *, field: str) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{field} is not timezone-aware")
    return moment.astimezone(UTC)


def _convert_order(order: SdkOrder) -> BrokerOrder:
    sdk_type = order.type or order.order_type
    if sdk_type is None:
        raise ValueError("order type is missing")
    if sdk_type not in _TYPES:
        raise _UnsupportedOrderError(f"order type {sdk_type.value} is not supported")
    if order.order_class not in _CLASSES:
        raise _UnsupportedOrderError(f"order class {order.order_class.value} is not supported")
    if order.qty is None:
        raise _UnsupportedOrderError("orders without qty (notional) are not supported")
    if order.symbol is None:
        raise ValueError("symbol is missing")
    if order.side is None:
        raise ValueError("side is missing")
    filled_avg_price = _decimal(order.filled_avg_price, field="filled_avg_price")
    return BrokerOrder(
        order_id=str(order.id),
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        side=_SIDES[order.side],
        order_type=_TYPES[sdk_type],
        order_class=_CLASSES[order.order_class],
        status=_STATUSES[order.status],
        qty=_required_decimal(order.qty, field="qty"),
        filled_qty=_decimal(order.filled_qty, field="filled_qty") or Decimal(0),
        # Alpaca reports 0 until the order is processed: no fill price yet.
        filled_avg_price=filled_avg_price if filled_avg_price else None,
        limit_price=_decimal(order.limit_price, field="limit_price"),
        stop_price=_decimal(order.stop_price, field="stop_price"),
        legs=tuple(_convert_order(leg) for leg in order.legs or ()),
        updated_at_utc=_utc(order.updated_at, field="updated_at"),
    )


def convert_order(order: SdkOrder) -> BrokerOrder:
    """Convert an SDK order (and its legs, recursively) into a domain ``BrokerOrder``.

    Raises:
        NonRetryableError: code ``UNSUPPORTED_BROKER_ORDER`` (trailing stop, multi-leg
            option or notional order) or ``INVALID_ORDER_DATA`` (missing or invalid field).
    """
    try:
        return _convert_order(order)
    except _UnsupportedOrderError as exc:
        raise NonRetryableError(
            f"order {order.id}: {exc}", code="UNSUPPORTED_BROKER_ORDER"
        ) from exc
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        raise NonRetryableError(
            f"order {order.id}: {str(exc)[:300]}", code="INVALID_ORDER_DATA"
        ) from exc


def convert_account(account: TradeAccount) -> AccountState:
    """Convert the SDK trading account (``buying_power`` is Alpaca ``buying_power``).

    Raises:
        NonRetryableError: code ``INVALID_ACCOUNT_DATA``.
    """
    try:
        return AccountState(
            equity=_required_decimal(account.equity, field="equity"),
            last_equity=_required_decimal(account.last_equity, field="last_equity"),
            buying_power=_required_decimal(account.buying_power, field="buying_power"),
            status=account.status.value.upper(),
        )
    except ValueError as exc:
        raise NonRetryableError(f"account: {str(exc)[:300]}", code="INVALID_ACCOUNT_DATA") from exc


def convert_position(position: SdkPosition) -> Position:
    """Convert an SDK position; short positions get a negative ``qty``.

    Raises:
        NonRetryableError: code ``INVALID_POSITION_DATA``.
    """
    try:
        qty = _required_decimal(position.qty, field="qty")
        if position.side is PositionSide.SHORT and qty > 0:
            qty = -qty
        return Position(
            symbol=position.symbol,
            qty=qty,
            avg_entry_price=_required_decimal(position.avg_entry_price, field="avg_entry_price"),
            market_value=_required_decimal(position.market_value, field="market_value"),
        )
    except ValueError as exc:
        raise NonRetryableError(
            f"position {position.symbol}: {str(exc)[:300]}", code="INVALID_POSITION_DATA"
        ) from exc


# --------------------------------------------------------------------------- requests


def _sdk_price(value: Decimal, *, field: str) -> float:
    """The SDK float for ``value``; refuses prices the float round trip would alter."""
    result = float(value)
    if Decimal(repr(result)) != value:
        raise NonRetryableError(
            f"{field} {value} cannot be sent exactly as a float", code="PRICE_NOT_REPRESENTABLE"
        )
    return result


def _check_client_order_id(client_order_id: str) -> None:
    if len(client_order_id) > MAX_CLIENT_ORDER_ID_LENGTH:
        raise NonRetryableError(
            f"client_order_id has {len(client_order_id)} characters "
            f"(Alpaca limit {MAX_CLIENT_ORDER_ID_LENGTH})",
            code="INVALID_CLIENT_ORDER_ID",
        )


def _bracket_order_request(request: BracketOrderRequest) -> OrderRequest:
    take_profit = TakeProfitRequest(
        limit_price=_sdk_price(request.take_profit_limit_price, field="take_profit_limit_price")
    )
    stop_loss = StopLossRequest(
        stop_price=_sdk_price(request.stop_loss_stop_price, field="stop_loss_stop_price")
    )
    time_in_force = _SDK_TIME_IN_FORCE[request.time_in_force]
    if request.order_type is OrderType.MARKET:
        return MarketOrderRequest(
            symbol=request.symbol,
            qty=request.qty,
            side=SdkOrderSide.BUY,
            time_in_force=time_in_force,
            order_class=SdkOrderClass.BRACKET,
            take_profit=take_profit,
            stop_loss=stop_loss,
            client_order_id=request.client_order_id,
            extended_hours=False,
        )
    if request.limit_price is None:  # pragma: no cover - guaranteed by the domain model
        raise NonRetryableError("LIMIT bracket without limit_price", code="INVALID_ORDER_REQUEST")
    return LimitOrderRequest(
        symbol=request.symbol,
        qty=request.qty,
        side=SdkOrderSide.BUY,
        time_in_force=time_in_force,
        order_class=SdkOrderClass.BRACKET,
        limit_price=_sdk_price(request.limit_price, field="limit_price"),
        take_profit=take_profit,
        stop_loss=stop_loss,
        client_order_id=request.client_order_id,
        extended_hours=False,
    )


def _simple_order_request(request: SimpleOrderRequest) -> OrderRequest:
    if request.side is not OrderSide.SELL:
        raise NonRetryableError(
            "simple BUY orders are not supported: entries are brackets (sec. 23.1)",
            code="UNSUPPORTED_ORDER",
        )
    side = _SDK_SIDES[request.side]
    time_in_force = _SDK_TIME_IN_FORCE[request.time_in_force]
    if request.order_type is OrderType.MARKET:
        return MarketOrderRequest(
            symbol=request.symbol,
            qty=request.qty,
            side=side,
            time_in_force=time_in_force,
            client_order_id=request.client_order_id,
            extended_hours=False,
        )
    if request.order_type is OrderType.LIMIT and request.limit_price is not None:
        return LimitOrderRequest(
            symbol=request.symbol,
            qty=request.qty,
            side=side,
            time_in_force=time_in_force,
            limit_price=_sdk_price(request.limit_price, field="limit_price"),
            client_order_id=request.client_order_id,
            extended_hours=False,
        )
    if request.order_type is OrderType.STOP and request.stop_price is not None:
        return StopOrderRequest(
            symbol=request.symbol,
            qty=request.qty,
            side=side,
            time_in_force=time_in_force,
            stop_price=_sdk_price(request.stop_price, field="stop_price"),
            client_order_id=request.client_order_id,
            extended_hours=False,
        )
    raise NonRetryableError(
        f"{request.order_type} simple orders are not supported (MARKET, LIMIT, STOP only)",
        code="UNSUPPORTED_ORDER",
    )


@dataclass(frozen=True, slots=True)
class _Expected:
    """What an adopted order (same ``client_order_id``) must match."""

    client_order_id: str
    symbol: str
    side: OrderSide
    qty: int
    order_class: OrderClass


def _expect(result: object, kind: type[T], operation: str) -> T:
    if not isinstance(result, kind):
        raise NonRetryableError(
            f"{operation}: expected {kind.__name__}, got {type(result).__name__}",
            code="INVALID_SCHEMA",
        )
    return result


def _expect_list(result: object, kind: type[T], operation: str) -> list[T]:
    if not isinstance(result, list) or not all(isinstance(item, kind) for item in result):
        raise NonRetryableError(
            f"{operation}: expected a list of {kind.__name__}", code="INVALID_SCHEMA"
        )
    return result


def _parse_order_id(order_id: str) -> UUID:
    try:
        return UUID(order_id)
    except ValueError as exc:
        raise NonRetryableError(
            f"order id {order_id!r} is not an Alpaca order id (UUID)", code="ORDER_NOT_FOUND"
        ) from exc


# --------------------------------------------------------------------------- adapter


class AlpacaBroker:
    """``IBroker`` on the Alpaca paper trading API (REST only until Phase 5).

    Args:
        client: alpaca-py ``TradingClient`` (paper, SDK retries disabled) or a test double.
        retry_policy: Bounded retry policy of reads, lookups and cancellations.
        submit_retry_policy: Attempts and backoff of order submissions.
        sleep: Async sleep used by retries and the pacer.
        pacer: Optional local request pacer.
        unit_random: ``U[0, 1)`` source of the retry jitter.
    """

    def __init__(
        self,
        client: TradingApiClient,
        *,
        retry_policy: RetryPolicy | None = None,
        submit_retry_policy: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
        pacer: RequestPacer | None = None,
        unit_random: Callable[[], float] = random.random,
    ) -> None:
        self._client = client
        self._policy = retry_policy or RetryPolicy()
        self._submit_policy = submit_retry_policy or DEFAULT_SUBMIT_RETRY_POLICY
        self._sleep = sleep
        self._pacer = pacer
        self._unit_random = unit_random

    @classmethod
    def from_credentials(
        cls,
        credentials: AlpacaCredentials,
        *,
        app_env: str,
        timeouts: HttpTimeouts | None = None,
        retry_policy: RetryPolicy | None = None,
        submit_retry_policy: RetryPolicy | None = None,
        min_request_interval_seconds: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
    ) -> AlpacaBroker:
        """Build the adapter over a real ``TradingClient(paper=True)`` (network).

        Only the composition root may call this (sec. 8.3.4). Every ``app_env`` other
        than ``live`` gets a paper client; ``live`` is refused before any client exists.

        Raises:
            NonRetryableError: code ``LIVE_BLOCKED`` for ``app_env == "live"``;
                ``SDK_INCOMPATIBLE`` if the SDK layout changed.
        """
        require_paper_environment(app_env)
        client = TradingClient(
            api_key=credentials.api_key.get_secret_value(),
            secret_key=credentials.secret_key.get_secret_value(),
            paper=True,
        )
        install_timeouts(client, timeouts or HttpTimeouts())
        disable_sdk_retries(client)
        return cls(
            client,
            retry_policy=retry_policy,
            submit_retry_policy=submit_retry_policy,
            pacer=RequestPacer(min_request_interval_seconds),
        )

    async def _call(
        self,
        call: Callable[[], T],
        *,
        operation: str,
        policy: RetryPolicy | None = None,
        translate: ErrorTranslator = translate_error,
    ) -> T:
        return await call_with_retry(
            call,
            operation=operation,
            policy=policy or self._policy,
            sleep=self._sleep,
            pacer=self._pacer,
            unit_random=self._unit_random,
            translate=translate,
        )

    # ------------------------------------------------------------------ queries

    async def get_account(self) -> AccountState:
        """Account equity, last equity, buying power and status.

        Raises:
            RetryableError: retries exhausted (rate limit, 5xx, network).
            NonRetryableError: auth, rejected request, invalid account data.
        """
        operation = "get_account"
        result = await self._call(self._client.get_account, operation=operation)
        return convert_account(_expect(result, TradeAccount, operation))

    async def get_positions(self) -> list[Position]:
        """Open positions as reported by Alpaca (source of truth, sec. 3.3)."""
        operation = "get_all_positions"
        result = await self._call(self._client.get_all_positions, operation=operation)
        return [convert_position(p) for p in _expect_list(result, SdkPosition, operation)]

    async def get_open_orders(self) -> list[BrokerOrder]:
        """Open orders with their legs nested (``status=open&nested=true``).

        Raises:
            NonRetryableError: code ``OPEN_ORDERS_LIMIT_REACHED`` when a full page came
                back (the listing could be truncated), plus the usual read errors.
        """
        operation = "get_orders open nested"
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN, nested=True, limit=OPEN_ORDERS_LIMIT
        )
        result = await self._call(
            lambda: self._client.get_orders(filter=request), operation=operation
        )
        orders = _expect_list(result, SdkOrder, operation)
        if len(orders) >= OPEN_ORDERS_LIMIT:
            raise NonRetryableError(
                f"{operation}: {len(orders)} orders returned, the listing may be truncated",
                code="OPEN_ORDERS_LIMIT_REACHED",
            )
        return [convert_order(order) for order in orders]

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """The order with ``client_order_id`` (legs included), ``None`` on HTTP 404."""
        operation = f"get_order_by_client_id {client_order_id}"
        try:
            result = await self._call(
                lambda: self._client.get_order_by_client_id(client_order_id),
                operation=operation,
                translate=translate_lookup_error,
            )
        except NonRetryableError as exc:
            if exc.code == "ORDER_NOT_FOUND":
                return None
            raise
        return await self._with_legs(_expect(result, SdkOrder, operation))

    async def _with_legs(self, order: SdkOrder) -> BrokerOrder:
        """Convert ``order``; a multi-leg order without ``legs`` is re-read nested."""
        if order.order_class in _SDK_MULTI_LEG and not order.legs:
            operation = f"get_order_by_id {order.id} nested"
            result = await self._call(
                lambda: self._client.get_order_by_id(
                    order.id, filter=GetOrderByIdRequest(nested=True)
                ),
                operation=operation,
                translate=translate_lookup_error,
            )
            order = _expect(result, SdkOrder, operation)
        return convert_order(order)

    # ------------------------------------------------------------------ commands

    async def submit_bracket(self, request: BracketOrderRequest) -> BrokerOrder:
        """Submit a BUY bracket entry (TP limit + SL stop legs), idempotent by
        ``client_order_id``.

        Raises:
            NonRetryableError: rejection (``INSUFFICIENT_BUYING_POWER``,
                ``INVALID_PRICE_INCREMENT``, ``INVALID_BRACKET_PRICES``, ``ORDER_REJECTED``,
                ``INVALID_CLIENT_ORDER_ID``, ``PRICE_NOT_REPRESENTABLE``...).
            RetryableError: rate limited on every attempt (no order was accepted).
            StateCriticalError: ``UNKNOWN_SUBMISSION`` or ``CLIENT_ORDER_ID_CONFLICT``.
        """
        _check_client_order_id(request.client_order_id)
        expected = _Expected(
            request.client_order_id, request.symbol, OrderSide.BUY, request.qty, OrderClass.BRACKET
        )
        return await self._submit(
            _bracket_order_request(request),
            expected,
            operation=f"submit_bracket {request.symbol} {request.client_order_id}",
        )

    async def submit_simple(self, request: SimpleOrderRequest) -> BrokerOrder:
        """Submit a simple SELL exit (``MARKET``, ``LIMIT`` or protective ``STOP``).

        Raises:
            NonRetryableError: ``UNSUPPORTED_ORDER`` (BUY or STOP_LIMIT) or a rejection
                (``INSUFFICIENT_QTY_AVAILABLE``, ``INVALID_PRICE_INCREMENT``...).
            RetryableError: rate limited on every attempt (no order was accepted).
            StateCriticalError: ``UNKNOWN_SUBMISSION`` or ``CLIENT_ORDER_ID_CONFLICT``.
        """
        _check_client_order_id(request.client_order_id)
        sdk_request = _simple_order_request(request)
        expected = _Expected(
            request.client_order_id, request.symbol, request.side, request.qty, OrderClass.SIMPLE
        )
        return await self._submit(
            sdk_request,
            expected,
            operation=f"submit_simple {request.symbol} {request.client_order_id}",
        )

    async def _submit(
        self, sdk_request: OrderRequest, expected: _Expected, *, operation: str
    ) -> BrokerOrder:
        policy = self._submit_policy
        last_error: RetryableError | None = None
        uncertain = False  # an attempt failed without proving that no order was created
        for attempt in range(1, policy.max_attempts + 1):
            try:
                result = await self._call(
                    lambda: self._client.submit_order(sdk_request),
                    operation=operation,
                    policy=_SINGLE_ATTEMPT,
                    translate=translate_submit_error,
                )
            except RetryableError as exc:
                last_error = exc
                uncertain = uncertain or exc.code != "RATE_LIMITED"
                if attempt == policy.max_attempts:
                    break
                await self._sleep(policy.delay(attempt, self._unit_random()))
                if uncertain:
                    # The order may exist: look it up before re-sending (never blindly).
                    existing = await self._lookup_after_failure(expected, operation, exc)
                    if existing is not None:
                        return existing
                continue
            except NonRetryableError as exc:
                if exc.code == "DUPLICATE_CLIENT_ORDER_ID":
                    return await self._adopt_duplicate(expected, operation, exc)
                if exc.code == "INVALID_SCHEMA":
                    raise StateCriticalError(
                        f"{operation}: the broker accepted the request but its response "
                        f"could not be read ({exc.message})",
                        code="UNKNOWN_SUBMISSION",
                    ) from exc
                if uncertain:
                    # A rejection now does not settle an earlier ambiguous attempt.
                    return await self._resolve_uncertain(expected, operation, exc)
                raise
            return await self._accepted(result, operation)
        assert last_error is not None  # the loop only breaks after a failure
        if not uncertain:
            raise RetryableError(
                f"{last_error.message} (gave up after {policy.max_attempts} attempts; "
                "no order was accepted)",
                code=last_error.code,
            )
        # Give a request lost in flight time to surface before the final lookup.
        await self._sleep(policy.delay(policy.max_attempts, self._unit_random()))
        return await self._resolve_uncertain(expected, operation, last_error)

    async def _resolve_uncertain(
        self, expected: _Expected, operation: str, cause: DomainError
    ) -> BrokerOrder:
        """Final lookup after an ambiguous attempt: the order, or ``UNKNOWN_SUBMISSION``."""
        existing = await self._lookup_after_failure(expected, operation, cause)
        if existing is not None:
            return existing
        raise StateCriticalError(
            f"{operation}: submission outcome unknown ({cause.message}); no order with "
            f"client_order_id {expected.client_order_id} is visible yet",
            code="UNKNOWN_SUBMISSION",
        ) from cause

    async def _accepted(self, result: object, operation: str) -> BrokerOrder:
        """Convert the response of an accepted submission (never a plain rejection)."""
        try:
            order = _expect(result, SdkOrder, operation)
            converted = convert_order(order)
        except NonRetryableError as exc:
            raise StateCriticalError(
                f"{operation}: accepted but unreadable response ({exc.message})",
                code="UNKNOWN_SUBMISSION",
            ) from exc
        if order.order_class in _SDK_MULTI_LEG and not order.legs:
            try:
                return await self._with_legs(order)
            except DomainError as exc:
                _LOGGER.warning("%s: accepted, legs not readable yet (%s)", operation, exc.code)
        return converted

    async def _lookup_after_failure(
        self, expected: _Expected, operation: str, cause: DomainError
    ) -> BrokerOrder | None:
        """Look the order up after an ambiguous failure; a failed lookup is critical."""
        try:
            existing = await self.get_order_by_client_id(expected.client_order_id)
        except DomainError as exc:
            raise StateCriticalError(
                f"{operation}: submission outcome unknown ({cause.message}) and the "
                f"lookup by client_order_id failed ({exc})",
                code="UNKNOWN_SUBMISSION",
            ) from exc
        return None if existing is None else _adopt(existing, expected, operation)

    async def _adopt_duplicate(
        self, expected: _Expected, operation: str, cause: NonRetryableError
    ) -> BrokerOrder:
        """Alpaca already holds this ``client_order_id``: return that order (sec. 22.3)."""
        existing = await self._lookup_after_failure(expected, operation, cause)
        if existing is None:
            raise StateCriticalError(
                f"{operation}: Alpaca reports a duplicate client_order_id that the lookup "
                "cannot find",
                code="UNKNOWN_SUBMISSION",
            ) from cause
        return existing

    async def cancel_order(self, order_id: str) -> None:
        """Request the cancellation of ``order_id`` (confirmation comes from a query or
        the trade-updates stream). Alpaca cancels the remaining legs of the group too.

        Raises:
            NonRetryableError: ``ORDER_NOT_FOUND`` (unknown or non-UUID id),
                ``ORDER_NOT_CANCELABLE`` (filled, expired, rejected...).
            RetryableError: retries exhausted (rate limit, 5xx, network).
        """
        broker_id = _parse_order_id(order_id)
        operation = f"cancel_order {order_id}"
        try:
            await self._call(
                lambda: self._client.cancel_order_by_id(broker_id),
                operation=operation,
                translate=translate_cancel_error,
            )
        except NonRetryableError as exc:
            if exc.code != "ORDER_NOT_CANCELABLE":
                raise
            lookup = f"get_order_by_id {order_id}"
            result = await self._call(
                lambda: self._client.get_order_by_id(broker_id),
                operation=lookup,
                translate=translate_lookup_error,
            )
            status = _STATUSES.get(_expect(result, SdkOrder, lookup).status)
            if status not in _ALREADY_CANCELING:
                raise
            # Already canceled or being canceled (retried cancel, sibling leg): no-op.

    def stream_trade_updates(self) -> AsyncIterator[TradeUpdate]:
        """The trade-updates stream (``TradingStream``) is Phase 5.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError("Phase 5")


def _adopt(order: BrokerOrder, expected: _Expected, operation: str) -> BrokerOrder:
    """``order`` if it is the one this ``client_order_id`` describes; else critical."""
    mismatches = [
        name
        for name, ok in (
            ("symbol", order.symbol == expected.symbol),
            ("side", order.side is expected.side),
            ("qty", order.qty == Decimal(expected.qty)),
            ("order_class", order.order_class is expected.order_class),
        )
        if not ok
    ]
    if mismatches:
        raise StateCriticalError(
            f"{operation}: broker order {order.order_id} with client_order_id "
            f"{expected.client_order_id} differs in {', '.join(mismatches)}",
            code="CLIENT_ORDER_ID_CONFLICT",
        )
    return order
