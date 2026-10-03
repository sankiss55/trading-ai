"""In-memory fake of the Alpaca trading REST API behind a REAL ``TradingClient``.

:func:`trading_client` returns an alpaca-py ``TradingClient`` (paper, fake credentials)
whose ``requests`` session is a :class:`FakeSession`: request building
(``to_request_fields``), the SDK HTTP error path (``raise_for_status`` -> ``APIError``)
and response parsing (``Order(**json)``) all run for real; only the network is replaced
by :class:`FakeTradingApi`. No network, no credentials (sec. 58.4).

Modelled Alpaca semantics (docs.alpaca.markets, verified 2026-10-02, sec. 60):

* ``POST /v2/orders``: a ``client_order_id`` already used answers HTTP 422
  ``{"code": 40010001, "message": "client_order_id must be unique"}`` (the fake checks
  every order, history included; whether Alpaca only checks active orders is VERIFICAR).
  A bracket creates a TP sell ``limit`` leg and an SL sell ``stop`` leg in ``held``.
* ``GET /v2/orders?status=open&nested=true``: top-level orders, legs nested. A parent is
  listed while it or one of its legs is open (VERIFICAR how Alpaca lists a filled parent
  whose legs are open).
* ``GET /v2/orders:by_client_order_id`` and ``GET /v2/orders/{id}``: 404
  ``{"code": 40410000, "message": "order not found"}`` when unknown.
* ``DELETE /v2/orders/{id}``: 204; 404 unknown; 422 when final. Canceling any order of a
  group cancels the remaining open orders of the group (Alpaca "Placing Orders": "If any
  one of the orders is canceled, any remaining open order in the group is canceled").
  The fake cancels synchronously (Alpaca goes through ``pending_cancel``).

Failures are scripted per ``(method, path)`` with :meth:`FakeTradingApi.fail`; an
``after_processing`` failure lets the server act first (e.g. the order is created and
then the response is lost), which is how a timeout after submission looks to the client.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import uuid4

import requests
from alpaca.trading.client import TradingClient

from adapters.alpaca._http import disable_sdk_retries
from tests.unit.alpaca.fakes import FAKE_KEY, FAKE_SECRET, clock_payload

BASE = "https://paper-api.alpaca.markets/v2"
NOW = "2026-10-01T13:31:00.123456Z"

ORDER_NOT_FOUND = {"code": 40410000, "message": "order not found"}
DUPLICATE_CLIENT_ORDER_ID = {"code": 40010001, "message": "client_order_id must be unique"}
NOT_CANCELABLE = {"code": 42210000, "message": "order is not cancelable"}
FINAL_STATUSES = frozenset({"filled", "canceled", "expired", "replaced", "rejected"})


def account_payload(**overrides: Any) -> dict[str, Any]:
    """An API-shaped ``GET /v2/account`` body."""
    payload: dict[str, Any] = {
        "id": "11111111-2222-3333-4444-555555555555",
        "account_number": "PA0000000000",
        "status": "ACTIVE",
        "currency": "USD",
        "cash": "100000",
        "buying_power": "200000",
        "equity": "100000.5",
        "last_equity": "99950.25",
        "portfolio_value": "100000.5",
        "pattern_day_trader": False,
        "trading_blocked": False,
        "transfers_blocked": False,
        "account_blocked": False,
        "multiplier": "2",
    }
    payload.update(overrides)
    return payload


def position_payload(
    symbol: str, qty: str, *, avg_entry_price: str = "100", side: str = "long", **overrides: Any
) -> dict[str, Any]:
    """An API-shaped position."""
    market_value = str(Decimal(qty) * Decimal(avg_entry_price))
    payload: dict[str, Any] = {
        "asset_id": str(uuid4()),
        "symbol": symbol,
        "exchange": "ARCA",
        "asset_class": "us_equity",
        "avg_entry_price": avg_entry_price,
        "qty": qty,
        "qty_available": qty,
        "side": side,
        "market_value": market_value,
        "cost_basis": market_value,
        "unrealized_pl": "0",
        "current_price": avg_entry_price,
    }
    payload.update(overrides)
    return payload


def order_payload(**overrides: Any) -> dict[str, Any]:
    """An API-shaped order (simple market buy unless overridden)."""
    payload: dict[str, Any] = {
        "id": str(uuid4()),
        "client_order_id": f"coid-{uuid4().hex[:12]}",
        "created_at": NOW,
        "updated_at": NOW,
        "submitted_at": NOW,
        "filled_at": None,
        "expired_at": None,
        "canceled_at": None,
        "failed_at": None,
        "replaced_at": None,
        "replaced_by": None,
        "replaces": None,
        "asset_id": str(uuid4()),
        "symbol": "SPY",
        "asset_class": "us_equity",
        "notional": None,
        "qty": "1",
        "filled_qty": "0",
        "filled_avg_price": None,
        "order_class": "simple",
        "order_type": "market",
        "type": "market",
        "side": "buy",
        "time_in_force": "day",
        "limit_price": None,
        "stop_price": None,
        "status": "new",
        "extended_hours": False,
        "legs": None,
        "trail_percent": None,
        "trail_price": None,
        "hwm": None,
    }
    payload.update(overrides)
    if "type" in overrides and "order_type" not in overrides:
        payload["order_type"] = overrides["type"]
    return payload


def _number_text(value: Any) -> str | None:
    """How Alpaca echoes a number: a plain decimal string (``3.0`` -> ``"3"``)."""
    if value is None:
        return None
    return format(Decimal(str(value)).normalize(), "f")


@dataclass
class Failure:
    """A scripted failure of one ``(method, path)`` request."""

    status: int | None = None
    body: dict[str, Any] | None = None
    exception: BaseException | None = None
    after_processing: bool = False


@dataclass
class _Record:
    payload: dict[str, Any]
    leg_ids: list[str] = field(default_factory=list)
    parent_id: str | None = None


class FakeTradingApi:
    """In-memory Alpaca trading endpoints used by ``AlpacaBroker`` (see module docstring).

    Args:
        account: ``GET /v2/account`` body.
        positions: ``GET /v2/positions`` body.
        clock: ``GET /v2/clock`` body.
        by_client_id_nests_legs: Whether ``orders:by_client_order_id`` includes ``legs``
            (unknown for Alpaca, both cases are tested).
    """

    def __init__(
        self,
        *,
        account: dict[str, Any] | None = None,
        positions: list[dict[str, Any]] | None = None,
        clock: dict[str, Any] | None = None,
        by_client_id_nests_legs: bool = True,
    ) -> None:
        self.account = account if account is not None else account_payload()
        self.positions = positions if positions is not None else []
        self.clock = clock if clock is not None else clock_payload()
        self.by_client_id_nests_legs = by_client_id_nests_legs
        self.records: dict[str, _Record] = {}
        self.top_level: list[str] = []
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.posted: list[dict[str, Any]] = []
        self._failures: dict[tuple[str, str], list[Failure]] = {}

    # ------------------------------------------------------------------ scripting

    def fail(self, method: str, path: str, *failures: Failure) -> None:
        """Queue ``failures`` for the next ``method path`` requests (in order)."""
        self._failures.setdefault((method.upper(), path), []).extend(failures)

    def take_failure(self, method: str, path: str) -> Failure | None:
        queue = self._failures.get((method.upper(), path))
        return queue.pop(0) if queue else None

    def add_order(self, payload: dict[str, Any], legs: Sequence[dict[str, Any]] = ()) -> str:
        """Store an existing order (with optional legs) and return its id."""
        order_id = str(payload["id"])
        self.records[order_id] = _Record(payload=dict(payload, legs=None))
        self.top_level.append(order_id)
        for leg in legs:
            leg_id = str(leg["id"])
            self.records[leg_id] = _Record(payload=dict(leg, legs=None), parent_id=order_id)
            self.records[order_id].leg_ids.append(leg_id)
        return order_id

    def set_status(self, order_id: str, status: str, **fields: Any) -> None:
        """Change an order's status (and any other field) as the venue would."""
        self.records[order_id].payload.update(status=status, **fields)

    def count(self, method: str, path: str) -> int:
        """Number of requests received for ``method path``."""
        return sum(1 for m, p, _ in self.calls if m == method.upper() and p == path)

    @property
    def order_count(self) -> int:
        """Top-level orders stored (each bracket counts once)."""
        return len(self.top_level)

    # ------------------------------------------------------------------ rendering

    def _render(self, order_id: str, *, nested: bool) -> dict[str, Any]:
        record = self.records[order_id]
        payload = dict(record.payload)
        payload["legs"] = (
            [self._render(leg, nested=False) for leg in record.leg_ids]
            if nested and record.leg_ids
            else None
        )
        return payload

    def _is_open(self, order_id: str) -> bool:
        return self.records[order_id].payload["status"] not in FINAL_STATUSES

    def _group(self, order_id: str) -> list[str]:
        record = self.records[order_id]
        root = record.parent_id or order_id
        return [root, *self.records[root].leg_ids]

    # ------------------------------------------------------------------ endpoints

    def handle(
        self, method: str, path: str, params: dict[str, Any], body: dict[str, Any] | None
    ) -> tuple[int, Any]:
        """Serve one request; returns ``(status, json body or None)``."""
        if method == "GET" and path == "/account":
            return 200, self.account
        if method == "GET" and path == "/positions":
            return 200, self.positions
        if method == "GET" and path == "/clock":
            return 200, self.clock
        if method == "GET" and path == "/orders":
            return 200, self._list_orders(params)
        if method == "GET" and path == "/orders:by_client_order_id":
            coid = params.get("client_order_id")
            for order_id, record in self.records.items():
                if record.payload["client_order_id"] == coid:
                    return 200, self._render(order_id, nested=self.by_client_id_nests_legs)
            return 404, ORDER_NOT_FOUND
        if path.startswith("/orders/"):
            order_id = path.removeprefix("/orders/")
            if order_id not in self.records:
                return 404, ORDER_NOT_FOUND
            if method == "GET":
                return 200, self._render(order_id, nested=_truthy(params.get("nested")))
            if method == "DELETE":
                return self._cancel(order_id)
        if method == "POST" and path == "/orders":
            assert body is not None
            return self._create(body)
        raise AssertionError(f"unexpected request {method} {path}")

    def _list_orders(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        assert params.get("status") == "open", params
        nested = _truthy(params.get("nested"))
        limit = int(params.get("limit") or 50)
        ids = self.top_level if nested else list(self.records)
        listed = [
            order_id
            for order_id in reversed(ids)  # direction=desc (Alpaca default)
            if self._is_open(order_id)
            or (nested and any(self._is_open(leg) for leg in self.records[order_id].leg_ids))
        ]
        return [self._render(order_id, nested=nested) for order_id in listed[:limit]]

    def _cancel(self, order_id: str) -> tuple[int, Any]:
        if not self._is_open(order_id):
            return 422, NOT_CANCELABLE
        for member in self._group(order_id):
            if self._is_open(member):
                self.set_status(member, "canceled", canceled_at=NOW, updated_at=NOW)
        return 204, None

    def _create(self, body: dict[str, Any]) -> tuple[int, Any]:
        self.posted.append(body)
        coid = body.get("client_order_id") or str(uuid4())
        if any(r.payload["client_order_id"] == coid for r in self.records.values()):
            return 422, DUPLICATE_CLIENT_ORDER_ID
        order_class = body.get("order_class", "simple")
        common = {
            "symbol": body["symbol"],
            "qty": _number_text(body["qty"]),
            "time_in_force": body["time_in_force"],
            "order_class": order_class,
            "extended_hours": body.get("extended_hours", False),
        }
        parent = order_payload(
            client_order_id=coid,
            side=body["side"],
            type=body["type"],
            limit_price=_number_text(body.get("limit_price")),
            stop_price=_number_text(body.get("stop_price")),
            status="new",
            **common,
        )
        legs: list[dict[str, Any]] = []
        if order_class == "bracket":
            leg_common = {**common, "side": "sell", "status": "held"}
            legs = [
                order_payload(
                    type="limit",
                    limit_price=_number_text(body["take_profit"]["limit_price"]),
                    **leg_common,
                ),
                order_payload(
                    type="stop",
                    stop_price=_number_text(body["stop_loss"]["stop_price"]),
                    **leg_common,
                ),
            ]
        order_id = self.add_order(parent, legs)
        return 200, self._render(order_id, nested=True)


def _truthy(value: Any) -> bool:
    return value in (True, "true", "True")


def _response(url: str, status: int, body: Any) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = url
    response.reason = "OK" if status < 400 else "Error"
    response.encoding = "utf-8"
    response._content = b"" if body is None else json.dumps(body).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


class FakeSession(requests.Session):
    """``requests.Session`` that serves ``TradingClient`` requests from a fake API."""

    def __init__(self, api: FakeTradingApi) -> None:
        super().__init__()
        self.api = api

    def request(self, method: str | bytes, url: str | bytes, *args: Any, **kwargs: Any) -> Any:
        method_text = method.decode() if isinstance(method, bytes) else method
        url_text = url.decode() if isinstance(url, bytes) else url
        assert url_text.startswith(BASE), url_text  # paper endpoint only
        path = url_text.removeprefix(BASE)
        params = dict(kwargs.get("params") or {})
        raw_body = kwargs.get("json")
        # What the server would see once ``requests`` has serialized the JSON body.
        body = json.loads(json.dumps(raw_body)) if raw_body is not None else None
        self.api.calls.append((method_text.upper(), path, params))
        failure = self.api.take_failure(method_text, path)
        if failure is not None and not failure.after_processing:
            return _failure_response(url_text, failure)
        status, payload = self.api.handle(method_text.upper(), path, params, body)
        if failure is not None:
            return _failure_response(url_text, failure)
        return _response(url_text, status, payload)


def _failure_response(url: str, failure: Failure) -> requests.Response:
    if failure.exception is not None:
        raise failure.exception
    assert failure.status is not None
    body = failure.body if failure.body is not None else {"code": 0, "message": "simulated"}
    return _response(url, failure.status, body)


def trading_client(api: FakeTradingApi, *, sdk_retries: bool = False) -> TradingClient:
    """A real paper ``TradingClient`` served by ``api`` (SDK retries disabled, as in
    production, unless ``sdk_retries``; their 3 s wait is then set to 0)."""
    client = TradingClient(api_key=FAKE_KEY, secret_key=FAKE_SECRET, paper=True)
    client._session = FakeSession(api)
    if sdk_retries:
        client._retry_wait = 0
    else:
        disable_sdk_retries(client)
    return client
