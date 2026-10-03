"""AlpacaBroker behavior over a REAL ``TradingClient`` served by ``FakeTradingApi``.

Covers request payloads, idempotency by ``client_order_id`` (sec. 22), the
no-blind-resubmission rule after ambiguous failures (sec. 29), bounded retries, error
translation, cancellation and the paper-only construction (sec. 7.2).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

import pytest
import requests
from alpaca.trading.client import TradingClient
from pydantic import SecretStr

from adapters.alpaca import AlpacaCredentials, RetryPolicy
from adapters.alpaca import broker as broker_module
from adapters.alpaca._http import _TimeoutAdapter, disable_sdk_retries
from adapters.alpaca.broker import AlpacaBroker
from domain.errors import NonRetryableError, RetryableError, StateCriticalError
from domain.models import (
    BracketOrderRequest,
    BrokerOrderStatus,
    OrderClass,
    OrderSide,
    OrderType,
    SimpleOrderRequest,
    TimeInForce,
)
from tests.unit.alpaca.fake_trading_api import (
    DUPLICATE_CLIENT_ORDER_ID,
    Failure,
    FakeTradingApi,
    order_payload,
    trading_client,
)
from tests.unit.alpaca.fakes import FAKE_KEY, FAKE_SECRET, SleepRecorder

READ_POLICY = RetryPolicy(max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=10.0)
SUBMIT_POLICY = RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=2.0)
COID = "paper-abc123-entry"
BY_CLIENT_ID = "/orders:by_client_order_id"


def _broker(api: FakeTradingApi, sleep: SleepRecorder | None = None) -> AlpacaBroker:
    return AlpacaBroker(
        trading_client(api),
        retry_policy=READ_POLICY,
        submit_retry_policy=SUBMIT_POLICY,
        sleep=sleep or SleepRecorder(),
        unit_random=lambda: 0.0,
    )


def _bracket(
    coid: str = COID,
    *,
    order_type: Literal[OrderType.MARKET, OrderType.LIMIT] = OrderType.MARKET,
    tif: TimeInForce = TimeInForce.DAY,
    symbol: str = "SPY",
    qty: int = 2,
) -> BracketOrderRequest:
    return BracketOrderRequest(
        symbol=symbol,
        qty=qty,
        order_type=order_type,
        limit_price=Decimal("500.25") if order_type is OrderType.LIMIT else None,
        time_in_force=tif,
        take_profit_limit_price=Decimal("510.10"),
        stop_loss_stop_price=Decimal("495.07"),
        client_order_id=coid,
    )


def _exit(order_type: OrderType = OrderType.MARKET, **prices: Decimal) -> SimpleOrderRequest:
    return SimpleOrderRequest(
        symbol="SPY",
        qty=2,
        side=OrderSide.SELL,
        order_type=order_type,
        time_in_force=TimeInForce.GTC,
        client_order_id="paper-abc123-exit",
        **prices,
    )


def _timeout(*, after_processing: bool) -> Failure:
    return Failure(
        exception=requests.ReadTimeout("read timed out"), after_processing=after_processing
    )


# --------------------------------------------------------------------------- submissions


async def test_market_bracket_payload_and_result() -> None:
    api = FakeTradingApi()
    order = await _broker(api).submit_bracket(_bracket())
    assert api.posted == [
        {
            "symbol": "SPY",
            "qty": 2.0,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "extended_hours": False,
            "client_order_id": COID,
            "take_profit": {"limit_price": 510.1},
            "stop_loss": {"stop_price": 495.07},
        }
    ]
    assert (order.client_order_id, order.order_class, order.status) == (
        COID,
        OrderClass.BRACKET,
        BrokerOrderStatus.NEW,
    )
    take_profit, stop_loss = order.legs
    assert (take_profit.order_type, take_profit.limit_price) == (OrderType.LIMIT, Decimal("510.1"))
    assert (stop_loss.order_type, stop_loss.stop_price) == (OrderType.STOP, Decimal("495.07"))
    assert take_profit.status is stop_loss.status is BrokerOrderStatus.HELD


async def test_limit_bracket_sends_limit_price_and_gtc() -> None:
    api = FakeTradingApi()
    order = await _broker(api).submit_bracket(
        _bracket(order_type=OrderType.LIMIT, tif=TimeInForce.GTC)
    )
    body = api.posted[0]
    assert (body["type"], body["limit_price"], body["time_in_force"]) == ("limit", 500.25, "gtc")
    assert order.limit_price == Decimal("500.25")


@pytest.mark.parametrize(
    ("order_type", "prices", "expected"),
    [
        (OrderType.MARKET, {}, {"type": "market"}),
        (
            OrderType.LIMIT,
            {"limit_price": Decimal("501.5")},
            {"type": "limit", "limit_price": 501.5},
        ),
        (OrderType.STOP, {"stop_price": Decimal("0.1234")}, {"type": "stop", "stop_price": 0.1234}),
    ],
)
async def test_simple_sell_payloads(
    order_type: OrderType, prices: dict[str, Decimal], expected: dict[str, Any]
) -> None:
    api = FakeTradingApi()
    order = await _broker(api).submit_simple(_exit(order_type, **prices))
    body = api.posted[0]
    assert body == {
        "symbol": "SPY",
        "qty": 2.0,
        "side": "sell",
        "time_in_force": "gtc",
        "extended_hours": False,
        "client_order_id": "paper-abc123-exit",
        **expected,
    }
    assert "order_class" not in body
    assert (order.order_class, order.order_type, order.side) == (
        OrderClass.SIMPLE,
        order_type,
        OrderSide.SELL,
    )


@pytest.mark.parametrize(
    "request_",
    [
        SimpleOrderRequest(
            symbol="SPY",
            qty=1,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id="buy",
        ),
        _exit(OrderType.STOP_LIMIT, limit_price=Decimal(9), stop_price=Decimal(10)),
    ],
    ids=["simple-buy", "stop-limit"],
)
async def test_unsupported_simple_orders_never_reach_the_broker(
    request_: SimpleOrderRequest,
) -> None:
    api = FakeTradingApi()
    with pytest.raises(NonRetryableError) as info:
        await _broker(api).submit_simple(request_)
    assert info.value.code == "UNSUPPORTED_ORDER"
    assert api.calls == []


async def test_too_long_client_order_id_is_refused_locally() -> None:
    api = FakeTradingApi()
    with pytest.raises(NonRetryableError) as info:
        await _broker(api).submit_bracket(_bracket("x" * 129))
    assert info.value.code == "INVALID_CLIENT_ORDER_ID"
    assert api.calls == []


async def test_duplicate_client_order_id_returns_the_existing_order() -> None:
    api = FakeTradingApi()
    broker = _broker(api)
    first = await broker.submit_bracket(_bracket())
    second = await broker.submit_bracket(_bracket())
    assert second.order_id == first.order_id
    assert {leg.order_id for leg in second.legs} == {leg.order_id for leg in first.legs}
    assert api.order_count == 1
    assert api.count("POST", "/orders") == 2
    assert api.count("GET", BY_CLIENT_ID) == 1


async def test_duplicate_that_is_a_different_order_is_state_critical() -> None:
    api = FakeTradingApi()
    api.add_order(order_payload(client_order_id=COID, symbol="QQQ", qty="2"))
    with pytest.raises(StateCriticalError) as info:
        await _broker(api).submit_bracket(_bracket())
    assert info.value.code == "CLIENT_ORDER_ID_CONFLICT"


async def test_duplicate_reply_without_a_visible_order_is_unknown_submission() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", Failure(status=422, body=DUPLICATE_CLIENT_ORDER_ID))
    with pytest.raises(StateCriticalError) as info:
        await _broker(api).submit_bracket(_bracket())
    assert info.value.code == "UNKNOWN_SUBMISSION"


async def test_timeout_after_creation_is_resolved_by_lookup_without_resending() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", _timeout(after_processing=True))
    sleep = SleepRecorder()
    order = await _broker(api, sleep).submit_bracket(_bracket())
    assert order.client_order_id == COID
    assert api.order_count == 1
    assert api.count("POST", "/orders") == 1  # never re-sent
    assert [(m, p) for m, p, _ in api.calls] == [("POST", "/orders"), ("GET", BY_CLIENT_ID)]
    assert sleep.delays == [0.5]  # waited before looking up


async def test_timeout_before_creation_looks_up_then_resends_once() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", _timeout(after_processing=False))
    order = await _broker(api).submit_bracket(_bracket())
    assert order.client_order_id == COID
    assert [(m, p) for m, p, _ in api.calls] == [
        ("POST", "/orders"),
        ("GET", BY_CLIENT_ID),
        ("POST", "/orders"),
    ]
    assert api.order_count == 1


async def test_gateway_timeout_after_creation_is_not_resent_by_the_sdk() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", Failure(status=504, after_processing=True))
    order = await _broker(api).submit_bracket(_bracket())
    assert order.client_order_id == COID
    assert api.count("POST", "/orders") == 1
    assert api.order_count == 1


async def test_sdk_default_resends_a_post_on_504_which_is_why_it_is_disabled() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", Failure(status=504, after_processing=True))
    client = trading_client(api, sdk_retries=True)
    broker = AlpacaBroker(client, sleep=SleepRecorder(), unit_random=lambda: 0.0)
    order = await broker.submit_bracket(_bracket())  # adopted through the duplicate reply
    assert api.count("POST", "/orders") == 2  # alpaca-py re-sent the POST on its own
    assert order.client_order_id == COID
    assert api.order_count == 1


async def test_unknown_outcome_after_bounded_attempts_is_state_critical() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", *[_timeout(after_processing=False)] * 3)
    sleep = SleepRecorder()
    with pytest.raises(StateCriticalError) as info:
        await _broker(api, sleep).submit_bracket(_bracket())
    assert info.value.code == "UNKNOWN_SUBMISSION"
    assert api.count("POST", "/orders") == SUBMIT_POLICY.max_attempts
    assert api.count("GET", BY_CLIENT_ID) == SUBMIT_POLICY.max_attempts  # before each retry + final
    assert sleep.delays == [0.5, 1.0, 2.0]
    assert api.order_count == 0


async def test_failed_lookup_after_timeout_is_state_critical_and_never_resends() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", _timeout(after_processing=False))
    api.fail("GET", BY_CLIENT_ID, *[Failure(status=500)] * READ_POLICY.max_attempts)
    with pytest.raises(StateCriticalError) as info:
        await _broker(api).submit_bracket(_bracket())
    assert info.value.code == "UNKNOWN_SUBMISSION"
    assert api.count("POST", "/orders") == 1
    assert api.count("GET", BY_CLIENT_ID) == READ_POLICY.max_attempts


async def test_rejection_after_an_ambiguous_attempt_is_settled_by_lookup() -> None:
    api = FakeTradingApi()
    api.fail(
        "POST",
        "/orders",
        _timeout(after_processing=False),
        Failure(status=403, body={"code": 40310000, "message": "insufficient buying power"}),
    )
    with pytest.raises(StateCriticalError) as info:
        await _broker(api).submit_bracket(_bracket())
    assert info.value.code == "UNKNOWN_SUBMISSION"
    assert api.count("GET", BY_CLIENT_ID) == 2


async def test_rate_limit_is_retried_without_lookup_and_bounded() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", *[Failure(status=429)] * 3)
    sleep = SleepRecorder()
    with pytest.raises(RetryableError) as info:
        await _broker(api, sleep).submit_bracket(_bracket())
    assert info.value.code == "RATE_LIMITED"
    assert api.count("POST", "/orders") == SUBMIT_POLICY.max_attempts
    assert api.count("GET", BY_CLIENT_ID) == 0
    assert sleep.delays == [0.5, 1.0]


async def test_rate_limit_then_success() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", Failure(status=429))
    order = await _broker(api).submit_bracket(_bracket())
    assert order.client_order_id == COID
    assert api.count("POST", "/orders") == 2


@pytest.mark.parametrize(
    ("status", "message", "code"),
    [
        (403, "insufficient buying power", "INSUFFICIENT_BUYING_POWER"),
        (
            403,
            "insufficient qty available for order (requested: 2, available: 0)",
            "INSUFFICIENT_QTY_AVAILABLE",
        ),
        (
            422,
            "invalid limit_price 1.234. sub-penny increment does not fulfill minimum "
            "pricing criteria",
            "INVALID_PRICE_INCREMENT",
        ),
        (422, "take_profit.limit_price must be >= base_price + 0.01", "INVALID_BRACKET_PRICES"),
        (403, "potential wash trade detected. use complex orders", "POTENTIAL_WASH_TRADE"),
        (422, "asset ZZZZ is not tradable", "ORDER_REJECTED"),
        (403, "forbidden.", "ALPACA_AUTH"),
        (401, "request is not authorized", "ALPACA_AUTH"),
    ],
)
async def test_rejections_are_translated_and_never_retried(
    status: int, message: str, code: str
) -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", Failure(status=status, body={"code": 1, "message": message}))
    with pytest.raises(NonRetryableError) as info:
        await _broker(api).submit_bracket(_bracket())
    assert info.value.code == code
    assert api.count("POST", "/orders") == 1
    assert api.count("GET", BY_CLIENT_ID) == 0
    assert FAKE_KEY not in str(info.value)
    assert FAKE_SECRET not in str(info.value)


async def test_unreadable_response_of_an_accepted_order_is_unknown_submission() -> None:
    api = FakeTradingApi()
    api.fail("POST", "/orders", Failure(status=200, body={"id": "not-an-order"}))
    with pytest.raises(StateCriticalError) as info:
        await _broker(api).submit_bracket(_bracket())
    assert info.value.code == "UNKNOWN_SUBMISSION"


async def test_price_that_a_float_cannot_carry_exactly_is_refused() -> None:
    api = FakeTradingApi()
    request = _exit(OrderType.LIMIT, limit_price=Decimal("500.1234567890123456789"))
    with pytest.raises(NonRetryableError) as info:
        await _broker(api).submit_simple(request)
    assert info.value.code == "PRICE_NOT_REPRESENTABLE"
    assert api.calls == []


# --------------------------------------------------------------------------- queries


async def test_reads_retry_rate_limits_with_backoff() -> None:
    api = FakeTradingApi()
    api.fail("GET", "/account", Failure(status=429), Failure(status=503))
    sleep = SleepRecorder()
    account = await _broker(api, sleep).get_account()
    assert account.status == "ACTIVE"
    assert sleep.delays == [1.0, 2.0]
    assert api.count("GET", "/account") == 3


async def test_reads_give_up_after_bounded_attempts() -> None:
    api = FakeTradingApi()
    api.fail("GET", "/positions", *[Failure(status=429)] * 5)
    with pytest.raises(RetryableError) as info:
        await _broker(api).get_positions()
    assert info.value.code == "RATE_LIMITED"
    assert api.count("GET", "/positions") == READ_POLICY.max_attempts


async def test_read_4xx_is_not_retried() -> None:
    api = FakeTradingApi()
    api.fail("GET", "/positions", Failure(status=403, body={"code": 1, "message": "forbidden."}))
    with pytest.raises(NonRetryableError) as info:
        await _broker(api).get_positions()
    assert info.value.code == "ALPACA_AUTH"
    assert api.count("GET", "/positions") == 1


async def test_network_errors_are_retried() -> None:
    api = FakeTradingApi()
    api.fail("GET", "/positions", Failure(exception=requests.ConnectionError("reset")))
    assert await _broker(api).get_positions() == []
    assert api.count("GET", "/positions") == 2


async def test_unknown_client_order_id_is_none() -> None:
    api = FakeTradingApi()
    assert await _broker(api).get_order_by_client_id("never-sent") is None
    assert api.calls == [("GET", BY_CLIENT_ID, {"client_order_id": "never-sent"})]


async def test_lookup_without_legs_is_re_read_nested() -> None:
    api = FakeTradingApi(by_client_id_nests_legs=False)
    broker = _broker(api)
    submitted = await broker.submit_bracket(_bracket())
    found = await broker.get_order_by_client_id(COID)
    assert found is not None
    assert len(found.legs) == 2
    assert found == submitted
    assert api.calls[-1] == ("GET", f"/orders/{submitted.order_id}", {"nested": True})


async def test_open_orders_are_listed_nested_and_closed_ones_skipped() -> None:
    api = FakeTradingApi()
    broker = _broker(api)
    filled = await broker.submit_bracket(_bracket("filled-entry"))
    api.set_status(filled.order_id, "filled", filled_qty="2", filled_avg_price="500.3")
    for item in filled.legs:
        api.set_status(item.order_id, "new")
    canceled = await broker.submit_bracket(_bracket("canceled-entry"))
    await broker.cancel_order(canceled.order_id)
    orders = await broker.get_open_orders()
    assert [o.order_id for o in orders] == [filled.order_id]
    assert orders[0].status is BrokerOrderStatus.FILLED
    assert {leg.status for leg in orders[0].legs} == {BrokerOrderStatus.NEW}
    assert api.calls[-1] == ("GET", "/orders", {"status": "open", "limit": 500, "nested": True})


async def test_a_full_page_of_open_orders_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(broker_module, "OPEN_ORDERS_LIMIT", 2)
    api = FakeTradingApi()
    for _ in range(2):
        api.add_order(order_payload(status="new"))
    with pytest.raises(NonRetryableError) as info:
        await _broker(api).get_open_orders()
    assert info.value.code == "OPEN_ORDERS_LIMIT_REACHED"


# --------------------------------------------------------------------------- cancellation


async def test_cancel_unfilled_bracket_cancels_its_legs() -> None:
    api = FakeTradingApi()
    broker = _broker(api)
    order = await broker.submit_bracket(_bracket())
    await broker.cancel_order(order.order_id)
    refreshed = await broker.get_order_by_client_id(COID)
    assert refreshed is not None
    assert refreshed.status is BrokerOrderStatus.CANCELED
    assert {leg.status for leg in refreshed.legs} == {BrokerOrderStatus.CANCELED}
    assert await broker.get_open_orders() == []


async def test_cancel_unknown_order_is_order_not_found() -> None:
    api = FakeTradingApi()
    with pytest.raises(NonRetryableError) as info:
        await _broker(api).cancel_order("6f1c2b0e-0000-4000-8000-000000000000")
    assert info.value.code == "ORDER_NOT_FOUND"
    assert api.count("DELETE", "/orders/6f1c2b0e-0000-4000-8000-000000000000") == 1


async def test_cancel_with_a_non_uuid_id_never_reaches_the_broker() -> None:
    api = FakeTradingApi()
    with pytest.raises(NonRetryableError) as info:
        await _broker(api).cancel_order("sim-000001")
    assert info.value.code == "ORDER_NOT_FOUND"
    assert api.calls == []


async def test_cancel_filled_order_is_not_cancelable() -> None:
    api = FakeTradingApi()
    order_id = api.add_order(order_payload(status="filled", filled_qty="1"))
    with pytest.raises(NonRetryableError) as info:
        await _broker(api).cancel_order(order_id)
    assert info.value.code == "ORDER_NOT_CANCELABLE"


async def test_cancel_of_an_already_canceled_leg_is_a_no_op() -> None:
    api = FakeTradingApi()
    broker = _broker(api)
    parent = await broker.submit_bracket(_bracket())
    api.set_status(parent.order_id, "filled", filled_qty="2", filled_avg_price="500.3")
    take_profit, stop_loss = parent.legs
    await broker.cancel_order(take_profit.order_id)  # Alpaca also cancels the sibling
    await broker.cancel_order(stop_loss.order_id)  # 422, already canceled: no error
    assert api.count("DELETE", f"/orders/{stop_loss.order_id}") == 1


async def test_cancel_is_retried_on_server_errors() -> None:
    api = FakeTradingApi()
    broker = _broker(api)
    order = await broker.submit_bracket(_bracket())
    api.fail("DELETE", f"/orders/{order.order_id}", Failure(status=500))
    await broker.cancel_order(order.order_id)
    assert api.count("DELETE", f"/orders/{order.order_id}") == 2


# --------------------------------------------------------------------------- construction


def _credentials() -> AlpacaCredentials:
    return AlpacaCredentials(api_key=SecretStr(FAKE_KEY), secret_key=SecretStr(FAKE_SECRET))


@pytest.mark.parametrize("app_env", ["dev", "test", "paper", "PAPER", "staging"])
def test_from_credentials_forces_paper_for_every_non_live_env(app_env: str) -> None:
    broker = AlpacaBroker.from_credentials(_credentials(), app_env=app_env)
    client = broker._client
    assert isinstance(client, TradingClient)
    assert client._base_url == "https://paper-api.alpaca.markets"
    assert client._sandbox is True
    assert client._retry == 0  # the SDK never re-sends a request on its own
    assert isinstance(
        client._session.get_adapter("https://paper-api.alpaca.markets"), _TimeoutAdapter
    )


@pytest.mark.parametrize("app_env", ["live", "LIVE", " live "])
def test_live_is_refused_before_any_client_is_built(
    app_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[object] = []
    monkeypatch.setattr(broker_module, "TradingClient", lambda **kw: built.append(kw))
    with pytest.raises(NonRetryableError) as info:
        AlpacaBroker.from_credentials(_credentials(), app_env=app_env)
    assert info.value.code == "LIVE_BLOCKED"
    assert built == []


def test_disable_sdk_retries_refuses_an_unknown_sdk_layout() -> None:
    with pytest.raises(NonRetryableError) as info:
        disable_sdk_retries(object())
    assert info.value.code == "SDK_INCOMPATIBLE"


def test_trade_updates_stream_is_phase_5() -> None:
    with pytest.raises(NotImplementedError, match="Phase 5"):
        _broker(FakeTradingApi()).stream_trade_updates()
