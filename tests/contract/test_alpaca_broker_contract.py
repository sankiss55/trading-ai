"""``BrokerContract`` (sec. 49.2) run against ``AlpacaBroker`` over a REAL ``TradingClient``
served by ``FakeTradingApi`` (no network).

What applies here: account shape, unknown ``client_order_id`` lookups, bracket legs,
idempotency of a repeated ``client_order_id`` and cancellation of an unfilled entry
(checked by a REST query, the ``CANCELED`` trade update needs the stream).

What does not, and why: the contract tests that wait for fills observe them through
``stream_trade_updates``, which is Phase 5 for Alpaca; fills and legs holding the
position quantity are venue behavior a fake cannot prove. Those are skipped here and
must be checked against the paper account (sec. 49.2) once the stream exists.
``BracketSemanticsContract`` (bar-level fills) only applies to simulated brokers.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from adapters.alpaca.broker import AlpacaBroker
from domain.models import BracketOrderRequest, BrokerOrderStatus, OrderType, TimeInForce
from domain.ports import IBroker
from tests.contract.broker_contract import BrokerContract, BrokerHarness, lookup
from tests.unit.alpaca.fake_trading_api import FakeTradingApi, trading_client
from tests.unit.alpaca.fakes import SleepRecorder

STREAM_PHASE_5 = "needs stream_trade_updates (Alpaca TradingStream is Phase 5) and real fills"


class AlpacaFakeHarness:
    """``BrokerHarness`` over ``AlpacaBroker`` + ``FakeTradingApi``."""

    def __init__(self) -> None:
        self.api = FakeTradingApi()
        self.alpaca = AlpacaBroker(trading_client(self.api), sleep=SleepRecorder())
        self._sequence = 0

    @property
    def broker(self) -> IBroker:
        return self.alpaca

    @property
    def symbol(self) -> str:
        return "SPY"

    def client_order_id(self, tag: str) -> str:
        self._sequence += 1
        return f"test-{tag}-{self._sequence}"

    def bracket_request(
        self, client_order_id: str, *, qty: int = 1, tif: TimeInForce = TimeInForce.DAY
    ) -> BracketOrderRequest:
        return BracketOrderRequest(
            symbol=self.symbol,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=tif,
            take_profit_limit_price=Decimal(102),
            stop_loss_stop_price=Decimal(98),
            client_order_id=client_order_id,
        )

    def resting_bracket_request(self, client_order_id: str) -> BracketOrderRequest:
        return BracketOrderRequest(
            symbol=self.symbol,
            qty=1,
            order_type=OrderType.LIMIT,
            limit_price=Decimal(50),
            time_in_force=TimeInForce.DAY,
            take_profit_limit_price=Decimal(150),
            stop_loss_stop_price=Decimal(49),
            client_order_id=client_order_id,
        )

    async def fill_pending_market_orders(self) -> None:
        raise NotImplementedError(STREAM_PHASE_5)


class TestAlpacaBrokerContract(BrokerContract):
    @pytest.fixture
    def broker_harness(self) -> AlpacaFakeHarness:
        return AlpacaFakeHarness()

    async def test_cancel_unfilled_entry(self, broker_harness: BrokerHarness) -> None:
        # REST-only variant: the CANCELED trade update needs the Phase 5 stream.
        h = broker_harness
        coid = h.client_order_id("cancel")
        order = await h.broker.submit_bracket(h.resting_bracket_request(coid))
        await h.broker.cancel_order(order.order_id)
        assert (await lookup(h.broker, coid)).status is BrokerOrderStatus.CANCELED
        assert order.order_id not in {o.order_id for o in await h.broker.get_open_orders()}

    @pytest.mark.skip(reason=STREAM_PHASE_5)
    async def test_entry_fill_opens_position_and_activates_legs(
        self, broker_harness: BrokerHarness
    ) -> None:
        raise AssertionError("skipped")

    @pytest.mark.skip(reason=STREAM_PHASE_5)
    async def test_active_legs_hold_the_position_quantity(
        self, broker_harness: BrokerHarness
    ) -> None:
        raise AssertionError("skipped")

    @pytest.mark.skip(reason=STREAM_PHASE_5)
    async def test_system_exit_after_canceling_legs(self, broker_harness: BrokerHarness) -> None:
        raise AssertionError("skipped")
