"""Unit tests for the domain model conventions (immutability, UTC, Decimal, contracts)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from domain.errors import DomainError, NonRetryableError, RetryableError, StateCriticalError
from domain.models import (
    AIValidity,
    AIVerdict,
    AIVerdictKind,
    AIVerdictResult,
    Bar,
    BarStatus,
    BracketOrderRequest,
    BrokerOrder,
    BrokerOrderStatus,
    OrderSide,
    OrderType,
    ProposedTrade,
    Signal,
    SimpleOrderRequest,
    SystemControl,
    Timeframe,
    TimeInForce,
    TradeUpdate,
    TradeUpdateEvent,
)

START = datetime(2026, 10, 1, 14, 30, tzinfo=UTC)
END = START + timedelta(minutes=5)


def bar_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbol": "SPY",
        "timeframe": Timeframe.MIN_5,
        "bar_start_utc": START,
        "bar_end_utc": END,
        "open": Decimal("500.10"),
        "high": Decimal("501.00"),
        "low": Decimal("499.90"),
        "close": Decimal("500.50"),
        "volume": 1_000,
        "feed": "iex",
        "status": BarStatus.COMPLETE,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- immutability


def test_models_are_frozen() -> None:
    bar = Bar(**bar_kwargs())
    with pytest.raises(ValidationError):
        bar.close = Decimal("1")  # type: ignore[misc]


def test_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Bar(**bar_kwargs(vwap=Decimal("500")))


# --------------------------------------------------------------------------- UTC datetimes


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError, match="naive"):
        Bar(**bar_kwargs(bar_start_utc=datetime(2026, 10, 1, 14, 30)))  # noqa: DTZ001


def test_non_utc_offset_is_rejected() -> None:
    eastern = timezone(timedelta(hours=-4))
    with pytest.raises(ValidationError, match="UTC"):
        Bar(**bar_kwargs(bar_start_utc=datetime(2026, 10, 1, 10, 30, tzinfo=eastern)))


def test_iso_z_string_is_parsed_as_utc() -> None:
    bar = Bar(**bar_kwargs(bar_start_utc="2026-10-01T14:30:00Z"))
    assert bar.bar_start_utc == START
    assert bar.bar_start_utc.tzinfo is UTC


# --------------------------------------------------------------------------- Decimal


def test_prices_are_decimal_when_given_as_strings() -> None:
    bar = Bar(**bar_kwargs(close="500.50"))
    assert isinstance(bar.close, Decimal)
    assert bar.close == Decimal("500.50")


def test_float_prices_are_rejected() -> None:
    with pytest.raises(ValidationError, match="float"):
        Bar(**bar_kwargs(close=500.5))


@pytest.mark.parametrize("value", ["NaN", "Infinity", "0", "-1"])
def test_non_positive_or_non_finite_prices_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        Bar(**bar_kwargs(low=Decimal(value)))


# --------------------------------------------------------------------------- Bar rules


def test_incomplete_bar_requires_minutes_present() -> None:
    with pytest.raises(ValidationError, match="minutes_present"):
        Bar(**bar_kwargs(status=BarStatus.INCOMPLETE))
    bar = Bar(**bar_kwargs(status=BarStatus.INCOMPLETE, minutes_present=3))
    assert bar.minutes_present == 3


def test_empty_bar_has_no_prices() -> None:
    empty = Bar(
        **bar_kwargs(status=BarStatus.EMPTY, open=None, high=None, low=None, close=None, volume=0)
    )
    assert empty.close is None
    with pytest.raises(ValidationError, match="EMPTY"):
        Bar(**bar_kwargs(status=BarStatus.EMPTY))


def test_ohlc_inconsistency_is_rejected() -> None:
    with pytest.raises(ValidationError, match="OHLC"):
        Bar(**bar_kwargs(high=Decimal("500.00")))


def test_bar_end_must_follow_start() -> None:
    with pytest.raises(ValidationError):
        Bar(**bar_kwargs(bar_end_utc=START))


# --------------------------------------------------------------------------- other models


def test_signal_expiry_cannot_precede_bar_end() -> None:
    with pytest.raises(ValidationError, match="expires_at_utc"):
        Signal(
            signal_id="sig_1",
            symbol="SPY",
            timeframe=Timeframe.MIN_5,
            bar_start_utc=START,
            bar_end_utc=END,
            created_at_utc=END,
            expires_at_utc=START,
            rule_results=(),
            strategy_version="0.1.0",
        )


def test_proposed_trade_is_long_only() -> None:
    with pytest.raises(ValidationError):
        ProposedTrade(
            signal_id="sig_1",
            symbol="SPY",
            side=OrderSide.SELL,  # type: ignore[arg-type]
            qty=1,
            entry_ref=Decimal("100"),
            stop_price=Decimal("99"),
            take_profit_price=Decimal("102"),
            risk_per_share=Decimal("1.05"),
            risk_amount=Decimal("1.05"),
            risk_pct_of_equity=Decimal("0.5"),
            r_multiple=Decimal("2"),
        )


def test_bracket_limit_requires_limit_price() -> None:
    kwargs: dict[str, Any] = {
        "symbol": "SPY",
        "qty": 10,
        "order_type": OrderType.LIMIT,
        "time_in_force": TimeInForce.DAY,
        "take_profit_limit_price": Decimal("510"),
        "stop_loss_stop_price": Decimal("495"),
        "client_order_id": "paper-abc-entry",
    }
    with pytest.raises(ValidationError, match="limit_price"):
        BracketOrderRequest(**kwargs)
    order = BracketOrderRequest(**kwargs, limit_price=Decimal("500.25"))
    assert order.extended_hours is False


def test_simple_stop_order_requires_stop_price() -> None:
    with pytest.raises(ValidationError, match="stop_price"):
        SimpleOrderRequest(
            symbol="SPY",
            qty=10,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            time_in_force=TimeInForce.GTC,
            client_order_id="paper-abc-protective",
        )


def test_fill_event_requires_fill() -> None:
    order = BrokerOrder(
        order_id="o1",
        client_order_id="paper-abc-entry",
        symbol="SPY",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=BrokerOrderStatus.FILLED,
        qty=Decimal(10),
    )
    with pytest.raises(ValidationError, match="fill"):
        TradeUpdate(event=TradeUpdateEvent.FILL, order=order, timestamp_utc=END)


def test_system_control_defaults_are_fail_closed() -> None:
    control = SystemControl()
    assert control.trading_enabled is False
    assert control.emergency_close is False
    assert control.ai_mode == "DISABLED"


# --------------------------------------------------------------------------- AI contract

VALID_VERDICT_JSON = (
    '{"signal_id": "sig_1", "verdict": "VETO", "reason_code": "EXTENDED_MOVE",'
    ' "risk_flags": ["LATE_IN_SESSION"], "confidence": 0.64, "rationale": "late entry"}'
)


def test_ai_verdict_parses_contract_json() -> None:
    verdict = AIVerdict.model_validate_json(VALID_VERDICT_JSON)
    assert verdict.verdict is AIVerdictKind.VETO
    assert verdict.confidence == pytest.approx(0.64)


@pytest.mark.parametrize(
    "payload",
    [
        VALID_VERDICT_JSON.replace("0.64", '"0.64"'),  # string instead of number
        VALID_VERDICT_JSON.replace('"VETO"', '"SELL"'),  # value outside the enum
        VALID_VERDICT_JSON.replace("}", ', "price": 1}'),  # extra field
        VALID_VERDICT_JSON.replace('"risk_flags": ["LATE_IN_SESSION"], ', ""),  # missing
    ],
)
def test_ai_verdict_rejects_schema_violations(payload: str) -> None:
    with pytest.raises(ValidationError):
        AIVerdict.model_validate_json(payload)


def test_ai_verdict_result_validity_coherence() -> None:
    verdict = AIVerdict.model_validate_json(VALID_VERDICT_JSON)
    assert AIVerdictResult(validity=AIValidity.VALID, verdict=verdict).verdict == verdict
    with pytest.raises(ValidationError):
        AIVerdictResult(validity=AIValidity.VALID)
    with pytest.raises(ValidationError):
        AIVerdictResult(validity=AIValidity.UNAVAILABLE, verdict=verdict, invalid_reason="x")
    with pytest.raises(ValidationError):
        AIVerdictResult(validity=AIValidity.INVALID)


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize("error_cls", [RetryableError, NonRetryableError, StateCriticalError])
def test_error_classes_share_domain_base(error_cls: type[DomainError]) -> None:
    error = error_cls("broker timeout", code="BROKER_TIMEOUT")
    assert isinstance(error, DomainError)
    assert str(error) == "[BROKER_TIMEOUT] broker timeout"
