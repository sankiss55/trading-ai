"""Shared builders for the simulation adapter unit tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from domain.models import (
    Bar,
    BarStatus,
    BracketOrderRequest,
    DataFeed,
    OrderType,
    Timeframe,
    TimeInForce,
)

T0 = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
"""2026-10-01 09:30 America/New_York (EDT)."""


def minute_bar(
    start: datetime,
    open_: str,
    high: str,
    low: str,
    close: str,
    *,
    symbol: str = "SPY",
    volume: int = 100,
) -> Bar:
    """COMPLETE 1-minute bar with Decimal prices parsed from strings."""
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.MIN_1,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(minutes=1),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=volume,
        feed=DataFeed.IEX,
        status=BarStatus.COMPLETE,
    )


def empty_bar(start: datetime, *, symbol: str = "SPY") -> Bar:
    """EMPTY 1-minute bar."""
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.MIN_1,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(minutes=1),
        open=None,
        high=None,
        low=None,
        close=None,
        volume=0,
        feed=DataFeed.IEX,
        status=BarStatus.EMPTY,
    )


def bracket(
    client_order_id: str = "paper-abc-entry",
    *,
    qty: int = 10,
    stop: str = "98",
    take_profit: str = "102",
    limit: str | None = None,
    tif: TimeInForce = TimeInForce.DAY,
    symbol: str = "SPY",
) -> BracketOrderRequest:
    """Bracket entry request (MARKET unless ``limit`` is given)."""
    return BracketOrderRequest(
        symbol=symbol,
        qty=qty,
        order_type=OrderType.MARKET if limit is None else OrderType.LIMIT,
        limit_price=None if limit is None else Decimal(limit),
        time_in_force=tif,
        take_profit_limit_price=Decimal(take_profit),
        stop_loss_stop_price=Decimal(stop),
        client_order_id=client_order_id,
    )
