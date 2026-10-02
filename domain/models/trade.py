"""Trade model: fields of the ``trades`` table (sec. 38.1) plus versions (sec. 35)."""

from decimal import Decimal
from typing import Literal

from pydantic import Field

from domain.models.base import (
    DomainModel,
    Money,
    NonEmptyStr,
    NonNegativeDecimal,
    Price,
    Symbol,
    UtcDatetime,
)
from domain.models.enums import ExitReason, OrderSide, TradeState

__all__ = ["Trade", "VersionInfo"]


class VersionInfo(DomainModel):
    """Versions recorded with every decision and trade (sec. 35).

    AI-related fields are ``None`` when the AI filter was not involved (``DISABLED``).
    """

    strategy_version: NonEmptyStr
    risk_version: NonEmptyStr
    config_version: NonEmptyStr
    config_hash: NonEmptyStr
    app_version: str | None = None
    model_id: str | None = None
    prompt_version: str | None = None
    snapshot_version: str | None = None


class Trade(DomainModel):
    """One long trade, one-to-one with its signal (``trade_id`` derives from ``signal_id``).

    Planned levels (``entry_ref``, ``stop_price``, ``take_profit_price``) come from the
    Risk Engine. Actual prices and P&L come exclusively from broker fills (sec. 39).
    """

    trade_id: NonEmptyStr
    signal_id: NonEmptyStr
    symbol: Symbol
    side: Literal[OrderSide.BUY] = OrderSide.BUY
    qty: int = Field(ge=1)
    filled_qty: NonNegativeDecimal = Decimal(0)
    entry_ref: Price
    stop_price: Price
    take_profit_price: Price
    entry_avg_price: Price | None = None
    entry_filled_at_utc: UtcDatetime | None = None
    exit_avg_price: Price | None = None
    exit_filled_at_utc: UtcDatetime | None = None
    exit_reason: ExitReason | None = None
    gross_pnl: Money | None = None
    net_pnl: Money | None = None
    result_r: Money | None = None
    versions: VersionInfo
    state: TradeState
