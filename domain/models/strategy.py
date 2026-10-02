"""Strategy, sizing and check models (sec. 8.4, 13.7, 15.2, 19)."""

from decimal import Decimal
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from domain.models.base import DomainModel, Money, NonEmptyStr, Price, Symbol, UtcDatetime
from domain.models.enums import OrderSide, Timeframe

__all__ = ["CheckResult", "ProposedTrade", "RuleResult", "RuleValue", "Signal"]

RuleValue = Decimal | float | int | bool | str | None
"""Value used by a rule evaluation. Indicators may be float (sec. 6.1)."""


class RuleResult(DomainModel):
    """Result of evaluating one strategy rule with a stable id (sec. 13.3)."""

    rule_id: NonEmptyStr
    result: bool
    values: dict[str, RuleValue] = Field(default_factory=dict)


class Signal(DomainModel):
    """Immutable BUY signal record (sec. 13.7).

    ``signal_id`` is a deterministic hash of
    ``(strategy_version, symbol, timeframe, bar_start_utc, "BUY")`` computed by
    ``domain/strategy/signals.py``; this model only carries it.
    """

    signal_id: NonEmptyStr
    symbol: Symbol
    timeframe: Timeframe
    bar_start_utc: UtcDatetime
    bar_end_utc: UtcDatetime
    created_at_utc: UtcDatetime
    expires_at_utc: UtcDatetime
    rule_results: tuple[RuleResult, ...]
    strategy_version: NonEmptyStr

    @model_validator(mode="after")
    def _check_times(self) -> Self:
        if self.bar_end_utc <= self.bar_start_utc:
            raise ValueError("bar_end_utc must be after bar_start_utc")
        if self.expires_at_utc < self.bar_end_utc:
            raise ValueError("expires_at_utc cannot be before bar_end_utc")
        return self


class ProposedTrade(DomainModel):
    """A priced, sized long entry proposal produced by the Risk Engine (sec. 15.2, 16).

    The model only enforces types and signs. Exit-level coherence (sec. 16.4) and risk
    limits (sec. 15.1) are checks of ``domain/guards/checks.py`` and are deliberately
    not duplicated here (sec. 19: a single implementation per check).
    """

    signal_id: NonEmptyStr
    symbol: Symbol
    side: Literal[OrderSide.BUY] = OrderSide.BUY
    qty: int = Field(ge=1)
    entry_ref: Price
    stop_price: Price
    take_profit_price: Price
    risk_per_share: Price
    """Effective risk per share: ``stop_distance + slippage`` (sec. 15.2)."""
    risk_amount: Money
    """Monetary risk of the trade: ``qty * risk_per_share``."""
    risk_pct_of_equity: Money
    r_multiple: Money
    """Take-profit distance expressed in R."""


class CheckResult(DomainModel):
    """Result of one check of the single check library (sec. 19)."""

    passed: bool
    code: NonEmptyStr
    detail: dict[str, Any] = Field(default_factory=dict)
