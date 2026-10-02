"""Persistence records exchanged with the ``IUnitOfWork`` repositories (sec. 8.5, 38.1).

These are not listed in sec. 8.4 but are required so that repository ports exchange
validated, immutable domain models instead of database rows (sec. 8.3.5).
"""

from typing import Any

from pydantic import Field

from domain.models.ai import AIVerdictResult, Snapshot
from domain.models.base import DomainModel, Money, NonEmptyStr, NonNegativeDecimal, UtcDatetime
from domain.models.broker import BracketOrderRequest, SimpleOrderRequest
from domain.models.enums import (
    AIMode,
    AIReasonCode,
    AIValidity,
    AIVerdictKind,
    BrokerOrderStatus,
    EquitySnapshotKind,
    OrderRole,
    ReconciliationOutcome,
    TradeState,
)
from domain.models.strategy import CheckResult

__all__ = [
    "AIDecisionRecord",
    "EquitySnapshot",
    "OrderEvent",
    "OrderRecord",
    "ReconciliationRecord",
    "RiskEvent",
    "ShadowOutcome",
    "SystemControl",
    "SystemEvent",
]


class OrderRecord(DomainModel):
    """Row of ``orders``: one order of a trade, keyed by its deterministic client id."""

    client_order_id: NonEmptyStr
    trade_id: NonEmptyStr
    role: OrderRole
    state: TradeState
    request: BracketOrderRequest | SimpleOrderRequest
    order_id: str | None = None
    broker_status: BrokerOrderStatus | None = None


class OrderEvent(DomainModel):
    """Row of ``order_events``: one persisted state transition (sec. 21.2)."""

    client_order_id: NonEmptyStr
    occurred_at_utc: UtcDatetime
    from_state: TradeState | None
    to_state: TradeState
    cause: NonEmptyStr
    broker_event: dict[str, Any] | None = None


class AIDecisionRecord(DomainModel):
    """Row of ``ai_decisions`` (sec. 38.1): snapshot sent, raw and parsed response."""

    signal_id: NonEmptyStr
    ai_mode: AIMode
    snapshot: Snapshot
    result: AIVerdictResult
    model_id: NonEmptyStr
    prompt_version: NonEmptyStr
    estimated_cost_usd: NonNegativeDecimal
    created_at_utc: UtcDatetime


class RiskEvent(DomainModel):
    """Row of ``risk_events``: all check results, full sizing and the decision."""

    signal_id: NonEmptyStr
    occurred_at_utc: UtcDatetime
    checks: tuple[CheckResult, ...]
    sizing: dict[str, Any] = Field(default_factory=dict)
    decision: NonEmptyStr


class EquitySnapshot(DomainModel):
    """Row of ``equity_snapshots``."""

    taken_at_utc: UtcDatetime
    kind: EquitySnapshotKind
    equity: Money
    last_equity: Money
    buying_power: Money


class ReconciliationRecord(DomainModel):
    """Row of ``reconciliations``: result, detected differences and actions (sec. 24)."""

    occurred_at_utc: UtcDatetime
    outcome: ReconciliationOutcome
    differences: tuple[dict[str, Any], ...] = ()
    actions: tuple[str, ...] = ()


class ShadowOutcome(DomainModel):
    """Row of ``shadow_outcomes``: counterfactual result of a SHADOW signal (sec. 46.2)."""

    signal_id: NonEmptyStr
    recorded_at_utc: UtcDatetime
    validity: AIValidity
    verdict: AIVerdictKind | None = None
    reason_code: AIReasonCode | None = None
    confidence: float | None = None
    executed: bool
    simulated: bool
    result_r: Money | None = None


class SystemControl(DomainModel):
    """Row of ``system_control`` (sec. 31.1). Defaults are fail-closed."""

    trading_enabled: bool = False
    emergency_close: bool = False
    ai_mode: AIMode = AIMode.DISABLED
    updated_at_utc: UtcDatetime | None = None
    updated_by: str | None = None
    reason: str | None = None


class SystemEvent(DomainModel):
    """Row of ``system_events``: startups, mode/breaker transitions, control changes."""

    occurred_at_utc: UtcDatetime
    event_type: NonEmptyStr
    detail: dict[str, Any] = Field(default_factory=dict)
