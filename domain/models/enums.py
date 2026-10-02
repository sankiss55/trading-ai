"""Domain enumerations (StrEnum so values serialize as plain strings).

Domain values are upper-case and broker-agnostic. Adapters translate to and from the
broker or SDK representation (sec. 8.3.5). Members marked VERIFICAR mirror an external
API and must be checked against its official documentation before relying on them.
"""

from enum import StrEnum

__all__ = [
    "AIMode",
    "AIReasonCode",
    "AIRiskFlag",
    "AIValidity",
    "AIVerdictKind",
    "BarStatus",
    "BrokerOrderStatus",
    "CircuitBreakerState",
    "DataFeed",
    "EquitySnapshotKind",
    "ExitReason",
    "HoldingMode",
    "NotificationEventType",
    "OrderClass",
    "OrderRole",
    "OrderSide",
    "OrderType",
    "ReconciliationOutcome",
    "Severity",
    "StrategyAction",
    "SystemMode",
    "TimeInForce",
    "Timeframe",
    "TradeState",
    "TradeUpdateEvent",
]


class Timeframe(StrEnum):
    """Bar timeframe. Values follow the spec/Alpaca notation (``"5Min"``)."""

    MIN_1 = "1Min"
    MIN_5 = "5Min"
    MIN_15 = "15Min"
    MIN_30 = "30Min"
    HOUR_1 = "1Hour"
    DAY_1 = "1Day"

    @property
    def minutes(self) -> int | None:
        """Fixed duration in minutes for intraday timeframes; ``None`` for daily bars."""
        return _TIMEFRAME_MINUTES[self]


_TIMEFRAME_MINUTES: dict[Timeframe, int | None] = {
    Timeframe.MIN_1: 1,
    Timeframe.MIN_5: 5,
    Timeframe.MIN_15: 15,
    Timeframe.MIN_30: 30,
    Timeframe.HOUR_1: 60,
    Timeframe.DAY_1: None,
}


class BarStatus(StrEnum):
    """Completeness of a (possibly aggregated) bar (sec. 10.3)."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    EMPTY = "EMPTY"


class DataFeed(StrEnum):
    """Market data feed (sec. 10.2). The choice is an OWNER_DECISION."""

    IEX = "iex"
    SIP = "sip"


class HoldingMode(StrEnum):
    """Holding mode (sec. 5.6). The choice is an OWNER_DECISION."""

    INTRADAY = "intraday"
    SWING = "swing"


class StrategyAction(StrEnum):
    """Output of the deterministic strategy (sec. 5.2). There is no SELL in the MVP."""

    BUY = "BUY"
    CLOSE = "CLOSE"
    HOLD = "HOLD"
    NO_ACTION = "NO_ACTION"


class AIMode(StrEnum):
    """AI filter mode stored in ``system_control`` (sec. 4.3). Changed only by a human."""

    DISABLED = "DISABLED"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"


class AIVerdictKind(StrEnum):
    """Verdict emitted by the AI veto filter (sec. 5.3, 18.1)."""

    APPROVE = "APPROVE"
    VETO = "VETO"


class AIReasonCode(StrEnum):
    """``reason_code`` enum of the decision contract (sec. 18.1)."""

    SIGNAL_CONFIRMED = "SIGNAL_CONFIRMED"
    CONTEXT_ADVERSE = "CONTEXT_ADVERSE"
    EXTENDED_MOVE = "EXTENDED_MOVE"
    ABNORMAL_VOLATILITY = "ABNORMAL_VOLATILITY"
    NEWS_RISK = "NEWS_RISK"
    LOW_QUALITY_SETUP = "LOW_QUALITY_SETUP"
    DATA_INCONSISTENT = "DATA_INCONSISTENT"
    OTHER = "OTHER"


class AIRiskFlag(StrEnum):
    """``risk_flags`` item enum of the decision contract (sec. 18.1)."""

    BENCHMARK_WEAK = "BENCHMARK_WEAK"
    LATE_IN_SESSION = "LATE_IN_SESSION"
    GAP_RISK = "GAP_RISK"
    ERRATIC_PRICE_ACTION = "ERRATIC_PRICE_ACTION"
    LOW_VOLUME_CONTEXT = "LOW_VOLUME_CONTEXT"
    RECENT_LOSSES_SYMBOL = "RECENT_LOSSES_SYMBOL"
    NEWS_PRESENT = "NEWS_PRESENT"
    DATA_ANOMALY = "DATA_ANOMALY"


class AIValidity(StrEnum):
    """Validity of an AI filter call result (sec. 8.4, 18.3)."""

    VALID = "VALID"
    INVALID = "INVALID"
    UNAVAILABLE = "UNAVAILABLE"


class OrderSide(StrEnum):
    """Order side. The MVP is long-only: entries are always BUY."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderClass(StrEnum):
    """Order class (VERIFICAR against Alpaca)."""

    SIMPLE = "SIMPLE"
    BRACKET = "BRACKET"
    OCO = "OCO"
    OTO = "OTO"


class TimeInForce(StrEnum):
    """Time in force. Brackets only allow DAY or GTC (sec. 23.2)."""

    DAY = "DAY"
    GTC = "GTC"


class OrderRole(StrEnum):
    """Role of an order inside a trade (``orders.type`` in sec. 38.1)."""

    ENTRY = "ENTRY"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    EXIT = "EXIT"
    PROTECTIVE = "PROTECTIVE"


class BrokerOrderStatus(StrEnum):
    """Normalized broker order status (VERIFICAR against Alpaca order statuses)."""

    NEW = "NEW"
    PENDING_NEW = "PENDING_NEW"
    ACCEPTED = "ACCEPTED"
    ACCEPTED_FOR_BIDDING = "ACCEPTED_FOR_BIDDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    DONE_FOR_DAY = "DONE_FOR_DAY"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REPLACED = "REPLACED"
    PENDING_CANCEL = "PENDING_CANCEL"
    PENDING_REPLACE = "PENDING_REPLACE"
    STOPPED = "STOPPED"
    REJECTED = "REJECTED"
    SUSPENDED = "SUSPENDED"
    CALCULATED = "CALCULATED"
    HELD = "HELD"


class TradeUpdateEvent(StrEnum):
    """Normalized trade-update event type (VERIFICAR against Alpaca trade_updates)."""

    NEW = "NEW"
    PENDING_NEW = "PENDING_NEW"
    FILL = "FILL"
    PARTIAL_FILL = "PARTIAL_FILL"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    DONE_FOR_DAY = "DONE_FOR_DAY"
    REPLACED = "REPLACED"
    REJECTED = "REJECTED"
    STOPPED = "STOPPED"
    PENDING_CANCEL = "PENDING_CANCEL"
    PENDING_REPLACE = "PENDING_REPLACE"
    CALCULATED = "CALCULATED"
    SUSPENDED = "SUSPENDED"
    ORDER_REPLACE_REJECTED = "ORDER_REPLACE_REJECTED"
    ORDER_CANCEL_REJECTED = "ORDER_CANCEL_REJECTED"


class TradeState(StrEnum):
    """Order/trade state machine states (sec. 21.1). Transitions live in
    ``domain/execution/state_machine.py``."""

    SIGNAL_CREATED = "SIGNAL_CREATED"
    CHECKS_PASSED = "CHECKS_PASSED"
    RISK_APPROVED = "RISK_APPROVED"
    AI_PENDING = "AI_PENDING"
    AI_VETOED = "AI_VETOED"
    AI_DONE = "AI_DONE"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    EXITS_ACTIVE = "EXITS_ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    UNKNOWN_SUBMISSION = "UNKNOWN_SUBMISSION"
    ORPHANED = "ORPHANED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class ExitReason(StrEnum):
    """Why a position was closed (sec. 13.6)."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIME_STOP = "TIME_STOP"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    END_OF_DAY = "END_OF_DAY"
    EMERGENCY_CLOSE = "EMERGENCY_CLOSE"


class SystemMode(StrEnum):
    """System lifecycle modes (sec. 32). Only RUNNING allows new entries."""

    BOOTING = "BOOTING"
    SYNCING = "SYNCING"
    READY = "READY"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"
    EMERGENCY = "EMERGENCY"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class CircuitBreakerState(StrEnum):
    """Circuit breaker states (sec. 30.1)."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HALTED = "HALTED"
    EMERGENCY = "EMERGENCY"


class ReconciliationOutcome(StrEnum):
    """Reconciliation results (sec. 24.3)."""

    RECONCILED_OK = "RECONCILED_OK"
    RECONCILED_APPLIED = "RECONCILED_APPLIED"
    STATE_MISMATCH = "STATE_MISMATCH"
    ORPHANED = "ORPHANED"


class EquitySnapshotKind(StrEnum):
    """When an equity snapshot was taken (sec. 38.1 ``equity_snapshots``)."""

    SESSION_OPEN = "SESSION_OPEN"
    SESSION_CLOSE = "SESSION_CLOSE"
    PERIODIC = "PERIODIC"


class NotificationEventType(StrEnum):
    """Notification events (sec. 40)."""

    TRADE_OPENED = "TRADE_OPENED"
    TRADE_CLOSED = "TRADE_CLOSED"
    TRADE_REJECTED = "TRADE_REJECTED"
    AI_UNAVAILABLE = "AI_UNAVAILABLE"
    RISK_HALTED = "RISK_HALTED"
    STATE_MISMATCH = "STATE_MISMATCH"
    ORPHANED_POSITION = "ORPHANED_POSITION"
    UNPROTECTED_POSITION = "UNPROTECTED_POSITION"
    EMERGENCY_CLOSE = "EMERGENCY_CLOSE"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    BROKER_ERROR = "BROKER_ERROR"
    DAILY_SUMMARY = "DAILY_SUMMARY"


class Severity(StrEnum):
    """Notification criticality (sec. 40)."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
