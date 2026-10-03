"""Persistence semantics shared by every ``IUnitOfWork`` adapter (sec. 8.5, 21.1, 38).

``SqliteUnitOfWork`` and ``InMemoryUnitOfWork`` both import these definitions, so the
in-memory adapter used by tests and simulation reports exactly the same final states,
UTC day/month windows and error codes as the SQLite adapter used in paper trading.

``FINAL_STATES`` mirrors sec. 21.1. It should move to ``domain/execution/state_machine.py``
once that module exists; until then this is the single definition used by persistence.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Final, TypeVar

from pydantic import BaseModel

from domain.errors import NonRetryableError
from domain.models import TradeState

__all__ = [
    "CONSTRAINT_VIOLATION",
    "CORRUPT_RECORD",
    "DB_BUSY",
    "DB_CORRUPT",
    "DB_ERROR",
    "DB_NOT_INITIALIZED",
    "DB_NOT_WRITABLE",
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DUPLICATE_KEY",
    "FINAL_STATES",
    "FOREIGN_KEY_VIOLATION",
    "INVALID_ARGUMENT",
    "ORDER_ID_MISMATCH",
    "ORDER_NOT_FOUND",
    "SCHEMA_VERSION_MISMATCH",
    "SIGNAL_NOT_FOUND",
    "TRADE_IDENTITY_CHANGED",
    "TRADE_NOT_FOUND",
    "TRANSACTION_ABORTED",
    "UOW_ALREADY_ACTIVE",
    "UOW_NOT_ACTIVE",
    "day_window",
    "month_window",
    "normalized",
]

M = TypeVar("M", bound=BaseModel)

FINAL_STATES: Final[frozenset[TradeState]] = frozenset(
    {
        TradeState.AI_VETOED,
        TradeState.CLOSED,
        TradeState.REJECTED,
        TradeState.EXPIRED,
        TradeState.CANCELLED,
        TradeState.FAILED,
    }
)
"""States marked final in sec. 21.1. ``UNKNOWN_SUBMISSION``, ``ORPHANED`` and
``RECOVERY_REQUIRED`` are NOT final: they stay visible to reconciliation (sec. 24)."""

DEFAULT_LOCK_TIMEOUT_SECONDS: Final = 5.0
"""How long a unit of work waits for another one's write transaction before failing
with ``RetryableError(code=DB_BUSY)``."""

# --------------------------------------------------------------------------- error codes
# RetryableError
DB_BUSY: Final = "DB_BUSY"
# NonRetryableError
DUPLICATE_KEY: Final = "DUPLICATE_KEY"
FOREIGN_KEY_VIOLATION: Final = "FOREIGN_KEY_VIOLATION"
CONSTRAINT_VIOLATION: Final = "CONSTRAINT_VIOLATION"
SIGNAL_NOT_FOUND: Final = "SIGNAL_NOT_FOUND"
TRADE_NOT_FOUND: Final = "TRADE_NOT_FOUND"
TRADE_IDENTITY_CHANGED: Final = "TRADE_IDENTITY_CHANGED"
ORDER_NOT_FOUND: Final = "ORDER_NOT_FOUND"
UOW_NOT_ACTIVE: Final = "UOW_NOT_ACTIVE"
UOW_ALREADY_ACTIVE: Final = "UOW_ALREADY_ACTIVE"
TRANSACTION_ABORTED: Final = "TRANSACTION_ABORTED"
INVALID_ARGUMENT: Final = "INVALID_ARGUMENT"
DB_NOT_INITIALIZED: Final = "DB_NOT_INITIALIZED"
SCHEMA_VERSION_MISMATCH: Final = "SCHEMA_VERSION_MISMATCH"
DB_ERROR: Final = "DB_ERROR"
# StateCriticalError
ORDER_ID_MISMATCH: Final = "ORDER_ID_MISMATCH"
CORRUPT_RECORD: Final = "CORRUPT_RECORD"
DB_CORRUPT: Final = "DB_CORRUPT"
DB_NOT_WRITABLE: Final = "DB_NOT_WRITABLE"


# --------------------------------------------------------------------------- time windows


def day_window(day: date) -> tuple[datetime, datetime]:
    """``[day 00:00 UTC, next day 00:00 UTC)``."""
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return start, start + timedelta(days=1)


def month_window(year: int, month: int) -> tuple[datetime, datetime]:
    """``[first day 00:00 UTC, first day of the next month 00:00 UTC)``.

    Raises:
        NonRetryableError: ``INVALID_ARGUMENT`` if ``month`` is not 1..12 or ``year`` is
            out of range.
    """
    if not 1 <= month <= 12 or not 1 <= year <= 9998:
        raise NonRetryableError(f"invalid month {year}-{month}", code=INVALID_ARGUMENT)
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC) if month == 12 else start.replace(month=month + 1)
    return start, end


def normalized(model: M) -> M:
    """``model`` after the JSON round trip that SQLite storage applies (see ``_codec``).

    Typed fields are unchanged; free-form payloads (``dict[str, Any]``, ``RuleValue``)
    become their JSON value. The result shares no mutable state with ``model``.
    """
    return type(model).model_validate_json(model.model_dump_json())
