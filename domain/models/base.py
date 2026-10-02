"""Base model and shared annotated types for every domain model (sec. 8.4).

Conventions enforced here:

* Every domain model is immutable (``frozen=True``) and rejects unknown fields
  (``extra="forbid"``).
* Prices and money are ``decimal.Decimal``. Floats are rejected for those fields so
  binary rounding never leaks into execution, risk or P&L (sec. 6.1, 57).
* Datetimes are timezone-aware and in UTC. Naive datetimes and non-UTC offsets are
  rejected; accepted values are normalized to ``datetime.UTC`` (sec. 8.3.6, 11.2).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
)

__all__ = [
    "DomainModel",
    "Money",
    "NonEmptyStr",
    "NonNegativeDecimal",
    "PositiveDecimal",
    "Price",
    "Symbol",
    "UtcDatetime",
]


class DomainModel(BaseModel):
    """Base class of all domain models: immutable, strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _ensure_utc(value: datetime) -> datetime:
    """Reject naive or non-UTC datetimes and normalize the tzinfo to ``UTC``."""
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError("datetime must be timezone-aware (UTC); naive datetimes are rejected")
    if offset != timedelta(0):
        raise ValueError(f"datetime must be in UTC, got offset {offset}")
    return value.replace(tzinfo=UTC)


def _reject_float(value: Any) -> Any:
    """Reject binary floats for Decimal fields; accept Decimal, int and numeric strings."""
    if isinstance(value, float):
        raise ValueError("float is not accepted for prices or money; use Decimal or str")
    return value


def _ensure_finite(value: Decimal) -> Decimal:
    """Reject NaN and infinity."""
    if not value.is_finite():
        raise ValueError("decimal value must be finite")
    return value


UtcDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]
"""Timezone-aware datetime whose offset is zero, normalized to ``datetime.UTC``."""

Money = Annotated[Decimal, BeforeValidator(_reject_float), AfterValidator(_ensure_finite)]
"""Finite Decimal amount of any sign (equity, P&L, market value)."""

NonNegativeDecimal = Annotated[Money, Field(ge=0)]
"""Finite Decimal greater than or equal to zero."""

PositiveDecimal = Annotated[Money, Field(gt=0)]
"""Finite Decimal strictly greater than zero (quantities reported by the broker)."""

Price = PositiveDecimal
"""Price: finite Decimal strictly greater than zero."""

Symbol = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9.\-]{0,14}$")]
"""Upper-case ticker symbol, e.g. ``SPY`` or ``BRK.B``."""

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
"""String with at least one character."""
