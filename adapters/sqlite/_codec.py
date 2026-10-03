"""Conversion between domain models and SQLite rows (sec. 8.3.5, 38).

Storage format:

* ``Decimal`` (prices, money, quantities): TEXT holding the exact decimal string
  (``"123.4500"`` stays ``"123.4500"``). Never REAL.
* ``datetime``: TEXT, UTC, fixed width ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` so that string
  order equals time order and SQL range filters are exact.
* Enums: TEXT with the enum value. ``bool``: INTEGER 0/1. ``int``: INTEGER. ``float``: REAL.
* Structured fields (nested models, tuples, dicts): TEXT column ``<field>_json`` holding
  the pydantic JSON of that field.

Reading always validates through the pydantic model in JSON mode, so every row leaves
the adapter as a validated, immutable domain model; ``sqlite3.Row`` never does.
Typed fields round-trip exactly. Free-form payloads (``dict[str, Any]``, the
``RuleValue`` union) round-trip as their JSON value, e.g. a ``Decimal`` inside
``RiskEvent.sizing`` comes back as a string. The in-memory adapter applies the same
normalization (``adapters.sqlite.semantics.normalized``), so callers never see
adapter-dependent values.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from adapters.sqlite.semantics import CORRUPT_RECORD
from domain.errors import StateCriticalError

__all__ = [
    "SqlValue",
    "format_utc",
    "from_row",
    "json_column",
    "parse_decimal",
    "to_row",
]

SqlValue = str | int | float | None
"""Python values exchanged with sqlite3 (TEXT, INTEGER, REAL, NULL)."""

M = TypeVar("M", bound=BaseModel)


def format_utc(value: datetime) -> str:
    """Fixed-width ISO-8601 UTC text (``2026-10-01T13:30:00.000000Z``).

    Raises:
        ValueError: if ``value`` is naive.
    """
    if value.utcoffset() is None:
        raise ValueError("naive datetimes cannot be stored; use timezone-aware UTC")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def json_column(name: str) -> str:
    """Column that stores the structured field ``name``."""
    return f"{name}_json"


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def to_row(model: BaseModel, *, json_fields: frozenset[str] = frozenset()) -> dict[str, SqlValue]:
    """Columns of ``model``: one per scalar field, ``<field>_json`` per structured field.

    Raises:
        TypeError: if a structured field is not listed in ``json_fields`` (programming
            error in the repository mapping).
    """
    dumped: dict[str, Any] = json.loads(model.model_dump_json())
    row: dict[str, SqlValue] = {}
    for name in type(model).model_fields:
        value = getattr(model, name)
        if name in json_fields:
            row[json_column(name)] = None if value is None else _dumps(dumped[name])
        elif isinstance(value, datetime):
            row[name] = format_utc(value)
        elif isinstance(value, bool):
            row[name] = int(value)
        else:
            scalar = dumped[name]
            if scalar is not None and not isinstance(scalar, str | int | float):
                raise TypeError(f"{type(model).__name__}.{name} needs a JSON column")
            row[name] = scalar
    return row


def from_row(
    model_type: type[M],
    row: Mapping[str, SqlValue],
    *,
    table: str,
    json_fields: frozenset[str] = frozenset(),
) -> M:
    """Rebuild and validate a domain model from its columns (extra columns are ignored).

    Raises:
        StateCriticalError: ``CORRUPT_RECORD`` if the stored row is not a valid model.
    """
    parts: list[str] = []
    for name in model_type.model_fields:
        if name in json_fields:
            raw = row[json_column(name)]
            parts.append(f"{_dumps(name)}:{'null' if raw is None else raw}")
        else:
            parts.append(f"{_dumps(name)}:{_dumps(row[name])}")
    document = "{" + ",".join(parts) + "}"
    try:
        return model_type.model_validate_json(document)
    except ValidationError as exc:
        raise StateCriticalError(
            f"{table}: stored record is not a valid {model_type.__name__} "
            f"({exc.error_count()} validation errors)",
            code=CORRUPT_RECORD,
        ) from exc


def parse_decimal(value: SqlValue, *, table: str, column: str) -> Decimal:
    """Exact ``Decimal`` from a TEXT money column.

    Raises:
        StateCriticalError: ``CORRUPT_RECORD`` if the column does not hold a finite decimal.
    """
    if isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation:
            parsed = None
        if parsed is not None and parsed.is_finite():
            return parsed
    raise StateCriticalError(
        f"{table}.{column}: stored value is not a decimal", code=CORRUPT_RECORD
    )
