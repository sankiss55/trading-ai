"""Safe declarative rule language of the deterministic strategy (sec. 13.3-13.5).

A rule is a single comparison ``left <op> right`` evaluated on the last closed bar
(offset ``-1``) or earlier bars. There is **no** ``eval``/``exec`` and no arbitrary
arithmetic: operands are one of

* a series reference with an offset: ``{"series": "ema_fast", "offset": -1}``;
* a constant Decimal: ``{"const": "50"}``;
* a threshold reference into ``strategy.no_trade_thresholds``:
  ``{"param": "rsi_overbought"}``.

The only derived quantities needed by sec. 13.5 are exposed as named series computed by
``domain/strategy/features.py`` (``atr_pct = atr / close`` and
``gap_pct = abs(open_today - close_yesterday) / close_yesterday``), never as expressions.

Series names and timeframes:

* Base names: :data:`SERIES_BASE_NAMES`. A base name resolves to the rule's own
  ``timeframe`` (``primary`` or ``confirmation``).
* The suffix ``_confirm`` (e.g. ``ema_fast_confirm``) always resolves to the
  confirmation timeframe, whatever the rule's ``timeframe``.

YAML format (one list item of ``strategy.entry_rules`` / ``no_trade_rules`` /
``exit_rules``)::

    - rule_id: ENTRY_TREND_01
      timeframe: primary
      left: {series: ema_fast, offset: -1}
      op: ">"
      right: {series: ema_slow, offset: -1}

Numbers must reach the model as ``Decimal``, ``int`` or numeric strings (floats are
rejected, like every Decimal field of the domain). Write ``const: "0.5"`` in YAML.

Evaluation (:func:`evaluate_rule`) returns a :class:`~domain.models.RuleResult` with the
rule id, the boolean result and every value used. If any operand is ``None``
(insufficient data or a pending threshold) the result is ``False`` (sec. 14.3).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Final, Protocol, Self

from pydantic import AfterValidator, Field, StringConstraints, model_validator

from domain.errors import NonRetryableError
from domain.models import DomainModel, Money, RuleResult, RuleValue

__all__ = [
    "CONFIRM_SUFFIX",
    "SERIES_BASE_NAMES",
    "SERIES_NAMES",
    "STRATEGY_CONFIG_INVALID",
    "STRATEGY_INVALID_INPUT",
    "ComparisonOperator",
    "ConstOperand",
    "Identifier",
    "Operand",
    "ParamOperand",
    "RuleId",
    "RuleSpec",
    "RuleTimeframe",
    "SeriesOperand",
    "SeriesSource",
    "StrategyConfigError",
    "StrategyInputError",
    "evaluate_rule",
    "has_missing_values",
    "operand_label",
    "resolve_series_name",
]

STRATEGY_CONFIG_INVALID: Final = "STRATEGY_CONFIG_INVALID"
STRATEGY_INVALID_INPUT: Final = "STRATEGY_INVALID_INPUT"

CONFIRM_SUFFIX: Final = "_confirm"

SERIES_BASE_NAMES: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ema_fast",
    "ema_slow",
    "rsi",
    "atr",
    "volume_avg",
    "atr_pct",
    "gap_pct",
)
"""Series computed per timeframe by ``domain/strategy/features.py``."""

SERIES_NAMES: Final[frozenset[str]] = frozenset(
    SERIES_BASE_NAMES + tuple(name + CONFIRM_SUFFIX for name in SERIES_BASE_NAMES)
)
"""Every series name a rule may reference."""


class StrategyConfigError(NonRetryableError):
    """The strategy configuration is structurally invalid (not merely pending)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=STRATEGY_CONFIG_INVALID)


class StrategyInputError(NonRetryableError):
    """Invalid input data passed to the strategy (programming or adapter error)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=STRATEGY_INVALID_INPUT)


class RuleTimeframe(StrEnum):
    """Timeframe a rule is evaluated on (sec. 13.1, 13.3)."""

    PRIMARY = "primary"
    CONFIRMATION = "confirmation"


class ComparisonOperator(StrEnum):
    """The only operators of the rule language."""

    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="


def _check_series_name(name: str) -> str:
    if name not in SERIES_NAMES:
        raise ValueError(f"unknown series {name!r}; allowed: {sorted(SERIES_NAMES)}")
    return name


Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
"""Lower-case snake_case identifier (threshold names)."""

RuleId = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]
"""Stable upper-case rule identifier, e.g. ``ENTRY_TREND_01``."""


class SeriesOperand(DomainModel):
    """Reference to a computed series at ``offset`` (``-1`` = last closed bar).

    Offsets must be ``<= -1``: index ``0`` or positive would be lookahead (sec. 14.2).
    """

    series: Annotated[str, AfterValidator(_check_series_name)]
    offset: int = Field(default=-1, le=-1)


class ConstOperand(DomainModel):
    """Constant Decimal operand (floats rejected)."""

    const: Money


class ParamOperand(DomainModel):
    """Reference to a named threshold of ``strategy.no_trade_thresholds``."""

    param: Identifier


Operand = SeriesOperand | ConstOperand | ParamOperand
"""Rule operand: exactly one of ``series`` (+ ``offset``), ``const`` or ``param``."""


class RuleSpec(DomainModel):
    """One declarative strategy rule ``left <op> right`` with a stable id (sec. 13.3).

    At least one operand must be a series; comparing two constants/thresholds is
    rejected because its result would not depend on market data.
    """

    rule_id: RuleId
    timeframe: RuleTimeframe
    left: Operand
    op: ComparisonOperator
    right: Operand

    @model_validator(mode="after")
    def _require_series(self) -> Self:
        if not any(isinstance(o, SeriesOperand) for o in (self.left, self.right)):
            raise ValueError(f"rule {self.rule_id}: at least one operand must be a series")
        return self

    def series_operands(self) -> tuple[SeriesOperand, ...]:
        """The series operands of this rule, left first."""
        return tuple(o for o in (self.left, self.right) if isinstance(o, SeriesOperand))

    def referenced_params(self) -> tuple[str, ...]:
        """Threshold names referenced by this rule, left first."""
        return tuple(o.param for o in (self.left, self.right) if isinstance(o, ParamOperand))

    def resolved_series(self) -> tuple[tuple[RuleTimeframe, str], ...]:
        """``(timeframe, base_name)`` for each series operand (see :func:`resolve_series_name`)."""
        return tuple(resolve_series_name(o.series, self.timeframe) for o in self.series_operands())


def resolve_series_name(name: str, rule_timeframe: RuleTimeframe) -> tuple[RuleTimeframe, str]:
    """Resolve a series name to ``(timeframe, base_name)``.

    ``<base>_confirm`` always means the confirmation timeframe; a plain base name means
    the rule's own timeframe.
    """
    _check_series_name(name)
    if name.endswith(CONFIRM_SUFFIX):
        return RuleTimeframe.CONFIRMATION, name[: -len(CONFIRM_SUFFIX)]
    return rule_timeframe, name


class SeriesSource(Protocol):
    """Anything that can return a series value (implemented by ``IndicatorContext``)."""

    def series_value(self, name: str, offset: int, rule_timeframe: RuleTimeframe) -> RuleValue:
        """Value of series ``name`` at ``offset`` (``None`` when unavailable)."""
        ...


def operand_label(operand: Operand) -> str:
    """Stable key used in ``RuleResult.values``: ``ema_fast[-1]``, ``rsi_overbought``, ``50``."""
    if isinstance(operand, SeriesOperand):
        return f"{operand.series}[{operand.offset}]"
    if isinstance(operand, ParamOperand):
        return operand.param
    return str(operand.const)


def _operand_value(
    operand: Operand,
    rule: RuleSpec,
    source: SeriesSource,
    thresholds: Mapping[str, Decimal | None],
) -> RuleValue:
    if isinstance(operand, SeriesOperand):
        return source.series_value(operand.series, operand.offset, rule.timeframe)
    if isinstance(operand, ParamOperand):
        if operand.param not in thresholds:
            raise StrategyConfigError(
                f"rule {rule.rule_id} references unknown threshold {operand.param!r}"
            )
        return thresholds[operand.param]
    return operand.const


def _as_decimal(value: RuleValue) -> Decimal | None:
    """Convert a rule value to Decimal for comparison; ``None`` if unusable."""
    if value is None or isinstance(value, bool | str):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, float):
        return Decimal(str(value)) if math.isfinite(value) else None
    return Decimal(value)


def _compare(left: Decimal, op: ComparisonOperator, right: Decimal) -> bool:
    if op is ComparisonOperator.GT:
        return left > right
    if op is ComparisonOperator.GE:
        return left >= right
    if op is ComparisonOperator.LT:
        return left < right
    if op is ComparisonOperator.LE:
        return left <= right
    if op is ComparisonOperator.EQ:
        return left == right
    return left != right


def evaluate_rule(
    rule: RuleSpec,
    source: SeriesSource,
    thresholds: Mapping[str, Decimal | None],
) -> RuleResult:
    """Evaluate ``rule`` against ``source`` and return its :class:`RuleResult`.

    Float indicator values are compared as ``Decimal(str(value))`` so the comparison is
    deterministic and exact on the recorded value. A ``None`` (or non-finite) operand
    makes the result ``False`` (sec. 14.3); the ``None`` is kept in ``values`` so
    :func:`has_missing_values` can detect it.

    Raises:
        StrategyConfigError: the rule references a threshold absent from ``thresholds``.
    """
    left_value = _operand_value(rule.left, rule, source, thresholds)
    right_value = _operand_value(rule.right, rule, source, thresholds)
    values: dict[str, RuleValue] = {
        operand_label(rule.left): left_value,
        operand_label(rule.right): right_value,
    }
    left_dec = _as_decimal(left_value)
    right_dec = _as_decimal(right_value)
    result = (
        left_dec is not None and right_dec is not None and _compare(left_dec, rule.op, right_dec)
    )
    return RuleResult(rule_id=rule.rule_id, result=result, values=values)


def has_missing_values(result: RuleResult) -> bool:
    """True when the rule could not be evaluated on real data (some value is ``None``)."""
    return any(_as_decimal(value) is None for value in result.values.values())
