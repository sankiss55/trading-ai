"""Configuration loading and validation (master spec sec. 7.3, 7.4).

* ``config.yaml`` is parsed with :class:`DecimalSafeLoader`: a ``yaml.SafeLoader`` whose
  YAML floats become :class:`~decimal.Decimal` (from their exact text), so unquoted
  values such as ``max_atr_pct: 0.02`` or a rule ``const: 50.5`` reach the domain
  without binary rounding (floats are rejected by every Decimal field of the domain).
  Non-finite YAML floats (``.inf``, ``.nan``) and duplicate mapping keys are rejected.
* :class:`AppConfig` mirrors the whole 7.3 template. Every section forbids unknown keys.
  An invalid type or an out-of-range value raises :class:`ConfigError` at startup; there
  is never a silent fallback (sec. 7.4.2).
* ``OWNER_DECISION`` parameters may be ``null``: the system may start in observation
  mode, but :func:`pending_owner_decisions` lists every missing value so that trading
  cannot be enabled (sec. 7.4.3, AC-20). No default is ever substituted.
* Mandatory cross rules (sec. 7.4.4) are enforced by :class:`AppConfig`:
  ``holding_mode = intraday`` requires ``flatten_minutes_before_close``;
  ``entry_order_type = limit`` requires ``limit_entry_offset_bps``;
  ``history_warmup_bars >= required_warmup_bars(strategy)`` (computed from the periods
  and offsets that are already set; no extra margin is invented). The whitelist
  tradability check needs the broker and belongs to startup (Phase 2).
  ``ai.provider = "anthropic_api"`` is rejected (owner decision 2026-10-02: the AI
  filter uses the Claude Code CLI session, see ``docs/DECISIONS.md``).
* :func:`load_config` returns the validated config together with ``config_hash`` (the
  sha256 of the file bytes) to be recorded with ``config_version`` (sec. 7.4.5).

``.env`` loading (secrets, ``APP_ENV``) is not part of this module yet (Phase 2).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)
from yaml.nodes import MappingNode, ScalarNode

from domain.errors import NonRetryableError
from domain.market.session import SessionWindowParams
from domain.models import DataFeed, HoldingMode, Money, NonEmptyStr, Symbol
from domain.risk.exits import ExitParams
from domain.risk.risk_engine import Bps, Fraction, RiskParams, pending_risk_params
from domain.strategy.strategy import StrategyParams, pending_strategy_params, required_warmup_bars

__all__ = [
    "AI_PROVIDER_ANTHROPIC_API",
    "AI_PROVIDER_CLAUDE_CLI",
    "CONFIG_INVALID",
    "OWNER_DECISION_PATHS",
    "AISection",
    "AppConfig",
    "BacktestSection",
    "BackupsSection",
    "ConfigError",
    "DecimalSafeLoader",
    "ExecutionSection",
    "LoadedConfig",
    "MarketDataSection",
    "NotificationsSection",
    "PaperSection",
    "RetentionSection",
    "SessionSection",
    "SystemSection",
    "UniverseSection",
    "config_hash",
    "exit_params",
    "load_config",
    "parse_config",
    "pending_backtest_decisions",
    "pending_owner_decisions",
    "risk_params",
    "session_window_params",
]

CONFIG_INVALID: Final = "CONFIG_INVALID"


class ConfigError(NonRetryableError):
    """``config.yaml`` is unreadable, malformed or invalid: the process must not start."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=CONFIG_INVALID)


# --------------------------------------------------------------------------- YAML loader


class DecimalSafeLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that parses YAML floats as ``Decimal`` and rejects duplicate keys."""


def _construct_decimal(loader: yaml.SafeLoader, node: ScalarNode) -> Decimal:
    text = str(loader.construct_scalar(node)).replace("_", "")
    if text.lower().lstrip("+-") in (".inf", ".nan"):
        raise yaml.constructor.ConstructorError(
            None, None, f"non-finite number {text!r} is not allowed", node.start_mark
        )
    try:
        value = Decimal(text)
    except InvalidOperation as exc:  # pragma: no cover - YAML already matched a float
        raise yaml.constructor.ConstructorError(
            None, None, f"invalid number {text!r}", node.start_mark
        ) from exc
    return value


def _construct_mapping(loader: yaml.SafeLoader, node: MappingNode) -> dict[Hashable, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Hashable, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


DecimalSafeLoader.add_constructor("tag:yaml.org,2002:float", _construct_decimal)
DecimalSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


# --------------------------------------------------------------------------- sections


class _Section(BaseModel):
    """Base of every config section: immutable and strict about unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveMoney = Annotated[Money, Field(gt=0)]
NonNegativeMoney = Annotated[Money, Field(ge=0)]
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class SystemSection(_Section):
    """``system``: paths and process-level technical settings."""

    timezone_internal: Literal["UTC"]
    timezone_market: NonEmptyStr
    db_path: NonEmptyStr
    log_dir: NonEmptyStr
    control_stop_file: NonEmptyStr
    control_poll_seconds: PositiveMoney
    max_clock_skew_seconds: PositiveMoney
    queue_max_size: PositiveInt


class UniverseSection(_Section):
    """``universe``: tradable symbols and liquidity filters (sec. 12)."""

    whitelist: Annotated[tuple[Symbol, ...], Field(min_length=1)] | None
    blacklist: tuple[Symbol, ...]
    min_price: PositiveMoney | None
    max_price: PositiveMoney | None
    min_avg_daily_volume: NonNegativeInt | None
    avg_volume_lookback_days: PositiveInt
    max_spread_bps: NonNegativeMoney | None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.whitelist is not None:
            duplicates = sorted({s for s in self.whitelist if self.whitelist.count(s) > 1})
            if duplicates:
                raise ValueError(f"duplicate whitelist symbols: {duplicates}")
            overlap = sorted(set(self.whitelist) & set(self.blacklist))
            if overlap:
                raise ValueError(f"symbols both whitelisted and blacklisted: {overlap}")
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price >= self.max_price
        ):
            raise ValueError("universe.min_price must be < universe.max_price")
        return self


class MarketDataSection(_Section):
    """``market_data`` (sec. 10).

    ``adjustment`` is the corporate-action price adjustment of historical bars
    (sec. 44, OWNER_DECISION): backtest and live must use the same convention.
    """

    feed: DataFeed | None
    adjustment: Literal["raw", "split", "dividend", "all"] | None
    max_bar_age_seconds: PositiveInt
    bar_close_grace_seconds: NonNegativeMoney
    history_warmup_bars: PositiveInt


class SessionSection(_Section):
    """``session`` (sec. 5.5, 11.4). Pre/after-hours are fixed to ``false`` in the MVP."""

    allow_premarket: Literal[False]
    allow_after_hours: Literal[False]
    no_entry_first_minutes: NonNegativeInt | None
    no_entry_last_minutes: NonNegativeInt | None


class ExecutionSection(_Section):
    """``execution`` (sec. 23)."""

    entry_order_type: Literal["market", "limit"] | None
    limit_entry_offset_bps: Bps | None
    entry_timeout_seconds: PositiveInt
    max_order_rejections_per_day: NonNegativeInt
    flatten_retry_seconds: PositiveInt
    reconcile_interval_seconds: PositiveInt


class AIPricingSection(_Section):
    """``ai.pricing`` (VERIFICAR values, used only to estimate cost)."""

    input_per_mtok_usd: NonNegativeMoney | None
    output_per_mtok_usd: NonNegativeMoney | None
    cache_write_per_mtok_usd: NonNegativeMoney | None
    cache_read_per_mtok_usd: NonNegativeMoney | None


AI_PROVIDER_CLAUDE_CLI: Final = "claude_cli"
AI_PROVIDER_ANTHROPIC_API: Final = "anthropic_api"


class AISection(_Section):
    """``ai`` (sec. 17, 34). Not used before Phase 8, but validated from day one.

    ``provider`` is an owner decision (2026-10-02, ``docs/DECISIONS.md``): the veto filter
    calls the Claude Code CLI headlessly with the machine's logged-in subscription session,
    so no API key exists. ``"anthropic_api"`` is accepted by the schema (it is the spec's
    original provider) but rejected at startup: this project fails closed on it.
    ``cli_max_concurrency`` is fixed to 1 (one CLI call at a time).
    """

    provider: Literal["claude_cli", "anthropic_api"]
    cli_command: NonEmptyStr
    cli_max_concurrency: Annotated[StrictInt, Field(ge=1, le=1)]
    model_id: NonEmptyStr
    effort: NonEmptyStr | None
    max_output_tokens: PositiveInt
    request_timeout_seconds: PositiveMoney
    max_retries: NonNegativeInt
    on_unavailable: Literal["block", "rules_only"]
    snapshot_bars_primary: PositiveInt
    snapshot_bars_confirmation: NonNegativeInt
    recent_outcomes_limit: NonNegativeInt
    max_rationale_chars: PositiveInt
    benchmark_symbol: Symbol
    include_news: StrictBool | None
    max_headlines: NonNegativeInt
    max_calls_per_day: NonNegativeInt | None
    max_cost_usd_per_day: NonNegativeMoney | None
    max_cost_usd_per_month: NonNegativeMoney | None
    pricing: AIPricingSection

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.cli_command.strip():
            raise ValueError("ai.cli_command must not be blank")
        if self.provider == AI_PROVIDER_ANTHROPIC_API:
            raise ValueError(
                'ai.provider = "anthropic_api" is not used in this project by owner '
                'decision (2026-10-02); use "claude_cli" (see docs/DECISIONS.md)'
            )
        return self


class NotificationsSection(_Section):
    """``notifications`` (sec. 40)."""

    policy: Literal["every_trade", "critical_plus_daily", "daily_only"] | None
    daily_summary_time_et: NonEmptyStr
    error_digest_minutes: PositiveInt

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not _HHMM.match(self.daily_summary_time_et):
            raise ValueError("notifications.daily_summary_time_et must be HH:MM (24h)")
        return self


class RetentionSection(_Section):
    """``retention`` in days."""

    market_bars_days: PositiveInt | None
    ai_snapshots_days: PositiveInt | None
    logs_days: PositiveInt | None


class BacktestSection(_Section):
    """``backtest`` (sec. 45). Thresholds are compared by the sec. 45.4 check."""

    start_date: date | None
    end_date: date | None
    out_of_sample_start: date | None
    walk_forward_train_months: PositiveInt | None
    walk_forward_test_months: PositiveInt | None
    min_expectancy_r: Money | None
    min_profit_factor: PositiveMoney | None
    max_drawdown_pct: Fraction | None
    min_trades_out_of_sample: NonNegativeInt | None

    @model_validator(mode="after")
    def _check(self) -> Self:
        start, end, oos = self.start_date, self.end_date, self.out_of_sample_start
        if start is not None and end is not None and start >= end:
            raise ValueError("backtest.start_date must be before backtest.end_date")
        if oos is not None:
            if start is not None and oos <= start:
                raise ValueError("backtest.out_of_sample_start must be after start_date")
            if end is not None and oos > end:
                raise ValueError("backtest.out_of_sample_start must be <= end_date")
        return self


class PaperSection(_Section):
    """``paper`` (sec. 46.4, 47)."""

    min_paper_trading_days: PositiveInt | None
    min_shadow_signals: NonNegativeInt | None
    min_expectancy_improvement_r: Money | None
    min_valid_ai_response_rate: Fraction | None


class BackupsSection(_Section):
    """``backups`` (sec. 58.2)."""

    frequency: NonEmptyStr | None
    retention_days: PositiveInt | None


# --------------------------------------------------------------------------- root


class AppConfig(_Section):
    """The whole ``config.yaml`` (sec. 7.3), validated at startup (sec. 7.4).

    ``strategy`` is the domain :class:`~domain.strategy.strategy.StrategyParams` and
    ``risk`` the domain :class:`~domain.risk.risk_engine.RiskParams` (identical keys),
    so the domain validates its own parameters; ``strategy.exit`` is additionally mapped
    to :class:`~domain.risk.exits.ExitParams` (see :func:`exit_params`).
    """

    config_version: NonEmptyStr
    strategy_version: NonEmptyStr
    risk_version: NonEmptyStr
    prompt_version: NonEmptyStr
    system: SystemSection
    universe: UniverseSection
    market_data: MarketDataSection
    session: SessionSection
    strategy: StrategyParams
    risk: RiskParams
    execution: ExecutionSection
    ai: AISection
    notifications: NotificationsSection
    retention: RetentionSection
    backtest: BacktestSection
    paper: PaperSection
    backups: BackupsSection

    @model_validator(mode="after")
    def _cross_rules(self) -> Self:
        # Building ExitParams validates the risk-side ranges of strategy.exit
        # (e.g. min_tp_distance_ticks >= 1), so they fail at startup, not at the first trade.
        exit_params(self)
        strategy = self.strategy
        if strategy.holding_mode is HoldingMode.INTRADAY and (
            strategy.flatten_minutes_before_close is None
        ):
            raise ValueError(
                "strategy.holding_mode = intraday requires "
                "strategy.flatten_minutes_before_close (sec. 7.4.4)"
            )
        if self.execution.entry_order_type == "limit" and (
            self.execution.limit_entry_offset_bps is None
        ):
            raise ValueError(
                "execution.entry_order_type = limit requires "
                "execution.limit_entry_offset_bps (sec. 7.4.4)"
            )
        needed = required_warmup_bars(strategy)
        if self.market_data.history_warmup_bars < needed:
            raise ValueError(
                f"market_data.history_warmup_bars={self.market_data.history_warmup_bars} "
                f"is below the {needed} bars the strategy indicators need (sec. 7.4.4)"
            )
        return self


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    """A validated configuration and the identity of the file it came from.

    Attributes:
        config: The validated configuration.
        config_hash: ``sha256`` hex digest of the exact file bytes (sec. 7.4.5).
        path: File the configuration was read from.
    """

    config: AppConfig
    config_hash: str
    path: Path


# --------------------------------------------------------------------------- loading


def config_hash(content: bytes) -> str:
    """``sha256`` hex digest of the raw configuration bytes."""
    return hashlib.sha256(content).hexdigest()


def _format_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"  {location}: {item['msg']}")
    return "\n".join(lines)


def parse_config(text: str | bytes, *, source: str = "<config>") -> AppConfig:
    """Parse and validate configuration text.

    Raises:
        ConfigError: malformed YAML, a non-mapping document, or any validation failure.
    """
    try:
        data = yaml.load(text, Loader=DecimalSafeLoader)  # SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: the document must be a mapping")
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"{source}: invalid configuration:\n{_format_validation_error(exc)}"
        ) from exc


def load_config(path: Path) -> LoadedConfig:
    """Read, validate and hash ``config.yaml``.

    Raises:
        ConfigError: the file cannot be read or is invalid (startup must fail).
    """
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {path}: {exc}") from exc
    config = parse_config(content, source=str(path))
    return LoadedConfig(config=config, config_hash=config_hash(content), path=path)


# --------------------------------------------------------------------------- domain mappings


def risk_params(config: AppConfig) -> RiskParams:
    """The ``risk`` section as the domain :class:`RiskParams`."""
    return config.risk


def exit_params(config: AppConfig) -> ExitParams:
    """``strategy.exit`` mapped to the risk engine :class:`ExitParams` (sec. 16).

    Raises:
        pydantic.ValidationError: a value valid for the strategy is out of the risk
            engine's range (e.g. ``min_tp_distance_ticks = 0``).
    """
    section = config.strategy.exit
    return ExitParams(
        stop_atr_multiplier=section.stop_atr_multiplier,
        take_profit_r_multiple=section.take_profit_r_multiple,
        min_tp_distance_ticks=section.min_tp_distance_ticks,
    )


def session_window_params(config: AppConfig) -> SessionWindowParams:
    """Parameters of :func:`domain.market.session.compute_session_windows`."""
    return SessionWindowParams(
        no_entry_first_minutes=config.session.no_entry_first_minutes,
        no_entry_last_minutes=config.session.no_entry_last_minutes,
        holding_mode=config.strategy.holding_mode,
        flatten_minutes_before_close=config.strategy.flatten_minutes_before_close,
    )


# --------------------------------------------------------------------------- pending decisions

OWNER_DECISION_PATHS: Final[tuple[str, ...]] = (
    "universe.whitelist",
    "universe.min_price",
    "universe.max_price",
    "universe.min_avg_daily_volume",
    "universe.max_spread_bps",
    "market_data.feed",
    "market_data.adjustment",
    "session.no_entry_first_minutes",
    "session.no_entry_last_minutes",
    "strategy.holding_mode",
    "strategy.exit.stop_method",
    "execution.entry_order_type",
    "ai.include_news",
    "ai.max_calls_per_day",
    "ai.max_cost_usd_per_day",
    "ai.max_cost_usd_per_month",
    "notifications.policy",
    "retention.market_bars_days",
    "retention.ai_snapshots_days",
    "retention.logs_days",
    "backtest.start_date",
    "backtest.end_date",
    "backtest.out_of_sample_start",
    "backtest.walk_forward_train_months",
    "backtest.walk_forward_test_months",
    "backtest.min_expectancy_r",
    "backtest.min_profit_factor",
    "backtest.max_drawdown_pct",
    "backtest.min_trades_out_of_sample",
    "paper.min_paper_trading_days",
    "paper.min_shadow_signals",
    "paper.min_expectancy_improvement_r",
    "paper.min_valid_ai_response_rate",
    "backups.frequency",
    "backups.retention_days",
)
"""OWNER_DECISION keys that must always be set (unconditional).

Not listed because ``null`` is a valid owner choice: ``strategy.confirmation_timeframe``
(``null`` = unused; pending only when a rule needs it, see ``pending_strategy_params``)
and ``strategy.exit.time_stop_bars`` (``null`` = no time stop). VERIFICAR keys
(``ai.effort``, cache pricing) are not owner decisions.
"""

_BACKTEST_PREFIXES: Final[tuple[str, ...]] = (
    "universe.whitelist",
    "market_data.",
    "session.",
    "strategy.",
    "risk.",
    "execution.",
    "backtest.start_date",
    "backtest.end_date",
    "backtest.out_of_sample_start",
    "backtest.walk_forward_",
)
"""Pending decisions that prevent running a backtest at all. The sec. 45.4 thresholds
are excluded: without them the backtest runs and reports ``PENDING_OWNER_DECISION``."""


def _get_path(config: AppConfig, path: str) -> object:
    value: object = config
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def pending_owner_decisions(config: AppConfig) -> tuple[str, ...]:
    """Dotted paths of every required OWNER_DECISION that is still ``null`` (AC-20).

    Includes the unconditional keys of :data:`OWNER_DECISION_PATHS`, the strategy
    parameters reported by ``pending_strategy_params`` (conditional on the active
    rules), the risk-engine exit parameters, every ``risk`` parameter, and the
    conditional keys ``strategy.flatten_minutes_before_close`` (intraday) and
    ``execution.limit_entry_offset_bps`` (limit entries). Order follows config.yaml.
    """
    pending: list[str] = []
    for path in OWNER_DECISION_PATHS:
        if _get_path(config, path) is None:
            pending.append(path)
        if path == "strategy.holding_mode":
            if (
                config.strategy.holding_mode is HoldingMode.INTRADAY
                and config.strategy.flatten_minutes_before_close is None
            ):  # pragma: no cover - rejected by the cross rule at load time
                pending.append("strategy.flatten_minutes_before_close")
            pending.extend(pending_strategy_params(config.strategy))
            exits = exit_params(config)
            pending.extend(
                f"strategy.exit.{name}"
                for name in ExitParams.model_fields
                if getattr(exits, name) is None
            )
            pending.extend(f"risk.{name}" for name in pending_risk_params(config.risk))
        if (
            path == "execution.entry_order_type"
            and config.execution.entry_order_type == "limit"
            and config.execution.limit_entry_offset_bps is None
        ):  # pragma: no cover - rejected by the cross rule at load time
            pending.append("execution.limit_entry_offset_bps")
    return _unique(pending)


def pending_backtest_decisions(config: AppConfig) -> tuple[str, ...]:
    """Subset of :func:`pending_owner_decisions` without which a backtest cannot run."""
    return tuple(
        path for path in pending_owner_decisions(config) if path.startswith(_BACKTEST_PREFIXES)
    )
