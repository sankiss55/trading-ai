"""Configuration loading and validation (sec. 7.3, 7.4, AC-20)."""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal
from pathlib import Path

import pytest

from app.config import (
    AppConfig,
    ConfigError,
    exit_params,
    load_config,
    parse_config,
    pending_backtest_decisions,
    pending_owner_decisions,
    risk_params,
    session_window_params,
)
from domain.models import DataFeed, HoldingMode, Timeframe
from domain.risk.exits import ExitParams
from domain.risk.risk_engine import RiskParams
from domain.strategy.strategy import Strategy, required_warmup_bars

ROOT = Path(__file__).resolve().parents[3]
REAL_CONFIG = ROOT / "config.yaml"
FIXTURE_CONFIG = ROOT / "tests" / "fixtures" / "config.backtest.yaml"
PENDING_CONFIG = ROOT / "tests" / "fixtures" / "config.pending.yaml"
"""The pre-2026-10-02 config.yaml with every OWNER_DECISION still null."""


def _fixture_text() -> str:
    return FIXTURE_CONFIG.read_text(encoding="utf-8")


def _replace(text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


def _parse(text: str) -> AppConfig:
    return parse_config(text, source="test")


# --------------------------------------------------------------------------- real config


def test_real_config_has_no_pending_decision_and_validates() -> None:
    loaded = load_config(REAL_CONFIG)
    assert loaded.config_hash == hashlib.sha256(REAL_CONFIG.read_bytes()).hexdigest()
    config = loaded.config
    assert config.config_version == "2.3.0"
    assert config.universe.liquidity_feed is DataFeed.SIP
    assert config.market_data.feed is DataFeed.IEX
    assert config.strategy_version == "1.0.0"
    assert config.risk_version == "1.0.0"
    assert config.market_data.adjustment == "split"
    assert pending_backtest_decisions(config) == ()
    assert pending_owner_decisions(config) == ()
    # the domain strategy accepts the owner rules (no pending parameter, warm-up covered)
    Strategy(config.strategy, strategy_version=config.strategy_version)
    assert config.market_data.history_warmup_bars >= required_warmup_bars(config.strategy)


def test_real_config_risk_values_are_fractions() -> None:
    risk = load_config(REAL_CONFIG).config.risk
    assert risk.risk_per_trade_pct == Decimal("0.005")
    assert risk.max_daily_loss_pct == Decimal("0.02")
    assert risk.max_drawdown_pct == Decimal("0.10")
    assert risk.max_aggregate_open_risk_pct == Decimal("0.015")


# --------------------------------------------------------------------------- pending config


def test_pending_config_loads_and_lists_pending_decisions() -> None:
    loaded = load_config(PENDING_CONFIG)
    assert loaded.config_hash == hashlib.sha256(PENDING_CONFIG.read_bytes()).hexdigest()
    pending = pending_owner_decisions(loaded.config)
    for expected in (
        "universe.whitelist",
        "universe.liquidity_feed",
        "market_data.feed",
        "market_data.adjustment",
        "strategy.holding_mode",
        "strategy.primary_timeframe",
        "strategy.entry_rules",
        "strategy.indicators.atr_period",
        "strategy.exit.stop_atr_multiplier",
        "risk.risk_per_trade_pct",
        "risk.slippage_buffer_bps",
        "backtest.start_date",
        "backtest.min_expectancy_r",
        "paper.min_paper_trading_days",
        "backups.frequency",
    ):
        assert expected in pending
    assert len(pending) == len(set(pending))
    # null is a valid owner choice for these: never reported as pending
    assert "strategy.confirmation_timeframe" not in pending
    assert "strategy.exit.time_stop_bars" not in pending
    # VERIFICAR keys are not owner decisions
    assert "ai.effort" not in pending


def test_pending_config_backtest_subset_excludes_continuation_thresholds() -> None:
    config = load_config(PENDING_CONFIG).config
    subset = pending_backtest_decisions(config)
    assert subset
    assert set(subset) <= set(pending_owner_decisions(config))
    assert "strategy.entry_rules" in subset
    assert "backtest.start_date" in subset
    assert "market_data.adjustment" in subset
    assert "backtest.min_expectancy_r" not in subset
    assert "paper.min_paper_trading_days" not in subset
    assert "universe.min_price" not in subset


def test_fixture_config_has_no_pending_decision() -> None:
    config = load_config(FIXTURE_CONFIG).config
    assert pending_owner_decisions(config) == ()
    assert config.strategy.holding_mode is HoldingMode.INTRADAY
    assert config.strategy.primary_timeframe is Timeframe.MIN_5


def test_real_config_uses_the_claude_cli_provider() -> None:
    ai = load_config(REAL_CONFIG).config.ai
    assert ai.provider == "claude_cli"
    assert ai.cli_command == "claude"
    assert ai.cli_max_concurrency == 1


# --------------------------------------------------------------------------- ai provider


def test_anthropic_api_provider_is_rejected_at_startup() -> None:
    text = _replace(_fixture_text(), 'provider: "claude_cli"', 'provider: "anthropic_api"')
    with pytest.raises(ConfigError, match="owner decision") as info:
        _parse(text)
    assert "docs/DECISIONS.md" in str(info.value)


@pytest.mark.parametrize("key", ["provider", "cli_command", "cli_max_concurrency"])
def test_missing_ai_cli_key_is_rejected(key: str) -> None:
    lines = [
        line
        for line in _fixture_text().splitlines(keepends=True)
        if not line.startswith(f"  {key}:")
    ]
    with pytest.raises(ConfigError, match=re.escape(f"ai.{key}")):
        _parse("".join(lines))


@pytest.mark.parametrize(
    "value",
    ['""', "null", '"   "'],
)
def test_empty_cli_command_is_rejected(value: str) -> None:
    text = _replace(_fixture_text(), 'cli_command: "claude"', f"cli_command: {value}")
    with pytest.raises(ConfigError, match=re.escape("ai.cli_command")):
        _parse(text)


@pytest.mark.parametrize("value", ["0", "2", "true", "1.0", '"1"', "null"])
def test_cli_concurrency_other_than_one_is_rejected(value: str) -> None:
    text = _replace(_fixture_text(), "cli_max_concurrency: 1", f"cli_max_concurrency: {value}")
    with pytest.raises(ConfigError, match=re.escape("ai.cli_max_concurrency")):
        _parse(text)


def test_unknown_provider_is_rejected() -> None:
    text = _replace(_fixture_text(), 'provider: "claude_cli"', 'provider: "openai"')
    with pytest.raises(ConfigError, match=re.escape("ai.provider")):
        _parse(text)


# --------------------------------------------------------------------------- YAML loader


def test_unquoted_floats_are_parsed_as_exact_decimals() -> None:
    config = _parse(_fixture_text())
    assert config.strategy.no_trade_thresholds["max_atr_pct"] == Decimal("0.02")
    assert isinstance(config.strategy.exit.stop_atr_multiplier, Decimal)
    assert config.strategy.exit.stop_atr_multiplier == Decimal("1.5")
    assert config.risk.max_drawdown_pct == Decimal("0.20")
    assert config.ai.pricing.input_per_mtok_usd == Decimal("2.00")


def test_non_finite_numbers_are_rejected() -> None:
    text = _replace(_fixture_text(), "max_atr_pct: 0.02", "max_atr_pct: .inf")
    with pytest.raises(ConfigError, match="non-finite"):
        _parse(text)


def test_duplicate_keys_are_rejected() -> None:
    text = _replace(_fixture_text(), "  min_qty: 1\n", "  min_qty: 1\n  min_qty: 2\n")
    with pytest.raises(ConfigError, match="duplicate key"):
        _parse(text)


def test_malformed_yaml_and_non_mapping_fail() -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        _parse("risk: [unclosed")
    with pytest.raises(ConfigError, match="mapping"):
        _parse("- just\n- a list\n")


def test_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")


# --------------------------------------------------------------------------- invalid values


@pytest.mark.parametrize(
    ("old", "new", "location"),
    [
        ("risk_per_trade_pct: 0.01", "risk_per_trade_pct: 1.5", "risk.risk_per_trade_pct"),
        ("max_total_exposure_pct: 1.0", "max_total_exposure_pct: 2", "max_total_exposure_pct"),
        ("max_positions: 2", 'max_positions: "two"', "risk.max_positions"),
        ("max_positions: 2", "max_positions: 2.5", "risk.max_positions"),
        ("cooldown_bars: 3", "cooldown_bars: -1", "strategy.cooldown_bars"),
        ('  feed: "iex"', '  feed: "nasdaq"', "market_data.feed"),
        ('liquidity_feed: "sip"', 'liquidity_feed: "nasdaq"', "universe.liquidity_feed"),
        ('adjustment: "split"', 'adjustment: "none"', "market_data.adjustment"),
        ("allow_premarket: false", "allow_premarket: true", "session.allow_premarket"),
        ("include_news: false", "include_news: 1", "ai.include_news"),
        ('daily_summary_time_et: "16:30"', 'daily_summary_time_et: "25:00"', "HH:MM"),
        ("min_price: 1\n", "min_price: 20000\n", "min_price must be < "),
        ("min_tp_distance_ticks: 2", "min_tp_distance_ticks: 0", "min_tp_distance_ticks"),
        ("start_date: 2025-01-02", "start_date: 2025-04-30", "start_date must be before"),
        ("out_of_sample_start: 2025-03-01", "out_of_sample_start: 2024-12-01", "after start"),
        ('whitelist: ["SYNTH"]', 'whitelist: ["synth"]', "universe.whitelist"),
        ("  queue_max_size: 100\n", "  queue_max_size: 100\n  unknown_key: 1\n", "unknown_key"),
    ],
)
def test_invalid_values_fail_at_startup(old: str, new: str, location: str) -> None:
    with pytest.raises(ConfigError) as info:
        _parse(_replace(_fixture_text(), old, new))
    assert location in str(info.value)


def test_missing_liquidity_feed_key_is_rejected() -> None:
    text = _replace(_fixture_text(), '  liquidity_feed: "sip"\n', "")
    with pytest.raises(ConfigError, match=re.escape("universe.liquidity_feed")):
        _parse(text)


def test_null_liquidity_feed_is_a_pending_decision() -> None:
    text = _replace(_fixture_text(), 'liquidity_feed: "sip"', "liquidity_feed: null")
    config = _parse(text)
    assert pending_owner_decisions(config) == ("universe.liquidity_feed",)
    # the backtest does not apply the liquidity filter yet: not a backtest blocker
    assert pending_backtest_decisions(config) == ()


def test_missing_adjustment_key_is_rejected() -> None:
    text = _replace(_fixture_text(), '  adjustment: "split"\n', "")
    with pytest.raises(ConfigError, match=re.escape("market_data.adjustment")):
        _parse(text)


def test_missing_section_fails() -> None:
    text = _fixture_text()
    start = text.index("backups:")
    with pytest.raises(ConfigError, match="backups"):
        _parse(text[:start])


# --------------------------------------------------------------------------- cross rules


def test_intraday_requires_flatten_minutes() -> None:
    text = _replace(
        _fixture_text(),
        "flatten_minutes_before_close: 10",
        "flatten_minutes_before_close: null",
    )
    with pytest.raises(ConfigError, match="flatten_minutes_before_close"):
        _parse(text)


def test_swing_does_not_require_flatten_minutes() -> None:
    text = _replace(_fixture_text(), 'holding_mode: "intraday"', 'holding_mode: "swing"')
    text = _replace(text, "flatten_minutes_before_close: 10", "flatten_minutes_before_close: null")
    assert _parse(text).strategy.holding_mode is HoldingMode.SWING


def test_limit_entries_require_offset() -> None:
    text = _replace(_fixture_text(), 'entry_order_type: "market"', 'entry_order_type: "limit"')
    with pytest.raises(ConfigError, match="limit_entry_offset_bps"):
        _parse(text)
    text = _replace(text, "limit_entry_offset_bps: null", "limit_entry_offset_bps: 5")
    assert _parse(text).execution.limit_entry_offset_bps == Decimal(5)


def test_history_warmup_must_cover_the_longest_indicator() -> None:
    text = _replace(_fixture_text(), "history_warmup_bars: 100", "history_warmup_bars: 10")
    with pytest.raises(ConfigError, match="history_warmup_bars"):
        _parse(text)


DAILY_CONFIG = ROOT / "tests" / "fixtures" / "config.daily.yaml"


def _daily_text() -> str:
    return DAILY_CONFIG.read_text(encoding="utf-8")


def test_daily_swing_config_loads_without_pending_decisions() -> None:
    config = load_config(DAILY_CONFIG).config
    assert config.strategy.primary_timeframe is Timeframe.DAY_1
    assert config.strategy.holding_mode is HoldingMode.SWING
    assert config.strategy.confirmation_timeframe is None
    assert config.strategy.indicators.sma_short_period == 5
    assert config.strategy.indicators.sma_long_period == 10
    assert pending_backtest_decisions(config) == ()
    Strategy(config.strategy, strategy_version=config.strategy_version)


def test_daily_primary_requires_swing() -> None:
    text = _replace(_daily_text(), 'holding_mode: "swing"', 'holding_mode: "intraday"')
    text = _replace(text, "flatten_minutes_before_close: null", "flatten_minutes_before_close: 5")
    with pytest.raises(ConfigError, match=r"1Day requires strategy.holding_mode = swing"):
        _parse(text)
    pending = _replace(_daily_text(), 'holding_mode: "swing"', "holding_mode: null")
    with pytest.raises(ConfigError, match=r"1Day requires strategy.holding_mode = swing"):
        _parse(pending)


def test_daily_primary_requires_no_confirmation_timeframe() -> None:
    text = _replace(_daily_text(), "confirmation_timeframe: null", 'confirmation_timeframe: "1Day"')
    with pytest.raises(ConfigError, match=r"1Day requires strategy.confirmation_timeframe = null"):
        _parse(text)


def test_daily_session_window_keys_stay_required() -> None:
    text = _replace(_daily_text(), "no_entry_first_minutes: 15", "no_entry_first_minutes: null")
    assert "session.no_entry_first_minutes" in pending_backtest_decisions(_parse(text))


def test_optional_indicator_keys_default_to_unused() -> None:
    config = _parse(_fixture_text())
    assert config.strategy.indicators.sma_short_period is None
    assert config.strategy.indicators.sma_long_period is None
    assert required_warmup_bars(config.strategy) == 21  # unchanged: ema_slow 21


def test_rule_on_an_unset_optional_indicator_is_a_config_error() -> None:
    text = _replace(_daily_text(), "sma_long_period: 10", "sma_long_period: null")
    text = _replace(
        text,
        "  no_trade_rules: []",
        "  no_trade_rules:\n"
        '    - {rule_id: NOTRADE_TREND, timeframe: primary, left: {series: close}, op: "<",'
        " right: {series: sma_long}}",
    )
    with pytest.raises(ConfigError, match=r"NOTRADE_TREND -> strategy.indicators.sma_long_period"):
        _parse(text)


def test_optional_indicator_periods_count_for_the_warm_up() -> None:
    text = _replace(_daily_text(), "sma_long_period: 10", "sma_long_period: 30")
    with pytest.raises(ConfigError, match="history_warmup_bars"):
        _parse(text)
    text = _replace(text, "history_warmup_bars: 20", "history_warmup_bars: 30")
    assert required_warmup_bars(_parse(text).strategy) == 30


def test_reversal_enabled_without_exit_rules_is_pending() -> None:
    text = _fixture_text()
    start = text.index("  exit_rules:\n")
    end = text.index("  no_trade_thresholds:")
    config = _parse(text[:start] + "  exit_rules: null\n" + text[end:])
    assert pending_owner_decisions(config) == ("strategy.exit_rules",)
    assert pending_backtest_decisions(config) == ("strategy.exit_rules",)


# --------------------------------------------------------------------------- mappings


def test_domain_parameter_mappings() -> None:
    config = _parse(_fixture_text())
    assert isinstance(risk_params(config), RiskParams)
    assert risk_params(config).slippage_buffer_bps == Decimal(5)
    exits = exit_params(config)
    assert exits == ExitParams(
        stop_atr_multiplier=Decimal("1.5"),
        take_profit_r_multiple=Decimal("2.0"),
        min_tp_distance_ticks=2,
    )
    window = session_window_params(config)
    assert window.no_entry_first_minutes == 15
    assert window.no_entry_last_minutes == 30
    assert window.holding_mode is HoldingMode.INTRADAY
    assert window.flatten_minutes_before_close == 10
