"""Research protocol and the 7 research configurations of family v2 (load and validate only).

Nothing here runs a configuration: the registered configs are only loaded and checked.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from app.config import load_config, pending_owner_decisions
from backtest.registry import TRIAL_BUDGET, V1_TRIAL_COUNT, BacktestResearchError, load_hypotheses
from backtest.research_cli import (
    DEV_BASE_METRICS,
    LOCKBOX_METRICS,
    dev_metric_names,
    dev_windows,
    load_protocol,
    load_study,
    validate_pass_rule,
)
from domain.models import DataFeed, HoldingMode, Timeframe
from domain.strategy.rules import ConstOperand, Operand, RuleSpec, SeriesOperand
from domain.strategy.strategy import required_warmup_bars

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "protocol.yaml"
HYPOTHESES = ROOT / "research" / "hypotheses"
CONFIGS = ROOT / "research" / "configs"
OWNER = load_config(ROOT / "config.yaml").config

# (entry rules, exit rules, time stop) per hypothesis, as (left, op, right) text.
EXPECTED_RULES: dict[str, tuple[set[str], set[str], int | None]] = {
    "mr_a1_ibs": ({"ibs < 0.2"}, {"ibs > 0.8"}, 5),
    "mr_a2_ibs_sma200": ({"ibs < 0.2", "close > sma_long"}, {"ibs > 0.8"}, 5),
    "mr_b1_rsi2_lt5": ({"close > sma_long", "rsi < 5"}, {"close > sma_short"}, 10),
    "mr_b2_rsi2_lt10": ({"close > sma_long", "rsi < 10"}, {"close > sma_short"}, 10),
    "mr_b3_rsi2_lt15": ({"close > sma_long", "rsi < 15"}, {"close > sma_short"}, 10),
    "tf_t1_sma200": ({"close > sma_long"}, {"close < sma_long"}, None),
    "tf_t2_sma210": ({"close > sma_long"}, {"close < sma_long"}, None),
}
SMA_LONG = {"mr_a1_ibs": None, "tf_t2_sma210": 210}


def _operand(operand: Operand) -> str:
    if isinstance(operand, SeriesOperand):
        assert operand.offset == -1  # the last complete daily bar
        return operand.series
    if isinstance(operand, ConstOperand):
        return format(operand.const.normalize(), "f")
    return operand.param


def _rule_texts(rules: Sequence[RuleSpec] | None) -> set[str]:
    return {f"{_operand(r.left)} {r.op.value} {_operand(r.right)}" for r in rules or ()}


# --------------------------------------------------------------------------- protocol


def test_protocol_has_the_owner_windows_budget_and_costs() -> None:
    protocol = load_protocol(PROTOCOL)
    assert (protocol.dev_window.start, protocol.dev_window.end) == (
        date(2016, 11, 1),
        date(2022, 12, 31),
    )
    assert (protocol.lockbox_window.start, protocol.lockbox_window.end) == (
        date(2023, 1, 1),
        date(2026, 6, 30),
    )
    assert protocol.embargo_start == date(2026, 7, 1)
    assert (protocol.trial_budget, protocol.v1_trials) == (TRIAL_BUDGET, V1_TRIAL_COUNT)
    assert protocol.costs.base_bps_per_side == 5
    assert protocol.costs.stress_multipliers == (Decimal(2), Decimal(3))
    assert (protocol.walk_forward.train_months, protocol.walk_forward.test_months) == (12, 3)
    assert protocol.bootstrap.resamples == 10_000
    assert protocol.monte_carlo.sims == 10_000
    assert protocol.random_entry.sims == 1_000
    assert protocol.pbo.blocks == 16
    assert protocol.gate_sample == "walk_forward_fresh_accounts"
    seeds = {protocol.bootstrap.seed, protocol.monte_carlo.seed, protocol.random_entry.seed}
    assert len(seeds) == 3


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("{start: 2023-01-01, end: 2026-06-30}", "{start: 2022-06-01, end: 2026-06-30}"),
        ("embargo_start: 2026-07-01", "embargo_start: 2026-06-01"),
        ("blocks: 16", "blocks: 15"),
        ("trial_budget: 12", "trial_budget: 20"),
        ("v1_trials: 1", "v1_trials: 0"),
        ("stress_multipliers: [2, 3]", "stress_multipliers: [3, 2]"),
        ("protocol_version", "unknown_key: 1\nprotocol_version"),
        ("gate_sample: walk_forward_fresh_accounts", "gate_sample: everything"),
        ("gate_sample: walk_forward_fresh_accounts", "# no gate sample"),
    ],
)
def test_invalid_protocols_are_refused(tmp_path: Path, old: str, new: str) -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "protocol.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(BacktestResearchError) as caught:
        load_protocol(path)
    assert caught.value.code == "PROTOCOL_INVALID"


# --------------------------------------------------------------------------- configs


def test_every_hypothesis_has_its_config() -> None:
    ids = {h.id for h in load_hypotheses(HYPOTHESES)}
    assert ids == set(EXPECTED_RULES)
    assert {p.stem for p in CONFIGS.glob("*.yaml")} == ids
    for hypothesis in load_hypotheses(HYPOTHESES):
        assert hypothesis.config_path == f"research/configs/{hypothesis.id}.yaml"


@pytest.mark.parametrize("hypothesis_id", sorted(EXPECTED_RULES))
def test_research_config_loads_and_matches_its_hypothesis(hypothesis_id: str) -> None:
    loaded = load_config(CONFIGS / f"{hypothesis_id}.yaml")
    config = loaded.config
    strategy = config.strategy
    assert pending_owner_decisions(config) == ()
    assert config.strategy_version == f"2.0.0-{hypothesis_id}"
    assert config.universe.whitelist == ("SPY", "QQQ", "IWM")
    assert config.market_data.feed is DataFeed.SIP
    assert config.universe.liquidity_feed is DataFeed.SIP
    assert config.market_data.adjustment == "split"
    assert strategy.holding_mode is HoldingMode.SWING
    assert strategy.primary_timeframe is Timeframe.DAY_1
    assert strategy.confirmation_timeframe is None
    assert strategy.no_trade_rules == ()
    assert strategy.cooldown_bars == 0
    entries, exits, time_stop = EXPECTED_RULES[hypothesis_id]
    assert _rule_texts(strategy.entry_rules) == entries
    assert _rule_texts(strategy.exit_rules) == exits
    assert strategy.exit.time_stop_bars == time_stop
    assert strategy.exit.exit_on_signal_reversal is True
    assert strategy.exit.stop_method == "atr"
    assert strategy.exit.stop_atr_multiplier == 3
    assert strategy.exit.take_profit_r_multiple == 10
    indicators = strategy.indicators
    assert indicators.atr_period == 14
    family_b = hypothesis_id.startswith("mr_b")
    assert indicators.rsi_period == (2 if family_b else None)
    assert indicators.sma_short_period == (5 if family_b else None)
    assert indicators.sma_long_period == SMA_LONG.get(hypothesis_id, 200)
    assert indicators.ema_fast is None
    assert indicators.ema_slow is None
    assert indicators.volume_avg_period is None
    assert config.market_data.history_warmup_bars >= max(required_warmup_bars(strategy), 210)
    risk = config.risk
    assert risk.risk_per_trade_pct == Decimal("0.01")
    assert risk.max_symbol_exposure_pct == Decimal("0.33")
    assert risk.max_positions == 3
    assert risk.max_total_exposure_pct == Decimal("0.99")
    assert risk.max_aggregate_open_risk_pct == Decimal("0.03")
    assert risk.slippage_buffer_bps == 5
    assert risk.max_daily_loss_pct == OWNER.risk.max_daily_loss_pct
    assert risk.max_weekly_loss_pct == OWNER.risk.max_weekly_loss_pct
    assert risk.max_drawdown_pct == OWNER.risk.max_drawdown_pct
    assert config.execution.entry_order_type == "market"
    backtest = config.backtest
    assert (backtest.start_date, backtest.end_date) == (date(2016, 11, 1), date(2022, 12, 31))
    assert (backtest.walk_forward_train_months, backtest.walk_forward_test_months) == (12, 3)
    for name in ("min_expectancy_r", "min_profit_factor", "max_drawdown_pct"):
        assert getattr(backtest, name) == getattr(OWNER.backtest, name)
    assert backtest.min_trades_out_of_sample == OWNER.backtest.min_trades_out_of_sample


def test_research_configs_pass_the_dev_checks_and_pass_rules_validate() -> None:
    protocol = load_protocol(PROTOCOL)
    for hypothesis in load_hypotheses(HYPOTHESES):
        study = load_study(hypothesis, ROOT, protocol)
        window, tests = dev_windows(study.loaded.config, protocol)
        assert (window.start, window.end) == (
            protocol.dev_window.start,
            protocol.dev_window.end,
        )
        assert tests
        assert all(t.end < protocol.lockbox_window.start for t in tests)
        assert tests[0].start == date(2017, 11, 1)
        validate_pass_rule(
            hypothesis.pass_rule["dev"], dev_metric_names(protocol), where=hypothesis.id
        )
        validate_pass_rule(hypothesis.pass_rule["lockbox"], LOCKBOX_METRICS, where=hypothesis.id)


def test_config_is_unchanged_on_v1() -> None:
    """config.yaml stays on strategy v1.0.0 until a v2 finalist is approved."""
    assert OWNER.strategy_version == "1.0.0"
    assert "expectancy_r" in DEV_BASE_METRICS
