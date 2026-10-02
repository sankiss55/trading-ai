"""Shared helpers for the deterministic backtest simulations.

Every parameter comes from ``tests/fixtures/config.backtest.yaml`` (TEST FIXTURE values,
not owner decisions) and every price from the seeded synthetic generator.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from app.config import AppConfig, LoadedConfig, load_config
from backtest.data import SyntheticSpec, generate_synthetic_data

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "tests" / "fixtures" / "config.backtest.yaml"
PENDING_CONFIG = ROOT / "tests" / "fixtures" / "config.pending.yaml"
"""The pre-2026-10-02 config.yaml with every OWNER_DECISION still null."""

DATA_START = date(2024, 12, 16)
"""First synthetic session: the sessions before ``backtest.start_date`` are warm-up."""
SEED = 42
DRIFT_BPS = 0.2


def make_dataset(directory: Path, *, end: date, seed: int = SEED) -> Path:
    """Write the standard synthetic dataset into ``directory`` and return it."""
    generate_synthetic_data(
        directory,
        SyntheticSpec(
            symbols=("SYNTH",),
            start=DATA_START,
            end=end,
            seed=seed,
            drift_bps_per_minute=DRIFT_BPS,
        ),
    )
    return directory


def fixture_loaded(
    *,
    backtest: dict[str, Any] | None = None,
    strategy: dict[str, Any] | None = None,
    exit_: dict[str, Any] | None = None,
) -> LoadedConfig:
    """The fixture config with optional section overrides (values must stay valid)."""
    loaded = load_config(FIXTURE_CONFIG)
    config: AppConfig = loaded.config
    strategy_section = config.strategy
    if exit_:
        strategy_section = strategy_section.model_copy(
            update={"exit": strategy_section.exit.model_copy(update=exit_)}
        )
    if strategy:
        strategy_section = strategy_section.model_copy(update=strategy)
    config = config.model_copy(
        update={
            "strategy": strategy_section,
            "backtest": config.backtest.model_copy(update=backtest or {}),
        }
    )
    return LoadedConfig(config=config, config_hash=loaded.config_hash, path=loaded.path)
