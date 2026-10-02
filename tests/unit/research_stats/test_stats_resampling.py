"""Block bootstrap, Monte-Carlo drawdown and PBO (CSCV) of :mod:`backtest.stats`."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from backtest.stats import (
    block_bootstrap,
    iso_week_blocks,
    max_drawdown,
    monte_carlo_drawdown,
    probability_of_backtest_overfitting,
)

# --------------------------------------------------------------------------- ISO weeks


def test_iso_week_blocks_group_by_iso_year_and_week() -> None:
    dates = [
        date(2021, 1, 11),  # Mon, 2021-W02
        date(2021, 1, 1),  # Fri, 2020-W53 (ISO year 2020)
        date(2021, 1, 4),  # Mon, 2021-W01
        date(2020, 12, 28),  # Mon, 2020-W53
        date(2021, 1, 10),  # Sun, 2021-W01
    ]
    assert iso_week_blocks(dates) == [(1, 3), (2, 4), (0,)]
    assert iso_week_blocks([]) == []


# --------------------------------------------------------------------------- bootstrap


def _sample_trades(n: int, seed: int) -> tuple[list[date], list[float], list[float]]:
    rng = np.random.default_rng(seed)
    start = date(2016, 1, 4)
    dates = sorted(start + timedelta(days=int(d)) for d in rng.integers(0, 2500, n))
    r_values = [float(v) for v in rng.normal(0.3, 1.0, n)]
    return dates, r_values, [100.0 * v for v in r_values]


def test_block_bootstrap_is_deterministic_for_a_seed() -> None:
    dates, r_values, pnl = _sample_trades(300, seed=11)
    first = block_bootstrap(dates, r_values, pnl, n_resamples=2000, seed=7)
    again = block_bootstrap(dates, r_values, pnl, n_resamples=2000, seed=7)
    other = block_bootstrap(dates, r_values, pnl, n_resamples=2000, seed=8)
    assert first == again
    assert (first.expectancy_ci_low, first.expectancy_ci_high) != (
        other.expectancy_ci_low,
        other.expectancy_ci_high,
    )


def test_block_bootstrap_point_estimates_and_interval() -> None:
    dates, r_values, pnl = _sample_trades(300, seed=3)
    result = block_bootstrap(dates, r_values, pnl, n_resamples=4000, seed=1)
    gains = sum(p for p in pnl if p > 0)
    losses = -sum(p for p in pnl if p < 0)
    assert result.n_trades == 300
    assert result.n_blocks == len(iso_week_blocks(dates))
    assert result.expectancy == pytest.approx(sum(r_values) / 300)
    assert result.profit_factor == pytest.approx(gains / losses)
    assert result.expectancy is not None
    assert result.expectancy_ci_low is not None
    assert result.expectancy_ci_high is not None
    assert result.expectancy_ci_low < result.expectancy < result.expectancy_ci_high
    assert result.expectancy_ci_low > 0  # mean 0.3 R, sd 1, n 300
    assert result.prob_expectancy_le_zero is not None
    assert result.prob_expectancy_le_zero < 0.01
    assert result.profit_factor_ci_low is not None
    assert result.profit_factor is not None
    assert result.profit_factor_ci_low < result.profit_factor


def test_block_bootstrap_resamples_whole_weeks() -> None:
    # Week 1: three trades of +1 R; week 2: one trade of -1 R. Two blocks drawn with
    # replacement give only {w1, w1} = 1, {w1, w2} = 0.5, {w2, w2} = -1 (trade-level
    # resampling would also produce other means such as 0).
    dates = [date(2021, 3, 1), date(2021, 3, 2), date(2021, 3, 3), date(2021, 3, 8)]
    result = block_bootstrap(dates, [1.0, 1.0, 1.0, -1.0], [1.0, 1.0, 1.0, -1.0], seed=5)
    assert result.n_blocks == 2
    assert result.expectancy_ci_low == pytest.approx(-1.0)
    assert result.expectancy_ci_high == pytest.approx(1.0)
    assert result.prob_expectancy_le_zero == pytest.approx(0.25, abs=0.02)


def test_block_bootstrap_profit_factor_without_losses_and_missing_r() -> None:
    dates = [date(2021, 3, 1), date(2021, 3, 9), date(2021, 3, 16)]
    result = block_bootstrap(dates, [0.5, None, 1.5], [10.0, 5.0, 30.0], n_resamples=500)
    assert result.expectancy == pytest.approx(1.0)  # the None R is left out
    assert result.profit_factor == math.inf
    assert result.profit_factor_ci_low == math.inf


def test_block_bootstrap_input_checks() -> None:
    empty = block_bootstrap([], [], [])
    assert (empty.n_trades, empty.expectancy, empty.profit_factor) == (0, None, None)
    with pytest.raises(ValueError, match="same length"):
        block_bootstrap([date(2021, 1, 4)], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="n_resamples"):
        block_bootstrap([date(2021, 1, 4)], [1.0], [1.0], n_resamples=0)


# --------------------------------------------------------------------------- Monte-Carlo DD


def _path_drawdown(returns: list[float]) -> float:
    path = [1.0]
    for r in returns:
        path.append(path[-1] * (1 + r))
    return max_drawdown(path)


def test_monte_carlo_drawdown_is_deterministic_and_ordered() -> None:
    returns = [float(v) for v in np.random.default_rng(4).normal(0.0005, 0.01, 500)]
    first = monte_carlo_drawdown(returns, mean_block_length=10, n_sims=2000, seed=3)
    assert first == monte_carlo_drawdown(returns, mean_block_length=10, n_sims=2000, seed=3)
    assert first != monte_carlo_drawdown(returns, mean_block_length=10, n_sims=2000, seed=4)
    assert 0 <= first.p5 <= first.p50 <= first.p95 <= 1
    assert first.observed == pytest.approx(_path_drawdown(returns))


def test_monte_carlo_drawdown_known_paths() -> None:
    up = monte_carlo_drawdown([0.01, 0.02, 0.005], n_sims=200)
    assert (up.p95, up.prob_drawdown_ge_threshold) == (0.0, 0.0)
    down = monte_carlo_drawdown([-0.1, -0.1, -0.1], n_sims=200, threshold=0.15)
    assert down.p50 == pytest.approx(1 - 0.9**3)
    assert down.prob_drawdown_ge_threshold == 1.0


def test_stationary_bootstrap_with_huge_blocks_rotates_the_series() -> None:
    # A block never restarts: every path is a circular rotation of the series.
    returns = [0.1, -0.2, 0.05, -0.1, 0.3]
    rotations = {round(_path_drawdown(returns[k:] + returns[:k]), 12) for k in range(5)}
    result = monte_carlo_drawdown(returns, mean_block_length=1e15, n_sims=300, seed=2)
    for value in (result.p5, result.p50, result.p95):
        assert round(value, 12) in rotations


def test_monte_carlo_drawdown_input_checks() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        monte_carlo_drawdown([])
    with pytest.raises(ValueError, match="mean_block_length"):
        monte_carlo_drawdown([0.01], mean_block_length=0.5)
    with pytest.raises(ValueError, match="n_sims"):
        monte_carlo_drawdown([0.01], n_sims=0)


# --------------------------------------------------------------------------- PBO


def test_pbo_of_pure_noise_is_about_one_half() -> None:
    values = []
    for seed in range(20):
        matrix = np.random.default_rng(100 + seed).normal(0.0, 0.01, size=(640, 8))
        values.append(probability_of_backtest_overfitting(matrix).pbo)
    assert float(np.mean(values)) == pytest.approx(0.5, abs=0.15)


def test_pbo_of_a_genuinely_strong_configuration_is_near_zero() -> None:
    matrix = np.random.default_rng(7).normal(0.0, 0.01, size=(1600, 10))
    matrix[:, 3] += 0.004
    result = probability_of_backtest_overfitting(matrix)
    assert result.n_combinations == 12_870  # C(16, 8)
    assert result.n_configs == 10
    assert result.pbo < 0.05
    assert result.prob_oos_loss < 0.05
    assert result.median_logit > 0


def test_pbo_is_deterministic_and_handles_ties() -> None:
    matrix = np.random.default_rng(9).normal(0.0, 0.01, size=(200, 4))
    first = probability_of_backtest_overfitting(matrix, n_blocks=8)
    assert first == probability_of_backtest_overfitting(matrix, n_blocks=8)
    assert first.n_combinations == 70  # C(8, 4)
    # Identical configurations: every OOS rank is the median -> logit 0 -> counted.
    flat = probability_of_backtest_overfitting(np.zeros((40, 3)), n_blocks=4)
    assert (flat.pbo, flat.slope) == (1.0, None)


def test_pbo_input_checks() -> None:
    with pytest.raises(ValueError, match="at least 2 configurations"):
        probability_of_backtest_overfitting(np.zeros((100, 1)))
    with pytest.raises(ValueError, match="even number"):
        probability_of_backtest_overfitting(np.zeros((100, 3)), n_blocks=5)
    with pytest.raises(ValueError, match="at least 32 periods"):
        probability_of_backtest_overfitting(np.zeros((20, 3)), n_blocks=16)
    with pytest.raises(ValueError, match="T x N matrix"):
        probability_of_backtest_overfitting(np.zeros(100))
