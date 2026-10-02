"""Distributions, return series, per-trade statistics, Sharpe inference and multiple
testing of :mod:`backtest.stats`."""

from __future__ import annotations

import math
import statistics
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from backtest.report import EquityPoint
from backtest.stats import (
    bhy_adjust,
    daily_returns,
    deflated_sharpe_ratio,
    effective_trials,
    expected_max_sharpe,
    holm_adjust,
    max_drawdown,
    min_backtest_length,
    min_track_record_length,
    performance_summary,
    probabilistic_sharpe_ratio,
    return_moments,
    session_close_equity,
    session_exposure_flags,
    t_cdf,
    t_quantile,
    t_to_p_value,
    trade_r_stats,
    wilson_interval,
    z_to_p_value,
)
from domain.models import SessionDay

D = Decimal

# --------------------------------------------------------------------------- Student t


@pytest.mark.parametrize(
    ("p", "df", "table"),
    [
        (0.975, 1, 12.706),
        (0.975, 2, 4.303),
        (0.975, 9, 2.262),  # n = 10
        (0.975, 29, 2.045),  # n = 30
        (0.975, 99, 1.984),  # n = 100
        (0.975, 999, 1.962),  # n = 1000
        (0.995, 9, 3.250),
        (0.95, 29, 1.699),
        (0.9, 5, 1.476),
    ],
)
def test_t_quantile_matches_table_values(p: float, df: int, table: float) -> None:
    assert round(t_quantile(p, df), 3) == table


def test_t_quantile_matches_closed_forms() -> None:
    for p in (0.6, 0.9, 0.975, 0.999):
        assert t_quantile(p, 1) == pytest.approx(math.tan(math.pi * (p - 0.5)), rel=1e-10)
        exact_df2 = (2 * p - 1) / math.sqrt(2 * p * (1 - p))
        assert t_quantile(p, 2) == pytest.approx(exact_df2, rel=1e-10)


def test_t_quantile_inverts_the_cdf_and_is_symmetric() -> None:
    for df in (1, 3, 9, 30, 250, 5000):
        for p in (0.001, 0.05, 0.3, 0.7, 0.975):
            assert t_cdf(t_quantile(p, df), df) == pytest.approx(p, abs=1e-12)
        assert t_quantile(0.025, df) == pytest.approx(-t_quantile(0.975, df), rel=1e-12)
    assert t_quantile(0.5, 7) == 0.0
    assert t_cdf(0.0, 7) == 0.5


def test_t_tends_to_the_normal_for_large_df() -> None:
    assert t_quantile(0.975, 1_000_000) == pytest.approx(1.959966, abs=1e-5)


@pytest.mark.parametrize(
    ("p", "df", "message"),
    [
        (0.0, 5, "p must be in"),
        (1.0, 5, "p must be in"),
        (0.5, 0, "degrees of freedom"),
        (0.5, -1, "degrees of freedom"),
    ],
)
def test_t_quantile_rejects_invalid_inputs(p: float, df: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        t_quantile(p, df)


def test_t_and_z_p_values() -> None:
    t = t_quantile(0.975, 9)
    assert t_to_p_value(t, 9) == pytest.approx(0.05, abs=1e-12)
    assert t_to_p_value(t, 9, "greater") == pytest.approx(0.025, abs=1e-12)
    assert t_to_p_value(t, 9, "less") == pytest.approx(0.975, abs=1e-12)
    assert t_to_p_value(0.0, 9) == 1.0
    assert z_to_p_value(1.959963984540054) == pytest.approx(0.05, abs=1e-12)
    assert z_to_p_value(1.6448536269514722, "greater") == pytest.approx(0.05, abs=1e-12)


# --------------------------------------------------------------------------- Wilson


@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [
        (8, 10, 0.4902, 0.9433),
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (50, 100, 0.4038, 0.5962),
    ],
)
def test_wilson_interval_known_values(successes: int, n: int, low: float, high: float) -> None:
    got_low, got_high = wilson_interval(successes, n)
    assert round(got_low, 4) == low
    assert round(got_high, 4) == high


def test_wilson_interval_edge_cases() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError, match="successes <= n"):
        wilson_interval(11, 10)


# --------------------------------------------------------------------------- series


def _session(day: date) -> SessionDay:
    opened = datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC)
    return SessionDay(
        session_date=day,
        open_utc=opened,
        close_utc=opened + timedelta(hours=6, minutes=30),
        is_early_close=False,
    )


def _point(day: date, hour: int, equity: str, exposure: str) -> EquityPoint:
    return EquityPoint(
        timestamp_utc=datetime(day.year, day.month, day.day, hour, 0, tzinfo=UTC),
        equity=D(equity),
        exposure=D(exposure),
    )


def test_session_close_equity_and_exposure_flags() -> None:
    days = [date(2021, 3, 1), date(2021, 3, 2), date(2021, 3, 3), date(2021, 3, 4)]
    sessions = [_session(d) for d in days]
    curve = [
        _point(days[1], 15, "1010", "0.5"),  # intraday sample, position open
        _point(days[1], 21, "1020", "0.5"),  # close of day 2
        _point(days[2], 21, "1000", "0"),  # close of day 3, flat
    ]
    equity = session_close_equity(curve, sessions, D(1000))
    assert equity == [D(1000), D(1000), D(1020), D(1000), D(1000)]
    assert session_exposure_flags(curve, sessions) == [False, True, False, False]
    assert daily_returns(equity) == pytest.approx([0.0, 0.02, 1000 / 1020 - 1, 0.0])


def test_daily_returns_values_and_refusal() -> None:
    assert daily_returns([D(100), D(110), D(99), D(99)]) == pytest.approx([0.1, -0.1, 0.0])
    assert daily_returns([D(100)]) == []
    with pytest.raises(ValueError, match="equity must stay positive"):
        daily_returns([D(100), D(0), D(10)])


def test_max_drawdown() -> None:
    assert max_drawdown([1.0, 1.2, 0.9, 1.3, 1.04]) == pytest.approx(0.25)
    assert max_drawdown([1.0, 1.1, 1.2]) == 0.0


def test_performance_summary_hand_values() -> None:
    returns = [0.01, -0.02, 0.03, 0.0]
    summary = performance_summary(returns, exposure_flags=[True, True, False, False])
    growth = 1.01 * 0.98 * 1.03
    cagr = growth ** (252 / 4) - 1
    sd = statistics.stdev(returns)
    assert summary.n_periods == 4
    assert summary.total_return == pytest.approx(growth - 1)
    assert summary.cagr == pytest.approx(cagr)
    assert summary.annualized_volatility == pytest.approx(sd * math.sqrt(252))
    assert summary.sharpe == pytest.approx(0.005 / sd * math.sqrt(252))
    # downside deviation = sqrt(0.02^2 / 4) = 0.01
    assert summary.sortino == pytest.approx(0.005 / 0.01 * math.sqrt(252))
    assert summary.max_drawdown == pytest.approx(0.02)
    assert summary.calmar == pytest.approx(cagr / 0.02)
    assert summary.exposure == 0.5
    assert summary.exposure_adjusted_return == pytest.approx(cagr / 0.5)


def test_performance_summary_risk_free_and_mar() -> None:
    returns = [0.01, -0.02, 0.03, 0.0]
    summary = performance_summary(returns, risk_free_rate=0.252, mar=0.252)
    sd = statistics.stdev(returns)
    assert summary.sharpe == pytest.approx((0.005 - 0.001) / sd * math.sqrt(252))
    excess = [r - 0.001 for r in returns]
    downside = math.sqrt(sum(min(e, 0.0) ** 2 for e in excess) / 4)
    assert summary.sortino == pytest.approx(statistics.mean(excess) / downside * math.sqrt(252))


def test_performance_summary_degenerate_series() -> None:
    empty = performance_summary([])
    assert (empty.n_periods, empty.total_return, empty.cagr, empty.sharpe) == (0, 0.0, None, None)
    flat = performance_summary([0.0, 0.0, 0.0], exposure_flags=[False, False, False])
    assert flat.sharpe is None
    assert flat.sortino is None
    assert flat.calmar is None
    assert flat.exposure == 0.0
    assert flat.exposure_adjusted_return is None
    with pytest.raises(ValueError, match="exposure_flags has 1 values"):
        performance_summary([0.01, 0.02], exposure_flags=[True])


def test_return_moments() -> None:
    moments = return_moments([1.0, 2.0, 3.0, 4.0, 5.0])
    assert moments.mean == 3.0
    assert moments.sd == pytest.approx(statistics.stdev([1, 2, 3, 4, 5]))
    assert moments.skewness == pytest.approx(0.0)
    assert moments.kurtosis == pytest.approx(6.8 / 4.0)  # m4 / m2^2 = 6.8 / 2^2
    assert return_moments([-1.0, 1.0]).kurtosis == pytest.approx(1.0)
    assert return_moments([0.5]).sd is None
    assert return_moments([]).mean is None


# --------------------------------------------------------------------------- per trade


def test_trade_r_stats_hand_values() -> None:
    stats = trade_r_stats([1.0, -1.0, 2.0, 0.5, -0.5])
    sd = math.sqrt(5.7 / 4)
    se = sd / math.sqrt(5)
    assert stats.n == 5
    assert stats.mean == pytest.approx(0.4)
    assert stats.sd == pytest.approx(sd)
    assert stats.t_stat == pytest.approx(0.4 / se)
    assert stats.ci_low == pytest.approx(0.4 - 2.776445105 * se, abs=1e-8)
    assert stats.ci_high == pytest.approx(0.4 + 2.776445105 * se, abs=1e-8)
    assert stats.p_value == pytest.approx(1 - t_cdf(0.4 / se, 4))
    assert stats.wins == 3
    assert stats.win_rate == pytest.approx(0.6)
    assert (stats.win_rate_ci_low, stats.win_rate_ci_high) == wilson_interval(3, 5)


def test_trade_r_stats_wins_override_and_small_samples() -> None:
    assert trade_r_stats([0.1, -0.2], wins=2).wins == 2
    single = trade_r_stats([0.3])
    assert (single.mean, single.sd, single.t_stat, single.ci_low) == (0.3, None, None, None)
    empty = trade_r_stats([])
    assert (empty.n, empty.mean, empty.win_rate) == (0, None, None)


# --------------------------------------------------------------------------- Sharpe inference


def test_deflated_sharpe_ratio_reproduces_bailey_lopez_de_prado_2014() -> None:
    # Paper example: annualised SR 2.5, variance of the trials' annualised SR 0.5,
    # N = 100 trials, T = 1250 daily observations (5 years of 250 sessions), skew -3,
    # raw kurtosis 10. Daily units: SR / sqrt(250), V / 250 (the paper's convention).
    result = deflated_sharpe_ratio(
        2.5 / math.sqrt(250), 1250, -3.0, 10.0, n_trials=100, trials_sr_variance=0.5 / 250
    )
    assert result.sr0 == pytest.approx(0.1132, abs=5e-5)
    assert result.dsr == pytest.approx(0.9004, abs=5e-5)
    # With this project's 252-session convention the DSR still rounds to 0.900.
    ours = deflated_sharpe_ratio(
        2.5 / math.sqrt(252), 1250, -3.0, 10.0, n_trials=100, trials_sr_variance=0.5 / 252
    )
    assert round(ours.dsr, 3) == 0.900


def test_psr_closed_form_and_monotonicity() -> None:
    sr, n_obs = 0.1, 500
    expected = statistics.NormalDist().cdf(sr * math.sqrt(n_obs - 1) / math.sqrt(1 + sr**2 / 2))
    assert probabilistic_sharpe_ratio(sr, 0.0, n_obs, 0.0, 3.0) == pytest.approx(expected)
    assert probabilistic_sharpe_ratio(0.05, 0.05, n_obs, -1.0, 6.0) == pytest.approx(0.5)
    values = [probabilistic_sharpe_ratio(s, 0.0, n_obs, 0.0, 3.0) for s in (0.0, 0.03, 0.06, 0.1)]
    assert values == sorted(values)
    longer = [probabilistic_sharpe_ratio(0.05, 0.0, t, 0.0, 3.0) for t in (50, 200, 1000)]
    assert longer == sorted(longer)
    benchmarks = [probabilistic_sharpe_ratio(0.1, b, n_obs, 0.0, 3.0) for b in (0.0, 0.05, 0.08)]
    assert benchmarks == sorted(benchmarks, reverse=True)
    # Negative skew and fat tails lower the confidence of a positive Sharpe ratio.
    assert probabilistic_sharpe_ratio(0.1, 0.0, n_obs, -2.0, 9.0) < values[-1]
    with pytest.raises(ValueError, match="n_obs must be"):
        probabilistic_sharpe_ratio(0.1, 0.0, 1, 0.0, 3.0)


def test_expected_max_sharpe() -> None:
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(50, 0.0) == 0.0
    growing = [expected_max_sharpe(n, 0.01) for n in (2, 5, 12, 100)]
    assert growing == sorted(growing)
    with pytest.raises(ValueError, match="n_trials must be"):
        expected_max_sharpe(0.5, 0.01)
    with pytest.raises(ValueError, match="trials_sr_variance"):
        expected_max_sharpe(5, -0.1)


def test_effective_trials() -> None:
    assert effective_trials(0.0, 10) == 10.0
    assert effective_trials(1.0, 10) == 1.0
    assert effective_trials(0.5, 10) == 5.5
    assert effective_trials(-0.3, 4) == 4.0  # clamped to rho = 0


def test_min_track_record_length() -> None:
    # Normal returns: 1 + (1 + SR^2 / 2) * (z_0.95 / SR)^2.
    expected = 1 + (1 + 0.01 / 2) * (1.6448536269514722 / 0.1) ** 2
    assert min_track_record_length(0.1, 0.0, 0.0, 3.0) == pytest.approx(expected)
    length = min_track_record_length(0.08, 0.02, -1.0, 7.0, confidence=0.9)
    # PSR evaluated at exactly MinTRL periods equals the requested confidence.
    factor = 1 + 0.08 + 6 / 4 * 0.08**2
    z = (0.08 - 0.02) * math.sqrt(length - 1) / math.sqrt(factor)
    assert statistics.NormalDist().cdf(z) == pytest.approx(0.9)
    assert min_track_record_length(0.02, 0.05, 0.0, 3.0) == math.inf


def test_min_backtest_length() -> None:
    # Bailey et al. (2014): with 7 zero-skill trials a 2-year backtest is expected to
    # show an annualised Sharpe of about 1.
    assert min_backtest_length(7, 1.0) == pytest.approx(1.9231, abs=1e-4)
    for n in (2, 10, 100, 1000):
        assert min_backtest_length(n, 1.0) < 2 * math.log(n)  # paper's upper bound
    assert min_backtest_length(45, 1.0) > min_backtest_length(7, 1.0)
    assert min_backtest_length(7, 2.0) == pytest.approx(min_backtest_length(7, 1.0) / 4)
    with pytest.raises(ValueError, match="n_trials must be > 1"):
        min_backtest_length(1, 1.0)


# --------------------------------------------------------------------------- multiple testing


def test_holm_known_example() -> None:
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> cummax
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_adjust([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []


def test_bhy_known_example() -> None:
    # c(4) = 1 + 1/2 + 1/3 + 1/4 = 25/12; m c(m) = 25/3; sorted p_(i) / i:
    # 0.005, 0.005, 0.01, 0.01 -> x 25/3 -> 0.041667, 0.041667, 0.083333, 0.083333
    adjusted = bhy_adjust([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([1 / 24, 1 / 12, 1 / 12, 1 / 24])
    assert bhy_adjust([0.02]) == pytest.approx([0.02])
    assert bhy_adjust([]) == []


def test_adjustments_are_at_least_the_raw_p_values() -> None:
    raw = [0.001, 0.2, 0.04, 0.8, 0.013]
    for adjusted in (holm_adjust(raw), bhy_adjust(raw)):
        assert all(a >= p for a, p in zip(adjusted, raw, strict=True))
        assert all(a <= 1.0 for a in adjusted)
    with pytest.raises(ValueError, match="p-values must be in"):
        holm_adjust([1.2])
