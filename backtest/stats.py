"""Research statistics for the family v2 pipeline (plan WU3; DECISIONS.md "family v2").

Pure, deterministic functions (no I/O, no wall clock). Randomised procedures take an
explicit ``seed`` and use ``numpy.random.Generator(PCG64(seed))``: the same seed and
inputs give the same result (for a given numpy version).

Unit conventions:

* Returns are simple per-period returns as floats (``0.01`` = 1 %). One period is one
  trading session; annualisation uses :data:`TRADING_DAYS_PER_YEAR` (252) unless a
  function takes ``periods_per_year``.
* Daily returns are session-close mark-to-market changes of the account equity: a flat
  session has a zero return because its equity does not move (:func:`daily_returns`).
* Sharpe ratios given to :func:`probabilistic_sharpe_ratio`, :func:`deflated_sharpe_ratio`
  and :func:`min_track_record_length` are PER PERIOD (not annualised): ``mean / sd`` of
  the per-period returns, with ``n_obs`` periods. The trial variance of DSR is the
  variance of the trials' per-period Sharpe ratios. :func:`min_backtest_length` uses
  ANNUALISED Sharpe ratios and returns years (as in Bailey et al. 2014).
* Kurtosis is the raw (non-excess) fourth standardised moment: 3 for a normal law.
* Student-t quantiles are exact (inverse of the regularized incomplete beta function),
  not a normal approximation.

References: Bailey & Lopez de Prado (2012) "The Sharpe ratio efficient frontier" (PSR,
MinTRL); Bailey & Lopez de Prado (2014) "The deflated Sharpe ratio" (DSR); Bailey,
Borwein, Lopez de Prado & Zhu (2014) "Pseudo-mathematics and financial charlatanism"
(MinBTL) and (2017) "The probability of backtest overfitting" (PBO, CSCV); Politis &
Romano (1994) (stationary bootstrap); Holm (1979); Benjamini & Yekutieli (2001).
"""

from __future__ import annotations

import itertools
import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from statistics import NormalDist
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from backtest.report import TRADING_DAYS_PER_YEAR, EquityPoint
from domain.models import SessionDay

__all__ = [
    "EULER_GAMMA",
    "TRADING_DAYS_PER_YEAR",
    "BootstrapResult",
    "DeflatedSharpe",
    "DrawdownMonteCarlo",
    "PBOResult",
    "PerformanceSummary",
    "ReturnMoments",
    "TradeRStats",
    "bhy_adjust",
    "block_bootstrap",
    "daily_returns",
    "deflated_sharpe_ratio",
    "effective_trials",
    "expected_max_sharpe",
    "holm_adjust",
    "iso_week_blocks",
    "max_drawdown",
    "min_backtest_length",
    "min_track_record_length",
    "monte_carlo_drawdown",
    "performance_summary",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "return_moments",
    "session_close_equity",
    "session_exposure_flags",
    "t_cdf",
    "t_quantile",
    "t_to_p_value",
    "trade_r_stats",
    "wilson_interval",
    "z_to_p_value",
]

EULER_GAMMA: Final = 0.5772156649015329
"""Euler-Mascheroni constant (expected maximum of normal draws, DSR / MinBTL)."""

Alternative = Literal["two-sided", "greater", "less"]
FloatArray = npt.NDArray[np.float64]

_NORMAL: Final = NormalDist()
_CF_MAX_ITERATIONS: Final = 1_000_000
_CF_EPSILON: Final = 1e-15
_FP_MIN: Final = 1e-300
_MAX_BATCH_ELEMENTS: Final = 2_000_000


# --------------------------------------------------------------------------- normal / t


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) >= _FP_MIN else _FP_MIN)
    h = d
    for m in range(1, _CF_MAX_ITERATIONS + 1):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) >= _FP_MIN else _FP_MIN)
            c = 1.0 + aa / c
            c = c if abs(c) >= _FP_MIN else _FP_MIN
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < _CF_EPSILON:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _regularized_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _check_df(df: float) -> None:
    if not (df > 0 and math.isfinite(df)):
        raise ValueError(f"degrees of freedom must be finite and > 0, got {df}")


def _t_upper_tail(t: float, df: float) -> float:
    """``P(T > t)`` for ``t >= 0``."""
    return 0.5 * _regularized_beta(df / 2.0, 0.5, df / (df + t * t))


def _t_pdf(t: float, df: float) -> float:
    log_c = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    return math.exp(log_c - (df + 1.0) / 2.0 * math.log1p(t * t / df))


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with ``df`` degrees of freedom (exact, incomplete beta)."""
    _check_df(df)
    if math.isnan(t):
        raise ValueError("t must not be NaN")
    if t == 0.0:
        return 0.5
    tail = _t_upper_tail(abs(t), df)
    return 1.0 - tail if t > 0 else tail


def _cornish_fisher_t(z: float, df: float) -> float:
    """Cornish-Fisher expansion of the t quantile around the normal quantile ``z``."""
    z2 = z * z
    g1 = (z2 + 1.0) * z / 4.0
    g2 = ((5.0 * z2 + 16.0) * z2 + 3.0) * z / 96.0
    g3 = (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) * z / 384.0
    g4 = ((((79.0 * z2 + 776.0) * z2 + 1482.0) * z2 - 1920.0) * z2 - 945.0) * z / 92160.0
    return z + g1 / df + g2 / df**2 + g3 / df**3 + g4 / df**4


def t_quantile(p: float, df: float) -> float:
    """Quantile (inverse CDF) of Student's t: Newton steps on the exact CDF, bracketed."""
    _check_df(df)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    target = 1.0 - p
    low, high = 0.0, 1.0
    while _t_upper_tail(high, df) > target:
        low, high = high, high * 2.0
    guess = _cornish_fisher_t(_NORMAL.inv_cdf(p), df)
    t = guess if low < guess < high else 0.5 * (low + high)
    for _ in range(500):
        excess = _t_upper_tail(t, df) - target  # decreasing in t
        if excess > 0:
            low = t
        else:
            high = t
        candidate = t + excess / _t_pdf(t, df)
        if not low < candidate < high:
            candidate = 0.5 * (low + high)
        if abs(candidate - t) <= 1e-14 * max(1.0, abs(t)):
            return candidate
        t = candidate
    return t


def t_to_p_value(t: float, df: float, alternative: Alternative = "two-sided") -> float:
    """p-value of a t statistic: ``greater`` tests ``mean > 0``, ``less`` tests ``mean < 0``."""
    if alternative == "greater":
        return 1.0 - t_cdf(t, df)
    if alternative == "less":
        return t_cdf(t, df)
    _check_df(df)
    return min(1.0, 2.0 * _t_upper_tail(abs(t), df))


def z_to_p_value(z: float, alternative: Alternative = "two-sided") -> float:
    """p-value of a standard normal statistic (same ``alternative`` as :func:`t_to_p_value`)."""
    if alternative == "greater":
        return 1.0 - _NORMAL.cdf(z)
    if alternative == "less":
        return _NORMAL.cdf(z)
    return min(1.0, 2.0 * (1.0 - _NORMAL.cdf(abs(z))))


# --------------------------------------------------------------------------- equity series


def session_close_equity(
    curve: Sequence[EquityPoint], sessions: Sequence[SessionDay], starting_equity: Decimal
) -> list[Decimal]:
    """Equity at every session close: ``[starting_equity, close_1, ..., close_n]``.

    The value at a close is the last curve sample at or before ``close_utc`` (the
    starting equity before the first sample). Feed the result to :func:`daily_returns`.
    """
    points = sorted(curve, key=lambda p: p.timestamp_utc)
    stamps = [p.timestamp_utc for p in points]
    equity = [starting_equity]
    for session in sessions:
        index = bisect_right(stamps, session.close_utc) - 1
        equity.append(points[index].equity if index >= 0 else starting_equity)
    return equity


def session_exposure_flags(
    curve: Sequence[EquityPoint], sessions: Sequence[SessionDay]
) -> list[bool]:
    """``True`` for every session with a position: a sample in ``(previous close, close]``
    reports ``exposure > 0``. Aligned with :func:`daily_returns` of
    :func:`session_close_equity`."""
    points = sorted(curve, key=lambda p: p.timestamp_utc)
    stamps = [p.timestamp_utc for p in points]
    flags: list[bool] = []
    previous = 0
    for session in sessions:
        end = bisect_right(stamps, session.close_utc)
        flags.append(any(p.exposure > 0 for p in points[previous:end]))
        previous = max(previous, end)
    return flags


def daily_returns(equity_by_session: Sequence[Decimal]) -> list[float]:
    """Session-close mark-to-market returns ``E_t / E_(t-1) - 1`` (``n - 1`` values).

    Pass ``[starting_equity, close_1, ..., close_n]``; flat sessions give zeros because
    the equity does not change. Raises ``ValueError`` on a non-positive equity.
    """
    returns: list[float] = []
    for previous, current in itertools.pairwise(equity_by_session):
        if previous <= 0:
            raise ValueError(f"equity must stay positive to compute returns, got {previous}")
        returns.append(float(current / previous - 1))
    return returns


def max_drawdown(equity: Sequence[float]) -> float:
    """Largest ``(peak - equity) / peak`` along ``equity`` (``0`` when it never falls)."""
    peak = -math.inf
    worst = 0.0
    for value in equity:
        peak = max(peak, float(value))
        if peak > 0:
            worst = max(worst, (peak - float(value)) / peak)
    return worst


def _equity_path(returns: Sequence[float]) -> list[float]:
    path = [1.0]
    for r in returns:
        path.append(path[-1] * (1.0 + float(r)))
    return path


# --------------------------------------------------------------------------- performance


@dataclass(frozen=True)
class ReturnMoments:
    """Sample moments of per-period returns (``sd`` with ddof=1; skew and raw kurtosis
    from the biased central moments, as in Bailey & Lopez de Prado)."""

    n: int
    mean: float | None
    sd: float | None
    sharpe: float | None
    """Per-period Sharpe ``mean / sd`` (risk-free 0); ``None`` when ``sd`` is 0 or undefined."""
    skewness: float | None
    kurtosis: float | None
    """Raw kurtosis (normal = 3)."""


def return_moments(returns: Sequence[float]) -> ReturnMoments:
    """Mean, sd, per-period Sharpe, skewness and raw kurtosis of ``returns``."""
    n = len(returns)
    if n == 0:
        return ReturnMoments(0, None, None, None, None, None)
    r = np.asarray(returns, dtype=np.float64)
    mean = float(r.mean())
    if n < 2:
        return ReturnMoments(n, mean, None, None, None, None)
    sd = float(r.std(ddof=1))
    centered = r - mean
    m2 = float(np.mean(centered**2))
    skew = kurt = None
    if m2 > 0:
        skew = float(np.mean(centered**3)) / m2**1.5
        kurt = float(np.mean(centered**4)) / m2**2
    return ReturnMoments(n, mean, sd, mean / sd if sd > 0 else None, skew, kurt)


@dataclass(frozen=True)
class PerformanceSummary:
    """Account-level performance of a per-period return series (all ratios as fractions).

    * ``cagr = (prod(1 + r)) ** (periods_per_year / n) - 1``.
    * ``annualized_volatility = sd(r, ddof=1) * sqrt(periods_per_year)``.
    * ``sharpe = mean(r - rf / periods_per_year) / sd(r) * sqrt(periods_per_year)``.
    * ``sortino = mean(r - mar_p) / sqrt(mean(min(0, r - mar_p) ** 2)) * sqrt(ppy)`` with
      ``mar_p = mar / periods_per_year`` (target downside deviation over all periods).
    * ``max_drawdown`` on the compounded path starting at 1; ``calmar = cagr / max_drawdown``.
    * ``exposure`` = fraction of periods with a position; ``exposure_adjusted_return =
      cagr / exposure`` (return per unit of time in the market).

    A field is ``None`` when undefined (too few periods, zero dispersion, no exposure).
    """

    n_periods: int
    total_return: float
    cagr: float | None
    annualized_volatility: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    calmar: float | None
    exposure: float | None
    exposure_adjusted_return: float | None


def performance_summary(
    returns: Sequence[float],
    *,
    exposure_flags: Sequence[bool] | None = None,
    risk_free_rate: float = 0.0,
    mar: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceSummary:
    """Summarise per-period ``returns`` (``risk_free_rate`` and ``mar`` are annual)."""
    n = len(returns)
    if exposure_flags is not None and len(exposure_flags) != n:
        raise ValueError(f"exposure_flags has {len(exposure_flags)} values, returns has {n}")
    path = _equity_path(returns)
    growth = path[-1]
    cagr: float | None = None
    if n > 0 and growth > 0:
        try:
            raw = growth ** (periods_per_year / n) - 1.0
        except OverflowError:
            raw = math.inf
        cagr = raw if math.isfinite(raw) else None
    vol = sharpe = sortino = None
    r = np.asarray(returns, dtype=np.float64)
    scale = math.sqrt(periods_per_year)
    if n >= 2:
        sd = float(r.std(ddof=1))
        vol = sd * scale
        if sd > 0:
            sharpe = float(np.mean(r - risk_free_rate / periods_per_year)) / sd * scale
        excess = r - mar / periods_per_year
        downside = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
        if downside > 0:
            sortino = float(np.mean(excess)) / downside * scale
    mdd = max_drawdown(path)
    calmar = cagr / mdd if cagr is not None and mdd > 0 else None
    exposure = None
    if exposure_flags is not None and n > 0:
        exposure = sum(1 for flag in exposure_flags if flag) / n
    adjusted = cagr / exposure if cagr is not None and exposure else None
    return PerformanceSummary(
        n_periods=n,
        total_return=growth - 1.0,
        cagr=cagr,
        annualized_volatility=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        exposure=exposure,
        exposure_adjusted_return=adjusted,
    )


# --------------------------------------------------------------------------- per trade


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion (``(0, 1)`` when ``n == 0``)."""
    if n < 0 or not 0 <= successes <= max(n, 0):
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")
    if n == 0:
        return 0.0, 1.0
    z = _NORMAL.inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / n
    z2n = z * z / n
    center = (p + z2n / 2.0) / (1.0 + z2n)
    half = z / (1.0 + z2n) * math.sqrt(p * (1.0 - p) / n + z2n / (4.0 * n))
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return low, high


@dataclass(frozen=True)
class TradeRStats:
    """Per-trade R statistics: ``t = mean / (sd / sqrt(n))``, ``mean +/- t_q * sd / sqrt(n)``
    with the exact Student-t quantile (``n - 1`` df), one-sided p-value for ``mean > 0``
    and the Wilson interval of the win rate."""

    n: int
    mean: float | None
    sd: float | None
    standard_error: float | None
    t_stat: float | None
    p_value: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    wins: int
    win_rate: float | None
    win_rate_ci_low: float | None
    win_rate_ci_high: float | None


def trade_r_stats(
    r_values: Sequence[float], *, wins: int | None = None, confidence: float = 0.95
) -> TradeRStats:
    """Statistics of per-trade R; ``wins`` defaults to the count of ``r > 0`` (pass the
    ``net_pnl > 0`` count to match :mod:`backtest.report`)."""
    n = len(r_values)
    win_count = sum(1 for r in r_values if r > 0) if wins is None else wins
    win_rate = win_low = win_high = None
    if n > 0:
        win_rate = win_count / n
        win_low, win_high = wilson_interval(win_count, n, confidence)
    mean = sd = se = t_stat = p_value = ci_low = ci_high = None
    if n > 0:
        mean = math.fsum(r_values) / n
    if n >= 2 and mean is not None:
        sd = float(np.std(np.asarray(r_values, dtype=np.float64), ddof=1))
        se = sd / math.sqrt(n)
        quantile = t_quantile(1.0 - (1.0 - confidence) / 2.0, n - 1)
        ci_low, ci_high = mean - quantile * se, mean + quantile * se
        if se > 0:
            t_stat = mean / se
            p_value = t_to_p_value(t_stat, n - 1, "greater")
    return TradeRStats(
        n=n,
        mean=mean,
        sd=sd,
        standard_error=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        wins=win_count,
        win_rate=win_rate,
        win_rate_ci_low=win_low,
        win_rate_ci_high=win_high,
    )


# --------------------------------------------------------------------------- bootstrap


def iso_week_blocks(entry_dates: Sequence[date]) -> list[tuple[int, ...]]:
    """Indices of the trades grouped by the ISO (year, week) of their entry date, in
    chronological order of the weeks (input order inside a week)."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, day in enumerate(entry_dates):
        iso = day.isocalendar()
        groups.setdefault((iso.year, iso.week), []).append(index)
    return [tuple(groups[key]) for key in sorted(groups)]


@dataclass(frozen=True)
class BootstrapResult:
    """Block-bootstrap percentile CIs of the expectancy (mean R) and the profit factor.

    ``profit_factor`` values may be ``inf`` (no losing trade); ``None`` when undefined.
    ``prob_expectancy_le_zero`` is the share of resamples with expectancy <= 0.
    """

    n_trades: int
    n_blocks: int
    n_resamples: int
    confidence: float
    seed: int
    expectancy: float | None
    expectancy_ci_low: float | None
    expectancy_ci_high: float | None
    prob_expectancy_le_zero: float | None
    profit_factor: float | None
    profit_factor_ci_low: float | None
    profit_factor_ci_high: float | None


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _profit_factor(gain: FloatArray, loss: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(loss > 0, gain / np.where(loss > 0, loss, 1.0), np.inf)
    return np.where((gain == 0) & (loss == 0), np.nan, ratio)


def _interval(values: FloatArray, confidence: float) -> tuple[float | None, float | None]:
    finite_or_inf = values[~np.isnan(values)]
    if finite_or_inf.size == 0:
        return None, None
    alpha = 1.0 - confidence
    low, high = np.quantile(finite_or_inf, [alpha / 2.0, 1.0 - alpha / 2.0], method="inverted_cdf")
    return float(low), float(high)


def block_bootstrap(
    entry_dates: Sequence[date],
    r_values: Sequence[float | None],
    pnl_values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap whole blocks of trades (one block per ISO entry week, with replacement,
    as many blocks as observed) to keep within-week dependence; percentile CIs
    (inverted CDF, no interpolation).

    Expectancy is the mean of the non-``None`` ``r_values`` of a resample; the profit
    factor uses ``pnl_values`` (gains / |losses|) of every trade, as the report does.
    """
    n = len(entry_dates)
    if len(r_values) != n or len(pnl_values) != n:
        raise ValueError("entry_dates, r_values and pnl_values must have the same length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if n == 0:
        return BootstrapResult(
            n_trades=0,
            n_blocks=0,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
            expectancy=None,
            expectancy_ci_low=None,
            expectancy_ci_high=None,
            prob_expectancy_le_zero=None,
            profit_factor=None,
            profit_factor_ci_low=None,
            profit_factor_ci_high=None,
        )
    blocks = iso_week_blocks(entry_dates)
    r = np.array([np.nan if v is None else v for v in r_values], dtype=np.float64)
    pnl = np.asarray(pnl_values, dtype=np.float64)
    r_sum = np.array([np.nansum(r[list(b)]) for b in blocks])
    r_count = np.array([np.count_nonzero(~np.isnan(r[list(b)])) for b in blocks], dtype=float)
    gain = np.array([pnl[list(b)][pnl[list(b)] > 0].sum() for b in blocks])
    loss = np.array([-pnl[list(b)][pnl[list(b)] < 0].sum() for b in blocks])

    k = len(blocks)
    rng = _rng(seed)
    rows = max(1, _MAX_BATCH_ELEMENTS // k)
    expectancies: list[FloatArray] = []
    factors: list[FloatArray] = []
    done = 0
    while done < n_resamples:
        size = min(rows, n_resamples - done)
        picks = rng.integers(0, k, size=(size, k))
        counts = r_count[picks].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            expectancies.append(np.where(counts > 0, r_sum[picks].sum(axis=1) / counts, np.nan))
        factors.append(_profit_factor(gain[picks].sum(axis=1), loss[picks].sum(axis=1)))
        done += size
    sample_e = np.concatenate(expectancies)
    sample_pf = np.concatenate(factors)

    valid_e = sample_e[~np.isnan(sample_e)]
    point_e = float(np.nanmean(r)) if np.any(~np.isnan(r)) else None
    point_pf = float(_profit_factor(np.array([gain.sum()]), np.array([loss.sum()]))[0])
    e_low, e_high = _interval(sample_e, confidence)
    pf_low, pf_high = _interval(sample_pf, confidence)
    return BootstrapResult(
        n_trades=n,
        n_blocks=k,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        expectancy=point_e,
        expectancy_ci_low=e_low,
        expectancy_ci_high=e_high,
        prob_expectancy_le_zero=float(np.mean(valid_e <= 0)) if valid_e.size else None,
        profit_factor=None if math.isnan(point_pf) else point_pf,
        profit_factor_ci_low=pf_low,
        profit_factor_ci_high=pf_high,
    )


@dataclass(frozen=True)
class DrawdownMonteCarlo:
    """Distribution of the maximum drawdown of resampled return paths (fractions)."""

    n_sims: int
    mean_block_length: float
    seed: int
    observed: float
    p5: float
    p50: float
    p95: float
    threshold: float
    prob_drawdown_ge_threshold: float


def monte_carlo_drawdown(
    returns: Sequence[float],
    *,
    mean_block_length: float = 1.0,
    n_sims: int = 10_000,
    threshold: float = 0.15,
    seed: int = 0,
) -> DrawdownMonteCarlo:
    """Max-drawdown percentiles by the stationary bootstrap (Politis & Romano 1994).

    Each path has ``len(returns)`` steps, compounds from 1 and starts its peak at 1. A
    new block starts with probability ``1 / mean_block_length`` (circular wrap), so
    ``mean_block_length = 1`` is the i.i.d. resampling used for per-trade returns (as
    fractions of equity); use e.g. 5-20 for daily returns to keep autocorrelation.
    """
    if not returns:
        raise ValueError("returns must not be empty")
    if mean_block_length < 1.0:
        raise ValueError("mean_block_length must be >= 1")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    r = np.asarray(returns, dtype=np.float64)
    length = r.size
    rng = _rng(seed)
    restart_probability = 1.0 / mean_block_length
    current = rng.integers(0, length, size=n_sims)
    equity = 1.0 + r[current]
    peak = np.maximum(equity, 1.0)
    worst = 1.0 - equity / peak
    for _ in range(1, length):
        restart = rng.random(n_sims) < restart_probability
        jump = rng.integers(0, length, size=n_sims)
        current = np.where(restart, jump, (current + 1) % length)
        equity = equity * (1.0 + r[current])
        np.maximum(peak, equity, out=peak)
        np.maximum(worst, 1.0 - equity / peak, out=worst)
    p5, p50, p95 = (float(v) for v in np.quantile(worst, [0.05, 0.5, 0.95]))
    return DrawdownMonteCarlo(
        n_sims=n_sims,
        mean_block_length=mean_block_length,
        seed=seed,
        observed=max_drawdown(_equity_path(returns)),
        p5=p5,
        p50=p50,
        p95=p95,
        threshold=threshold,
        prob_drawdown_ge_threshold=float(np.mean(worst >= threshold)),
    )


# --------------------------------------------------------------------------- Sharpe inference


def _sharpe_variance_factor(sr: float, skew: float, kurtosis: float) -> float:
    factor = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if factor <= 0:
        raise ValueError(f"non-positive Sharpe variance factor {factor} (check skew/kurtosis)")
    return factor


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """PSR = ``Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2))``.

    ``sr`` and ``sr_benchmark`` per period, ``n_obs`` = T periods, raw ``kurtosis``.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    factor = _sharpe_variance_factor(sr, skew, kurtosis)
    return _NORMAL.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(factor))


def _expected_max_standard_normal(n_trials: float) -> float:
    """``(1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e))`` (Bailey et al. 2014)."""
    return (1.0 - EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials) + EULER_GAMMA * (
        _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )


def expected_max_sharpe(n_trials: float, trials_sr_variance: float) -> float:
    """SR0: expected maximum Sharpe of ``n_trials`` zero-skill trials whose Sharpe
    ratios have variance ``trials_sr_variance`` (same units as the Sharpe ratios).
    ``n_trials == 1`` means no selection: SR0 = 0."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if trials_sr_variance < 0:
        raise ValueError("trials_sr_variance must be >= 0")
    if n_trials == 1:
        return 0.0
    return math.sqrt(trials_sr_variance) * _expected_max_standard_normal(n_trials)


@dataclass(frozen=True)
class DeflatedSharpe:
    """DSR = PSR with the benchmark SR0 (all Sharpe ratios per period)."""

    sr: float
    sr0: float
    dsr: float
    n_trials: float
    n_obs: int


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    *,
    n_trials: float,
    trials_sr_variance: float,
) -> DeflatedSharpe:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014); per-period units throughout."""
    sr0 = expected_max_sharpe(n_trials, trials_sr_variance)
    dsr = probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurtosis)
    return DeflatedSharpe(sr=sr, sr0=sr0, dsr=dsr, n_trials=n_trials, n_obs=n_obs)


def effective_trials(avg_correlation: float, n_trials: int) -> float:
    """Effective number of independent trials ``rho + (1 - rho) M`` for ``M`` trials
    with average pairwise correlation ``rho`` (clamped to ``[0, 1]``)."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    rho = min(1.0, max(0.0, avg_correlation))
    return rho + (1.0 - rho) * n_trials


def min_track_record_length(
    sr: float,
    sr_benchmark: float,
    skew: float,
    kurtosis: float,
    *,
    confidence: float = 0.95,
) -> float:
    """MinTRL in periods: ``1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (Z_c / (SR - SR*))^2``.

    The track record length at which PSR reaches ``confidence``; ``inf`` when
    ``sr <= sr_benchmark``.
    """
    if sr <= sr_benchmark:
        return math.inf
    factor = _sharpe_variance_factor(sr, skew, kurtosis)
    return 1.0 + factor * (_NORMAL.inv_cdf(confidence) / (sr - sr_benchmark)) ** 2


def min_backtest_length(n_trials: float, target_sharpe: float) -> float:
    """MinBTL in YEARS: backtest length below which ``n_trials`` zero-skill trials are
    expected to show a best ANNUALISED Sharpe of ``target_sharpe`` (Bailey et al. 2014):
    ``((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))^2 / target^2``."""
    if n_trials <= 1:
        raise ValueError("n_trials must be > 1")
    if target_sharpe <= 0:
        raise ValueError("target_sharpe must be > 0")
    return (_expected_max_standard_normal(n_trials) / target_sharpe) ** 2


# --------------------------------------------------------------------------- PBO / CSCV


@dataclass(frozen=True)
class PBOResult:
    """CSCV result: ``pbo`` = share of splits whose in-sample best is at or below the
    out-of-sample median (logit <= 0). ``slope``/``intercept`` regress the OOS Sharpe
    of the selected configuration on its IS Sharpe (per period; ``None`` when IS has no
    spread); ``prob_oos_loss`` = share of splits where the selected one has OOS Sharpe < 0.
    """

    pbo: float
    n_combinations: int
    n_configs: int
    n_blocks: int
    median_logit: float
    slope: float | None
    intercept: float | None
    prob_oos_loss: float


def _sharpe_from_sums(total: FloatArray, squares: FloatArray, count: FloatArray) -> FloatArray:
    mean = total / count
    variance = np.maximum((squares - total * mean) / (count - 1.0), 0.0)
    sd = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mean / np.where(sd > 0, sd, 1.0), 0.0)


def probability_of_backtest_overfitting(
    returns: Sequence[Sequence[float]] | FloatArray, *, n_blocks: int = 16
) -> PBOResult:
    """PBO by combinatorially symmetric cross-validation (Bailey et al. 2017).

    ``returns`` is a ``T x N`` matrix of aligned per-period returns of N configurations.
    Rows are split into ``n_blocks`` contiguous blocks (``numpy.array_split``: no row
    dropped); for each of the ``C(S, S/2)`` choices of in-sample blocks the configuration
    with the best in-sample Sharpe is ranked out of sample (average rank on ties),
    ``omega = rank / (N + 1)`` and ``logit = ln(omega / (1 - omega))``. A configuration
    with zero dispersion has Sharpe 0.
    """
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns must be a T x N matrix")
    periods, configs = matrix.shape
    if configs < 2:
        raise ValueError("PBO needs at least 2 configurations")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number >= 2")
    if periods < 2 * n_blocks:
        raise ValueError(f"need at least {2 * n_blocks} periods for {n_blocks} blocks")

    parts = np.array_split(np.arange(periods), n_blocks)
    block_sum = np.stack([matrix[p].sum(axis=0) for p in parts])
    block_sq = np.stack([(matrix[p] ** 2).sum(axis=0) for p in parts])
    block_n = np.array([float(p.size) for p in parts])

    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    indicator = np.zeros((len(combos), n_blocks))
    for row, combo in enumerate(combos):
        indicator[row, list(combo)] = 1.0
    is_sum, is_sq, is_n = indicator @ block_sum, indicator @ block_sq, indicator @ block_n
    oos_sum = block_sum.sum(axis=0) - is_sum
    oos_sq = block_sq.sum(axis=0) - is_sq
    oos_n = block_n.sum() - is_n
    is_sr = _sharpe_from_sums(is_sum, is_sq, is_n[:, None])
    oos_sr = _sharpe_from_sums(oos_sum, oos_sq, oos_n[:, None])

    rows = np.arange(len(combos))
    best = np.argmax(is_sr, axis=1)
    selected_oos = oos_sr[rows, best]
    below = (oos_sr < selected_oos[:, None]).sum(axis=1)
    ties = (oos_sr == selected_oos[:, None]).sum(axis=1)
    omega = (below + (ties + 1) / 2.0) / (configs + 1)
    logits = np.log(omega / (1.0 - omega))

    selected_is = is_sr[rows, best]
    slope = intercept = None
    if float(np.ptp(selected_is)) > 0:
        fit = np.polyfit(selected_is, selected_oos, 1)
        slope, intercept = float(fit[0]), float(fit[1])
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        n_combinations=len(combos),
        n_configs=configs,
        n_blocks=n_blocks,
        median_logit=float(np.median(logits)),
        slope=slope,
        intercept=intercept,
        prob_oos_loss=float(np.mean(selected_oos < 0)),
    )


# --------------------------------------------------------------------------- multiple testing


def _check_p_values(p_values: Sequence[float]) -> None:
    if any(not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p-values must be in [0, 1]")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error), in input order."""
    _check_p_values(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def bhy_adjust(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg-Yekutieli adjusted p-values (FDR under any dependence),
    ``min_(j >= i) m c(m) p_(j) / j`` with ``c(m) = sum 1/k``, in input order."""
    _check_p_values(p_values)
    m = len(p_values)
    if m == 0:
        return []
    harmonic = math.fsum(1.0 / k for k in range(1, m + 1))
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        running = min(running, min(1.0, m * harmonic * p_values[index] / (rank + 1)))
        adjusted[index] = running
    return adjusted
