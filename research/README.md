# Research protocol: strategy family v2

Every strategy idea is registered **before** it is run, every run is logged, and the
final test data (the lockbox) is opened **once** per finalist. This keeps the count of
attempts honest, so the statistics can correct for multiple testing. Owner decision
2026-10-02 (`docs/DECISIONS.md`, "Strategy v1.0.0 result and family v2").

"Nothing passes" is an acceptable outcome.

## Quick path

1. **Register.** Write `research/hypotheses/<id>.yaml` (rationale, sources, exact rules,
   costs, config path, pass rule) and commit it before any run. Its full config lives in
   `research/configs/<id>.yaml`.
2. **Develop.** Run the registered configs on the development window only. Each run is
   appended to `research/trials.jsonl` (never edited).
3. **Gate.** Apply the development acceptance table below. The owner picks at most two
   finalists.
4. **Lockbox.** For each finalist, with a clean git tree at a recorded commit, open the
   lockbox window once and apply its pass rule, which was written before opening.
5. **Forward.** A lockbox pass goes to forward paper trading on the embargo window.

## Data windows

| Window | Dates | Use |
| --- | --- | --- |
| Development | 2016-11 .. 2022-12 (2016-01..2016-10 = indicator warm-up; SIP daily starts 2016-01-04) | Walk-forward evaluation, all statistics, any number of looks |
| Lockbox | 2023-01 .. 2026-06 | Opened **once** per finalist, frozen commit |
| Embargo | 2026-07 onward | Forward paper trading only, never backtested |

## Trial budget

| Rule | Value |
| --- | --- |
| Registered configurations in total | at most 12 |
| Strategy v1.0.0 (rejected) | counts as 1 (`V1_TRIAL_COUNT`, not written to the log) |
| Pre-registered family v2 | 7 configurations (below), so 8 of 12 used |
| What counts as a trial | each distinct `(hypothesis_id, config_hash)` in the log; rerunning the same config is not a new trial, changing it is |
| N for the deflated Sharpe ratio | `count_trials(research/trials.jsonl)` (`backtest/registry.py`) |

## Development acceptance (all must pass)

Measured on the pooled walk-forward out-of-sample trades of the development window.

| Criterion | Threshold |
| --- | --- |
| Expectancy | >= 0.10 R |
| t statistic of the mean R | >= 2 |
| Block-bootstrap 95% CI of the expectancy (ISO-week blocks) | lower bound > 0 |
| Profit factor | >= 1.2 |
| Monte-Carlo 95th-percentile max drawdown | <= 15% |
| Deflated Sharpe ratio (N from the registry) | >= 0.95 |
| PBO across the family (CSCV, 16 blocks) | < 0.5 |
| Random-entry benchmark (1,000 seeded simulations) | >= 95th percentile |
| Cost stress | still positive at 2x cost (also reported at 3x, and the break-even bps) |
| Buy-and-hold of the same ETFs | Sharpe >= B&H, **or** max drawdown <= 1/2 B&H **and** CAGR >= 60% of B&H |

## Lockbox pass rule (written before opening)

| Criterion | Threshold |
| --- | --- |
| Expectancy | > 0 R |
| Profit factor | > 1.1 |
| Expectancy inside the development bootstrap CI | yes |
| Max drawdown | <= 15% |

The lockbox guard (`assert_lockbox_allowed`) refuses to open when the hypothesis is not
registered, has no lockbox pass rule, the commit is not recorded, the git tree is dirty,
or the lockbox was already opened for that hypothesis.

## Pre-registered hypotheses

Universe SPY, QQQ, IWM; daily SIP bars; signal on the complete daily bar, market order
at the next session open; disaster stop entry - 3 x ATR(14); costs 5 bps per side
(stress x2, x3).

| Id | Family | Entry | Exit |
| --- | --- | --- | --- |
| `mr_a1_ibs` | MR-A (IBS) | IBS < 0.2 | IBS > 0.8, or 5 sessions |
| `mr_a2_ibs_sma200` | MR-A (IBS) | IBS < 0.2 and close > SMA200 | IBS > 0.8, or 5 sessions |
| `mr_b1_rsi2_lt5` | MR-B (RSI(2)) | close > SMA200 and RSI(2) < 5 | close > SMA5, or 10 sessions |
| `mr_b2_rsi2_lt10` | MR-B (RSI(2)) | close > SMA200 and RSI(2) < 10 | close > SMA5, or 10 sessions |
| `mr_b3_rsi2_lt15` | MR-B (RSI(2)) | close > SMA200 and RSI(2) < 15 | close > SMA5, or 10 sessions |
| `tf_t1_sma200` | TF (baseline) | close > SMA200 | close < SMA200 |
| `tf_t2_sma210` | TF (baseline) | close > SMA210 (~10 months) | close < SMA210 |

## Files

| Path | Content |
| --- | --- |
| `hypotheses/<id>.yaml` | Pre-registration (`backtest.registry.Hypothesis`) |
| `configs/<id>.yaml` | Full `AppConfig` of the hypothesis, `strategy_version 2.0.0-<id>` |
| `trials.jsonl` | Append-only trial log (`backtest.registry.TrialRecord`, one JSON object per line) |

Statistics: `backtest/stats.py`; benchmarks: `backtest/benchmark.py`,
`backtest/random_entry.py`. Backtest results do not imply future profitability.
