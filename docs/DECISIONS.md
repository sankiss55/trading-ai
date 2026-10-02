# Decisions

Owner decisions (master spec v2.1, sec. 59) and phase approvals (sec. 56.4).
Only the project owner can check an item or record an approval. The coding agent
never invents these values: until they are set, the matching `config.yaml` keys stay
`null` and trading cannot be enabled (sec. 7.4, AC-20).

## Owner decisions (sec. 59)

- [x] Whitelist symbols (owner, 2026-10-02)
- [x] Data feed: iex or sip (owner, 2026-10-02)
- [x] holding_mode: intraday or swing (owner, 2026-10-02)
- [x] flatten_minutes_before_close (if intraday) (owner, 2026-10-02)
- [x] No-entry windows at the start and end of the session (owner, 2026-10-02)
- [x] Primary and confirmation timeframes (owner, 2026-10-02)
- [x] Indicators and all their parameters (owner, 2026-10-02)
- [x] Entry rules (owner, 2026-10-02)
- [x] No-trade rules and their thresholds (owner, 2026-10-02)
- [x] Exit rules: stop ATR multiplier, take-profit R multiple, time stop, reversal (owner, 2026-10-02)
- [x] Cooldown after an exit (owner, 2026-10-02)
- [x] Minimum minutes per aggregated bar (owner, 2026-10-02)
- [x] Entry order type: market or limit (owner, 2026-10-02)
- [x] Risk per trade (owner, 2026-10-02)
- [x] Maximum daily and weekly loss (owner, 2026-10-02)
- [x] Maximum drawdown (owner, 2026-10-02)
- [x] Maximum positions (owner, 2026-10-02)
- [x] Maximum total exposure, per-symbol exposure and aggregate open risk (owner, 2026-10-02)
- [x] Slippage buffer (owner, 2026-10-02)
- [x] Price, volume and spread filters (owner, 2026-10-02)
- [x] Dividend adjustment convention (owner, 2026-10-02)
- [x] Backtest period, walk-forward windows and sec. 45.4 thresholds (owner, 2026-10-02)
- [ ] Anthropic model for the filter. Provider decided by the owner on 2026-10-02:
  Claude Code CLI with the subscription session (`ai.provider = "claude_cli"`, see
  "AI provider" below); the model id (`ai.model_id`) is still VERIFICAR
- [x] Use of news in the snapshot (yes/no) (owner, 2026-10-02)
- [x] Daily and monthly AI budget, and maximum calls per day (owner, 2026-10-02)
- [x] Sec. 46.4 criteria to activate the veto (owner, 2026-10-02)
- [x] Notification policy
- [x] Data retention (owner, 2026-10-02)
- [x] Minimum paper trading and shadow mode periods (owner, 2026-10-02)
- [x] Backup frequency and retention (owner, 2026-10-02)

## Owner decisions 2026-10-02

Approved by the project owner on 2026-10-02 and written into `config.yaml`
(`config_version` 2.2.0, 2.3.0 for group 18; `strategy_version` 1.0.0, `risk_version` 1.0.0; every key is
marked `# OWNER_DECISION 2026-10-02`). How they were decided:

- **Chosen one by one by the owner:** `market_data.feed`, `universe.whitelist`,
  `session.no_entry_first_minutes`, `session.no_entry_last_minutes`, `strategy.holding_mode`,
  `strategy.flatten_minutes_before_close` and `strategy.primary_timeframe`.
- **Approved as a package:** every other value in the table, proposed by the coding agent and
  approved by the owner as a whole.
- **Approved separately by the owner:** group 16 (price adjustment convention), group 17
  (notification policy) and group 18 (liquidity feed, `config_version` 2.3.0).

Units: every `*_pct` of `risk`, `strategy.no_trade_thresholds` and `backtest` is a
fraction (`0.005` = 0.5 %; see `domain/risk/risk_engine.py` and `backtest/report.py`);
`*_bps` are basis points. These values are decisions to evaluate in backtest and paper
trading, not evidence of profitability (sec. 45.4).

| # | Group | Key | Value | How decided |
| --- | --- | --- | --- | --- |
| 1 | Data feed | `market_data.feed` | `iex` | Owner, one by one |
| 2 | Universe | `universe.whitelist` | `SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, META` | Owner, one by one |
| 2 | Universe | `universe.min_price` | 10 | Owner, package |
| 2 | Universe | `universe.max_price` | 2000 | Owner, package |
| 2 | Universe | `universe.min_avg_daily_volume` | 5000000 shares/day, measured on SIP (consolidated) historical daily bars (`universe.liquidity_feed`, group 18) | Owner, package |
| 2 | Universe | `universe.max_spread_bps` | 20 bps | Owner, package |
| 3 | Session | `session.no_entry_first_minutes` | 15 | Owner, one by one |
| 3 | Session | `session.no_entry_last_minutes` | 30 | Owner, one by one |
| 4 | Strategy | `strategy.holding_mode` | `intraday` | Owner, one by one |
| 4 | Strategy | `strategy.flatten_minutes_before_close` | 10 | Owner, one by one |
| 4 | Strategy | `strategy.primary_timeframe` | `5Min` | Owner, one by one |
| 4 | Strategy | `strategy.confirmation_timeframe` | `15Min` | Owner, package |
| 4 | Strategy | `strategy.cooldown_bars` | 3 | Owner, package |
| 4 | Strategy | `strategy.min_minutes_per_bar` | 3 | Owner, package |
| 4 | Strategy | `strategy.indicators.ema_fast` / `ema_slow` | 9 / 21 | Owner, package |
| 4 | Strategy | `strategy.indicators.rsi_period` / `atr_period` | 14 / 14 | Owner, package |
| 4 | Strategy | `strategy.indicators.volume_avg_period` | 20 | Owner, package |
| 5 | Entry rules (all true) | `ENTRY_TREND_01` | `ema_fast[-1] > ema_slow[-1]` (primary) | Owner, package |
| 5 | Entry rules (all true) | `ENTRY_PRICE_01` | `close[-1] > ema_fast[-1]` (primary) | Owner, package |
| 5 | Entry rules (all true) | `ENTRY_MOMENTUM_01` | `rsi[-1] > 50` (primary) | Owner, package |
| 5 | Entry rules (all true) | `ENTRY_VOLUME_01` | `volume[-1] > volume_avg[-1]` (primary) | Owner, package |
| 5 | Entry rules (all true) | `ENTRY_CONFIRM_01` | `ema_fast_confirm[-1] > ema_slow_confirm[-1]` (confirmation) | Owner, package |
| 6 | No-trade rules (any true) | `NOTRADE_RSI_HIGH` | `rsi[-1] > rsi_overbought` | Owner, package |
| 6 | No-trade rules (any true) | `NOTRADE_VOLATILITY` | `atr_pct[-1] > max_atr_pct` | Owner, package |
| 6 | No-trade rules (any true) | `NOTRADE_GAP` | `gap_pct[-1] > max_gap_pct` | Owner, package |
| 6 | No-trade thresholds | `rsi_overbought` / `max_atr_pct` / `max_gap_pct` | 70 / 0.01 / 0.02 | Owner, package |
| 7 | Exit | `strategy.exit.stop_method` | `atr` (unchanged) | Owner, package |
| 7 | Exit | `strategy.exit.stop_atr_multiplier` | 1.5 | Owner, package |
| 7 | Exit | `strategy.exit.take_profit_r_multiple` | 2.0 | Owner, package |
| 7 | Exit | `strategy.exit.time_stop_bars` | 12 | Owner, package |
| 7 | Exit | `strategy.exit.exit_on_signal_reversal` | `true` | Owner, package |
| 7 | Exit | `EXIT_REVERSAL_01` (`strategy.exit_rules`) | `ema_fast[-1] < ema_slow[-1]` (primary) | Owner, package |
| 7 | Exit | `strategy.exit.min_tp_distance_ticks` | 2 (unchanged) | Owner, package |
| 8 | Risk | `risk.risk_per_trade_pct` | 0.005 | Owner, package |
| 8 | Risk | `risk.max_daily_loss_pct` | 0.02 | Owner, package |
| 8 | Risk | `risk.max_weekly_loss_pct` | 0.04 | Owner, package |
| 8 | Risk | `risk.max_drawdown_pct` | 0.10 | Owner, package |
| 8 | Risk | `risk.max_positions` | 3 | Owner, package |
| 8 | Risk | `risk.max_total_exposure_pct` | 0.90 | Owner, package |
| 8 | Risk | `risk.max_symbol_exposure_pct` | 0.30 | Owner, package |
| 8 | Risk | `risk.max_aggregate_open_risk_pct` | 0.015 | Owner, package |
| 8 | Risk | `risk.slippage_buffer_bps` | 5 bps | Owner, package |
| 9 | Execution | `execution.entry_order_type` | `market` (unchanged) | Owner, package |
| 10 | AI | `ai.include_news` | `false` (unchanged) | Owner, package |
| 10 | AI | `ai.max_calls_per_day` | 50 | Owner, package |
| 10 | AI | `ai.max_cost_usd_per_day` | 5 USD | Owner, package |
| 10 | AI | `ai.max_cost_usd_per_month` | 100 USD | Owner, package |
| 11 | Retention | `retention.market_bars_days` | 365 | Owner, package |
| 11 | Retention | `retention.ai_snapshots_days` | 365 | Owner, package |
| 11 | Retention | `retention.logs_days` | 90 | Owner, package |
| 12 | Backtest | `backtest.start_date` / `end_date` | 2022-01-01 / 2026-06-30 | Owner, package |
| 12 | Backtest | `backtest.out_of_sample_start` | 2025-07-01 | Owner, package |
| 12 | Backtest | `backtest.walk_forward_train_months` / `walk_forward_test_months` | 12 / 3 | Owner, package |
| 13 | Sec. 45.4 thresholds | `backtest.min_expectancy_r` | 0.10 R | Owner, package |
| 13 | Sec. 45.4 thresholds | `backtest.min_profit_factor` | 1.2 | Owner, package |
| 13 | Sec. 45.4 thresholds | `backtest.max_drawdown_pct` | 0.15 | Owner, package |
| 13 | Sec. 45.4 thresholds | `backtest.min_trades_out_of_sample` | 100 | Owner, package |
| 14 | Paper / sec. 46.4 | `paper.min_paper_trading_days` | 30 | Owner, package |
| 14 | Paper / sec. 46.4 | `paper.min_shadow_signals` | 100 | Owner, package |
| 14 | Paper / sec. 46.4 | `paper.min_expectancy_improvement_r` | 0.05 R | Owner, package |
| 14 | Paper / sec. 46.4 | `paper.min_valid_ai_response_rate` | 0.95 | Owner, package |
| 15 | Backups | `backups.frequency` | `daily_after_close` | Owner, package |
| 15 | Backups | `backups.retention_days` | 30 | Owner, package |
| 16 | Price adjustment (sec. 44) | `market_data.adjustment` | `split` (split-adjusted only; dividends not adjusted: intraday, no overnight positions). Backtest and live must use the same convention | Owner, separate approval |
| 17 | Notifications (sec. 40) | `notifications.policy` | `critical_plus_daily` (critical events immediately, daily summary, error digest every `error_digest_minutes`) | Owner, separate approval |
| 18 | Liquidity feed (sec. 10.2.1, 12) | `universe.liquidity_feed` | `sip`: the daily bars of the liquidity filter (`min_avg_daily_volume`) come from SIP; minute bars and the strategy keep `market_data.feed = iex` (sec. 10.2.2). Reason: IEX daily volume is a small slice of the market (SPY about 1.2-1.4M shares/day on IEX vs about 60M consolidated), so a 5M threshold on IEX would reject every whitelisted symbol. Threshold unchanged | Owner, separate approval |

### Alpaca historical data

- **SIP historical daily bars: access verified on 2026-10-02** with the project paper
  account (`AlpacaMarketData`, feed `sip`, adjustment `split`): SPY 2025-06-02..2025-06-03
  returned 2 daily bars (61,630,502 and 63,606,204 shares). Minute bars stay on IEX.
- The downloader (`python -m backtest.data download`) takes `--feed` (minute bars,
  `market_data.feed`) and `--daily-feed` (daily bars, `universe.liquidity_feed`), both
  required. Both feeds are recorded in `manifest.json` (resume refused on a mismatch),
  and the dataset copy of the manifest makes the loader tag daily bars with their real
  feed (`sip`) and minute bars with `iex`. The aggregator's `FEED_MISMATCH` check only
  sees minute bars (daily bars are never streamed to it).
- The Phase 1 backtest does not apply the universe checks yet, so the daily bars are
  downloaded and loaded but not consumed by the runner.

Still open after this approval: `ai.model_id` (VERIFICAR), `ai.effort` and the cache prices of `ai.pricing` (VERIFICAR,
not owner decisions).

## Strategy v1.0.0 result and family v2 (owner decision 2026-10-02)

**v1.0.0 rejected.** 5-min EMA trend, intraday, flat 15:50: official run 2022-2026 hit the 10% drawdown
halt on 2022-03-11 (283 trades, PF 0.56, -0.28R); segmented research run lost in every walk-forward
window and every year (5,656 trades, win rate 33%, PF 0.54, -0.34R). Diagnosis: 1R ~ 0.27% of price,
round-trip cost ~ 0.37R, gross edge ~ 0. Web research found intraday edges on SPY/QQQ of 2-5 bps gross
(ORB, "Beat the Market", last half hour), below costs.

**Family v2: daily swing mean reversion on index ETFs.** Approved by the owner as part of the
research plan (the values below were proposed by the coding agent and approved as a whole):

| Topic | Value |
| --- | --- |
| Universe | SPY, QQQ, IWM (mega caps excluded: survivorship bias, index-level mechanism) |
| Holding mode | `swing` (overnight gap risk accepted; disaster stop GTC) |
| Timeframe / feed | primary `1Day`, SIP daily bars (decided overnight on the previous complete bar: same feed live and in backtest) |
| Execution | signal after the close on the complete daily bar; market order at the next session open; exits the same way |
| Protection | bracket kept (spec 23.1): disaster stop entry - 3 x ATR(14, daily); take profit 10R (exits are rule-based) |
| Sizing | risk 1.0% per trade; max 0.33 equity per ETF; max 3 positions; total exposure <= 0.99 (cash only) |
| Costs | 5 bps per side base; stress x2 and x3; break-even bps reported |
| Price adjustment | `split` (dividends not credited: small conservative bias) |
| Data split | development 2016-11..2022-12 (owner decision 2026-10-02 after verifying that Alpaca SIP daily history starts 2016-01-04: January-October 2016 is indicator warm-up only, SMA200 needs ~200 sessions); lockbox 2023-01..2026-06 opened once per finalist with a frozen commit and a pass rule written before opening; embargo 2026-07+ = forward paper trading |
| Trial budget | <= 12 registered configurations in total (v1.0.0 counts as 1) |
| Acceptance (extends sec. 45.4) | pooled development walk-forward OOS: expectancy >= 0.10R and t >= 2 and bootstrap 95% CI lower bound > 0; PF >= 1.2; Monte-Carlo 95th-percentile max DD <= 15%; DSR >= 0.95 (N from the trial registry); PBO < 0.5; >= 95th percentile vs random entry; positive at 2x cost; vs buy-and-hold of the same ETFs: Sharpe >= B&H, or max DD <= 1/2 B&H with CAGR >= 60% of B&H. Lockbox pass rule: expectancy > 0, PF > 1.1, inside the development CI, max DD <= 15%. "Nothing passes" is an acceptable outcome |

Pre-registered hypotheses (7 configurations + v1.0.0 = 8 of 12):
- **MR-A (IBS)**: entry IBS < 0.2; A1 without filter, A2 with close > SMA200. Exit IBS > 0.8 or after 5 days.
- **MR-B (Connors RSI(2))**: close > SMA200 and RSI(2) < 5 / 10 / 15 (B1-B3). Exit close > SMA5 or after 10 days.
- **TF (baseline)**: hold while close > SMA200 (T1) or SMA210 (T2, ~10 months); exit when the close falls below it.

`config.yaml` stays on strategy v1.0.0 until a v2 finalist passes the lockbox and the owner approves it.

**Development evaluation 2026-10-02 (7 trials recorded in `research/trials.jsonl`; report in
`research/results/dev_eval_2026-10-02.txt`): all 7 FAIL.** Best: `mr_a1_ibs` (494 gate trades,
E[R] +0.043, t 2.16, PF 1.29, Sharpe 0.71 vs buy-and-hold 0.57, max DD 7.6% vs 33.5%, CAGR 4.0% vs 10.4%,
100th percentile vs random entry); it fails expectancy >= 0.10R, bootstrap CI lower bound > 0 (-0.004)
and DSR >= 0.95 (0.78). The criteria were not changed after seeing the results.

**Owner exception 2026-10-02:** `mr_a1_ibs` is sent to the lockbox as the only finalist although it
failed 3 development criteria. It is judged once with its pre-written lockbox rule (expectancy > 0,
PF > 1.1, inside the development CI, max DD <= 15%). If it fails it is discarded permanently.

**Owner decision 2026-10-02: build the paper-trading system (Phases 2-7) after the lockbox, whatever its
result.** This deviates from sec. 55 (Phase 1 exit criterion not met): the goal of paper trading is
to validate the system (orders, stops, recovery, reconciliation) with paper money only; any strategy
run in paper is labelled with its validation status. Live trading stays blocked (sec. 48).

## Change log (sec. 58.6)

Every change to strategy, risk, prompts, model or universe bumps the matching version
and is recorded here.

| Date | Version bumped | Change | Reason | Author |
| --- | --- | --- | --- | --- |
| 2026-10-02 | `config_version` 2.0.0 -> 2.1.0 | `ai.provider = "claude_cli"`, `ai.cli_command`, `ai.cli_max_concurrency = 1`; `ANTHROPIC_API_KEY` removed from `.env.example`; `subprocess` forbidden in `domain/` and `application/` | Owner decision: the AI filter uses the Claude Code CLI session, no API key | Owner (decision); coding agent (config) |
| 2026-10-02 | `config_version` 2.1.0 -> 2.2.0; `strategy_version` 0.1.0 -> 1.0.0; `risk_version` 0.1.0 -> 1.0.0 | Every remaining unconditional OWNER_DECISION of `config.yaml` filled (universe, feed, session, strategy, rules, exit, risk, AI budgets, retention, backtest, paper, backups) and new required key `market_data.adjustment = "split"` (sec. 44), see "Owner decisions 2026-10-02"; `tests/fixtures/config.pending.yaml` keeps the previous all-null config for the refusal tests | Owner decisions 2026-10-02 (groups 1-6 one by one; groups 7-15 approved as a package proposed by the coding agent; group 16 approved separately) | Owner (decision); coding agent (config) |
| 2026-10-02 | `config_version` 2.2.0 -> 2.3.0 | New required key `universe.liquidity_feed = "sip"` (group 18): the liquidity filter uses SIP daily bars, minute bars and strategy stay on IEX; downloader `--daily-feed` (required) and per-timeframe feeds in `manifest.json` | Owner decision 2026-10-02: IEX daily volume is too small for the 5M shares/day threshold | Owner (decision); coding agent (config, downloader) |

## AI provider: Claude Code CLI session (owner decision 2026-10-02)

**Decision (owner, 2026-10-02).** The AI veto filter does not use the Anthropic API or
an API key. It calls the Claude Code CLI headlessly with the Claude subscription session
already logged in on the machine. This is an owner decision, not a technical choice of
the coding agent. `ai.provider = "anthropic_api"` is rejected at startup (fail closed).

**What it is.** One call per signal runs, with `shell=False`:

```text
claude -p --output-format json --system-prompt-file <temp file> --allowedTools "" --model <ai.model_id>
```

The snapshot goes in through STDIN. The system prompt and filter rules go in a temporary
file. The CLI prints a JSON envelope (`result`, `is_error`, `subtype`, ...); the verdict
object is extracted from `result`. The owner's other project uses the same pattern.

**Why.** Owner choice: no API key to manage, and the subscription is already paid.

**Where it will live.** `adapters/claude_cli/claude_filter.py`, implementing the existing
`IAIFilter` port unchanged. It is built in Phase 8, with contract tests that run a fake
CLI executable: tests never use a real session (sec. 58.4). Only `adapters/` may start a
process: `subprocess`, `asyncio.subprocess`, `os.system` and similar are forbidden in
`domain/` and `application/` by `tests/architecture/` and the import-linter contracts.

**Config (sec. 7.3).** `ai.provider`, `ai.cli_command` (VERIFICAR) and
`ai.cli_max_concurrency` (fixed to 1). `ai.model_id` is passed to `--model`.

### Deviations from the spec and how each guarantee is kept

| Spec | Original requirement | With the CLI session |
| --- | --- | --- |
| 6.1 | Official `anthropic` SDK, async client | Claude Code CLI run as a child process from the adapter |
| 6.2 rule 1 | Check at startup that `ai.model_id` answers | A minimal `claude -p` probe at startup with the configured model (Phase 8). If it fails and `ai_mode` is not `DISABLED`, the system cannot reach `RUNNING` with AI |
| 6.4 | API structured outputs (JSON Schema) | CLI JSON envelope, then extract the JSON object from `result`. The pydantic validation of sec. 18.3 stays mandatory and unchanged. VERIFICAR whether the installed CLI has a JSON-schema output flag; use it if it does |
| 6.5 | Cacheable system-prompt prefix, cache tokens reported | Caching cannot be controlled. The stable system prompt goes through `--system-prompt-file`; the variable snapshot goes through STDIN, last |
| 4.4 | Timeout, API error, bad stop reason, invalid JSON = unavailable | Timeout, non-zero exit, `is_error = true`, quota or rate-limit text, and invalid JSON all map to an `AIVerdictResult` with status UNAVAILABLE or INVALID. The adapter never raises (`IAIFilter` contract). `on_unavailable: block` stays the default |
| 17.5 | Retries on retryable errors only | Quota and rate-limit errors are detected by the regex `usage limit\|rate limit\|quota\|too many requests\|overloaded` (case-insensitive). Retries use bounded exponential backoff and never continue past `expires_at_utc` |
| 34 | Tokens and cost from the SDK `usage` object | Tokens and cost come from the CLI envelope when it reports them; otherwise cost is estimated from `ai.pricing`. Budgets (`max_calls_per_day`, `max_cost_usd_*`) are enforced either way, counted from `ai_decisions` |
| 43 | Secrets in `.env`, never sent to Claude | No AI key exists in `.env`. The CLI runs with `--allowedTools ""` (no tools, sec. 4.5), `shell=False`, prompt through STDIN and system prompt through a temporary file. The snapshot holds no secrets (AC-19) |
| Concurrency | Not specified | `ai.cli_max_concurrency = 1`: one CLI call at a time, other signals wait |

### Risks

- **Latency vs. signal TTL.** About 20 s per call was observed in the other project.
  With one call at a time, queued signals plus retries can exceed
  `strategy.signal_ttl_seconds` (120 s). An expired signal fails closed: no entry
  (sec. 17.5, 17.6).
- **Subscription usage limits.** When the session hits its quota, the AI is unavailable
  until the limit resets. With `on_unavailable: block` no new entries happen in that time.
- **CLI version changes.** The envelope format and flags may change between CLI versions
  (VERIFICAR at each upgrade). The contract tests pin the expected envelope.
- **Terms of use.** Whether the subscription terms allow automated use of this kind is
  for the owner to VERIFICAR.

## Phase approvals (sec. 56.4)

| Date | Phase | Approver | Notes |
| --- | --- | --- | --- |

## Technical decisions pending owner confirmation

Technical choices made by the coding agent where the spec leaves room. They are not
owner decisions; the owner confirms or changes them.

| Decision | Where | Rationale |
| --- | --- | --- |
| `NOTRADE_DATA_UNAVAILABLE` built-in no-trade rule: if an owner no-trade rule cannot be evaluated because a value is missing (`None`), the entry is blocked | `domain/strategy/strategy.py` | Sec. 14.3 makes a rule with missing data `False`, which for a no-trade rule would silently disable the protection; blocking restores the fail-closed default of sec. 3.4 |
| RSI = 50 on perfectly flat prices (no gains and no losses in the window) | `domain/market/indicators.py` | Wilder's formula is undefined (0/0) there; 50 is the neutral value, so neither an overbought nor a momentum rule fires on flat data |
| Stop-first literal rule in the simulated broker: if a bar can touch both the stop and the take profit, the stop fills (and a gap through the stop fills at the open) | `adapters/simulation/simulated_broker.py` | Conservative assumption of sec. 45.2.6, applied literally whenever the stop is touchable in the bar, even if the take profit may have been reached first |
| Exposure fractions (`max_total_exposure_pct`, `max_symbol_exposure_pct`, every `*_pct`) are capped at 1.0: no margin or leverage | `domain/risk/risk_engine.py` | Cash account in the MVP; a value above 1 is rejected at startup instead of silently allowing leverage |

## Declared deviations (Phase 1, sec. 56.4.2)

| Deviation | Where | Resolution |
| --- | --- | --- |
| Simulation adapters are built in the backtest runner, not in `app/` (rule 8.3.4) | `backtest/runner.py` | `app/container.py` is Phase 2 scope; the runner moves to it then |
| System exit, end-of-day flatten and limit-entry timeout live in the runner instead of `application/exit_procedures.py` (AC-22 pending) | `backtest/runner.py` | Phase 5 moves them to the shared use cases and the Phase 1 backtest is re-run (sec. 55) |
| Price-increment rule duplicated as an independent broker-side validation | `adapters/simulation/simulated_broker.py` vs `domain/risk/exits.py` | Intentional: the simulated broker mimics Alpaca rejecting invalid prices independently of the domain; both are VERIFICAR |
| 390-minute regular session constant used only to size the warm-up history | `backtest/runner.py` | Follow-up: derive from `SessionDay` open/close |
| Family v2 uses holding mode `swing` and a `1Day` primary timeframe (spec 5.6 recommends intraday for 5-min systems; v2 is a daily system) | research configs for v2 | Owner decision 2026-10-02 |
| Family v2 strategy feed is SIP daily (sec. 10.2 consistency kept: the same feed live and in backtest) | research configs for v2 | Owner decision 2026-10-02 |
| Sec. 45.4 thresholds extended with statistical gates; development/lockbox split replaces the single out-of-sample cut for v2 | research pipeline | Owner decision 2026-10-02 |
| Primary per-trade gate sample = trades of the continuous development run entered inside the walk-forward span (`research/protocol.yaml` `gate_sample: continuous_walk_forward_span`) instead of pooled fresh-account windows; both are reported | `research/protocol.yaml` | Decided 2026-10-02 before any dev trial; owner delegated the choice. Fresh-account windows drop trades open at window ends (synthetic check: TF +0.54R continuous vs -0.51R fresh windows). Chosen for being unbiased, not for its result |
| Daily signals expire at the next session open + `signal_ttl_seconds` instead of bar end + TTL (sec. 13.7): the decision is taken after the close and executed at the next open | `application/market_flow.py` | Technical; follows the owner-approved execution model |
| SIP daily bars (labelled New York midnight) are re-stamped to their session `[open, close)` before use; a missing daily bar becomes an EMPTY session bar (time stop and cooldown still count the session) | `domain/market/session.py`, `backtest/runner.py` | Technical; prevents lookahead and keeps bar counting consistent |
| Daily warm-up uses the flow's full window (`window_size` >= `history_warmup_bars` sessions), so a segment starts with exactly the indicator window of the continuous run; a daily limit entry lives for one session | `backtest/runner.py` | Technical |
| v2 executes at the next open with market orders (open-gap slippage risk) instead of auction orders (MOO/MOC) | daily engine path | Possible later improvement; Alpaca paper does not simulate auctions |
| Part of Phase 2 pulled forward (owner request 2026-10-02): Alpaca historical market data + market calendar adapters, secrets loader and downloader, so the Phase 1 backtest can run on real data. No orders, no trading client usage beyond the calendar, no streaming | `adapters/alpaca/`, `app/secrets.py`, `backtest/data.py` | Rest of Phase 2 (container, SQLite, control CLI, SMTP, broker adapter) unchanged |
