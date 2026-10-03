"""Risk gates end to end on the backtest runner (sec. 49.5): AC-05, AC-15 logic, invariants.

The real runner path (``backtest.runner._Simulator``: SimClock, SimulatedBroker, MarketFlow
with the check library) runs on the synthetic fixture dataset. Hooks record every
``submit_bracket`` call and every flow outcome; the state the checks see is altered only
through the runner's own inputs (broker account, trade book, runtime environment).

* AC-05: a proposed trade that violates a risk limit never reaches the broker. Limits
  violated before the AI step are rejected by ``run_pre_ai_checks``; limits that only
  break after the proposal (equity drop before submission) are caught by
  ``run_execution_checks``, which the Phase 5 Execution Guard runs before every
  submission (here a stand-in hook calls it with the refreshed state).
* AC-15 (logic): the kill switch (``trading_enabled = false``, STOP file, emergency close)
  blocks every entry from the first decision after it is set, without a restart. The
  polling of ``control_poll_seconds`` is the runtime's job (Phase 5, ``app/``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.config import AppConfig
from application.market_flow import BarOutcome, EntryProposal, Rejected, TradeBook
from application.risk_gate import RiskGate
from backtest.data import BacktestData, load_backtest_data
from backtest.runner import (
    SimulationGuardEnvironment,
    SimulationResult,
    _Simulator,
    guard_settings,
)
from domain.guards.checks import (
    CIRCUIT_BREAKER_NOT_NORMAL,
    EMERGENCY_CLOSE_ACTIVE,
    FEED_DISCONNECTED,
    KILL_SWITCH_CODES,
    STOP_FILE_PRESENT,
    TRADING_DISABLED,
    ExecutionFacts,
    IntradayTiming,
    RuntimeFacts,
    SpreadFilter,
    all_passed,
    failed_codes,
    run_execution_checks,
)
from domain.market.session import compute_session_windows
from domain.models import (
    AccountState,
    Bar,
    BracketOrderRequest,
    BrokerOrder,
    CircuitBreakerState,
    DataFeed,
)
from domain.risk.risk_engine import (
    AGGREGATE_OPEN_RISK,
    DAILY_LOSS_LIMIT,
    MAX_DRAWDOWN,
    MAX_POSITIONS,
    RISK_PER_TRADE_EXCEEDED,
    SYMBOL_EXPOSURE,
    TOTAL_EXPOSURE,
    WEEKLY_LOSS_LIMIT,
    PendingEntry,
)
from tests.simulation.helpers import fixture_loaded, make_dataset

CASH = Decimal(100000)
PERIOD = {
    "start_date": date(2025, 1, 2),
    "end_date": date(2025, 1, 15),
    "out_of_sample_start": date(2025, 1, 10),
}


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> BacktestData:
    directory: Path = make_dataset(tmp_path_factory.mktemp("gates"), end=date(2025, 1, 15))
    return load_backtest_data(directory, feed=DataFeed.IEX)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return fixture_loaded(backtest=PERIOD).config


@dataclass
class Run:
    """A runner simulation with its recorded submissions and outcomes."""

    simulator: _Simulator
    submissions: list[tuple[datetime, BracketOrderRequest]] = field(default_factory=list)
    outcomes: list[tuple[datetime, BarOutcome]] = field(default_factory=list)
    result: SimulationResult | None = None

    @property
    def proposals(self) -> list[EntryProposal]:
        return [o for _, o in self.outcomes if isinstance(o, EntryProposal)]

    @property
    def rejections(self) -> list[tuple[datetime, Rejected]]:
        return [(t, o) for t, o in self.outcomes if isinstance(o, Rejected)]

    @property
    def buy_signals(self) -> int:
        return len(self.proposals) + len(self.rejections)


def _prepare(config: AppConfig, data: BacktestData) -> Run:
    simulator = _Simulator(
        config,
        data,
        starting_cash=CASH,
        commission_per_fill=Decimal(0),
        slippage_multiplier=Decimal(1),
    )
    run = Run(simulator)
    flow, broker, clock = simulator._flow, simulator._broker, simulator._clock
    decide, submit = flow.on_closed_bar, broker.submit_bracket

    async def recording_decide(bar: Bar, *, book: TradeBook) -> BarOutcome:
        outcome = await decide(bar, book=book)
        run.outcomes.append((clock.now_utc(), outcome))
        return outcome

    async def recording_submit(request: BracketOrderRequest) -> BrokerOrder:
        run.submissions.append((clock.now_utc(), request))
        return await submit(request)

    flow.on_closed_bar = recording_decide  # type: ignore[method-assign]
    broker.submit_bracket = recording_submit  # type: ignore[method-assign]
    return run


async def _finish(run: Run) -> Run:
    run.result = await run.simulator.run()
    return run


def _patch_book(run: Run, change: Callable[[TradeBook], TradeBook]) -> None:
    original = run.simulator._book
    run.simulator._book = lambda: change(original())  # type: ignore[method-assign]


def _assert_no_violation_reached_the_broker(run: Run) -> None:
    """Every submitted entry comes from a proposal whose every check passed (51.4)."""
    proposed = {p.trade.signal_id[4:20]: p for p in run.proposals}
    for _, request in run.submissions:
        short = request.client_order_id.split("-")[1]
        proposal = proposed[short]
        assert all_passed(proposal.checks)
        assert request.qty == proposal.trade.qty
        assert request.stop_loss_stop_price == proposal.trade.stop_price
    rejected = {r.signal.signal_id[4:20] for _, r in run.rejections}
    assert not rejected & {r.client_order_id.split("-")[1] for _, r in run.submissions}


async def test_baseline_submits_only_fully_checked_entries(
    config: AppConfig, data: BacktestData
) -> None:
    run = await _finish(_prepare(config, data))
    assert run.submissions  # the scenarios below are not vacuous
    _assert_no_violation_reached_the_broker(run)


# --------------------------------------------------------------------------- AC-05


def _pending(symbol: str, qty: int, entry: str, stop: str) -> PendingEntry:
    return PendingEntry(symbol=symbol, qty=qty, entry_ref=Decimal(entry), stop_price=Decimal(stop))


def _violation(config: AppConfig, limit: str) -> Callable[[Run], None]:
    risk = config.risk
    assert risk.max_positions is not None
    assert risk.max_total_exposure_pct is not None
    assert risk.max_aggregate_open_risk_pct is not None
    max_positions = risk.max_positions
    exposure_qty = int(CASH * risk.max_total_exposure_pct / 10) + 1
    risk_qty = int(CASH * risk.max_aggregate_open_risk_pct / 5) + 1

    def book(
        change: dict[str, Any] | Callable[[TradeBook], dict[str, Any]],
    ) -> Callable[[Run], None]:
        def apply(run: Run) -> None:
            _patch_book(
                run,
                lambda b: b.model_copy(update=change(b) if callable(change) else change),
            )

        return apply

    def daily(run: Run) -> None:
        broker = run.simulator._broker
        account = broker.get_account

        async def inflated() -> AccountState:
            state = await account()
            return state.model_copy(update={"last_equity": state.equity * Decimal("1.5")})

        broker.get_account = inflated  # type: ignore[method-assign]

    return {
        DAILY_LOSS_LIMIT: daily,
        WEEKLY_LOSS_LIMIT: book(lambda b: {"week_start_equity": b.week_start_equity * 2}),
        MAX_DRAWDOWN: book(lambda b: {"peak_equity": b.peak_equity * 2}),
        MAX_POSITIONS: book(
            lambda b: {
                "pending_entries": b.pending_entries
                + tuple(_pending(f"P{i}", 1, "10", "9") for i in range(max_positions))
            }
        ),
        TOTAL_EXPOSURE: book(
            lambda b: {
                "pending_entries": (*b.pending_entries, _pending("PX", exposure_qty, "10", "9.99"))
            }
        ),
        AGGREGATE_OPEN_RISK: book(
            lambda b: {"pending_entries": (*b.pending_entries, _pending("PR", risk_qty, "10", "5"))}
        ),
    }[limit]


@pytest.mark.parametrize(
    "limit",
    [
        DAILY_LOSS_LIMIT,
        WEEKLY_LOSS_LIMIT,
        MAX_DRAWDOWN,
        MAX_POSITIONS,
        TOTAL_EXPOSURE,
        AGGREGATE_OPEN_RISK,
    ],
)
async def test_ac05_a_risk_violation_before_the_ai_step_never_reaches_the_broker(
    config: AppConfig, data: BacktestData, limit: str
) -> None:
    run = _prepare(config, data)
    _violation(config, limit)(run)
    await _finish(run)
    assert run.buy_signals > 0
    assert run.proposals == []
    assert run.submissions == []
    assert all(limit in rejected.codes for _, rejected in run.rejections)


async def test_ac05_a_limit_broken_after_the_proposal_is_caught_by_the_execution_checks(
    config: AppConfig, data: BacktestData
) -> None:
    """Equity falls between the proposal and the submission: the guard checks block it."""
    run = _prepare(config, data)
    simulator = run.simulator
    gate = RiskGate(
        settings=guard_settings(config, spread_filter=SpreadFilter.NOT_APPLIED),
        environment=SimulationGuardEnvironment(
            sessions=data.sessions,
            clock=simulator._clock,
            tradable=("SYNTH",),
            pending_decisions=(),
        ),
        liquidity_data=data.feed,
        quote_data=None,
    )
    submit_entry = simulator._submit_entry
    blocked: list[tuple[str, ...]] = []
    scale = Decimal("0.25")

    async def guarded(proposal: EntryProposal) -> None:
        state = await gate.broker_state(simulator._broker)
        account = state.account
        refreshed = state.model_copy(
            update={
                "account": account.model_copy(
                    update={
                        "equity": account.equity * scale,
                        "last_equity": account.last_equity * scale,
                        "buying_power": account.buying_power * scale,
                    }
                )
            }
        )
        book = simulator._book()
        session = simulator._flow.session_of(proposal.signal.bar_start_utc)
        assert session is not None
        context = await gate.context(
            now_utc=simulator._clock.now_utc(),
            signal=proposal.signal,
            signal_bar=simulator._flow.window(proposal.symbol)[-1],
            bars_since_last_exit=None,
            trade=proposal.trade,
            params=simulator._flow._check_params,
            timing=IntradayTiming(
                windows=compute_session_windows(session, simulator._session_params),
                last_minute_bar_end_utc=simulator._flow._last_minute_end.get(proposal.symbol),
            ),
            broker=refreshed,
            pending_entries=book.pending_entries,
            position_stops=book.position_stops,
            week_start_equity=book.week_start_equity * scale,
            peak_equity=book.peak_equity * scale,
            executed_signal_ids=book.executed_signal_ids,
        )
        results = run_execution_checks(
            context,
            ExecutionFacts(
                client_order_id=f"backtest-{proposal.trade.signal_id[4:20]}-entry",
                client_order_id_seen=False,
                symbol_lock_held=True,
                ai_result=None,
            ),
        )
        if not all_passed(results):
            blocked.append(failed_codes(results))
            return
        await submit_entry(proposal)  # pragma: no cover - every proposal is blocked here

    simulator._submit_entry = guarded  # type: ignore[method-assign]
    await _finish(run)
    assert run.proposals  # pre-AI passed: the limits only break at submission time
    assert len(blocked) == len(run.proposals)
    assert run.submissions == []
    assert all({RISK_PER_TRADE_EXCEEDED, SYMBOL_EXPOSURE} & set(codes) for codes in blocked)


# --------------------------------------------------------------------------- AC-15 and invariants


class SwitchingEnvironment:
    """Simulated runtime facts with one fact changed from ``switch_at`` on."""

    def __init__(
        self, base: SimulationGuardEnvironment, switch_at: datetime, change: dict[str, Any]
    ) -> None:
        self._base = base
        self._switch_at = switch_at
        self._change = change
        self.clock = base._clock

    async def runtime_facts(self) -> RuntimeFacts:
        facts = await self._base.runtime_facts()
        if self.clock.now_utc() < self._switch_at:
            return facts
        update = dict(self._change)
        control = update.pop("control", None)
        if control is not None:
            assert facts.control is not None
            update["control"] = facts.control.model_copy(update=control)
        return facts.model_copy(update=update)


def _switch(
    run: Run, data: BacktestData, config: AppConfig, at: datetime, change: dict[str, Any]
) -> None:
    simulator = run.simulator
    base = SimulationGuardEnvironment(
        sessions=data.sessions, clock=simulator._clock, tradable=("SYNTH",), pending_decisions=()
    )
    simulator._flow._gate = RiskGate(
        settings=guard_settings(config, spread_filter=SpreadFilter.NOT_APPLIED),
        environment=SwitchingEnvironment(base, at, change),
        liquidity_data=data.feed,
        quote_data=None,
    )


async def _switch_time(config: AppConfig, data: BacktestData) -> datetime:
    """An instant with entries both before and after it in the baseline run."""
    baseline = await _finish(_prepare(config, data))
    times = [t for t, _ in baseline.submissions]
    assert len(times) >= 2
    return times[len(times) // 2]


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"control": {"stop_file_present": True}}, STOP_FILE_PRESENT),
        ({"control": {"trading_enabled": False}}, TRADING_DISABLED),
        ({"control": {"emergency_close": True}}, EMERGENCY_CLOSE_ACTIVE),
    ],
    ids=["stop-file", "trading-disabled", "emergency-close"],
)
async def test_ac15_kill_switch_blocks_entries_from_the_next_decision_without_restart(
    config: AppConfig, data: BacktestData, change: dict[str, Any], code: str
) -> None:
    switch_at = await _switch_time(config, data)
    run = _prepare(config, data)
    _switch(run, data, config, switch_at, change)
    await _finish(run)
    assert run.submissions
    assert all(t < switch_at for t, _ in run.submissions)
    after = [r for t, r in run.rejections if t >= switch_at]
    assert after  # BUY signals kept coming after the switch
    assert all(code in r.codes and set(r.codes) & KILL_SWITCH_CODES for r in after)
    assert not [p for t, p in run.outcomes if t >= switch_at and isinstance(p, EntryProposal)]


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"breaker_state": CircuitBreakerState.HALTED}, CIRCUIT_BREAKER_NOT_NORMAL),
        ({"breaker_state": CircuitBreakerState.EMERGENCY}, CIRCUIT_BREAKER_NOT_NORMAL),
        ({"breaker_state": CircuitBreakerState.WARNING}, CIRCUIT_BREAKER_NOT_NORMAL),
        ({"feed_connected": False}, FEED_DISCONNECTED),
    ],
    ids=["halted", "emergency", "warning", "feed-down"],
)
async def test_breaker_or_data_outage_blocks_every_new_entry(
    config: AppConfig, data: BacktestData, change: dict[str, Any], code: str
) -> None:
    """Invariants 51.11 and 51.13 (AC-07 logic): no entry while data or breaker are not OK."""
    run = _prepare(config, data)
    _switch(run, data, config, data.sessions[0].open_utc - timedelta(days=1), change)
    await _finish(run)
    assert run.buy_signals > 0
    assert run.submissions == []
    assert all(code in r.codes for _, r in run.rejections)
