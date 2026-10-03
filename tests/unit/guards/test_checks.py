"""Single check library (sec. 19): every check passes, fails and fails closed (sec. 49.1)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest

from domain.guards.checks import (
    ASSET_STATUS_UNAVAILABLE,
    BAR_NOT_CLOSED,
    BAR_NOT_ENTERABLE,
    BROKER_STATE_UNAVAILABLE,
    BUYING_POWER_INSUFFICIENT,
    BUYING_POWER_UNAVAILABLE,
    CIRCUIT_BREAKER_NOT_NORMAL,
    CIRCUIT_BREAKER_UNAVAILABLE,
    CONFIG_INCOMPLETE,
    CONFIG_STATUS_UNAVAILABLE,
    CONTROL_STATE_UNAVAILABLE,
    COOLDOWN_ACTIVE,
    DAILY_BAR_SUPERSEDED,
    DUPLICATE_CLIENT_ORDER_ID,
    DUPLICATE_SIGNAL,
    EMERGENCY_CLOSE_ACTIVE,
    ENTRY_PENDING,
    EXIT_LEVELS_UNAVAILABLE,
    FEED_DISCONNECTED,
    FEED_STATUS_UNAVAILABLE,
    KILL_SWITCH_CODES,
    LIQUIDITY_FEED_MISMATCH,
    LIQUIDITY_UNAVAILABLE,
    MARKET_CLOCK_UNAVAILABLE,
    MARKET_CLOSED,
    ORDER_HISTORY_UNAVAILABLE,
    OUTSIDE_ENTRY_WINDOW,
    PARAM_PENDING,
    POSITION_EXISTS,
    PRE_AI_CHECKS,
    RECONCILIATION_STALE,
    RECONCILIATION_UNAVAILABLE,
    RISK_LIMITS_UNAVAILABLE,
    SESSION_UNKNOWN,
    SIGNAL_EXPIRED,
    SIGNAL_HISTORY_UNAVAILABLE,
    SPREAD_NOT_APPLIED,
    STATE_MISMATCH_UNRESOLVED,
    STOP_FILE_PRESENT,
    SYMBOL_LOCK_NOT_HELD,
    SYSTEM_MODE_UNAVAILABLE,
    SYSTEM_NOT_RUNNING,
    TRADING_DISABLED,
    AIApprovalCode,
    CheckCode,
    CheckContext,
    ExecutionFacts,
    ReconciliationFacts,
    RuntimeFacts,
    SpreadFilter,
    all_passed,
    checks_of,
    failed_codes,
    kill_switch_reasons,
    run_execution_checks,
    run_pre_ai_checks,
)
from domain.market.quality import QualityCode
from domain.market.universe import UniverseCode
from domain.models import (
    AIMode,
    AIReasonCode,
    AIValidity,
    AIVerdict,
    AIVerdictKind,
    AIVerdictResult,
    BarStatus,
    BrokerOrder,
    BrokerOrderStatus,
    CheckResult,
    CircuitBreakerState,
    DataFeed,
    OrderSide,
    OrderType,
    Position,
    SystemMode,
)
from domain.risk.exits import INVALID_EXIT_LEVELS
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
    PositionStop,
)
from tests.unit.guards import factories as f

SEC_19_TABLE = (
    "SYMBOL_ALLOWED",
    "MARKET_OPEN",
    "ENTRY_WINDOW",
    "BAR_CLOSED",
    "SIGNAL_NOT_EXPIRED",
    "DATA_FRESH",
    "NO_EXISTING_POSITION",
    "NO_PENDING_ORDER",
    "NOT_DUPLICATE_SIGNAL",
    "COOLDOWN_OK",
    "PRICE_RANGE",
    "LIQUIDITY_OK",
    "SPREAD_OK",
    "EXIT_LEVELS_VALID",
    "RISK_LIMITS_OK",
    "BUYING_POWER_OK",
    "CIRCUIT_BREAKER_NORMAL",
    "TRADING_ENABLED",
    "SYSTEM_RUNNING",
    "STATE_RECONCILED",
    "CONFIG_COMPLETE",
)
"""The sec. 19 table, in order."""


def _order(**overrides: Any) -> BrokerOrder:
    values: dict[str, Any] = {
        "order_id": "o-1",
        "client_order_id": "paper-x-entry",
        "symbol": f.SYMBOL,
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "status": BrokerOrderStatus.NEW,
        "qty": Decimal(10),
    }
    values.update(overrides)
    return BrokerOrder(**values)


def _failed(ctx: CheckContext) -> tuple[str, ...]:
    return failed_codes(run_pre_ai_checks(ctx))


def _check_of(ctx: CheckContext, code: CheckCode) -> tuple[CheckResult, ...]:
    return checks_of(run_pre_ai_checks(ctx), code)


# --------------------------------------------------------------------------- structure


def test_the_library_covers_the_sec_19_table_in_order_once() -> None:
    assert tuple(code.value for code, _ in PRE_AI_CHECKS) == SEC_19_TABLE


@pytest.mark.parametrize("build", [f.context, f.daily_context], ids=["intraday", "daily"])
def test_a_clean_context_passes_every_check(build: Callable[[], CheckContext]) -> None:
    results = run_pre_ai_checks(build())
    assert all_passed(results), failed_codes(results)
    assert {r.detail["check"] for r in results} == set(SEC_19_TABLE)
    risk = [r for r in results if r.detail["check"] == "RISK_LIMITS_OK"]
    assert len(risk) == 8  # every limit of sec. 15.1


def test_no_short_circuit_every_check_reports_even_when_all_fail() -> None:
    ctx = f.context(
        trade=None,
        runtime=RuntimeFacts.unavailable(),
        market=f.market(daily_bars=None, quote=None),
        portfolio=f.portfolio(
            account=None, positions=None, open_orders=None, executed_signal_ids=None
        ),
        params=f.params(whitelist=None, min_price=None, cooldown_bars=None),
        timing=f.intraday_timing(windows=None, last_minute_bar_end_utc=None),
        now_utc=f.BAR_END + timedelta(hours=1),
        signal_bar=f.signal_bar(status=BarStatus.INCOMPLETE, minutes_present=1),
    )
    results = run_pre_ai_checks(ctx)
    assert {r.detail["check"] for r in results} == set(SEC_19_TABLE)
    by_check = {code: checks_of(results, CheckCode(code)) for code in SEC_19_TABLE}
    assert all(any(not r.passed for r in items) for items in by_check.values())
    assert not all_passed(results)


def test_every_result_names_its_check() -> None:
    ctx = f.context(runtime=RuntimeFacts.unavailable())
    results = run_execution_checks(
        ctx,
        ExecutionFacts(
            client_order_id="paper-x-entry",
            client_order_id_seen=False,
            symbol_lock_held=True,
            ai_result=None,
        ),
    )
    assert all(r.detail["check"] in {c.value for c in CheckCode} for r in results)


def test_all_passed_never_approves_an_empty_set() -> None:
    assert not all_passed(())


def test_failed_codes_keep_order_and_drop_repetitions() -> None:
    results = (
        CheckResult(passed=False, code="B"),
        CheckResult(passed=True, code="X"),
        CheckResult(passed=False, code="A"),
        CheckResult(passed=False, code="B"),
    )
    assert failed_codes(results) == ("B", "A")


def test_context_refuses_a_trade_of_another_signal() -> None:
    with pytest.raises(ValueError, match="trade must belong"):
        f.context(trade=f.trade(signal_id="sig_other"))
    with pytest.raises(ValueError, match=r"signal_bar\.symbol"):
        f.context(signal_bar=f.signal_bar(symbol="BBB"))


# --------------------------------------------------------------------------- per check


Case = tuple[str, Callable[[], CheckContext], CheckCode, str]

CASES: list[Case] = [
    # SYMBOL_ALLOWED
    (
        "not whitelisted",
        lambda: f.context(params=f.params(whitelist=("BBB",))),
        CheckCode.SYMBOL_ALLOWED,
        UniverseCode.SYMBOL_NOT_WHITELISTED.value,
    ),
    (
        "blacklisted",
        lambda: f.context(params=f.params(blacklist=(f.SYMBOL,))),
        CheckCode.SYMBOL_ALLOWED,
        UniverseCode.SYMBOL_BLACKLISTED.value,
    ),
    (
        "not tradable",
        lambda: f.context(runtime=f.runtime(tradable_symbols=frozenset({"BBB"}))),
        CheckCode.SYMBOL_ALLOWED,
        UniverseCode.ASSET_NOT_TRADABLE.value,
    ),
    (
        "asset status unknown",
        lambda: f.context(runtime=f.runtime(tradable_symbols=None)),
        CheckCode.SYMBOL_ALLOWED,
        ASSET_STATUS_UNAVAILABLE,
    ),
    (
        "whitelist pending",
        lambda: f.context(params=f.params(whitelist=None)),
        CheckCode.SYMBOL_ALLOWED,
        PARAM_PENDING,
    ),
    # MARKET_OPEN
    (
        "market closed",
        lambda: f.context(runtime=f.runtime(market_open=False)),
        CheckCode.MARKET_OPEN,
        MARKET_CLOSED,
    ),
    (
        "no broker clock",
        lambda: f.context(runtime=f.runtime(market_open=None)),
        CheckCode.MARKET_OPEN,
        MARKET_CLOCK_UNAVAILABLE,
    ),
    (
        "daily without next session",
        lambda: f.daily_context(timing=f.daily_timing(next_session=None)),
        CheckCode.MARKET_OPEN,
        SESSION_UNKNOWN,
    ),
    (
        "daily without broker clock",
        lambda: f.daily_context(runtime=f.runtime(market_open=None)),
        CheckCode.MARKET_OPEN,
        MARKET_CLOCK_UNAVAILABLE,
    ),
    # ENTRY_WINDOW
    (
        "before entries_allowed_from",
        lambda: f.context(now_utc=f.WINDOWS.entries_allowed_from_utc - timedelta(seconds=1)),
        CheckCode.ENTRY_WINDOW,
        OUTSIDE_ENTRY_WINDOW,
    ),
    (
        "at entries_allowed_until",
        lambda: f.context(now_utc=f.WINDOWS.entries_allowed_until_utc),
        CheckCode.ENTRY_WINDOW,
        OUTSIDE_ENTRY_WINDOW,
    ),
    (
        "session unknown",
        lambda: f.context(timing=f.intraday_timing(windows=None)),
        CheckCode.ENTRY_WINDOW,
        SESSION_UNKNOWN,
    ),
    (
        "daily before the close",
        lambda: f.daily_context(now_utc=f.SESSION.close_utc - timedelta(minutes=1)),
        CheckCode.ENTRY_WINDOW,
        OUTSIDE_ENTRY_WINDOW,
    ),
    (
        "daily session unknown",
        lambda: f.daily_context(timing=f.daily_timing(signal_session=None)),
        CheckCode.ENTRY_WINDOW,
        SESSION_UNKNOWN,
    ),
    # BAR_CLOSED
    (
        "bar not closed",
        lambda: f.context(now_utc=f.BAR_END - timedelta(seconds=1)),
        CheckCode.BAR_CLOSED,
        BAR_NOT_CLOSED,
    ),
    (
        "bar incomplete",
        lambda: f.context(signal_bar=f.signal_bar(status=BarStatus.INCOMPLETE, minutes_present=2)),
        CheckCode.BAR_CLOSED,
        BAR_NOT_ENTERABLE,
    ),
    (
        "min minutes pending",
        lambda: f.context(params=f.params(min_minutes_per_bar=None)),
        CheckCode.BAR_CLOSED,
        PARAM_PENDING,
    ),
    # SIGNAL_NOT_EXPIRED
    (
        "expired at expires_at",
        lambda: f.context(now_utc=f.BAR_END + timedelta(seconds=120)),
        CheckCode.SIGNAL_NOT_EXPIRED,
        SIGNAL_EXPIRED,
    ),
    # DATA_FRESH
    (
        "stale symbol",
        lambda: f.context(
            timing=f.intraday_timing(last_minute_bar_end_utc=f.NOW - timedelta(seconds=91))
        ),
        CheckCode.DATA_FRESH,
        QualityCode.STALE.value,
    ),
    (
        "no minute bar yet",
        lambda: f.context(timing=f.intraday_timing(last_minute_bar_end_utc=None)),
        CheckCode.DATA_FRESH,
        QualityCode.STALE.value,
    ),
    (
        "feed disconnected",
        lambda: f.context(runtime=f.runtime(feed_connected=False)),
        CheckCode.DATA_FRESH,
        FEED_DISCONNECTED,
    ),
    (
        "feed status unknown",
        lambda: f.context(runtime=f.runtime(feed_connected=None)),
        CheckCode.DATA_FRESH,
        FEED_STATUS_UNAVAILABLE,
    ),
    (
        "max bar age pending",
        lambda: f.context(params=f.params(max_bar_age_seconds=None)),
        CheckCode.DATA_FRESH,
        PARAM_PENDING,
    ),
    (
        "daily bar superseded",
        lambda: f.daily_context(now_utc=f.NEXT_SESSION.close_utc),
        CheckCode.DATA_FRESH,
        DAILY_BAR_SUPERSEDED,
    ),
    # NO_EXISTING_POSITION
    (
        "position exists",
        lambda: f.context(
            portfolio=f.portfolio(
                positions=(
                    Position(
                        symbol=f.SYMBOL,
                        qty=Decimal(5),
                        avg_entry_price=Decimal(90),
                        market_value=Decimal(500),
                    ),
                )
            )
        ),
        CheckCode.NO_EXISTING_POSITION,
        POSITION_EXISTS,
    ),
    (
        "positions unknown",
        lambda: f.context(portfolio=f.portfolio(positions=None)),
        CheckCode.NO_EXISTING_POSITION,
        BROKER_STATE_UNAVAILABLE,
    ),
    # NO_PENDING_ORDER
    (
        "broker pending entry",
        lambda: f.context(portfolio=f.portfolio(open_orders=(_order(),))),
        CheckCode.NO_PENDING_ORDER,
        ENTRY_PENDING,
    ),
    (
        "db pending entry",
        lambda: f.context(
            portfolio=f.portfolio(
                pending_entries=(
                    PendingEntry(
                        symbol=f.SYMBOL, qty=1, entry_ref=Decimal(100), stop_price=Decimal(99)
                    ),
                )
            )
        ),
        CheckCode.NO_PENDING_ORDER,
        ENTRY_PENDING,
    ),
    (
        "open orders unknown",
        lambda: f.context(portfolio=f.portfolio(open_orders=None)),
        CheckCode.NO_PENDING_ORDER,
        BROKER_STATE_UNAVAILABLE,
    ),
    # NOT_DUPLICATE_SIGNAL
    (
        "signal executed before",
        lambda: f.context(portfolio=f.portfolio(executed_signal_ids=frozenset({"sig_test_0001"}))),
        CheckCode.NOT_DUPLICATE_SIGNAL,
        DUPLICATE_SIGNAL,
    ),
    (
        "signal history unknown",
        lambda: f.context(portfolio=f.portfolio(executed_signal_ids=None)),
        CheckCode.NOT_DUPLICATE_SIGNAL,
        SIGNAL_HISTORY_UNAVAILABLE,
    ),
    # COOLDOWN_OK
    (
        "in cooldown",
        lambda: f.context(bars_since_last_exit=2),
        CheckCode.COOLDOWN_OK,
        COOLDOWN_ACTIVE,
    ),
    (
        "cooldown pending",
        lambda: f.context(params=f.params(cooldown_bars=None)),
        CheckCode.COOLDOWN_OK,
        PARAM_PENDING,
    ),
    # PRICE_RANGE
    (
        "price below min",
        lambda: f.context(params=f.params(min_price=Decimal("100.01"))),
        CheckCode.PRICE_RANGE,
        UniverseCode.PRICE_BELOW_MIN.value,
    ),
    (
        "price above max",
        lambda: f.context(params=f.params(max_price=Decimal("99.99"))),
        CheckCode.PRICE_RANGE,
        UniverseCode.PRICE_ABOVE_MAX.value,
    ),
    (
        "no close price",
        lambda: f.context(
            signal_bar=f.signal_bar(
                open=None, high=None, low=None, close=None, volume=0, status=BarStatus.EMPTY
            )
        ),
        CheckCode.PRICE_RANGE,
        UniverseCode.PRICE_UNAVAILABLE.value,
    ),
    (
        "price range pending",
        lambda: f.context(params=f.params(max_price=None)),
        CheckCode.PRICE_RANGE,
        PARAM_PENDING,
    ),
    # LIQUIDITY_OK
    (
        "volume too low",
        lambda: f.context(market=f.market(daily_bars=f.daily_bars(volume=999))),
        CheckCode.LIQUIDITY_OK,
        UniverseCode.AVG_VOLUME_TOO_LOW.value,
    ),
    (
        "not enough sessions",
        lambda: f.context(market=f.market(daily_bars=f.daily_bars(count=4))),
        CheckCode.LIQUIDITY_OK,
        UniverseCode.AVG_VOLUME_INSUFFICIENT_DATA.value,
    ),
    (
        "liquidity source down",
        lambda: f.context(market=f.market(daily_bars=None)),
        CheckCode.LIQUIDITY_OK,
        LIQUIDITY_UNAVAILABLE,
    ),
    (
        "liquidity bars from iex",
        lambda: f.context(market=f.market(daily_bars=f.daily_bars(feed=DataFeed.IEX))),
        CheckCode.LIQUIDITY_OK,
        LIQUIDITY_FEED_MISMATCH,
    ),
    (
        "liquidity pending",
        lambda: f.context(params=f.params(min_avg_daily_volume=None)),
        CheckCode.LIQUIDITY_OK,
        PARAM_PENDING,
    ),
    # SPREAD_OK
    (
        "spread too wide",
        lambda: f.context(market=f.market(quote=f.quote("99.80", "100.20"))),
        CheckCode.SPREAD_OK,
        UniverseCode.SPREAD_TOO_WIDE.value,
    ),
    (
        "no quote",
        lambda: f.context(market=f.market(quote=None)),
        CheckCode.SPREAD_OK,
        UniverseCode.QUOTE_UNAVAILABLE.value,
    ),
    (
        "spread pending",
        lambda: f.context(params=f.params(max_spread_bps=None)),
        CheckCode.SPREAD_OK,
        PARAM_PENDING,
    ),
    # EXIT_LEVELS_VALID
    (
        "take profit too close",
        lambda: f.context(trade=f.trade(take_profit_price=Decimal("100.01"))),
        CheckCode.EXIT_LEVELS_VALID,
        INVALID_EXIT_LEVELS,
    ),
    (
        "no priced trade",
        lambda: f.context(trade=None),
        CheckCode.EXIT_LEVELS_VALID,
        EXIT_LEVELS_UNAVAILABLE,
    ),
    (
        "min tp distance pending",
        lambda: f.context(params=f.params(min_tp_distance_ticks=None)),
        CheckCode.EXIT_LEVELS_VALID,
        PARAM_PENDING,
    ),
    # RISK_LIMITS_OK (one case per limit of sec. 15.1)
    (
        "risk per trade",
        lambda: f.context(trade=f.trade(risk_amount=Decimal(501))),
        CheckCode.RISK_LIMITS_OK,
        RISK_PER_TRADE_EXCEEDED,
    ),
    (
        "daily loss",
        lambda: f.context(portfolio=f.portfolio(account=f.account(last_equity=Decimal(102041)))),
        CheckCode.RISK_LIMITS_OK,
        DAILY_LOSS_LIMIT,
    ),
    (
        "weekly loss",
        lambda: f.context(portfolio=f.portfolio(week_start_equity=Decimal(104200))),
        CheckCode.RISK_LIMITS_OK,
        WEEKLY_LOSS_LIMIT,
    ),
    (
        "drawdown",
        lambda: f.context(portfolio=f.portfolio(peak_equity=Decimal(111200))),
        CheckCode.RISK_LIMITS_OK,
        MAX_DRAWDOWN,
    ),
    (
        "max positions",
        lambda: f.context(
            portfolio=f.portfolio(
                pending_entries=tuple(
                    PendingEntry(symbol=s, qty=1, entry_ref=Decimal(10), stop_price=Decimal(9))
                    for s in ("B1", "B2", "B3")
                )
            )
        ),
        CheckCode.RISK_LIMITS_OK,
        MAX_POSITIONS,
    ),
    (
        "total exposure",
        lambda: f.context(
            portfolio=f.portfolio(
                pending_entries=(
                    PendingEntry(
                        symbol="BBB", qty=850, entry_ref=Decimal(100), stop_price=Decimal("99.5")
                    ),
                )
            )
        ),
        CheckCode.RISK_LIMITS_OK,
        TOTAL_EXPOSURE,
    ),
    (
        "symbol exposure",
        lambda: f.context(trade=f.trade(qty=301, risk_amount=Decimal(300))),
        CheckCode.RISK_LIMITS_OK,
        SYMBOL_EXPOSURE,
    ),
    (
        "aggregate open risk",
        lambda: f.context(
            portfolio=f.portfolio(
                positions=(
                    Position(
                        symbol="BBB",
                        qty=Decimal(100),
                        avg_entry_price=Decimal(100),
                        market_value=Decimal(10000),
                    ),
                ),
                position_stops=(PositionStop(symbol="BBB", stop_price=Decimal(88)),),
            )
        ),
        CheckCode.RISK_LIMITS_OK,
        AGGREGATE_OPEN_RISK,
    ),
    (
        "account unknown",
        lambda: f.context(portfolio=f.portfolio(account=None)),
        CheckCode.RISK_LIMITS_OK,
        RISK_LIMITS_UNAVAILABLE,
    ),
    (
        "risk param pending",
        lambda: f.context(params=f.params(risk=f.risk_params(max_drawdown_pct=None))),
        CheckCode.RISK_LIMITS_OK,
        PARAM_PENDING,
    ),
    # BUYING_POWER_OK
    (
        "buying power short",
        lambda: f.context(portfolio=f.portfolio(account=f.account(buying_power=Decimal(10004)))),
        CheckCode.BUYING_POWER_OK,
        BUYING_POWER_INSUFFICIENT,
    ),
    (
        "buying power without trade",
        lambda: f.context(trade=None),
        CheckCode.BUYING_POWER_OK,
        BUYING_POWER_UNAVAILABLE,
    ),
    # CIRCUIT_BREAKER_NORMAL
    (
        "breaker warning",
        lambda: f.context(runtime=f.runtime(breaker_state=CircuitBreakerState.WARNING)),
        CheckCode.CIRCUIT_BREAKER_NORMAL,
        CIRCUIT_BREAKER_NOT_NORMAL,
    ),
    (
        "breaker unknown",
        lambda: f.context(runtime=f.runtime(breaker_state=None)),
        CheckCode.CIRCUIT_BREAKER_NORMAL,
        CIRCUIT_BREAKER_UNAVAILABLE,
    ),
    # TRADING_ENABLED
    (
        "trading disabled",
        lambda: f.context(runtime=f.runtime(control=f.control(trading_enabled=False))),
        CheckCode.TRADING_ENABLED,
        TRADING_DISABLED,
    ),
    (
        "stop file",
        lambda: f.context(runtime=f.runtime(control=f.control(stop_file_present=True))),
        CheckCode.TRADING_ENABLED,
        STOP_FILE_PRESENT,
    ),
    (
        "emergency close",
        lambda: f.context(runtime=f.runtime(control=f.control(emergency_close=True))),
        CheckCode.TRADING_ENABLED,
        EMERGENCY_CLOSE_ACTIVE,
    ),
    (
        "control unknown",
        lambda: f.context(runtime=f.runtime(control=None)),
        CheckCode.TRADING_ENABLED,
        CONTROL_STATE_UNAVAILABLE,
    ),
    # SYSTEM_RUNNING
    (
        "system degraded",
        lambda: f.context(runtime=f.runtime(system_mode=SystemMode.DEGRADED)),
        CheckCode.SYSTEM_RUNNING,
        SYSTEM_NOT_RUNNING,
    ),
    (
        "mode unknown",
        lambda: f.context(runtime=f.runtime(system_mode=None)),
        CheckCode.SYSTEM_RUNNING,
        SYSTEM_MODE_UNAVAILABLE,
    ),
    # STATE_RECONCILED
    (
        "unresolved mismatch",
        lambda: f.context(
            runtime=f.runtime(
                reconciliation=ReconciliationFacts(
                    last_clean_at_utc=f.NOW, unresolved_mismatch=True
                )
            )
        ),
        CheckCode.STATE_RECONCILED,
        STATE_MISMATCH_UNRESOLVED,
    ),
    (
        "reconciliation too old",
        lambda: f.context(
            runtime=f.runtime(
                reconciliation=ReconciliationFacts(
                    last_clean_at_utc=f.NOW - timedelta(seconds=61), unresolved_mismatch=False
                )
            )
        ),
        CheckCode.STATE_RECONCILED,
        RECONCILIATION_STALE,
    ),
    (
        "never reconciled",
        lambda: f.context(
            runtime=f.runtime(
                reconciliation=ReconciliationFacts(
                    last_clean_at_utc=None, unresolved_mismatch=False
                )
            )
        ),
        CheckCode.STATE_RECONCILED,
        RECONCILIATION_STALE,
    ),
    (
        "reconciliation unknown",
        lambda: f.context(runtime=f.runtime(reconciliation=None)),
        CheckCode.STATE_RECONCILED,
        RECONCILIATION_UNAVAILABLE,
    ),
    # CONFIG_COMPLETE
    (
        "owner decision pending",
        lambda: f.context(runtime=f.runtime(pending_owner_decisions=("risk.max_positions",))),
        CheckCode.CONFIG_COMPLETE,
        CONFIG_INCOMPLETE,
    ),
    (
        "config status unknown",
        lambda: f.context(runtime=f.runtime(pending_owner_decisions=None)),
        CheckCode.CONFIG_COMPLETE,
        CONFIG_STATUS_UNAVAILABLE,
    ),
]


@pytest.mark.parametrize(("name", "build", "check", "code"), CASES, ids=[c[0] for c in CASES])
def test_each_check_fails_with_its_code(
    name: str, build: Callable[[], CheckContext], check: CheckCode, code: str
) -> None:
    ctx = build()
    results = run_pre_ai_checks(ctx)
    mine = checks_of(results, check)
    assert code in [r.code for r in mine if not r.passed], (name, [r.code for r in mine])
    assert code in failed_codes(results)
    # The result belongs to the named check and the other checks still ran (no short-circuit).
    assert {r.detail["check"] for r in results} == set(SEC_19_TABLE)


def test_failures_are_isolated_to_their_check() -> None:
    """Changing one input fails only the checks that read it."""
    results = run_pre_ai_checks(f.context(runtime=f.runtime(market_open=False)))
    assert failed_codes(results) == (MARKET_CLOSED,)


def test_signal_valid_until_just_before_expiry() -> None:
    almost = f.context(now_utc=f.BAR_END + timedelta(seconds=119))
    assert all(r.passed for r in _check_of(almost, CheckCode.SIGNAL_NOT_EXPIRED))


def test_daily_bars_of_the_decision_session_are_never_used() -> None:
    """Only bars that ended at or before the signal bar start count (no lookahead)."""
    bars = f.daily_bars(count=4)
    today = bars[-1].model_copy(
        update={
            "bar_start_utc": bars[-1].bar_start_utc.replace(day=10),
            "bar_end_utc": bars[-1].bar_end_utc.replace(day=11),
        }
    )
    ctx = f.context(market=f.market(daily_bars=(*bars, today)))
    (result,) = _check_of(ctx, CheckCode.LIQUIDITY_OK)
    assert result.code == UniverseCode.AVG_VOLUME_INSUFFICIENT_DATA.value


def test_liquidity_ignores_other_symbols_bars() -> None:
    other = tuple(b.model_copy(update={"symbol": "BBB", "volume": 1}) for b in f.daily_bars())
    ctx = f.context(market=f.market(daily_bars=(*f.daily_bars(), *other)))
    assert all(r.passed for r in _check_of(ctx, CheckCode.LIQUIDITY_OK))


def test_spread_not_applied_passes_explicitly() -> None:
    ctx = f.context(
        params=f.params(spread_filter=SpreadFilter.NOT_APPLIED), market=f.market(quote=None)
    )
    (result,) = _check_of(ctx, CheckCode.SPREAD_OK)
    assert result.passed
    assert result.code == SPREAD_NOT_APPLIED


@pytest.mark.parametrize(
    "status",
    [
        s
        for s in BrokerOrderStatus
        if s.value not in {"FILLED", "CANCELED", "EXPIRED", "REJECTED", "REPLACED"}
    ],
)
def test_any_non_final_buy_order_blocks_the_symbol(status: BrokerOrderStatus) -> None:
    """Invariant 51.9: uncertainty about an order -> no new order in that symbol."""
    ctx = f.context(portfolio=f.portfolio(open_orders=(_order(status=status),)))
    assert ENTRY_PENDING in _failed(ctx)


@pytest.mark.parametrize("status", ["FILLED", "CANCELED", "EXPIRED", "REJECTED", "REPLACED"])
def test_final_orders_and_exit_legs_do_not_block(status: str) -> None:
    orders = (
        _order(status=BrokerOrderStatus(status)),
        _order(
            order_id="o-2", side=OrderSide.SELL, order_type=OrderType.STOP, stop_price=Decimal(95)
        ),
        _order(order_id="o-3", symbol="BBB"),
    )
    ctx = f.context(portfolio=f.portfolio(open_orders=orders))
    assert all(r.passed for r in _check_of(ctx, CheckCode.NO_PENDING_ORDER))


def test_a_flat_position_row_is_not_a_position() -> None:
    flat = Position(
        symbol=f.SYMBOL, qty=Decimal(0), avg_entry_price=Decimal(1), market_value=Decimal(0)
    )
    ctx = f.context(portfolio=f.portfolio(positions=(flat,)))
    assert all(r.passed for r in _check_of(ctx, CheckCode.NO_EXISTING_POSITION))


def test_risk_limits_delegate_to_the_risk_engine_codes() -> None:
    results = _check_of(f.context(), CheckCode.RISK_LIMITS_OK)
    assert [r.code for r in results] == [
        RISK_PER_TRADE_EXCEEDED,
        DAILY_LOSS_LIMIT,
        WEEKLY_LOSS_LIMIT,
        MAX_DRAWDOWN,
        MAX_POSITIONS,
        TOTAL_EXPOSURE,
        SYMBOL_EXPOSURE,
        AGGREGATE_OPEN_RISK,
    ]


def test_several_limits_are_all_reported() -> None:
    ctx = f.context(
        portfolio=f.portfolio(peak_equity=Decimal(200000), week_start_equity=Decimal(200000))
    )
    assert {WEEKLY_LOSS_LIMIT, MAX_DRAWDOWN} <= set(_failed(ctx))


# --------------------------------------------------------------------------- kill switch (AC-15)


@pytest.mark.parametrize("stop", [False, True])
@pytest.mark.parametrize("emergency", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_kill_switch_reasons_order(stop: bool, emergency: bool, enabled: bool) -> None:
    reasons = kill_switch_reasons(
        f.control(stop_file_present=stop, emergency_close=emergency, trading_enabled=enabled)
    )
    expected = tuple(
        code
        for code, active in (
            (STOP_FILE_PRESENT, stop),
            (EMERGENCY_CLOSE_ACTIVE, emergency),
            (TRADING_DISABLED, not enabled),
        )
        if active
    )
    assert reasons == expected
    assert set(reasons) <= KILL_SWITCH_CODES


def _execution(**overrides: Any) -> ExecutionFacts:
    values: dict[str, Any] = {
        "client_order_id": "paper-sig-entry",
        "client_order_id_seen": False,
        "symbol_lock_held": True,
        "ai_result": None,
    }
    values.update(overrides)
    return ExecutionFacts(**values)


@pytest.mark.parametrize(
    ("control_overrides", "code"),
    [
        ({"stop_file_present": True}, STOP_FILE_PRESENT),
        ({"trading_enabled": False}, TRADING_DISABLED),
        ({"emergency_close": True}, EMERGENCY_CLOSE_ACTIVE),
    ],
)
def test_kill_switch_fails_pre_ai_and_execution_checks(
    control_overrides: dict[str, bool], code: str
) -> None:
    """AC-15 (logic): the DB flag or the STOP file blocks both stages with a kill-switch code."""
    ctx = f.context(runtime=f.runtime(control=f.control(**control_overrides)))
    pre = run_pre_ai_checks(ctx)
    execution = run_execution_checks(ctx, _execution())
    for results in (pre, execution):
        assert not all_passed(results)
        assert failed_codes(results) == (code,)
        assert set(failed_codes(results)) <= KILL_SWITCH_CODES


def test_stop_file_overrides_trading_enabled() -> None:
    ctx = f.context(
        runtime=f.runtime(control=f.control(trading_enabled=True, stop_file_present=True))
    )
    assert STOP_FILE_PRESENT in _failed(ctx)


# --------------------------------------------------------------------------- execution guard


def test_execution_checks_rerun_every_pre_ai_check_first() -> None:
    ctx = f.context()
    pre = run_pre_ai_checks(ctx)
    execution = run_execution_checks(ctx, _execution())
    assert execution[: len(pre)] == pre
    assert [r.detail["check"] for r in execution[len(pre) :]] == [
        "SYMBOL_LOCK_HELD",
        "CLIENT_ORDER_ID_UNUSED",
        "AI_VERDICT_OK",
    ]
    assert all_passed(execution)


@pytest.mark.parametrize(
    ("facts", "code"),
    [
        ({"symbol_lock_held": False}, SYMBOL_LOCK_NOT_HELD),
        ({"client_order_id_seen": True}, DUPLICATE_CLIENT_ORDER_ID),
        ({"client_order_id_seen": None}, ORDER_HISTORY_UNAVAILABLE),
    ],
)
def test_execution_preconditions(facts: dict[str, Any], code: str) -> None:
    results = run_execution_checks(f.context(), _execution(**facts))
    assert failed_codes(results) == (code,)


def _ai(
    verdict: AIVerdictKind,
    *,
    signal_id: str = "sig_test_0001",
    validity: AIValidity = AIValidity.VALID,
) -> AIVerdictResult:
    return AIVerdictResult(
        validity=validity,
        verdict=AIVerdict(
            signal_id=signal_id,
            verdict=verdict,
            reason_code=AIReasonCode.SIGNAL_CONFIRMED,
            risk_flags=(),
            confidence=0.8,
            rationale="fixture",
        ),
    )


@pytest.mark.parametrize(
    ("ai_result", "code"),
    [
        (None, AIApprovalCode.AI_APPROVAL_MISSING.value),
        (
            AIVerdictResult(validity=AIValidity.UNAVAILABLE, invalid_reason="timeout"),
            AIApprovalCode.AI_RESPONSE_NOT_VALID.value,
        ),
        (
            _ai(AIVerdictKind.APPROVE, signal_id="sig_other"),
            AIApprovalCode.AI_SIGNAL_MISMATCH.value,
        ),
        (_ai(AIVerdictKind.VETO), AIApprovalCode.AI_VETOED.value),
    ],
)
def test_active_ai_mode_requires_a_valid_approval(
    ai_result: AIVerdictResult | None, code: str
) -> None:
    ctx = f.context(runtime=f.runtime(control=f.control(ai_mode=AIMode.ACTIVE)))
    results = run_execution_checks(ctx, _execution(ai_result=ai_result))
    assert failed_codes(results) == (code,)
    approved = run_execution_checks(ctx, _execution(ai_result=_ai(AIVerdictKind.APPROVE)))
    assert all_passed(approved)


@pytest.mark.parametrize("mode", [AIMode.DISABLED, AIMode.SHADOW])
def test_shadow_and_disabled_never_gate_on_the_verdict(mode: AIMode) -> None:
    """AC-17: in SHADOW the verdict never changes which orders are sent."""
    ctx = f.context(runtime=f.runtime(control=f.control(ai_mode=mode)))
    for ai_result in (None, _ai(AIVerdictKind.VETO)):
        assert all_passed(run_execution_checks(ctx, _execution(ai_result=ai_result)))


def test_execution_guard_catches_state_that_changed_after_the_proposal() -> None:
    """The guard re-runs the limits on refreshed state: an equity drop blocks the order."""
    refreshed = f.context(
        portfolio=f.portfolio(
            account=f.account(
                equity=Decimal(30000), last_equity=Decimal(100000), buying_power=Decimal(30000)
            )
        )
    )
    codes = failed_codes(run_execution_checks(refreshed, _execution()))
    assert {RISK_PER_TRADE_EXCEEDED, DAILY_LOSS_LIMIT, SYMBOL_EXPOSURE} <= set(codes)
