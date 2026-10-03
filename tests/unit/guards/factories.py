"""Factories of a check context that passes every check (TEST FIXTURE values only).

Each test changes exactly the field it is about, so a failure is attributable to it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from domain.guards.checks import (
    CheckContext,
    CheckParams,
    ControlFacts,
    DailyTiming,
    IntradayTiming,
    MarketFacts,
    PortfolioFacts,
    ReconciliationFacts,
    RuntimeFacts,
    SpreadFilter,
)
from domain.market.session import SessionWindowParams, compute_session_windows
from domain.models import (
    AccountState,
    AIMode,
    Bar,
    BarStatus,
    CircuitBreakerState,
    DataFeed,
    HoldingMode,
    ProposedTrade,
    Quote,
    SessionDay,
    Signal,
    SystemMode,
    Timeframe,
)
from domain.risk.risk_engine import RiskParams

SYMBOL = "AAA"
SESSION = SessionDay(
    session_date=date(2025, 3, 10),
    open_utc=datetime(2025, 3, 10, 13, 30, tzinfo=UTC),
    close_utc=datetime(2025, 3, 10, 20, 0, tzinfo=UTC),
    is_early_close=False,
)
NEXT_SESSION = SessionDay(
    session_date=date(2025, 3, 11),
    open_utc=datetime(2025, 3, 11, 13, 30, tzinfo=UTC),
    close_utc=datetime(2025, 3, 11, 20, 0, tzinfo=UTC),
    is_early_close=False,
)
WINDOWS = compute_session_windows(
    SESSION,
    SessionWindowParams(
        no_entry_first_minutes=15,
        no_entry_last_minutes=30,
        holding_mode=HoldingMode.INTRADAY,
        flatten_minutes_before_close=10,
    ),
)
BAR_START = datetime(2025, 3, 10, 15, 0, tzinfo=UTC)
BAR_END = BAR_START + timedelta(minutes=5)
NOW = BAR_END + timedelta(seconds=10)
EQUITY = Decimal(100000)


def risk_params(**overrides: Any) -> RiskParams:
    values: dict[str, Any] = {
        "risk_per_trade_pct": Decimal("0.005"),
        "max_daily_loss_pct": Decimal("0.02"),
        "max_weekly_loss_pct": Decimal("0.04"),
        "max_drawdown_pct": Decimal("0.10"),
        "max_positions": 3,
        "max_total_exposure_pct": Decimal("0.90"),
        "max_symbol_exposure_pct": Decimal("0.30"),
        "max_aggregate_open_risk_pct": Decimal("0.015"),
        "slippage_buffer_bps": Decimal(5),
        "min_qty": 1,
    }
    values.update(overrides)
    return RiskParams(**values)


def params(**overrides: Any) -> CheckParams:
    values: dict[str, Any] = {
        "whitelist": (SYMBOL, "BBB"),
        "blacklist": ("ZZZ",),
        "min_price": Decimal(10),
        "max_price": Decimal(2000),
        "min_avg_daily_volume": 1000,
        "avg_volume_lookback_days": 5,
        "liquidity_feed": DataFeed.SIP,
        "max_spread_bps": Decimal(20),
        "spread_filter": SpreadFilter.ENFORCED,
        "max_bar_age_seconds": 90,
        "max_reconcile_age_seconds": 60,
        "cooldown_bars": 3,
        "min_minutes_per_bar": 3,
        "min_tp_distance_ticks": 2,
        "risk": risk_params(),
    }
    values.update(overrides)
    return CheckParams(**values)


def control(**overrides: Any) -> ControlFacts:
    values: dict[str, Any] = {
        "trading_enabled": True,
        "emergency_close": False,
        "stop_file_present": False,
        "ai_mode": AIMode.DISABLED,
    }
    values.update(overrides)
    return ControlFacts(**values)


def runtime(**overrides: Any) -> RuntimeFacts:
    values: dict[str, Any] = {
        "control": control(),
        "breaker_state": CircuitBreakerState.NORMAL,
        "system_mode": SystemMode.RUNNING,
        "reconciliation": ReconciliationFacts(
            last_clean_at_utc=NOW - timedelta(seconds=30), unresolved_mismatch=False
        ),
        "pending_owner_decisions": (),
        "tradable_symbols": frozenset({SYMBOL, "BBB"}),
        "market_open": True,
        "feed_connected": True,
    }
    values.update(overrides)
    return RuntimeFacts(**values)


def signal_bar(**overrides: Any) -> Bar:
    values: dict[str, Any] = {
        "symbol": SYMBOL,
        "timeframe": Timeframe.MIN_5,
        "bar_start_utc": BAR_START,
        "bar_end_utc": BAR_END,
        "open": Decimal("99.50"),
        "high": Decimal("100.20"),
        "low": Decimal("99.40"),
        "close": Decimal("100.00"),
        "volume": 50_000,
        "feed": DataFeed.IEX,
        "status": BarStatus.COMPLETE,
    }
    values.update(overrides)
    return Bar(**values)


def signal(**overrides: Any) -> Signal:
    values: dict[str, Any] = {
        "signal_id": "sig_test_0001",
        "symbol": SYMBOL,
        "timeframe": Timeframe.MIN_5,
        "bar_start_utc": BAR_START,
        "bar_end_utc": BAR_END,
        "created_at_utc": NOW,
        "expires_at_utc": BAR_END + timedelta(seconds=120),
        "rule_results": (),
        "strategy_version": "fixture",
    }
    values.update(overrides)
    return Signal(**values)


def trade(**overrides: Any) -> ProposedTrade:
    values: dict[str, Any] = {
        "signal_id": "sig_test_0001",
        "symbol": SYMBOL,
        "qty": 100,
        "entry_ref": Decimal("100.00"),
        "stop_price": Decimal("97.00"),
        "take_profit_price": Decimal("106.00"),
        "risk_per_share": Decimal("3.05"),
        "risk_amount": Decimal("305"),
        "risk_pct_of_equity": Decimal("0.00305"),
        "r_multiple": Decimal(2),
    }
    values.update(overrides)
    return ProposedTrade(**values)


def daily_bars(
    count: int = 10, *, volume: int = 1_000_000, feed: DataFeed = DataFeed.SIP
) -> tuple[Bar, ...]:
    """``count`` SIP daily bars of the sessions before :data:`SESSION`, labelled 04:00Z."""
    bars: list[Bar] = []
    day = SESSION.session_date
    while len(bars) < count:
        day -= timedelta(days=1)
        if day.weekday() >= 5:
            continue
        start = datetime(day.year, day.month, day.day, 4, 0, tzinfo=UTC)
        bars.append(
            Bar(
                symbol=SYMBOL,
                timeframe=Timeframe.DAY_1,
                bar_start_utc=start,
                bar_end_utc=start + timedelta(days=1),
                open=Decimal(100),
                high=Decimal(101),
                low=Decimal(99),
                close=Decimal(100),
                volume=volume,
                feed=feed,
                status=BarStatus.COMPLETE,
            )
        )
    return tuple(reversed(bars))


def quote(bid: str = "99.99", ask: str = "100.01") -> Quote:
    return Quote(
        symbol=SYMBOL,
        bid_price=Decimal(bid),
        ask_price=Decimal(ask),
        bid_size=Decimal(100),
        ask_size=Decimal(100),
        timestamp_utc=NOW,
    )


def market(**overrides: Any) -> MarketFacts:
    values: dict[str, Any] = {"daily_bars": daily_bars(), "quote": quote()}
    values.update(overrides)
    return MarketFacts(**values)


def account(**overrides: Any) -> AccountState:
    values: dict[str, Any] = {
        "equity": EQUITY,
        "last_equity": EQUITY,
        "buying_power": EQUITY,
        "status": "ACTIVE",
    }
    values.update(overrides)
    return AccountState(**values)


def portfolio(**overrides: Any) -> PortfolioFacts:
    values: dict[str, Any] = {
        "account": account(),
        "positions": (),
        "open_orders": (),
        "pending_entries": (),
        "position_stops": (),
        "week_start_equity": EQUITY,
        "peak_equity": EQUITY,
        "executed_signal_ids": frozenset(),
    }
    values.update(overrides)
    return PortfolioFacts(**values)


def intraday_timing(**overrides: Any) -> IntradayTiming:
    values: dict[str, Any] = {"windows": WINDOWS, "last_minute_bar_end_utc": BAR_END}
    values.update(overrides)
    return IntradayTiming(**values)


def daily_timing(**overrides: Any) -> DailyTiming:
    values: dict[str, Any] = {"signal_session": SESSION, "next_session": NEXT_SESSION}
    values.update(overrides)
    return DailyTiming(**values)


def context(**overrides: Any) -> CheckContext:
    """A context in which every check passes (intraday timing)."""
    values: dict[str, Any] = {
        "now_utc": NOW,
        "signal": signal(),
        "signal_bar": signal_bar(),
        "bars_since_last_exit": None,
        "trade": trade(),
        "params": params(),
        "timing": intraday_timing(),
        "runtime": runtime(),
        "market": market(),
        "portfolio": portfolio(),
    }
    values.update(overrides)
    return CheckContext(**values)


def daily_context(**overrides: Any) -> CheckContext:
    """A context in which every check passes (daily next-open timing, after the close)."""
    bar = signal_bar(
        timeframe=Timeframe.DAY_1, bar_start_utc=SESSION.open_utc, bar_end_utc=SESSION.close_utc
    )
    values: dict[str, Any] = {
        "now_utc": SESSION.close_utc + timedelta(seconds=11),
        "signal": signal(
            timeframe=Timeframe.DAY_1,
            bar_start_utc=SESSION.open_utc,
            bar_end_utc=SESSION.close_utc,
            created_at_utc=SESSION.close_utc + timedelta(seconds=11),
            expires_at_utc=NEXT_SESSION.open_utc + timedelta(seconds=120),
        ),
        "signal_bar": bar,
        "timing": daily_timing(),
        "runtime": runtime(
            market_open=False,
            reconciliation=ReconciliationFacts(
                last_clean_at_utc=SESSION.close_utc, unresolved_mismatch=False
            ),
        ),
    }
    values.update(overrides)
    return context(**values)
