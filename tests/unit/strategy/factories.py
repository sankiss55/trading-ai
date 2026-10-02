"""Test factories for strategy tests.

Every parameter value here is an explicit TEST FIXTURE chosen to exercise the code. None
of them is an owner value (OWNER_DECISION) and none should be copied into config.yaml.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from domain.models import Bar, BarStatus, DataFeed, SessionDay, Timeframe
from domain.strategy.strategy import Strategy, StrategyParams

FIXTURE_STRATEGY_VERSION = "test-0.0.1"

DAY_1 = SessionDay(
    session_date=date(2026, 6, 15),
    open_utc=datetime(2026, 6, 15, 13, 30, tzinfo=UTC),
    close_utc=datetime(2026, 6, 15, 20, 0, tzinfo=UTC),
    is_early_close=False,
)
DAY_2 = SessionDay(
    session_date=date(2026, 6, 16),
    open_utc=datetime(2026, 6, 16, 13, 30, tzinfo=UTC),
    close_utc=datetime(2026, 6, 16, 20, 0, tzinfo=UTC),
    is_early_close=False,
)

CREATED_AT = datetime(2026, 6, 15, 14, 30, 2, tzinfo=UTC)

# Zig-zag uptrend: EMA fast > EMA slow, close > EMA fast, RSI between 50 and 100.
UPTREND_CLOSES: tuple[str, ...] = (
    "100", "101", "100.5", "102", "101.5", "103", "102.5", "104", "103.5", "105", "104.5", "106",
)  # fmt: skip


def rule(
    rule_id: str,
    left: dict[str, Any],
    op: str,
    right: dict[str, Any],
    *,
    timeframe: str = "primary",
) -> dict[str, Any]:
    """A rule spec dict as it would appear in config.yaml."""
    return {"rule_id": rule_id, "timeframe": timeframe, "left": left, "op": op, "right": right}


def s(name: str, offset: int = -1) -> dict[str, Any]:
    """Series operand."""
    return {"series": name, "offset": offset}


# Conceptual example rules of sec. 13.4 / 13.5 (TEST FIXTURES).
ENTRY_TREND_01 = rule("ENTRY_TREND_01", s("ema_fast"), ">", s("ema_slow"))
ENTRY_PRICE_01 = rule("ENTRY_PRICE_01", s("close"), ">", s("ema_fast"))
ENTRY_MOMENTUM_01 = rule("ENTRY_MOMENTUM_01", s("rsi"), ">", {"const": "50"})
ENTRY_VOLUME_01 = rule("ENTRY_VOLUME_01", s("volume"), ">", s("volume_avg"))
ENTRY_CONFIRM_01 = rule(
    "ENTRY_CONFIRM_01", s("ema_fast_confirm"), ">", s("ema_slow_confirm"), timeframe="confirmation"
)
NOTRADE_RSI_HIGH = rule("NOTRADE_RSI_HIGH", s("rsi"), ">", {"param": "rsi_overbought"})
NOTRADE_VOLATILITY = rule("NOTRADE_VOLATILITY", s("atr_pct"), ">", {"param": "max_atr_pct"})
NOTRADE_GAP = rule("NOTRADE_GAP", s("gap_pct"), ">", {"param": "max_gap_pct"})
EXIT_REVERSAL_01 = rule("EXIT_REVERSAL_01", s("ema_fast"), "<", s("ema_slow"))

_BASE_PARAMS: dict[str, Any] = {
    "holding_mode": "intraday",
    "flatten_minutes_before_close": 5,
    "primary_timeframe": "5Min",
    "confirmation_timeframe": None,
    "signal_ttl_seconds": 120,
    "cooldown_bars": 3,
    "min_minutes_per_bar": 3,
    "indicators": {
        "ema_fast": 3,
        "ema_slow": 5,
        "rsi_period": 3,
        "atr_period": 3,
        "volume_avg_period": 3,
    },
    "entry_rules": [ENTRY_TREND_01, ENTRY_PRICE_01, ENTRY_MOMENTUM_01, ENTRY_VOLUME_01],
    "no_trade_rules": [NOTRADE_RSI_HIGH, NOTRADE_VOLATILITY],
    "no_trade_thresholds": {
        "rsi_overbought": "99",
        "max_atr_pct": "0.05",
        "max_gap_pct": "0.02",
    },
    "exit": {
        "stop_method": "atr",
        "stop_atr_multiplier": "2",
        "take_profit_r_multiple": "2",
        "time_stop_bars": 10,
        "exit_on_signal_reversal": True,
        "min_tp_distance_ticks": 2,
    },
    "exit_rules": [EXIT_REVERSAL_01],
}


def params_dict(**overrides: Any) -> dict[str, Any]:
    """Deep copy of the fixture params with top-level overrides."""
    data = copy.deepcopy(_BASE_PARAMS)
    data.update(copy.deepcopy(overrides))
    return data


def make_params(**overrides: Any) -> StrategyParams:
    """Validated fixture :class:`StrategyParams`."""
    return StrategyParams.model_validate(params_dict(**overrides))


def make_strategy(version: str = FIXTURE_STRATEGY_VERSION, **overrides: Any) -> Strategy:
    """A :class:`Strategy` built from fixture params."""
    return Strategy(make_params(**overrides), strategy_version=version)


def bar(
    start: datetime,
    close: str,
    *,
    open_: str | None = None,
    volume: int = 1000,
    timeframe: Timeframe = Timeframe.MIN_5,
    status: BarStatus = BarStatus.COMPLETE,
    minutes_present: int | None = None,
    symbol: str = "SPY",
) -> Bar:
    """A priced bar; high/low are 0.5 around open/close."""
    minutes = timeframe.minutes or 1
    end = start + timedelta(minutes=minutes)
    close_d = Decimal(close)
    open_d = Decimal(open_) if open_ is not None else close_d
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        bar_start_utc=start,
        bar_end_utc=end,
        open=open_d,
        high=max(open_d, close_d) + Decimal("0.5"),
        low=min(open_d, close_d) - Decimal("0.5"),
        close=close_d,
        volume=volume,
        feed=DataFeed.IEX,
        status=status,
        minutes_present=minutes_present,
    )


def empty_bar(start: datetime, *, timeframe: Timeframe = Timeframe.MIN_5) -> Bar:
    """An EMPTY bar (no prices, zero volume)."""
    return Bar(
        symbol="SPY",
        timeframe=timeframe,
        bar_start_utc=start,
        bar_end_utc=start + timedelta(minutes=timeframe.minutes or 1),
        open=None,
        high=None,
        low=None,
        close=None,
        volume=0,
        feed=DataFeed.IEX,
        status=BarStatus.EMPTY,
    )


def series_bars(
    closes: Sequence[str],
    *,
    start: datetime = DAY_1.open_utc,
    volumes: Sequence[int] | None = None,
    timeframe: Timeframe = Timeframe.MIN_5,
) -> list[Bar]:
    """Consecutive bars; each opens at the previous close."""
    step = timedelta(minutes=timeframe.minutes or 1)
    out: list[Bar] = []
    for index, close in enumerate(closes):
        volume = volumes[index] if volumes is not None else 1000
        open_ = closes[index - 1] if index else close
        out.append(
            bar(start + index * step, close, open_=open_, volume=volume, timeframe=timeframe)
        )
    return out


def uptrend_bars(last_volume: int = 3000) -> list[Bar]:
    """Bars on which every fixture entry rule is true (volume spike on the last bar)."""
    volumes = [1000] * (len(UPTREND_CLOSES) - 1) + [last_volume]
    return series_bars(UPTREND_CLOSES, volumes=volumes)
