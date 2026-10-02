"""Strategy decisions (sec. 5.2, 13.4-13.7). All parameter values are TEST FIXTURES."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from domain.market.session import OwnerDecisionPendingError
from domain.models import BarStatus, ExitReason, StrategyAction, Timeframe
from domain.strategy.rules import StrategyConfigError, StrategyInputError
from domain.strategy.signals import signal_id_for
from domain.strategy.strategy import (
    EXIT_TIME_STOP,
    NOTRADE_COOLDOWN,
    NOTRADE_DATA_UNAVAILABLE,
    NOTRADE_INCOMPLETE_BAR,
    Strategy,
    StrategyDecision,
    StrategyParams,
    bar_allows_entry,
    cooldown_active,
    count_bars_since,
    pending_strategy_params,
    required_warmup_bars,
)
from tests.unit.strategy.factories import (
    CREATED_AT,
    DAY_1,
    DAY_2,
    ENTRY_CONFIRM_01,
    ENTRY_MOMENTUM_01,
    ENTRY_PRICE_01,
    ENTRY_TREND_01,
    ENTRY_VOLUME_01,
    FIXTURE_STRATEGY_VERSION,
    NOTRADE_GAP,
    NOTRADE_RSI_HIGH,
    NOTRADE_VOLATILITY,
    bar,
    empty_bar,
    make_params,
    make_strategy,
    params_dict,
    rule,
    s,
    series_bars,
    uptrend_bars,
)

PENDING_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "config.pending.yaml"
)
"""The pre-2026-10-02 config.yaml with every OWNER_DECISION still null."""


def _results(decision: StrategyDecision) -> dict[str, bool]:
    return {r.rule_id: r.result for r in decision.rule_results}


def _entry(
    strategy: Strategy,
    bars: list[Any],
    *,
    confirmation: list[Any] | None = None,
    sessions: list[Any] | None = None,
    since: int | None = None,
) -> StrategyDecision:
    context = strategy.build_context(bars, confirmation, sessions=sessions or ())
    return strategy.evaluate_entry(context, bars_since_last_exit=since, created_at_utc=CREATED_AT)


# --------------------------------------------------------------------------- params


def test_pending_config_parses_and_is_pending() -> None:
    config = yaml.safe_load(PENDING_CONFIG_PATH.read_text(encoding="utf-8"))
    params = StrategyParams.model_validate(config["strategy"])
    pending = pending_strategy_params(params)
    assert "strategy.primary_timeframe" in pending
    assert "strategy.entry_rules" in pending
    assert "strategy.indicators.atr_period" in pending
    with pytest.raises(OwnerDecisionPendingError) as excinfo:
        Strategy(params, strategy_version=config["strategy_version"])
    assert excinfo.value.parameter == pending[0]


@pytest.mark.parametrize(
    ("override", "missing"),
    [
        ({"primary_timeframe": None}, "strategy.primary_timeframe"),
        ({"signal_ttl_seconds": None}, "strategy.signal_ttl_seconds"),
        ({"cooldown_bars": None}, "strategy.cooldown_bars"),
        ({"min_minutes_per_bar": None}, "strategy.min_minutes_per_bar"),
        ({"entry_rules": None}, "strategy.entry_rules"),
        ({"no_trade_rules": None}, "strategy.no_trade_rules"),
        ({"exit_rules": None}, "strategy.exit_rules"),
    ],
)
def test_pending_owner_decision_raises(override: dict[str, Any], missing: str) -> None:
    with pytest.raises(OwnerDecisionPendingError) as excinfo:
        make_strategy(**override)
    assert excinfo.value.parameter == missing


@pytest.mark.parametrize(
    ("section", "key", "missing"),
    [
        ("exit", "exit_on_signal_reversal", "strategy.exit.exit_on_signal_reversal"),
        ("indicators", "atr_period", "strategy.indicators.atr_period"),
        ("indicators", "ema_fast", "strategy.indicators.ema_fast"),
        ("indicators", "rsi_period", "strategy.indicators.rsi_period"),
        ("indicators", "volume_avg_period", "strategy.indicators.volume_avg_period"),
        ("no_trade_thresholds", "rsi_overbought", "strategy.no_trade_thresholds.rsi_overbought"),
    ],
)
def test_pending_nested_owner_decision_raises(section: str, key: str, missing: str) -> None:
    data = params_dict()
    data[section][key] = None
    params = StrategyParams.model_validate(data)
    assert missing in pending_strategy_params(params)
    with pytest.raises(OwnerDecisionPendingError) as excinfo:
        Strategy(params, strategy_version=FIXTURE_STRATEGY_VERSION)
    assert excinfo.value.parameter == missing


def test_unreferenced_values_are_not_required() -> None:
    data = params_dict(
        entry_rules=[ENTRY_TREND_01],
        no_trade_rules=[],
        exit_rules=None,
    )
    data["indicators"]["rsi_period"] = None
    data["indicators"]["volume_avg_period"] = None
    data["no_trade_thresholds"] = {"rsi_overbought": None, "max_gap_pct": None}
    data["exit"]["exit_on_signal_reversal"] = False
    data["exit"]["time_stop_bars"] = None
    params = StrategyParams.model_validate(data)
    assert pending_strategy_params(params) == ()
    Strategy(params, strategy_version=FIXTURE_STRATEGY_VERSION)


def test_confirmation_timeframe_required_when_referenced() -> None:
    params = make_params(entry_rules=[ENTRY_TREND_01, ENTRY_CONFIRM_01])
    assert pending_strategy_params(params) == ("strategy.confirmation_timeframe",)
    with pytest.raises(OwnerDecisionPendingError):
        Strategy(params, strategy_version=FIXTURE_STRATEGY_VERSION)


@pytest.mark.parametrize(
    "override",
    [
        {"entry_rules": []},
        {"entry_rules": [ENTRY_TREND_01, ENTRY_TREND_01]},
        {"no_trade_rules": [rule("NOTRADE_COOLDOWN", s("rsi"), ">", {"const": "1"})]},
        {"no_trade_rules": [rule("NOTRADE_X", s("rsi"), ">", {"param": "unknown"})]},
        {"min_minutes_per_bar": 6},
        {"min_minutes_per_bar": 0},
        {"cooldown_bars": -1},
        {"signal_ttl_seconds": 0},
        {"primary_timeframe": "7Min"},
        {"unknown_key": 1},
        {"no_trade_thresholds": {"rsi_overbought": 70.5}},
    ],
)
def test_structurally_invalid_params_are_rejected(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make_params(**override)


def test_time_stop_bars_must_be_positive() -> None:
    data = params_dict()
    data["exit"]["time_stop_bars"] = 0
    with pytest.raises(ValidationError):
        StrategyParams.model_validate(data)


def test_empty_strategy_version_rejected() -> None:
    with pytest.raises(StrategyConfigError):
        Strategy(make_params(), strategy_version="")


def test_required_warmup_bars() -> None:
    assert required_warmup_bars(make_params()) == 5  # ema_slow period 5
    deeper = make_params(entry_rules=[rule("ENTRY_X", s("close", -1), ">", s("ema_slow", -3))])
    assert required_warmup_bars(deeper) == 7


def test_optional_indicators_leave_the_warm_up_unchanged_until_set() -> None:
    explicit = params_dict()
    explicit["indicators"] |= {"sma_short_period": None, "sma_long_period": None}
    assert required_warmup_bars(StrategyParams.model_validate(explicit)) == 5
    longer = params_dict()
    longer["indicators"] |= {"sma_short_period": 2, "sma_long_period": 8}
    assert required_warmup_bars(StrategyParams.model_validate(longer)) == 8


@pytest.mark.parametrize(
    ("series", "param"), [("sma_short", "sma_short_period"), ("sma_long", "sma_long_period")]
)
def test_rule_on_an_unset_optional_indicator_is_a_config_error(series: str, param: str) -> None:
    for section in ("entry_rules", "no_trade_rules", "exit_rules"):
        data = params_dict()
        data[section] = [*data[section], rule("RULE_SMA_01", s("close"), ">", s(series))]
        with pytest.raises(ValidationError, match=f"RULE_SMA_01 -> strategy.indicators.{param}"):
            StrategyParams.model_validate(data)
    data = params_dict(entry_rules=[rule("RULE_SMA_01", s("close"), ">", s(series))])
    data["indicators"][param] = 3
    assert pending_strategy_params(StrategyParams.model_validate(data)) == ()


def test_ibs_rule_needs_no_indicator_period() -> None:
    strategy = make_strategy(entry_rules=[rule("ENTRY_IBS_01", s("ibs"), "<", {"const": "0.9"})])
    decision = _entry(strategy, uptrend_bars())
    assert decision.action is StrategyAction.BUY
    assert decision.rule_results[0].values["ibs[-1]"] is not None


def test_entry_signal_expiry_can_be_set_by_the_caller() -> None:
    strategy = make_strategy()
    context = strategy.build_context(uptrend_bars())
    expires = CREATED_AT + timedelta(hours=20)
    decision = strategy.evaluate_entry(
        context, bars_since_last_exit=None, created_at_utc=CREATED_AT, expires_at_utc=expires
    )
    assert decision.signal is not None
    assert decision.signal.expires_at_utc == expires
    default = strategy.evaluate_entry(context, bars_since_last_exit=None, created_at_utc=CREATED_AT)
    assert default.signal is not None
    assert default.signal.expires_at_utc == default.signal.bar_end_utc + timedelta(seconds=120)


# --------------------------------------------------------------------------- entries


def test_buy_when_all_entry_rules_true_and_no_no_trade() -> None:
    strategy = make_strategy()
    bars = uptrend_bars()
    decision = _entry(strategy, bars)
    assert decision.action is StrategyAction.BUY
    assert decision.signal is not None
    assert decision.exit_reason is None
    results = _results(decision)
    assert results == {
        "ENTRY_TREND_01": True,
        "ENTRY_PRICE_01": True,
        "ENTRY_MOMENTUM_01": True,
        "ENTRY_VOLUME_01": True,
        "NOTRADE_RSI_HIGH": False,
        "NOTRADE_VOLATILITY": False,
        NOTRADE_INCOMPLETE_BAR: False,
        NOTRADE_COOLDOWN: False,
        NOTRADE_DATA_UNAVAILABLE: False,
    }
    signal = decision.signal
    assert signal.rule_results == decision.rule_results
    assert signal.bar_start_utc == bars[-1].bar_start_utc
    assert signal.bar_end_utc == bars[-1].bar_end_utc
    assert signal.expires_at_utc == bars[-1].bar_end_utc + timedelta(seconds=120)
    assert signal.created_at_utc == CREATED_AT
    assert signal.timeframe is Timeframe.MIN_5
    trend = decision.rule_results[0]
    assert set(trend.values) == {"ema_fast[-1]", "ema_slow[-1]"}


def test_no_action_when_one_entry_rule_false() -> None:
    decision = _entry(make_strategy(), uptrend_bars(last_volume=500))
    assert decision.action is StrategyAction.NO_ACTION
    assert decision.signal is None
    assert _results(decision)["ENTRY_VOLUME_01"] is False


@pytest.mark.parametrize(
    ("thresholds", "fired"),
    [
        (
            {"rsi_overbought": "80", "max_atr_pct": "0.05", "max_gap_pct": "0.02"},
            "NOTRADE_RSI_HIGH",
        ),
        (
            {"rsi_overbought": "99", "max_atr_pct": "0.01", "max_gap_pct": "0.02"},
            "NOTRADE_VOLATILITY",
        ),
    ],
)
def test_no_trade_rule_blocks_buy(thresholds: dict[str, str], fired: str) -> None:
    decision = _entry(make_strategy(no_trade_thresholds=thresholds), uptrend_bars())
    assert decision.action is StrategyAction.NO_ACTION
    results = _results(decision)
    assert results[fired] is True
    assert all(results[r] for r in ("ENTRY_TREND_01", "ENTRY_PRICE_01", "ENTRY_MOMENTUM_01"))


def _two_session_bars(day2_open: str) -> list[Any]:
    day1 = series_bars(
        ["100", "101", "100.5", "102", "101.5", "103", "102.5", "104", "103.5", "105"],
        start=DAY_1.open_utc,
    )
    day2 = [
        bar(DAY_2.open_utc, "105.2", open_=day2_open),
        bar(DAY_2.open_utc + timedelta(minutes=5), "106", open_="105.2", volume=3000),
    ]
    return [*day1, *day2]


def test_gap_no_trade_rule() -> None:
    strategy = make_strategy(no_trade_rules=[NOTRADE_RSI_HIGH, NOTRADE_GAP])
    sessions = [DAY_1, DAY_2]
    small_gap = _entry(strategy, _two_session_bars("105.1"), sessions=sessions)
    big_gap = _entry(strategy, _two_session_bars("108"), sessions=sessions)
    assert _results(small_gap)["NOTRADE_GAP"] is False
    assert small_gap.action is StrategyAction.BUY
    assert _results(big_gap)["NOTRADE_GAP"] is True
    assert big_gap.action is StrategyAction.NO_ACTION


def test_unevaluable_no_trade_rule_fails_closed() -> None:
    strategy = make_strategy(no_trade_rules=[NOTRADE_GAP])
    decision = _entry(strategy, uptrend_bars())  # no sessions -> gap_pct is None
    results = _results(decision)
    assert results["NOTRADE_GAP"] is False  # sec. 14.3: None -> False
    assert results[NOTRADE_DATA_UNAVAILABLE] is True
    assert decision.action is StrategyAction.NO_ACTION
    unavailable = next(r for r in decision.rule_results if r.rule_id == NOTRADE_DATA_UNAVAILABLE)
    assert unavailable.values == {"rules": "NOTRADE_GAP"}


def test_insufficient_history_gives_no_action() -> None:
    decision = _entry(make_strategy(), uptrend_bars()[:3])
    assert decision.action is StrategyAction.NO_ACTION
    assert _results(decision)["ENTRY_TREND_01"] is False  # ema_slow None -> False


def test_confirmation_rule() -> None:
    strategy = make_strategy(
        confirmation_timeframe="15Min",
        entry_rules=[ENTRY_TREND_01, ENTRY_PRICE_01, ENTRY_CONFIRM_01],
    )
    primary = uptrend_bars()  # signal bar ends 14:30 UTC
    start = DAY_1.open_utc - timedelta(minutes=90)
    rising_closes = [str(100 + i) for i in range(12)]
    rising = series_bars(rising_closes, timeframe=Timeframe.MIN_15, start=start)
    falling = series_bars(rising_closes[::-1], timeframe=Timeframe.MIN_15, start=start)
    # 10 confirmation bars end by 14:30; the last 2 would be lookahead and are dropped.
    context = strategy.build_context(primary, rising)
    assert context.confirmation is not None
    assert len(context.confirmation) == 10
    assert _entry(strategy, primary, confirmation=rising).action is StrategyAction.BUY
    falling_decision = _entry(strategy, primary, confirmation=falling)
    assert falling_decision.action is StrategyAction.NO_ACTION
    assert _results(falling_decision)["ENTRY_CONFIRM_01"] is False


@pytest.mark.parametrize(
    ("minutes_present", "action", "blocked"),
    [
        (3, StrategyAction.BUY, False),
        (4, StrategyAction.BUY, False),
        (2, StrategyAction.NO_ACTION, True),
    ],
)
def test_incomplete_bar_uses_min_minutes_per_bar(
    minutes_present: int, action: StrategyAction, blocked: bool
) -> None:
    bars = uptrend_bars()
    last = bars[-1].model_copy(
        update={"status": BarStatus.INCOMPLETE, "minutes_present": minutes_present}
    )
    decision = _entry(make_strategy(min_minutes_per_bar=3), [*bars[:-1], last])
    assert decision.action is action
    assert _results(decision)[NOTRADE_INCOMPLETE_BAR] is blocked
    incomplete = next(r for r in decision.rule_results if r.rule_id == NOTRADE_INCOMPLETE_BAR)
    assert incomplete.values == {
        "bar_status": "INCOMPLETE",
        "minutes_present": minutes_present,
        "min_minutes_per_bar": 3,
    }


def test_empty_signal_bar_is_never_evaluated() -> None:
    bars = uptrend_bars()
    decision = _entry(make_strategy(), [*bars, empty_bar(bars[-1].bar_end_utc)])
    assert decision.action is StrategyAction.NO_ACTION
    assert decision.bar_status is BarStatus.EMPTY
    assert [r.rule_id for r in decision.rule_results] == [NOTRADE_INCOMPLETE_BAR, NOTRADE_COOLDOWN]
    assert _results(decision)[NOTRADE_INCOMPLETE_BAR] is True


@pytest.mark.parametrize(
    ("since", "action"),
    [
        (0, StrategyAction.NO_ACTION),
        (2, StrategyAction.NO_ACTION),
        (3, StrategyAction.BUY),
        (10, StrategyAction.BUY),
        (None, StrategyAction.BUY),
    ],
)
def test_cooldown(since: int | None, action: StrategyAction) -> None:
    decision = _entry(make_strategy(cooldown_bars=3), uptrend_bars(), since=since)
    assert decision.action is action
    assert _results(decision)[NOTRADE_COOLDOWN] is (action is StrategyAction.NO_ACTION)


def test_entry_input_errors() -> None:
    strategy = make_strategy()
    context = strategy.build_context(uptrend_bars())
    with pytest.raises(StrategyInputError):
        strategy.evaluate_entry(context, bars_since_last_exit=-1, created_at_utc=CREATED_AT)
    with pytest.raises(StrategyInputError):
        strategy.evaluate_entry(
            strategy.build_context([]), bars_since_last_exit=None, created_at_utc=CREATED_AT
        )
    with pytest.raises(StrategyInputError):
        strategy.build_context(series_bars(["1", "2"], timeframe=Timeframe.MIN_15))


# --------------------------------------------------------------------------- AC-03


def test_ac03_duplicate_bar_gives_same_signal_id() -> None:
    bars = uptrend_bars()
    first = _entry(make_strategy(), bars)
    # A restarted process (new Strategy) seeing the same bar again (duplicate delivery).
    later = datetime(2026, 6, 15, 14, 31, tzinfo=UTC)
    restarted = make_strategy()
    second = restarted.evaluate_entry(
        restarted.build_context([b.model_copy() for b in bars]),
        bars_since_last_exit=None,
        created_at_utc=later,
    )
    assert first.signal is not None
    assert second.signal is not None
    assert first.signal.signal_id == second.signal.signal_id
    assert first.signal.signal_id == signal_id_for(
        strategy_version=FIXTURE_STRATEGY_VERSION,
        symbol="SPY",
        timeframe=Timeframe.MIN_5,
        bar_start_utc=bars[-1].bar_start_utc,
    )


def test_ac03_strategy_version_changes_signal_id() -> None:
    bars = uptrend_bars()
    v1 = _entry(make_strategy(version="1.0.0"), bars)
    v2 = _entry(make_strategy(version="1.0.1"), bars)
    assert v1.signal is not None
    assert v2.signal is not None
    assert v1.signal.signal_id != v2.signal.signal_id


def test_decisions_are_deterministic() -> None:
    bars = uptrend_bars()
    assert _entry(make_strategy(), bars) == _entry(make_strategy(), bars)


# --------------------------------------------------------------------------- positions


def _position(strategy: Strategy, bars: list[Any], bars_held: int) -> StrategyDecision:
    return strategy.evaluate_position(strategy.build_context(bars), bars_held=bars_held)


def test_hold_when_no_exit_fires() -> None:
    decision = _position(make_strategy(), uptrend_bars(), bars_held=2)
    assert decision.action is StrategyAction.HOLD
    assert decision.exit_reason is None
    assert _results(decision) == {"EXIT_REVERSAL_01": False, EXIT_TIME_STOP: False}


@pytest.mark.parametrize(("held", "closes"), [(9, False), (10, True), (11, True)])
def test_time_stop(held: int, closes: bool) -> None:
    decision = _position(make_strategy(), uptrend_bars(), bars_held=held)
    assert (decision.action is StrategyAction.CLOSE) is closes
    if closes:
        assert decision.exit_reason is ExitReason.TIME_STOP


def test_time_stop_disabled_when_null() -> None:
    data = params_dict()
    data["exit"]["time_stop_bars"] = None
    strategy = Strategy(StrategyParams.model_validate(data), strategy_version="x")
    decision = _position(strategy, uptrend_bars(), bars_held=10_000)
    assert decision.action is StrategyAction.HOLD
    time_stop = next(r for r in decision.rule_results if r.rule_id == EXIT_TIME_STOP)
    assert time_stop.values == {"bars_held": 10_000, "time_stop_bars": None}


DOWNTREND = ["106", "105", "105.5", "104", "104.5", "103", "103.5", "102", "102.5", "101"]


def test_signal_reversal_closes() -> None:
    decision = _position(make_strategy(), series_bars(DOWNTREND), bars_held=1)
    assert decision.action is StrategyAction.CLOSE
    assert decision.exit_reason is ExitReason.SIGNAL_REVERSAL


def test_signal_reversal_ignored_when_disabled() -> None:
    data = params_dict()
    data["exit"]["exit_on_signal_reversal"] = False
    strategy = Strategy(StrategyParams.model_validate(data), strategy_version="x")
    decision = _position(strategy, series_bars(DOWNTREND), bars_held=1)
    assert decision.action is StrategyAction.HOLD
    assert [r.rule_id for r in decision.rule_results] == [EXIT_TIME_STOP]


def test_time_stop_wins_over_reversal() -> None:
    decision = _position(make_strategy(), series_bars(DOWNTREND), bars_held=10)
    assert decision.exit_reason is ExitReason.TIME_STOP
    assert _results(decision)["EXIT_REVERSAL_01"] is True


def test_empty_bar_skips_reversal_but_not_time_stop() -> None:
    bars = series_bars(DOWNTREND)
    with_empty = [*bars, empty_bar(bars[-1].bar_end_utc)]
    hold = _position(make_strategy(), with_empty, bars_held=1)
    assert hold.action is StrategyAction.HOLD
    assert [r.rule_id for r in hold.rule_results] == [EXIT_TIME_STOP]
    stop = _position(make_strategy(), with_empty, bars_held=10)
    assert stop.exit_reason is ExitReason.TIME_STOP


def test_position_rejects_negative_bars_held() -> None:
    with pytest.raises(StrategyInputError):
        _position(make_strategy(), uptrend_bars(), bars_held=-1)


# --------------------------------------------------------------------------- helpers


def test_bar_allows_entry() -> None:
    complete = bar(DAY_1.open_utc, "1")
    incomplete = bar(DAY_1.open_utc, "1", status=BarStatus.INCOMPLETE, minutes_present=3)
    assert bar_allows_entry(complete, 5)
    assert bar_allows_entry(incomplete, 3)
    assert not bar_allows_entry(incomplete, 4)
    assert not bar_allows_entry(empty_bar(DAY_1.open_utc), 1)


def test_cooldown_active() -> None:
    assert cooldown_active(0, 1)
    assert not cooldown_active(1, 1)
    assert not cooldown_active(None, 5)
    assert not cooldown_active(0, 0)


def test_count_bars_since() -> None:
    bars = series_bars(["1", "2", "3", "4"])  # ends 13:35, 13:40, 13:45, 13:50
    assert count_bars_since(bars, datetime(2026, 6, 15, 13, 37, tzinfo=UTC)) == 3
    assert count_bars_since(bars, datetime(2026, 6, 15, 13, 40, tzinfo=UTC)) == 2
    assert count_bars_since(bars, datetime(2026, 6, 15, 13, 50, tzinfo=UTC)) == 0
    with pytest.raises(StrategyInputError):
        count_bars_since(bars, datetime(2026, 6, 15, 13, 40))  # noqa: DTZ001


def test_decision_consistency_is_enforced() -> None:
    decision = _entry(make_strategy(), uptrend_bars())
    with pytest.raises(ValidationError):
        StrategyDecision.model_validate({**decision.model_dump(), "signal": None})
    with pytest.raises(ValidationError):
        StrategyDecision.model_validate(
            {**decision.model_dump(), "action": "CLOSE", "signal": None}
        )


def test_fixture_entry_rules_mirror_spec_examples() -> None:
    # The conceptual sec. 13.4 rules are expressible in the rule language.
    for data in (
        ENTRY_TREND_01,
        ENTRY_PRICE_01,
        ENTRY_MOMENTUM_01,
        ENTRY_VOLUME_01,
        ENTRY_CONFIRM_01,
        NOTRADE_RSI_HIGH,
        NOTRADE_VOLATILITY,
        NOTRADE_GAP,
    ):
        assert make_params(entry_rules=[data], no_trade_rules=[]).entry_rules is not None
