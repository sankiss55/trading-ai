"""Sec. 19 / 56.3: one implementation per check, reused by every caller (AST-level checks).

The safety checks live in ``domain/guards/checks.py``. Outside ``domain/`` nothing may call
the check primitives directly (entry windows, expiry comparisons, limits, universe and
freshness checks...), so ``application.market_flow`` and the backtest cannot grow an inline
duplicate of a library check again.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest

import domain.guards.checks as checks
from application.control_rules import trading_block_reasons
from domain.guards.checks import ControlFacts, kill_switch_reasons
from domain.models import AIMode, SystemControl

ROOT = Path(__file__).resolve().parents[3]
OUTER_LAYERS = ("application", "app", "backtest", "adapters", "shadow")

CHECK_PRIMITIVES = frozenset(
    {
        # domain.market.session / universe / quality
        "is_entry_window",
        "check_symbol_listed",
        "check_tradable",
        "check_price_range",
        "check_avg_daily_volume",
        "average_daily_volume",
        "check_spread",
        "check_quote_freshness",
        "check_data_fresh",
        "check_corporate_action",
        "check_staleness",
        # domain.risk
        "check_exit_levels",
        "evaluate_limits",
        "check_risk_per_trade",
        "check_daily_loss",
        "check_weekly_loss",
        "check_drawdown",
        "check_max_positions",
        "check_total_exposure",
        "check_symbol_exposure",
        "check_aggregate_open_risk",
        # domain.strategy built-ins reused by the library
        "bar_allows_entry",
        "cooldown_active",
    }
)
"""Names only the check library (and the domain itself) may call."""

ALLOWED_OUTER_USES: dict[str, frozenset[str]] = {
    # Not a check: the runner's end-of-day flatten procedure (sec. 23.5, Phase 5 moves it).
    "backtest/runner.py": frozenset({"is_flatten_time"}),
}


def _python_files(layer: str) -> list[Path]:
    return sorted(p for p in (ROOT / layer).rglob("*.py") if "__pycache__" not in p.parts)


def _used_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("layer", OUTER_LAYERS)
def test_outer_layers_never_call_check_primitives(layer: str) -> None:
    offenders: list[str] = []
    for path in _python_files(layer):
        relative = path.relative_to(ROOT).as_posix()
        used = _used_names(ast.parse(path.read_text(encoding="utf-8")))
        forbidden = (used & CHECK_PRIMITIVES) - ALLOWED_OUTER_USES.get(relative, frozenset())
        offenders.extend(f"{relative}: {name}" for name in sorted(forbidden))
    assert not offenders, offenders


def _library_codes() -> set[str]:
    codes = {code.value for code in checks.CheckCode}
    codes |= {
        value
        for name, value in vars(checks).items()
        if name.isupper() and isinstance(value, str) and name in checks.__all__
    }
    return codes


def test_market_flow_has_no_inline_check() -> None:
    """``market_flow`` routes and prices; every check result comes from the library."""
    path = ROOT / "application" / "market_flow.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exported = {  # names listed in __all__ (re-exports of library constants)
        id(node)
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in statement.targets)
        for node in ast.walk(statement.value)
    }
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in exported
    }
    assert not (literals & _library_codes()), sorted(literals & _library_codes())
    comparisons = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(side, ast.Attribute)
            and side.attr
            in {
                "expires_at_utc",
                "entries_allowed_from_utc",
                "entries_allowed_until_utc",
                "flatten_at_utc",
                "trading_enabled",
                "breaker_state",
            }
            for side in (node.left, *node.comparators)
        )
    ]
    assert not comparisons, [ast.unparse(node) for node in comparisons]
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "run_pre_ai_checks" in calls


@pytest.mark.parametrize(
    ("stop", "emergency", "enabled"), list(itertools.product((False, True), repeat=3))
)
def test_control_rules_delegate_the_kill_switch(stop: bool, emergency: bool, enabled: bool) -> None:
    control = SystemControl(trading_enabled=enabled, emergency_close=emergency)
    assert trading_block_reasons(control, stop_file_present=stop) == kill_switch_reasons(
        ControlFacts(
            trading_enabled=enabled,
            emergency_close=emergency,
            stop_file_present=stop,
            ai_mode=AIMode.DISABLED,
        )
    )
