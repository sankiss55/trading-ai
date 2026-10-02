"""Architecture test (AC-21): dependency rules of sec. 8.3.

Two independent mechanisms verify the same rules:

1. An AST walk over ``domain/`` and ``application/`` (always runs, no extra tooling).
2. ``lint-imports`` (import-linter contracts in ``pyproject.toml``), when installed.

The AST walk is slightly stricter than import-linter: third-party imports from
``domain/`` and ``application/`` must belong to an explicit allowlist, and reading the
system wall clock is forbidden (sec. 8.3.6: time comes from ``IClock``).

Spawning OS processes (``subprocess``, ``asyncio.subprocess``,
``asyncio.create_subprocess_*``, ``os.system``, ``os.popen``, ``os.spawn*``, ``os.exec*``)
is forbidden in ``domain/`` and ``application/``: the Claude Code CLI used by the AI
filter (owner decision 2026-10-02, ``docs/DECISIONS.md``) may only run from ``adapters/``.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERNAL_PACKAGES = frozenset({"domain", "application", "adapters", "app", "backtest", "shadow"})

INFRASTRUCTURE_MODULES = frozenset(
    {
        "sqlite3",
        "smtplib",
        "socket",
        "http",
        "urllib",
        "requests",
        "alpaca",
        "anthropic",
        "subprocess",
    }
)

# Process-spawning APIs reachable without importing ``subprocess`` (stdlib modules that
# are otherwise allowed). import-linter only sees top-level external modules, so these
# are enforced by the AST walk alone.
FORBIDDEN_SUBMODULES = frozenset({"asyncio.subprocess"})
PROCESS_SPAWN_NAMES: dict[str, frozenset[str]] = {
    "asyncio": frozenset({"subprocess", "create_subprocess_exec", "create_subprocess_shell"}),
    "os": frozenset(
        {
            "system",
            "popen",
            "startfile",
            "posix_spawn",
            "posix_spawnp",
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
            "execl",
            "execle",
            "execlp",
            "execlpe",
            "execv",
            "execve",
            "execvp",
            "execvpe",
        }
    ),
}

# Pure libraries allowed in domain/ and application/ (sec. 8.3.1: stdlib, pydantic and
# pure numeric libraries).
ALLOWED_THIRD_PARTY = frozenset(
    {"pydantic", "pydantic_core", "annotated_types", "typing_extensions", "numpy"}
)

LAYER_RULES: dict[str, frozenset[str]] = {
    # layer -> forbidden top-level modules
    "domain": frozenset({"application", "adapters", "app", "backtest", "shadow"})
    | INFRASTRUCTURE_MODULES,
    "application": frozenset({"adapters", "app", "backtest", "shadow"}) | INFRASTRUCTURE_MODULES,
}

WALL_CLOCK_ATTRIBUTES = frozenset({"now", "utcnow", "today"})
WALL_CLOCK_OWNERS = frozenset({"datetime", "date"})
WALL_CLOCK_TIME_FUNCTIONS = frozenset({"time", "time_ns", "localtime", "gmtime", "ctime"})


@dataclass(frozen=True)
class Violation:
    """One forbidden dependency found in a source file."""

    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.detail}"


def _imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(line, absolute module name)`` for every absolute import in ``tree``."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            # Relative imports (level > 0) cannot leave their top-level package.
            found.append((node.lineno, node.module))
    return found


def check_imports(source: str, layer: str, path: str = "<string>") -> list[Violation]:
    """Return the import violations of ``source`` for the given layer."""
    forbidden = LAYER_RULES[layer]
    violations: list[Violation] = []
    for line, module in _imported_modules(ast.parse(source)):
        top = module.split(".")[0]
        if top in forbidden or any(
            module == name or module.startswith(f"{name}.") for name in FORBIDDEN_SUBMODULES
        ):
            violations.append(Violation(path, line, f"{layer} must not import {module!r}"))
        elif (
            top not in INTERNAL_PACKAGES
            and top not in sys.stdlib_module_names
            and top not in ALLOWED_THIRD_PARTY
            and top != "__future__"
        ):
            violations.append(
                Violation(path, line, f"{layer} imports non-allowlisted package {module!r}")
            )
    return violations


def _terminal_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def check_wall_clock(source: str, path: str = "<string>") -> list[Violation]:
    """Return reads of the system wall clock (``datetime.now()``, ``time.time()``...)."""
    violations: list[Violation] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            owner = _terminal_name(node.value)
            is_datetime_read = node.attr in WALL_CLOCK_ATTRIBUTES and owner in WALL_CLOCK_OWNERS
            is_time_read = owner == "time" and node.attr in WALL_CLOCK_TIME_FUNCTIONS
            if is_datetime_read or is_time_read:
                detail = f"reads wall clock: {owner}.{node.attr}"
                violations.append(Violation(path, node.lineno, detail))
        elif isinstance(node, ast.ImportFrom) and node.module == "time":
            for alias in node.names:
                if alias.name in WALL_CLOCK_TIME_FUNCTIONS:
                    violations.append(
                        Violation(path, node.lineno, f"imports wall clock: time.{alias.name}")
                    )
    return violations


def check_process_spawn(source: str, path: str = "<string>") -> list[Violation]:
    """Return uses of process-spawning APIs (``os.system``, ``asyncio.create_subprocess_*``).

    ``import subprocess`` itself is reported by :func:`check_imports`.
    """
    violations: list[Violation] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            owner = _terminal_name(node.value)
            if owner in PROCESS_SPAWN_NAMES and node.attr in PROCESS_SPAWN_NAMES[owner]:
                detail = f"spawns a process: {owner}.{node.attr}"
                violations.append(Violation(path, node.lineno, detail))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = PROCESS_SPAWN_NAMES.get(node.module, frozenset())
            violations.extend(
                Violation(path, node.lineno, f"imports process spawning: {node.module}.{a.name}")
                for a in node.names
                if a.name in names
            )
    return violations


def _python_files(layer: str) -> list[Path]:
    return sorted((PROJECT_ROOT / layer).rglob("*.py"))


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


# --------------------------------------------------------------------------- real tree


@pytest.mark.parametrize("layer", sorted(LAYER_RULES))
def test_layer_has_no_forbidden_imports(layer: str) -> None:
    files = _python_files(layer)
    assert files, f"no python files found under {layer}/"
    violations = [
        violation
        for path in files
        for violation in check_imports(path.read_text(encoding="utf-8"), layer, _relative(path))
    ]
    assert not violations, "\n".join(str(v) for v in violations)


@pytest.mark.parametrize("layer", sorted(LAYER_RULES))
def test_layer_never_reads_wall_clock(layer: str) -> None:
    violations = [
        violation
        for path in _python_files(layer)
        for violation in check_wall_clock(path.read_text(encoding="utf-8"), _relative(path))
    ]
    assert not violations, "\n".join(str(v) for v in violations)


@pytest.mark.parametrize("layer", sorted(LAYER_RULES))
def test_layer_never_spawns_processes(layer: str) -> None:
    violations = [
        violation
        for path in _python_files(layer)
        for violation in check_process_spawn(path.read_text(encoding="utf-8"), _relative(path))
    ]
    assert not violations, "\n".join(str(v) for v in violations)


def test_import_linter_contracts() -> None:
    executable = shutil.which("lint-imports", path=str(Path(sys.executable).parent))
    executable = executable or shutil.which("lint-imports")
    if executable is None:
        pytest.skip("import-linter is not installed")
    result = subprocess.run(
        [executable],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- checker self-tests


@pytest.mark.parametrize(
    ("layer", "source"),
    [
        ("domain", "import sqlite3"),
        ("domain", "from adapters.alpaca import broker"),
        ("domain", "import application.market_flow"),
        ("domain", "from app.container import build"),
        ("domain", "import urllib.request"),
        ("domain", "from http import client"),
        ("domain", "import anthropic"),
        ("domain", "from alpaca.trading.client import TradingClient"),
        ("domain", "import yaml"),
        ("domain", "import subprocess"),
        ("domain", "from subprocess import run"),
        ("domain", "import asyncio.subprocess"),
        ("domain", "from asyncio.subprocess import PIPE"),
        ("application", "import subprocess"),
        ("application", "import asyncio.subprocess"),
        ("application", "from adapters.sqlite.unit_of_work import SqliteUnitOfWork"),
        ("application", "import smtplib"),
        ("application", "import requests"),
        ("application", "from backtest.runner import run"),
    ],
)
def test_checker_flags_forbidden_imports(layer: str, source: str) -> None:
    assert check_imports(source, layer)


@pytest.mark.parametrize(
    ("layer", "source"),
    [
        ("domain", "from decimal import Decimal"),
        ("domain", "from pydantic import BaseModel"),
        ("domain", "from domain.models import Bar"),
        ("domain", "from . import models"),
        ("domain", "from __future__ import annotations"),
        ("application", "from domain.ports import IBroker"),
        ("application", "import asyncio"),
    ],
)
def test_checker_allows_permitted_imports(layer: str, source: str) -> None:
    assert not check_imports(source, layer)


@pytest.mark.parametrize(
    "source",
    [
        "from datetime import datetime\nx = datetime.now()",
        "import datetime\nx = datetime.datetime.utcnow()",
        "from datetime import date\nx = date.today()",
        "import time\nx = time.time()",
        "from time import time_ns",
    ],
)
def test_checker_flags_wall_clock_reads(source: str) -> None:
    assert check_wall_clock(source)


@pytest.mark.parametrize(
    "source",
    [
        "import asyncio\nasyncio.create_subprocess_exec('claude')",
        "import asyncio\nasyncio.create_subprocess_shell('claude')",
        "import asyncio\nx = asyncio.subprocess.PIPE",
        "from asyncio import create_subprocess_exec",
        "from asyncio import subprocess",
        "import os\nos.system('claude')",
        "import os\nos.popen('claude')",
        "import os\nos.execv('claude', [])",
        "import os\nos.spawnv(0, 'claude', [])",
        "from os import system",
    ],
)
def test_checker_flags_process_spawning(source: str) -> None:
    assert check_process_spawn(source)


def test_checker_allows_non_spawning_os_and_asyncio_usage() -> None:
    source = (
        "import asyncio\nimport os\n"
        "x = os.path.join('a', 'b')\ny = asyncio.Queue()\nz = asyncio.wait_for\n"
    )
    assert not check_process_spawn(source)


def test_checker_allows_clock_free_datetime_usage() -> None:
    source = (
        "from datetime import UTC, datetime, timedelta\n"
        "x = datetime(2026, 10, 1, tzinfo=UTC) + timedelta(minutes=5)\n"
    )
    assert not check_wall_clock(source)
