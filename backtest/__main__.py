"""CLI: ``python -m backtest --config <yaml> --data <dir> --starting-cash <amount> [--out f]``.

Prints the text summary; ``--out`` also writes the full JSON report. ``--workers N``
runs the base and slippage-sensitivity simulations in up to N processes (default: one
per scenario, at most the CPU count; ``1`` = sequential); the report is identical for
any N.

``--mode official`` (default) is the sec. 45.4 run: one continuous account over the
whole period. ``--mode segmented`` is the RESEARCH mode (:mod:`backtest.research`):
every segment (walk-forward test windows, in/out-of-sample, calendar years) is an
independent fresh account, so halts reset per segment; it is NOT the official run.
Exit codes:
``0`` report produced, ``2`` refused (pending OWNER_DECISIONs) or invalid input.
The starting capital is not part of config.yaml (sec. 7.3), so it must be passed
explicitly; there is no default.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.config import ConfigError, load_config
from backtest.report import render_text
from backtest.research import render_research_text, run_research
from backtest.runner import BacktestRefusedError, default_workers, run_backtest
from domain.errors import NonRetryableError


def _decimal(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal {text!r}") from exc
    if not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError(f"expected a finite non-negative amount, got {text!r}")
    return value


def _workers(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid worker count {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected at least 1 worker, got {text!r}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the backtest CLI and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m backtest",
        description="Backtest of the deterministic strategy (sec. 45).",
    )
    parser.add_argument("--config", type=Path, required=True, help="config.yaml to use")
    parser.add_argument("--data", type=Path, required=True, help="dataset directory")
    parser.add_argument(
        "--starting-cash", type=_decimal, required=True, help="initial capital (no default)"
    )
    parser.add_argument(
        "--commission-per-fill",
        type=_decimal,
        default=Decimal(0),
        help="flat commission per fill (default 0; VERIFICAR broker fees)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    parser.add_argument(
        "--workers",
        type=_workers,
        default=default_workers(),
        help="processes for the simulations (default: %(default)s; 1 = sequential; "
        "the report does not depend on it)",
    )
    parser.add_argument(
        "--mode",
        choices=("official", "segmented"),
        default="official",
        help="official: the sec. 45.4 continuous run (default); segmented: RESEARCH mode, "
        "a fresh account per segment (halts reset per segment), NOT the official run",
    )
    args = parser.parse_args(argv)
    try:
        loaded = load_config(args.config)
        if args.mode == "segmented":
            research = run_research(
                loaded,
                args.data,
                starting_cash=args.starting_cash,
                commission_per_fill=args.commission_per_fill,
                workers=args.workers,
            )
            text, payload = render_research_text(research), research.model_dump_json(indent=2)
        else:
            report = run_backtest(
                loaded,
                args.data,
                starting_cash=args.starting_cash,
                commission_per_fill=args.commission_per_fill,
                workers=args.workers,
            )
            text, payload = render_text(report), report.model_dump_json(indent=2)
    except BacktestRefusedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ConfigError, NonRetryableError) as exc:
        print(f"backtest failed: {exc}", file=sys.stderr)
        return 2
    if args.mode == "segmented" and isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")  # the research banner is not ASCII
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"\nJSON report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
