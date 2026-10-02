"""CLI: ``python -m backtest --config <yaml> --data <dir> --starting-cash <amount> [--out f]``.

Prints the text summary; ``--out`` also writes the full JSON report. Exit codes:
``0`` report produced, ``2`` refused (pending OWNER_DECISIONs) or invalid input.
The starting capital is not part of config.yaml (sec. 7.3), so it must be passed
explicitly; there is no default.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.config import ConfigError, load_config
from backtest.report import render_text
from backtest.runner import BacktestRefusedError, run_backtest
from domain.errors import NonRetryableError


def _decimal(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal {text!r}") from exc
    if not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError(f"expected a finite non-negative amount, got {text!r}")
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
    args = parser.parse_args(argv)
    try:
        loaded = load_config(args.config)
        report = run_backtest(
            loaded,
            args.data,
            starting_cash=args.starting_cash,
            commission_per_fill=args.commission_per_fill,
        )
    except BacktestRefusedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ConfigError, NonRetryableError) as exc:
        print(f"backtest failed: {exc}", file=sys.stderr)
        return 2
    print(render_text(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
