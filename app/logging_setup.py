"""Structured logging: one JSON object per line (sec. 41), secrets redacted (sec. 43.1).

Convention for every module: ``logger.info("message", extra=fields(event="X", ...))``.
The keys of ``fields`` become top-level JSON keys next to ``ts`` (UTC, ISO 8601),
``level``, ``logger`` and ``msg``; ``trade_id``, ``signal_id``, ``client_order_id`` and
``order_id`` go there when they apply (sec. 41). Adapters, which cannot import ``app/``,
pass the same ``extra={"fields": {...}}`` mapping directly.

:func:`configure_logging` installs, on the root logger:

* a file handler on ``<log_dir>/trading-agent.jsonl`` rotated at UTC midnight, keeping
  ``retention.logs_days`` files (30 while that owner decision is ``null``);
* a stderr handler with the same JSON format.

Every registered secret value (Alpaca keys, SMTP password) is replaced by
``[REDACTED]`` in the message, the fields and any exception text, before and after JSON
encoding.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Final

from adapters.smtp.notifier import Redactor

__all__ = [
    "DEFAULT_LOG_RETENTION_DAYS",
    "LOG_FILE_NAME",
    "JsonLineFormatter",
    "close_logging",
    "configure_logging",
    "fields",
]

LOG_FILE_NAME: Final = "trading-agent.jsonl"
DEFAULT_LOG_RETENTION_DAYS: Final = 30
"""Rotated files kept while ``retention.logs_days`` is still ``null`` (SUGERIDO)."""
_HANDLER_MARK: Final = "_trading_agent_handler"
_RESERVED: Final = frozenset({"ts", "level", "logger", "msg", "exc_type", "exc"})


def fields(**values: Any) -> dict[str, Any]:
    """``extra=`` mapping carrying structured fields: ``extra=fields(event="X")``."""
    return {"fields": values}


def _redact_value(value: Any, redact: Redactor) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {str(key): _redact_value(item, redact) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_redact_value(item, redact) for item in value]
    return value


class JsonLineFormatter(logging.Formatter):
    """Formats a record as one JSON line with secrets redacted.

    Args:
        redactor: Secret redactor shared with the rest of the composition root.
    """

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redact = redactor

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, Mapping):
            for key, value in extra.items():
                name = str(key)
                payload[name if name not in _RESERVED else f"field_{name}"] = value
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc"] = self.formatException(record.exc_info)
        line = json.dumps(_redact_value(payload, self._redact), default=str, ensure_ascii=False)
        return self._redact(line)


def _remove_installed_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()


def configure_logging(
    log_dir: Path,
    *,
    redactor: Redactor,
    level: int | str = logging.INFO,
    retention_days: int | None = None,
    console: bool = True,
) -> Path:
    """Install the JSON file (and console) handlers; returns the log file path.

    Calling it again replaces the handlers it installed before (never duplicates).

    Raises:
        OSError: ``log_dir`` cannot be created.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / LOG_FILE_NAME
    root = logging.getLogger()
    _remove_installed_handlers(root)
    formatter = JsonLineFormatter(redactor)
    file_handler = TimedRotatingFileHandler(
        path,
        when="midnight",
        utc=True,
        backupCount=retention_days or DEFAULT_LOG_RETENTION_DAYS,
        encoding="utf-8",
    )
    handlers: list[logging.Handler] = [file_handler]
    if console:
        handlers.append(logging.StreamHandler(sys.stderr))
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARK, True)
        root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("urllib3", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


def close_logging() -> None:
    """Flush, close and remove the handlers installed by :func:`configure_logging`."""
    _remove_installed_handlers(logging.getLogger())
