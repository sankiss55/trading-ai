"""SMTP secrets loading (sec. 7.2) and JSON logging with redaction (sec. 41, 43.1)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from adapters.smtp.notifier import REDACTED, Redactor
from app.logging_setup import LOG_FILE_NAME, close_logging, configure_logging, fields
from app.notifier_secrets import load_notifier_secrets, read_env_value

FAKE_PASSWORD = "fake app password"


def _env_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "test.env"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_all_three_from_the_env_file(tmp_path: Path) -> None:
    env_file = _env_file(
        tmp_path,
        "GMAIL_USER=bot@example.com\n"
        f"GMAIL_APP_PASSWORD={FAKE_PASSWORD}\n"
        "NOTIFICATION_EMAIL=owner@example.com\n",
    )
    secrets, problems = load_notifier_secrets(env_file, environ={})
    assert problems == ()
    assert secrets is not None
    assert secrets.gmail_user == "bot@example.com"
    assert secrets.gmail_app_password.get_secret_value() == FAKE_PASSWORD
    assert secrets.notification_email == "owner@example.com"
    assert FAKE_PASSWORD not in repr(secrets)


def test_environment_wins_over_the_file(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "NOTIFICATION_EMAIL=file@example.com\n")
    environ = {"NOTIFICATION_EMAIL": "env@example.com"}
    assert read_env_value("NOTIFICATION_EMAIL", env_file, environ=environ) == "env@example.com"
    assert read_env_value("NOTIFICATION_EMAIL", env_file, environ={"NOTIFICATION_EMAIL": " "}) == (
        "file@example.com"
    )
    assert read_env_value("MISSING", env_file, environ={}) is None
    assert read_env_value("NOTIFICATION_EMAIL", None, environ={}) is None


def test_missing_and_invalid_values_are_named_without_values(tmp_path: Path) -> None:
    env_file = _env_file(
        tmp_path, f"GMAIL_USER=not-an-address\nGMAIL_APP_PASSWORD={FAKE_PASSWORD}\n"
    )
    secrets, problems = load_notifier_secrets(env_file, environ={})
    assert secrets is None
    assert problems == (
        "NOTIFICATION_EMAIL is missing or empty",
        "GMAIL_USER is not an email address",
    )
    assert all(FAKE_PASSWORD not in problem for problem in problems)


def test_missing_env_file_is_empty(tmp_path: Path) -> None:
    secrets, problems = load_notifier_secrets(tmp_path / "absent.env", environ={})
    assert secrets is None
    assert len(problems) == 3


@pytest.fixture
def log_dir(tmp_path: Path) -> Iterator[Path]:
    directory = tmp_path / "logs"
    yield directory
    close_logging()


def _lines(log_dir: Path) -> list[dict[str, object]]:
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = (log_dir / LOG_FILE_NAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_json_lines_with_fields(log_dir: Path) -> None:
    path = configure_logging(log_dir, redactor=Redactor(), console=False, retention_days=7)
    assert path == log_dir / LOG_FILE_NAME
    logging.getLogger("app.test").info(
        "order seen", extra=fields(event="X", trade_id="T1", client_order_id="C1", msg="clash")
    )
    (line,) = _lines(log_dir)
    assert line["level"] == "INFO"
    assert line["logger"] == "app.test"
    assert line["msg"] == "order seen"
    assert line["event"] == "X"
    assert line["trade_id"] == "T1"
    assert line["client_order_id"] == "C1"
    assert line["field_msg"] == "clash"
    assert str(line["ts"]).endswith("+00:00")


def test_secrets_are_redacted_in_messages_fields_and_exceptions(log_dir: Path) -> None:
    secret = 'PK"SECRET\\value123'
    configure_logging(log_dir, redactor=Redactor([secret]), console=False)
    logger = logging.getLogger("app.test")
    logger.warning(f"msg {secret}", extra=fields(nested={"k": [secret]}))
    try:
        raise RuntimeError(f"boom {secret}")
    except RuntimeError:
        logger.exception("failed")
    raw = (log_dir / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "SECRET" not in raw
    first, second = _lines(log_dir)
    assert first["msg"] == f"msg {REDACTED}"
    assert first["nested"] == {"k": [REDACTED]}
    assert second["exc_type"] == "RuntimeError"
    assert REDACTED in str(second["exc"])


def test_reconfiguring_never_duplicates_handlers(log_dir: Path) -> None:
    configure_logging(log_dir, redactor=Redactor(), console=False)
    configure_logging(log_dir, redactor=Redactor(), console=False)
    logging.getLogger("app.test").info("once")
    assert len(_lines(log_dir)) == 1
