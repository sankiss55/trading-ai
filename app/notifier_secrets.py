"""Notification secrets and single ``.env`` values for the composition root (sec. 7.2, 43.1).

``app/secrets.py`` loads ``APP_ENV`` and the Alpaca keys (both mandatory). The SMTP
secrets are optional: without them the runtime falls back to the console notifier and
logs a warning, so this module reads them separately:

* ``GMAIL_USER``: the Gmail account that sends (also the ``From`` address).
* ``GMAIL_APP_PASSWORD``: a Google App Password (requires 2-step verification), kept as
  ``SecretStr`` (masked in ``repr`` and logs).
* ``NOTIFICATION_EMAIL``: the recipient.

Precedence matches ``app/secrets.py``: a non-blank process environment variable wins over
the ``.env`` file, which is read with ``dotenv_values`` (``os.environ`` is never mutated).
Errors and warnings name variables, never values.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr

__all__ = [
    "NOTIFIER_VARIABLES",
    "NotifierSecrets",
    "load_notifier_secrets",
    "read_env_value",
]

NOTIFIER_VARIABLES: Final[tuple[str, ...]] = (
    "GMAIL_USER",
    "GMAIL_APP_PASSWORD",
    "NOTIFICATION_EMAIL",
)


class NotifierSecrets(BaseModel):
    """SMTP account and recipient. Frozen; the password is masked in ``repr``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gmail_user: str
    gmail_app_password: SecretStr
    notification_email: str


def _file_values(env_file: Path | None) -> Mapping[str, str | None]:
    if env_file is not None and env_file.is_file():
        return dotenv_values(env_file, encoding="utf-8")
    return {}


def read_env_value(
    name: str,
    env_file: Path | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Stripped value of ``name`` (process environment first, then ``env_file``).

    Returns ``None`` when the variable is missing or blank everywhere.
    """
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None or not raw.strip():
        raw = _file_values(env_file).get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _looks_like_address(value: str) -> bool:
    local, _, domain = value.partition("@")
    return bool(local) and "." in domain and not any(char.isspace() for char in value)


def load_notifier_secrets(
    env_file: Path | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[NotifierSecrets | None, tuple[str, ...]]:
    """Load the SMTP secrets.

    Returns:
        ``(secrets, ())`` when all three are present and the two addresses look valid;
        otherwise ``(None, problems)`` where ``problems`` names each missing or invalid
        variable (never its value).
    """
    values = {name: read_env_value(name, env_file, environ=environ) for name in NOTIFIER_VARIABLES}
    problems = [f"{name} is missing or empty" for name, value in values.items() if value is None]
    for name in ("GMAIL_USER", "NOTIFICATION_EMAIL"):
        value = values[name]
        if value is not None and not _looks_like_address(value):
            problems.append(f"{name} is not an email address")
    user, password, recipient = (values[name] for name in NOTIFIER_VARIABLES)
    if problems or user is None or password is None or recipient is None:
        return None, tuple(problems)
    return (
        NotifierSecrets(
            gmail_user=user,
            gmail_app_password=SecretStr(password),
            notification_email=recipient,
        ),
        (),
    )
