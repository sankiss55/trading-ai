"""Secrets loading for the composition root (sec. 7.1, 7.2, 43.1).

Only ``app/`` and the CLIs that act as composition roots call this module.

* Values are read with ``python-dotenv`` (``dotenv_values``, which does NOT mutate
  ``os.environ``) from the project ``.env`` (path injectable). Variables already set
  in the process environment take precedence over the file, like ``load_dotenv``.
* ``APP_ENV`` must be one of ``dev | test | paper | live``. ``live`` is refused in the
  MVP (sec. 7.2, until the sec. 48 approval record exists); every other value forces
  the Alpaca clients into paper mode.
* Alpaca keys are ``pydantic.SecretStr``: masked in ``repr``/``str`` and logs. Error
  messages name the missing variable, never a value (sec. 43.1).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr

from adapters.alpaca import AlpacaCredentials
from domain.errors import NonRetryableError

__all__ = ["DEFAULT_ENV_FILE", "AppEnv", "Secrets", "SecretsError", "load_secrets"]

DEFAULT_ENV_FILE = Path(".env")


class SecretsError(NonRetryableError):
    """Invalid or missing secrets/environment selection. Never carries a secret value."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message, code=code)


class AppEnv(StrEnum):
    """Execution environment (sec. 58.4)."""

    DEV = "dev"
    TEST = "test"
    PAPER = "paper"
    LIVE = "live"


class Secrets(BaseModel):
    """Loaded secrets. Frozen; ``repr`` masks every key.

    Attributes:
        app_env: Environment selection (never ``live`` in the MVP).
        alpaca_api_key: ``ALPACA_API_KEY``.
        alpaca_secret_key: ``ALPACA_SECRET_KEY``.
        alpaca_paper: Always ``True`` (sec. 7.2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_env: AppEnv
    alpaca_api_key: SecretStr
    alpaca_secret_key: SecretStr
    alpaca_paper: Literal[True] = True

    def alpaca_credentials(self) -> AlpacaCredentials:
        """Credentials for the Alpaca adapters (paper)."""
        return AlpacaCredentials(api_key=self.alpaca_api_key, secret_key=self.alpaca_secret_key)


def _read(values: Mapping[str, str | None], environ: Mapping[str, str], name: str) -> str | None:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        raw = values.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def load_secrets(
    env_file: Path | None = DEFAULT_ENV_FILE,
    *,
    environ: Mapping[str, str] | None = None,
) -> Secrets:
    """Load and validate the secrets.

    Args:
        env_file: ``.env`` file to read (``None`` reads only ``environ``). A missing
            file is treated as empty; the variables may come from the environment.
        environ: Process environment (defaults to ``os.environ``; tests pass ``{}``).

    Raises:
        SecretsError: code ``MISSING_SECRET`` (names the variable only),
            ``INVALID_APP_ENV`` or ``LIVE_BLOCKED``.
    """
    env = os.environ if environ is None else environ
    values: Mapping[str, str | None] = {}
    if env_file is not None and env_file.is_file():
        values = dotenv_values(env_file, encoding="utf-8")
    source = f" (env file: {env_file})" if env_file is not None else ""

    app_env_text = _read(values, env, "APP_ENV")
    if app_env_text is None:
        raise SecretsError(f"APP_ENV is missing or empty{source}", code="MISSING_SECRET")
    try:
        app_env = AppEnv(app_env_text.lower())
    except ValueError as exc:
        allowed = " | ".join(member.value for member in AppEnv)
        raise SecretsError(
            f"APP_ENV must be one of {allowed}{source}", code="INVALID_APP_ENV"
        ) from exc
    if app_env is AppEnv.LIVE:
        raise SecretsError(
            "APP_ENV=live is blocked in the MVP (sec. 7.2, 48): refusing to start",
            code="LIVE_BLOCKED",
        )

    keys: dict[str, str] = {}
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        value = _read(values, env, name)
        if value is None:
            raise SecretsError(f"{name} is missing or empty{source}", code="MISSING_SECRET")
        keys[name] = value
    return Secrets(
        app_env=app_env,
        alpaca_api_key=SecretStr(keys["ALPACA_API_KEY"]),
        alpaca_secret_key=SecretStr(keys["ALPACA_SECRET_KEY"]),
    )
