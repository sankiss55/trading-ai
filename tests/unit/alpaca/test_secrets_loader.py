"""app.secrets.load_secrets (sec. 7.2, 43.1). Fake values in tmp_path .env files only."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.secrets import AppEnv, SecretsError, load_secrets
from tests.unit.alpaca.fakes import FAKE_KEY, FAKE_SECRET


def _env_file(tmp_path: Path, **values: str) -> Path:
    path = tmp_path / ".env"
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    return path


def _valid(tmp_path: Path, app_env: str = "paper") -> Path:
    return _env_file(
        tmp_path, APP_ENV=app_env, ALPACA_API_KEY=FAKE_KEY, ALPACA_SECRET_KEY=FAKE_SECRET
    )


@pytest.mark.parametrize("app_env", ["dev", "test", "paper", "PAPER"])
def test_non_live_environments_load_and_force_paper(tmp_path: Path, app_env: str) -> None:
    secrets = load_secrets(_valid(tmp_path, app_env), environ={})
    assert secrets.app_env is AppEnv(app_env.lower())
    assert secrets.alpaca_paper is True
    assert isinstance(secrets.alpaca_api_key, SecretStr)
    assert secrets.alpaca_api_key.get_secret_value() == FAKE_KEY
    assert secrets.alpaca_secret_key.get_secret_value() == FAKE_SECRET


def test_secret_values_are_masked(tmp_path: Path) -> None:
    secrets = load_secrets(_valid(tmp_path), environ={})
    credentials = secrets.alpaca_credentials()
    for text in (repr(secrets), str(secrets), repr(credentials), secrets.model_dump_json()):
        assert FAKE_KEY not in text
        assert FAKE_SECRET not in text


def test_live_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SecretsError) as info:
        load_secrets(_valid(tmp_path, "live"), environ={})
    assert info.value.code == "LIVE_BLOCKED"


@pytest.mark.parametrize("missing", ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"])
def test_missing_key_names_the_variable_never_a_value(tmp_path: Path, missing: str) -> None:
    values = {"APP_ENV": "paper", "ALPACA_API_KEY": FAKE_KEY, "ALPACA_SECRET_KEY": FAKE_SECRET}
    values[missing] = "  "  # present but empty counts as missing
    with pytest.raises(SecretsError) as info:
        load_secrets(_env_file(tmp_path, **values), environ={})
    assert info.value.code == "MISSING_SECRET"
    assert missing in str(info.value)
    assert FAKE_KEY not in str(info.value)
    assert FAKE_SECRET not in str(info.value)


def test_missing_or_invalid_app_env(tmp_path: Path) -> None:
    path = _env_file(tmp_path, ALPACA_API_KEY=FAKE_KEY, ALPACA_SECRET_KEY=FAKE_SECRET)
    with pytest.raises(SecretsError, match="APP_ENV") as info:
        load_secrets(path, environ={})
    assert info.value.code == "MISSING_SECRET"
    with pytest.raises(SecretsError) as info:
        load_secrets(path, environ={"APP_ENV": "production"})
    assert info.value.code == "INVALID_APP_ENV"


def test_process_environment_overrides_file(tmp_path: Path) -> None:
    other = "PKOTHERFAKEKEY11111"
    secrets = load_secrets(_valid(tmp_path), environ={"ALPACA_API_KEY": other})
    assert secrets.alpaca_api_key.get_secret_value() == other


def test_missing_file_reads_environment_only(tmp_path: Path) -> None:
    environ = {"APP_ENV": "dev", "ALPACA_API_KEY": FAKE_KEY, "ALPACA_SECRET_KEY": FAKE_SECRET}
    secrets = load_secrets(tmp_path / "absent.env", environ=environ)
    assert secrets.app_env is AppEnv.DEV
    with pytest.raises(SecretsError):
        load_secrets(tmp_path / "absent.env", environ={})


def test_secrets_model_is_frozen(tmp_path: Path) -> None:
    secrets = load_secrets(_valid(tmp_path), environ={})
    with pytest.raises(ValidationError):
        secrets.app_env = AppEnv.LIVE  # type: ignore[misc]
