"""Credentials and HTTP timeouts for the alpaca-py REST clients.

alpaca-py (verified on 0.44.0) sends every request through ``RESTClient._session`` (a
``requests.Session``) **without a timeout**, so a stalled connection could block a
worker thread forever. :func:`install_timeouts` mounts an ``HTTPAdapter`` that applies
a default ``(connect, read)`` timeout to every request of that session (sec. 29).
It relies on the private ``_session`` attribute: if a future SDK version removes it,
the adapter refuses to start (``SDK_INCOMPATIBLE``) instead of running without timeouts.

alpaca-py also re-sends **any** request, ``POST /v2/orders`` included, when the response is
HTTP 429 or 504 (``RESTClient._request``: ``DEFAULT_RETRY_ATTEMPTS = 3``,
``DEFAULT_RETRY_WAIT_SECONDS = 3``, ``DEFAULT_RETRY_EXCEPTION_CODES = [429, 504]``,
verified on 0.44.0). A 504 does not prove the order was not created, so the trading
client used for orders disables that layer (:func:`disable_sdk_retries`, private
``_retry`` attribute, same fail-closed policy) and every retry decision is taken by the
adapter, which looks the order up by ``client_order_id`` first (sec. 22, 29).

:func:`require_paper_environment` is the adapter-side guard of sec. 7.2: ``APP_ENV=live``
is refused before any client exists; every other value gets a ``paper=True`` client.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import requests
from pydantic import SecretStr
from requests.adapters import HTTPAdapter

from domain.errors import NonRetryableError

__all__ = [
    "AlpacaCredentials",
    "HttpTimeouts",
    "disable_sdk_retries",
    "install_timeouts",
    "require_paper_environment",
]


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    """Alpaca API credentials. ``SecretStr`` keeps them out of ``repr`` and logs.

    There is no ``paper`` flag: the trading client is always built with
    ``paper=True`` in the MVP (sec. 7.2); the composition root refuses ``APP_ENV=live``.
    """

    api_key: SecretStr
    secret_key: SecretStr


@dataclass(frozen=True, slots=True)
class HttpTimeouts:
    """Per-request timeouts in seconds (SUGERIDO technical defaults, sec. 29)."""

    connect_seconds: float = 10.0
    read_seconds: float = 60.0


class _TimeoutAdapter(HTTPAdapter):
    def __init__(self, timeouts: HttpTimeouts) -> None:
        super().__init__()
        self._timeout = (timeouts.connect_seconds, timeouts.read_seconds)

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: float | tuple[float, float] | tuple[float, None] | None = None,
        verify: bool | str = True,
        cert: bytes | str | tuple[bytes | str, bytes | str] | None = None,
        proxies: Mapping[str, str] | None = None,
    ) -> requests.Response:
        return super().send(
            request,
            stream=stream,
            timeout=self._timeout if timeout is None else timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )


def install_timeouts(client: object, timeouts: HttpTimeouts) -> None:
    """Make every request of an alpaca-py REST ``client`` use ``timeouts``.

    Raises:
        NonRetryableError: code ``SDK_INCOMPATIBLE`` if the client has no session.
    """
    session = getattr(client, "_session", None)
    if not isinstance(session, requests.Session):
        raise NonRetryableError(
            f"{type(client).__name__} has no requests session; verify the alpaca-py version",
            code="SDK_INCOMPATIBLE",
        )
    adapter = _TimeoutAdapter(timeouts)
    session.mount("https://", adapter)
    session.mount("http://", adapter)


def disable_sdk_retries(client: object) -> None:
    """Make an alpaca-py REST ``client`` send every request exactly once.

    With ``_retry = 0`` ``RESTClient._request`` performs a single ``_one_request`` and
    raises ``APIError`` for 429/504 instead of sleeping and re-sending.

    Raises:
        NonRetryableError: code ``SDK_INCOMPATIBLE`` if the client has no ``_retry``.
    """
    retry = getattr(client, "_retry", None)
    if not isinstance(retry, int) or isinstance(retry, bool):
        raise NonRetryableError(
            f"{type(client).__name__} has no request retry counter; verify the alpaca-py version",
            code="SDK_INCOMPATIBLE",
        )
    client._retry = 0  # type: ignore[attr-defined]  # checked above; private SDK attribute


def require_paper_environment(app_env: str) -> None:
    """Refuse ``APP_ENV=live`` (any case or padding) before any client is built.

    Raises:
        NonRetryableError: code ``LIVE_BLOCKED`` (sec. 7.2, 48).
    """
    if app_env.strip().lower() == "live":
        raise NonRetryableError(
            "APP_ENV=live is blocked in the MVP (sec. 7.2, 48): refusing to build a trading client",
            code="LIVE_BLOCKED",
        )
