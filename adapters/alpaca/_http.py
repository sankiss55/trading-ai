"""Credentials and HTTP timeouts for the alpaca-py REST clients.

alpaca-py (verified on 0.44.0) sends every request through ``RESTClient._session`` (a
``requests.Session``) **without a timeout**, so a stalled connection could block a
worker thread forever. :func:`install_timeouts` mounts an ``HTTPAdapter`` that applies
a default ``(connect, read)`` timeout to every request of that session (sec. 29).
It relies on the private ``_session`` attribute: if a future SDK version removes it,
the adapter refuses to start (``SDK_INCOMPATIBLE``) instead of running without timeouts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import requests
from pydantic import SecretStr
from requests.adapters import HTTPAdapter

from domain.errors import NonRetryableError

__all__ = ["AlpacaCredentials", "HttpTimeouts", "install_timeouts"]


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
