"""Translation of alpaca-py / HTTP errors into domain errors (sec. 8.3.7, 29).

Classification:

* HTTP 429 (rate limit) and 5xx: ``RetryableError``.
* Network failures (connection error, timeout, broken stream): ``RetryableError``.
* HTTP 401/403: ``NonRetryableError`` code ``ALPACA_AUTH``.
* Any other HTTP 4xx: ``NonRetryableError`` code ``ALPACA_REQUEST_REJECTED``.
* Unparseable / schema-invalid responses: ``NonRetryableError`` code ``INVALID_SCHEMA``.

Messages never contain credentials (sec. 43.1): the API key travels in request headers,
which are never read here, and the Alpaca error body only carries ``code``/``message``.
"""

from __future__ import annotations

import json

import requests
from alpaca.common.exceptions import APIError
from pydantic import ValidationError

from domain.errors import DomainError, NonRetryableError, RetryableError

__all__ = ["SDK_ERRORS", "translate_error"]

SDK_ERRORS: tuple[type[Exception], ...] = (
    APIError,
    requests.RequestException,
    ValidationError,
)
"""Exceptions raised by alpaca-py calls that the adapters translate."""

_MAX_DETAIL = 200
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_SERVER_ERROR = 500
_HTTP_AUTH = frozenset({401, 403})


def _api_message(exc: APIError) -> str:
    """The ``message`` field of an Alpaca error body, truncated; empty if unparseable."""
    try:
        body = json.loads(str(exc))
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    message = body.get("message")
    return str(message)[:_MAX_DETAIL] if message is not None else ""


def _status_code(exc: APIError) -> int | None:
    http_error = getattr(exc, "_http_error", None)
    response = getattr(http_error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def translate_error(exc: Exception, *, operation: str) -> DomainError:
    """Map an SDK/network exception to ``RetryableError`` or ``NonRetryableError``.

    Args:
        exc: Exception raised by an alpaca-py call (see ``SDK_ERRORS``).
        operation: Short description of the call, e.g. ``"get_stock_bars SPY 1Min"``.
    """
    if isinstance(exc, APIError):
        status = _status_code(exc)
        detail = _api_message(exc)
        suffix = f": {detail}" if detail else ""
        where = f"{operation}: HTTP {status if status is not None else '?'}{suffix}"
        if status == _HTTP_TOO_MANY_REQUESTS:
            return RetryableError(where, code="RATE_LIMITED")
        if status is not None and status >= _HTTP_SERVER_ERROR:
            return RetryableError(where, code="ALPACA_SERVER_ERROR")
        if status in _HTTP_AUTH:
            return NonRetryableError(where, code="ALPACA_AUTH")
        return NonRetryableError(where, code="ALPACA_REQUEST_REJECTED")
    if isinstance(
        exc,
        requests.ConnectionError | requests.Timeout | requests.exceptions.ChunkedEncodingError,
    ):
        # The exception text may contain the request URL; only the type is reported.
        return RetryableError(f"{operation}: {type(exc).__name__}", code="NETWORK_ERROR")
    if isinstance(exc, requests.RequestException):
        return NonRetryableError(f"{operation}: {type(exc).__name__}", code="INVALID_SCHEMA")
    if isinstance(exc, ValidationError):
        return NonRetryableError(
            f"{operation}: response failed SDK validation ({exc.error_count()} errors)",
            code="INVALID_SCHEMA",
        )
    return NonRetryableError(f"{operation}: {type(exc).__name__}", code="ALPACA_UNEXPECTED")
