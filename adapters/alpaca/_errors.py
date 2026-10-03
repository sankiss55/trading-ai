"""Translation of alpaca-py / HTTP errors into domain errors (sec. 8.3.7, 29).

Classification:

* HTTP 429 (rate limit) and 5xx: ``RetryableError``.
* Network failures (connection error, timeout, broken stream): ``RetryableError``.
* HTTP 401/403: ``NonRetryableError`` code ``ALPACA_AUTH``.
* Any other HTTP 4xx: ``NonRetryableError`` code ``ALPACA_REQUEST_REJECTED``.
* Unparseable / schema-invalid responses: ``NonRetryableError`` code ``INVALID_SCHEMA``.

Messages never contain credentials (sec. 43.1): the API key travels in request headers,
which are never read here, and the Alpaca error body only carries ``code``/``message``.

Order endpoints (``translate_submit_error`` / ``translate_lookup_error`` /
``translate_cancel_error``) refine that classification with the Alpaca order semantics
(verified 2026-10-02 against docs.alpaca.markets ``postorder``, "Placing Orders" and the
"30 common Trading API errors" guide, sec. 60):

* ``POST /v2/orders`` answers HTTP 403 "Buying power or shares is not sufficient" (body
  code 40310000: ``insufficient buying power`` / ``insufficient qty available for order``)
  and HTTP 422 for invalid input (code 42210000, e.g. ``sub-penny increment does not
  fulfill minimum pricing criteria``; code 40010001 ``client_order_id must be unique``).
  The status of the duplicate reply is matched on its message (409 is accepted too).
* ``GET /v2/orders:by_client_order_id`` and ``DELETE /v2/orders/{id}`` answer 404 for an
  unknown order; ``DELETE`` answers 422 when the order is no longer cancelable.

Stable codes produced: ``DUPLICATE_CLIENT_ORDER_ID``, ``INSUFFICIENT_BUYING_POWER``,
``INSUFFICIENT_QTY_AVAILABLE``, ``INVALID_PRICE_INCREMENT``, ``INVALID_BRACKET_PRICES``,
``POTENTIAL_WASH_TRADE``, ``ORDER_REJECTED`` (any other 403/409/422 of a submission),
``ORDER_NOT_FOUND`` and ``ORDER_NOT_CANCELABLE``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import requests
from alpaca.common.exceptions import APIError
from pydantic import ValidationError

from domain.errors import DomainError, NonRetryableError, RetryableError

__all__ = [
    "SDK_ERRORS",
    "ApiErrorDetail",
    "api_error_detail",
    "translate_cancel_error",
    "translate_error",
    "translate_lookup_error",
    "translate_submit_error",
]

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


# --------------------------------------------------------------------------- order endpoints

_HTTP_FORBIDDEN = 403
_HTTP_NOT_FOUND = 404
_HTTP_CONFLICT = 409
_HTTP_UNPROCESSABLE = 422
_ORDER_REJECTION_STATUSES = frozenset({_HTTP_FORBIDDEN, _HTTP_CONFLICT, _HTTP_UNPROCESSABLE})

_ORDER_REJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("client_order_id must be unique", "DUPLICATE_CLIENT_ORDER_ID"),
    ("insufficient buying power", "INSUFFICIENT_BUYING_POWER"),
    ("insufficient qty", "INSUFFICIENT_QTY_AVAILABLE"),
    ("sub-penny", "INVALID_PRICE_INCREMENT"),
    ("minimum pricing criteria", "INVALID_PRICE_INCREMENT"),
    ("take_profit", "INVALID_BRACKET_PRICES"),
    ("stop_loss", "INVALID_BRACKET_PRICES"),
    ("wash trade", "POTENTIAL_WASH_TRADE"),
)
"""Case-insensitive substrings of the Alpaca ``message`` and the domain code they map to.
Only the duplicate and buying-power/qty messages are documented verbatim; the others
are matched loosely (VERIFICAR the exact texts against the paper account, sec. 49.2)."""


@dataclass(frozen=True, slots=True)
class ApiErrorDetail:
    """The parts of an ``APIError`` that are safe to inspect (no request data)."""

    status: int | None
    code: int | None
    message: str


def api_error_detail(exc: APIError) -> ApiErrorDetail:
    """HTTP status, Alpaca body ``code`` and truncated ``message`` of ``exc``."""
    code: int | None = None
    try:
        body = json.loads(str(exc))
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("code"), int):
        code = body["code"]
    return ApiErrorDetail(status=_status_code(exc), code=code, message=_api_message(exc))


def _where(operation: str, detail: ApiErrorDetail) -> str:
    status = detail.status if detail.status is not None else "?"
    suffix = f": {detail.message}" if detail.message else ""
    return f"{operation}: HTTP {status}{suffix}"


def translate_submit_error(exc: Exception, *, operation: str) -> DomainError:
    """Translation for ``POST /v2/orders``: order rejections get specific stable codes.

    A 403 whose message is empty or ``forbidden`` stays ``ALPACA_AUTH``; 401, 429, 5xx
    and network errors follow :func:`translate_error`.
    """
    if isinstance(exc, APIError):
        detail = api_error_detail(exc)
        if detail.status in _ORDER_REJECTION_STATUSES:
            lowered = detail.message.lower()
            for needle, code in _ORDER_REJECTION_PATTERNS:
                if needle in lowered:
                    return NonRetryableError(_where(operation, detail), code=code)
            if detail.status == _HTTP_FORBIDDEN and lowered.strip(" .") in ("", "forbidden"):
                return NonRetryableError(_where(operation, detail), code="ALPACA_AUTH")
            return NonRetryableError(_where(operation, detail), code="ORDER_REJECTED")
    return translate_error(exc, operation=operation)


def translate_lookup_error(exc: Exception, *, operation: str) -> DomainError:
    """Translation for single-order reads: HTTP 404 becomes ``ORDER_NOT_FOUND``."""
    if isinstance(exc, APIError):
        detail = api_error_detail(exc)
        if detail.status == _HTTP_NOT_FOUND:
            return NonRetryableError(_where(operation, detail), code="ORDER_NOT_FOUND")
    return translate_error(exc, operation=operation)


def translate_cancel_error(exc: Exception, *, operation: str) -> DomainError:
    """Translation for ``DELETE /v2/orders/{id}``: 404 ``ORDER_NOT_FOUND``, 422
    ``ORDER_NOT_CANCELABLE``."""
    if isinstance(exc, APIError):
        detail = api_error_detail(exc)
        if detail.status == _HTTP_NOT_FOUND:
            return NonRetryableError(_where(operation, detail), code="ORDER_NOT_FOUND")
        if detail.status == _HTTP_UNPROCESSABLE:
            return NonRetryableError(_where(operation, detail), code="ORDER_NOT_CANCELABLE")
    return translate_error(exc, operation=operation)
