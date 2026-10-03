"""Bounded retry with exponential backoff and a local request pacer (sec. 29).

alpaca-py clients are synchronous: each call runs in a worker thread
(``asyncio.to_thread``) so the event loop is never blocked. Sleeping happens only here,
through an injectable ``sleep`` coroutine, so tests never wait.

Note: alpaca-py itself retries HTTP 429/504 up to 3 times with a 3 s wait
(``alpaca.common.constants.DEFAULT_RETRY_*``, verified on 0.44.0) before raising
``APIError``. The effective upper bound of HTTP requests per call is therefore
``max_attempts * 4``; both layers are bounded, there is no infinite loop.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from adapters.alpaca._errors import SDK_ERRORS, translate_error
from domain.errors import DomainError, RetryableError

__all__ = ["ErrorTranslator", "RequestPacer", "RetryPolicy", "Sleep", "call_with_retry"]

T = TypeVar("T")

Sleep = Callable[[float], Awaitable[None]]
"""Async sleep function (``asyncio.sleep`` in production, a recorder in tests)."""


class ErrorTranslator(Protocol):
    """Maps an SDK/network exception to a domain error (``translate_error`` and the
    order-endpoint variants in ``adapters.alpaca._errors``)."""

    def __call__(self, exc: Exception, *, operation: str) -> DomainError:
        """Domain error for ``exc`` raised by ``operation``."""
        ...


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry parameters of one integration (SUGERIDO technical defaults, sec. 29).

    Attributes:
        max_attempts: Total attempts including the first one (>= 1).
        base_delay_seconds: Delay before the 2nd attempt; doubles on each retry.
        max_delay_seconds: Cap of a single delay (before jitter).
        jitter_ratio: Each delay is multiplied by ``1 + jitter_ratio * U[0, 1)``.
    """

    max_attempts: int = 5
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0 or self.jitter_ratio < 0:
            raise ValueError("delays and jitter_ratio must be >= 0")

    def delay(self, retry_number: int, unit_random: float) -> float:
        """Delay before retry ``retry_number`` (1 = first retry)."""
        growth = 2.0 ** (retry_number - 1)
        base = min(self.max_delay_seconds, self.base_delay_seconds * growth)
        return float(base * (1.0 + self.jitter_ratio * unit_random))


class RequestPacer:
    """Local limiter: at most one request every ``min_interval_seconds`` (sec. 29).

    Args:
        min_interval_seconds: Minimum spacing between request starts (0 disables it).
        monotonic: Monotonic time source in seconds.
        sleep: Async sleep function.
    """

    def __init__(
        self,
        min_interval_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be >= 0")
        self._interval = min_interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._next_allowed: float | None = None
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        """Wait until the next request may start, then reserve that slot."""
        async with self._lock:
            now = self._monotonic()
            if self._next_allowed is not None and now < self._next_allowed:
                await self._sleep(self._next_allowed - now)
                now = self._next_allowed
            self._next_allowed = now + self._interval


async def call_with_retry(
    call: Callable[[], T],
    *,
    operation: str,
    policy: RetryPolicy,
    sleep: Sleep,
    pacer: RequestPacer | None = None,
    unit_random: Callable[[], float] = random.random,
    translate: ErrorTranslator = translate_error,
) -> T:
    """Run the synchronous SDK ``call`` in a thread with bounded retries.

    Args:
        call: Zero-argument SDK call.
        operation: Description used in error messages (no secrets).
        policy: Retry policy.
        sleep: Async sleep used between attempts.
        pacer: Optional request pacer awaited before every attempt.
        unit_random: Source of ``U[0, 1)`` for the jitter.
        translate: Exception classifier (only its ``RetryableError`` results are retried).

    Raises:
        NonRetryableError: On the first non-retryable failure (never retried).
        RetryableError: When every attempt failed with a retryable error.
    """
    for attempt in range(1, policy.max_attempts + 1):
        if pacer is not None:
            await pacer.wait()
        try:
            return await asyncio.to_thread(call)
        except SDK_ERRORS as exc:
            error = translate(exc, operation=operation)
            if not isinstance(error, RetryableError):
                raise error from exc
            if attempt == policy.max_attempts:
                raise RetryableError(
                    f"{error.message} (gave up after {attempt} attempts)", code=error.code
                ) from exc
        await sleep(policy.delay(attempt, unit_random()))
    raise AssertionError("unreachable")  # pragma: no cover - loop always returns or raises
