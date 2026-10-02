"""Domain exception hierarchy (sec. 8.3.7, 29).

Adapters translate every SDK/infrastructure error into one of the three classes below;
no SDK-specific exception may leave an adapter.

* ``RetryableError``: network timeout, 5xx, rate limit, overload.
  Bounded retry with exponential backoff and jitter.
* ``NonRetryableError``: invalid order, insufficient buying power, auth, invalid schema.
  No blind retry; log and reject.
* ``StateCriticalError``: state mismatch, unknown submission, unprotected position.
  System goes ``HALTED`` and runs the specific procedure.
"""

__all__ = ["DomainError", "NonRetryableError", "RetryableError", "StateCriticalError"]


class DomainError(Exception):
    """Base class of all domain errors.

    Args:
        message: Human-readable description. MUST NOT contain secrets (sec. 43.1).
        code: Optional stable machine-readable code (e.g. ``"INVALID_STOP"``).
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}" if self.code else self.message


class RetryableError(DomainError):
    """Transient failure: may be retried a bounded number of times with backoff."""


class NonRetryableError(DomainError):
    """Permanent failure for this request: never retried blindly."""


class StateCriticalError(DomainError):
    """The system state is uncertain or unsafe: the system must go HALTED."""
