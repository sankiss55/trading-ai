"""AI veto filter port (sec. 8.5, 17)."""

from typing import Protocol

from domain.models import AIVerdictResult, Snapshot

__all__ = ["IAIFilter"]


class IAIFilter(Protocol):
    """Stateless evaluation of one immutable snapshot."""

    async def evaluate(self, snapshot: Snapshot) -> AIVerdictResult:
        """Always returns a result. A model or network failure is expressed as validity
        ``INVALID`` or ``UNAVAILABLE``, never as an exception towards the use case."""
        ...
