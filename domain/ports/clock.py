"""Clock port (sec. 8.5). The domain never reads system time directly (sec. 8.3.6)."""

from datetime import datetime
from typing import Protocol

__all__ = ["IClock"]


class IClock(Protocol):
    """Single source of time for ``application/`` and ``domain/``."""

    def now_utc(self) -> datetime:
        """Current time in UTC (timezone-aware)."""
        ...
