"""In-memory notifiers for tests, simulation and backtest (sec. 8.6).

* :class:`RecordingNotifier` (tests and simulation): keeps every event it receives so a
  test can assert what would have been emailed. It can be told to fail, to exercise
  the retry and "never blocks trading" paths of the policy notifier.
* :class:`NullNotifier` (backtest): discards everything.

Both implement ``INotifier`` (``send``) and the delivery-transport shape
(``deliver -> bool``) used by ``adapters.smtp.notifier.PolicyNotifier``. Neither touches
the network.
"""

from __future__ import annotations

from domain.models import NotificationEvent

__all__ = ["NullNotifier", "RecordingNotifier"]


class RecordingNotifier:
    """Records notifications in memory.

    Args:
        fail_deliveries: Number of upcoming ``deliver`` calls that fail (return
            ``False``) before deliveries succeed again; ``-1`` fails forever.
    """

    def __init__(self, *, fail_deliveries: int = 0) -> None:
        self.events: list[NotificationEvent] = []
        self.failed_attempts = 0
        self._fail_remaining = fail_deliveries

    async def send(self, event: NotificationEvent) -> None:
        """``INotifier``: record ``event`` (failures configured for ``deliver`` only)."""
        self.events.append(event)

    async def deliver(self, event: NotificationEvent) -> bool:
        """Record ``event`` unless a failure is configured."""
        if self._fail_remaining != 0:
            if self._fail_remaining > 0:
                self._fail_remaining -= 1
            self.failed_attempts += 1
            return False
        self.events.append(event)
        return True

    def fail_next(self, count: int) -> None:
        """Make the next ``count`` deliveries fail (``-1``: all of them)."""
        self._fail_remaining = count


class NullNotifier:
    """Discards every notification."""

    async def send(self, event: NotificationEvent) -> None:
        """``INotifier``: do nothing."""

    async def deliver(self, event: NotificationEvent) -> bool:
        """Accept and discard ``event``."""
        return True
