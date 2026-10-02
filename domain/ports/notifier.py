"""Notifier port (sec. 8.5, 40)."""

from typing import Protocol

from domain.models import NotificationEvent

__all__ = ["INotifier"]


class INotifier(Protocol):
    """Outbound notifications."""

    async def send(self, event: NotificationEvent) -> None:
        """Send a notification. Never blocks trading."""
        ...
