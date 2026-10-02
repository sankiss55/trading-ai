"""Notification event model (sec. 40)."""

from domain.models.base import DomainModel, NonEmptyStr
from domain.models.enums import NotificationEventType, Severity

__all__ = ["NotificationEvent"]


class NotificationEvent(DomainModel):
    """A notification to deliver. Subject and body MUST NOT contain secrets (sec. 43.1)."""

    event_type: NotificationEventType
    severity: Severity
    subject: NonEmptyStr
    body: str
