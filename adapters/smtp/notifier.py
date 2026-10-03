"""Notifications (sec. 40): Gmail SMTP transport, policy routing and a delivery queue.

Three pieces, composed by ``app/container.py``:

* :class:`SmtpNotifier`: the transport. One email per event over ``smtplib.SMTP_SSL``
  (Gmail, port 465, App Password). The blocking SMTP dialogue runs in a worker thread
  (``asyncio.to_thread``) with a socket timeout. It never raises: a failure is logged
  (error type and SMTP code only, never the server text or a credential), recorded in
  :attr:`SmtpNotifier.failures`, and ``deliver`` returns ``False``.
* :class:`ConsoleNotifier`: fallback transport when the SMTP secrets are missing. It
  writes the notification to the structured log.
* :class:`PolicyNotifier`: the ``INotifier`` the system uses. It applies
  ``notifications.policy`` (:func:`route_event`) and hands immediate emails to a bounded
  queue served by one background worker (the ``notification_worker`` of sec. 27), so
  ``send`` returns at once and never blocks trading (sec. 33: "SMTP down: trading
  continues; notifications are queued and logged").

Policy (sec. 40). Critical events are always sent immediately, whatever the policy:

    policy                CRITICAL   WARNING                            INFO
    every_trade           immediate  immediate                          immediate
    critical_plus_daily   immediate  error digest (error_digest_min.)   daily summary
    daily_only / null     immediate  daily summary                      daily summary

``null`` (owner decision pending) is handled as ``daily_only``, the most conservative
choice; trading cannot be enabled while it is ``null`` anyway (AC-20). The criticality of
an event is the higher of its own ``severity`` and the sec. 40 table for its type, so a
critical event type is never downgraded by a caller. A ``DAILY_SUMMARY`` event is the
summary itself: it is sent immediately with the buffered informational events appended.
The error digest is flushed by :meth:`PolicyNotifier.flush_due` (called by the runtime
loop) once ``error_digest_minutes`` have passed since its first buffered event.

Secrets (sec. 43.1): subjects and bodies are redacted against every registered secret
value before they are sent or logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import smtplib
import ssl
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from enum import StrEnum
from typing import Final, Protocol

from pydantic import SecretStr

from domain.models import NotificationEvent, NotificationEventType, Severity
from domain.ports import IClock

__all__ = [
    "EVENT_SEVERITY",
    "GMAIL_SMTP_HOST",
    "GMAIL_SMTP_SSL_PORT",
    "REDACTED",
    "ConsoleNotifier",
    "DeliveryFailure",
    "DeliveryTransport",
    "NotificationPolicy",
    "NotificationRoute",
    "PolicyNotifier",
    "Redactor",
    "SmtpClient",
    "SmtpNotifier",
    "SmtpSettings",
    "effective_severity",
    "gmail_ssl_factory",
    "parse_policy",
    "route_event",
]

_LOGGER = logging.getLogger(__name__)

REDACTED: Final = "[REDACTED]"
GMAIL_SMTP_HOST: Final = "smtp.gmail.com"
GMAIL_SMTP_SSL_PORT: Final = 465
_MIN_SECRET_LENGTH: Final = 4
"""Shorter values are not redacted: they would mangle ordinary text."""


# --------------------------------------------------------------------------- policy


class NotificationPolicy(StrEnum):
    """``notifications.policy`` values (sec. 40, OWNER_DECISION)."""

    EVERY_TRADE = "every_trade"
    CRITICAL_PLUS_DAILY = "critical_plus_daily"
    DAILY_ONLY = "daily_only"


class NotificationRoute(StrEnum):
    """Where an event goes under the active policy."""

    IMMEDIATE = "IMMEDIATE"
    ERROR_DIGEST = "ERROR_DIGEST"
    DAILY_SUMMARY = "DAILY_SUMMARY"


EVENT_SEVERITY: Final[Mapping[NotificationEventType, Severity]] = {
    NotificationEventType.TRADE_OPENED: Severity.INFO,
    NotificationEventType.TRADE_CLOSED: Severity.INFO,
    NotificationEventType.TRADE_REJECTED: Severity.INFO,
    NotificationEventType.AI_UNAVAILABLE: Severity.WARNING,
    NotificationEventType.RISK_HALTED: Severity.CRITICAL,
    NotificationEventType.STATE_MISMATCH: Severity.CRITICAL,
    NotificationEventType.ORPHANED_POSITION: Severity.CRITICAL,
    NotificationEventType.UNPROTECTED_POSITION: Severity.CRITICAL,
    NotificationEventType.EMERGENCY_CLOSE: Severity.CRITICAL,
    NotificationEventType.SYSTEM_ERROR: Severity.CRITICAL,
    NotificationEventType.BROKER_ERROR: Severity.CRITICAL,
    NotificationEventType.DAILY_SUMMARY: Severity.INFO,
}
"""Criticality of every event type (sec. 40 table)."""

_SEVERITY_RANK: Final[Mapping[Severity, int]] = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
}


def effective_severity(event: NotificationEvent) -> Severity:
    """The higher of the event's own severity and the sec. 40 severity of its type."""
    table = EVENT_SEVERITY.get(event.event_type, Severity.CRITICAL)
    return max(event.severity, table, key=lambda severity: _SEVERITY_RANK[severity])


def parse_policy(value: NotificationPolicy | str | None) -> NotificationPolicy | None:
    """``notifications.policy`` as a :class:`NotificationPolicy` (``None`` stays pending).

    Raises:
        ValueError: unknown policy text.
    """
    if value is None or isinstance(value, NotificationPolicy):
        return value
    return NotificationPolicy(value)


def route_event(event: NotificationEvent, policy: NotificationPolicy | None) -> NotificationRoute:
    """Route of ``event`` under ``policy`` (see the module table)."""
    if event.event_type is NotificationEventType.DAILY_SUMMARY:
        return NotificationRoute.IMMEDIATE
    severity = effective_severity(event)
    if severity is Severity.CRITICAL or policy is NotificationPolicy.EVERY_TRADE:
        return NotificationRoute.IMMEDIATE
    if policy is NotificationPolicy.CRITICAL_PLUS_DAILY and severity is Severity.WARNING:
        return NotificationRoute.ERROR_DIGEST
    return NotificationRoute.DAILY_SUMMARY


# --------------------------------------------------------------------------- redaction


class Redactor:
    """Replaces every registered secret value in a text with :data:`REDACTED`.

    Args:
        secrets: Initial secret values (``SecretStr`` or plain text).
    """

    def __init__(self, secrets: Iterable[SecretStr | str] = ()) -> None:
        self._values: list[str] = []
        self.add(*secrets)

    def add(self, *secrets: SecretStr | str) -> None:
        """Register more secret values (blank or very short values are ignored).

        A value containing spaces (Gmail shows App Passwords in groups of four) is also
        registered without them.
        """
        for secret in secrets:
            raw = secret.get_secret_value() if isinstance(secret, SecretStr) else secret
            for value in {raw.strip(), "".join(raw.split())}:
                if len(value) >= _MIN_SECRET_LENGTH and value not in self._values:
                    self._values.append(value)
        self._values.sort(key=len, reverse=True)

    def __call__(self, text: str) -> str:
        """``text`` with every registered secret replaced."""
        for value in self._values:
            if value in text:
                text = text.replace(value, REDACTED)
        return text


# --------------------------------------------------------------------------- transports


class DeliveryTransport(Protocol):
    """Sends one notification now. Never raises: returns ``False`` on failure."""

    async def deliver(self, event: NotificationEvent) -> bool:
        """Deliver ``event``; ``True`` if it was accepted by the channel."""
        ...


class SmtpClient(Protocol):
    """The subset of ``smtplib.SMTP`` used here (injectable in tests)."""

    def login(self, user: str, password: str) -> object:
        """Authenticate."""
        ...

    def send_message(self, msg: EmailMessage) -> object:
        """Send one message."""
        ...

    def quit(self) -> object:
        """End the session politely."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


SmtpFactory = Callable[[str, int, float], SmtpClient]
"""``(host, port, timeout_seconds) -> connected client``."""


def gmail_ssl_factory(host: str, port: int, timeout: float) -> SmtpClient:
    """Implicit-TLS SMTP connection with certificate verification (Gmail port 465)."""
    return smtplib.SMTP_SSL(host, port, timeout=timeout, context=ssl.create_default_context())


@dataclass(frozen=True, slots=True)
class SmtpSettings:
    """SMTP account and recipient (``GMAIL_USER``, ``GMAIL_APP_PASSWORD``,
    ``NOTIFICATION_EMAIL``; sec. 7.2, 40). ``password`` is masked in ``repr``."""

    username: str
    password: SecretStr
    recipient: str
    host: str = GMAIL_SMTP_HOST
    port: int = GMAIL_SMTP_SSL_PORT
    timeout_seconds: float = 30.0
    subject_prefix: str = "[trading-agent]"


@dataclass(frozen=True, slots=True)
class DeliveryFailure:
    """One failed delivery attempt, without any server text or credential."""

    event_type: NotificationEventType
    error_type: str
    smtp_code: int | None


def _single_line(text: str) -> str:
    return " ".join(text.split())


class SmtpNotifier:
    """Gmail SMTP transport (also a plain ``INotifier`` that sends every event at once).

    Args:
        settings: Account, recipient and connection settings.
        smtp_factory: Opens a connected SMTP client (default: verified ``SMTP_SSL``).
        redactor: Secret redactor; the SMTP password is always registered in it.
        max_recorded_failures: Size of the in-memory failure record.
    """

    def __init__(
        self,
        settings: SmtpSettings,
        *,
        smtp_factory: SmtpFactory | None = None,
        redactor: Redactor | None = None,
        max_recorded_failures: int = 100,
    ) -> None:
        self._settings = settings
        self._factory = smtp_factory or gmail_ssl_factory
        self._redact = redactor or Redactor()
        self._redact.add(settings.password)
        self._failures: deque[DeliveryFailure] = deque(maxlen=max_recorded_failures)
        self._sent = 0

    @property
    def failures(self) -> tuple[DeliveryFailure, ...]:
        """Recorded failed attempts, oldest first (bounded)."""
        return tuple(self._failures)

    @property
    def sent_count(self) -> int:
        """Number of messages accepted by the SMTP server."""
        return self._sent

    def build_message(self, event: NotificationEvent) -> EmailMessage:
        """The redacted email for ``event`` (subject on one line: no header injection)."""
        settings = self._settings
        severity = effective_severity(event)
        subject = _single_line(
            self._redact(
                f"{settings.subject_prefix} {severity} {event.event_type}: {event.subject}"
            )
        )
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = settings.username
        message["To"] = settings.recipient
        message["Date"] = formatdate(usegmt=True)
        message["Message-ID"] = make_msgid(domain="trading-agent.local")
        message.set_content(self._redact(event.body) or "(no body)")
        return message

    async def deliver(self, event: NotificationEvent) -> bool:
        """Send ``event`` now in a worker thread. Never raises."""
        try:
            message = self.build_message(event)
            await asyncio.wait_for(
                asyncio.to_thread(self._send_blocking, message),
                timeout=self._settings.timeout_seconds * 2,
            )
        except Exception as exc:  # noqa: BLE001 - a notification never breaks its caller (sec. 40)
            self._record_failure(event, exc)
            return False
        self._sent += 1
        return True

    async def send(self, event: NotificationEvent) -> None:
        """``INotifier``: deliver immediately, swallowing (and recording) failures."""
        await self.deliver(event)

    def _send_blocking(self, message: EmailMessage) -> None:
        settings = self._settings
        client = self._factory(settings.host, settings.port, settings.timeout_seconds)
        try:
            client.login(settings.username, settings.password.get_secret_value())
            client.send_message(message)
        except BaseException:
            client.close()
            raise
        with contextlib.suppress(smtplib.SMTPException, OSError):
            client.quit()

    def _record_failure(self, event: NotificationEvent, exc: BaseException) -> None:
        code = getattr(exc, "smtp_code", None)
        failure = DeliveryFailure(
            event_type=event.event_type,
            error_type=type(exc).__name__,
            smtp_code=code if isinstance(code, int) else None,
        )
        self._failures.append(failure)
        _LOGGER.error(
            "SMTP notification delivery failed",
            extra={
                "fields": {
                    "event": "NOTIFICATION_SEND_FAILED",
                    "notification_type": str(event.event_type),
                    "error_type": failure.error_type,
                    "smtp_code": failure.smtp_code,
                }
            },
        )


class ConsoleNotifier:
    """Fallback transport: writes notifications to the log (no SMTP secrets configured).

    Args:
        redactor: Secret redactor applied to subject and body.
        logger: Destination logger.
    """

    def __init__(
        self, *, redactor: Redactor | None = None, logger: logging.Logger | None = None
    ) -> None:
        self._redact = redactor or Redactor()
        self._logger = logger or logging.getLogger("notifications.console")
        self.delivered: int = 0

    async def deliver(self, event: NotificationEvent) -> bool:
        """Log ``event``. Always succeeds."""
        severity = effective_severity(event)
        level = logging.WARNING if severity is Severity.CRITICAL else logging.INFO
        self._logger.log(
            level,
            "notification (console fallback, SMTP not configured)",
            extra={
                "fields": {
                    "event": "NOTIFICATION_CONSOLE",
                    "notification_type": str(event.event_type),
                    "severity": str(severity),
                    "subject": _single_line(self._redact(event.subject)),
                    "body": self._redact(event.body),
                }
            },
        )
        self.delivered += 1
        return True

    async def send(self, event: NotificationEvent) -> None:
        """``INotifier``: log immediately."""
        await self.deliver(event)


# --------------------------------------------------------------------------- policy notifier

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _Buffered:
    received_at_utc: datetime
    event: NotificationEvent


def _describe(item: _Buffered) -> str:
    event = item.event
    line = (
        f"- {item.received_at_utc.isoformat()} {effective_severity(event)} "
        f"{event.event_type}: {_single_line(event.subject)}"
    )
    body = event.body.strip()
    if body:
        line += "\n" + "\n".join(f"    {part}" for part in body.splitlines())
    return line


class PolicyNotifier:
    """``INotifier`` applying ``notifications.policy`` with a non-blocking delivery queue.

    Args:
        transport: Channel that sends one notification (SMTP, console, recording...).
        policy: ``notifications.policy`` (``None`` = pending, handled as ``daily_only``).
        clock: Time source of the digest window and of buffered entries.
        error_digest_minutes: Error digest period (``critical_plus_daily``).
        queue_max_size: Bound of the delivery queue (``system.queue_max_size``). When it
            is full the oldest non-critical queued email is dropped (``QUEUE_OVERFLOW``).
        buffer_max_size: Bound of the daily-summary and digest buffers (oldest dropped).
        max_attempts: Delivery attempts per email (bounded retry, sec. 29).
        retry_delay_seconds: First retry delay, doubled on each further attempt.
        sleep: Async sleep used between attempts (injectable in tests).
        redactor: Secret redactor applied to composed summaries and digests.
    """

    def __init__(
        self,
        transport: DeliveryTransport,
        *,
        policy: NotificationPolicy | str | None,
        clock: IClock,
        error_digest_minutes: int,
        queue_max_size: int = 100,
        buffer_max_size: int = 500,
        max_attempts: int = 3,
        retry_delay_seconds: float = 2.0,
        sleep: Sleep = asyncio.sleep,
        redactor: Redactor | None = None,
    ) -> None:
        if error_digest_minutes < 1 or queue_max_size < 1 or buffer_max_size < 1:
            raise ValueError("digest period and queue/buffer sizes must be >= 1")
        if max_attempts < 1 or retry_delay_seconds < 0:
            raise ValueError("max_attempts must be >= 1 and retry_delay_seconds >= 0")
        self._transport = transport
        self._policy = parse_policy(policy)
        self._clock = clock
        self._digest_period = timedelta(minutes=error_digest_minutes)
        self._queue_max = queue_max_size
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay_seconds
        self._sleep = sleep
        self._redact = redactor or Redactor()
        self._daily: deque[_Buffered] = deque(maxlen=buffer_max_size)
        self._digest: deque[_Buffered] = deque(maxlen=buffer_max_size)
        self._digest_started: datetime | None = None
        self._daily_dropped = 0
        self._pending: deque[NotificationEvent] = deque()
        self._wakeup = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._worker: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self.delivered = 0
        self.failed = 0
        self.dropped = 0

    @property
    def policy(self) -> NotificationPolicy | None:
        """Active policy (``None`` = pending owner decision)."""
        return self._policy

    @property
    def pending_count(self) -> int:
        """Emails waiting in the delivery queue."""
        return len(self._pending)

    @property
    def buffered_daily(self) -> tuple[NotificationEvent, ...]:
        """Informational events waiting for the next daily summary."""
        return tuple(item.event for item in self._daily)

    @property
    def buffered_digest(self) -> tuple[NotificationEvent, ...]:
        """Warning events waiting for the next error digest."""
        return tuple(item.event for item in self._digest)

    async def send(self, event: NotificationEvent) -> None:
        """Route ``event`` per the policy. Returns at once; never raises for delivery."""
        if event.event_type is NotificationEventType.DAILY_SUMMARY:
            self._enqueue(self._compose_daily_summary(event))
            return
        route = route_event(event, self._policy)
        if route is NotificationRoute.IMMEDIATE:
            self._enqueue(event)
        elif route is NotificationRoute.ERROR_DIGEST:
            if not self._digest:
                self._digest_started = self._clock.now_utc()
            self._digest.append(_Buffered(self._clock.now_utc(), event))
        else:
            if len(self._daily) == self._daily.maxlen:
                self._daily_dropped += 1
            self._daily.append(_Buffered(self._clock.now_utc(), event))

    async def flush_due(self) -> None:
        """Queue the error digest if its window (``error_digest_minutes``) has elapsed."""
        started = self._digest_started
        if (
            self._digest
            and started is not None
            and (self._clock.now_utc() >= started + self._digest_period)
        ):
            self._enqueue(self._compose_digest())

    async def drain(self, timeout: float | None = None) -> bool:
        """Wait until the delivery queue is empty. ``False`` if ``timeout`` expired."""
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
        except TimeoutError:
            return False
        return True

    async def aclose(self, timeout: float = 30.0) -> None:
        """Flush the pending digest, deliver the queue (bounded by ``timeout``) and stop."""
        if self._closed:
            return
        if self._digest:
            self._enqueue(self._compose_digest())
        self._closing = True
        self._wakeup.set()
        worker = self._worker
        if worker is not None:
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout)
            except TimeoutError:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
                _LOGGER.error(
                    "notification queue not fully delivered before shutdown",
                    extra={
                        "fields": {
                            "event": "NOTIFICATIONS_UNDELIVERED",
                            "count": len(self._pending),
                        }
                    },
                )
        if self._daily:
            _LOGGER.info(
                "informational notifications not summarized before shutdown",
                extra={
                    "fields": {"event": "NOTIFICATIONS_UNSUMMARIZED", "count": len(self._daily)}
                },
            )
        self._closed = True

    # ------------------------------------------------------------------ internals

    def _enqueue(self, event: NotificationEvent) -> None:
        if self._closed:
            _LOGGER.warning(
                "notifier closed: notification logged only",
                extra={
                    "fields": {
                        "event": "NOTIFICATION_AFTER_CLOSE",
                        "notification_type": str(event.event_type),
                        "subject": _single_line(self._redact(event.subject)),
                    }
                },
            )
            return
        if len(self._pending) >= self._queue_max:
            self._make_room()
        self._pending.append(event)
        self._idle.clear()
        self._wakeup.set()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(
                self._run(), name="notification_worker"
            )

    def _make_room(self) -> None:
        dropped: NotificationEvent | None = None
        for index, queued in enumerate(self._pending):
            if effective_severity(queued) is not Severity.CRITICAL:
                dropped = queued
                del self._pending[index]
                break
        if dropped is None:
            dropped = self._pending.popleft()
        self.dropped += 1
        _LOGGER.error(
            "notification queue full: oldest email dropped",
            extra={
                "fields": {
                    "event": "QUEUE_OVERFLOW",
                    "queue": "notifications",
                    "notification_type": str(dropped.event_type),
                }
            },
        )

    async def _run(self) -> None:
        while True:
            if not self._pending:
                self._idle.set()
                if self._closing:
                    return
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            await self._deliver_with_retries(self._pending.popleft())

    async def _deliver_with_retries(self, event: NotificationEvent) -> None:
        for attempt in range(1, self._max_attempts + 1):
            try:
                delivered = await self._transport.deliver(event)
            except Exception:  # noqa: BLE001 - transports must not raise; keep the worker alive
                delivered = False
            if delivered:
                self.delivered += 1
                return
            if attempt < self._max_attempts:
                await self._sleep(self._retry_delay * 2 ** (attempt - 1))
        self.failed += 1
        _LOGGER.error(
            "notification not delivered",
            extra={
                "fields": {
                    "event": "NOTIFICATION_FAILED",
                    "notification_type": str(event.event_type),
                    "attempts": self._max_attempts,
                }
            },
        )

    def _compose_digest(self) -> NotificationEvent:
        items = list(self._digest)
        self._digest.clear()
        self._digest_started = None
        types = {item.event.event_type for item in items}
        event_type = types.pop() if len(types) == 1 else NotificationEventType.SYSTEM_ERROR
        minutes = int(self._digest_period.total_seconds() // 60)
        body = "\n".join(_describe(item) for item in items)
        return NotificationEvent(
            event_type=event_type,
            severity=Severity.WARNING,
            subject=f"Error digest: {len(items)} warning event(s) (window {minutes} min)",
            body=self._redact(body),
        )

    def _compose_daily_summary(self, event: NotificationEvent) -> NotificationEvent:
        items = list(self._daily)
        dropped = self._daily_dropped
        self._daily.clear()
        self._daily_dropped = 0
        header = f"Informational events since the last summary: {len(items)}"
        if dropped:
            header += f" ({dropped} older event(s) dropped: buffer full)"
        summary = "\n".join([header, *(_describe(item) for item in items)])
        body = f"{event.body.rstrip()}\n\n{summary}" if event.body.strip() else summary
        return NotificationEvent(
            event_type=NotificationEventType.DAILY_SUMMARY,
            severity=event.severity,
            subject=event.subject,
            body=self._redact(body),
        )
