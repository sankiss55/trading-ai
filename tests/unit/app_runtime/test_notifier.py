"""Notifications (sec. 40, 43.1): policy routing, non-blocking queue, SMTP transport.

SMTP is exercised with the real ``smtplib`` client against an in-thread fake server on
127.0.0.1 (plain SMTP; production uses ``SMTP_SSL`` through the same factory hook).
No real email is ever sent.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
from datetime import timedelta

import pytest
from pydantic import SecretStr

from adapters.simulation.recording_notifier import NullNotifier, RecordingNotifier
from adapters.simulation.sim_clock import FixedClock
from adapters.smtp.notifier import (
    REDACTED,
    ConsoleNotifier,
    NotificationPolicy,
    NotificationRoute,
    PolicyNotifier,
    Redactor,
    SmtpClient,
    SmtpNotifier,
    SmtpSettings,
    effective_severity,
    parse_policy,
    route_event,
)
from domain.models import NotificationEvent, NotificationEventType, Severity
from tests.unit.app_runtime.fake_smtp_server import FakeSmtpServer
from tests.unit.app_runtime.fakes import at

T = NotificationEventType
PASSWORD = "abcd efgh ijkl mnop"
ALPACA_SECRET = "PKSECRET0123456789"


def _event(
    event_type: NotificationEventType = T.TRADE_OPENED,
    severity: Severity = Severity.INFO,
    subject: str = "subject",
    body: str = "body",
) -> NotificationEvent:
    return NotificationEvent(event_type=event_type, severity=severity, subject=subject, body=body)


CRITICAL = _event(T.RISK_HALTED, Severity.CRITICAL, "halted")
WARNING = _event(T.AI_UNAVAILABLE, Severity.WARNING, "ai down")
INFO = _event(T.TRADE_CLOSED, Severity.INFO, "closed")

# --------------------------------------------------------------------------- routing

ALL_POLICIES = [*NotificationPolicy, None]


@pytest.mark.parametrize("policy", ALL_POLICIES)
@pytest.mark.parametrize(
    "event_type",
    [
        T.RISK_HALTED,
        T.STATE_MISMATCH,
        T.ORPHANED_POSITION,
        T.UNPROTECTED_POSITION,
        T.EMERGENCY_CLOSE,
        T.SYSTEM_ERROR,
        T.BROKER_ERROR,
    ],
)
def test_critical_events_are_always_immediate(
    policy: NotificationPolicy | None, event_type: NotificationEventType
) -> None:
    event = _event(event_type, Severity.CRITICAL)
    assert route_event(event, policy) is NotificationRoute.IMMEDIATE


@pytest.mark.parametrize(
    ("policy", "info_route", "warning_route"),
    [
        (NotificationPolicy.EVERY_TRADE, NotificationRoute.IMMEDIATE, NotificationRoute.IMMEDIATE),
        (
            NotificationPolicy.CRITICAL_PLUS_DAILY,
            NotificationRoute.DAILY_SUMMARY,
            NotificationRoute.ERROR_DIGEST,
        ),
        (
            NotificationPolicy.DAILY_ONLY,
            NotificationRoute.DAILY_SUMMARY,
            NotificationRoute.DAILY_SUMMARY,
        ),
        (None, NotificationRoute.DAILY_SUMMARY, NotificationRoute.DAILY_SUMMARY),
    ],
)
def test_policy_table(
    policy: NotificationPolicy | None,
    info_route: NotificationRoute,
    warning_route: NotificationRoute,
) -> None:
    for event_type in (T.TRADE_OPENED, T.TRADE_CLOSED, T.TRADE_REJECTED):
        assert route_event(_event(event_type), policy) is info_route
    assert route_event(WARNING, policy) is warning_route


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_daily_summary_is_sent_immediately(policy: NotificationPolicy | None) -> None:
    assert route_event(_event(T.DAILY_SUMMARY), policy) is NotificationRoute.IMMEDIATE


def test_a_critical_event_type_is_never_downgraded_by_its_severity() -> None:
    event = _event(T.SYSTEM_ERROR, Severity.INFO)
    assert effective_severity(event) is Severity.CRITICAL
    assert route_event(event, NotificationPolicy.DAILY_ONLY) is NotificationRoute.IMMEDIATE


def test_an_informational_type_may_be_raised_by_its_severity() -> None:
    event = _event(T.TRADE_REJECTED, Severity.CRITICAL)
    assert route_event(event, NotificationPolicy.CRITICAL_PLUS_DAILY) is NotificationRoute.IMMEDIATE


def test_parse_policy() -> None:
    assert parse_policy("critical_plus_daily") is NotificationPolicy.CRITICAL_PLUS_DAILY
    assert parse_policy(None) is None
    with pytest.raises(ValueError, match="hourly"):
        parse_policy("hourly")


# --------------------------------------------------------------------------- redaction


def test_redactor_replaces_secrets_including_the_spaceless_app_password() -> None:
    redact = Redactor([SecretStr(PASSWORD), ALPACA_SECRET, "", "ab"])
    text = f"pw={PASSWORD} pw2={PASSWORD.replace(' ', '')} key={ALPACA_SECRET} ab"
    assert redact(text) == f"pw={REDACTED} pw2={REDACTED} key={REDACTED} ab"


# --------------------------------------------------------------------------- policy notifier


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class GatedTransport:
    """Transport whose deliveries wait until ``gate`` is set."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.events: list[NotificationEvent] = []
        self.started = asyncio.Event()

    async def deliver(self, event: NotificationEvent) -> bool:
        self.started.set()
        await self.gate.wait()
        self.events.append(event)
        return True


class RaisingTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def deliver(self, event: NotificationEvent) -> bool:
        self.calls += 1
        raise RuntimeError("transport bug")


def _notifier(
    transport: object,
    *,
    policy: NotificationPolicy | None = NotificationPolicy.CRITICAL_PLUS_DAILY,
    clock: FixedClock | None = None,
    **kwargs: object,
) -> PolicyNotifier:
    return PolicyNotifier(
        transport,  # type: ignore[arg-type]
        policy=policy,
        clock=clock or FixedClock(at(14)),
        error_digest_minutes=30,
        sleep=SleepRecorder(),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_critical_is_delivered_by_the_worker() -> None:
    transport = RecordingNotifier()
    notifier = _notifier(transport)
    await notifier.send(CRITICAL)
    assert await notifier.drain(timeout=5)
    assert transport.events == [CRITICAL]
    assert notifier.delivered == 1
    await notifier.aclose()


async def test_send_returns_immediately_while_the_transport_is_blocked() -> None:
    transport = GatedTransport()
    notifier = _notifier(transport)
    await asyncio.wait_for(notifier.send(CRITICAL), timeout=1)
    await asyncio.wait_for(transport.started.wait(), timeout=1)
    assert transport.events == []
    assert not await notifier.drain(timeout=0.05)
    transport.gate.set()
    assert await notifier.drain(timeout=5)
    assert transport.events == [CRITICAL]
    await notifier.aclose()


async def test_informational_events_go_to_the_daily_summary() -> None:
    transport = RecordingNotifier()
    notifier = _notifier(transport)
    await notifier.send(INFO)
    await notifier.send(_event(T.TRADE_OPENED, subject="opened SPY", body="qty 1"))
    assert await notifier.drain(timeout=5)
    assert transport.events == []
    assert len(notifier.buffered_daily) == 2
    await notifier.send(_event(T.DAILY_SUMMARY, subject="Daily summary", body="Mode: READY"))
    assert await notifier.drain(timeout=5)
    (summary,) = transport.events
    assert summary.event_type is T.DAILY_SUMMARY
    assert summary.subject == "Daily summary"
    assert summary.body.startswith("Mode: READY\n\nInformational events since the last summary: 2")
    assert "TRADE_CLOSED: closed" in summary.body
    assert "TRADE_OPENED: opened SPY" in summary.body
    assert "    qty 1" in summary.body
    assert len(list(notifier.buffered_daily)) == 0
    await notifier.aclose()


async def test_every_trade_sends_informational_events_immediately() -> None:
    transport = RecordingNotifier()
    notifier = _notifier(transport, policy=NotificationPolicy.EVERY_TRADE)
    await notifier.send(INFO)
    await notifier.send(WARNING)
    assert await notifier.drain(timeout=5)
    assert transport.events == [INFO, WARNING]
    await notifier.aclose()


async def test_warnings_are_digested_every_error_digest_minutes() -> None:
    clock = FixedClock(at(14))
    transport = RecordingNotifier()
    notifier = _notifier(transport, clock=clock)
    await notifier.send(WARNING)
    clock.advance(timedelta(minutes=10))
    await notifier.send(_event(T.AI_UNAVAILABLE, Severity.WARNING, "ai still down"))
    clock.advance(timedelta(minutes=19))
    await notifier.flush_due()
    assert await notifier.drain(timeout=5)
    assert transport.events == []
    clock.advance(timedelta(minutes=1))  # 30 min after the first buffered warning
    await notifier.flush_due()
    assert await notifier.drain(timeout=5)
    (digest,) = transport.events
    assert digest.event_type is T.AI_UNAVAILABLE
    assert digest.severity is Severity.WARNING
    assert digest.subject == "Error digest: 2 warning event(s) (window 30 min)"
    assert "ai down" in digest.body
    assert "ai still down" in digest.body
    assert notifier.buffered_digest == ()
    await notifier.flush_due()
    assert await notifier.drain(timeout=5)
    assert len(transport.events) == 1
    await notifier.aclose()


async def test_bounded_retry_then_delivery() -> None:
    transport = RecordingNotifier(fail_deliveries=2)
    sleep = SleepRecorder()
    notifier = PolicyNotifier(
        transport,
        policy=NotificationPolicy.CRITICAL_PLUS_DAILY,
        clock=FixedClock(at(14)),
        error_digest_minutes=30,
        max_attempts=3,
        retry_delay_seconds=2.0,
        sleep=sleep,
    )
    await notifier.send(CRITICAL)
    assert await notifier.drain(timeout=5)
    assert transport.events == [CRITICAL]
    assert transport.failed_attempts == 2
    assert sleep.calls == [2.0, 4.0]
    assert (notifier.delivered, notifier.failed) == (1, 0)
    await notifier.aclose()


async def test_permanent_failure_never_raises_and_the_worker_keeps_going(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = RecordingNotifier(fail_deliveries=3)
    notifier = _notifier(transport, max_attempts=3)
    with caplog.at_level(logging.ERROR):
        await notifier.send(CRITICAL)
        await notifier.send(_event(T.BROKER_ERROR, Severity.CRITICAL, "second"))
        assert await notifier.drain(timeout=5)
    assert notifier.failed == 1
    assert [event.subject for event in transport.events] == ["second"]
    assert any(
        getattr(record, "fields", {}).get("event") == "NOTIFICATION_FAILED"
        for record in caplog.records
    )
    await notifier.aclose()


async def test_a_raising_transport_does_not_kill_the_worker() -> None:
    transport = RaisingTransport()
    notifier = _notifier(transport, max_attempts=2)
    await notifier.send(CRITICAL)
    await notifier.send(CRITICAL)
    assert await notifier.drain(timeout=5)
    assert transport.calls == 4
    assert notifier.failed == 2
    await notifier.aclose()


async def test_queue_overflow_drops_the_oldest_non_critical_email() -> None:
    transport = GatedTransport()
    notifier = _notifier(transport, policy=NotificationPolicy.EVERY_TRADE, queue_max_size=2)
    first = _event(T.TRADE_OPENED, subject="in flight")
    await notifier.send(first)
    await asyncio.wait_for(transport.started.wait(), timeout=1)
    info = _event(T.TRADE_OPENED, subject="queued info")
    await notifier.send(info)
    await notifier.send(CRITICAL)
    critical_2 = _event(T.EMERGENCY_CLOSE, Severity.CRITICAL, "second critical")
    await notifier.send(critical_2)
    assert notifier.dropped == 1
    transport.gate.set()
    assert await notifier.drain(timeout=5)
    assert transport.events == [first, CRITICAL, critical_2]
    await notifier.aclose()


async def test_aclose_flushes_the_pending_digest_and_later_sends_are_logged_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = RecordingNotifier()
    notifier = _notifier(transport)
    await notifier.send(WARNING)
    await notifier.aclose(timeout=5)
    assert [event.subject for event in transport.events] == [
        "Error digest: 1 warning event(s) (window 30 min)"
    ]
    with caplog.at_level(logging.WARNING):
        await notifier.send(CRITICAL)
    assert len(transport.events) == 1
    assert any(
        getattr(record, "fields", {}).get("event") == "NOTIFICATION_AFTER_CLOSE"
        for record in caplog.records
    )


async def test_aclose_is_bounded_when_the_transport_hangs() -> None:
    transport = GatedTransport()
    notifier = _notifier(transport)
    await notifier.send(CRITICAL)
    await asyncio.wait_for(notifier.aclose(timeout=0.1), timeout=5)
    assert transport.events == []


async def test_recording_and_null_notifiers() -> None:
    recording = RecordingNotifier()
    await recording.send(INFO)
    assert recording.events == [INFO]
    recording.fail_next(1)
    assert not await recording.deliver(INFO)
    assert await recording.deliver(INFO)
    null = NullNotifier()
    await null.send(INFO)
    assert await null.deliver(INFO)


# --------------------------------------------------------------------------- SMTP


def _plain_smtp(host: str, port: int, timeout: float) -> SmtpClient:
    return smtplib.SMTP(host, port, timeout=timeout)


def _settings(port: int) -> SmtpSettings:
    return SmtpSettings(
        username="bot@example.com",
        password=SecretStr(PASSWORD),
        recipient="owner@example.com",
        host="127.0.0.1",
        port=port,
        timeout_seconds=5,
        subject_prefix="[trading-agent test]",
    )


async def test_smtp_sends_through_a_fake_server() -> None:
    with FakeSmtpServer() as server:
        notifier = SmtpNotifier(_settings(server.port), smtp_factory=_plain_smtp)
        assert await notifier.deliver(CRITICAL)
    assert server.state.logins == [("bot@example.com", PASSWORD)]
    (mail,) = server.state.messages
    assert mail.sender == "bot@example.com"
    assert mail.recipients == ["owner@example.com"]
    message = mail.message
    assert message["Subject"] == "[trading-agent test] CRITICAL RISK_HALTED: halted"
    assert message["From"] == "bot@example.com"
    assert message["To"] == "owner@example.com"
    assert notifier.sent_count == 1
    assert notifier.failures == ()


async def test_smtp_never_puts_secrets_in_subject_or_body() -> None:
    event = _event(
        T.SYSTEM_ERROR,
        Severity.CRITICAL,
        subject=f"leak {PASSWORD}",
        body=f"key={ALPACA_SECRET} pw={PASSWORD.replace(' ', '')}",
    )
    with FakeSmtpServer() as server:
        notifier = SmtpNotifier(
            _settings(server.port),
            smtp_factory=_plain_smtp,
            redactor=Redactor([ALPACA_SECRET]),
        )
        assert await notifier.deliver(event)
    (mail,) = server.state.messages
    raw = mail.data.decode("utf-8", "replace")
    for secret in (PASSWORD, PASSWORD.replace(" ", ""), ALPACA_SECRET):
        assert secret not in raw
    assert REDACTED in str(mail.message["Subject"])
    assert REDACTED in raw


async def test_rejected_login_is_recorded_and_never_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with FakeSmtpServer(reject_login=True) as server, caplog.at_level(logging.ERROR):
        notifier = SmtpNotifier(_settings(server.port), smtp_factory=_plain_smtp)
        assert not await notifier.deliver(CRITICAL)
    (failure,) = notifier.failures
    assert failure.error_type == "SMTPAuthenticationError"
    assert failure.smtp_code == 535
    assert failure.event_type is T.RISK_HALTED
    assert server.state.messages == []
    logged = " ".join(f"{r.getMessage()} {getattr(r, 'fields', {})}" for r in caplog.records)
    assert "SMTPAuthenticationError" in logged
    assert PASSWORD not in logged
    assert "not accepted" not in logged  # server text is never logged


async def test_unreachable_server_is_recorded_and_never_raises() -> None:
    def refused(host: str, port: int, timeout: float) -> SmtpClient:
        raise ConnectionRefusedError(10061, "refused")

    notifier = SmtpNotifier(_settings(465), smtp_factory=refused)
    assert not await notifier.deliver(CRITICAL)
    (failure,) = notifier.failures
    assert failure.smtp_code is None
    assert failure.error_type == "ConnectionRefusedError"
    await notifier.send(CRITICAL)  # INotifier.send swallows too
    assert len(notifier.failures) == 2


async def test_factory_errors_never_escape() -> None:
    def broken(host: str, port: int, timeout: float) -> SmtpClient:
        raise ValueError("bad host")

    notifier = SmtpNotifier(_settings(465), smtp_factory=broken)
    assert not await notifier.deliver(CRITICAL)
    assert notifier.failures[0].error_type == "ValueError"


def test_subject_is_one_line_no_header_injection() -> None:
    notifier = SmtpNotifier(_settings(465), smtp_factory=_plain_smtp)
    event = _event(T.SYSTEM_ERROR, Severity.CRITICAL, subject="a\r\nBcc: evil@example.com")
    message = notifier.build_message(event)
    assert "\n" not in str(message["Subject"])
    assert message["Bcc"] is None


def test_settings_repr_masks_the_password() -> None:
    assert PASSWORD not in repr(_settings(465))


async def test_smtp_through_the_policy_notifier() -> None:
    with FakeSmtpServer() as server:
        transport = SmtpNotifier(_settings(server.port), smtp_factory=_plain_smtp)
        notifier = _notifier(transport)
        await notifier.send(CRITICAL)
        await notifier.send(INFO)
        await notifier.aclose(timeout=10)
    assert [m.message["Subject"] for m in server.state.messages] == [
        "[trading-agent test] CRITICAL RISK_HALTED: halted"
    ]


async def test_console_notifier_logs_redacted(caplog: pytest.LogCaptureFixture) -> None:
    notifier = ConsoleNotifier(redactor=Redactor([ALPACA_SECRET]))
    with caplog.at_level(logging.INFO, logger="notifications.console"):
        assert await notifier.deliver(_event(T.BROKER_ERROR, Severity.CRITICAL, body=ALPACA_SECRET))
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert record.fields["body"] == REDACTED  # type: ignore[attr-defined]
    assert record.fields["notification_type"] == "BROKER_ERROR"  # type: ignore[attr-defined]
