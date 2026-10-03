"""Minimal in-thread ESMTP server on 127.0.0.1 for notifier tests (no real SMTP).

Speaks just enough of RFC 5321 / 4954 for ``smtplib``: EHLO (advertising
``AUTH PLAIN``), AUTH PLAIN, MAIL, RCPT, DATA, RSET, NOOP, QUIT. Received messages and
login attempts are recorded. ``reject_login`` answers 535 to every AUTH.
"""

from __future__ import annotations

import base64
import socketserver
import threading
from dataclasses import dataclass, field
from email import message_from_bytes
from email.message import Message
from types import TracebackType
from typing import Self


@dataclass
class ReceivedMail:
    """One message accepted by the fake server."""

    sender: str
    recipients: list[str]
    data: bytes

    @property
    def message(self) -> Message:
        """Parsed message."""
        return message_from_bytes(self.data)


@dataclass
class ServerState:
    """What the server saw."""

    reject_login: bool = False
    logins: list[tuple[str, str]] = field(default_factory=list)
    messages: list[ReceivedMail] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class _Handler(socketserver.StreamRequestHandler):
    server: _Server

    def _reply(self, line: str) -> None:
        self.wfile.write(line.encode("ascii") + b"\r\n")
        self.wfile.flush()

    def handle(self) -> None:
        state = self.server.state
        sender = ""
        recipients: list[str] = []
        self._reply("220 fake.smtp ESMTP ready")
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            verb = line.split(" ", 1)[0].upper()
            if verb in ("EHLO", "HELO"):
                self.wfile.write(b"250-fake.smtp\r\n250-AUTH PLAIN\r\n250 8BITMIME\r\n")
                self.wfile.flush()
            elif verb == "AUTH":
                parts = line.split()
                decoded = base64.b64decode(parts[2]).split(b"\0") if len(parts) > 2 else []
                user = decoded[1].decode() if len(decoded) > 2 else ""
                password = decoded[2].decode() if len(decoded) > 2 else ""
                with state.lock:
                    state.logins.append((user, password))
                if state.reject_login:
                    self._reply("535 5.7.8 Username and Password not accepted")
                else:
                    self._reply("235 2.7.0 Accepted")
            elif verb == "MAIL":
                sender = line.split(":", 1)[1].strip().strip("<>").split(">")[0]
                recipients = []
                self._reply("250 OK")
            elif verb == "RCPT":
                recipients.append(line.split(":", 1)[1].strip().strip("<>"))
                self._reply("250 OK")
            elif verb == "DATA":
                self._reply("354 End data with <CR><LF>.<CR><LF>")
                chunks: list[bytes] = []
                while True:
                    data_line = self.rfile.readline()
                    if data_line in (b".\r\n", b""):
                        break
                    chunks.append(data_line[1:] if data_line.startswith(b"..") else data_line)
                with state.lock:
                    state.messages.append(ReceivedMail(sender, recipients, b"".join(chunks)))
                self._reply("250 OK queued")
            elif verb in ("RSET", "NOOP"):
                self._reply("250 OK")
            elif verb == "QUIT":
                self._reply("221 Bye")
                return
            else:
                self._reply("502 Command not implemented")


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, state: ServerState) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.state = state


class FakeSmtpServer:
    """Context manager running the fake server in a background thread."""

    def __init__(self, *, reject_login: bool = False) -> None:
        self.state = ServerState(reject_login=reject_login)
        self._server = _Server(self.state)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        """Listening port on 127.0.0.1."""
        return int(self._server.server_address[1])

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
