"""``SystemClock`` satisfies the ``IClock`` contract (sec. 8.6)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from adapters.clock.system_clock import SystemClock
from domain.ports import IClock
from tests.contract.clock_contract import ClockContract


class TestSystemClockContract(ClockContract):
    @pytest.fixture
    def clock(self) -> IClock:
        return SystemClock()


def test_system_clock_is_close_to_the_os_clock() -> None:
    now = SystemClock().now_utc()
    assert now.tzinfo is UTC
    assert abs(now - datetime.now(UTC)) < timedelta(seconds=5)
