"""Backtest data: stored calendar, CSV loading and the synthetic generator."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

import pytest

from backtest.data import (
    BARS_DIR,
    CALENDAR_FILE,
    SyntheticSpec,
    business_days,
    generate_synthetic_data,
    load_backtest_data,
    load_calendar,
)
from domain.errors import NonRetryableError
from domain.models import DataFeed, Timeframe

SPEC = SyntheticSpec(
    symbols=("SYNTH", "SYNB"),
    start=date(2025, 1, 2),
    end=date(2025, 1, 8),
    seed=7,
    early_closes={date(2025, 1, 3): time(13, 0)},
)


def _files(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob("*.csv")}


def test_same_seed_gives_identical_files(tmp_path: Path) -> None:
    generate_synthetic_data(tmp_path / "a", SPEC)
    generate_synthetic_data(tmp_path / "b", SPEC)
    a, b = _files(tmp_path / "a"), _files(tmp_path / "b")
    assert a == b
    assert set(a) == {
        CALENDAR_FILE,
        f"{BARS_DIR}/SYNTH_1Min.csv",
        f"{BARS_DIR}/SYNTH_1Day.csv",
        f"{BARS_DIR}/SYNB_1Min.csv",
        f"{BARS_DIR}/SYNB_1Day.csv",
    }


def test_different_seed_gives_different_prices(tmp_path: Path) -> None:
    generate_synthetic_data(tmp_path / "a", SPEC)
    other = SyntheticSpec(symbols=SPEC.symbols, start=SPEC.start, end=SPEC.end, seed=8)
    generate_synthetic_data(tmp_path / "b", other)
    path = f"{BARS_DIR}/SYNTH_1Min.csv"
    assert _files(tmp_path / "a")[path] != _files(tmp_path / "b")[path]


async def test_generated_dataset_loads_with_calendar_and_early_close(tmp_path: Path) -> None:
    sessions = generate_synthetic_data(tmp_path, SPEC)
    assert [s.session_date for s in sessions] == business_days(SPEC.start, SPEC.end)
    data = load_backtest_data(tmp_path, feed=DataFeed.IEX)
    assert data.sessions == sessions
    early = next(s for s in data.sessions if s.session_date == date(2025, 1, 3))
    assert early.is_early_close
    assert early.close_utc == datetime(2025, 1, 3, 18, 0, tzinfo=UTC)
    minutes = await data.feed.get_minute_bars("SYNTH", early.open_utc, early.close_utc)
    assert len(minutes) == 210  # 09:30-13:00
    assert all(b.timeframe is Timeframe.MIN_1 for b in minutes)
    regular = data.sessions[0]
    minutes = await data.feed.get_minute_bars("SYNTH", regular.open_utc, regular.close_utc)
    assert len(minutes) == 390
    daily = await data.feed.get_daily_bars("SYNTH", SPEC.start, SPEC.end)
    assert len(daily) == len(sessions)
    assert daily[0].volume == sum(b.volume for b in minutes)
    assert all(isinstance(b.close, Decimal) for b in minutes)


def test_missing_minutes_are_dropped(tmp_path: Path) -> None:
    spec = SyntheticSpec(
        symbols=("SYNTH",), start=date(2025, 1, 2), end=date(2025, 1, 2), seed=1,
        missing_minute_rate=0.2,
    )  # fmt: skip
    generate_synthetic_data(tmp_path, spec)
    lines = (tmp_path / BARS_DIR / "SYNTH_1Min.csv").read_text(encoding="utf-8").splitlines()
    assert 250 < len(lines) - 1 < 390


def test_calendar_is_required(tmp_path: Path) -> None:
    generate_synthetic_data(tmp_path, SPEC)
    (tmp_path / CALENDAR_FILE).unlink()
    with pytest.raises(NonRetryableError) as info:
        load_backtest_data(tmp_path, feed=DataFeed.IEX)
    assert info.value.code == "INVALID_CALENDAR"


@pytest.mark.parametrize(
    "content",
    [
        "date,open,close\n2025-01-02,09:30,16:00\n",
        "session_date,open_local,close_local\n2025-13-02,09:30,16:00\n",
        "session_date,open_local,close_local\n2025-01-02,9h30,16:00\n",
        "session_date,open_local,close_local\n2025-01-02,09:30,16:00\n2025-01-02,09:30,16:00\n",
        "session_date,open_local,close_local\n2025-01-02,09:30,16:00\n2025-01-03,10:00,16:00\n",
    ],
)
def test_invalid_calendar_fails(tmp_path: Path, content: str) -> None:
    path = tmp_path / CALENDAR_FILE
    path.write_text(content, encoding="utf-8")
    with pytest.raises(NonRetryableError) as info:
        load_calendar(path)
    assert info.value.code == "INVALID_CALENDAR"


def test_duplicate_stored_bar_is_rejected(tmp_path: Path) -> None:
    generate_synthetic_data(tmp_path, SPEC)
    path = tmp_path / BARS_DIR / "SYNTH_1Min.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, lines[5]]) + "\n", encoding="utf-8")
    with pytest.raises(NonRetryableError) as info:
        load_backtest_data(tmp_path, feed=DataFeed.IEX)
    assert info.value.code == "DUPLICATE_BAR"


def test_missing_bars_directory_fails(tmp_path: Path) -> None:
    (tmp_path / CALENDAR_FILE).write_text(
        "session_date,open_local,close_local\n2025-01-02,09:30,16:00\n", encoding="utf-8"
    )
    with pytest.raises(NonRetryableError) as info:
        load_backtest_data(tmp_path, feed=DataFeed.IEX)
    assert info.value.code == "INVALID_BAR_DATA"
