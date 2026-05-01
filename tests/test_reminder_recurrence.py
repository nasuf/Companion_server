"""Spec Part 5 §4.2 提醒 recurrence 字段单测.

覆盖 _reminder_matches_date 各 recurrence 路径 + None 兼容性.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.proactive.special_dates import (
    _extract_reminders_for_date,
    _reminder_matches_date,
)


_TZ = timezone.utc


def _row(occur=None, recurrence=None, summary="提醒事项"):
    return SimpleNamespace(
        occurTime=occur, recurrence=recurrence,
        content=summary, summary=summary,
        subCategory="提醒",
    )


# ── _reminder_matches_date 直接单测 ─────────────────────────────────


def test_recurrence_yearly_matches_same_month_day():
    """spec §4.2: yearly → (month, day) 相同则命中, 跨年无所谓."""
    occur = datetime(2025, 3, 20, tzinfo=_TZ)
    assert _reminder_matches_date(occur, "yearly", date(2026, 3, 20)) is True
    assert _reminder_matches_date(occur, "yearly", date(2030, 3, 20)) is True
    assert _reminder_matches_date(occur, "yearly", date(2026, 3, 21)) is False


def test_recurrence_monthly_matches_same_day():
    """spec §4.2: monthly → day 相同则命中, 任何月."""
    occur = datetime(2025, 1, 15, tzinfo=_TZ)
    assert _reminder_matches_date(occur, "monthly", date(2026, 1, 15)) is True
    assert _reminder_matches_date(occur, "monthly", date(2026, 7, 15)) is True
    assert _reminder_matches_date(occur, "monthly", date(2026, 7, 16)) is False


def test_recurrence_weekly_matches_weekday():
    """spec §4.2: weekly → weekday() 相同则命中."""
    # 2025-01-15 是周三 (weekday=2)
    occur = datetime(2025, 1, 15, tzinfo=_TZ)
    assert _reminder_matches_date(occur, "weekly", date(2025, 1, 22)) is True  # 下周三
    assert _reminder_matches_date(occur, "weekly", date(2026, 4, 22)) is True  # 也是周三
    assert _reminder_matches_date(occur, "weekly", date(2026, 4, 23)) is False  # 周四


def test_recurrence_daily_always_matches():
    """spec §4.2: daily → 任何日期命中, 无需 occur_time."""
    assert _reminder_matches_date(None, "daily", date(2026, 1, 1)) is True
    assert _reminder_matches_date(
        datetime(2020, 1, 1, tzinfo=_TZ), "daily", date(2026, 12, 31),
    ) is True


def test_recurrence_once_only_matches_exact_date():
    """spec §4.2: once → 仅 occur_time.date() == the_date 时命中."""
    occur = datetime(2026, 4, 22, tzinfo=_TZ)
    assert _reminder_matches_date(occur, "once", date(2026, 4, 22)) is True
    assert _reminder_matches_date(occur, "once", date(2026, 4, 23)) is False
    assert _reminder_matches_date(occur, "once", date(2027, 4, 22)) is False


def test_recurrence_none_treated_as_once():
    """null/missing recurrence → once 语义 (向后兼容老数据)."""
    occur = datetime(2026, 4, 22, tzinfo=_TZ)
    # 调用方走 `recurrence or "once"`, 这里直接验 once 路径
    assert _reminder_matches_date(occur, "once", date(2026, 4, 22)) is True
    assert _reminder_matches_date(occur, "once", date(2026, 4, 23)) is False


def test_recurrence_yearly_no_occur_returns_false():
    """yearly/monthly/weekly 没 occur_time → 跳过 (无法计算)."""
    assert _reminder_matches_date(None, "yearly", date(2026, 1, 1)) is False
    assert _reminder_matches_date(None, "monthly", date(2026, 1, 1)) is False
    assert _reminder_matches_date(None, "weekly", date(2026, 1, 1)) is False


# ── _extract_reminders_for_date 集成路径 ─────────────────────────────


@pytest.mark.asyncio
async def test_extract_reminders_yearly_past_occur_kept():
    """P1-A 核心: 1995 yearly 提醒在 2026 同月日仍命中."""
    rows = [_row(
        occur=datetime(1995, 3, 20, tzinfo=_TZ),
        recurrence="yearly",
        summary="每年体检",
    )]
    with patch(
        "app.services.proactive.special_dates.memory_repo.find_many",
        new_callable=AsyncMock, return_value=rows,
    ):
        result = await _extract_reminders_for_date("u1", "user", date(2026, 3, 20))
    assert result == ["每年体检"]


@pytest.mark.asyncio
async def test_extract_reminders_once_past_occur_skipped():
    """once 提醒过去时间不命中今天 — 防 spec §4.2 脏数据."""
    rows = [_row(
        occur=datetime(1995, 3, 20, tzinfo=_TZ),
        recurrence="once",
        summary="过去提醒",
    )]
    with patch(
        "app.services.proactive.special_dates.memory_repo.find_many",
        new_callable=AsyncMock, return_value=rows,
    ):
        result = await _extract_reminders_for_date("u1", "user", date(2026, 3, 20))
    assert result == []


@pytest.mark.asyncio
async def test_extract_reminders_legacy_null_recurrence_treated_as_once():
    """老数据 recurrence=NULL → once 语义."""
    today = date(2026, 4, 22)
    rows = [
        _row(occur=datetime(2026, 4, 22, tzinfo=_TZ), recurrence=None, summary="今日"),
        _row(occur=datetime(2025, 4, 22, tzinfo=_TZ), recurrence=None, summary="去年"),
    ]
    with patch(
        "app.services.proactive.special_dates.memory_repo.find_many",
        new_callable=AsyncMock, return_value=rows,
    ):
        result = await _extract_reminders_for_date("u1", "user", today)
    assert result == ["今日"]
