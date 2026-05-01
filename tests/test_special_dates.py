"""Spec Part 5 §4.3 + §5 特殊日期单测.

覆盖: 生日多格式提取 (P0-4) / 起床后空闲触发 / important_date 命名 (P1-5).
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.proactive.special_dates import (
    Occasion,
    _extract_birthday_from_memories,
    collect_special_dates_today,
    find_first_idle_after_wakeup,
)
from app.services.schedule_domain.time_service import _TZ


def _mem(occur=None, content="", summary=""):
    return SimpleNamespace(
        occurTime=occur, content=content, summary=summary,
        mainCategory="身份", subCategory="生日",
    )


# ── P0-4: 生日多格式提取 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_birthday_extracted_from_occur_time():
    """parser 已写过 occur_time → 直接读字段, 最权威."""
    rows = [_mem(occur=datetime(1995, 3, 20, tzinfo=_TZ), content="生日记录", summary="")]
    with patch(
        "app.services.proactive.special_dates.memory_repo.find_many",
        new_callable=AsyncMock, return_value=rows,
    ):
        md = await _extract_birthday_from_memories("u1", "user")
    assert md == (3, 20)


@pytest.mark.asyncio
async def test_birthday_extracted_from_iso_string():
    """occur_time 没有时, ISO 串 "1995-03-20" 兜底."""
    rows = [_mem(content="我的生日是 1995-03-20", summary="")]
    with patch(
        "app.services.proactive.special_dates.memory_repo.find_many",
        new_callable=AsyncMock, return_value=rows,
    ):
        md = await _extract_birthday_from_memories("u1", "user")
    assert md == (3, 20)


@pytest.mark.asyncio
async def test_birthday_extracted_from_chinese_format():
    """occur_time 没有时, 中文格式 "3月20日生" 兜底 (现有行为不退化)."""
    rows = [_mem(content="3月20日生", summary="")]
    with patch(
        "app.services.proactive.special_dates.memory_repo.find_many",
        new_callable=AsyncMock, return_value=rows,
    ):
        md = await _extract_birthday_from_memories("u1", "user")
    assert md == (3, 20)


@pytest.mark.asyncio
async def test_birthday_returns_none_when_no_match():
    """三种格式都没命中 → None."""
    rows = [_mem(content="我喜欢猫", summary="")]
    with patch(
        "app.services.proactive.special_dates.memory_repo.find_many",
        new_callable=AsyncMock, return_value=rows,
    ):
        md = await _extract_birthday_from_memories("u1", "user")
    assert md is None


# ── 起床后第一个空闲 ─────────────────────────────────────────────


def test_find_first_idle_with_wake_and_idle_slot():
    """起床事件 + 后续 idle slot, 返回正确 datetime."""
    schedule = [
        {"start": "07:00", "end": "07:30", "event": "起床", "type": "wake", "status": "busy"},
        {"start": "07:30", "end": "08:30", "event": "早餐", "type": "leisure", "status": "idle"},
        {"start": "08:30", "end": "12:00", "event": "工作", "type": "work", "status": "busy"},
    ]
    d = date(2026, 4, 22)
    result = find_first_idle_after_wakeup(schedule, d)
    assert result.date() == d
    assert result.hour == 7 and result.minute == 30


def test_find_first_idle_no_wake_event_falls_back_8am():
    """作息表无起床事件 → fallback 08:00."""
    schedule = [
        {"start": "09:00", "end": "12:00", "event": "工作", "type": "work", "status": "busy"},
    ]
    d = date(2026, 4, 22)
    result = find_first_idle_after_wakeup(schedule, d)
    assert result.hour == 8 and result.minute == 0


def test_find_first_idle_all_busy_uses_wake_end():
    """全 busy 无 idle slot → 用起床结束时刻."""
    schedule = [
        {"start": "07:00", "end": "07:30", "event": "起床", "type": "wake", "status": "busy"},
        {"start": "07:30", "end": "12:00", "event": "工作", "type": "work", "status": "busy"},
    ]
    d = date(2026, 4, 22)
    result = find_first_idle_after_wakeup(schedule, d)
    assert result.hour == 7 and result.minute == 30


def test_find_first_idle_empty_schedule_falls_back_8am():
    """作息表为空 → fallback 08:00."""
    d = date(2026, 4, 22)
    result = find_first_idle_after_wakeup([], d)
    assert result.hour == 8 and result.minute == 0


# ── P1-5: important_date 命名清晰化 ──────────────────────────────


@pytest.mark.asyncio
async def test_important_date_does_not_solo_trigger():
    """spec §5.1: 重要日期不独立触发. 仅当日"考试"无其他命中 → occasions 空."""
    d = date(2026, 4, 22)
    # 模拟: 无节日, 无生日, 无提醒; 仅 "今日重要事项: 面试"
    important_rows = [SimpleNamespace(
        occurTime=datetime(2026, 4, 22, 14, 0, tzinfo=_TZ),
        content="面试", summary="今日有面试",
        mainCategory="生活", subCategory="纪念",
    )]
    with (
        patch("app.services.schedule_domain.time_service.is_holiday", return_value=None),
        patch(
            "app.services.proactive.special_dates._extract_birthday_from_memories",
            new_callable=AsyncMock, return_value=None,
        ),
        patch(
            "app.services.proactive.special_dates._extract_reminders_for_date",
            new_callable=AsyncMock, return_value=[],
        ),
        patch(
            "app.services.proactive.special_dates._extract_important_dates_for_date",
            new_callable=AsyncMock, return_value=["面试"],
        ),
    ):
        occasions = await collect_special_dates_today(user_id="u1", the_date=d)
    assert occasions == [], (
        f"spec §5.1 重要日期不独立触发, 但返回了: {occasions}"
    )


@pytest.mark.asyncio
async def test_important_date_appended_when_birthday_hits():
    """生日命中 + 当日重要日期 → occasions 含 birthday + important_date."""
    d = date(2026, 4, 22)
    with (
        patch("app.services.schedule_domain.time_service.is_holiday", return_value=None),
        patch(
            "app.services.proactive.special_dates._extract_birthday_from_memories",
            new_callable=AsyncMock,
            side_effect=lambda uid, owner: (4, 22) if owner == "user" else None,
        ),
        patch(
            "app.services.proactive.special_dates._extract_reminders_for_date",
            new_callable=AsyncMock, return_value=[],
        ),
        patch(
            "app.services.proactive.special_dates._extract_important_dates_for_date",
            new_callable=AsyncMock,
            side_effect=lambda uid, owner, dd: ["面试"] if owner == "user" else [],
        ),
    ):
        occasions = await collect_special_dates_today(user_id="u1", the_date=d)

    types = [o.type for o in occasions]
    assert "birthday" in types
    assert "important_date" in types, (
        f"重要日期应附加, 但返回 types={types}"
    )
    # P1-5 防回归: 不能误归到 reminder
    assert types.count("reminder") == 0


def test_occasion_type_includes_important_date():
    """OccasionType Literal 含 important_date (P1-5)."""
    o = Occasion(type="important_date", name="测试", owner="user")
    assert o.type == "important_date"
