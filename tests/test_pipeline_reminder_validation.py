"""Spec Part 5 §4.2 提醒记忆 occur_time 必须未来 — pipeline 校验单测.

P0-3: LLM 误把 "上周提醒过我" 归为提醒子类时, occur_time 是历史 →
pipeline 应降级到 sub_category="其他", 防特殊日期触发链路脏数据.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.services.schedule_domain.time_service import _TZ


def _now() -> datetime:
    return datetime(2026, 4, 22, 14, 30, tzinfo=_TZ)


def _mem_extraction(occur_time_iso: str | None, summary="提醒事项"):
    """构造 extract_memories 的返回 (mock 用)."""
    mem = {
        "summary": summary,
        "importance": 0.6,
        "type": "reminder",
        "main_category": "生活",
        "sub_category": "提醒",
        "occur_time": occur_time_iso,
    }
    return {"memories": [mem], "entities": [], "topics": [], "preferences": []}


@pytest.mark.asyncio
async def test_reminder_with_past_occur_time_demoted_to_other():
    """sub_category="提醒" + occur_time<now → 降级"其他"."""
    from app.services.memory.recording.pipeline import process_memory_pipeline

    past = (_now() - timedelta(days=7)).isoformat()
    captured: dict = {}

    async def fake_store(**kwargs):
        captured.update(kwargs)
        return "mem-1"

    with (
        patch(
            "app.services.memory.recording.pipeline.resolve_workspace_id",
            new_callable=AsyncMock, return_value="ws1",
        ),
        patch(
            "app.services.memory.recording.pipeline.should_extract_memory",
            return_value=True,
        ),
        patch(
            "app.services.memory.recording.pipeline.should_memorize",
            new_callable=AsyncMock, return_value=True,
        ),
        patch(
            "app.services.memory.recording.pipeline.extract_memories",
            new_callable=AsyncMock, return_value=_mem_extraction(past, "上周说过的事"),
        ),
        patch(
            "app.services.memory.recording.pipeline.has_explicit_time",
            return_value=False,  # rule engine 不参与, 用 LLM occur_time
        ),
        patch(
            "app.services.memory.recording.pipeline.store_memory",
            new=fake_store,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_entities_for_memory",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_topics_for_memory",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_preferences_for_memory",
            new_callable=AsyncMock,
        ),
    ):
        await process_memory_pipeline(
            user_id="u1",
            new_conversation="user: 上周说过的事",
            statement_time=_now(),
            side="user",
        )

    assert captured["sub_category"] == "其他", (
        f"过去 occur_time 的提醒应降级, 实际 sub={captured['sub_category']}"
    )


@pytest.mark.asyncio
async def test_reminder_with_no_occur_time_demoted_to_other():
    """sub_category="提醒" + occur_time=None → 降级"其他"."""
    from app.services.memory.recording.pipeline import process_memory_pipeline

    captured: dict = {}

    async def fake_store(**kwargs):
        captured.update(kwargs)
        return "mem-1"

    with (
        patch(
            "app.services.memory.recording.pipeline.resolve_workspace_id",
            new_callable=AsyncMock, return_value="ws1",
        ),
        patch(
            "app.services.memory.recording.pipeline.should_extract_memory",
            return_value=True,
        ),
        patch(
            "app.services.memory.recording.pipeline.should_memorize",
            new_callable=AsyncMock, return_value=True,
        ),
        patch(
            "app.services.memory.recording.pipeline.extract_memories",
            new_callable=AsyncMock, return_value=_mem_extraction(None),
        ),
        patch(
            "app.services.memory.recording.pipeline.has_explicit_time",
            return_value=False,
        ),
        patch(
            "app.services.memory.recording.pipeline.store_memory",
            new=fake_store,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_entities_for_memory",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_topics_for_memory",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_preferences_for_memory",
            new_callable=AsyncMock,
        ),
    ):
        await process_memory_pipeline(
            user_id="u1",
            new_conversation="user: 提醒我",
            statement_time=_now(),
            side="user",
        )

    assert captured["sub_category"] == "其他"


@pytest.mark.asyncio
async def test_reminder_demotion_logs_changelog_audit_trail():
    """P2-C: 降级后写 log_memory_changelog(reminder_past_time_demoted)
    供 admin grep 审计 LLM 抽取质量."""
    from app.services.memory.recording.pipeline import process_memory_pipeline

    past = (_now() - timedelta(days=7)).isoformat()
    changelog_calls: list[tuple] = []

    async def fake_changelog(*args, **kwargs):
        changelog_calls.append((args, kwargs))

    async def fake_store(**kwargs):
        return "mem-1"

    with (
        patch(
            "app.services.memory.recording.pipeline.resolve_workspace_id",
            new_callable=AsyncMock, return_value="ws1",
        ),
        patch(
            "app.services.memory.recording.pipeline.should_extract_memory",
            return_value=True,
        ),
        patch(
            "app.services.memory.recording.pipeline.should_memorize",
            new_callable=AsyncMock, return_value=True,
        ),
        patch(
            "app.services.memory.recording.pipeline.extract_memories",
            new_callable=AsyncMock, return_value=_mem_extraction(past, "上周事"),
        ),
        patch(
            "app.services.memory.recording.pipeline.has_explicit_time",
            return_value=False,
        ),
        patch(
            "app.services.memory.recording.pipeline.store_memory",
            new=fake_store,
        ),
        patch(
            "app.services.memory.recording.pipeline.log_memory_changelog",
            new=fake_changelog,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_entities_for_memory",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_topics_for_memory",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_preferences_for_memory",
            new_callable=AsyncMock,
        ),
    ):
        await process_memory_pipeline(
            user_id="u1",
            new_conversation="user: 上周事",
            statement_time=_now(),
            side="user",
        )

    # 应该至少有一条 reminder_past_time_demoted changelog
    demoted_calls = [
        c for c in changelog_calls
        if len(c[0]) >= 3 and c[0][2] == "reminder_past_time_demoted"
    ]
    assert demoted_calls, (
        f"应写 reminder_past_time_demoted changelog, 实际 calls={changelog_calls}"
    )


@pytest.mark.asyncio
async def test_reminder_with_future_occur_time_kept():
    """sub_category="提醒" + occur_time>now → 保留 (回归)."""
    from app.services.memory.recording.pipeline import process_memory_pipeline

    future = (_now() + timedelta(days=3)).isoformat()
    captured: dict = {}

    async def fake_store(**kwargs):
        captured.update(kwargs)
        return "mem-1"

    with (
        patch(
            "app.services.memory.recording.pipeline.resolve_workspace_id",
            new_callable=AsyncMock, return_value="ws1",
        ),
        patch(
            "app.services.memory.recording.pipeline.should_extract_memory",
            return_value=True,
        ),
        patch(
            "app.services.memory.recording.pipeline.should_memorize",
            new_callable=AsyncMock, return_value=True,
        ),
        patch(
            "app.services.memory.recording.pipeline.extract_memories",
            new_callable=AsyncMock, return_value=_mem_extraction(future, "明天面试"),
        ),
        patch(
            "app.services.memory.recording.pipeline.has_explicit_time",
            return_value=False,
        ),
        patch(
            "app.services.memory.recording.pipeline.store_memory",
            new=fake_store,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_entities_for_memory",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_topics_for_memory",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.recording.pipeline.record_preferences_for_memory",
            new_callable=AsyncMock,
        ),
    ):
        await process_memory_pipeline(
            user_id="u1",
            new_conversation="user: 提醒我明天面试",
            statement_time=_now(),
            side="user",
        )

    assert captured["sub_category"] == "提醒", (
        f"未来 occur_time 的提醒应保留, 实际 sub={captured['sub_category']}"
    )
