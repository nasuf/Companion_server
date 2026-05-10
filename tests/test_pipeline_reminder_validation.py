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


def _mem_extraction(occur_time_iso: str | None, summary="提醒事项", recurrence=None):
    """构造 extract_memories 的返回 (mock 用)."""
    mem = {
        "summary": summary,
        "importance": 0.6,
        "type": "reminder",
        "main_category": "生活",
        "sub_category": "提醒",
        "occur_time": occur_time_iso,
        "recurrence": recurrence,
    }
    return {"memories": [mem], "entities": [], "topics": [], "preferences": []}


@pytest.mark.asyncio
async def test_ai_side_skips_user_fact_acknowledgement_after_extraction():
    """即使 AI extraction 误抽了“我记住了用户事实”，pipeline 也不写入 A 库."""
    from app.services.memory.recording.pipeline import process_memory_pipeline

    extraction = {
        "memories": [{
            "summary": "我记住了用户的名字叫馒头，并觉得这个名字很可爱。",
            "importance": 0.8,
            "type": "life",
            "main_category": "生活",
            "sub_category": "人际",
            "occur_time": None,
            "recurrence": None,
        }],
        "entities": [], "topics": [], "preferences": [],
    }

    with (
        patch("app.services.memory.recording.pipeline.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws1"),
        patch("app.services.memory.recording.pipeline.should_extract_memory",
              return_value=True),
        patch("app.services.memory.recording.pipeline.should_memorize",
              new_callable=AsyncMock, return_value=True),
        patch("app.services.memory.recording.pipeline.extract_memories",
              new_callable=AsyncMock, return_value=extraction),
        patch("app.services.memory.recording.pipeline.has_explicit_time",
              return_value=False),
        patch("app.services.memory.recording.pipeline.store_memory",
              new_callable=AsyncMock) as mock_store,
    ):
        stored = await process_memory_pipeline(
            user_id="u1",
            new_conversation="assistant: 好的，馒头。这个名字很可爱，我记住了。",
            statement_time=_now(),
            side="ai",
        )

    assert stored == []
    mock_store.assert_not_called()


@pytest.mark.asyncio
async def test_ai_side_skips_uncertain_self_memory_after_extraction():
    """AI 侧不把记不清/猜测式自述固化成稳定自我记忆。"""
    from app.services.memory.recording.pipeline import process_memory_pipeline

    extraction = {
        "memories": [{
            "summary": "我高中是在本地一所普通学校读的，具体名字记不清了。",
            "importance": 0.9,
            "type": "identity",
            "main_category": "身份",
            "sub_category": "教育背景",
            "occur_time": None,
            "recurrence": None,
        }],
        "entities": [], "topics": [], "preferences": [],
    }

    with (
        patch("app.services.memory.recording.pipeline.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws1"),
        patch("app.services.memory.recording.pipeline.should_extract_memory",
              return_value=True),
        patch("app.services.memory.recording.pipeline.should_memorize",
              new_callable=AsyncMock, return_value=True),
        patch("app.services.memory.recording.pipeline.extract_memories",
              new_callable=AsyncMock, return_value=extraction),
        patch("app.services.memory.recording.pipeline.has_explicit_time",
              return_value=False),
        patch("app.services.memory.recording.pipeline.store_memory",
              new_callable=AsyncMock) as mock_store,
    ):
        stored = await process_memory_pipeline(
            user_id="u1",
            new_conversation=(
                "assistant: 我高中在本地一所普通学校读的，"
                "不过具体名字记不太清了。"
            ),
            statement_time=_now(),
            side="ai",
        )

    assert stored == []
    mock_store.assert_not_called()


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


@pytest.mark.asyncio
async def test_reminder_yearly_recurrence_kept_with_past_occur():
    """spec §4.2: yearly 提醒 occur_time 是历史 (e.g. 1995 生日) 不应降级,
    后续按 (month, day) 自动重复触发."""
    from app.services.memory.recording.pipeline import process_memory_pipeline

    past = (_now() - timedelta(days=365 * 30)).isoformat()
    captured: dict = {}

    async def fake_store(**kwargs):
        captured.update(kwargs)
        return "mem-1"

    with (
        patch("app.services.memory.recording.pipeline.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws1"),
        patch("app.services.memory.recording.pipeline.should_extract_memory",
              return_value=True),
        patch("app.services.memory.recording.pipeline.should_memorize",
              new_callable=AsyncMock, return_value=True),
        patch("app.services.memory.recording.pipeline.extract_memories",
              new_callable=AsyncMock,
              return_value=_mem_extraction(past, "每年体检", recurrence="yearly")),
        patch("app.services.memory.recording.pipeline.has_explicit_time",
              return_value=False),
        patch("app.services.memory.recording.pipeline.store_memory", new=fake_store),
        patch("app.services.memory.recording.pipeline.record_entities_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_topics_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_preferences_for_memory",
              new_callable=AsyncMock),
    ):
        await process_memory_pipeline(
            user_id="u1", new_conversation="user: 每年体检",
            statement_time=_now(), side="user",
        )

    assert captured["sub_category"] == "提醒", (
        f"yearly 提醒不应降级, 实际 sub={captured['sub_category']}"
    )
    assert captured["recurrence"] == "yearly"


@pytest.mark.asyncio
async def test_reminder_recurrence_extracted_from_llm_monthly():
    """LLM 输出 recurrence='monthly' → store_memory 收到 monthly."""
    from app.services.memory.recording.pipeline import process_memory_pipeline

    future = (_now() + timedelta(days=3)).isoformat()
    captured: dict = {}

    async def fake_store(**kwargs):
        captured.update(kwargs)
        return "mem-1"

    with (
        patch("app.services.memory.recording.pipeline.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws1"),
        patch("app.services.memory.recording.pipeline.should_extract_memory",
              return_value=True),
        patch("app.services.memory.recording.pipeline.should_memorize",
              new_callable=AsyncMock, return_value=True),
        patch("app.services.memory.recording.pipeline.extract_memories",
              new_callable=AsyncMock,
              return_value=_mem_extraction(future, "每月房租", recurrence="monthly")),
        patch("app.services.memory.recording.pipeline.has_explicit_time",
              return_value=False),
        patch("app.services.memory.recording.pipeline.store_memory", new=fake_store),
        patch("app.services.memory.recording.pipeline.record_entities_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_topics_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_preferences_for_memory",
              new_callable=AsyncMock),
    ):
        await process_memory_pipeline(
            user_id="u1", new_conversation="user: 每月 1 号提醒交房租",
            statement_time=_now(), side="user",
        )

    assert captured["recurrence"] == "monthly"


@pytest.mark.asyncio
async def test_reminder_subcategory_alias_resolved_before_recurrence():
    """LLM 输出 sub_category 别名 (e.g. "备忘") → 应识别为提醒并保留 recurrence.

    P0 回归: 之前 pipeline 比对原始 sub_category=="提醒", "备忘"被跳过 →
    recurrence=None, 后续 store_memory 把 alias 解析后的"提醒"写入但 recurrence
    丢失, yearly/monthly 提醒一次性化.
    """
    from app.services.memory.recording.pipeline import process_memory_pipeline

    extraction = {
        "memories": [{
            "summary": "每月 1 号交房租",
            "importance": 0.6,
            "type": "life",
            "main_category": "生活",
            "sub_category": "备忘",  # alias of "提醒"
            "occur_time": (_now() + timedelta(days=3)).isoformat(),
            "recurrence": "monthly",
        }],
        "entities": [], "topics": [], "preferences": [],
    }
    captured: dict = {}

    async def fake_store(**kwargs):
        captured.update(kwargs)
        return "mem-1"

    with (
        patch("app.services.memory.recording.pipeline.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws1"),
        patch("app.services.memory.recording.pipeline.should_extract_memory",
              return_value=True),
        patch("app.services.memory.recording.pipeline.should_memorize",
              new_callable=AsyncMock, return_value=True),
        patch("app.services.memory.recording.pipeline.extract_memories",
              new_callable=AsyncMock, return_value=extraction),
        patch("app.services.memory.recording.pipeline.has_explicit_time",
              return_value=False),
        patch("app.services.memory.recording.pipeline.store_memory", new=fake_store),
        patch("app.services.memory.recording.pipeline.record_entities_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_topics_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_preferences_for_memory",
              new_callable=AsyncMock),
    ):
        await process_memory_pipeline(
            user_id="u1", new_conversation="user: 备忘 1 号交租",
            statement_time=_now(), side="user",
        )

    # alias 解析后 sub_category 应为 "提醒"
    assert captured["sub_category"] == "提醒"
    # recurrence 应保留 (alias 不导致丢失)
    assert captured["recurrence"] == "monthly"


@pytest.mark.asyncio
async def test_reminder_non_reminder_subcategory_no_recurrence():
    """非提醒子类传 recurrence=None (避免脏数据)."""
    from app.services.memory.recording.pipeline import process_memory_pipeline

    captured: dict = {}

    async def fake_store(**kwargs):
        captured.update(kwargs)
        return "mem-1"

    # 模拟 LLM 错把生日 (身份/生日) 标了 recurrence=yearly
    extraction = {
        "memories": [{
            "summary": "用户生日 3-20",
            "importance": 0.9,
            "type": "identity",
            "main_category": "身份",
            "sub_category": "生日",
            "occur_time": "1995-03-20T00:00:00",
            "recurrence": "yearly",  # 错的, pipeline 应清空
        }],
        "entities": [], "topics": [], "preferences": [],
    }

    with (
        patch("app.services.memory.recording.pipeline.resolve_workspace_id",
              new_callable=AsyncMock, return_value="ws1"),
        patch("app.services.memory.recording.pipeline.should_extract_memory",
              return_value=True),
        patch("app.services.memory.recording.pipeline.should_memorize",
              new_callable=AsyncMock, return_value=True),
        patch("app.services.memory.recording.pipeline.extract_memories",
              new_callable=AsyncMock, return_value=extraction),
        patch("app.services.memory.recording.pipeline.has_explicit_time",
              return_value=False),
        patch("app.services.memory.recording.pipeline.store_memory", new=fake_store),
        patch("app.services.memory.recording.pipeline.record_entities_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_topics_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_preferences_for_memory",
              new_callable=AsyncMock),
    ):
        await process_memory_pipeline(
            user_id="u1", new_conversation="user: 我生日 3-20",
            statement_time=_now(), side="user",
        )

    assert captured["recurrence"] is None, (
        f"非提醒子类 recurrence 应清空, 实际 {captured['recurrence']}"
    )
