"""AI-side extraction must not fossilize self-authored persona claims.

Production bug: the AI hallucinated a preference ("我喜欢浅紫色和奶白色") that
contradicted its profile. Chat-time AI-side extraction then pulled that claim as
a 偏好/审美爱好 self-memory, risking a fake persona fact that drifts further from
the real profile. The AI's stable persona (偏好 + 身份) is seeded once from the
creation profile; new such self-memories from the AI's own replies are dropped.
Episodic self-memory (生活/情绪 experiences) still flows.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.services.schedule_domain.time_service import _TZ


def _now() -> datetime:
    return datetime(2026, 7, 20, 14, 30, tzinfo=_TZ)


def _extraction(main_category: str, sub_category: str, summary: str):
    return {
        "memories": [{
            "summary": summary,
            "importance": 0.6,
            "type": "preference",
            "main_category": main_category,
            "sub_category": sub_category,
            "occur_time": None,
            "recurrence": None,
        }],
        "entities": [], "topics": [], "preferences": [],
    }


def _pipeline_patches(extraction, store):
    return (
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
        patch("app.services.memory.recording.pipeline.store_memory", new=store),
        patch("app.services.memory.recording.pipeline.record_entities_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_topics_for_memory",
              new_callable=AsyncMock),
        patch("app.services.memory.recording.pipeline.record_preferences_for_memory",
              new_callable=AsyncMock),
    )


async def _run(side, extraction, conversation):
    from app.services.memory.recording.pipeline import process_memory_pipeline
    calls: list[dict] = []

    async def _store(**kwargs):
        calls.append(kwargs)
        return f"mem-{len(calls)}"

    import contextlib
    with contextlib.ExitStack() as stack:
        for p in _pipeline_patches(extraction, _store):
            stack.enter_context(p)
        await process_memory_pipeline(
            user_id="u1", new_conversation=conversation,
            statement_time=_now(), side=side,
        )
    return calls


@pytest.mark.asyncio
async def test_ai_side_preference_claim_blocked():
    calls = await _run(
        "ai",
        _extraction("偏好", "审美爱好", "我喜欢浅紫色和奶白色"),
        "assistant: 我喜欢浅紫色和奶白色",
    )
    assert calls == []  # self-authored preference not persisted


@pytest.mark.asyncio
async def test_ai_side_identity_claim_blocked():
    calls = await _run(
        "ai",
        _extraction("身份", "年龄", "我今年30岁"),
        "assistant: 我今年30岁啦",
    )
    assert calls == []


@pytest.mark.asyncio
async def test_ai_side_episodic_memory_still_stored():
    """生活/情绪 experiences are NOT persona facts — they must still be stored."""
    calls = await _run(
        "ai",
        _extraction("生活", "工作", "今天处理了一个很难的客服投诉，最后解决了"),
        "assistant: 今天处理了一个很难的投诉，总算解决了",
    )
    assert len(calls) == 1
    assert calls[0]["main_category"] == "生活"


@pytest.mark.asyncio
async def test_user_side_preference_not_blocked():
    """The block is AI-side only — user preferences must still be recorded."""
    calls = await _run(
        "user",
        _extraction("偏好", "审美爱好", "用户喜欢浅紫色"),
        "user: 我喜欢浅紫色",
    )
    assert len(calls) == 1
    assert calls[0]["main_category"] == "偏好"
