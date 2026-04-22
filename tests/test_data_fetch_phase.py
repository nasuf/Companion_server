"""data_fetch_phase 单测：覆盖 happy path + L3 awakening 触发条件。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.intent_dispatcher import IntentResult, IntentType


@pytest.mark.asyncio
async def test_fetch_parallel_context_happy_path():
    """关键字段从 9 个并行 fetch 中正确解包。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    classified = []
    for s in (0.7, 0.5, 0.3):
        m = SimpleNamespace(text=f"mem-{s}", similarity=s, importance=0.6, created_at=None)
        classified.append(m)

    with patch(
        "app.services.chat.data_fetch_phase.classify_memory_relevance",
        AsyncMock(return_value="medium"),
    ), patch(
        "app.services.chat.data_fetch_phase.hybrid_retrieve",
        AsyncMock(return_value={"memories": classified, "memory_strings": ["a"], "graph_context": None}),
    ), patch(
        "app.services.chat.data_fetch_phase.compute_ai_pad",
        AsyncMock(return_value={"pleasure": 0.1, "arousal": 0.5, "dominance": 0.5}),
    ), patch(
        "app.services.chat.data_fetch_phase.extract_emotion",
        AsyncMock(return_value={"pleasure": 0.2, "arousal": 0.4, "dominance": 0.5}),
    ), patch(
        "app.services.chat.data_fetch_phase.cache_summarizer",
        AsyncMock(return_value={"layer1": "..."}),
    ), patch(
        "app.services.chat.data_fetch_phase.get_latest_portrait",
        AsyncMock(return_value="user portrait"),
    ), patch(
        "app.services.chat.data_fetch_phase.get_cached_schedule",
        AsyncMock(return_value=[{"start": "09:00", "activity": "工作"}]),
    ), patch(
        "app.services.chat.data_fetch_phase.get_topic_intimacy",
        AsyncMock(return_value=65.0),
    ), patch(
        "app.services.chat.data_fetch_phase.get_current_status",
        return_value={"activity": "工作", "status": "busy"},
    ), patch(
        "app.services.chat.data_fetch_phase.format_schedule_context",
        return_value="(工作中)",
    ):
        ctx = await fetch_parallel_context(
            user_id="u1", agent_id="a1", workspace_id=None,
            user_message="嗨",
            messages_dicts=[{"role": "user", "content": "嗨"}],
            parsed_times=[],
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
        )

    assert ctx.memory_relevance == "medium"
    assert ctx.classified_memories is not None
    assert len(ctx.classified_memories) == 3
    # rerank 后按 display_score 降序
    scores = [m.display_score for m in ctx.classified_memories]
    assert scores == sorted(scores, reverse=True)
    assert ctx.emotion == {"pleasure": 0.1, "arousal": 0.5, "dominance": 0.5}
    assert ctx.user_emotion == {"pleasure": 0.2, "arousal": 0.4, "dominance": 0.5}
    assert ctx.portrait == "user portrait"
    assert ctx.topic_intimacy == 65.0
    assert ctx.ai_status == {"activity": "工作", "status": "busy"}
    assert ctx.schedule_context == "(工作中)"
    assert ctx.l3_memories == []
    assert ctx.l3_trigger_label == "无"


@pytest.mark.asyncio
async def test_fetch_parallel_context_skips_retrieval_on_weak():
    """relevance=weak 时跳过 retrieval 后处理，classified_memories=None。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    with patch(
        "app.services.chat.data_fetch_phase.classify_memory_relevance",
        AsyncMock(return_value="weak"),
    ), patch(
        "app.services.chat.data_fetch_phase.hybrid_retrieve",
        AsyncMock(return_value={"memories": ["should_not_use"], "memory_strings": [], "graph_context": None}),
    ), patch(
        "app.services.chat.data_fetch_phase.compute_ai_pad", AsyncMock(return_value={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}),
    ), patch(
        "app.services.chat.data_fetch_phase.extract_emotion", AsyncMock(return_value={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}),
    ), patch(
        "app.services.chat.data_fetch_phase.cache_summarizer", AsyncMock(return_value=None),
    ), patch(
        "app.services.chat.data_fetch_phase.get_latest_portrait", AsyncMock(return_value=None),
    ), patch(
        "app.services.chat.data_fetch_phase.get_cached_schedule", AsyncMock(return_value=None),
    ), patch(
        "app.services.chat.data_fetch_phase.get_topic_intimacy", AsyncMock(return_value=50.0),
    ):
        ctx = await fetch_parallel_context(
            user_id="u1", agent_id="a1", workspace_id=None,
            user_message="嗨",
            messages_dicts=[{"role": "user", "content": "嗨"}],
            parsed_times=[],
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
        )

    assert ctx.memory_relevance == "weak"
    assert ctx.classified_memories is None
    assert ctx.l3_memories == []


@pytest.mark.asyncio
async def test_fetch_parallel_context_l3_awakened_on_strong_relevance():
    """relevance=strong + L3 trigger 命中"请求更久" → 召回 L3 记忆。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    with patch(
        "app.services.chat.data_fetch_phase.classify_memory_relevance",
        AsyncMock(return_value="strong"),
    ), patch(
        "app.services.chat.data_fetch_phase.hybrid_retrieve",
        AsyncMock(return_value={"memories": [], "memory_strings": [], "graph_context": None}),
    ), patch(
        "app.services.chat.data_fetch_phase.compute_ai_pad", AsyncMock(return_value={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}),
    ), patch(
        "app.services.chat.data_fetch_phase.extract_emotion", AsyncMock(return_value={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}),
    ), patch(
        "app.services.chat.data_fetch_phase.cache_summarizer", AsyncMock(return_value=None),
    ), patch(
        "app.services.chat.data_fetch_phase.get_latest_portrait", AsyncMock(return_value=None),
    ), patch(
        "app.services.chat.data_fetch_phase.get_cached_schedule", AsyncMock(return_value=None),
    ), patch(
        "app.services.chat.data_fetch_phase.get_topic_intimacy", AsyncMock(return_value=50.0),
    ), patch(
        "app.services.chat.data_fetch_phase.search_l3_memories",
        AsyncMock(return_value=[{"content": "很久以前你说过喜欢下雨"}]),
    ):
        ctx = await fetch_parallel_context(
            user_id="u1", agent_id="a1", workspace_id=None,
            user_message="还记得我以前喜欢什么天气吗",
            messages_dicts=[{"role": "user", "content": "..."}],
            parsed_times=[],
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="请求更久"),
        )

    assert ctx.l3_trigger_label == "请求更久"
    assert ctx.l3_memories == ["很久以前你说过喜欢下雨"]
