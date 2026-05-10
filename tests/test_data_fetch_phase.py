"""data_fetch_phase 单测：覆盖 happy path + L3 awakening 触发条件。"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.intent_dispatcher import IntentResult, IntentType


_NEUTRAL_EMOTION = {"emotion": "中性", "intensity": 0, "confidence": 0.0}
_BUSY_STATUS = {"activity": "工作", "status": "busy"}


def _patch_data_fetch(**overrides) -> ExitStack:
    """Patch data_fetch_phase 中的所有外部依赖；overrides 覆盖默认 mock 值。

    返回的 ExitStack 用作 `with` 上下文，进入时所有 patch 生效。
    """
    defaults = {
        "classify_memory_relevance": AsyncMock(return_value="medium"),
        "hybrid_retrieve": AsyncMock(return_value={"memories": [], "memory_strings": [], "graph_context": None}),
        "analyze_user_emotion": AsyncMock(return_value=dict(_NEUTRAL_EMOTION)),
        "get_latest_portrait": AsyncMock(return_value=None),
        "get_cached_schedule": AsyncMock(return_value=None),
        "get_topic_intimacy": AsyncMock(return_value=50.0),
    }
    defaults.update(overrides)

    stack = ExitStack()
    for name, mock in defaults.items():
        stack.enter_context(patch(f"app.services.chat.data_fetch_phase.{name}", mock))
    return stack


_DEFAULT_CALL = dict(
    user_id="u1", agent_id="a1", workspace_id=None,
    parsed_times=[],
    detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
)


def test_fast_weak_relevance_gate_is_conservative():
    """语气词可跳过; 省略追问/回忆/危机线索不能被规则误挡。"""
    from app.services.chat.data_fetch_phase import (
        _apply_memory_relevance_floor,
        _should_fast_weak_relevance,
    )

    assert _should_fast_weak_relevance("哈哈哈") is True
    assert _should_fast_weak_relevance("哈哈哈！") is True
    assert _should_fast_weak_relevance("嗯嗯") is True
    assert _should_fast_weak_relevance("ok!") is True
    assert _should_fast_weak_relevance("🙂🙂") is True

    assert _should_fast_weak_relevance("妈妈呢") is False
    assert _should_fast_weak_relevance("颜色呢") is False
    assert _should_fast_weak_relevance("还记得去年那次吗") is False
    assert _should_fast_weak_relevance("不是") is False

    for message in (
        "你知道王家卫吗",
        "你听过黑胶唱片吗",
        "你看过重庆森林吗",
        "你喜欢拿铁吗",
        "你觉得上海怎么样",
        "你多大了",
        "你叫什么名字",
        "你是做什么的",
        "你大学学什么专业",
    ):
        assert _should_fast_weak_relevance(message) is False
        assert _apply_memory_relevance_floor("weak", message) == "medium"

    assert _apply_memory_relevance_floor("weak", "你知道吗") == "weak"


@pytest.mark.asyncio
async def test_fetch_parallel_context_happy_path():
    """关键字段从并行 fetch 中正确解包。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    classified = [
        SimpleNamespace(text=f"mem-{s}", similarity=s, importance=0.6, created_at=None)
        for s in (0.7, 0.5, 0.3)
    ]

    with _patch_data_fetch(
        hybrid_retrieve=AsyncMock(return_value={"memories": classified, "memory_strings": ["a"], "graph_context": None}),
        analyze_user_emotion=AsyncMock(return_value={"emotion": "焦虑", "intensity": 62, "confidence": 0.8}),
        get_latest_portrait=AsyncMock(return_value="user portrait"),
        get_cached_schedule=AsyncMock(return_value=[{"start": "09:00", "activity": "工作"}]),
        get_topic_intimacy=AsyncMock(return_value=65.0),
    ), patch(
        "app.services.chat.data_fetch_phase.get_current_status", return_value=_BUSY_STATUS,
    ), patch(
        "app.services.chat.data_fetch_phase.format_schedule_context", return_value="(工作中)",
    ):
        ctx = await fetch_parallel_context(
            user_message="最近工作有点忙",
            messages_dicts=[{"role": "user", "content": "最近工作有点忙"}],
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
            **_DEFAULT_CALL,
        )

    assert ctx.memory_relevance == "medium"
    assert ctx.classified_memories is not None
    assert len(ctx.classified_memories) == 3
    scores = [m.display_score for m in ctx.classified_memories]
    assert scores == sorted(scores, reverse=True)  # rerank 后按 display_score 降序
    assert ctx.user_emotion == {"emotion": "焦虑", "intensity": 62, "confidence": 0.8}
    assert ctx.portrait == "user portrait"
    assert ctx.topic_intimacy == 65.0
    assert ctx.ai_status == _BUSY_STATUS
    assert ctx.schedule_context == "(工作中)"
    assert ctx.l3_memories == []
    assert ctx.l3_trigger_label == "无"


@pytest.mark.asyncio
async def test_fetch_parallel_context_skips_retrieval_on_weak():
    """relevance=weak 时不调用 hybrid retrieval，classified_memories=None。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    retrieval = AsyncMock(return_value={
        "memories": ["should_not_use"],
        "memory_strings": [],
        "graph_context": None,
    })
    with _patch_data_fetch(
        classify_memory_relevance=AsyncMock(return_value="weak"),
        hybrid_retrieve=retrieval,
    ):
        ctx = await fetch_parallel_context(
            user_message="今天随便聊点别的吧",
            messages_dicts=[{"role": "user", "content": "今天随便聊点别的吧"}],
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
            **_DEFAULT_CALL,
        )

    assert ctx.memory_relevance == "weak"
    assert ctx.classified_memories is None
    assert ctx.l3_memories == []
    retrieval.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_parallel_context_upgrades_ai_relation_query_from_weak():
    """询问 AI 与具体对象的关系时, weak 结果也要升到 medium 并召回 A/B 库。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    retrieval = AsyncMock(return_value={
        "memories": [],
        "memory_strings": [],
        "graph_context": None,
    })
    with _patch_data_fetch(
        classify_memory_relevance=AsyncMock(return_value="weak"),
        hybrid_retrieve=retrieval,
    ):
        ctx = await fetch_parallel_context(
            user_message="你知道某个独立音乐人吗",
            messages_dicts=[{"role": "user", "content": "你知道某个独立音乐人吗"}],
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
            **_DEFAULT_CALL,
        )

    assert ctx.memory_relevance == "medium"
    retrieval.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_parallel_context_upgrades_ai_profile_query_and_expands_retrieval():
    """询问 AI 稳态资料时, weak 也要召回 AI 自己资料和用户同类资料。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    retrieval = AsyncMock(return_value={
        "memories": [],
        "memory_strings": [],
        "graph_context": None,
    })
    with _patch_data_fetch(
        classify_memory_relevance=AsyncMock(return_value="weak"),
        hybrid_retrieve=retrieval,
    ):
        ctx = await fetch_parallel_context(
            user_message="你多大了？",
            messages_dicts=[{"role": "user", "content": "你多大了？"}],
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
            **_DEFAULT_CALL,
        )

    assert ctx.memory_relevance == "medium"
    retrieval.assert_awaited_once()
    _, _, kwargs = retrieval.mock_calls[0]
    assert "AI 年龄" in kwargs["enhanced_query"]
    assert "用户 年龄" in kwargs["enhanced_query"]


@pytest.mark.asyncio
async def test_fetch_parallel_context_preserves_profile_query_anchors():
    """AI profile 增强检索必须保留原始具体锚点, 不能只传泛化资料词。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    retrieval = AsyncMock(return_value={
        "memories": [],
        "memory_strings": [],
        "graph_context": None,
    })
    with _patch_data_fetch(
        classify_memory_relevance=AsyncMock(return_value="weak"),
        hybrid_retrieve=retrieval,
    ):
        await fetch_parallel_context(
            user_message="你还记得自己是哪个学校读的高中？",
            messages_dicts=[{"role": "user", "content": "你还记得自己是哪个学校读的高中？"}],
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
            **_DEFAULT_CALL,
        )

    retrieval.assert_awaited_once()
    _, _, kwargs = retrieval.mock_calls[0]
    enhanced_query = kwargs["enhanced_query"]
    assert enhanced_query.startswith("你还记得自己是哪个学校读的高中？")
    assert "高中" in enhanced_query
    assert "AI 教育背景" in enhanced_query
    assert "用户 教育背景" in enhanced_query


@pytest.mark.asyncio
async def test_fetch_parallel_context_fast_weak_skips_relevance_and_retrieval():
    """明显无记忆价值的短消息不应调用 relevance LLM / hybrid retrieval。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    relevance = AsyncMock(return_value="medium")
    retrieval = AsyncMock(return_value={
        "memories": ["should_not_use"],
        "memory_strings": [],
        "graph_context": None,
    })

    with _patch_data_fetch(
        classify_memory_relevance=relevance,
        hybrid_retrieve=retrieval,
    ):
        ctx = await fetch_parallel_context(
            user_message="哈哈哈",
            messages_dicts=[{"role": "user", "content": "哈哈哈"}],
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
            **_DEFAULT_CALL,
        )

    assert ctx.memory_relevance == "weak"
    assert ctx.classified_memories is None
    relevance.assert_not_awaited()
    retrieval.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_parallel_context_fast_weak_preserves_entity_followup():
    """短实体追问仍需 LLM 解上下文/召回, 不能被 fast weak gate 吞掉。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    relevance = AsyncMock(return_value="medium")
    retrieval = AsyncMock(return_value={
        "memories": [],
        "memory_strings": [],
        "graph_context": None,
    })

    with _patch_data_fetch(
        classify_memory_relevance=relevance,
        hybrid_retrieve=retrieval,
    ):
        ctx = await fetch_parallel_context(
            user_message="妈妈呢",
            messages_dicts=[{"role": "user", "content": "妈妈最近怎么样"}],
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
            **_DEFAULT_CALL,
        )

    assert ctx.memory_relevance == "medium"
    relevance.assert_awaited_once()
    retrieval.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_parallel_context_time_memories_use_workspace_scope():
    """显式时间记忆段必须限制在当前 conversation workspace 内。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    now = datetime.now(timezone.utc)
    parsed_time = SimpleNamespace(
        is_future=False,
        start=now - timedelta(days=7),
        end=now,
    )
    search_mock = AsyncMock(return_value=[{
        "id": "m1",
        "summary": "当前 workspace 的时间记忆",
        "content": "当前 workspace 的时间记忆",
    }])

    with _patch_data_fetch(), patch(
        "app.services.memory.retrieval.vector_search.search_by_time_range",
        search_mock,
    ):
        ctx = await fetch_parallel_context(
            user_message="上周那件事",
            messages_dicts=[{"role": "user", "content": "上周那件事"}],
            parsed_times=[parsed_time],
            workspace_id="ws-current",
            user_id="u1",
            agent_id="a1",
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
        )

    search_mock.assert_awaited_once_with(
        "u1",
        parsed_time.start,
        parsed_time.end,
        limit=5,
        workspace_id="ws-current",
    )
    assert ctx.time_memories == ["当前 workspace 的时间记忆"]


@pytest.mark.asyncio
async def test_fetch_parallel_context_l3_awakened_on_strong_relevance():
    """relevance=strong + L3 trigger 命中"请求更久" → 召回 L3 记忆。"""
    from app.services.chat.data_fetch_phase import fetch_parallel_context

    with _patch_data_fetch(
        classify_memory_relevance=AsyncMock(return_value="strong"),
    ), patch(
        "app.services.chat.data_fetch_phase.search_l3_memories",
        AsyncMock(return_value=[{"content": "很久以前你说过喜欢下雨"}]),
    ):
        ctx = await fetch_parallel_context(
            user_message="还记得我以前喜欢什么天气吗",
            messages_dicts=[{"role": "user", "content": "..."}],
            l3_trigger_classify_fn=AsyncMock(return_value="请求更久"),
            **_DEFAULT_CALL,
        )

    assert ctx.l3_trigger_label == "请求更久"
    assert ctx.l3_memories == ["很久以前你说过喜欢下雨"]


@pytest.mark.asyncio
async def test_l3_awakening_uses_trigger_retrieval_query_when_enhanced_query_empty():
    """L3 判定产出的 retrieval_query 应接到同一次 L3 检索。"""
    from app.services.chat.data_fetch_phase import maybe_awaken_l3

    search = AsyncMock(return_value=[{"content": "第一次见面时用户说了你好"}])
    trigger = AsyncMock(return_value=SimpleNamespace(
        label="请求更久",
        retrieval_query="用户第一次见面时说的话",
    ))
    with patch("app.services.chat.data_fetch_phase.search_l3_memories", search):
        memories, label = await maybe_awaken_l3(
            user_message="就是第一次说的话",
            user_id="u1",
            workspace_id="ws1",
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            memory_relevance="strong",
            l3_trigger_classify_fn=trigger,
            enhanced_query="",
            l1_l2_count=1,
            recent_context="用户: 我想你回想一下，第一次见面我说了啥",
        )

    assert label == "请求更久"
    assert memories == ["第一次见面时用户说了你好"]
    trigger.assert_awaited_once_with(
        "就是第一次说的话",
        "用户: 我想你回想一下，第一次见面我说了啥",
    )
    search.assert_awaited_once_with(
        "用户第一次见面时说的话",
        "u1",
        workspace_id="ws1",
    )


@pytest.mark.asyncio
async def test_l3_intent_does_not_awaken_when_trigger_says_none():
    """统一意图误判 L3 时，专门 trigger 返回"无"必须阻止 L3 召回。"""
    from app.services.chat.data_fetch_phase import maybe_awaken_l3

    search = AsyncMock(return_value=[{"content": "不该被召回"}])
    trigger = AsyncMock(return_value="无")
    with patch("app.services.chat.data_fetch_phase.search_l3_memories", search):
        memories, label = await maybe_awaken_l3(
            user_message="你记得我上次和你说的那家书店吗 我五一准备去",
            user_id="u1",
            workspace_id="ws1",
            detected_intent=IntentResult(intent=IntentType.L3_RECALL, confidence=0.75),
            memory_relevance="strong",
            l3_trigger_classify_fn=trigger,
            enhanced_query="用户上次说过的书店",
            l1_l2_count=5,
            recent_context="",
        )

    assert label == "无"
    assert memories == []
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_l3_intent_awakens_when_trigger_confirms_old_recall():
    """明确久远记忆请求仍可通过 L3 intent + trigger 召回。"""
    from app.services.chat.data_fetch_phase import maybe_awaken_l3

    search = AsyncMock(return_value=[{"content": "半年前你说过想去独立书店"}])
    trigger = AsyncMock(return_value=SimpleNamespace(
        label="请求更久",
        retrieval_query="半年前用户说过的书店",
    ))
    with patch("app.services.chat.data_fetch_phase.search_l3_memories", search):
        memories, label = await maybe_awaken_l3(
            user_message="你还记得半年前我说的那家书店吗",
            user_id="u1",
            workspace_id="ws1",
            detected_intent=IntentResult(intent=IntentType.L3_RECALL, confidence=0.75),
            memory_relevance="strong",
            l3_trigger_classify_fn=trigger,
            enhanced_query="",
            l1_l2_count=0,
            recent_context="",
        )

    assert label == "请求更久"
    assert memories == ["半年前你说过想去独立书店"]
    search.assert_awaited_once_with(
        "半年前用户说过的书店",
        "u1",
        workspace_id="ws1",
    )


@pytest.mark.asyncio
async def test_l3_sparse_fallback_runs_for_medium_recall_with_few_l1_l2():
    """medium + 明确回忆线索 + L1/L2 稀疏时, 应补搜 L3。"""
    from app.services.chat.data_fetch_phase import maybe_awaken_l3

    search = AsyncMock(return_value=[{"content": "去年那次旅行你很开心"}])
    with patch("app.services.chat.data_fetch_phase.search_l3_memories", search):
        memories, label = await maybe_awaken_l3(
            user_message="还记得去年那次旅行吗",
            user_id="u1",
            workspace_id="ws1",
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            memory_relevance="medium",
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
            l1_l2_count=1,
        )

    assert label == "稀疏补召"
    assert memories == ["去年那次旅行你很开心"]
    search.assert_awaited_once()


@pytest.mark.asyncio
async def test_l3_sparse_fallback_skips_medium_without_recall_hint():
    """medium + L1/L2 稀疏但没有回忆线索时, 不应随便唤醒 L3。"""
    from app.services.chat.data_fetch_phase import maybe_awaken_l3

    search = AsyncMock(return_value=[{"content": "不该出现"}])
    trigger = AsyncMock(return_value="无")
    with patch("app.services.chat.data_fetch_phase.search_l3_memories", search):
        memories, label = await maybe_awaken_l3(
            user_message="今天好累啊",
            user_id="u1",
            workspace_id="ws1",
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            memory_relevance="medium",
            l3_trigger_classify_fn=trigger,
            l1_l2_count=1,
        )

    assert label == "无"
    assert memories == []
    search.assert_not_awaited()
    trigger.assert_not_awaited()
