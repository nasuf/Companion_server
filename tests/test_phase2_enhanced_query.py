"""Phase 2.4: enhanced_query 多轮上下文增强检索 集成测试.

历史 bug: relevance LLM 已经看 context 解了"那他怎样" → "妈妈情况", 但 hybrid
retrieval 还是用原文"那他怎样" 做 embedding → 召回噪声/空.

修复后流程:
1. relevance 输出 {"level": "强", "enhanced_query": "用户的妈妈现状"}
2. data_fetch_phase 拿到 enhanced_query, 调 _do_retrieval(..., enhanced_query=...)
3. hybrid_retrieve 用 enhanced_query 做 vector search (effective_query)
4. 时间解析仍用原 message (时间词通常在原话)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_hybrid_retrieve_uses_enhanced_query_for_embedding():
    """enhanced_query 非空 → search_similar 用 enhanced_query 做 embedding."""
    from app.services.memory.retrieval import hybrid

    captured_query = []
    async def _capture_search(query, user_id, tiers, **kwargs):
        captured_query.append(query)
        return []

    with (
        patch.object(hybrid, "search_similar_tiers", side_effect=_capture_search),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval",
                     new_callable=AsyncMock, return_value=None),
        patch.object(hybrid, "cache_set_retrieval",
                     new_callable=AsyncMock),
    ):
        await hybrid.hybrid_retrieve(
            message="那他怎样了?",
            user_id="u1",
            workspace_id="w1",
            enhanced_query="用户的妈妈最近住院的情况",
        )

    # 热层与 L3 采样共用这一次调用 (共享嵌入), 用的必须是改写后的 query。
    assert captured_query == ["用户的妈妈最近住院的情况"]


@pytest.mark.asyncio
async def test_hybrid_retrieve_falls_back_to_message_when_no_enhanced():
    """enhanced_query 空 → search_similar 用原 message (向后兼容)."""
    from app.services.memory.retrieval import hybrid

    captured_query = []
    async def _capture_search(query, user_id, tiers, **kwargs):
        captured_query.append(query)
        return []

    with (
        patch.object(hybrid, "search_similar_tiers", side_effect=_capture_search),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval",
                     new_callable=AsyncMock, return_value=None),
        patch.object(hybrid, "cache_set_retrieval",
                     new_callable=AsyncMock),
    ):
        await hybrid.hybrid_retrieve(
            message="我喜欢咖啡",
            user_id="u1",
            workspace_id="w1",
            # enhanced_query 不传, 默认 None
        )

    assert captured_query == ["我喜欢咖啡"]


@pytest.mark.asyncio
async def test_hybrid_cache_key_uses_enhanced_query():
    """cache key 用 effective_query (enhanced 或 message), 避免不同指代复用同 cache.

    场景: 用户先说"那他怎样" (enhanced='妈妈现状') → cache 写 key='妈妈现状'
    后续用户说"她呢?" (enhanced='妹妹现状') → 不同 key, cache miss, 重新搜.
    防 bug: 用 message 做 cache key, 两条指代消息会共用第一次的 cache.
    """
    from app.services.memory.retrieval import hybrid

    captured_keys = []
    async def _capture_cache_get(key, user_id, workspace_id=None):
        captured_keys.append(key)
        return None

    with (
        patch.object(hybrid, "search_similar_tiers",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval", side_effect=_capture_cache_get),
        patch.object(hybrid, "cache_set_retrieval", new_callable=AsyncMock),
    ):
        await hybrid.hybrid_retrieve(
            message="那他呢?", user_id="u1", workspace_id="w1",
            enhanced_query="妈妈情况",
        )
        await hybrid.hybrid_retrieve(
            message="她呢?", user_id="u1", workspace_id="w1",
            enhanced_query="妹妹情况",
        )

    # 两次 cache 查询用了不同的 key (enhanced_query 各异)
    assert captured_keys == ["妈妈情况", "妹妹情况"]


@pytest.mark.asyncio
async def test_data_fetch_enhanced_first_for_elliptical_followup():
    """data_fetch_phase: 省略追问先等 enhanced_query, 避免用错误 query 搜两次.

    流程: "那他怎样了?" 这类消息向量信号不完整, 先等 relevance 还原成
    enhanced_query, 然后只检索一次.
    """
    from app.services.chat.data_fetch_phase import fetch_parallel_context
    from app.services.memory.retrieval.relevance import RelevanceResult
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    relevance_result = RelevanceResult(level="strong", enhanced_query="妈妈最近情况")

    retrieval_calls = []
    async def _track_retrieve(message, user_id, workspace_id=None,
                               enhanced_query=None, **kw):
        retrieval_calls.append({"message": message, "enhanced": enhanced_query})
        return {"memories": [], "memory_strings": [], "graph_context": None}

    with (
        patch("app.services.chat.data_fetch_phase.classify_memory_relevance",
              new_callable=AsyncMock, return_value=relevance_result),
        patch("app.services.chat.data_fetch_phase.hybrid_retrieve",
              side_effect=_track_retrieve),
        patch("app.services.chat.data_fetch_phase.analyze_user_emotion",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_latest_portrait",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_cached_schedule",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_topic_intimacy",
              new_callable=AsyncMock, return_value=50.0),
    ):
        ctx = await fetch_parallel_context(
            user_id="u1", agent_id="a1", workspace_id="w1",
            user_message="那他怎样了?",
            messages_dicts=[{"role": "user", "content": "..."}],
            parsed_times=[],
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
        )

    assert len(retrieval_calls) == 1
    assert retrieval_calls[0]["enhanced"] == "妈妈最近情况"
    assert ctx.memory_relevance == "strong"


@pytest.mark.asyncio
async def test_data_fetch_retrieves_for_ai_self_preference_even_when_llm_says_weak():
    """Agent self-preference questions must still search memory."""
    from app.services.chat.data_fetch_phase import fetch_parallel_context
    from app.services.memory.retrieval.context_selector import ClassifiedMemory
    from app.services.memory.retrieval.relevance import RelevanceResult
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    relevance_result = RelevanceResult(level="weak", enhanced_query="")
    retrieval_calls = []
    ai_memory = ClassifiedMemory(
        id="ai-movie",
        text="我喜欢烧脑科幻电影",
        relevance="medium",
        score=0.6,
        source="ai",
    )

    async def _track_retrieve(message, user_id, workspace_id=None,
                              enhanced_query=None, **kw):
        retrieval_calls.append({"message": message, "enhanced": enhanced_query})
        return {"memories": [ai_memory], "memory_strings": [ai_memory.text], "graph_context": None}

    with (
        patch("app.services.chat.data_fetch_phase.classify_memory_relevance",
              new_callable=AsyncMock, return_value=relevance_result),
        patch("app.services.chat.data_fetch_phase.hybrid_retrieve",
              side_effect=_track_retrieve),
        patch("app.services.chat.data_fetch_phase.analyze_user_emotion",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_latest_portrait",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_cached_schedule",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_topic_intimacy",
              new_callable=AsyncMock, return_value=50.0),
    ):
        ctx = await fetch_parallel_context(
            user_id="u1", agent_id="a1", workspace_id="w1",
            user_message="你喜欢什么电影啊",
            messages_dicts=[],
            parsed_times=[],
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
        )

    assert ctx.memory_relevance == "medium"
    assert len(retrieval_calls) == 1
    assert ctx.classified_memories == [ai_memory]


def test_ai_self_relation_query_pattern_is_generic_and_user_safe():
    from app.services.memory.retrieval.query_patterns import (
        ai_profile_search_query,
        asks_ai_profile_relation,
        asks_ai_stable_relation,
    )

    assert asks_ai_stable_relation("你喜欢什么电影啊")
    assert asks_ai_stable_relation("你最喜欢哪类电影")
    assert asks_ai_stable_relation("电影方面你偏哪一类")
    assert asks_ai_stable_relation("我想知道你喜欢什么电影")
    assert asks_ai_stable_relation("你有没有喜欢的电影")
    assert asks_ai_stable_relation("你去过哪些城市")
    assert asks_ai_stable_relation("哪些城市你去过")
    assert asks_ai_stable_relation("你怎么看科幻片")
    assert asks_ai_stable_relation("你多大了")
    assert asks_ai_stable_relation("我想知道你多大了")
    assert asks_ai_stable_relation("你什么时候出生")
    assert asks_ai_stable_relation("你叫什么名字")
    assert asks_ai_stable_relation("你是做什么的")
    assert asks_ai_stable_relation("你做什么工作")
    assert asks_ai_stable_relation("你大学学什么专业")
    assert asks_ai_stable_relation("你还记得自己是哪个学校读的高中？")

    assert asks_ai_profile_relation("你多大了")
    assert asks_ai_profile_relation("你什么时候出生")
    assert asks_ai_profile_relation("我想知道你多大了")
    assert asks_ai_profile_relation("你生日是哪天")
    assert asks_ai_profile_relation("你是哪里人")
    assert asks_ai_profile_relation("你是什么职业")
    assert asks_ai_profile_relation("你在哪里工作")
    assert asks_ai_profile_relation("你还记得自己是哪个学校读的高中？")
    assert "AI 年龄" in ai_profile_search_query("你多大了")
    assert "用户 年龄" in ai_profile_search_query("你多大了")
    high_school_query = ai_profile_search_query("你还记得自己是哪个学校读的高中？")
    assert high_school_query.startswith("你还记得自己是哪个学校读的高中？")
    assert "AI 教育背景" in high_school_query
    assert "用户 教育背景" in high_school_query
    assert "高中" in high_school_query

    assert not asks_ai_stable_relation("你觉得我怎么样")
    assert not asks_ai_stable_relation("你知道我喜欢什么电影吗")
    assert not asks_ai_stable_relation("你还记得我喜欢什么电影吗")
    assert not asks_ai_stable_relation("你猜我多大")
    assert not asks_ai_profile_relation("你知道我多大吗")
    assert not ai_profile_search_query("你知道我多大吗")
    assert not asks_ai_profile_relation("你工作忙吗")
    assert not asks_ai_stable_relation("那个呢")


@pytest.mark.asyncio
async def test_data_fetch_still_retrieves_in_parallel_for_complete_message_with_enhanced_query():
    """完整消息保持 retrieval/relevance 并行; 初次检索稀疏时再重检索."""
    from app.services.chat.data_fetch_phase import fetch_parallel_context
    from app.services.memory.retrieval.relevance import RelevanceResult
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    relevance_result = RelevanceResult(level="strong", enhanced_query="用户的妈妈最近情况")

    retrieval_calls = []
    async def _track_retrieve(message, user_id, workspace_id=None,
                               enhanced_query=None, **kw):
        retrieval_calls.append({"message": message, "enhanced": enhanced_query})
        return {"memories": [], "memory_strings": [], "graph_context": None}

    with (
        patch("app.services.chat.data_fetch_phase.classify_memory_relevance",
              new_callable=AsyncMock, return_value=relevance_result),
        patch("app.services.chat.data_fetch_phase.hybrid_retrieve",
              side_effect=_track_retrieve),
        patch("app.services.chat.data_fetch_phase.analyze_user_emotion",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_latest_portrait",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_cached_schedule",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_topic_intimacy",
              new_callable=AsyncMock, return_value=50.0),
    ):
        await fetch_parallel_context(
            user_id="u1", agent_id="a1", workspace_id="w1",
            user_message="我想聊聊妈妈最近的事",
            messages_dicts=[{"role": "user", "content": "..."}],
            parsed_times=[],
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
        )

    assert len(retrieval_calls) == 2
    assert retrieval_calls[0]["enhanced"] is None
    assert retrieval_calls[1]["enhanced"] == "用户的妈妈最近情况"


@pytest.mark.asyncio
async def test_data_fetch_skips_enhanced_reretrieve_when_initial_result_is_enough():
    """完整消息初次检索已有足够记忆时, 不为轻微 enhanced_query 重跑."""
    from app.services.chat.data_fetch_phase import fetch_parallel_context
    from app.services.memory.retrieval.context_selector import ClassifiedMemory
    from app.services.memory.retrieval.relevance import RelevanceResult
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    relevance_result = RelevanceResult(level="strong", enhanced_query="用户的妈妈最近情况")
    initial_memories = [
        ClassifiedMemory(id=f"m{i}", text=f"妈妈相关记忆 {i}", relevance="medium", score=0.6)
        for i in range(3)
    ]

    retrieval_calls = []
    async def _track_retrieve(message, user_id, workspace_id=None,
                               enhanced_query=None, **kw):
        retrieval_calls.append({"message": message, "enhanced": enhanced_query})
        return {
            "memories": initial_memories,
            "memory_strings": [m.text for m in initial_memories],
            "graph_context": None,
        }

    with (
        patch("app.services.chat.data_fetch_phase.classify_memory_relevance",
              new_callable=AsyncMock, return_value=relevance_result),
        patch("app.services.chat.data_fetch_phase.hybrid_retrieve",
              side_effect=_track_retrieve),
        patch("app.services.chat.data_fetch_phase.analyze_user_emotion",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_latest_portrait",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_cached_schedule",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_topic_intimacy",
              new_callable=AsyncMock, return_value=50.0),
    ):
        ctx = await fetch_parallel_context(
            user_id="u1", agent_id="a1", workspace_id="w1",
            user_message="我想聊聊妈妈最近的事",
            messages_dicts=[{"role": "user", "content": "..."}],
            parsed_times=[],
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
        )

    assert len(retrieval_calls) == 1
    assert retrieval_calls[0]["enhanced"] is None
    assert ctx.classified_memories == initial_memories


@pytest.mark.asyncio
async def test_data_fetch_no_re_retrieve_when_weak():
    """relevance=weak 即便有 enhanced_query 也不重检索 (反正 weak 跳过 retrieval)."""
    from app.services.chat.data_fetch_phase import fetch_parallel_context
    from app.services.memory.retrieval.relevance import RelevanceResult
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    relevance_result = RelevanceResult(level="weak", enhanced_query="any")

    retrieval_calls = []
    async def _track_retrieve(message, user_id, workspace_id=None,
                               enhanced_query=None, **kw):
        retrieval_calls.append({"enhanced": enhanced_query})
        return {"memories": [], "memory_strings": [], "graph_context": None}

    with (
        patch("app.services.chat.data_fetch_phase.classify_memory_relevance",
              new_callable=AsyncMock, return_value=relevance_result),
        patch("app.services.chat.data_fetch_phase.hybrid_retrieve",
              side_effect=_track_retrieve),
        patch("app.services.chat.data_fetch_phase.analyze_user_emotion",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_latest_portrait",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_cached_schedule",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_topic_intimacy",
              new_callable=AsyncMock, return_value=50.0),
    ):
        ctx = await fetch_parallel_context(
            user_id="u1", agent_id="a1", workspace_id="w1",
            user_message="今天随便聊点没有记忆关系的话题吧",
            messages_dicts=[{"role": "user", "content": "..."}],
            parsed_times=[],
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
        )

    # weak relevance 直接跳过 retrieval, enhanced_query 也不会触发补救检索。
    assert len(retrieval_calls) == 0
    assert ctx.memory_relevance == "weak"


@pytest.mark.asyncio
async def test_data_fetch_no_re_retrieve_when_no_enhanced():
    """enhanced_query 空 (用户消息已完整) → 不重 retrieve."""
    from app.services.chat.data_fetch_phase import fetch_parallel_context
    from app.services.memory.retrieval.relevance import RelevanceResult
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    relevance_result = RelevanceResult(level="medium", enhanced_query="")

    retrieval_calls = []
    async def _track_retrieve(message, user_id, workspace_id=None,
                               enhanced_query=None, **kw):
        retrieval_calls.append({"enhanced": enhanced_query})
        return {"memories": [], "memory_strings": [], "graph_context": None}

    with (
        patch("app.services.chat.data_fetch_phase.classify_memory_relevance",
              new_callable=AsyncMock, return_value=relevance_result),
        patch("app.services.chat.data_fetch_phase.hybrid_retrieve",
              side_effect=_track_retrieve),
        patch("app.services.chat.data_fetch_phase.analyze_user_emotion",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_latest_portrait",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_cached_schedule",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.chat.data_fetch_phase.get_topic_intimacy",
              new_callable=AsyncMock, return_value=50.0),
    ):
        await fetch_parallel_context(
            user_id="u1", agent_id="a1", workspace_id="w1",
            user_message="我喜欢咖啡",
            messages_dicts=[{"role": "user", "content": "..."}],
            parsed_times=[],
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            l3_trigger_classify_fn=AsyncMock(return_value="无"),
        )

    assert len(retrieval_calls) == 1  # 仅初次并行


@pytest.mark.asyncio
async def test_relevance_parser_handles_malformed_json():
    """LLM 输出不规范 JSON (引号错/缺括号) → fallback 到旧单字符模式 + 默认 medium."""
    from app.services.memory.retrieval.relevance import _parse_relevance_response

    # 引号错: 单引号
    r = _parse_relevance_response("{'level': '强', 'enhanced_query': ''}")
    # JSON 解析失败 → fallback 找单字符 → 找到 '强' → strong
    assert r.level == "strong"

    # 完全乱码
    r = _parse_relevance_response("xyz123!@#")
    assert r.level == "medium"

    # 中文混合
    r = _parse_relevance_response("我觉得是中等吧")
    # fallback 找到 '中' → medium
    assert r.level == "medium"


@pytest.mark.asyncio
async def test_relevance_parser_extracts_json_from_noise():
    """LLM 输出含前后冗余文字 → 提取 {} 部分解析."""
    from app.services.memory.retrieval.relevance import _parse_relevance_response

    raw = '好的, 这里是判断: {"level": "强", "enhanced_query": "用户的妈妈"} 完毕.'
    r = _parse_relevance_response(raw)
    assert r.level == "strong"
    assert r.enhanced_query == "用户的妈妈"


# ═══════════════════════════════════════════════════════════════════
# CR Round 1 发现的 bug 修复回归
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_cache_write_uses_same_key_as_read_with_enhanced_query():
    """Phase 2.4 CR bug fix: cache_set_retrieval 必须用跟 cache_retrieval 相同
    的 cache_key (effective_query), 否则 enhanced_query 路径永远 cache miss.
    """
    from app.services.memory.retrieval import hybrid

    set_calls = []
    async def _capture_set(key, user_id, result, workspace_id=None):
        set_calls.append(key)

    with (
        patch.object(hybrid, "search_similar_tiers",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval",
                     new_callable=AsyncMock, return_value=None),
        patch.object(hybrid, "cache_set_retrieval", side_effect=_capture_set),
    ):
        # 场景 1: enhanced_query 非空 → set/get 都用 enhanced_query
        await hybrid.hybrid_retrieve(
            message="那他呢?", user_id="u1", workspace_id="w1",
            enhanced_query="妈妈情况",
        )

    assert set_calls == ["妈妈情况"], (
        f"cache_set_retrieval 必须用 enhanced_query (跟 read 一致); got {set_calls}"
    )


@pytest.mark.asyncio
async def test_cache_write_uses_message_when_no_enhanced_query():
    """无 enhanced_query 时, cache write 仍用 message (跟 read 一致)."""
    from app.services.memory.retrieval import hybrid

    set_calls = []
    async def _capture_set(key, user_id, result, workspace_id=None):
        set_calls.append(key)

    with (
        patch.object(hybrid, "search_similar_tiers",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval",
                     new_callable=AsyncMock, return_value=None),
        patch.object(hybrid, "cache_set_retrieval", side_effect=_capture_set),
    ):
        await hybrid.hybrid_retrieve(
            message="我喜欢咖啡", user_id="u1", workspace_id="w1",
            # enhanced_query 不传
        )

    assert set_calls == ["我喜欢咖啡"]


@pytest.mark.asyncio
async def test_l3_awakening_uses_enhanced_query():
    """Phase 2.4: L3 也走 enhanced_query (省略指代场景)."""
    from app.services.chat import data_fetch_phase
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    captured_query = []
    async def _capture_search(query, user_id, workspace_id=None):
        captured_query.append(query)
        return []

    with patch.object(data_fetch_phase, "search_l3_memories",
                      side_effect=_capture_search):
        await data_fetch_phase.maybe_awaken_l3(
            user_message="那他呢?",
            user_id="u1", workspace_id="w1",
            detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
            memory_relevance="strong",
            l3_trigger_classify_fn=AsyncMock(return_value="请求更久"),
            enhanced_query="妈妈情况",
        )

    assert captured_query == ["妈妈情况"], (
        f"L3 awakening 必须用 enhanced_query; got {captured_query}"
    )


@pytest.mark.asyncio
async def test_l3_awakening_falls_back_to_message():
    """L3: 无 enhanced_query → 用原 message (向后兼容)."""
    from app.services.chat import data_fetch_phase
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    captured_query = []
    async def _capture_search(query, user_id, workspace_id=None):
        captured_query.append(query)
        return []

    with patch.object(data_fetch_phase, "search_l3_memories",
                      side_effect=_capture_search):
        await data_fetch_phase.maybe_awaken_l3(
            user_message="还记得我以前的事吗?",
            user_id="u1", workspace_id="w1",
            detected_intent=IntentResult(intent=IntentType.L3_RECALL, confidence=0.9),
            memory_relevance="strong",
            l3_trigger_classify_fn=AsyncMock(return_value="请求更久"),
            # enhanced_query 默认 ""
        )

    assert captured_query == ["还记得我以前的事吗?"]
