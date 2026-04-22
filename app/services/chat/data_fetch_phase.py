"""Spec §3.1 + §3.2 step 2-3：聊天热路径的并行数据拉取阶段。

把 orchestrator 中 9 个 _load_* / _classify_relevance / _do_retrieval 的
asyncio.gather 块和后续的 L3 awakening、ai_status 派生、reranking 全部封装。

输出 `FetchedContext` 数据类，下游 prompt 构建只需读字段。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.services.chat.intent_dispatcher import IntentResult, IntentType
from app.services.memory.retrieval.hybrid import hybrid_retrieve
from app.services.memory.retrieval.l3_awakening import search_l3_memories
from app.services.memory.retrieval.relevance import (
    classify_memory_relevance,
    compute_display_score,
)
from app.services.portrait import get_latest_portrait
from app.services.relationship.emotion import compute_ai_pad, extract_emotion
from app.services.relationship.intimacy import get_topic_intimacy
from app.services.runtime.cache import cache_summarizer
from app.services.schedule_domain.schedule import (
    format_schedule_context,
    get_cached_schedule,
    get_current_status,
)
from app.services.schedule_domain.time_service import get_current_time

logger = logging.getLogger(__name__)


@dataclass
class FetchedContext:
    """spec §3.1+§3.2 并行拉取后聚合的所有上下文信号。"""

    memory_relevance: str = "medium"        # "weak" | "medium" | "strong"
    classified_memories: list | None = None
    memory_strings: list[str] | None = None
    graph_context: dict | None = None
    emotion: dict | None = None             # AI PAD (spec §3.2)
    user_emotion: dict | None = None        # 用户 PAD (spec §3.2 用户侧)
    summaries: dict | None = None
    portrait: Any = None
    schedule: Any = None
    topic_intimacy: float = 50.0
    time_memories: list[str] = field(default_factory=list)
    l3_memories: list[str] = field(default_factory=list)
    l3_trigger_label: str = "无"            # "无" | "不满纠正" | "请求更久"
    ai_status: dict | None = None
    schedule_context: str | None = None


async def _classify_relevance(user_message: str) -> str:
    return await classify_memory_relevance(user_message)


async def _do_retrieval(user_message: str, user_id: str, workspace_id: str | None) -> dict:
    return await hybrid_retrieve(user_message, user_id, workspace_id=workspace_id)


_T = Any  # gather result type alias


def _unwrap(result: _T, default: _T, label: str) -> _T:
    """Unwrap one slot of an asyncio.gather(return_exceptions=True) call: log + fallback on Exception."""
    if isinstance(result, Exception):
        logger.warning(f"{label} failed: {result}")
        return default
    return result


def _format_recent_context(messages_dicts: list[dict], *, turns: int = 4, max_chars: int = 400) -> str:
    """Spec §3.2 AIPAD值判断 的 recent_context 输入：最近 N 条用户/AI 消息。"""
    if not messages_dicts:
        return "（无）"
    tail = messages_dicts[-turns:]
    lines: list[str] = []
    for m in tail:
        role = m.get("role") or "user"
        text = (m.get("content") or "").strip()
        if not text:
            continue
        prefix = "AI" if role == "assistant" else "用户"
        lines.append(f"{prefix}: {text[:120]}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text or "（无）"


async def _load_cached_summaries(messages_dicts: list[dict]) -> dict | None:
    from app.services.summarizer import _conv_hash
    prev_msgs = messages_dicts[:-1]
    prev_user_content = next(
        (m["content"] for m in reversed(prev_msgs) if m["role"] == "user"),
        "",
    )
    ch = _conv_hash(prev_msgs, prev_user_content)
    return await cache_summarizer(ch)


async def _load_portrait(user_id: str, agent_id: str | None) -> Any:
    if agent_id:
        return await get_latest_portrait(user_id, agent_id)
    return None


async def _load_schedule(agent_id: str | None) -> Any:
    if agent_id:
        return await get_cached_schedule(agent_id)
    return None


async def _load_topic_intimacy(agent_id: str | None, user_id: str) -> float:
    if agent_id and user_id:
        return await get_topic_intimacy(agent_id, user_id)
    return 50.0


async def _load_time_memories(user_id: str, parsed_times: list) -> list[str]:
    """spec §9.3.4：按解析出的过去时间区间召回记忆。"""
    past_times = [pt for pt in parsed_times if not pt.is_future]
    if not past_times:
        return []
    from app.services.memory.retrieval.vector_search import search_by_time_range
    all_rows = await asyncio.gather(
        *[search_by_time_range(user_id, pt.start, pt.end, limit=5) for pt in past_times]
    )
    seen: set[str] = set()
    results: list[str] = []
    for rows in all_rows:
        for r in rows:
            content = r.get("summary") or r.get("content", "")
            if content and content not in seen:
                seen.add(content)
                results.append(content)
    return results[:10]


def _post_process_retrieval(
    memory_relevance: str,
    retrieval_result: Any,
) -> tuple[list | None, list[str] | None, dict | None]:
    """Spec §3.2/3.3：rerank by display_score, cap at top 10。返回 (memories, strings, graph)。"""
    if memory_relevance == "weak":
        logger.info("[DEBUG-MEM] SKIPPED — weak relevance, no memories injected")
        return None, None, None
    if isinstance(retrieval_result, Exception):
        logger.warning(f"Hybrid retrieval failed: {retrieval_result}")
        return None, None, None

    classified_memories = retrieval_result.get("memories")
    memory_strings = retrieval_result.get("memory_strings")
    graph_context = retrieval_result.get("graph_context")
    logger.info(
        f"[DEBUG-MEM] retrieval returned "
        f"{len(classified_memories) if classified_memories else 0} memories"
    )
    if not classified_memories:
        logger.info("[DEBUG-MEM] no classified_memories from retrieval (empty result)")
        return None, memory_strings, graph_context

    for m in classified_memories[:5]:
        logger.info(
            f"[DEBUG-MEM]   sim={m.similarity:.3f} imp={m.importance:.2f} "
            f"text='{m.text[:60]}'"
        )
    for m in classified_memories:
        m.display_score = compute_display_score(
            importance=getattr(m, "importance", 0.5),
            last_accessed_at=getattr(m, "created_at", None),
            similarity=getattr(m, "similarity", 0.8),
        )
    classified_memories.sort(key=lambda m: m.display_score, reverse=True)
    classified_memories = classified_memories[:10]
    logger.info(
        f"[DEBUG-MEM] after rerank, top {len(classified_memories)} injected into prompt:"
    )
    for m in classified_memories[:5]:
        logger.info(f"[DEBUG-MEM]   ds={m.display_score:.3f} text='{m.text[:60]}'")
    return classified_memories, memory_strings, graph_context


async def _maybe_awaken_l3(
    user_message: str,
    user_id: str,
    workspace_id: str | None,
    detected_intent: IntentResult,
    memory_relevance: str,
    l3_trigger_classify_fn: Callable[[str], Awaitable[str]],
) -> tuple[list[str], str]:
    """spec §4 step 5 + §3.4.5：强相关或调用久远记忆意图 → 调 L3 trigger 判定。"""
    should_call_l3 = detected_intent.intent == IntentType.L3_RECALL
    if not (memory_relevance == "strong" or should_call_l3):
        return [], "无"

    try:
        label = await l3_trigger_classify_fn(user_message)
    except Exception as e:
        logger.warning(f"L3 trigger classify failed: {e}")
        label = "无"
    logger.info(f"[L3-TRIGGER] label='{label}' for '{user_message[:40]}'")

    # §3.4.5 调用久远记忆意图 → 无论分类结果都召回；§4 强相关 → 仅前两类召回
    if not (should_call_l3 or label in ("不满纠正", "请求更久")):
        return [], label

    l3_results = await search_l3_memories(user_message, user_id, workspace_id=workspace_id)
    l3_memories = [r.get("content") or r.get("summary", "") for r in l3_results if r]
    if l3_memories:
        logger.info(
            f"L3 awakening: {len(l3_memories)} memories injected (label='{label}')"
        )
    return l3_memories, label


async def fetch_parallel_context(
    *,
    user_id: str,
    agent_id: str | None,
    workspace_id: str | None,
    user_message: str,
    messages_dicts: list[dict],
    parsed_times: list,
    detected_intent: IntentResult,
    l3_trigger_classify_fn: Callable[[str], Awaitable[str]],
) -> FetchedContext:
    """spec §3.1+§3.2 step 2-3：并行拉取记忆/情绪/画像/作息 + L3 awakening。"""
    # Schedule 提前 (Redis 缓存)，使 compute_ai_pad 能进 gather 并行块
    schedule = await _load_schedule(agent_id)
    ai_status = get_current_status(schedule) if schedule else None
    schedule_context = format_schedule_context(ai_status) if ai_status else None
    status_label = (ai_status or {}).get("status", "空闲")
    activity_label = (ai_status or {}).get("activity", "自由活动")
    time_info = get_current_time()
    current_time_str = time_info.now.strftime("%Y-%m-%d %H:%M") + f" {time_info.weekday}"
    recent_context = _format_recent_context(messages_dicts)

    (
        relevance_result, retrieval_result, summaries,
        portrait, topic_intimacy,
        time_memories_result, user_emotion_result, emotion_result,
    ) = await asyncio.gather(
        _classify_relevance(user_message),
        _do_retrieval(user_message, user_id, workspace_id),
        _load_cached_summaries(messages_dicts),
        _load_portrait(user_id, agent_id),
        _load_topic_intimacy(agent_id, user_id),
        _load_time_memories(user_id, parsed_times),
        extract_emotion(user_message),
        compute_ai_pad(
            current_time=current_time_str,
            schedule_status=status_label,
            current_activity=activity_label,
            recent_context=recent_context,
        ),
        return_exceptions=True,
    )

    memory_relevance = "medium"
    if isinstance(relevance_result, Exception):
        logger.warning(f"Memory relevance classification failed: {relevance_result}")
    elif isinstance(relevance_result, str):
        memory_relevance = relevance_result
    logger.info(f"[DEBUG-MEM] relevance='{memory_relevance}' for '{user_message[:60]}'")

    classified_memories, memory_strings, graph_context = _post_process_retrieval(
        memory_relevance, retrieval_result,
    )

    summaries = _unwrap(summaries, None, "Loading cached summaries")
    portrait = _unwrap(portrait, None, "Loading portrait")
    topic_intimacy = _unwrap(topic_intimacy, 50.0, "Loading topic intimacy")
    time_memories: list[str] = _unwrap(time_memories_result, [], "Loading time memories") or []
    user_emotion: dict | None = _unwrap(user_emotion_result, None, "extract_emotion")
    emotion: dict | None = _unwrap(emotion_result, None, "compute_ai_pad")

    l3_memories, l3_trigger_label = await _maybe_awaken_l3(
        user_message, user_id, workspace_id,
        detected_intent, memory_relevance,
        l3_trigger_classify_fn,
    )

    return FetchedContext(
        memory_relevance=memory_relevance,
        classified_memories=classified_memories,
        memory_strings=memory_strings,
        graph_context=graph_context,
        emotion=emotion,
        user_emotion=user_emotion,
        summaries=summaries,
        portrait=portrait,
        schedule=schedule,
        topic_intimacy=float(topic_intimacy) if topic_intimacy is not None else 50.0,
        time_memories=time_memories,
        l3_memories=l3_memories,
        l3_trigger_label=l3_trigger_label,
        ai_status=ai_status,
        schedule_context=schedule_context,
    )
