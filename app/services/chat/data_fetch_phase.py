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

from app.observability.events import (
    EVT_LLM_FAIL,
    EVT_MEMORY_L3_AWAKEN,
    EVT_MEMORY_RELEVANCE,
    EVT_MEMORY_RETRIEVED,
)
from app.services.chat.intent_dispatcher import IntentResult, IntentType
from app.services.memory.retrieval.hybrid import hybrid_retrieve
from app.services.memory.retrieval.l3_awakening import search_l3_memories
from app.services.memory.retrieval.relevance import (
    classify_memory_relevance,
    compute_display_score,
)
from app.services.memory.retrieval.ranking import is_recall_query
from app.services.portrait import get_latest_portrait
from app.services.prompting.utils import EMPTY_RECENT_CONTEXT
from app.services.relationship.emotion import compute_ai_pad, extract_emotion
from app.services.relationship.intimacy import get_topic_intimacy
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
    portrait: Any = None
    schedule: Any = None
    topic_intimacy: float = 50.0
    time_memories: list[str] = field(default_factory=list)
    l3_memories: list[str] = field(default_factory=list)
    l3_trigger_label: str = "无"            # "无" | "不满纠正" | "请求更久" | "稀疏补召"
    enhanced_query: str = ""
    ai_status: dict | None = None
    schedule_context: str | None = None


async def _classify_relevance(user_message: str, context: str = ""):
    """Phase 2.4: 返回 RelevanceResult(level, enhanced_query). 调用方拆字段."""
    return await classify_memory_relevance(user_message, context=context)


async def _do_retrieval(
    user_message: str, user_id: str, workspace_id: str | None,
    enhanced_query: str = "",
) -> dict:
    """Phase 2.4: 优先用 enhanced_query 做向量检索 (省略式追问还原后的完整短语).
    enhanced_query 空时 fallback 到原 user_message.
    """
    return await hybrid_retrieve(
        user_message, user_id,
        workspace_id=workspace_id,
        enhanced_query=enhanced_query or None,
    )


_T = Any  # gather result type alias
_EMPTY_RETRIEVAL_RESULT = {"memories": None, "memory_strings": None, "graph_context": None}

_ENHANCED_QUERY_FIRST_HINTS = (
    "他呢", "她呢", "它呢", "这个呢", "那个呢", "这件事", "那件事",
    "那次", "上次", "当时", "后来呢", "然后呢", "颜色呢", "名字呢",
    "情况呢", "怎么样了", "怎样了", "怎么了", "咋样了",
)


def _should_wait_for_enhanced_query(user_message: str) -> bool:
    """省略式追问先等 enhanced_query, 避免用错误 query 做一次无效检索."""
    text = user_message.strip()
    if not text or len(text) > 24:
        return False
    return any(hint in text for hint in _ENHANCED_QUERY_FIRST_HINTS)


def _unwrap(result: _T, default: _T, label: str) -> _T:
    """Unwrap one slot of an asyncio.gather(return_exceptions=True) call: log + fallback on Exception."""
    if isinstance(result, Exception):
        logger.warning(f"{label} failed: {result}")
        return default
    return result


def format_recent_context(
    messages_dicts: list[dict],
    *,
    turns: int = 4,
    max_chars: int = 400,
    exclude_message_id: str | None = None,
) -> str:
    """Spec §3.2 AIPAD值判断 的 recent_context 输入：最近 N 条用户/AI 消息。

    exclude_message_id: 排除指定 ID 的消息. 用法: short-circuit handler 的 prompt
    同时有 {message} (当前用户消息) + {context} (recent_context) 占位符, 如果不
    排除当前消息, LLM 会看到它两遍 (生产 trace 2026-05-07 16:57 实测). AI PAD
    路径不传 exclude — 那条路径需要看完整含当前消息的上下文做情绪判断.
    """
    if not messages_dicts:
        return EMPTY_RECENT_CONTEXT
    tail = messages_dicts[-turns:]
    lines: list[str] = []
    for m in tail:
        # 排除指定 ID (典型: 当前 user_message 已经在 prompt 的 {message} 占位符)
        if exclude_message_id and m.get("id") == exclude_message_id:
            continue
        role = m.get("role") or "user"
        text = (m.get("content") or "").strip()
        if not text:
            continue
        prefix = "AI" if role == "assistant" else "用户"
        lines.append(f"{prefix}: {text[:120]}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text or EMPTY_RECENT_CONTEXT


async def _load_portrait(user_id: str, agent_id: str | None) -> Any:
    if agent_id:
        return await get_latest_portrait(user_id, agent_id)
    return None


async def _load_schedule(agent_id: str | None) -> Any:
    if agent_id:
        return await get_cached_schedule(agent_id)
    return None


async def _load_topic_intimacy(agent_id: str | None, user_id: str) -> float:
    """spec §2.1 无数据归 0 (cold start). 跟 proactive.state._load_topic_intimacy 一致."""
    if agent_id and user_id:
        return await get_topic_intimacy(agent_id, user_id)
    return 0.0


async def _load_time_memories(
    user_id: str,
    parsed_times: list,
    workspace_id: str | None,
) -> list[str]:
    """spec §9.3.4：按解析出的过去时间区间召回记忆。"""
    past_times = [pt for pt in parsed_times if not pt.is_future]
    if not past_times:
        return []
    from app.services.memory.retrieval.vector_search import search_by_time_range
    all_rows = await asyncio.gather(
        *[
            search_by_time_range(
                user_id, pt.start, pt.end, limit=5, workspace_id=workspace_id,
            )
            for pt in past_times
        ]
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
    n_retrieved = len(classified_memories) if classified_memories else 0
    logger.info(
        f"[DEBUG-MEM] retrieval returned {n_retrieved} memories",
        extra={
            "event": EVT_MEMORY_RETRIEVED,
            "memory_relevance": memory_relevance,
            "n_retrieved": n_retrieved,
            "has_graph_context": graph_context is not None,
        },
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
        rank_score = float(getattr(m, "score", 0.0) or 0.0)
        if rank_score > 0:
            m.display_score = rank_score
        else:
            m.display_score = compute_display_score(
                importance=getattr(m, "importance", 0.5),
                last_accessed_at=(
                    getattr(m, "last_accessed_at", None)
                    or getattr(m, "created_at", None)
                ),
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


async def maybe_awaken_l3(
    user_message: str,
    user_id: str,
    workspace_id: str | None,
    detected_intent: IntentResult,
    memory_relevance: str,
    l3_trigger_classify_fn: Callable[[str], Awaitable[str]],
    enhanced_query: str = "",
    l1_l2_count: int | None = None,
) -> tuple[list[str], str]:
    """spec §4 step 5 + §3.4.5：强相关或调用久远记忆意图 → 调 L3 trigger 判定.

    Phase 2.4: 加 enhanced_query 参数. 省略指代场景下 (e.g. "那他呢") L3 也用
    enhanced_query 做向量检索, 提升久远记忆召回率. trigger 分类用原 message
    (LLM 看不到上下文也能判"不满纠正/请求更久").
    """
    should_call_l3 = detected_intent.intent == IntentType.L3_RECALL
    sparse_fallback = (
        l1_l2_count is not None
        and l1_l2_count < 3
        and memory_relevance in ("medium", "strong")
        and is_recall_query(user_message)
    )
    if not (memory_relevance == "strong" or should_call_l3 or sparse_fallback):
        return [], "无"

    if sparse_fallback and memory_relevance == "medium" and not should_call_l3:
        label = "稀疏补召"
    else:
        try:
            label = await l3_trigger_classify_fn(user_message)
        except Exception as e:
            logger.warning(
                f"L3 trigger classify failed: {e}",
                extra={"event": EVT_LLM_FAIL, "stage": "l3_trigger_classify"},
            )
            label = "无"

    # §3.4.5 调用久远记忆意图 → 无论分类结果都召回；§4 强相关 → 仅前两类召回
    if not (should_call_l3 or sparse_fallback or label in ("不满纠正", "请求更久")):
        logger.info(
            f"[L3-TRIGGER] label='{label}' for '{user_message[:40]}' — skip awaken",
            extra={
                "event": EVT_MEMORY_L3_AWAKEN,
                "trigger_label": label,
                "awakened": False,
                "n_l3_retrieved": 0,
            },
        )
        return [], label

    # Phase 2.4: enhanced_query 优先 (省略指代场景), fallback 到原 message
    search_query = enhanced_query or user_message
    l3_results = await search_l3_memories(search_query, user_id, workspace_id=workspace_id)
    l3_memories = [r.get("content") or r.get("summary", "") for r in l3_results if r]
    logger.info(
        f"[L3-TRIGGER] label='{label}' awakened {len(l3_memories)} memories "
        f"(query='{search_query[:40]}')",
        extra={
            "event": EVT_MEMORY_L3_AWAKEN,
            "trigger_label": label,
            "awakened": bool(l3_memories),
            "n_l3_retrieved": len(l3_memories),
            "used_enhanced_query": bool(enhanced_query),
        },
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
    detected_intent: IntentResult | None = None,
    l3_trigger_classify_fn: Callable[[str], Awaitable[str]] | None = None,
) -> FetchedContext:
    """spec §3.1+§3.2 step 2-3：并行拉取记忆/情绪/画像/作息 + L3 awakening。

    detected_intent / l3_trigger_classify_fn 二者均给定时, 内部会做 L3 唤醒;
    任一为 None 时跳过 L3 (调用方负责后续单独调 maybe_awaken_l3). 这是 P0-2
    优化打开的口子: 让 intent 与本函数能并行, 短路意图早返回时无需等 fetch.
    """
    # Schedule 提前 (Redis 缓存)，使 compute_ai_pad 能进 gather 并行块
    schedule = await _load_schedule(agent_id)
    ai_status = get_current_status(schedule) if schedule else None
    schedule_context = format_schedule_context(ai_status) if ai_status else None
    status_label = (ai_status or {}).get("status", "空闲")
    activity_label = (ai_status or {}).get("activity", "自由活动")
    time_info = get_current_time()
    current_time_str = time_info.now.strftime("%Y-%m-%d %H:%M") + f" {time_info.weekday}"
    recent_context = format_recent_context(messages_dicts)

    wait_for_enhanced_query = _should_wait_for_enhanced_query(user_message)
    retrieval_awaitable = (
        asyncio.sleep(0, result=_EMPTY_RETRIEVAL_RESULT)
        if wait_for_enhanced_query else _do_retrieval(user_message, user_id, workspace_id)
    )
    (
        relevance_result, retrieval_result,
        portrait, topic_intimacy,
        time_memories_result, user_emotion_result, emotion_result,
    ) = await asyncio.gather(
        _classify_relevance(user_message, context=recent_context),
        retrieval_awaitable,
        _load_portrait(user_id, agent_id),
        _load_topic_intimacy(agent_id, user_id),
        _load_time_memories(user_id, parsed_times, workspace_id),
        extract_emotion(user_message),
        compute_ai_pad(
            current_time=current_time_str,
            schedule_status=status_label,
            current_activity=activity_label,
            recent_context=recent_context,
        ),
        return_exceptions=True,
    )

    # Phase 2.4: relevance 返 RelevanceResult(level, enhanced_query). 历史返 str.
    memory_relevance = "medium"
    enhanced_query = ""
    if isinstance(relevance_result, Exception):
        logger.warning(f"Memory relevance classification failed: {relevance_result}")
    else:
        # 兼容: 新 RelevanceResult 取 .level + .enhanced_query;
        # 历史路径 (test mock 返 str) 直接用 str.
        level_attr = getattr(relevance_result, "level", None)
        if level_attr:
            memory_relevance = level_attr
            enhanced_query = getattr(relevance_result, "enhanced_query", "") or ""
        elif isinstance(relevance_result, str):
            memory_relevance = relevance_result

    logger.info(
        f"[DEBUG-MEM] relevance='{memory_relevance}' enhanced='{enhanced_query[:40]}' "
        f"for '{user_message[:60]}'",
        extra={
            "event": EVT_MEMORY_RELEVANCE,
            "memory_relevance": memory_relevance,
            "msg_len": len(user_message),
            "has_enhanced_query": bool(enhanced_query),
        },
    )

    # Phase 2.4 + retrieval latency fix:
    # - 明显省略式追问 ("那他呢?" / "颜色呢?") 先等 enhanced_query, 再只检索一次。
    # - 完整消息仍保持 relevance/retrieval 并行; 若 LLM 额外给 enhanced_query, 再重检索。
    if wait_for_enhanced_query and memory_relevance != "weak":
        try:
            retrieval_result = await _do_retrieval(
                user_message, user_id, workspace_id,
                enhanced_query=enhanced_query,
            )
        except Exception as e:
            logger.warning(f"Enhanced-first retrieval failed: {e}")
            retrieval_result = _EMPTY_RETRIEVAL_RESULT
    elif enhanced_query and memory_relevance != "weak":
        logger.info(
            f"[DEBUG-MEM] re-retrieve with enhanced_query='{enhanced_query[:40]}'"
        )
        try:
            retrieval_result = await _do_retrieval(
                user_message, user_id, workspace_id,
                enhanced_query=enhanced_query,
            )
        except Exception as e:
            logger.warning(f"Enhanced retrieval failed, keep original: {e}")

    classified_memories, memory_strings, graph_context = _post_process_retrieval(
        memory_relevance, retrieval_result,
    )

    portrait = _unwrap(portrait, None, "Loading portrait")
    topic_intimacy = _unwrap(topic_intimacy, 50.0, "Loading topic intimacy")
    time_memories: list[str] = _unwrap(time_memories_result, [], "Loading time memories") or []
    user_emotion: dict | None = _unwrap(user_emotion_result, None, "extract_emotion")
    emotion: dict | None = _unwrap(emotion_result, None, "compute_ai_pad")

    if detected_intent is not None and l3_trigger_classify_fn is not None:
        # Phase 2.4: L3 也用 enhanced_query 做向量检索 (省略指代场景)
        l1_l2_count = len(classified_memories or [])
        l3_memories, l3_trigger_label = await maybe_awaken_l3(
            user_message, user_id, workspace_id,
            detected_intent, memory_relevance,
            l3_trigger_classify_fn,
            enhanced_query=enhanced_query,
            l1_l2_count=l1_l2_count,
        )
    else:
        l3_memories, l3_trigger_label = [], "无"

    return FetchedContext(
        memory_relevance=memory_relevance,
        classified_memories=classified_memories,
        memory_strings=memory_strings,
        graph_context=graph_context,
        emotion=emotion,
        user_emotion=user_emotion,
        portrait=portrait,
        schedule=schedule,
        topic_intimacy=float(topic_intimacy) if topic_intimacy is not None else 50.0,
        time_memories=time_memories,
        l3_memories=l3_memories,
        l3_trigger_label=l3_trigger_label,
        enhanced_query=enhanced_query,
        ai_status=ai_status,
        schedule_context=schedule_context,
    )
