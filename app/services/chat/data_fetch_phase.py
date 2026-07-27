"""Spec §3.1 + §3.2 step 2-3：聊天热路径的数据拉取阶段。

把 orchestrator 中 9 个 _load_* / _classify_relevance / _do_retrieval 的
上下文拉取和后续的 L3 awakening、ai_status 派生、reranking 全部封装。
弱相关消息会在 relevance gate 后跳过 hybrid retrieval。

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
from app.services.memory.retrieval.knowledge_hits import probe_knowledge_memories
from app.services.memory.retrieval.l3_awakening import search_l3_memories
from app.services.memory.retrieval.query_patterns import (
    ai_profile_search_query,
    asks_ai_stable_relation,
)
from app.services.memory.retrieval.relevance import (
    RelevanceResult,
    classify_memory_relevance,
    compute_display_score,
)
from app.services.memory.retrieval.ranking import is_recall_query
from app.services.memory.retrieval.trace import (
    record_retrieval_session,
    replace_latest_retrieval_selection,
)
from app.services.portrait import get_latest_portrait
from app.services.prompting.utils import EMPTY_RECENT_CONTEXT
from app.services.relationship.emotion import analyze_user_emotion
from app.services.relationship.intimacy import get_topic_intimacy
from app.services.rules.chat_keywords import (
    ENHANCED_QUERY_FIRST_HINTS,
    FAST_WEAK_EMOJI_RE,
    FAST_WEAK_NOISE_RE,
    FAST_WEAK_PROTECTED_HINTS,
    FAST_WEAK_REPEAT_CHARS,
    FAST_WEAK_WORDS,
)
from app.services.schedule_domain.schedule import (
    format_schedule_context,
    get_cached_schedule,
    get_current_status,
)
from app.services.schedule_domain.time_parser import has_explicit_time

logger = logging.getLogger(__name__)


@dataclass
class FetchedContext:
    """spec §3.1+§3.2 并行拉取后聚合的所有上下文信号。"""

    memory_relevance: str = "medium"        # "weak" | "medium" | "strong"
    classified_memories: list | None = None
    memory_strings: list[str] | None = None
    graph_context: dict | None = None
    user_emotion: dict | None = None        # 用户情绪标签: emotion/intensity/confidence
    portrait: Any = None
    schedule: Any = None
    topic_intimacy: float = 50.0
    time_memories: list[str] = field(default_factory=list)
    l3_memories: list[str] = field(default_factory=list)
    l3_trigger_label: str = "无"            # "无" | "不满纠正" | "请求更久" | "稀疏补召"
    enhanced_query: str = ""
    ai_status: dict | None = None
    schedule_context: str | None = None
    # True → 主回复走方舟 Responses API 并强制 web_search (见 llm/web_search_gate).
    needs_web_search: bool = False


async def _classify_relevance(user_message: str, context: str = ""):
    """Phase 2.4: 返回 RelevanceResult(level, enhanced_query). 调用方拆字段."""
    return await classify_memory_relevance(user_message, context=context)


def _web_search_route_available() -> bool:
    """联网搜索开关开启 + 当前生效大模型路由是方舟 (工具只在方舟可用)."""
    try:
        from app.services.runtime_config import resolve_for_current

        resolved = resolve_for_current()
    except Exception:  # noqa: BLE001 — 配置读取失败按不可用处理
        return False
    return bool(
        resolved.web_search_enabled
        and resolved.online_model
        and resolved.remote_chat_provider == "ark"
    )


async def _decide_web_search(user_message: str, context: str) -> bool:
    """小模型判定本轮是否需要实时外部信息.

    路由不可用 (开关关 / 非方舟) 时零开销返回; 否则除极短应答外每条都判定 —
    关键词粗筛覆盖不了专有名词 (见 web_search_gate 模块注释). 与其他 fetch
    任务同批 gather, 不增加串行延迟.
    """
    from app.services.llm.web_search_gate import is_worth_classifying, needs_web_search

    if not _web_search_route_available():
        return False
    if not is_worth_classifying(user_message):
        return False
    return await needs_web_search(user_message, context=context)


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


def _should_wait_for_enhanced_query(user_message: str) -> bool:
    """省略式追问先等 enhanced_query, 避免用错误 query 做一次无效检索."""
    text = user_message.strip()
    if not text or len(text) > 24:
        return False
    return any(hint in text for hint in ENHANCED_QUERY_FIRST_HINTS)


def _has_fast_gate_protected_hint(text: str) -> bool:
    """有记忆/时间/指代线索时, 即使很短也交给 relevance LLM 判断。"""
    return (
        _should_wait_for_enhanced_query(text)
        or is_recall_query(text)
        or has_explicit_time(text)
        or any(hint in text for hint in FAST_WEAK_PROTECTED_HINTS)
    )


def _should_fast_weak_relevance(user_message: str) -> bool:
    """规则 fast path: 明显无记忆价值的短消息直接判 weak。

    这里刻意比 hybrid 的 trivial gate 更保守: 不把"不是/没有"这类否定纠正词
    直接判弱，避免挡住后续纠错/上下文判断；实体短追问如"妈妈呢"也会继续走 LLM。
    """
    text = user_message.strip()
    if not text:
        return True
    if _has_fast_gate_protected_hint(text):
        return False
    if asks_ai_stable_relation(text):
        return False

    cleaned = FAST_WEAK_NOISE_RE.sub("", text)
    cleaned = FAST_WEAK_EMOJI_RE.sub("", cleaned)
    if not cleaned:
        return True
    normalized = cleaned.lower()
    if normalized in FAST_WEAK_WORDS:
        return True
    if (
        len(cleaned) <= 6
        and len(set(cleaned)) <= 2
        and all(ch in FAST_WEAK_REPEAT_CHARS for ch in cleaned)
    ):
        return True
    return False


def _apply_memory_relevance_floor(
    memory_relevance: str,
    user_message: str,
) -> str:
    """Apply deterministic lower bounds for recall-sensitive message classes."""
    if memory_relevance == "weak" and asks_ai_stable_relation(user_message):
        return "medium"
    return memory_relevance


def _retrieved_memory_count(retrieval_result: Any) -> int:
    if isinstance(retrieval_result, Exception) or not isinstance(retrieval_result, dict):
        return 0
    memories = retrieval_result.get("memories")
    if isinstance(memories, list):
        return len(memories)
    strings = retrieval_result.get("memory_strings")
    if isinstance(strings, list):
        return len(strings)
    return 0


def _should_reretrieve_with_enhanced_query(
    *,
    enhanced_query: str,
    retrieval_result: Any,
    sparse_threshold: int = 3,
) -> bool:
    """完整消息的 enhanced_query 只在初次检索稀疏时重跑。

    Relevance LLM 常把完整句子轻微改写成 "用户的xxx", 这类增强不值得
    额外支付一次向量/实体/时间检索。若初次检索少于 sparse_threshold 条,
    再用 enhanced_query 作为补救查询。
    """
    if not enhanced_query:
        return False
    if isinstance(retrieval_result, Exception):
        return True
    return _retrieved_memory_count(retrieval_result) < sparse_threshold


def _extract_l3_trigger_decision(raw: Any) -> tuple[str, str]:
    labels = ("不满纠正", "请求更久", "稀疏补召", "无")
    label = ""
    retrieval_query = ""
    if isinstance(raw, dict):
        label = str(raw.get("label", "")).strip()
        retrieval_query = str(raw.get("retrieval_query", "")).strip()
    else:
        label = str(getattr(raw, "label", "") or "").strip()
        retrieval_query = str(getattr(raw, "retrieval_query", "") or "").strip()
        if not label and isinstance(raw, str):
            for candidate in labels:
                if candidate in raw:
                    label = candidate
                    break
    if label not in labels:
        label = "无"
    if label == "无":
        retrieval_query = ""
    if len(retrieval_query) > 50:
        retrieval_query = retrieval_query[:50]
    return label, retrieval_query


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
    exclude_message_ids: set[str] | None = None,
) -> str:
    """Format recent user/assistant turns for lightweight classifiers and prompts.

    exclude_message_id: 排除指定 ID 的消息. 用法: short-circuit handler 的 prompt
    同时有 {message} (当前用户消息) + {context} (recent_context) 占位符, 如果不
    排除当前消息, LLM 会看到它两遍 (生产 trace 2026-05-07 16:57 实测).
    """
    if not messages_dicts:
        return EMPTY_RECENT_CONTEXT
    excluded_ids = set(exclude_message_ids or set())
    if exclude_message_id:
        excluded_ids.add(exclude_message_id)
    tail = messages_dicts[-turns:]
    lines: list[str] = []
    for m in tail:
        # 排除指定 ID (典型: 当前 user_message 已经在 prompt 的 {message} 占位符)
        if m.get("id") in excluded_ids:
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
    selected_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for rows in all_rows:
        for r in rows:
            candidate_rows.append(r)
            content = r.get("content", "")
            if content and content not in seen:
                seen.add(content)
                results.append(content)
                selected_rows.append(r)
    selected = results[:10]
    record_retrieval_session(
        strategy="explicit_time",
        query="; ".join(
            f"{getattr(pt, 'start', '')}..{getattr(pt, 'end', '')}"
            for pt in past_times
        ),
        workspace_id=workspace_id,
        raw_count=len(candidate_rows),
        candidate_count=len(candidate_rows),
        selected_count=len(selected),
        candidates=candidate_rows,
        selected=selected_rows[:10],
        notes={"time_range_count": len(past_times)},
    )
    return selected


def _post_process_retrieval(
    memory_relevance: str,
    retrieval_result: Any,
) -> tuple[list | None, list[str] | None, dict | None]:
    """Spec §3.2/3.3：补齐 display_score。返回 (memories, strings, graph)。

    最终 top-N / quota 选择权在 context_selector.select_context 内完成。
    这里不再二次排序/截断，避免破坏安全、字面命中、用户记忆保护槽。
    """
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
    memory_strings = [m.text for m in classified_memories]
    logger.info(
        f"[DEBUG-MEM] after post-process, {len(classified_memories)} injected into prompt:"
    )
    for m in classified_memories[:5]:
        logger.info(f"[DEBUG-MEM]   ds={m.display_score:.3f} text='{m.text[:60]}'")
    return classified_memories, memory_strings, graph_context


def _merge_knowledge_hits(
    memory_relevance: str,
    classified_memories: list | None,
    memory_strings: list[str] | None,
    hits: list,
) -> tuple[str, list | None, list[str] | None]:
    """Union literal-hit knowledge rows into the injected memory set.

    On weak relevance the label is escalated to "medium": the weak tier
    prompt carries no memory placeholder at all, so without escalation the
    injected knowledge would never reach the reply LLM. Medium keeps the
    lightweight tier path while feeding user/ai memory slots.
    """
    if not hits:
        return memory_relevance, classified_memories, memory_strings
    merged = (classified_memories or []) + hits
    strings = (memory_strings or []) + [m.text for m in hits]
    relevance = "medium" if memory_relevance == "weak" else memory_relevance
    return relevance, merged, strings


async def maybe_awaken_l3(
    user_message: str,
    user_id: str,
    workspace_id: str | None,
    detected_intent: IntentResult,
    memory_relevance: str,
    l3_trigger_classify_fn: Callable[..., Awaitable[Any]],
    enhanced_query: str = "",
    l1_l2_count: int | None = None,
    recent_context: str = "",
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
        trigger_query = ""
    else:
        try:
            try:
                trigger_result = await l3_trigger_classify_fn(user_message, recent_context)
            except TypeError:
                trigger_result = await l3_trigger_classify_fn(user_message)
            label, trigger_query = _extract_l3_trigger_decision(trigger_result)
        except Exception as e:
            logger.warning(
                f"L3 trigger classify failed: {e}",
                extra={"event": EVT_LLM_FAIL, "stage": "l3_trigger_classify"},
            )
            label = "无"
            trigger_query = ""

    # L3_RECALL 只负责进入专门判定；是否召回由 trigger label 确认。
    # 这样 broad intent 的误判不会绕过 L3 trigger 直接拉久远记忆。
    if not (sparse_fallback or label in ("不满纠正", "请求更久")):
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

    # L3 trigger query 优先，因为它和"是否唤醒 L3"来自同一次上下文判断；
    # 没有 trigger query 时再退回通用 enhanced_query / 原消息。
    search_query = trigger_query or enhanced_query or user_message
    l3_results = await search_l3_memories(search_query, user_id, workspace_id=workspace_id)
    l3_memories = [r.get("content", "") for r in l3_results if r]
    record_retrieval_session(
        strategy="l3_awaken",
        query=search_query,
        enhanced_query=trigger_query or enhanced_query or None,
        workspace_id=workspace_id,
        memory_relevance=memory_relevance,
        trigger_label=label,
        raw_count=len(l3_results),
        candidate_count=len(l3_results),
        selected_count=len(l3_memories),
        candidates=l3_results,
        selected=l3_results,
        notes={
            "intent": detected_intent.intent.value,
            "l1_l2_count": l1_l2_count,
            "trigger_query": trigger_query or None,
        },
    )
    logger.info(
        f"[L3-TRIGGER] label='{label}' awakened {len(l3_memories)} memories "
        f"(query='{search_query[:40]}')",
        extra={
            "event": EVT_MEMORY_L3_AWAKEN,
            "trigger_label": label,
            "awakened": bool(l3_memories),
            "n_l3_retrieved": len(l3_memories),
            "used_enhanced_query": bool(enhanced_query and not trigger_query),
            "used_trigger_query": bool(trigger_query),
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
    l3_trigger_classify_fn: Callable[..., Awaitable[Any]] | None = None,
) -> FetchedContext:
    """spec §3.1+§3.2 step 2-3：拉取记忆/用户情绪/画像/作息 + L3 awakening。

    detected_intent / l3_trigger_classify_fn 二者均给定时, 内部会做 L3 唤醒;
    任一为 None 时跳过 L3 (调用方负责后续单独调 maybe_awaken_l3). 这是 P0-2
    优化打开的口子: 让 intent 与本函数能并行, 短路意图早返回时无需等 fetch.
    """
    # Schedule 提前 (Redis 缓存)，供状态类短路和 prompt 自洽性约束复用。
    schedule = await _load_schedule(agent_id)
    ai_status = get_current_status(schedule) if schedule else None
    schedule_context = format_schedule_context(ai_status) if ai_status else None
    recent_context = format_recent_context(messages_dicts)

    fast_weak_relevance = _should_fast_weak_relevance(user_message)
    wait_for_enhanced_query = (
        False if fast_weak_relevance else _should_wait_for_enhanced_query(user_message)
    )
    if fast_weak_relevance:
        relevance_awaitable = asyncio.sleep(
            0, result=RelevanceResult(level="weak", enhanced_query="")
        )
        retrieval_awaitable = asyncio.sleep(0, result=_EMPTY_RETRIEVAL_RESULT)
    else:
        relevance_awaitable = _classify_relevance(user_message, context=recent_context)
        retrieval_awaitable = asyncio.sleep(0, result=_EMPTY_RETRIEVAL_RESULT)
    (
        relevance_result, retrieval_result,
        portrait, topic_intimacy,
        time_memories_result, user_emotion_result,
        web_search_result,
    ) = await asyncio.gather(
        relevance_awaitable,
        retrieval_awaitable,
        _load_portrait(user_id, agent_id),
        _load_topic_intimacy(agent_id, user_id),
        _load_time_memories(user_id, parsed_times, workspace_id),
        analyze_user_emotion(user_message),
        _decide_web_search(user_message, recent_context),
        return_exceptions=True,
    )
    needs_web_search = _unwrap(web_search_result, False, "web search gate") is True

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

    memory_relevance = _apply_memory_relevance_floor(memory_relevance, user_message)
    profile_enhanced_query = ai_profile_search_query(user_message)
    if profile_enhanced_query and not enhanced_query:
        enhanced_query = profile_enhanced_query

    logger.info(
        f"[DEBUG-MEM] relevance='{memory_relevance}' enhanced='{enhanced_query[:40]}' "
        f"for '{user_message[:60]}'",
        extra={
            "event": EVT_MEMORY_RELEVANCE,
            "memory_relevance": memory_relevance,
            "msg_len": len(user_message),
            "has_enhanced_query": bool(enhanced_query),
            "fast_gate": fast_weak_relevance,
        },
    )

    # Phase 2.4 + retrieval latency fix:
    # - 明显省略式追问 ("那他呢?" / "颜色呢?") 先等 enhanced_query, 再只检索一次。
    # - 完整消息仍保持 relevance/retrieval 并行; 若 LLM 额外给 enhanced_query,
    #   仅在初次检索稀疏/失败时重检索，避免完整消息无意义多跑一次。
    if wait_for_enhanced_query and memory_relevance != "weak":
        try:
            retrieval_result = await _do_retrieval(
                user_message, user_id, workspace_id,
                enhanced_query=enhanced_query,
            )
        except Exception as e:
            logger.warning(f"Enhanced-first retrieval failed: {e}")
            retrieval_result = _EMPTY_RETRIEVAL_RESULT
    elif memory_relevance != "weak":
        try:
            retrieval_result = await _do_retrieval(
                user_message, user_id, workspace_id,
                enhanced_query=profile_enhanced_query,
            )
        except Exception as e:
            logger.warning(f"Hybrid retrieval failed: {e}")
            retrieval_result = e
    if (
        memory_relevance != "weak"
        and not wait_for_enhanced_query
        and not profile_enhanced_query
        and _should_reretrieve_with_enhanced_query(
            enhanced_query=enhanced_query,
            retrieval_result=retrieval_result,
        )
    ):
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

    # Knowledge literal-hit floor: admin-published knowledge rows must stay
    # reachable even when the relevance gate says weak (the classifier cannot
    # know the memory bank contains these topics), when vector ranking drops
    # them, or when an elliptical follow-up ("啥时候开始？") names its topic
    # only in prior turns. Deterministic and cheap; failures never break chat.
    # See knowledge_hits module docstring for the full rationale.
    knowledge_hits: list = []
    try:
        knowledge_hits = await probe_knowledge_memories(
            user_message=user_message,
            enhanced_query=enhanced_query,
            # Raw message texts (not the formatted "用户:/AI:" transcript) so
            # role prefixes never gram-match row contents like "有生命的AI".
            # 10 rows ≈ 3-4 turns with multi-bubble replies (each AI turn is
            # 2-4 rows); a 4-row window missed topic anchors like "西甲你知道
            # 不" sitting 6-7 rows back (2026-07-24 trace).
            context_texts=[
                str(m.get("content") or "") for m in (messages_dicts or [])[-10:]
            ],
            workspace_id=workspace_id,
            exclude_texts=(
                {m.text for m in classified_memories} if classified_memories else frozenset()
            ),
        )
    except Exception as e:
        logger.warning(f"[DEBUG-MEM] knowledge literal-hit probe failed: {e}")
    if knowledge_hits:
        memory_relevance, classified_memories, memory_strings = _merge_knowledge_hits(
            memory_relevance, classified_memories, memory_strings, knowledge_hits,
        )
        logger.info(
            f"[DEBUG-MEM] +{len(knowledge_hits)} knowledge literal hits "
            f"(relevance→{memory_relevance}): "
            + "; ".join(m.text[:40] for m in knowledge_hits),
            extra={
                "event": EVT_MEMORY_RETRIEVED,
                "memory_relevance": memory_relevance,
                "n_knowledge_hits": len(knowledge_hits),
            },
        )

    replace_latest_retrieval_selection(
        strategy="hybrid_l1_l2",
        selected=classified_memories or [],
        final_injected=memory_relevance != "weak",
    )

    portrait = _unwrap(portrait, None, "Loading portrait")
    topic_intimacy = _unwrap(topic_intimacy, 50.0, "Loading topic intimacy")
    time_memories: list[str] = _unwrap(time_memories_result, [], "Loading time memories") or []
    user_emotion: dict | None = _unwrap(user_emotion_result, None, "analyze_user_emotion")

    if detected_intent is not None and l3_trigger_classify_fn is not None:
        # Phase 2.4: L3 也用 enhanced_query 做向量检索 (省略指代场景)
        l1_l2_count = len(classified_memories or [])
        l3_memories, l3_trigger_label = await maybe_awaken_l3(
            user_message, user_id, workspace_id,
            detected_intent, memory_relevance,
            l3_trigger_classify_fn,
            enhanced_query=enhanced_query,
            l1_l2_count=l1_l2_count,
            recent_context=recent_context,
        )
    else:
        l3_memories, l3_trigger_label = [], "无"

    return FetchedContext(
        memory_relevance=memory_relevance,
        classified_memories=classified_memories,
        memory_strings=memory_strings,
        graph_context=graph_context,
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
        needs_web_search=needs_web_search,
    )
