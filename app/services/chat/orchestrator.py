"""Chat service — optimized for low latency.

Hot path (user-facing, ~2s):
  save msg → parallel(vector retrieval + load cached context) → prompt → stream LLM

Background (fire-and-forget, after response):
  emotion extraction + summarizer update + memory pipeline
"""

import asyncio
import contextlib
import json
import logging
import random
import re
from collections.abc import AsyncGenerator
from prisma import Json

from app.config import settings
from app.db import db
from app.services.llm.models import get_chat_model, convert_messages, invoke_text
from app.services.chat.prompt_builder import build_system_prompt, build_chat_messages
from app.services.prompts.system_prompts import (
    MAX_PER_REPLY, MAX_REPLY_COUNT, MAX_TOTAL_CHARS,
)
from app.services.memory.retrieval.hybrid import hybrid_retrieve
from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.summarizer import summarize
from app.services.relationship.emotion import (
    extract_emotion,
    get_ai_emotion,
    quick_emotion_estimate,
    update_emotion_state,
    save_ai_emotion,
)
from app.services.runtime.cache import cache_summarizer
from app.services.portrait import get_latest_portrait
from app.services.schedule_domain.timing import (
    calculate_reply_delay, calculate_typing_duration,
    explain_delay_reason,
)
from app.services.memory.retrieval.relevance import classify_memory_relevance, compute_display_score
from app.services.memory.retrieval.l3_awakening import search_l3_memories
from app.services.memory.interaction.contradiction import (
    detect_l1_contradiction, generate_contradiction_inquiry,
    save_pending_contradiction,
)  # analyze/apply/load/clear 已由 preflight.resolve_pending_contradiction 接管
from app.services.memory.retrieval.access_log import log_memory_access
from app.services.topic import push_topic, format_topic_context
from app.services.schedule_domain.schedule import (
    get_cached_schedule, get_current_status, format_schedule_context,
)
from app.services.schedule_domain.time_service import build_time_context
from app.services.schedule_domain.time_parser import parse_time_expressions, has_explicit_time
from app.services.interaction.boundary import (
    detect_apology, handle_apology,
    check_positive_recovery,
    get_patience_prompt_instruction, get_patience_zone,
    PATIENCE_MAX,
)  # detect_apology/handle_apology: _bg_apology_check + pending 分支仍用
from app.services.relationship.intimacy import get_topic_intimacy, get_relationship_stage
from app.services.trait_adjustment import infer_feedback, detect_direct_feedback, apply_trait_adjustment
from app.services.chat.intent_dispatcher import (
    detect_intent, detect_intent_llm, IntentType, IntentResult,
    LABEL_TO_INTENT, INTENT_PRIORITY,
)
from app.services.chat.multi_intent import (
    finalize_short_circuit as _finalize_short_circuit,
    process_sub_intents as _process_sub_intents,
    short_circuit_reply as _short_circuit_reply_impl,
)
from app.services.chat.intent_handlers import (
    ShortCircuitCtx,
    handle_apology_promise,
    handle_conversation_end,
    handle_current_state,
    handle_deletion,
    handle_schedule_adjust,
    handle_schedule_query,
)
from app.services.chat.intent_replies import (
    delay_explanation_reply as _delay_explanation_reply,
    memory_weak_reply as _memory_weak_reply,
    memory_medium_reply as _memory_medium_reply,
    memory_strong_reply as _memory_strong_reply,
    memory_l3_reply as _memory_l3_reply,
    l3_trigger_classify as _l3_trigger_classify,
    split_reply_to_n_sentences as _split_reply_to_n_sentences,
    ai_reply_emotion as _ai_reply_emotion,
)
from app.services.chat.reply_post_process import emit_replies as _emit_replies
from app.services.chat.reply_generate import generate_reply as _generate_reply
from app.services.chat.preflight import (
    PreflightCtx,
    resolve_pending_contradiction,
    resolve_pending_deletion,
)
from app.services.chat.boundary_phase import BoundaryPhaseCtx, run_boundary
from app.services.mbti import get_mbti
from app.services.chat.fast_fact import update_working_facts, facts_for_prompt
from app.services.interaction.reply_context import actual_delay_seconds, save_last_reply_timestamp
from app.services.proactive.state import start_or_restart_proactive_session

logger = logging.getLogger(__name__)


def _on_task_error(t: asyncio.Task) -> None:
    """Log unhandled exceptions from background tasks."""
    if not t.cancelled() and t.exception():
        logger.error(f"Background post-processing failed: {t.exception()}")


def _fire_background(coro) -> None:
    """Schedule a background coroutine as a fire-and-forget task."""
    task = asyncio.create_task(coro)
    task.add_done_callback(_on_task_error)


async def _short_circuit_reply(
    reply: str,
    conversation_id: str,
    agent_id: str | None,
    user_id: str,
    *,
    sub_intent_mode: bool = False,
    reply_index_offset: int = 0,
    include_done: bool = True,
    extra_metadata: dict | None = None,
) -> list[dict]:
    """Orchestrator-side adapter that injects `_save_replies`."""
    return await _short_circuit_reply_impl(
        reply, conversation_id, agent_id, user_id, _save_replies,
        sub_intent_mode=sub_intent_mode,
        reply_index_offset=reply_index_offset,
        include_done=include_done,
        extra_metadata=extra_metadata,
    )


async def _intent_llm_reply(
    agent,
    user_message: str,
    instruction: str,
) -> str:
    """Generate a short LLM reply for a special intent (farewell, reconciliation, etc.)."""
    prompt = await build_system_prompt(agent=agent, reply_count=1, reply_total=60)
    prompt += f"\n\n## 特殊指令\n{instruction}"
    model = get_chat_model()
    result = await model.ainvoke(convert_messages([
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message},
    ]))
    return result.content.strip().split("||")[0][:60]


# --- Multi-reply split & validate (PRD §3.2.1/§3.2.2) ---

_SENTENCE_END = re.compile(r'[。！？…～~!?]+')

_RELATIONAL_COMPLAINT_KEYWORDS = [
    "怎么不理我", "不理我", "不回我", "不想理我", "你在忙吗", "你还在吗",
    "你是不是不想理我", "是不是不想聊", "是不是烦我", "怎么才回",
]
_DISTRESS_KEYWORDS = [
    "不好", "难受", "烦", "委屈", "崩溃", "糟糕", "不开心", "很累", "想哭",
    "好难过", "撑不住", "心情不好",
]


def truncate_at_sentence(text: str, max_len: int) -> str:
    """截断至max_len内最后一个句子边界。"""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    match = None
    for m in _SENTENCE_END.finditer(truncated):
        match = m
    if match and match.end() > max_len // 2:
        return truncated[:match.end()]
    return truncated


def split_and_validate_replies(
    raw: str,
    max_count: int = MAX_REPLY_COUNT,
    max_per_reply: int = MAX_PER_REPLY,
    max_total: int = MAX_TOTAL_CHARS,
) -> list[str]:
    """按||分割LLM输出，校验条数/单条长度/总长度。"""
    parts = [p.strip() for p in raw.split("||") if p.strip()]
    if not parts:
        return [raw.strip() or "..."]
    parts = parts[:max_count]
    parts = [truncate_at_sentence(p, max_per_reply) for p in parts]
    result: list[str] = []
    total = 0
    for p in parts:
        if total + len(p) > max_total:
            remaining = max_total - total
            if remaining > 5:
                result.append(truncate_at_sentence(p, remaining))
            break
        result.append(p)
        total += len(p)
    return result or [parts[0][:max_per_reply]]


def detect_relational_context(message: str, user_emotion: dict | None) -> str | None:
    """Detect relationship repair / distress cues that need more human handling."""
    text = message.strip()
    if any(keyword in text for keyword in _RELATIONAL_COMPLAINT_KEYWORDS):
        return (
            "用户这句更像是在确认你有没有在意Ta，或者在表达被忽略感。"
            "先短促地接住关系情绪，比如安抚、解释半句、表明你不是故意的；"
            "不要一上来就长解释，也不要立刻抛万能反问。"
        )

    negative_emotion = bool(
        user_emotion
        and float(user_emotion.get("pleasure", 0.0)) < -0.2
        and float(user_emotion.get("arousal", 0.0)) > 0.25
    )
    if any(keyword in text for keyword in _DISTRESS_KEYWORDS) or negative_emotion:
        return (
            "用户这句带明显低落或烦闷情绪。"
            "先回应当下感受，语气真一点、短一点；"
            "不要套模板安慰，不要一下子给很多建议，追问也只问最贴当前情绪的一句。"
        )
    return None


def _langsmith_trace_ctx(user_message: str, conversation_id: str):
    """Create a LangSmith parent trace context if tracing is enabled."""
    if settings.langsmith_tracing:
        from langsmith import trace as ls_trace
        return ls_trace(
            name="chat_request",
            run_type="chain",
            inputs={"message": user_message, "conversation_id": conversation_id},
            project_name="ai-companion",
        )
    return contextlib.nullcontext()


def _get_langsmith_client():
    """Return a cached LangSmith Client singleton."""
    if not hasattr(_get_langsmith_client, "_instance"):
        from langsmith import Client
        _get_langsmith_client._instance = Client()
    return _get_langsmith_client._instance


async def _bg_share_trace(trace_id: str, conversation_id: str) -> None:
    """Background: share the LangSmith trace and update DB message metadata with public URL."""
    try:
        loop = asyncio.get_running_loop()
        client = _get_langsmith_client()
        public_url = await loop.run_in_executor(None, client.share_run, trace_id)
        logger.info(f"Trace shared: {public_url}")
        updated_message_id: str | None = None
        # Update the first assistant reply for this trace.
        msgs = await db.message.find_many(
            where={"conversationId": conversation_id, "role": "assistant"},
            order={"createdAt": "desc"},
            take=20,
        )
        for msg in msgs:
            meta = msg.metadata or {}
            if isinstance(meta, dict) and meta.get("trace_id") == trace_id:
                await db.message.update(
                    where={"id": msg.id},
                    data={
                        "metadata": Json(
                            {
                                **meta,
                                "trace_url": public_url,
                                "trace_pending": False,
                            }
                        )
                    },
                )
                updated_message_id = msg.id
                break
        if updated_message_id:
            from app.services.runtime.ws_manager import manager

            ws = manager.get(conversation_id)
            if ws:
                await ws.send_json(
                    {
                        "type": "trace_ready",
                        "data": {
                            "message_id": updated_message_id,
                            "trace_url": public_url,
                        },
                    }
                )
    except Exception as e:
        logger.warning(f"Failed to share trace: {e}")


def _end_trace(trace_ctx, trace_id: str | None, conversation_id: str) -> None:
    """Safely close the LangSmith trace and fire background share task."""
    trace_ctx.__exit__(None, None, None)
    if trace_id and settings.langsmith_tracing:
        _fire_background(_bg_share_trace(trace_id, conversation_id))


async def stream_chat_response(
    conversation_id: str,
    user_message: str,
    agent,
    user_id: str,
    reply_context: dict | None = None,
    *,
    save_user_message: bool = True,
    user_message_id: str | None = None,
    delivered_from_queue: bool = False,
    sub_intent_mode: bool = False,
    forced_intent: IntentType | None = None,
    reply_index_offset: int = 0,
    parent_patience: int | None = None,
) -> AsyncGenerator[dict, None]:
    """spec §3.3 step 3：多意图拆分后递归调用本函数处理每个子片段。

    sub_intent_mode=True 的子调用：跳过用户消息 DB 写入、边界/pending 检查、
    延迟解释、done 事件、save_last_reply_timestamp 与后台任务；由父调用统一完成。
    forced_intent 指定片段意图不再识别；reply_index_offset 让回复 index 顺延；
    parent_patience 复用父调用的耐心值，避免每个子片段再读一次 Redis。
    子调用共享 reply_context 沿用首条消息的 due_at（spec §6 延迟批处理）。
    """
    pending_sub_fragments: dict[str, str] = {}

    # 碎片聚合/延迟队列在入队时已落库；sub_intent_mode 共享父调用的原始消息
    if save_user_message and not sub_intent_mode:
        saved_msg = await db.message.create(
            data={
                "conversation": {"connect": {"id": conversation_id}},
                "role": "user",
                "content": user_message,
            }
        )
        user_message_id = saved_msg.id

    agent_id = getattr(agent, "id", None)
    conversation = await db.conversation.find_unique(where={"id": conversation_id})
    workspace_id = getattr(conversation, "workspaceId", None)

    # --- LangSmith parent trace (groups all LLM calls for this request) ---
    _trace_ctx = _langsmith_trace_ctx(user_message, conversation_id)
    _run_tree = _trace_ctx.__enter__()
    trace_id = str(_run_tree.id) if _run_tree else None

    # spec §2.6 边界系统全流程（含步骤 2-6 + 步骤 6 中/低耐心短路）
    boundary_ctx = BoundaryPhaseCtx(
        conversation_id=conversation_id,
        agent_id=agent_id,
        user_id=user_id,
        agent=agent,
        user_message=user_message,
        sub_intent_mode=sub_intent_mode,
        parent_patience=parent_patience,
        trace_ctx=_trace_ctx,
        trace_id=trace_id,
        end_trace_fn=_end_trace,
        short_circuit_fn=_short_circuit_reply,
        fire_background_fn=_fire_background,
        bg_apology_check_fn=_bg_apology_check,
        bg_memory_pipeline_fn=_bg_memory_pipeline,
    )
    async for evt in run_boundary(boundary_ctx):
        yield evt
    if boundary_ctx.stopped:
        return
    cached_patience = boundary_ctx.cached_patience

    # Pending 跨消息状态：矛盾追问 / 删除确认。用户的回答不会带意图关键词，
    # 必须在意图识别前先匹配 Redis 里的待处理状态。sub_intent_mode 下跳过。
    if not sub_intent_mode:
        preflight_ctx = PreflightCtx(
            conversation_id=conversation_id,
            agent_id=agent_id,
            user_id=user_id,
            agent=agent,
            trace_ctx=_trace_ctx,
            trace_id=trace_id,
            end_trace_fn=_end_trace,
            short_circuit_fn=_short_circuit_reply,
        )

        async def _chat_text(prompt: str) -> str:
            return await invoke_text(get_chat_model(), prompt)

        async for evt in resolve_pending_contradiction(user_message, preflight_ctx, _chat_text):
            yield evt
        if preflight_ctx.stopped:
            return

        async for evt in resolve_pending_deletion(user_message, preflight_ctx):
            yield evt
        if preflight_ctx.stopped:
            return

    # --- 统一意图识别：关键字快路 + LLM 兜底（spec §3.3 step 1-2） ---
    patience_zone = get_patience_zone(cached_patience)
    if forced_intent is not None:
        # sub_intent_mode 的子片段，意图由父调用指定
        detected_intent = IntentResult(intent=forced_intent, confidence=1.0)
    else:
        detected_intent = detect_intent(user_message, patience_zone)
        # 关键字扫描落空且消息足够长 → 调小模型统一识别，覆盖没有关键词的意图表达
        if detected_intent.intent == IntentType.NONE and len(user_message.strip()) > 4:
            try:
                llm_intent = await detect_intent_llm(user_message)
                if llm_intent.intent != IntentType.NONE:
                    detected_intent = llm_intent
                    logger.info(
                        f"[INTENT-LLM] '{user_message[:30]}' → {llm_intent.intent.value} "
                        f"(labels={llm_intent.metadata.get('llm_labels')})"
                    )
            except Exception as e:
                logger.warning(f"LLM intent recognition failed: {e}")
        # spec §3.3 step 3: 多意图 → 待处理子片段列表（主意图片段替换 user_message，其它稍后递归处理）
        fragments = detected_intent.metadata.get("fragments") if detected_intent.metadata else None
        if fragments and len(fragments) > 1:
            primary_label = next(
                (lb for lb, it in LABEL_TO_INTENT.items()
                 if it == detected_intent.intent and lb in fragments),
                None,
            )
            if primary_label and fragments.get(primary_label):
                user_message = str(fragments[primary_label]).strip() or user_message
            pending_sub_fragments = {
                lb: str(txt).strip()
                for lb, txt in fragments.items()
                if lb != primary_label and str(txt).strip()
            }
            if pending_sub_fragments:
                logger.info(
                    f"[INTENT-MULTI] primary={detected_intent.intent.value} "
                    f"sub={list(pending_sub_fragments.keys())}"
                )

    # 统一短路上下文：6 个意图 handler 共用
    sc_ctx = ShortCircuitCtx(
        conversation_id=conversation_id,
        agent_id=agent_id,
        user_id=user_id,
        agent=agent,
        reply_context=reply_context,
        trace_ctx=_trace_ctx,
        trace_id=trace_id,
        end_trace_fn=_end_trace,
        save_replies_fn=_save_replies,
        pending_sub_fragments=pending_sub_fragments,
        sub_intent_mode=sub_intent_mode,
        reply_index_offset=reply_index_offset,
        cached_patience=cached_patience,
    )

    # §3.4.6 终结意图
    if detected_intent.intent == IntentType.CONVERSATION_END:
        async for evt in handle_conversation_end(user_message, sc_ctx, _intent_llm_reply):
            yield evt
        if workspace_id and agent_id and not sub_intent_mode:
            _fire_background(start_or_restart_proactive_session(
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id=agent_id,
                reason="farewell",
            ))
        return

    # §3.4.4 道歉承诺热路径
    if detected_intent.intent == IntentType.APOLOGY_PROMISE:
        handled, events = await handle_apology_promise(user_message, sc_ctx)
        if handled and events is not None:
            async for evt in events:
                yield evt
            return

    # §5 step 1-2 删除意图：找候选 → 请求确认
    elif detected_intent.intent == IntentType.DELETION:
        handled, events = await handle_deletion(user_message, sc_ctx)
        if handled and events is not None:
            async for evt in events:
                yield evt
            return

    # NOTE: SCHEDULE_ADJUST/SCHEDULE_QUERY/CURRENT_STATE 在 parallel data fetch 之后处理

    # Load recent messages (for prompt context)
    recent_messages = await db.message.find_many(
        where={"conversationId": conversation_id},
        order={"createdAt": "desc"},
        take=30,
    )
    recent_messages.reverse()

    messages_dicts = [
        {"role": m.role, "content": m.content} for m in recent_messages
    ]

    # --- Load previous user emotion from message metadata (no LLM) ---
    prev_user_emotion = None
    for m in reversed(recent_messages[:-1]):  # skip current message
        if m.role == "user" and m.metadata:
            meta = m.metadata if isinstance(m.metadata, dict) else {}
            if "emotion" in meta:
                prev_user_emotion = meta["emotion"]
                break

    # --- Quick keyword emotion estimate for current message (no LLM) ---
    current_user_emotion = quick_emotion_estimate(user_message)
    prompt_user_emotion = current_user_emotion or prev_user_emotion

    # --- Topic tracking (Redis, no LLM) ---
    topic_info = await push_topic(conversation_id, user_message)
    topic_context = format_topic_context(topic_info) if topic_info else None

    # --- Time system: parse explicit time expressions (PRD §9.3.2) ---
    parsed_times = parse_time_expressions(user_message) if has_explicit_time(user_message) else []

    # --- Pre-compute personality (MBTI) for downstream timing/emotion calls ---
    mbti = get_mbti(agent)

    # --- HOT PATH: parallel data fetches ---
    # Spec §3.1: classify memory relevance (强/中/弱) in parallel with other
    # fetches. Result gates whether retrieval results are injected into prompt.

    async def _classify_relevance():
        return await classify_memory_relevance(user_message)

    async def _do_retrieval():
        return await hybrid_retrieve(user_message, user_id, workspace_id=workspace_id)

    async def _load_cached_emotion():
        """Load last known emotion from cache/DB — no LLM call."""
        if agent_id:
            return await get_ai_emotion(agent_id)
        return None

    async def _load_cached_summaries():
        """Load previously cached summarizer results — no LLM call."""
        from app.services.summarizer import _conv_hash
        prev_msgs = messages_dicts[:-1]
        prev_user_content = next(
            (m["content"] for m in reversed(prev_msgs) if m["role"] == "user"),
            ""
        )
        ch = _conv_hash(prev_msgs, prev_user_content)
        return await cache_summarizer(ch)

    async def _load_portrait():
        """Load latest user portrait — no LLM call."""
        if agent_id:
            return await get_latest_portrait(user_id, agent_id)
        return None

    async def _load_schedule():
        """Load cached schedule for AI status context."""
        if agent_id:
            return await get_cached_schedule(agent_id)
        return None

    async def _load_topic_intimacy():
        """Load topic intimacy score for prompt injection."""
        if agent_id and user_id:
            return await get_topic_intimacy(agent_id, user_id)
        return 50.0

    async def _load_time_memories():
        """Load memories matching parsed time ranges (PRD §9.3.4)."""
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

    async def _load_working_facts():
        """Synchronously update hot-path working memory for the current user message."""
        return await update_working_facts(conversation_id, user_message)

    relevance_result, retrieval_result, emotion, summaries, portrait, schedule, topic_intimacy, time_memories_result, working_facts_result = await asyncio.gather(
        _classify_relevance(),
        _do_retrieval(),
        _load_cached_emotion(),
        _load_cached_summaries(),
        _load_portrait(),
        _load_schedule(),
        _load_topic_intimacy(),
        _load_time_memories(),
        _load_working_facts(),
        return_exceptions=True,
    )

    # Spec §3.1: determine memory relevance level
    memory_relevance = "medium"
    if isinstance(relevance_result, Exception):
        logger.warning(f"Memory relevance classification failed: {relevance_result}")
    elif isinstance(relevance_result, str):
        memory_relevance = relevance_result
    logger.info(f"[DEBUG-MEM] relevance='{memory_relevance}' for '{user_message[:60]}'")

    # Process retrieval results — gate by relevance
    classified_memories = None  # ClassifiedMemory list for prompt_builder
    memory_strings = None       # plain text list for summarizer/other consumers
    graph_context = None
    if memory_relevance == "weak":
        # Spec §3.4: weak relevance → don't inject any retrieved memories
        logger.info("[DEBUG-MEM] SKIPPED — weak relevance, no memories injected")
    elif isinstance(retrieval_result, Exception):
        logger.warning(f"Hybrid retrieval failed: {retrieval_result}")
    else:
        classified_memories = retrieval_result.get("memories")
        memory_strings = retrieval_result.get("memory_strings")
        graph_context = retrieval_result.get("graph_context")
        logger.info(f"[DEBUG-MEM] retrieval returned {len(classified_memories) if classified_memories else 0} memories")
        if classified_memories:
            for m in classified_memories[:5]:
                logger.info(f"[DEBUG-MEM]   sim={m.similarity:.3f} imp={m.importance:.2f} text='{m.text[:60]}'")

        # Spec §3.2/3.3: rerank by display_score and cap at top 10
        if classified_memories:
            for m in classified_memories:
                m.display_score = compute_display_score(
                    importance=getattr(m, "importance", 0.5),
                    last_accessed_at=getattr(m, "created_at", None),
                    similarity=getattr(m, "similarity", 0.8),
                )
            classified_memories.sort(key=lambda m: m.display_score, reverse=True)
            classified_memories = classified_memories[:10]
            logger.info(f"[DEBUG-MEM] after rerank, top {len(classified_memories)} injected into prompt:")
            for m in classified_memories[:5]:
                logger.info(f"[DEBUG-MEM]   ds={m.display_score:.3f} text='{m.text[:60]}'")
        else:
            logger.info("[DEBUG-MEM] no classified_memories from retrieval (empty result)")

    if isinstance(emotion, Exception):
        logger.warning(f"Loading cached emotion failed: {emotion}")
        emotion = None

    if isinstance(summaries, Exception):
        logger.warning(f"Loading cached summaries failed: {summaries}")
        summaries = None

    # spec §4 step 5 & §3.4.5: 强相关或调用久远记忆意图 → 小模型「调用L3」判断
    # 输出 "不满纠正" / "请求更久" / "无"。前两类触发 L3 召回。
    l3_memories: list[str] = []
    l3_trigger_label: str = "无"
    should_call_l3 = detected_intent.intent == IntentType.L3_RECALL
    if memory_relevance == "strong" or should_call_l3:
        try:
            l3_trigger_label = await _l3_trigger_classify(user_message)
        except Exception as e:
            logger.warning(f"L3 trigger classify failed: {e}")
            l3_trigger_label = "无"
        logger.info(f"[L3-TRIGGER] label='{l3_trigger_label}' for '{user_message[:40]}'")
        # §3.4.5 调用久远记忆意图 → 无论分类结果都召回；§4 强相关 → 仅前两类召回
        if should_call_l3 or l3_trigger_label in ("不满纠正", "请求更久"):
            l3_results = await search_l3_memories(user_message, user_id, workspace_id=workspace_id)
            l3_memories = [r.get("content") or r.get("summary", "") for r in l3_results if r]
            if l3_memories:
                logger.info(f"L3 awakening: {len(l3_memories)} memories injected "
                            f"(label='{l3_trigger_label}')")

    if isinstance(portrait, Exception):
        logger.warning(f"Loading portrait failed: {portrait}")
        portrait = None

    if isinstance(schedule, Exception):
        logger.warning(f"Loading schedule failed: {schedule}")
        schedule = None

    if isinstance(topic_intimacy, Exception):
        logger.warning(f"Loading topic intimacy failed: {topic_intimacy}")
        topic_intimacy = 50.0

    time_memories: list[str] = []
    if isinstance(time_memories_result, Exception):
        logger.warning(f"Loading time memories failed: {time_memories_result}")
    elif time_memories_result:
        time_memories = time_memories_result

    working_facts: list[str] | None = None
    if isinstance(working_facts_result, Exception):
        logger.warning(f"Loading working facts failed: {working_facts_result}")
    elif working_facts_result:
        working_facts = facts_for_prompt(working_facts_result)

    delay_context = None
    if reply_context:
        received_status = reply_context.get("received_status") or {}
        received_activity = str(received_status.get("activity", "")).strip() or "处理自己的事"
        received_status_label = str(received_status.get("status", "idle"))
        received_at = str(reply_context.get("received_at", ""))
        elapsed = actual_delay_seconds(reply_context)
        # spec §6.5: ≥1min 时会单独推送"延迟解释回复"，主回复不再重复注入解释
        if elapsed is not None and elapsed < 60:
            rounded_delay = max(1, round(elapsed))
            delay_reason_text = explain_delay_reason(
                str(reply_context.get("delay_reason", "")),
                activity=received_activity,
                status=received_status_label,
            )
            delay_context = (
                f"你在 {received_at} 收到用户消息时，正在{received_activity}"
                f"（状态：{received_status_label}）。\n"
                f"现在距离收到消息已经过去约 {rounded_delay} 秒。\n"
                f"{delay_reason_text}\n"
                "只有在确实需要时，才用半句自然带过刚才在忙什么；"
                "优先回应用户当下情绪或关系信号，解释不要压过聊天本身。"
            )

    relational_context = detect_relational_context(user_message, prompt_user_emotion)

    # --- Time context for prompt (PRD §9.2) ---
    time_context = build_time_context()

    # --- Intimacy stage for prompt (PRD §4.6.2.1) ---
    intimacy_stage = get_relationship_stage(topic_intimacy)

    # --- AI status context from schedule (pure computation) ---
    schedule_context = None
    ai_status = get_current_status(schedule) if schedule else None
    if ai_status:
        schedule_context = format_schedule_context(ai_status)

    # §3.4.2 作息调整
    if detected_intent.intent == IntentType.SCHEDULE_ADJUST:
        handled, events = await handle_schedule_adjust(
            user_message, sc_ctx,
            schedule=schedule, ai_status=ai_status,
            topic_intimacy=topic_intimacy, mbti=mbti,
        )
        if handled and events is not None:
            async for evt in events:
                yield evt
            return

    # §3.4.1 计划查询
    if detected_intent.intent == IntentType.SCHEDULE_QUERY:
        query_type = detected_intent.metadata.get("query_type", "current")
        handled, events, schedule_ctx_for_prompt = await handle_schedule_query(
            user_message, sc_ctx,
            schedule=schedule, ai_status=ai_status,
            portrait=portrait, user_emotion=prompt_user_emotion,
            query_type=query_type,
        )
        if schedule_ctx_for_prompt is not None:
            schedule_context = schedule_ctx_for_prompt  # 供下方 rich prompt 复用
        if handled and events is not None:
            async for evt in events:
                yield evt
            return

    # §3.4.3 询问当前状态
    if detected_intent.intent == IntentType.CURRENT_STATE:
        handled, events = await handle_current_state(
            user_message, sc_ctx,
            ai_status=ai_status, schedule_context=schedule_context,
            portrait=portrait, user_emotion=prompt_user_emotion,
        )
        if handled and events is not None:
            async for evt in events:
                yield evt
            return

    # 5B.4: Get patience prompt instruction (reuse value from check_boundary)
    patience_instruction = get_patience_prompt_instruction(cached_patience)

    # Spec §4 step 1-2: detect NEW contradictions (resolution already handled
    # at the top of the function via pending state check)
    contradiction_inquiry: str | None = None
    if memory_relevance in ("strong", "medium"):
        try:
            conflict = await detect_l1_contradiction(user_message, user_id, workspace_id=workspace_id)
            if conflict:
                inquiry = await generate_contradiction_inquiry(conflict, agent_name=agent.name if agent else "AI")
                contradiction_inquiry = inquiry
                await save_pending_contradiction(conversation_id, conflict)
                logger.info(f"L1 contradiction detected: {conflict.get('conflict_description', '')}")
        except Exception as e:
            logger.warning(f"Contradiction detection failed: {e}")

    # --- spec §5.5: n = random.randint(1, 3) 均匀分布 ---
    if relational_context:
        reply_count = 1
    elif contradiction_inquiry:
        reply_count = 1  # contradiction inquiry is a single focused question
    else:
        reply_count = random.randint(1, MAX_REPLY_COUNT)
    max_reply_count = MAX_REPLY_COUNT
    max_total = MAX_TOTAL_CHARS

    # Build prompt (pure string operations — instant)
    system_prompt = await build_system_prompt(
        agent=agent,
        memories=classified_memories,
        working_facts=working_facts,
        delay_context=delay_context,
        relational_context=relational_context,
        emotion=emotion,
        graph_context=graph_context,
        summaries=summaries,
        portrait=portrait,
        topic_context=topic_context,
        user_emotion=prompt_user_emotion,
        schedule_context=schedule_context,
        patience_instruction=patience_instruction,
        reply_count=reply_count,
        reply_total=max_total,
        intimacy_stage=intimacy_stage,
        time_context=time_context,
        time_memories=time_memories or None,
        l3_memories=l3_memories or None,
    )
    chat_messages = build_chat_messages(system_prompt, messages_dicts)

    # Log memory access for L2 frequency tracking (background, non-blocking)
    accessed_ids: list[str] = []
    if classified_memories:
        accessed_ids.extend(getattr(m, "id", "") for m in classified_memories if getattr(m, "id", ""))
    if accessed_ids:
        _fire_background(log_memory_access(user_id, accessed_ids, workspace_id=workspace_id))

    # --- Send typing event before response ---
    typing_duration = calculate_typing_duration(len(user_message))

    # Delay decision is frozen at receipt time via reply_context.
    # For live WS/SSE, we keep a short blocking sleep while exposing the conceptual delay.
    reply_delay = calculate_reply_delay(len(user_message), mbti=mbti)
    queued_delay = float((reply_context or {}).get("delay_seconds", 0.0) or 0.0)
    conceptual_delay = max(reply_delay, queued_delay)
    if delivered_from_queue:
        actual_sleep = min(reply_delay, 1.5)
    else:
        actual_sleep = min(conceptual_delay, 2.0)
        if conceptual_delay > 5.0:
            yield {"event": "delay", "data": json.dumps({"duration": conceptual_delay})}
    yield {"event": "typing", "data": json.dumps({"duration": typing_duration})}
    await asyncio.sleep(actual_sleep)

    replies, raw_response = await _generate_reply(
        contradiction_inquiry=contradiction_inquiry,
        detected_intent=detected_intent,
        memory_relevance=memory_relevance,
        relational_context=relational_context,
        schedule_context=schedule_context,
        delay_context=delay_context,
        l3_memories=l3_memories,
        classified_memories=classified_memories or [],
        messages_dicts=messages_dicts,
        portrait=portrait,
        prompt_user_emotion=prompt_user_emotion,
        user_message=user_message,
        agent=agent,
        chat_messages=chat_messages,
        reply_count=reply_count,
        max_reply_count=max_reply_count,
        max_total=max_total,
        tier_fns={
            "weak": _memory_weak_reply,
            "medium": _memory_medium_reply,
            "strong": _memory_strong_reply,
            "l3": _memory_l3_reply,
        },
        split_llm_fn=_split_reply_to_n_sentences,
        truncate_fn=truncate_at_sentence,
        pipe_fallback_fn=split_and_validate_replies,
    )

    # spec §5 step 1：AI 语句情绪识别（基于回复文本，不是 AI PAD 缓存）
    full_response = " ".join(replies)
    reply_emotion = await _ai_reply_emotion(full_response)
    if reply_emotion.get("emotion"):
        logger.info(
            f"[REPLY-EMO] emotion={reply_emotion['emotion']} "
            f"intensity={reply_emotion.get('intensity', 0)}"
        )

    # spec §5/§6.4-§6.5: emoji/sticker + 延迟解释 + 推送
    emitted_replies: list[dict] = []
    async for evt in _emit_replies(
        replies,
        reply_context=reply_context,
        reply_index_offset=reply_index_offset,
        sub_intent_mode=sub_intent_mode,
        emotion=emotion,
        agent=agent,
        user_message=user_message,
        delay_reply_fn=_delay_explanation_reply,
        fallback_fn=_intent_llm_reply,
        emitted_replies=emitted_replies,
        reply_emotion=reply_emotion,
    ):
        yield evt

    # Persist replies immediately; trace links become clickable only after public share completes.
    first_assistant_message_id = await _save_replies(
        conversation_id,
        emitted_replies,
        trace_id=trace_id if settings.langsmith_tracing else None,
    )

    if sub_intent_mode:
        # 父调用负责后台任务、save_last_reply_timestamp、done、trace 关闭
        return

    # Update conversation title if first exchange (non-blocking)
    if len(recent_messages) <= 1:
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        _fire_background(db.conversation.update(
            where={"id": conversation_id},
            data={"title": title},
        ))

    # --- BACKGROUND: fire-and-forget post-processing ---
    _fire_background(_background_post_process(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        user_message=user_message,
        user_message_id=user_message_id,
        full_response=full_response,
        messages_dicts=messages_dicts,
        memory_strings=memory_strings,
        cached_emotion=emotion,
        mbti=mbti,
        topic_intimacy=topic_intimacy,
    ))

    # spec §3.3 step 3: 主意图回复完成后，依次处理拆分出的子意图片段
    if pending_sub_fragments:
        start_idx = reply_index_offset + len(emitted_replies)
        async for evt in _process_sub_intents(
            pending_sub_fragments, conversation_id, agent, user_id,
            reply_context, start_index=start_idx,
            parent_patience=cached_patience,
        ):
            yield evt

    await save_last_reply_timestamp(agent_id, user_id)
    if workspace_id and agent_id:
        _fire_background(start_or_restart_proactive_session(
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            agent_id=agent_id,
            reason="conversation_end",
        ))
    done_data: dict = {"message_id": "complete"}
    if first_assistant_message_id and trace_id and settings.langsmith_tracing:
        done_data["assistant_message_id"] = first_assistant_message_id
        done_data["trace_pending"] = True
    yield {"event": "done", "data": json.dumps(done_data)}

    # End trace and share publicly in background (updates DB with public URL)
    _end_trace(_trace_ctx, trace_id, conversation_id)


async def _background_post_process(
    user_id: str,
    agent_id: str | None,
    conversation_id: str,
    user_message: str,
    user_message_id: str | None,
    full_response: str,
    messages_dicts: list[dict],
    memory_strings: list[str] | None,
    cached_emotion: dict | None = None,
    mbti: dict | None = None,
    topic_intimacy: float = 50.0,
) -> None:
    """Run all background tasks after response is sent.

    1. Emotion extraction + state update (small model)
    2. Summarizer → cache to Redis for next request (small model)
    3. Memory extraction pipeline (small model)
    """
    try:
        # Add the assistant response to messages for summarizer context
        full_messages = messages_dicts + [{"role": "assistant", "content": full_response}]

        # Run all background tasks concurrently
        tasks = [
            _bg_emotion(agent_id, user_message_id, user_message, cached_emotion, topic_intimacy, mbti),
            _bg_summarizer(full_messages, user_message, memory_strings),
            _bg_memory_pipeline(user_id, full_messages),
        ]
        # 道歉和删除检查已在热路径处理，不再后台重复执行
        if agent_id:
            tasks.append(_bg_trait_adjustment(agent_id, user_message))
        # Positive recovery: messages reaching here passed boundary check (no banned words)
        if agent_id:
            tasks.append(_bg_positive_recovery(agent_id, user_id))
        # AI self-memory: now handled by _bg_memory_pipeline which processes
        # both user and AI messages through the unified 3-step pipeline (spec §2.2).
        # The extraction prompt distinguishes owner=user vs owner=ai.
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Background post-processing failed: {e}")


async def _bg_emotion(
    agent_id: str | None,
    user_message_id: str | None,
    user_message: str,
    cached_emotion: dict | None = None,
    topic_intimacy: float = 50.0,
    mbti: dict | None = None,
) -> None:
    """Extract emotion from user message, update AI emotion state, and save to message metadata."""
    if not agent_id:
        return
    try:
        user_emotion = await extract_emotion(user_message)
        current_emotion = cached_emotion or await get_ai_emotion(agent_id)
        new_emotion = update_emotion_state(current_emotion, user_emotion, topic_intimacy, mbti=mbti)
        await save_ai_emotion(agent_id, new_emotion)

        # Save user emotion to message metadata (use known ID, no re-query)
        if user_message_id:
            await db.message.update(
                where={"id": user_message_id},
                data={"metadata": Json({"emotion": user_emotion})},
            )
    except Exception as e:
        logger.warning(f"Background emotion update failed: {e}")


async def _bg_summarizer(
    messages: list[dict],
    current_message: str,
    memories: list[str] | None,
) -> None:
    """Run 3-layer summarizer and cache results for next request."""
    try:
        result = await summarize(messages, current_message, memories)
        if result:
            logger.debug("Background summarizer completed and cached")
    except Exception as e:
        logger.warning(f"Background summarizer failed: {e}")


async def _save_replies(
    conversation_id: str,
    replies: list[str | dict],
    trace_id: str | None = None,
) -> str | None:
    """Save split replies as individual DB messages."""
    try:
        first_message_id: str | None = None
        for i, reply in enumerate(replies):
            if isinstance(reply, dict):
                text = str(reply.get("text", ""))
                metadata: dict = {"reply_index": i}
                # 允许任意白名单之外的 metadata 合并进来（如 boundary/zone/attack_level/sticker_url）
                for k, v in reply.items():
                    if k not in ("text", "index") and v is not None:
                        metadata[k] = v
            else:
                text = reply
                metadata = {"reply_index": i}

            # First reply carries trace-pending metadata until a public LangSmith link is ready.
            if i == 0 and trace_id:
                metadata["trace_id"] = trace_id
                metadata["trace_pending"] = True

            created = await db.message.create(
                data={
                    "conversation": {"connect": {"id": conversation_id}},
                    "role": "assistant",
                    "content": text,
                    "metadata": Json(metadata),
                }
            )
            if i == 0:
                first_message_id = created.id
        return first_message_id
    except Exception as e:
        logger.error(f"Failed to save replies: {e}")
        return None


async def _bg_memory_pipeline(user_id: str, messages: list[dict]) -> None:
    """Run memory extraction pipeline for BOTH user and AI messages.

    Spec §2.1/§2.2: both user and AI messages go through the same 3-step
    pipeline (filter → small model → big model). The extraction prompt
    distinguishes owner (user vs ai) via the "owner" field in output.

    Input: last 3 rounds of dialogue (user+assistant alternating), which
    gives the big model enough context per spec §2.1.3.
    """
    try:
        # Spec §2.1.3: "用户消息 + 最近3轮对话上下文"
        # Take last 6 messages (3 rounds of user+assistant)
        recent = messages[-6:]
        if not recent:
            return
        conv_text = "\n".join(
            f"{m.get('role', 'user')}: {m['content']}" for m in recent
        )
        await process_memory_pipeline(user_id, conv_text)
    except Exception as e:
        logger.error(f"Background memory pipeline failed: {e}")





async def _bg_trait_adjustment(agent_id: str, user_message: str) -> None:
    """Check for trait adjustment signals in user message."""
    try:
        adjustments = detect_direct_feedback(user_message) or infer_feedback(user_message)
        if adjustments:
            await apply_trait_adjustment(agent_id, adjustments)
    except Exception as e:
        logger.warning(f"Background trait adjustment failed: {e}")


async def _bg_apology_check(agent_id: str, user_id: str, user_message: str) -> None:
    """Check if user is apologizing and restore patience."""
    try:
        result = await detect_apology(user_message)
        if result.get("is_apology") and result.get("sincerity", 0) >= 0.8:
            new_patience = await handle_apology(agent_id, user_id)
            logger.info(f"Apology detected: patience restored to {new_patience}")
    except Exception as e:
        logger.warning(f"Background apology check failed: {e}")


async def _bg_positive_recovery(agent_id: str, user_id: str) -> None:
    """Positive interaction recovery. Only called for messages that passed boundary check."""
    try:
        await check_positive_recovery(agent_id, user_id)
    except Exception as e:
        logger.warning(f"Background positive recovery failed: {e}")
