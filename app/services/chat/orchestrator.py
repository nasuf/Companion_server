"""Chat service — optimized for low latency.

Hot path (user-facing, ~2s):
  save msg → parallel(vector retrieval + load cached context) → prompt → stream LLM

Background (fire-and-forget, after response):
  user PAD metadata + AI PAD cache write + memory pipeline + trait/patience updates
"""

import asyncio
import json
import logging
import random
import re
from collections.abc import AsyncGenerator

from app.db import db
from app.services.llm.models import get_chat_model, convert_messages
from app.services.chat.prompt_builder import build_system_prompt, build_chat_messages
from app.services.prompts.system_prompts import (
    MAX_PER_REPLY, MAX_REPLY_COUNT, MAX_TOTAL_CHARS,
)
from app.services.schedule_domain.timing import (
    calculate_reply_delay, calculate_typing_duration,
    explain_delay_reason,
)
from app.services.memory.interaction.contradiction import (
    detect_l1_contradiction, generate_contradiction_inquiry,
    save_pending_contradiction,
)  # analyze/apply/load/clear 已由 preflight.resolve_pending_contradiction 接管
from app.services.memory.retrieval.access_log import log_memory_access
from app.services.topic import push_topic, format_topic_context
from app.services.schedule_domain.time_service import build_time_context
from app.services.schedule_domain.time_parser import parse_time_expressions, has_explicit_time
from app.services.interaction.boundary import get_patience_prompt_instruction
from app.services.relationship.intimacy import get_relationship_stage
from app.services.chat.intent_dispatcher import (
    detect_intent_unified, IntentType, IntentResult, LABEL_TO_INTENT,
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
from app.services.chat.data_fetch_phase import fetch_parallel_context
from app.services.chat.post_process import (
    save_replies as _save_replies,
    run_post_process as _background_post_process,
    _bg_memory_pipeline,
)
from app.services.chat.tracing import LangSmithTracer
from app.services.mbti import get_mbti
from app.services.interaction.reply_context import actual_delay_seconds, save_last_reply_timestamp
from app.services.proactive.state import start_or_restart_proactive_session
from app.services.runtime.tasks import fire_background as _fire_background

logger = logging.getLogger(__name__)


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


_INTENT_CONTEXT_WINDOW = 6  # 最近 6 条消息 (3-4 轮对话), spec §3.3 "及上下文"


async def _fetch_intent_context(
    conversation_id: str,
    *,
    exclude_id: str | None = None,
    exclude_content: str | None = None,
) -> str:
    """拉最近 N 条消息拼成意图识别 prompt 的上下文段落。

    spec §3.3 step 1 要求识别 "用户消息及上下文". 常见场景:
    AI 问 "要我再陪你一会儿吗?" + 用户回 "好" — 必须结合 AI 上一句
    才能判定用户 "好" 是 作息调整 意图.

    当前消息已经作为 {user_message} 传给 prompt, 不应再出现在 context 里.
    优先用 `exclude_id` (已落库场景) 精确排除; 若消息尚未入库或只有内容,
    回退到 `exclude_content` 字符串匹配 (仅第一条命中者).
    """
    try:
        rows = await db.message.find_many(
            where={"conversationId": conversation_id},
            order={"createdAt": "desc"},
            take=_INTENT_CONTEXT_WINDOW + 1,
        )
    except Exception as e:
        logger.debug(f"intent context fetch failed: {e}")
        return ""

    # Prisma desc 排序下, 首条即最新; 当前消息通常在这里.
    # 回退按内容匹配时只过滤第一条 (即最新那条) 命中的用户消息,
    # 避免用户连发两条相同短消息 ("好" / "好") 把上一轮的 "好" 也丢掉.
    lines: list[str] = []
    content_fallback_consumed = False
    for row in rows:
        content = (getattr(row, "content", "") or "").strip()
        if not content:
            continue
        role = "AI" if getattr(row, "role", "") == "assistant" else "用户"
        if exclude_id and getattr(row, "id", None) == exclude_id:
            continue
        if (
            not exclude_id
            and exclude_content
            and not content_fallback_consumed
            and role == "用户"
            and content == exclude_content
        ):
            content_fallback_consumed = True
            continue
        lines.append(f"{role}: {content}")

    # desc 顺序 → 反转为时间顺序, 只保留最近 N 条
    lines.reverse()
    lines = lines[-_INTENT_CONTEXT_WINDOW:]
    return "\n".join(lines)


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
    tracer = LangSmithTracer(user_message, conversation_id).enter()

    # spec §2.6 边界系统全流程（含步骤 2-6 + 步骤 6 中/低耐心短路）
    boundary_ctx = BoundaryPhaseCtx(
        conversation_id=conversation_id,
        agent_id=agent_id,
        user_id=user_id,
        agent=agent,
        user_message=user_message,
        sub_intent_mode=sub_intent_mode,
        parent_patience=parent_patience,
        tracer=tracer,
        short_circuit_fn=_short_circuit_reply,
        fire_background_fn=_fire_background,
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
            tracer=tracer,
            short_circuit_fn=_short_circuit_reply,
        )

        async for evt in resolve_pending_contradiction(user_message, preflight_ctx):
            yield evt
        if preflight_ctx.stopped:
            return

        async for evt in resolve_pending_deletion(user_message, preflight_ctx):
            yield evt
        if preflight_ctx.stopped:
            return

    # --- 统一意图识别：spec §3.3 step 1 严格实现 ---
    # 每条用户消息都调小模型做意图分类, 并把最近对话历史作为上下文注入.
    # 不再区分消息长度 — 短消息如 "好" / "嗯" 只有结合 AI 上一句
    # ("要我再陪你一会儿吗?") 才能识别出 "作息调整" 意图.
    if forced_intent is not None:
        # sub_intent_mode 的子片段, 意图由父调用指定, 不再识别
        detected_intent = IntentResult(intent=forced_intent, confidence=1.0)
    else:
        context_text = await _fetch_intent_context(
            conversation_id,
            exclude_id=user_message_id,
            exclude_content=user_message if not user_message_id else None,
        )
        detected_intent = await detect_intent_unified(user_message, context=context_text)
        if detected_intent.intent != IntentType.NONE:
            logger.info(
                f"[INTENT-LLM] '{user_message[:30]}' → {detected_intent.intent.value} "
                f"(labels={detected_intent.metadata.get('llm_labels')})"
            )
    if forced_intent is None:
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
        tracer=tracer,
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

    # --- Topic tracking (Redis, no LLM) ---
    topic_info = await push_topic(conversation_id, user_message)
    topic_context = format_topic_context(topic_info) if topic_info else None

    # --- Time system: parse explicit time expressions (PRD §9.3.2) ---
    parsed_times = parse_time_expressions(user_message) if has_explicit_time(user_message) else []

    # --- Pre-compute personality (MBTI) for downstream timing/emotion calls ---
    mbti = get_mbti(agent)

    # spec §3.1+§3.2 step 2-3: 并行拉取记忆/情绪/画像/作息 + L3 awakening
    fetched = await fetch_parallel_context(
        user_id=user_id, agent_id=agent_id, workspace_id=workspace_id,
        user_message=user_message,
        messages_dicts=messages_dicts, parsed_times=parsed_times,
        detected_intent=detected_intent,
        l3_trigger_classify_fn=_l3_trigger_classify,
    )
    memory_relevance = fetched.memory_relevance
    classified_memories = fetched.classified_memories
    emotion = fetched.emotion
    prompt_user_emotion = fetched.user_emotion
    portrait = fetched.portrait
    schedule = fetched.schedule
    topic_intimacy = fetched.topic_intimacy
    time_memories = fetched.time_memories
    l3_memories = fetched.l3_memories
    ai_status = fetched.ai_status
    schedule_context = fetched.schedule_context

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

    # ai_status / schedule_context 已由 fetch_parallel_context 在上面计算并赋值

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
        delay_context=delay_context,
        relational_context=relational_context,
        graph_context=fetched.graph_context,
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
        trace_id=tracer.trace_id if tracer.is_active else None,
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
        user_emotion=prompt_user_emotion,
        ai_emotion=emotion,
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
    if first_assistant_message_id and tracer.trace_id and tracer.is_active:
        done_data["assistant_message_id"] = first_assistant_message_id
        done_data["trace_pending"] = True
    yield {"event": "done", "data": json.dumps(done_data)}

    # End trace and share publicly in background (updates DB with public URL)
    tracer.close()



