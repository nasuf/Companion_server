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
from datetime import datetime

from prisma import Json

from app.config import settings
from app.db import db
from app.services.llm.models import get_chat_model, convert_messages
from app.services.prompt_builder import build_system_prompt, build_chat_messages
from app.services.prompts.system_prompts import (
    MAX_PER_REPLY, MAX_REPLY_COUNT, EXPAND_MAX_REPLY_COUNT,
    MAX_TOTAL_CHARS, EXPAND_MAX_TOTAL_CHARS,
)
from app.services.memory.hybrid_retrieval import hybrid_retrieve
from app.services.memory.pipeline import process_memory_pipeline
from app.services.summarizer import summarize
from app.services.emotion import (
    extract_emotion,
    get_ai_emotion,
    quick_emotion_estimate,
    update_emotion_state,
    save_ai_emotion,
)
from app.services.cache import cache_summarizer
from app.services.portrait import get_latest_portrait
from app.services.timing import (
    calculate_reply_delay, calculate_typing_duration,
    explain_delay_reason,
)
from app.services.memory.deletion import detect_deletion_intent, delete_memories_by_description, DELETION_KEYWORDS
from app.services.topic import push_topic, format_topic_context
from app.services.schedule import get_cached_schedule, get_current_status, format_schedule_context
from app.services.time_service import build_time_context
from app.services.time_parser import parse_time_expressions, has_explicit_time
from app.services.boundary import (
    check_boundary, process_boundary_violation, detect_apology, handle_apology,
    has_apology_keyword, check_positive_recovery, get_patience_prompt_instruction,
    PATIENCE_MAX,
)
from app.services.intimacy import get_topic_intimacy, get_relationship_stage
from app.services.trait_adjustment import infer_feedback, detect_direct_feedback, apply_trait_adjustment
from app.services.conversation_end import check_conversation_end
from app.services.emoji import should_add_emoji, should_add_sticker, pick_one_emoji
from app.services.sticker import recommend_sticker
from app.services.trait_model import get_seven_dim
from app.services.fast_fact import update_working_facts, facts_for_prompt
from app.services.reply_context import actual_delay_seconds, save_last_reply_timestamp

logger = logging.getLogger(__name__)


def _on_task_error(t: asyncio.Task) -> None:
    """Log unhandled exceptions from background tasks."""
    if not t.cancelled() and t.exception():
        logger.error(f"Background post-processing failed: {t.exception()}")


def _fire_background(coro) -> None:
    """Schedule a background coroutine as a fire-and-forget task."""
    task = asyncio.create_task(coro)
    task.add_done_callback(_on_task_error)


# --- Multi-reply split & validate (PRD §3.2.1/§3.2.2) ---

_SENTENCE_END = re.compile(r'[。！？…～~!?]+')

_MORE_KEYWORDS = ["多说", "详细", "展开", "继续说", "说多点", "多聊聊"]
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


def detect_special_expand(message: str, emotion: dict | None) -> bool:
    """检测是否需要放宽回复限制（用户要求多说 或 高唤醒+负面情绪）。"""
    if any(kw in message for kw in _MORE_KEYWORDS):
        return True
    if emotion:
        if emotion.get("arousal", 0.5) > 0.7 and emotion.get("pleasure", 0.0) < -0.3:
            return True
    return False


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
            from app.services.ws_manager import manager

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
) -> AsyncGenerator[dict, None]:
    # Save user message unless it has already been persisted at receipt time.
    if save_user_message:
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

    # --- Boundary check (keyword + Redis, no LLM) ---
    cached_patience = PATIENCE_MAX
    if agent_id:
        boundary_result, cached_patience = await check_boundary(agent_id, user_id, user_message)
        if boundary_result:
            response = boundary_result["response"]
            await db.message.create(
                data={
                    "conversation": {"connect": {"id": conversation_id}},
                    "role": "assistant",
                    "content": response,
                    "metadata": Json({"boundary": True, "zone": boundary_result["zone"]}),
                }
            )
            yield {"event": "reply", "data": json.dumps({"text": response, "index": 0})}
            await save_last_reply_timestamp(agent_id, user_id)
            yield {"event": "done", "data": json.dumps({"message_id": "complete"})}
            # Background: classify + deduct patience
            _fire_background(process_boundary_violation(agent_id, user_id, user_message))
            # 记忆管道（攻击性消息也需记录）
            _fire_background(_bg_memory_pipeline(user_id, [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response},
            ]))
            # 拉黑状态下检测道歉（PRD §6.6.2.2）
            if boundary_result.get("zone") == "blocked" and has_apology_keyword(user_message):
                _fire_background(_bg_apology_check(agent_id, user_id, user_message))
            _end_trace(_trace_ctx, trace_id, conversation_id)
            return

    # --- Conversation end detection (PRD §3.2.3, keyword trigger + LLM farewell) ---
    if check_conversation_end(user_message):
        farewell_prompt = await build_system_prompt(agent=agent, reply_count=1, reply_total=60)
        farewell_prompt += (
            "\n\n## 特殊指令\n"
            "用户要结束对话了。用你的性格风格生成一句简短的道别，不超过30字。不要用||分隔。"
        )
        model = get_chat_model()
        result = await model.ainvoke(convert_messages([
            {"role": "system", "content": farewell_prompt},
            {"role": "user", "content": user_message},
        ]))
        farewell = result.content.strip().split("||")[0][:60]
        _fire_background(_save_replies(conversation_id, [farewell]))
        yield {"event": "reply", "data": json.dumps({"text": farewell, "index": 0})}
        await save_last_reply_timestamp(agent_id, user_id)
        yield {"event": "done", "data": json.dumps({"message_id": "complete"})}
        _end_trace(_trace_ctx, trace_id, conversation_id)
        return

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

    # --- Check deletion intent (keyword-only, no LLM on hot path) ---
    has_deletion_keyword = any(kw in user_message for kw in DELETION_KEYWORDS)

    # --- Topic tracking (Redis, no LLM) ---
    topic_info = await push_topic(conversation_id, user_message)
    topic_context = format_topic_context(topic_info) if topic_info else None

    # --- Time system: parse explicit time expressions (PRD §9.3.2) ---
    parsed_times = parse_time_expressions(user_message) if has_explicit_time(user_message) else []

    # --- Pre-compute personality and topic fatigue (needed after parallel fetch) ---
    agent_personality = getattr(agent, "personality", None) or {}
    seven_dim = get_seven_dim(agent)

    # --- HOT PATH: parallel data fetches (no LLM calls) ---
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

    async def _load_core_memories():
        """Load L1 core memories (user only) — always present in prompt."""
        from app.services.memory.core_memory import load_core_memory_strings
        rows = await load_core_memory_strings(
            user_id=user_id,
            workspace_id=workspace_id,
            source="user",
        )
        return rows if rows else None

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
        from app.services.memory.vector_search import search_by_time_range
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

    retrieval_result, emotion, summaries, core_memories, portrait, schedule, topic_intimacy, time_memories_result, working_facts_result = await asyncio.gather(
        _do_retrieval(),
        _load_cached_emotion(),
        _load_cached_summaries(),
        _load_core_memories(),
        _load_portrait(),
        _load_schedule(),
        _load_topic_intimacy(),
        _load_time_memories(),
        _load_working_facts(),
        return_exceptions=True,
    )

    # Process retrieval results
    memory_strings = None
    graph_context = None
    if isinstance(retrieval_result, Exception):
        logger.warning(f"Hybrid retrieval failed: {retrieval_result}")
    else:
        memory_strings = retrieval_result.get("memories")
        graph_context = retrieval_result.get("graph_context")

    if isinstance(emotion, Exception):
        logger.warning(f"Loading cached emotion failed: {emotion}")
        emotion = None

    if isinstance(summaries, Exception):
        logger.warning(f"Loading cached summaries failed: {summaries}")
        summaries = None

    if isinstance(core_memories, Exception):
        logger.warning(f"Loading core memories failed: {core_memories}")
        core_memories = None

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
        if elapsed is not None:
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

    # 5B.4: Get patience prompt instruction (reuse value from check_boundary)
    patience_instruction = get_patience_prompt_instruction(cached_patience)

    # --- Multi-reply parameters (PRD §3.2.1) ---
    is_expand = detect_special_expand(user_message, emotion)
    if relational_context:
        reply_count = 1
    else:
        reply_count = random.randint(1, MAX_REPLY_COUNT)
    max_reply_count = EXPAND_MAX_REPLY_COUNT if is_expand else MAX_REPLY_COUNT
    max_total = EXPAND_MAX_TOTAL_CHARS if is_expand else MAX_TOTAL_CHARS

    # Build prompt (pure string operations — instant)
    system_prompt = await build_system_prompt(
        agent=agent,
        memories=memory_strings,
        working_facts=working_facts,
        delay_context=delay_context,
        relational_context=relational_context,
        core_memories=core_memories,
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
    )
    chat_messages = build_chat_messages(system_prompt, messages_dicts)

    # --- Send typing event before response ---
    typing_duration = calculate_typing_duration(len(user_message))

    # Delay decision is frozen at receipt time via reply_context.
    # For live WS/SSE, we keep a short blocking sleep while exposing the conceptual delay.
    reply_delay = calculate_reply_delay(len(user_message), agent_personality, seven_dim=seven_dim)
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

    # --- Collect full LLM response (the ONLY LLM call in hot path) ---
    model = get_chat_model()
    lc_messages = convert_messages(chat_messages)

    response_chunks: list[str] = []
    async for chunk in model.astream(lc_messages):
        token = chunk.content
        if token:
            response_chunks.append(token)

    raw_response = "".join(response_chunks)

    # --- Split & validate into multiple replies (PRD §3.2.1/§3.2.2) ---
    replies = split_and_validate_replies(raw_response, max_reply_count, MAX_PER_REPLY, max_total)

    # --- Yield reply events with emoji/sticker (PRD §3.3.2/§3.3.3) ---
    emo = emotion if isinstance(emotion, dict) else {}
    ai_arousal = emo.get("arousal", 0.0)
    ai_pleasure = emo.get("pleasure", 0.0)
    ai_dominance = emo.get("dominance", 0.5)
    ai_primary_emotion = emo.get("primary_emotion")

    sticker_used = False  # 一个回合最多一个表情包

    emitted_replies: list[dict] = []

    for i, reply_text in enumerate(replies):
        added_emoji = False
        # PRD §3.3.2: emoji概率
        if should_add_emoji(ai_arousal):
            emoji = pick_one_emoji(ai_pleasure, ai_arousal, ai_primary_emotion)
            if emoji:
                reply_text += emoji
                added_emoji = True

        # PRD §3.3.3: 表情包互斥 — 该条未加emoji且本回合未用过表情包
        sticker_url = None
        if not added_emoji and not sticker_used and should_add_sticker(ai_arousal):
            try:
                result = await recommend_sticker(ai_pleasure, ai_arousal, ai_dominance, ai_primary_emotion)
                if result:
                    sticker_url = result["url"]
                    sticker_used = True
            except Exception:
                pass

        if i > 0:
            await asyncio.sleep(random.uniform(0.3, 0.8))

        data: dict = {"text": reply_text, "index": i}
        if sticker_url:
            data["sticker_url"] = sticker_url
        emitted_replies.append(data)
        yield {"event": "reply", "data": json.dumps(data)}

    full_response = " ".join(replies)

    # Update conversation title if first exchange (non-blocking)
    if len(recent_messages) <= 1:
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        _fire_background(db.conversation.update(
            where={"id": conversation_id},
            data={"title": title},
        ))

    # --- BACKGROUND: fire-and-forget post-processing ---
    # Background tasks inherit trace context via asyncio.create_task context copy,
    # so their LLM calls appear as children in LangSmith.
    _fire_background(_background_post_process(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        user_message=user_message,
        user_message_id=user_message_id,
        full_response=full_response,
        messages_dicts=messages_dicts,
        memory_strings=memory_strings,
        has_deletion_keyword=has_deletion_keyword,
        cached_emotion=emotion,
        seven_dim=seven_dim,
        topic_intimacy=topic_intimacy,
    ))

    # Persist replies immediately; trace links become clickable only after public share completes.
    first_assistant_message_id = await _save_replies(
        conversation_id,
        emitted_replies,
        trace_id=trace_id if settings.langsmith_tracing else None,
    )

    await save_last_reply_timestamp(agent_id, user_id)
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
    has_deletion_keyword: bool = False,
    cached_emotion: dict | None = None,
    seven_dim: dict | None = None,
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
            _bg_emotion(agent_id, user_message_id, user_message, cached_emotion, topic_intimacy, seven_dim),
            _bg_summarizer(full_messages, user_message, memory_strings),
            _bg_memory_pipeline(user_id, full_messages),
        ]
        if has_deletion_keyword:
            tasks.append(_bg_deletion_check(user_id, user_message))
        if agent_id and has_apology_keyword(user_message):
            tasks.append(_bg_apology_check(agent_id, user_id, user_message))
        if agent_id:
            tasks.append(_bg_trait_adjustment(agent_id, user_message))
        # Positive recovery: messages reaching here passed boundary check (no banned words)
        if agent_id:
            tasks.append(_bg_positive_recovery(agent_id, user_id))
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Background post-processing failed: {e}")


async def _bg_emotion(
    agent_id: str | None,
    user_message_id: str | None,
    user_message: str,
    cached_emotion: dict | None = None,
    topic_intimacy: float = 50.0,
    seven_dim: dict | None = None,
) -> None:
    """Extract emotion from user message, update AI emotion state, and save to message metadata."""
    if not agent_id:
        return
    try:
        user_emotion = await extract_emotion(user_message)
        current_emotion = cached_emotion or await get_ai_emotion(agent_id)
        new_emotion = update_emotion_state(current_emotion, user_emotion, topic_intimacy, seven_dim=seven_dim)
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
                sticker_url = reply.get("sticker_url")
                metadata: dict = {"reply_index": i}
                if sticker_url:
                    metadata["sticker_url"] = sticker_url
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
    """Run memory extraction pipeline."""
    try:
        conv_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages[-10:]
        )
        await process_memory_pipeline(user_id, conv_text)
    except Exception as e:
        logger.error(f"Background memory pipeline failed: {e}")


async def _bg_deletion_check(user_id: str, user_message: str) -> None:
    """Check if user wants to delete a memory and execute deletion."""
    try:
        intent = await detect_deletion_intent(user_message)
        if intent and intent.get("target_description"):
            deleted = await delete_memories_by_description(
                user_id, intent["target_description"]
            )
            logger.info(f"Deletion check: removed {deleted} memories for user {user_id}")
    except Exception as e:
        logger.warning(f"Background deletion check failed: {e}")


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
