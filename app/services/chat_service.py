"""Chat service — optimized for low latency.

Hot path (user-facing, ~2s):
  save msg → parallel(vector retrieval + load cached context) → prompt → stream LLM

Background (fire-and-forget, after response):
  emotion extraction + summarizer update + memory pipeline
"""

import asyncio
import json
import logging
import random
import re
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from prisma import Json

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
    compute_message_interval_delay,
)
from app.services.memory.deletion import detect_deletion_intent, delete_memories_by_description, DELETION_KEYWORDS
from app.services.topic import push_topic, format_topic_context
from app.services.schedule import get_cached_schedule, get_current_status, format_schedule_context
from app.services.boundary import (
    check_boundary, process_boundary_violation, detect_apology, handle_apology,
    has_apology_keyword, check_positive_recovery, get_patience_prompt_instruction,
    PATIENCE_MAX,
)
from app.services.intimacy import get_topic_intimacy, get_relationship_stage
from app.services.trait_adjustment import infer_feedback, detect_direct_feedback, apply_trait_adjustment
from app.services.conversation_end import check_conversation_end
from app.services.emoji import should_add_emoji, pick_one_emoji
from app.services.trait_model import get_seven_dim

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


async def stream_chat_response(
    conversation_id: str,
    user_message: str,
    agent,
    user_id: str,
) -> AsyncGenerator[dict, None]:
    # Save user message
    saved_msg = await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "user",
            "content": user_message,
        }
    )
    user_message_id = saved_msg.id

    agent_id = getattr(agent, "id", None)

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
            return

    # --- Conversation end detection (PRD §3.2.3, keyword trigger + LLM farewell) ---
    if check_conversation_end(user_message):
        farewell_prompt = build_system_prompt(agent=agent, reply_count=1, reply_total=60)
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
        yield {"event": "done", "data": json.dumps({"message_id": "complete"})}
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
    if prev_user_emotion is None:
        prev_user_emotion = quick_emotion_estimate(user_message)

    # --- Check deletion intent (keyword-only, no LLM on hot path) ---
    has_deletion_keyword = any(kw in user_message for kw in DELETION_KEYWORDS)

    # --- Topic tracking (Redis, no LLM) ---
    topic_info = await push_topic(conversation_id, user_message)
    topic_context = format_topic_context(topic_info) if topic_info else None

    # --- Pre-compute personality and topic fatigue (needed after parallel fetch) ---
    agent_personality = getattr(agent, "personality", None) or {}
    seven_dim = get_seven_dim(agent)

    # --- HOT PATH: parallel data fetches (no LLM calls) ---
    async def _do_retrieval():
        return await hybrid_retrieve(user_message, user_id)

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
        """Load L1 core memories — always present in prompt."""
        rows = await db.memory.find_many(
            where={"userId": user_id, "level": 1, "isArchived": False},
            order={"importance": "desc"},
            take=20,
        )
        return [r.summary or r.content for r in rows] if rows else None

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

    retrieval_result, emotion, summaries, core_memories, portrait, schedule, topic_intimacy = await asyncio.gather(
        _do_retrieval(),
        _load_cached_emotion(),
        _load_cached_summaries(),
        _load_core_memories(),
        _load_portrait(),
        _load_schedule(),
        _load_topic_intimacy(),
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

    # --- Intimacy stage for prompt (PRD §4.6.2.1) ---
    intimacy_stage = get_relationship_stage(topic_intimacy)

    # --- AI status context from schedule (pure computation) ---
    schedule_context = None
    ai_status = get_current_status(schedule) if schedule else None
    if ai_status:
        schedule_context = format_schedule_context(ai_status)
    ai_status_str = ai_status["status"] if ai_status else "idle"

    # 5B.4: Get patience prompt instruction (reuse value from check_boundary)
    patience_instruction = get_patience_prompt_instruction(cached_patience)

    # --- Multi-reply parameters (PRD §3.2.1) ---
    is_expand = detect_special_expand(user_message, emotion)
    reply_count = random.randint(1, MAX_REPLY_COUNT)
    max_reply_count = EXPAND_MAX_REPLY_COUNT if is_expand else MAX_REPLY_COUNT
    max_total = EXPAND_MAX_TOTAL_CHARS if is_expand else MAX_TOTAL_CHARS

    # Build prompt (pure string operations — instant)
    system_prompt = build_system_prompt(
        agent=agent,
        memories=memory_strings,
        core_memories=core_memories,
        emotion=emotion,
        graph_context=graph_context,
        summaries=summaries,
        portrait=portrait,
        topic_context=topic_context,
        user_emotion=prev_user_emotion,
        schedule_context=schedule_context,
        patience_instruction=patience_instruction,
        reply_count=reply_count,
        reply_total=max_total,
        intimacy_stage=intimacy_stage,
    )
    chat_messages = build_chat_messages(system_prompt, messages_dicts)

    # --- Send typing event before response ---
    typing_duration = calculate_typing_duration(len(user_message))

    # Message interval delay (PRD §6.2.1.2)
    # compute_message_interval_delay handles all cases:
    #   交流状态(<30min) → 1-5s, 高情绪 → 0-5s/60-180s, 其他 → calculate_status_delay
    status_delay = 0.0
    if len(recent_messages) >= 2:
        prev_time = recent_messages[-2].createdAt
        if prev_time.tzinfo is None:
            prev_time = prev_time.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - prev_time).total_seconds()
        status_delay = compute_message_interval_delay(age, emotion, ai_status_str)

    # Conceptual delay (sent to client) vs actual sleep (server-side cap)
    reply_delay = calculate_reply_delay(len(user_message), agent_personality, seven_dim=seven_dim)
    conceptual_delay = max(reply_delay, status_delay)
    actual_sleep = min(reply_delay, 2.0)

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
    ai_arousal = emotion.get("arousal", 0.0) if isinstance(emotion, dict) else 0.0
    ai_pleasure = emotion.get("pleasure", 0.0) if isinstance(emotion, dict) else 0.0
    ai_primary_emotion = emotion.get("primary_emotion") if isinstance(emotion, dict) else None

    for i, reply_text in enumerate(replies):
        # 12C: emoji概率
        if should_add_emoji(ai_arousal):
            emoji = pick_one_emoji(ai_pleasure, ai_arousal, ai_primary_emotion)
            if emoji:
                reply_text += emoji

        # 12D: 表情包互斥 — 框架预留，sticker资源接入后启用
        # from app.services.emoji import should_add_sticker
        # if not added_emoji and not sticker_used and should_add_sticker(ai_arousal):
        #     data["sticker_url"] = sticker_url

        if i > 0:
            await asyncio.sleep(random.uniform(0.3, 0.8))

        data: dict = {"text": reply_text, "index": i}
        yield {"event": "reply", "data": json.dumps(data)}

    # Save all replies to DB in background (don't block SSE stream)
    _fire_background(_save_replies(conversation_id, replies))

    full_response = " ".join(replies)

    # Update conversation title if first exchange (non-blocking)
    if len(recent_messages) <= 1:
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        _fire_background(db.conversation.update(
            where={"id": conversation_id},
            data={"title": title},
        ))

    yield {"event": "done", "data": json.dumps({"message_id": "complete"})}

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
        has_deletion_keyword=has_deletion_keyword,
        cached_emotion=emotion,
        seven_dim=seven_dim,
        topic_intimacy=topic_intimacy,
    ))


async def _background_post_process(
    user_id: str,
    agent_id: str | None,
    conversation_id: str,
    user_message: str,
    user_message_id: str,
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
    user_message_id: str,
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


async def _save_replies(conversation_id: str, replies: list[str]) -> None:
    """Save split replies as individual DB messages."""
    try:
        for i, text in enumerate(replies):
            await db.message.create(
                data={
                    "conversation": {"connect": {"id": conversation_id}},
                    "role": "assistant",
                    "content": text,
                    "metadata": Json({"reply_index": i}),
                }
            )
    except Exception as e:
        logger.error(f"Failed to save replies: {e}")


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
