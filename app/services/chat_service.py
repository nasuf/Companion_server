"""Chat service — optimized for low latency.

Hot path (user-facing, ~2s):
  save msg → parallel(vector retrieval + load cached context) → prompt → stream LLM

Background (fire-and-forget, after response):
  emotion extraction + summarizer update + memory pipeline
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from prisma import Json

from app.db import db
from app.services.llm.models import get_chat_model, convert_messages
from app.services.prompt_builder import build_system_prompt, build_chat_messages
from app.services.memory.hybrid_retrieval import hybrid_retrieve
from app.services.memory.pipeline import process_memory_pipeline
from app.services.summarizer import summarize
from app.services.emotion import (
    extract_emotion,
    get_ai_emotion,
    update_emotion_state,
    save_ai_emotion,
)
from app.services.cache import cache_summarizer, cache_set_summarizer

logger = logging.getLogger(__name__)


async def stream_chat_response(
    conversation_id: str,
    user_message: str,
    agent,
    user_id: str,
) -> AsyncGenerator[dict, None]:
    # Save user message
    await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "user",
            "content": user_message,
        }
    )

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

    agent_id = getattr(agent, "id", None)

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
        ch = _conv_hash(messages_dicts[:-1], messages_dicts[-2]["content"] if len(messages_dicts) >= 2 else "")
        return await cache_summarizer(ch)

    async def _load_core_memories():
        """Load L1 core memories — always present in prompt."""
        rows = await db.memory.find_many(
            where={"userId": user_id, "level": 1, "isArchived": False},
            order={"importance": "desc"},
            take=20,
        )
        return [r.summary or r.content for r in rows] if rows else None

    retrieval_result, emotion, summaries, core_memories = await asyncio.gather(
        _do_retrieval(),
        _load_cached_emotion(),
        _load_cached_summaries(),
        _load_core_memories(),
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

    # Build prompt (pure string operations — instant)
    system_prompt = build_system_prompt(
        agent=agent,
        memories=memory_strings,
        core_memories=core_memories,
        emotion=emotion,
        graph_context=graph_context,
        summaries=summaries,
    )
    chat_messages = build_chat_messages(system_prompt, messages_dicts)

    # --- Stream response from large model (the ONLY LLM call in hot path) ---
    model = get_chat_model()
    lc_messages = convert_messages(chat_messages)

    response_chunks: list[str] = []
    async for chunk in model.astream(lc_messages):
        token = chunk.content
        if token:
            response_chunks.append(token)
            yield {"event": "token", "data": json.dumps({"token": token})}

    full_response = "".join(response_chunks)

    # Save assistant message
    await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "assistant",
            "content": full_response,
            "metadata": Json({}),
        }
    )

    # Update conversation title if first exchange
    if len(recent_messages) <= 1:
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        await db.conversation.update(
            where={"id": conversation_id},
            data={"title": title},
        )

    yield {"event": "done", "data": json.dumps({"message_id": "complete"})}

    # --- BACKGROUND: fire-and-forget post-processing ---
    asyncio.create_task(
        _background_post_process(
            user_id=user_id,
            agent_id=agent_id,
            user_message=user_message,
            full_response=full_response,
            messages_dicts=messages_dicts,
            memory_strings=memory_strings,
        )
    )


async def _background_post_process(
    user_id: str,
    agent_id: str | None,
    user_message: str,
    full_response: str,
    messages_dicts: list[dict],
    memory_strings: list[str] | None,
) -> None:
    """Run all background tasks after response is sent.

    1. Emotion extraction + state update (small model)
    2. Summarizer → cache to Redis for next request (small model)
    3. Memory extraction pipeline (small model)
    """
    try:
        # Add the assistant response to messages for summarizer context
        full_messages = messages_dicts + [{"role": "assistant", "content": full_response}]

        # Run all 3 background tasks concurrently
        await asyncio.gather(
            _bg_emotion(agent_id, user_message),
            _bg_summarizer(full_messages, user_message, memory_strings),
            _bg_memory_pipeline(user_id, full_messages),
            return_exceptions=True,
        )
    except Exception as e:
        logger.error(f"Background post-processing failed: {e}")


async def _bg_emotion(agent_id: str | None, user_message: str) -> None:
    """Extract emotion from user message and update AI emotion state."""
    if not agent_id:
        return
    try:
        user_emotion = await extract_emotion(user_message)
        current_emotion = await get_ai_emotion(agent_id)
        new_emotion = update_emotion_state(current_emotion, user_emotion)
        await save_ai_emotion(agent_id, new_emotion)
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


async def _bg_memory_pipeline(user_id: str, messages: list[dict]) -> None:
    """Run memory extraction pipeline."""
    try:
        conv_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages[-10:]
        )
        await process_memory_pipeline(user_id, conv_text)
    except Exception as e:
        logger.error(f"Background memory pipeline failed: {e}")
