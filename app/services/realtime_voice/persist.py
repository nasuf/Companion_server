"""Store voice transcripts on the same conversation as text chat.

Memory extraction is NOT done per-utterance. Instead, transcripts are
persisted as normal messages and extracted in batch at call end via the
standard text-chat watermark pipeline (_bg_memory_pipeline). This avoids:
  1. Duplicate extraction (text-chat pipeline would re-extract voice rows)
  2. Context-free extraction (single utterances lack conversation context)
  3. Blocking playback with LLM calls
"""

from __future__ import annotations

import logging

from prisma import Json

from app.db import db
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)


def schedule_user_present(conversation_id: str) -> None:
    fire_background(_mark_present(conversation_id))


async def save_user_transcript(
    *,
    conversation_id: str,
    user_id: str,
    workspace_id: str | None,
    model: str,
    text: str,
) -> str | None:
    """Persist a user utterance. No memory extraction here."""
    content = " ".join(text.split())
    if not content:
        return None
    created = await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "user",
            "content": content[:4000],
            "metadata": Json(_metadata(model)),
        }
    )
    fire_background(_mark_present(conversation_id))
    return created.id


async def save_assistant_transcript(
    *,
    conversation_id: str,
    user_id: str,
    workspace_id: str | None,
    model: str,
    text: str,
    user_message_id: str | None,
    interrupted: bool,
) -> str | None:
    """Persist an assistant utterance. No memory extraction here."""
    content = " ".join(text.split())
    if not content:
        return None
    metadata = _metadata(model)
    if interrupted:
        metadata["interrupted"] = True
    if user_message_id:
        metadata["reply_to"] = user_message_id
    created = await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "assistant",
            "content": content[:4000],
            "metadata": Json(metadata),
        }
    )
    return created.id


def schedule_end_of_call_extraction(
    *,
    user_id: str,
    workspace_id: str | None,
    conversation_id: str,
) -> None:
    """Trigger the standard text-chat memory pipeline for this conversation.

    The pipeline uses watermarks to find un-extracted messages (including
    the voice transcripts we just persisted) and processes them with full
    context. This is the same path text chat uses.
    """
    fire_background(_run_extraction(
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
    ))


def _metadata(model: str) -> dict:
    return {
        "channel": "realtime_voice",
        "provider": "stepfun",
        "model": model,
    }


async def _mark_present(conversation_id: str) -> None:
    try:
        from app.services.proactive.state import mark_user_replied_for_conversation
        await mark_user_replied_for_conversation(conversation_id)
    except Exception as exc:
        logger.warning("[realtime-voice] presence update failed: %s", exc)


async def _run_extraction(
    *,
    user_id: str,
    workspace_id: str | None,
    conversation_id: str,
) -> None:
    """Run the standard dual-side memory pipeline.

    This picks up all un-extracted messages in the conversation (both
    voice transcripts and any text messages), using the watermark system
    to avoid re-processing.
    """
    try:
        from app.services.chat.post_process import _bg_memory_pipeline
        logger.info("[realtime-voice] running end-of-call extraction conv=%s", conversation_id[:8])
        await _bg_memory_pipeline(
            user_id,
            [],
            conversation_id=conversation_id,
            workspace_id=workspace_id,
        )
        logger.info("[realtime-voice] end-of-call extraction completed conv=%s", conversation_id[:8])
    except Exception as exc:
        logger.warning("[realtime-voice] end-of-call extraction failed: %s", exc)
