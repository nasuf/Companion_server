"""Store voice transcripts on the same conversation as text chat.

Rows are marked hiddenFromTimeline: the call is heard, not read, so the chat
history never lists them, while memory extraction and chat context still do.

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
            "hiddenFromTimeline": True,
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
            "hiddenFromTimeline": True,
        }
    )
    return created.id


# One extraction run reads at most 30 new rows per side, and a source batch
# may name at most 100 rows. 60 ids keep both sides near one run's page.
_EXTRACTION_BATCH = 60
# A 28-minute call stores ~200 rows; leave room for a text backlog before it.
_MAX_EXTRACTION_ROUNDS = 12


def schedule_end_of_call_extraction(
    *,
    user_id: str,
    workspace_id: str | None,
    conversation_id: str,
    agent_id: str | None,
    message_ids: list[str],
) -> None:
    """Run the text-chat memory pipeline over this call's transcript rows.

    The pipeline only extracts persisted rows it can identify, so the call
    passes the ids it stored; without them every run failed with
    source_identity_missing and the call waited for the next text message.
    """
    if not message_ids:
        return
    fire_background(_run_extraction(
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        agent_id=agent_id,
        message_ids=list(dict.fromkeys(message_ids)),
    ))


def _metadata(model: str) -> dict:
    return {
        "channel": "realtime_voice",
        "provider": "dashscope",
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
    agent_id: str | None,
    message_ids: list[str],
) -> None:
    """Extract the call in chronological batches until both sides catch up.

    Each run advances the per-side watermark by one bounded page, so a long
    call needs several runs. A batch's ids are the run's identified sources
    and context; once batches run out, the last one is reused while a side
    still reports more backlog (rows saved before the call).
    """
    from app.services.chat.post_process import _bg_memory_pipeline
    from app.services.llm.usage_tracker import usage_session

    batches = [
        message_ids[i:i + _EXTRACTION_BATCH]
        for i in range(0, len(message_ids), _EXTRACTION_BATCH)
    ]
    stored = 0
    rounds = 0
    outcome = "completed"
    logger.info(
        "[realtime-voice] end-of-call extraction conv=%s rows=%d",
        conversation_id[:8], len(message_ids),
    )
    try:
        async with usage_session(
            scope="post_process", conversation_id=conversation_id,
            agent_id=agent_id, user_id=user_id,
        ):
            while rounds < _MAX_EXTRACTION_ROUNDS:
                batch = batches[min(rounds, len(batches) - 1)]
                rounds += 1
                results = await _bg_memory_pipeline(
                    user_id,
                    [],
                    conversation_id=conversation_id,
                    workspace_id=workspace_id,
                    source_message_ids=batch,
                )
                stored += sum(r.memory_count or 0 for r in results)
                if any(r.status in {"failed", "disabled"} for r in results):
                    outcome = ",".join(f"{r.side}:{r.reason}" for r in results if r.status in {"failed", "disabled"})
                    break
                if rounds >= len(batches) and not any(r.has_more for r in results):
                    break
            else:
                outcome = "round_limit"
    except Exception as exc:
        outcome = f"error:{type(exc).__name__}"
        logger.warning("[realtime-voice] end-of-call extraction failed: %s", exc)
    logger.log(
        logging.INFO if outcome == "completed" else logging.WARNING,
        "[realtime-voice] end-of-call extraction %s conv=%s rounds=%d memories=%d",
        outcome, conversation_id[:8], rounds, stored,
    )
