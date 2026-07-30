from __future__ import annotations

import logging

from app.db import db
from app.services.speech_output.client import SynthesizedSpeech

logger = logging.getLogger(__name__)


async def record_tts_usage(
    *,
    speech: SynthesizedSpeech,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    message_id: str | None,
    source: str,
) -> None:
    """Best-effort immutable billing record for one delivered audio message."""
    try:
        await db.execute_raw(
            """
            INSERT INTO tts_usage (
                id, user_id, agent_id, conversation_id, message_id,
                source, provider, model, voice_id, request_id,
                raw_characters, billable_characters,
                duration_milliseconds, audio_bytes,
                unit_price_cny, cost_cny, created_at
            )
            VALUES (
                gen_random_uuid(), $1, $2, $3, $4,
                $5, 'dashscope', $6, $7, $8,
                $9, $10, $11, $12, $13, $14, NOW()
            )
            ON CONFLICT (request_id) DO NOTHING
            """,
            user_id,
            agent_id,
            conversation_id,
            message_id,
            source,
            speech.model,
            speech.voice_id,
            speech.request_id,
            speech.raw_characters,
            speech.billable_characters,
            speech.duration_milliseconds,
            len(speech.audio),
            speech.unit_price_cny,
            speech.cost_cny,
        )
    except Exception:
        logger.warning(
            "[TTS] usage metering failed user=%s agent=%s conversation=%s",
            user_id,
            agent_id,
            conversation_id,
            exc_info=True,
        )


async def link_tts_usage_to_message(
    *,
    request_id: str | None,
    message_id: str,
) -> None:
    if not request_id:
        return
    try:
        await db.execute_raw(
            """
            UPDATE tts_usage
            SET message_id = $1
            WHERE request_id = $2 AND message_id IS NULL
            """,
            message_id,
            request_id,
        )
    except Exception:
        logger.warning(
            "[TTS] usage message linkage failed request_id=%s",
            request_id,
            exc_info=True,
        )
