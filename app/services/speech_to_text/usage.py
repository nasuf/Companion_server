"""Speech-to-text usage metering.

Records one row per successful ``/chat/transcribe`` call so that voice-message
("voice") and voice-to-text ("text") durations can be reported separately in the
admin media-usage overview. This is a standalone ledger (no foreign keys) and is
written on a best-effort basis — a metering failure must never break the
transcription response.
"""

from __future__ import annotations

import logging

from app.db import db

logger = logging.getLogger(__name__)


async def record_speech_usage(
    *,
    user_id: str,
    conversation_id: str,
    display_mode: str,
    duration_seconds: int,
    model: str | None,
    request_id: str | None,
) -> None:
    """Insert a usage row. Swallows errors (best-effort, off the hot path)."""
    try:
        await db.query_raw(
            """
            INSERT INTO speech_usage (
                user_id, conversation_id, display_mode,
                duration_seconds, model, request_id, source
            )
            VALUES ($1, $2, $3, $4, $5, $6, 'live')
            """,
            user_id,
            conversation_id,
            display_mode,
            int(duration_seconds),
            model,
            request_id,
        )
    except Exception:
        logger.warning(
            "[speech-to-text] speech usage metering failed "
            "user_id=%s conversation_id=%s display_mode=%s",
            user_id,
            conversation_id,
            display_mode,
            exc_info=True,
        )
