from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from app.services.chat_media import repo as media_repo
from app.services.chat_media import storage as media_storage
from app.services.chat_media.prompt import attachment_to_metadata
from app.services.speech_output.client import SynthesizedSpeech, synthesize_speech
from app.services.speech_output.style import build_style_instruction
from app.services.speech_output.usage import (
    link_tts_usage_to_message,
    record_tts_usage,
)
from app.services.speech_output.voices import ensure_agent_voice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedVoiceOutput:
    speech: SynthesizedSpeech
    attachment: Any
    metadata: dict[str, Any]
    transcript: str
    user_id: str
    agent_id: str
    conversation_id: str
    source: str


async def prepare_voice_output(
    *,
    text: str,
    user_id: str,
    agent: Any,
    conversation_id: str,
    source: str,
    emotion: str | None = None,
    intensity: int | float | None = None,
) -> PreparedVoiceOutput:
    transcript = " ".join((text or "").split())
    from app.services.emoji import limit_emojis

    spoken_text = " ".join(limit_emojis(transcript, max_keep=0).split())
    if not spoken_text:
        spoken_text = transcript
    voice_id = await ensure_agent_voice(agent)
    speech = await synthesize_speech(
        text=spoken_text,
        voice_id=voice_id,
        instruction=build_style_instruction(emotion, intensity),
    )
    agent_id = str(getattr(agent, "id"))
    # Meter the successful provider call before local persistence. DashScope has
    # already billed it even if disk/message delivery later fails.
    await record_tts_usage(
        speech=speech,
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        message_id=None,
        source=source,
    )
    storage_key = media_storage.save_audio_blob(
        user_id=user_id,
        conversation_id=conversation_id,
        blob=speech.audio,
        mime=speech.mime,
    )
    try:
        attachment = await media_repo.create_generated_audio_attachment(
            user_id=user_id,
            conversation_id=conversation_id,
            storage_key=storage_key,
            url=media_storage.media_url(storage_key),
            mime=speech.mime,
            size=len(speech.audio),
            duration_seconds=max(
                1,
                math.ceil(speech.duration_milliseconds / 1000),
            ),
            transcript=transcript,
            request_id=speech.request_id,
            name="agent_voice.wav",
        )
    except Exception:
        media_storage.delete_media_file(storage_key)
        raise
    metadata = attachment_to_metadata(attachment)
    metadata["generated_by"] = "assistant_tts"
    return PreparedVoiceOutput(
        speech=speech,
        attachment=attachment,
        metadata=metadata,
        transcript=transcript,
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        source=source,
    )


async def bind_prepared_voice_output(
    prepared: PreparedVoiceOutput,
    *,
    message_id: str,
) -> None:
    await media_repo.bind_attachments_to_message(
        attachment_ids=[prepared.attachment.id],
        message_id=message_id,
        user_id=prepared.user_id,
        conversation_id=prepared.conversation_id,
    )
    await link_tts_usage_to_message(
        request_id=prepared.speech.request_id,
        message_id=message_id,
    )


async def discard_prepared_voice_output(
    prepared: PreparedVoiceOutput,
) -> None:
    deleted = await media_repo.delete_unbound_attachment(
        attachment_id=prepared.attachment.id,
        user_id=prepared.user_id,
        conversation_id=prepared.conversation_id,
    )
    media_storage.delete_media_file(
        getattr(deleted, "storage_key", None)
        or getattr(prepared.attachment, "storage_key", None)
    )
