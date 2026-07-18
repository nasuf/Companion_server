from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.config import settings
from app.models.speech import (
    ChatAudioTranscriptionRequest,
    ChatAudioTranscriptionResponse,
)
from app.redis_client import get_redis
from app.models.chat_media import ChatAttachmentResponse
from app.services.chat_media import repo as chat_media_repo
from app.services.chat_media import storage as chat_media_storage
from app.services.runtime.tasks import fire_background
from app.services.speech_to_text import (
    SpeechTranscriptionEmpty,
    SpeechTranscriptionNotConfigured,
    SpeechTranscriptionProviderError,
    SpeechTranscriptionRateLimited,
    SpeechTranscriptionTimeout,
    transcribe_audio,
)
from app.services.speech_to_text.audio import (
    analyze_audio_activity,
    audio_format_for_mime,
    decode_audio_base64,
    normalize_audio_mime,
    validate_audio,
    validate_chat_m4a_duration,
)

router = APIRouter(prefix="/chat", tags=["speech-to-text"])
logger = logging.getLogger(__name__)


@router.post("/transcribe", response_model=ChatAudioTranscriptionResponse)
async def transcribe_chat_audio(
    data: ChatAudioTranscriptionRequest,
    user: dict = Depends(require_user),
) -> ChatAudioTranscriptionResponse:
    user_id = user["sub"]
    if not await chat_media_repo.conversation_belongs_to_user(
        data.conversation_id,
        user_id,
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")

    mime = normalize_audio_mime(data.mime, data.name)
    audio_format = audio_format_for_mime(mime)
    audio = decode_audio_base64(data.base64)
    validate_audio(
        audio,
        declared_size=data.size,
        duration_seconds=data.duration_seconds,
    )
    duration_seconds = validate_chat_m4a_duration(
        audio,
        mime=mime,
        declared_duration_seconds=data.duration_seconds,
    )
    await _enforce_rate_limit(user_id)
    activity = await analyze_audio_activity(audio)
    if activity is None:
        raise HTTPException(status_code=422, detail="语音文件无法解析，请重新录制")
    if not activity.has_meaningful_speech(
        settings.chat_voice_min_active_milliseconds
    ):
        logger.info(
            "[speech-to-text] silent chat audio rejected user_id=%s "
            "conversation_id=%s active_ms=%s total_ms=%s peak_dbfs=%.1f",
            user_id,
            data.conversation_id,
            activity.active_milliseconds,
            activity.total_milliseconds,
            activity.peak_dbfs,
        )
        raise HTTPException(status_code=422, detail="没有检测到清晰的语音，请重新录制")
    try:
        result = await transcribe_audio(
            audio=audio,
            mime=mime,
            audio_format=audio_format,
        )
    except SpeechTranscriptionNotConfigured as exc:
        logger.error("[speech-to-text] DashScope ASR is not configured")
        raise HTTPException(status_code=503, detail="语音识别服务尚未配置") from exc
    except SpeechTranscriptionRateLimited as exc:
        raise HTTPException(status_code=429, detail="语音识别请求较多，请稍后重试") from exc
    except SpeechTranscriptionTimeout as exc:
        raise HTTPException(status_code=504, detail="语音识别超时，请重试") from exc
    except SpeechTranscriptionEmpty as exc:
        raise HTTPException(status_code=422, detail="没有识别到清晰的语音") from exc
    except SpeechTranscriptionProviderError as exc:
        raise HTTPException(status_code=502, detail="语音识别暂时不可用，请重试") from exc

    if data.display_mode == "text":
        logger.info(
            "[speech-to-text] transient chat transcription user_id=%s "
            "conversation_id=%s duration_seconds=%s model=%s request_id=%s",
            user_id,
            data.conversation_id,
            duration_seconds,
            result.model,
            result.request_id,
        )
        return ChatAudioTranscriptionResponse(
            text=result.text,
            duration_seconds=duration_seconds,
            model=result.model,
            request_id=result.request_id,
        )

    storage_key = chat_media_storage.save_audio_blob(
        user_id=user_id,
        conversation_id=data.conversation_id,
        blob=audio,
        mime=mime,
    )
    try:
        attachment = await chat_media_repo.create_audio_attachment(
            user_id=user_id,
            conversation_id=data.conversation_id,
            storage_key=storage_key,
            url=chat_media_storage.media_url(storage_key),
            mime=mime,
            size=len(audio),
            duration_seconds=duration_seconds,
            transcription_text=result.text,
            transcription_model=result.model,
            transcription_request_id=result.request_id,
            name=data.name,
        )
    except Exception:
        chat_media_storage.delete_media_file(storage_key)
        raise
    fire_background(_cleanup_orphan_files(user_id))

    logger.info(
        "[speech-to-text] chat transcription user_id=%s conversation_id=%s "
        "duration_seconds=%s model=%s request_id=%s",
        user_id,
        data.conversation_id,
        duration_seconds,
        result.model,
        result.request_id,
    )
    return ChatAudioTranscriptionResponse(
        text=result.text,
        duration_seconds=duration_seconds,
        model=result.model,
        request_id=result.request_id,
        attachment=_attachment_response(attachment),
    )


async def _cleanup_orphan_files(user_id: str) -> None:
    try:
        removed = await chat_media_repo.cleanup_unbound_attachments(user_id)
        for attachment in removed:
            chat_media_storage.delete_media_file(attachment.storage_key)
    except Exception:
        logger.warning("[speech-to-text] orphan cleanup failed", exc_info=True)


async def _enforce_rate_limit(user_id: str) -> None:
    try:
        redis = await get_redis()
        key = f"speech:chat:minute:{user_id}"
        count = int(await redis.incr(key))
        if count == 1:
            await redis.expire(key, 60)
        if count > settings.chat_voice_max_requests_per_minute:
            raise HTTPException(status_code=429, detail="语音识别请求较多，请稍后重试")
    except HTTPException:
        raise
    except Exception:
        # Redis availability must not make ordinary chat unusable. Provider
        # rate limits remain the final backstop while this warning is alerted.
        logger.warning("[speech-to-text] rate-limit check failed open", exc_info=True)


def _attachment_response(
    attachment: chat_media_repo.ChatAttachment,
) -> ChatAttachmentResponse:
    return ChatAttachmentResponse(
        id=attachment.id,
        kind=attachment.kind,
        name=attachment.name,
        mime=attachment.mime,
        size=attachment.size,
        width=attachment.width,
        height=attachment.height,
        duration_seconds=attachment.duration_seconds,
        url=attachment.url,
        vision_status=attachment.vision_status,
        vision_summary=attachment.vision_summary,
        transcription_status=attachment.transcription_status,
        transcription_text=attachment.transcription_text,
        transcription_model=attachment.transcription_model,
        transcription_request_id=attachment.transcription_request_id,
        created_at=str(attachment.created_at) if attachment.created_at else None,
    )
