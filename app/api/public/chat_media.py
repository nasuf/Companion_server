from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.models.chat_media import ChatAttachmentResponse, ChatImageUpload
from app.services.chat_media import repo, storage
from app.services.runtime.tasks import fire_background

router = APIRouter(prefix="/chat/media", tags=["chat-media"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatAttachmentResponse)
async def upload_chat_image(
    data: ChatImageUpload,
    user: dict = Depends(require_user),
) -> ChatAttachmentResponse:
    user_id = user["sub"]
    if not await repo.conversation_belongs_to_user(data.conversation_id, user_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    mime = storage.normalize_image_mime(data.mime)
    blob = storage.decode_image_base64(data.base64)
    storage.validate_image_size(blob)
    storage_key = storage.save_image_blob(user_id=user_id, blob=blob, mime=mime)
    try:
        attachment = await repo.create_attachment(
            user_id=user_id,
            conversation_id=data.conversation_id,
            storage_key=storage_key,
            url=storage.media_url(storage_key),
            mime=mime,
            size=len(blob),
            name=data.name,
            width=data.width,
            height=data.height,
        )
    except Exception:
        storage.delete_media_file(storage_key)
        raise
    fire_background(_cleanup_orphan_files(user_id))
    return _response(attachment)


@router.get("/{storage_key}")
async def get_chat_media(
    storage_key: str,
    user: dict = Depends(require_user),
):
    return storage.serve_media(
        storage_key,
        user_id=user["sub"],
        is_admin=user.get("role") == "admin",
    )


async def _cleanup_orphan_files(user_id: str) -> None:
    try:
        removed = await repo.cleanup_unbound_attachments(user_id)
        for attachment in removed:
            storage.delete_media_file(attachment.storage_key)
    except Exception:
        logger.warning("[chat-media] orphan cleanup failed", exc_info=True)


def _response(attachment: repo.ChatAttachment) -> ChatAttachmentResponse:
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
