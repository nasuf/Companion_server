from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.jwt_auth import require_user
from app.models.chat_media import ChatAttachmentResponse, ChatImageUpload
from app.services.chat_media import repo, storage
from app.services.runtime.tasks import fire_background

router = APIRouter(prefix="/chat/media", tags=["chat-media"])
logger = logging.getLogger(__name__)


async def _store_chat_image(
    *,
    user_id: str,
    conversation_id: str,
    blob: bytes,
    mime: str,
    name: str | None,
    fallback_width: int | None = None,
    fallback_height: int | None = None,
) -> ChatAttachmentResponse:
    """Shared ingest for the base64 and multipart upload routes: normalizes the
    image, stores original + thumbnail, and records server-measured metadata."""
    if not await repo.conversation_belongs_to_user(conversation_id, user_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    normalized_mime = storage.normalize_image_mime(mime)
    storage.validate_image_size(blob)
    storage_key, processed = storage.save_image_with_thumbnail(
        user_id=user_id,
        conversation_id=conversation_id,
        blob=blob,
        mime=normalized_mime,
    )
    try:
        attachment = await repo.create_attachment(
            user_id=user_id,
            conversation_id=conversation_id,
            storage_key=storage_key,
            url=storage.media_url(storage_key),
            mime=processed.mime,
            size=len(processed.blob),
            name=name,
            # Server-measured dimensions win: H5 historically sent 0x0, which
            # broke bubble aspect ratios on every client.
            width=processed.width or fallback_width,
            height=processed.height or fallback_height,
        )
    except Exception:
        storage.delete_media_file(storage_key)
        raise
    fire_background(_cleanup_orphan_files(user_id))
    return _response(attachment)


@router.post("", response_model=ChatAttachmentResponse)
async def upload_chat_image(
    data: ChatImageUpload,
    user: dict = Depends(require_user),
) -> ChatAttachmentResponse:
    return await _store_chat_image(
        user_id=user["sub"],
        conversation_id=data.conversation_id,
        blob=storage.decode_image_base64(data.base64),
        mime=data.mime,
        name=data.name,
        fallback_width=data.width or None,
        fallback_height=data.height or None,
    )


@router.post("/upload", response_model=ChatAttachmentResponse)
async def upload_chat_image_multipart(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    name: str | None = Form(None),
    user: dict = Depends(require_user),
) -> ChatAttachmentResponse:
    """Multipart variant of the image upload: no base64 inflation (-25% wire
    size), used by newer app builds. The base64 JSON route stays for H5 and
    the mini-program."""
    blob = await file.read()
    return await _store_chat_image(
        user_id=user["sub"],
        conversation_id=conversation_id,
        blob=blob,
        mime=file.content_type or "image/jpeg",
        name=name or file.filename,
    )


@router.get("/{storage_key}")
async def get_chat_media(
    storage_key: str,
    v: str | None = Query(default=None, description="thumb = bubble thumbnail"),
    user: dict = Depends(require_user),
):
    return storage.serve_media(
        storage_key,
        user_id=user["sub"],
        is_admin=user.get("role") == "admin",
        variant=v,
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
