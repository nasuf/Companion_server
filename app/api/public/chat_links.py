from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.models.chat_links import ChatLinkAnalyzeRequest, ChatLinkCardResponse
from app.services.chat_links import (
    component_card_for_link,
    create_or_update_link_card,
    extract_link_metadata,
)
from app.services.chat_media import repo as chat_media_repo

router = APIRouter(prefix="/chat/links", tags=["chat-links"])


@router.post("/preview", response_model=ChatLinkCardResponse)
async def preview_chat_link(
    data: ChatLinkAnalyzeRequest,
    user: dict = Depends(require_user),
) -> ChatLinkCardResponse:
    user_id = user["sub"]
    if not await chat_media_repo.conversation_belongs_to_user(data.conversation_id, user_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        metadata = await extract_link_metadata(url=data.url, shared_text=data.shared_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    link = await create_or_update_link_card(
        user_id=user_id,
        conversation_id=data.conversation_id,
        metadata=metadata,
        role="user",
        source_app=data.source_app,
        extra_metadata={"preview": True},
    )
    return _response(link)


def _response(link) -> ChatLinkCardResponse:
    return ChatLinkCardResponse(
        id=link.id,
        conversation_id=link.conversation_id,
        message_id=link.message_id,
        role=link.role,
        source_app=link.source_app,
        source_url=link.source_url,
        final_url=link.final_url,
        platform=link.platform,
        title=link.title,
        description=link.description,
        author=link.author,
        image_url=link.image_url,
        content_text=link.content_text,
        summary=link.summary,
        status=link.status,
        error=link.error,
        created_at=str(link.created_at) if link.created_at else None,
        component_card=component_card_for_link(link),
    )
