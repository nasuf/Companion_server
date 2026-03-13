from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.db import db
from app.models.message import ChatRequest
from app.services.chat_service import stream_chat_response

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/{conversation_id}")
async def chat(conversation_id: str, data: ChatRequest):
    conv = await db.conversation.find_unique(
        where={"id": conversation_id},
        include={"agent": True},
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.isDeleted:
        raise HTTPException(status_code=410, detail="Conversation deleted")

    return EventSourceResponse(
        stream_chat_response(
            conversation_id=conversation_id,
            user_message=data.message,
            agent=conv.agent,
            user_id=conv.userId,
        )
    )
