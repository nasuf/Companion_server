from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.db import db
from app.models.message import ChatRequest
from app.services.chat_service import stream_chat_response
from app.services.proactive import generate_proactive_message, get_proactive_history

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


@router.post("/proactive/{agent_id}")
async def trigger_proactive(agent_id: str, user_id: str):
    """触发AI主动消息。"""
    message = await generate_proactive_message(user_id, agent_id)
    if not message:
        return {"message": None, "reason": "no_content_or_limit_reached"}
    return {"message": message}


@router.get("/proactive/{agent_id}/history")
async def proactive_history(agent_id: str, user_id: str, limit: int = 10):
    """获取主动消息历史。"""
    history = await get_proactive_history(agent_id, user_id, limit)
    return {"history": history}
