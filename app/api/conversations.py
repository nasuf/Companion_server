from fastapi import APIRouter, HTTPException, Query

from app.db import db
from app.models.conversation import ConversationCreate, ConversationResponse
from app.models.message import MessageResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse)
async def create_conversation(data: ConversationCreate):
    conv = await db.conversation.create(
        data={
            "user": {"connect": {"id": data.user_id}},
            "agent": {"connect": {"id": data.agent_id}},
            "title": data.title,
        }
    )
    return ConversationResponse(
        id=conv.id,
        user_id=conv.userId,
        agent_id=conv.agentId,
        title=conv.title,
        created_at=str(conv.createdAt),
        updated_at=str(conv.updatedAt),
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    conv = await db.conversation.find_unique(where={"id": conversation_id})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        id=conv.id,
        user_id=conv.userId,
        agent_id=conv.agentId,
        title=conv.title,
        created_at=str(conv.createdAt),
        updated_at=str(conv.updatedAt),
    )


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    user_id: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
):
    where = {"isDeleted": False}
    if user_id:
        where["userId"] = user_id
    convs = await db.conversation.find_many(
        where=where, order={"createdAt": "desc"}, take=limit, skip=offset
    )
    return [
        ConversationResponse(
            id=c.id,
            user_id=c.userId,
            agent_id=c.agentId,
            title=c.title,
            created_at=str(c.createdAt),
            updated_at=str(c.updatedAt),
        )
        for c in convs
    ]


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def list_messages(
    conversation_id: str,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
):
    conv = await db.conversation.find_unique(where={"id": conversation_id})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await db.message.find_many(
        where={"conversationId": conversation_id},
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
    )
    return [
        MessageResponse(
            id=m.id,
            conversation_id=m.conversationId,
            role=m.role,
            content=m.content,
            metadata=m.metadata,
            created_at=str(m.createdAt),
        )
        for m in messages
    ]


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conv = await db.conversation.find_unique(where={"id": conversation_id})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.conversation.update(
        where={"id": conversation_id}, data={"isDeleted": True}
    )
    return {"status": "deleted"}
