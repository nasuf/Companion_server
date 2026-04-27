from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_user
from app.api.ownership import require_conversation_owner, require_user_self
from app.db import db
from app.models.conversation import ConversationCreate, ConversationResponse
from app.models.message import MessageResponse
from app.services.workspace.workspaces import ensure_workspace, get_workspace_by_id

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    data: ConversationCreate,
    user: dict = Depends(require_user),
):
    if data.user_id != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your user_id")
    agent = await db.aiagent.find_unique(where={"id": data.agent_id})
    # 放宽到 active / provisioning: agent 创建后端 LLM 后台 30-60s, 前端在
    # 此期间立刻创建会话, workspace 已 active, conversation 可正常落库;
    # 待 agent 激活后用户即可发送消息. archived 仍 404.
    if not agent or getattr(agent, "status", "active") == "archived":
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your agent")
    workspace = None
    if data.workspace_id:
        workspace = await get_workspace_by_id(data.workspace_id)
        if not workspace or workspace.status != "active":
            raise HTTPException(status_code=404, detail="Workspace not found")
        if workspace.userId != data.user_id or workspace.agentId != data.agent_id:
            raise HTTPException(status_code=400, detail="Workspace does not match user/agent")
    else:
        try:
            workspace = await ensure_workspace(data.user_id, data.agent_id)
        except ValueError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc

    existing = await db.conversation.find_first(
        where={
            "workspaceId": workspace.id,
            "isDeleted": False,
        },
        order={"updatedAt": "desc"},
    )
    if existing:
        return ConversationResponse(
            id=existing.id,
            user_id=existing.userId,
            agent_id=existing.agentId,
            workspace_id=existing.workspaceId,
            title=existing.title,
            created_at=str(existing.createdAt),
            updated_at=str(existing.updatedAt),
        )

    try:
        conv = await db.conversation.create(
            data={
                "user": {"connect": {"id": data.user_id}},
                "agent": {"connect": {"id": data.agent_id}},
                "workspace": {"connect": {"id": workspace.id}},
                "title": data.title,
            }
        )
    except Exception:
        existing = await db.conversation.find_first(
            where={
                "workspaceId": workspace.id,
                "isDeleted": False,
            },
            order={"updatedAt": "desc"},
        )
        if not existing:
            raise
        conv = existing

    return ConversationResponse(
        id=conv.id,
        user_id=conv.userId,
        agent_id=conv.agentId,
        workspace_id=conv.workspaceId,
        title=conv.title,
        created_at=str(conv.createdAt),
        updated_at=str(conv.updatedAt),
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conv=Depends(require_conversation_owner)):
    return ConversationResponse(
        id=conv.id,
        user_id=conv.userId,
        agent_id=conv.agentId,
        workspace_id=conv.workspaceId,
        title=conv.title,
        created_at=str(conv.createdAt),
        updated_at=str(conv.updatedAt),
    )


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    user_id: str = Query(...),
    workspace_id: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    _user=Depends(require_user_self),
):
    where: dict = {"isDeleted": False, "userId": user_id}
    if workspace_id:
        where["workspaceId"] = workspace_id
    convs = await db.conversation.find_many(
        where=where, order={"createdAt": "desc"}, take=limit, skip=offset
    )
    return [
        ConversationResponse(
            id=c.id,
            user_id=c.userId,
            agent_id=c.agentId,
            workspace_id=c.workspaceId,
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
    _conv=Depends(require_conversation_owner),
):
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
async def delete_conversation(conv=Depends(require_conversation_owner)):
    await db.conversation.update(
        where={"id": conv.id},
        data={"isDeleted": True, "archivedAt": datetime.now(UTC)},
    )
    return {"status": "deleted"}
