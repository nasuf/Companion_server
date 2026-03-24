from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.db import db
from app.api.jwt_auth import require_admin_jwt

router = APIRouter(prefix="/admin-api/users", tags=["admin-users"])


@router.get("")
async def list_users(
    search: str = "",
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: dict = Depends(require_admin_jwt),
):
    where = {}
    if search.strip():
        where["username"] = {"contains": search.strip(), "mode": "insensitive"}

    total = await db.user.count(where=where)
    users = await db.user.find_many(
        where=where,
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
        include={"agents": True},
    )

    return {
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "name": u.name,
                "role": u.role,
                "created_at": str(u.createdAt),
                "agent_count": len(u.agents) if u.agents else 0,
            }
            for u in users
        ],
        "total": total,
    }


@router.get("/{user_id}/detail")
async def get_user_detail(
    user_id: str,
    _: dict = Depends(require_admin_jwt),
):
    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    agents = await db.aiagent.find_many(
        where={"userId": user_id},
        include={"conversations": {"include": {"messages": True}}},
    )

    agent_list = []
    for a in agents:
        conv_count = len(a.conversations) if a.conversations else 0
        msg_count = sum(
            len(c.messages) for c in (a.conversations or []) if c.messages
        )
        agent_list.append({
            "id": a.id,
            "name": a.name,
            "gender": a.gender,
            "created_at": str(a.createdAt),
            "conversation_count": conv_count,
            "message_count": msg_count,
            "conversations": [
                {
                    "id": c.id,
                    "created_at": str(c.createdAt),
                    "updated_at": str(c.updatedAt),
                    "message_count": len(c.messages) if c.messages else 0,
                }
                for c in (a.conversations or [])
            ],
        })

    return {
        "user": {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "role": user.role,
            "created_at": str(user.createdAt),
        },
        "agents": agent_list,
    }


@router.get("/{user_id}/conversations/{conv_id}/messages")
async def get_user_messages(
    user_id: str,
    conv_id: str,
    limit: int = Query(default=200, ge=1, le=500),
    _: dict = Depends(require_admin_jwt),
):
    conv = await db.conversation.find_unique(where={"id": conv_id})
    if not conv or conv.userId != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = await db.message.find_many(
        where={"conversationId": conv_id},
        order={"createdAt": "asc"},
        take=limit,
    )

    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "metadata": m.metadata,
            "created_at": str(m.createdAt),
        }
        for m in messages
    ]
