from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator

from app.db import db
from app.api.jwt_auth import require_admin_jwt

_ALLOWED_ROLES = {"user", "admin"}


class UpdateUserRoleRequest(BaseModel):
    role: str

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in _ALLOWED_ROLES:
            raise ValueError(f"role must be one of {sorted(_ALLOWED_ROLES)}")
        return v

router = APIRouter(prefix="/admin-api/users", tags=["admin-users"])


@router.get("/memory-overview")
async def memory_overview(_: dict = Depends(require_admin_jwt)):
    workspaces = await db.chatworkspace.find_many()
    user_memories = await db.usermemory.find_many(where={"isArchived": False})
    ai_memories = await db.aimemory.find_many(where={"isArchived": False})
    since = datetime.now(UTC) - timedelta(days=7)
    recent_user_memories = await db.usermemory.find_many(
        where={"isArchived": False, "createdAt": {"gte": since}}
    )
    recent_ai_memories = await db.aimemory.find_many(
        where={"isArchived": False, "createdAt": {"gte": since}}
    )

    all_memories = [*user_memories, *ai_memories]
    recent_memories = [*recent_user_memories, *recent_ai_memories]
    by_main_category: dict[str, int] = {}
    by_sub_category: dict[str, int] = {}
    by_level: dict[str, int] = {}
    by_workspace_status: dict[str, int] = {}
    recent_by_main_category: dict[str, int] = {}

    workspace_status_map = {workspace.id: workspace.status for workspace in workspaces}
    active_workspace_count = sum(1 for workspace in workspaces if workspace.status == "active")

    for memory in all_memories:
        main_category = getattr(memory, "mainCategory", None) or "未分类"
        sub_category = getattr(memory, "subCategory", None) or "其他"
        level = f"L{memory.level}"
        workspace_status = workspace_status_map.get(getattr(memory, "workspaceId", None), "unknown")

        by_main_category[main_category] = by_main_category.get(main_category, 0) + 1
        by_sub_category[sub_category] = by_sub_category.get(sub_category, 0) + 1
        by_level[level] = by_level.get(level, 0) + 1
        by_workspace_status[workspace_status] = by_workspace_status.get(workspace_status, 0) + 1

    for memory in recent_memories:
        main_category = getattr(memory, "mainCategory", None) or "未分类"
        recent_by_main_category[main_category] = recent_by_main_category.get(main_category, 0) + 1

    def _serialize(data: dict[str, int], limit: int | None = None):
        items = sorted(data.items(), key=lambda item: (-item[1], item[0]))
        if limit is not None:
            items = items[:limit]
        return [{"key": key, "count": count} for key, count in items]

    return {
        "totals": {
            "workspaces": len(workspaces),
            "active_workspaces": active_workspace_count,
            "memories": len(all_memories),
            "user_memories": len(user_memories),
            "ai_memories": len(ai_memories),
            "recent_memories_7d": len(recent_memories),
        },
        "by_level": _serialize(by_level),
        "by_main_category": _serialize(by_main_category),
        "by_sub_category": _serialize(by_sub_category, limit=15),
        "by_workspace_status": _serialize(by_workspace_status),
        "recent_by_main_category": _serialize(recent_by_main_category),
    }


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
                "role": u.role,
                "created_at": str(u.createdAt),
                "status": getattr(u, "status", "active"),
                "archived_at": str(u.archivedAt) if getattr(u, "archivedAt", None) else None,
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

    user_workspaces = await db.chatworkspace.find_many(
        where={"userId": user_id},
        include={"agent": True, "conversations": {"include": {"messages": True}}},
        order={"createdAt": "desc"},
    )

    agents = await db.aiagent.find_many(
        where={"userId": user_id},
        include={
            "conversations": {"include": {"messages": True}},
            "workspaces": {"include": {"conversations": {"include": {"messages": True}}}},
        },
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
            "status": getattr(a, "status", "active"),
            "archived_at": str(a.archivedAt) if getattr(a, "archivedAt", None) else None,
            "conversation_count": conv_count,
            "message_count": msg_count,
            "workspaces": [
                {
                    "id": w.id,
                    "status": w.status,
                    "created_at": str(w.createdAt),
                    "archived_at": str(w.archivedAt) if getattr(w, "archivedAt", None) else None,
                    "conversation_count": len(w.conversations) if w.conversations else 0,
                    "message_count": sum(
                        len(c.messages) for c in (w.conversations or []) if c.messages
                    ),
                }
                for w in (a.workspaces or [])
            ],
            "conversations": [
                {
                    "id": c.id,
                    "created_at": str(c.createdAt),
                    "updated_at": str(c.updatedAt),
                    "is_deleted": c.isDeleted,
                    "workspace_id": c.workspaceId,
                    "archived_at": str(c.archivedAt) if getattr(c, "archivedAt", None) else None,
                    "message_count": len(c.messages) if c.messages else 0,
                }
                for c in (a.conversations or [])
            ],
        })

    return {
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "created_at": str(user.createdAt),
        },
        "workspaces": [
            {
                "id": w.id,
                "status": w.status,
                "agent_id": w.agentId,
                "agent_name": getattr(getattr(w, "agent", None), "name", None),
                "created_at": str(w.createdAt),
                "archived_at": str(w.archivedAt) if getattr(w, "archivedAt", None) else None,
                "conversation_count": len(w.conversations) if w.conversations else 0,
                "message_count": sum(
                    len(c.messages) for c in (w.conversations or []) if c.messages
                ),
            }
            for w in user_workspaces
        ],
        "agents": agent_list,
    }


@router.get("/{user_id}/agents/{agent_id}/proactive")
async def get_agent_proactive_detail(
    user_id: str,
    agent_id: str,
    _: dict = Depends(require_admin_jwt),
):
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or agent.userId != user_id:
        raise HTTPException(status_code=404, detail="Agent not found for this user")

    workspace = await db.chatworkspace.find_first(
        where={"userId": user_id, "agentId": agent_id},
        order={"createdAt": "desc"},
    )
    if not workspace:
        return {"workspace_id": None, "state": None, "events": [], "logs": []}

    state_rows = await db.query_raw(
        """
        SELECT *
        FROM proactive_states
        WHERE workspace_id = $1
        LIMIT 1
        """,
        workspace.id,
    )
    event_rows = await db.query_raw(
        """
        SELECT event_type, window_name, trigger_type, payload, created_at
        FROM proactive_event_logs
        WHERE workspace_id = $1
        ORDER BY created_at DESC
        LIMIT 30
        """,
        workspace.id,
    )
    logs = await db.proactivechatlog.find_many(
        where={"workspaceId": workspace.id},
        order={"createdAt": "desc"},
        take=20,
    )

    state = None
    if state_rows:
        row = state_rows[0]
        state = {
            "status": row.get("status"),
            "stage": row.get("stage"),
            "silence_level_n": int(row.get("silence_level_n") or 0),
            "followup_plan_type": row.get("followup_plan_type"),
            "remaining_forced_triggers": row.get("remaining_forced_triggers"),
            "current_window_index": row.get("current_window_index"),
            "window_due_at": str(row["window_due_at"]) if row.get("window_due_at") else None,
            "response_deadline_at": str(row["response_deadline_at"]) if row.get("response_deadline_at") else None,
            "t0_at": str(row["t0_at"]) if row.get("t0_at") else None,
            "last_proactive_at": str(row["last_proactive_at"]) if row.get("last_proactive_at") else None,
            "last_user_reply_at": str(row["last_user_reply_at"]) if row.get("last_user_reply_at") else None,
            "last_assistant_reply_at": str(row["last_assistant_reply_at"]) if row.get("last_assistant_reply_at") else None,
            "stop_reason": row.get("stop_reason"),
            "metadata": row.get("metadata"),
        }

    return {
        "workspace_id": workspace.id,
        "state": state,
        "events": [
            {
                "event_type": row.get("event_type"),
                "window_name": row.get("window_name"),
                "trigger_type": row.get("trigger_type"),
                "payload": row.get("payload"),
                "created_at": str(row["created_at"]),
            }
            for row in event_rows
        ],
        "logs": [
            {
                "message": log.message,
                "event_type": log.eventType,
                "created_at": str(log.createdAt),
            }
            for log in logs
        ],
    }


@router.patch("/{user_id}/role")
async def update_user_role(
    user_id: str,
    payload: UpdateUserRoleRequest,
    claims: dict = Depends(require_admin_jwt),
):
    """修改用户角色（user / admin）。禁止自己改自己，避免误锁定最后一个 admin。"""
    if claims.get("user_id") == user_id or claims.get("sub") == user_id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.role == payload.role:
        return {"ok": True, "id": user.id, "role": user.role, "changed": False}

    # 若将最后一个 admin 降级为普通用户，拒绝
    if user.role == "admin" and payload.role != "admin":
        admin_count = await db.user.count(where={"role": "admin"})
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot demote the last admin")

    updated = await db.user.update(
        where={"id": user_id},
        data={"role": payload.role},
    )
    return {"ok": True, "id": updated.id, "role": updated.role, "changed": True}


@router.delete("/{user_id}/agents/{agent_id}")
async def delete_user_agent(
    user_id: str,
    agent_id: str,
    _: dict = Depends(require_admin_jwt),
):
    """彻底删除用户与指定 Agent 的全部数据。"""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or agent.userId != user_id:
        raise HTTPException(status_code=404, detail="Agent not found for this user")

    from app.services.runtime.data_reset import hard_delete_agent_data

    stats = await hard_delete_agent_data(agent_id, user_id)
    return {"ok": True, "stats": stats}


