from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator

from app.db import db
from app.api.jwt_auth import require_admin_jwt
from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME

_ALLOWED_ROLES = {"user", "admin"}
_WECHAT_PROVIDER = "wechat"
_PHONE_PROVIDER = "phone"


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


def _text_or_none(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _serialize_wechat_identity(identity) -> dict | None:
    if not identity:
        return None
    profile = getattr(identity, "rawProfile", None)
    if not isinstance(profile, dict):
        profile = {}
    privilege = profile.get("privilege")
    return {
        "provider": getattr(identity, "provider", _WECHAT_PROVIDER),
        "provider_account_id": _text_or_none(
            getattr(identity, "providerAccountId", None)
        ),
        "openid": _text_or_none(getattr(identity, "openid", None))
        or _text_or_none(profile.get("openid")),
        "unionid": _text_or_none(getattr(identity, "unionid", None))
        or _text_or_none(profile.get("unionid")),
        "scope": _text_or_none(getattr(identity, "scope", None))
        or _text_or_none(profile.get("scope")),
        "nickname": _text_or_none(profile.get("nickname")),
        "avatar_url": _text_or_none(profile.get("headimgurl")),
        "sex": profile.get("sex"),
        "province": _text_or_none(profile.get("province")),
        "city": _text_or_none(profile.get("city")),
        "country": _text_or_none(profile.get("country")),
        "privilege": privilege if isinstance(privilege, list) else [],
        "last_login_at": str(identity.lastLoginAt)
        if getattr(identity, "lastLoginAt", None)
        else None,
        "created_at": str(identity.createdAt)
        if getattr(identity, "createdAt", None)
        else None,
        "updated_at": str(identity.updatedAt)
        if getattr(identity, "updatedAt", None)
        else None,
    }


def _serialize_phone_identity(identity) -> dict | None:
    if not identity:
        return None
    profile = getattr(identity, "rawProfile", None)
    if not isinstance(profile, dict):
        profile = {}
    phone = (
        _text_or_none(getattr(identity, "providerAccountId", None))
        or _text_or_none(profile.get("phone"))
    )
    if not phone:
        return None
    return {
        "provider": getattr(identity, "provider", _PHONE_PROVIDER),
        "provider_account_id": _text_or_none(
            getattr(identity, "providerAccountId", None)
        ),
        "phone": phone,
        "phone_masked": _mask_phone(phone),
        "last_login_at": str(identity.lastLoginAt)
        if getattr(identity, "lastLoginAt", None)
        else None,
        "created_at": str(identity.createdAt)
        if getattr(identity, "createdAt", None)
        else None,
        "updated_at": str(identity.updatedAt)
        if getattr(identity, "updatedAt", None)
        else None,
    }


def _mask_phone(phone: str | None) -> str | None:
    if not phone or len(phone) != 11:
        return phone
    return f"{phone[:3]}****{phone[-4:]}"


def _serialize_password_method(user) -> dict | None:
    if not getattr(user, "hashedPassword", None):
        return None
    email = _text_or_none(getattr(user, "email", None))
    username = _text_or_none(getattr(user, "username", None))
    return {
        "type": "password",
        "label": "邮箱密码" if email else "账号密码",
        "identifier": email or username,
        "email": email,
        "username": username,
        "signup_source": _text_or_none(getattr(user, "signupSource", None)),
    }


def _serialize_auth_methods(user, identities: list[object]) -> list[dict]:
    methods: list[dict] = []
    password = _serialize_password_method(user)
    if password:
        methods.append(password)

    wechat_identity = _first_identity(identities, _WECHAT_PROVIDER)
    wechat = _serialize_wechat_identity(wechat_identity)
    if wechat:
        methods.append(
            {
                "type": "wechat",
                "label": "微信",
                "identifier": wechat.get("nickname")
                or wechat.get("openid")
                or wechat.get("provider_account_id"),
                "provider_account_id": wechat.get("provider_account_id"),
                "openid": wechat.get("openid"),
                "unionid": wechat.get("unionid"),
                "nickname": wechat.get("nickname"),
                "avatar_url": wechat.get("avatar_url"),
                "last_login_at": wechat.get("last_login_at"),
            }
        )

    phone_identity = _first_identity(identities, _PHONE_PROVIDER)
    phone = _serialize_phone_identity(phone_identity)
    if phone:
        methods.append(
            {
                "type": "phone",
                "label": "手机号",
                "identifier": phone.get("phone_masked") or phone.get("phone"),
                "phone": phone.get("phone"),
                "phone_masked": phone.get("phone_masked"),
                "last_login_at": phone.get("last_login_at"),
            }
        )
    return methods


def _first_identity(identities: list[object], provider: str):
    for identity in identities:
        if getattr(identity, "provider", None) == provider:
            return identity
    return None


async def _identities_by_user(user_ids: list[str]) -> dict[str, list[object]]:
    if not user_ids:
        return {}
    identities = await db.authidentity.find_many(
        where={
            "userId": {"in": user_ids},
            "provider": {"in": [_WECHAT_PROVIDER, _PHONE_PROVIDER]},
        },
        order={"updatedAt": "desc"},
    )
    by_user: dict[str, list[object]] = {}
    for identity in identities:
        user_id = getattr(identity, "userId", None)
        if user_id:
            by_user.setdefault(user_id, []).append(identity)
    return by_user


def _serialize_admin_user(
    user,
    identities: list[object],
    *,
    agent_count: int | None = None,
) -> dict:
    resolved_agent_count = (
        agent_count
        if agent_count is not None
        else len(user.agents)
        if getattr(user, "agents", None)
        else 0
    )
    return {
        "id": user.id,
        "username": user.username,
        "email": getattr(user, "email", None),
        "role": user.role,
        "created_at": str(user.createdAt),
        "status": getattr(user, "status", "active"),
        "archived_at": str(user.archivedAt)
        if getattr(user, "archivedAt", None)
        else None,
        "signup_source": getattr(user, "signupSource", None),
        "agent_count": resolved_agent_count,
        "wechat": _serialize_wechat_identity(
            _first_identity(identities, _WECHAT_PROVIDER)
        ),
        "phone": _serialize_phone_identity(
            _first_identity(identities, _PHONE_PROVIDER)
        ),
        "auth_methods": _serialize_auth_methods(user, identities),
    }


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
    # The reserved template system user owns all agent templates; hide it from
    # the admin user list so list-driven operations (e.g. batch delete) can
    # never touch it.
    conditions: list[dict] = [{"username": {"not": TEMPLATE_SYSTEM_USERNAME}}]
    if search.strip():
        conditions.append(
            {"username": {"contains": search.strip(), "mode": "insensitive"}}
        )
    where = {"AND": conditions}

    total = await db.user.count(where=where)
    users = await db.user.find_many(
        where=where,
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
        include={"agents": True},
    )
    identities_by_user = await _identities_by_user([u.id for u in users])

    return {
        "users": [
            _serialize_admin_user(
                u,
                identities_by_user.get(u.id, []),
                agent_count=len(u.agents) if u.agents else 0,
            )
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
    identities = await db.authidentity.find_many(
        where={
            "userId": user_id,
            "provider": {"in": [_WECHAT_PROVIDER, _PHONE_PROVIDER]},
        },
        order={"updatedAt": "desc"},
    )

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
        "user": _serialize_admin_user(user, identities, agent_count=len(agents)),
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


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    claims: dict = Depends(require_admin_jwt),
):
    """彻底删除用户及该用户关联的全部 Agent 和用户级数据。"""
    if claims.get("user_id") == user_id or claims.get("sub") == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Deleting the template system user would cascade-delete every agent
    # template it owns (2026-07 production incident via batch delete).
    if user.username == TEMPLATE_SYSTEM_USERNAME:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the template system user (owns agent templates)",
        )

    if user.role == "admin":
        admin_count = await db.user.count(where={"role": "admin"})
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last admin")

    from app.services.runtime.data_reset import hard_delete_user_data

    stats = await hard_delete_user_data(user_id)
    return {"ok": True, "stats": stats}
