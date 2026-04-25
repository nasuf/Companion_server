"""Ownership 校验 dependency.

与 jwt_auth 职责分离: jwt_auth 只解码 token; 本文件负责 "token 对应用户
是否真的拥有目标资源 (agent / conversation / memory / 自身 user_id)" 的 ORM 校验.

公开 endpoint 严格只让 owner 通过; admin 走 /admin-api/* (require_admin_jwt),
不在 public 端给 admin 后门.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from app.api.jwt_auth import require_user
from app.db import db


async def require_agent_owner_any_status(
    agent_id: str,
    user: dict = Depends(require_user),
):
    """JWT 解码 + agent 查询 + owner 校验, 不限 agent.status.

    用法: provision-status 等需要在 status="provisioning" 阶段访问的 endpoint.
    其他默认场景应该用 `require_agent_owner` (active-only).
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your agent")
    return agent


async def require_agent_owner(
    agent=Depends(require_agent_owner_any_status),
):
    """同 require_agent_owner_any_status 但额外要求 agent.status="active".

    agent_id 由 FastAPI 路径参数自动注入到上游 require_agent_owner_any_status
    (形参名必须与 URL `{agent_id}` 对齐). FastAPI 同一 request 内会 cache
    dependency 结果, 共享只查一次 DB.
    """
    if getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


async def require_user_self(
    user_id: str,
    user: dict = Depends(require_user),
) -> dict:
    """路径/查询参数 user_id 必须等于 JWT 的 sub.

    用法 (path param):
        @router.get("/intimacy/{agent_id}/{user_id}")
        async def h(... _=Depends(require_user_self)): ...

    用法 (query param): 把 user_id 也作为 endpoint 形参声明 (str 类型 +
    无默认值或 = Query(...)), FastAPI 会从 query string 解析后注入到
    dependency 的 user_id 形参里.
    """
    if user_id != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your data")
    return user


async def require_conversation_owner(
    conversation_id: str,
    user: dict = Depends(require_user),
):
    """JWT 用户必须是 conversation 的 owner. 返回 Conversation 对象.

    isDeleted=True 的会话视为 not found (与已有 GET /conversations 行为对齐).
    """
    conv = await db.conversation.find_unique(where={"id": conversation_id})
    if not conv or conv.isDeleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your conversation")
    return conv


async def require_memory_owner(
    memory_id: str,
    user: dict = Depends(require_user),
):
    """JWT 用户必须是 memory 的 owner. 返回 memory 对象.

    memory_repo.find_unique 内部跨 memories_user / memories_ai 两表查找,
    任一表命中即可校验 owner.
    """
    from app.services.memory.storage import repo as memory_repo
    m = await memory_repo.find_unique(memory_id)
    if not m:
        raise HTTPException(status_code=404, detail="Memory not found")
    if m.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your memory")
    return m
