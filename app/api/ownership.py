"""Agent / workspace ownership 校验 dependencies.

职责与 jwt_auth 分离: jwt_auth 只解码 token; 本文件负责 "token 对应的用户
是否真的拥有目标 agent_id" 类 ORM 校验. 任何 public API 需要 per-agent
鉴权时都应 `Depends(require_agent_owner)` 直接拿到已验证的 agent 对象.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from app.api.jwt_auth import require_user
from app.db import db


async def verify_agent_owner(agent_id: str, user_payload: dict):
    """查 agent 并确认调用者是其 owner. 返回 agent 或抛 404/403.

    独立函数形式，便于非 endpoint 上下文（比如后台任务、其他 helper）复用。
    Endpoint 请优先用 `require_agent_owner` dependency factory。
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.userId != user_payload.get("sub"):
        raise HTTPException(status_code=403, detail="Not your agent")
    return agent


async def require_agent_owner(
    agent_id: str,
    user: dict = Depends(require_user),
):
    """FastAPI 依赖: 合并 JWT 解码 + agent ownership 校验, 单个 Depends 搞定.

    用法:
        @router.get("/emotions/{agent_id}/current")
        async def handler(agent = Depends(require_agent_owner)):
            ...  # agent 对象保证已通过 404 + 403

    agent_id 由 FastAPI 路径参数注入 (函数形参同名即可).
    """
    return await verify_agent_owner(agent_id, user)
