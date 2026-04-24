"""Agent ownership 校验 dependency.

与 jwt_auth 职责分离: jwt_auth 只解码 token; 本文件负责 "token 对应用户
是否真的拥有目标 agent_id" 的 ORM 校验.

当前仅覆盖路径参数为 `{agent_id}` 的 endpoint (emotions 等). Conversation
/ workspace scoped endpoint (如 `/conversations/{conversation_id}`)
未来需单独加 `require_conversation_owner` helper (通过 conversationId
→ workspaceId → userId 链路校验), 不能直接复用本文件的 factory.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from app.api.jwt_auth import require_user
from app.db import db


async def require_agent_owner(
    agent_id: str,
    user: dict = Depends(require_user),
):
    """FastAPI 依赖: JWT 解码 + agent 查询 + owner 校验, 单个 Depends 搞定.

    用法:
        @router.get("/emotions/{agent_id}/current")
        async def handler(agent = Depends(require_agent_owner)):
            ...  # agent 对象已通过 404 / 403 校验

    agent_id 由 FastAPI 路径参数自动注入 (形参名必须与 URL `{agent_id}` 对齐).
    FastAPI 同一 request 内会 cache dependency 结果, 多个 endpoint 共享
    本 dependency 只查一次 DB.
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your agent")
    return agent
