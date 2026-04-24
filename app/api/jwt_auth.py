from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.db import db
from app.services.auth import decode_jwt

bearer_scheme = HTTPBearer(auto_error=False)


async def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    """Decode Bearer JWT and return payload with 'sub' (user_id) and 'role'."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token required",
        )
    try:
        payload = decode_jwt(credentials.credentials)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return payload


async def require_admin_jwt(payload: dict = Depends(require_user)) -> dict:
    """Like require_user but enforces role=='admin'."""
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return payload


async def verify_agent_owner(agent_id: str, user_payload: dict):
    """查 agent + 校验当前用户是 owner. 返回 agent 对象或抛 404/403.

    Agent 天然 per-user (AiAgent.userId 非空), 本 helper 确保 API 调用者
    持有的 JWT 对应 user 确实是该 agent 的拥有者, 防止通过猜 agent_id
    访问他人数据.
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.userId != user_payload.get("sub"):
        raise HTTPException(status_code=403, detail="Not your agent")
    return agent
