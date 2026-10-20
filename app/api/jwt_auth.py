from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

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
    # JWT signatures survive deletion. Current SQL identity is authoritative
    # for account lifecycle and admin role, including retries while deleting.
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Invalid account")
    from app.db import db
    user = await db.user.find_unique(where={"id": user_id})
    if not user or user.status != "active" or user.archivedAt is not None:
        raise HTTPException(status_code=401, detail="Account unavailable")
    payload["role"] = user.role
    return payload


async def require_admin_jwt(payload: dict = Depends(require_user)) -> dict:
    """Like require_user but enforces role=='admin'."""
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return payload


