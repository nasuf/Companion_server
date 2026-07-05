from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.websockets import WebSocket

from app.services.auth import decode_jwt

bearer_scheme = HTTPBearer(auto_error=False)


def _extract_ws_token(websocket: WebSocket) -> str | None:
    """Pull a bearer token from a WebSocket handshake.

    Browsers cannot set custom headers on the WebSocket constructor, so the
    query param `?token=` is the portable transport; native clients (Flutter /
    miniprogram) may use either. Prefer the query param, fall back to the
    Authorization header for header-capable clients.
    """
    token = websocket.query_params.get("token")
    if token:
        return token.strip() or None
    header = websocket.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


def authenticate_ws(websocket: WebSocket) -> dict | None:
    """Decode the WebSocket JWT. Returns the payload or None when absent/invalid.

    Never raises — the caller decides how to close the socket so a specific
    close code can be surfaced to the client.
    """
    token = _extract_ws_token(websocket)
    if not token:
        return None
    try:
        return decode_jwt(token)
    except Exception:
        return None


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


