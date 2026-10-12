"""Admin inbox WebSocket. Separate from the one-socket-per-conversation map."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket

from app.redis_client import is_redis_healthy
from app.services.runtime.ws_auth import (
    CHAT_PROTOCOL,
    TicketInvalid,
    TicketUnavailable,
    consume_ticket,
    offered_ticket,
    origin_allowed,
)
from app.services.runtime.ws_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin-inbox"])
_IDLE_TIMEOUT = 90.0


@router.websocket("/ws/admin-inbox/{admin_id}")
async def admin_inbox(websocket: WebSocket, admin_id: str):
    if not is_redis_healthy():
        await websocket.close(code=1011, reason="redis_unavailable")
        return
    try:
        ticket = offered_ticket(websocket.scope.get("subprotocols", []))
        if not origin_allowed(websocket.headers.get("origin")):
            raise TicketInvalid()
        principal = await consume_ticket(ticket, f"admin-inbox:{admin_id}")
    except TicketInvalid:
        await websocket.close(code=4401, reason="authentication_required")
        return
    except TicketUnavailable:
        await websocket.close(code=1011, reason="authentication_unavailable")
        return
    if principal.role != "admin" or principal.user_id != admin_id:
        await websocket.close(code=4403, reason="admin_inbox_denied")
        return

    await websocket.accept(subprotocol=CHAT_PROTOCOL)
    await manager.connect_admin(admin_id, websocket)
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_json(), timeout=_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                await websocket.close(code=4008, reason="Idle timeout")
                break
            if (raw or {}).get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except Exception as exc:
        logger.debug("admin inbox closed admin=%s: %s", admin_id[:8], exc)
    finally:
        await manager.disconnect_admin(admin_id, expected=websocket)
