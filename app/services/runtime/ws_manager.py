"""WebSocket 连接管理器。

管理活跃的 WebSocket 连接，支持按 conversation_id 或 user_id 推送消息。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from starlette.websockets import WebSocket, WebSocketState

logger = logging.getLogger(__name__)


class ConnectionManager:
    """In-memory WebSocket connection tracker (asyncio-safe)."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._user_convs: dict[str, set[str]] = {}
        self._conv_users: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def connect(self, conv_id: str, user_id: str, ws: WebSocket) -> None:
        async with self._lock:
            old = self._connections.get(conv_id)
            if old and old.client_state == WebSocketState.CONNECTED:
                try:
                    await old.close(code=4001, reason="replaced")
                except Exception:
                    pass

            self._connections[conv_id] = ws
            self._conv_users[conv_id] = user_id
            self._user_convs.setdefault(user_id, set()).add(conv_id)
        logger.debug(f"WS connected: conv={conv_id[:8]} user={user_id[:8]}")

    async def disconnect(self, conv_id: str) -> None:
        async with self._lock:
            self._connections.pop(conv_id, None)
            user_id = self._conv_users.pop(conv_id, None)
            if user_id and user_id in self._user_convs:
                self._user_convs[user_id].discard(conv_id)
                if not self._user_convs[user_id]:
                    del self._user_convs[user_id]
        logger.debug(f"WS disconnected: conv={conv_id[:8]}")

    def get(self, conv_id: str) -> WebSocket | None:
        ws = self._connections.get(conv_id)
        if ws and ws.client_state == WebSocketState.CONNECTED:
            return ws
        return None

    async def send_event(self, conv_id: str, event_type: str, data: Any = None) -> bool:
        ws = self.get(conv_id)
        if not ws:
            return False
        try:
            await ws.send_json({"type": event_type, "data": data or {}})
            return True
        except Exception as e:
            logger.warning(f"WS send failed for conv={conv_id[:8]}: {e}")
            await self.disconnect(conv_id)
            return False

    async def send_to_user(self, user_id: str, event_type: str, data: Any = None) -> int:
        conv_ids = self._user_convs.get(user_id, set()).copy()
        if not conv_ids:
            return 0
        results = await asyncio.gather(
            *[self.send_event(cid, event_type, data) for cid in conv_ids],
            return_exceptions=True,
        )
        return sum(1 for r in results if r is True)


manager = ConnectionManager()
