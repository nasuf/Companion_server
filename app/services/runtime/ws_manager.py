"""WebSocket 连接管理器 (跨进程 Pub/Sub).

管理活跃 WebSocket 连接, 支持:
- 按 conversation_id 推送 (LLM stream / aggregation status / trace_url)
- 按 workspace_id 推送 (proactive / first_greeting / special_dates)
- 跨进程广播 (Redis Pub/Sub) — 多 worker / scheduler 拆容器后仍能推到正确连接

Fast path: 本地 _connections 命中 → 直送 ws.send_json (零 Redis 开销)
Slow path: 本地未命中 → publish 到 Redis channel, 持有连接的 worker 收到后本地推

Channel 命名预留 Redis Cluster Sharded Pub/Sub (Redis 7+ SPUBLISH/SSUBSCRIBE):
- ws:conv:{conv_id}        — hash tag {conv_id} 决定 shard
- ws:workspace:{workspace_id}  — hash tag {workspace_id} 决定 shard

不在 user_id 维度路由: 一个 user 多 agent (多 workspace) 时 send_to_user 会
广播到所有 agent 的 WS, 业务上是 bug. workspace_id 是正确的粒度.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from starlette.websockets import WebSocket, WebSocketState

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_CONV_CHANNEL_PREFIX = "ws:conv:{"
_CHANNEL_SUFFIX = "}"
_WS_CHANNEL_PREFIX = "ws:workspace:{"
_PSUB_PATTERN_CONV = "ws:conv:*"
_PSUB_PATTERN_WS = "ws:workspace:*"


def _conv_channel(conv_id: str) -> str:
    return f"{_CONV_CHANNEL_PREFIX}{conv_id}{_CHANNEL_SUFFIX}"


def _workspace_channel(workspace_id: str) -> str:
    return f"{_WS_CHANNEL_PREFIX}{workspace_id}{_CHANNEL_SUFFIX}"


def _parse_channel(channel: str) -> tuple[str, str] | None:
    """`ws:conv:{abc}` → ("conv", "abc"); `ws:workspace:{xyz}` → ("workspace", "xyz")."""
    for prefix, kind in ((_CONV_CHANNEL_PREFIX, "conv"), (_WS_CHANNEL_PREFIX, "workspace")):
        if channel.startswith(prefix) and channel.endswith(_CHANNEL_SUFFIX):
            return kind, channel[len(prefix):-len(_CHANNEL_SUFFIX)]
    return None


class ConnectionManager:
    """In-memory + Redis Pub/Sub WebSocket connection tracker (asyncio-safe)."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._user_convs: dict[str, set[str]] = {}
        self._workspace_convs: dict[str, set[str]] = {}
        self._conv_users: dict[str, str] = {}
        self._conv_workspace: dict[str, str | None] = {}
        self._lock = asyncio.Lock()
        self._subscriber_task: asyncio.Task | None = None

    # ────────────────────────── lifecycle ──────────────────────────

    async def connect(
        self,
        conv_id: str,
        user_id: str,
        ws: WebSocket,
        *,
        workspace_id: str | None = None,
    ) -> None:
        """注册一条 WS 连接. workspace_id 为 None 时仅 user 维度可达 (兼容历史 conv)."""
        async with self._lock:
            old = self._connections.get(conv_id)
            if old and old.client_state == WebSocketState.CONNECTED:
                try:
                    await old.close(code=4001, reason="replaced")
                except Exception:
                    pass

            self._connections[conv_id] = ws
            self._conv_users[conv_id] = user_id
            self._conv_workspace[conv_id] = workspace_id
            self._user_convs.setdefault(user_id, set()).add(conv_id)
            if workspace_id:
                self._workspace_convs.setdefault(workspace_id, set()).add(conv_id)
        logger.debug(
            f"WS connected: conv={conv_id[:8]} user={user_id[:8]} "
            f"workspace={(workspace_id or 'none')[:8]}"
        )

    async def disconnect(self, conv_id: str) -> None:
        async with self._lock:
            self._connections.pop(conv_id, None)
            user_id = self._conv_users.pop(conv_id, None)
            workspace_id = self._conv_workspace.pop(conv_id, None)
            if user_id and user_id in self._user_convs:
                self._user_convs[user_id].discard(conv_id)
                if not self._user_convs[user_id]:
                    del self._user_convs[user_id]
            if workspace_id and workspace_id in self._workspace_convs:
                self._workspace_convs[workspace_id].discard(conv_id)
                if not self._workspace_convs[workspace_id]:
                    del self._workspace_convs[workspace_id]
        logger.debug(f"WS disconnected: conv={conv_id[:8]}")

    def get(self, conv_id: str) -> WebSocket | None:
        ws = self._connections.get(conv_id)
        if ws and ws.client_state == WebSocketState.CONNECTED:
            return ws
        return None

    # ────────────────────────── 推送 (跨进程) ──────────────────────────

    async def send_event(self, conv_id: str, event_type: str, data: Any = None) -> bool:
        """conv 维度推送. Fast path 本地直送, slow path 跨进程 publish."""
        if await self._send_local_conv(conv_id, event_type, data):
            return True
        return await self._publish(_conv_channel(conv_id), event_type, data)

    async def send_to_workspace(
        self, workspace_id: str | None, event_type: str, data: Any = None,
    ) -> int:
        """workspace 维度推送 (proactive / first_greeting / special_dates).

        Returns 本地命中并成功推送的连接数 (远端 worker 的成功数不可见).
        workspace_id is None → 兜底退化到 user 维度兼容历史 conv 无 workspace 的场景.
        """
        if not workspace_id:
            user_id = (data or {}).get("user_id") if isinstance(data, dict) else None
            if user_id:
                logger.warning("send_to_workspace 收到 workspace_id=None, 回退 send_to_user")
                return await self.send_to_user(user_id, event_type, data)
            return 0

        local_count = 0
        for cid in self._workspace_convs.get(workspace_id, set()).copy():
            if await self._send_local_conv(cid, event_type, data):
                local_count += 1

        # 同时 publish: 该 workspace 的 conv 可能在别的 worker
        await self._publish(_workspace_channel(workspace_id), event_type, data)
        return local_count

    async def send_to_user(
        self, user_id: str, event_type: str, data: Any = None,
    ) -> int:
        """user 维度推送 (deprecated: 跨 agent 广播, 业务正确性差).

        保留作为 send_to_workspace(workspace_id=None) 的兜底, 不应被新代码调用.
        """
        local_count = 0
        for cid in self._user_convs.get(user_id, set()).copy():
            if await self._send_local_conv(cid, event_type, data):
                local_count += 1
        return local_count

    # ────────────────────────── 内部 ──────────────────────────

    async def _send_local_conv(
        self, conv_id: str, event_type: str, data: Any,
    ) -> bool:
        ws = self._connections.get(conv_id)
        if not ws or ws.client_state != WebSocketState.CONNECTED:
            return False
        try:
            await ws.send_json({"type": event_type, "data": data or {}})
            return True
        except Exception as e:
            logger.warning(f"WS send failed conv={conv_id[:8]} type={event_type}: {e}")
            await self.disconnect(conv_id)
            return False

    async def _publish(self, channel: str, event_type: str, data: Any) -> bool:
        try:
            redis = await get_redis()
            payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
            await redis.publish(channel, payload)
            return True
        except Exception as e:
            logger.warning(f"WS publish failed channel={channel}: {e}")
            return False

    # ────────────────────────── 跨进程订阅 ──────────────────────────

    async def start_subscriber(self) -> None:
        """lifespan startup 调用. Redis 挂时降级到本地直送, 不抛."""
        if self._subscriber_task and not self._subscriber_task.done():
            return
        self._subscriber_task = asyncio.create_task(self._subscribe_loop())

    async def stop_subscriber(self) -> None:
        if self._subscriber_task and not self._subscriber_task.done():
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except (asyncio.CancelledError, Exception):
                pass
            self._subscriber_task = None

    async def _subscribe_loop(self) -> None:
        """长期运行的订阅 task. Redis 异常时静默重试, 不向上抛打挂主进程."""
        while True:
            try:
                redis = await get_redis()
                pubsub = redis.pubsub()
                await pubsub.psubscribe(_PSUB_PATTERN_CONV, _PSUB_PATTERN_WS)
                logger.info(
                    f"WS subscriber started: psubscribe "
                    f"{_PSUB_PATTERN_CONV} {_PSUB_PATTERN_WS}"
                )
                async for msg in pubsub.listen():
                    if msg.get("type") not in ("pmessage", "message"):
                        continue
                    await self._handle_message(msg)
            except asyncio.CancelledError:
                logger.info("WS subscriber cancelled")
                return
            except Exception as e:
                logger.warning(f"WS subscriber loop error, retry in 5s: {e}")
                await asyncio.sleep(5)

    async def _handle_message(self, msg: dict) -> None:
        channel = msg.get("channel")
        data_raw = msg.get("data")
        if isinstance(channel, bytes):
            channel = channel.decode()
        if isinstance(data_raw, bytes):
            data_raw = data_raw.decode()
        if not isinstance(channel, str) or not isinstance(data_raw, str):
            return
        parsed = _parse_channel(channel)
        if parsed is None:
            return
        kind, scope_id = parsed
        try:
            payload = json.loads(data_raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"WS subscriber bad payload on {channel}")
            return
        event_type = payload.get("type", "")
        data = payload.get("data")

        if kind == "conv":
            await self._send_local_conv(scope_id, event_type, data)
        elif kind == "workspace":
            for cid in self._workspace_convs.get(scope_id, set()).copy():
                await self._send_local_conv(cid, event_type, data)


manager = ConnectionManager()
