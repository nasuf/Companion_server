"""Foreground presence used to suppress remote notifications."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_TTL_SECONDS = 90

# 统一"实时在线"计数. 与 push:presence:* (推送抑制, 按 device 存, 仅原生 App 写)
# 解耦: 这个 ZSET 由所有平台的活跃信号喂 (WS 连接/ping/消息 + 登录/auth_me +
# App 前台 presence), member=user_id / score=最近活跃 epoch 秒. 读取时按 score
# 剔除 _ONLINE_TTL_SECONDS 之前的陈旧成员再计数 → 跨进程准确, App / H5 一视同仁.
_ONLINE_ZKEY = "presence:online"
_ONLINE_TTL_SECONDS = 90


async def record_online(user_id: str | None) -> None:
    """标记某用户此刻在线 (刷新其在 online ZSET 的时间戳). Best-effort, 不抛."""
    if not user_id:
        return
    try:
        redis = await get_redis()
        await redis.zadd(_ONLINE_ZKEY, {user_id: time.time()})
    except Exception as e:
        logger.debug(f"[PRESENCE] record_online skipped user={user_id[:8]}: {e}")


def _key(user_id: str, device_id: str) -> str:
    return f"push:presence:{user_id}:{device_id}"


def _user_pattern(user_id: str) -> str:
    return f"push:presence:{user_id}:*"


async def update_presence(
    *,
    user_id: str,
    device_id: str,
    foreground: bool,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
) -> None:
    if not device_id:
        return
    try:
        redis = await get_redis()
        if not foreground:
            await redis.delete(_key(user_id, device_id))
            logger.info(
                f"[PUSH] presence user={user_id[:8]} device={device_id[:16]} "
                "foreground=false"
            )
            return
        payload = {
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        await redis.set(_key(user_id, device_id), json.dumps(payload), ex=_TTL_SECONDS)
        # 同时喂统一在线计数 (App 前台但未进聊天也算在线).
        await record_online(user_id)
        logger.info(
            f"[PUSH] presence user={user_id[:8]} device={device_id[:16]} "
            f"foreground=true workspace={workspace_id} conversation={conversation_id}"
        )
    except Exception as e:
        logger.debug(f"[PUSH] presence update skipped: {e}")


async def count_online_users() -> tuple[int, bool]:
    """Count distinct users active within the last _ONLINE_TTL_SECONDS.

    Reads the unified `presence:online` ZSET fed by every platform (WS
    connect/ping/message, login/auth_me, App foreground presence), so both App
    and H5 users are counted. Stale members (older than the TTL) are pruned on
    read, then ZCARD gives the live distinct-user count (cross-process via Redis).

    Returns (distinct_user_count, redis_ok). On Redis failure returns (0, False)
    so the admin dashboard can surface degraded state instead of a wrong zero.
    """
    try:
        redis = await get_redis()
        cutoff = time.time() - _ONLINE_TTL_SECONDS
        await redis.zremrangebyscore(_ONLINE_ZKEY, "-inf", cutoff)
        count = await redis.zcard(_ONLINE_ZKEY)
        return int(count or 0), True
    except Exception as e:
        logger.debug(f"[PRESENCE] online count skipped: {e}")
        return 0, False


async def is_user_foreground(
    *,
    user_id: str,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
) -> bool:
    try:
        redis = await get_redis()
        async for key in redis.scan_iter(_user_pattern(user_id), count=20):
            raw = await redis.get(key)
            if not raw:
                continue
            try:
                payload = json.loads(raw if isinstance(raw, str) else raw.decode())
            except Exception:
                return True
            active_workspace = payload.get("workspace_id")
            active_conversation = payload.get("conversation_id")
            if conversation_id and active_conversation == conversation_id:
                return True
            if workspace_id and active_workspace == workspace_id:
                return True
            if not workspace_id and not conversation_id:
                return True
    except Exception as e:
        logger.debug(f"[PUSH] presence read skipped: {e}")
    return False
