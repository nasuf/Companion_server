"""Foreground presence used to suppress remote notifications."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_TTL_SECONDS = 90

# 实时在线用两个池, 取并集去重按 user_id 计数 (跨进程, App/H5 一视同仁):
#
# 1) WS 连接池 (presence:online:ws) — member="{user_id}|{conn_id}", score=最近活跃.
#    连接即 ZADD, **断开即 ZREM (瞬时下线)**, ping/消息刷新 score. 这是"连接数"语义,
#    离开聊天页会立刻减少. TTL 仅作 worker 崩溃/漏发 disconnect 的兜底清理.
# 2) 心跳池 (presence:online:hb) — member=user_id, score=最近活跃. 由天然轮询式信号
#    喂: App 前台 presence(45s) / H5 页面可见心跳(40s) / 登录. 这类"前台开着但不在
#    聊天"无法瞬时感知离开, 只能靠 TTL 过期; 但 App 切后台 / H5 页面隐藏会显式移除.
#
# 一个用户只要在任一池活跃即算在线.
_ONLINE_WS_ZKEY = "presence:online:ws"
_ONLINE_HB_ZKEY = "presence:online:hb"
_ONLINE_TTL_SECONDS = 90


def _ws_member(user_id: str, conn_id: str) -> str:
    return f"{user_id}|{conn_id}"


async def record_ws_online(user_id: str | None, conn_id: str) -> None:
    """WS 连接建立 / 收到帧时调用: 把该连接标记为在线. Best-effort."""
    if not user_id or not conn_id:
        return
    try:
        redis = await get_redis()
        await redis.zadd(_ONLINE_WS_ZKEY, {_ws_member(user_id, conn_id): time.time()})
    except Exception as e:
        logger.debug(f"[PRESENCE] record_ws_online skipped user={user_id[:8]}: {e}")


async def remove_ws_online(user_id: str | None, conn_id: str) -> None:
    """WS 断开时调用: 立即摘除该连接 → 实时在线瞬时反映离开. Best-effort."""
    if not user_id or not conn_id:
        return
    try:
        redis = await get_redis()
        await redis.zrem(_ONLINE_WS_ZKEY, _ws_member(user_id, conn_id))
    except Exception as e:
        logger.debug(f"[PRESENCE] remove_ws_online skipped user={user_id[:8]}: {e}")


async def record_online(user_id: str | None) -> None:
    """心跳池: 标记用户前台在线 (App 前台 / H5 可见 / 登录). Best-effort."""
    if not user_id:
        return
    try:
        redis = await get_redis()
        await redis.zadd(_ONLINE_HB_ZKEY, {user_id: time.time()})
    except Exception as e:
        logger.debug(f"[PRESENCE] record_online skipped user={user_id[:8]}: {e}")


async def remove_online(user_id: str | None) -> None:
    """心跳池显式下线 (App 切后台 / H5 页面隐藏或关闭). Best-effort."""
    if not user_id:
        return
    try:
        redis = await get_redis()
        await redis.zrem(_ONLINE_HB_ZKEY, user_id)
    except Exception as e:
        logger.debug(f"[PRESENCE] remove_online skipped user={user_id[:8]}: {e}")


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
            # App 切后台 → 从心跳池移除, 实时在线及时下降 (原生每次切后台都会发).
            await remove_online(user_id)
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
    """Count distinct users currently online = union of the WS + heartbeat pools.

    WS pool gives instant connection-count semantics (leaving the chat drops the
    number immediately); heartbeat pool covers app/page foreground outside a chat.
    Both pools are pruned by _ONLINE_TTL_SECONDS as a crash/lost-signal backstop,
    then their members are unioned by user_id.

    Returns (distinct_user_count, redis_ok). On Redis failure returns (0, False)
    so the admin dashboard can surface degraded state instead of a wrong zero.
    """
    try:
        redis = await get_redis()
        cutoff = time.time() - _ONLINE_TTL_SECONDS
        await redis.zremrangebyscore(_ONLINE_WS_ZKEY, "-inf", cutoff)
        await redis.zremrangebyscore(_ONLINE_HB_ZKEY, "-inf", cutoff)
        ws_members, hb_members = await asyncio.gather(
            redis.zrange(_ONLINE_WS_ZKEY, 0, -1),
            redis.zrange(_ONLINE_HB_ZKEY, 0, -1),
        )
        users: set[str] = set()
        for m in ws_members:
            member = m if isinstance(m, str) else m.decode()
            users.add(member.split("|", 1)[0])  # "{user_id}|{conn_id}" → user_id
        for m in hb_members:
            users.add(m if isinstance(m, str) else m.decode())
        return len(users), True
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
