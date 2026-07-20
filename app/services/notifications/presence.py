"""Foreground presence used to suppress remote notifications."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_TTL_SECONDS = 90


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
        logger.info(
            f"[PUSH] presence user={user_id[:8]} device={device_id[:16]} "
            f"foreground=true workspace={workspace_id} conversation={conversation_id}"
        )
    except Exception as e:
        logger.debug(f"[PUSH] presence update skipped: {e}")


async def count_online_users() -> tuple[int, bool]:
    """Count distinct users with a live foreground presence key.

    Presence keys carry a 90s TTL and the app refreshes them every ~45s while
    foregrounded, so distinct user_ids across live `push:presence:*` keys is a
    reliable real-time "online now" signal (cross-process via Redis).

    Returns (distinct_user_count, redis_ok). On Redis failure returns (0, False)
    so the admin dashboard can surface degraded state instead of a wrong zero.
    """
    try:
        redis = await get_redis()
        users: set[str] = set()
        async for key in redis.scan_iter("push:presence:*", count=200):
            key_str = key if isinstance(key, str) else key.decode()
            # key format: push:presence:{user_id}:{device_id}
            parts = key_str.split(":")
            if len(parts) >= 4:
                users.add(parts[2])
        return len(users), True
    except Exception as e:
        logger.debug(f"[PUSH] presence online count skipped: {e}")
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
