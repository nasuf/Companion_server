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
            return
        payload = {
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        await redis.set(_key(user_id, device_id), json.dumps(payload), ex=_TTL_SECONDS)
    except Exception as e:
        logger.debug(f"[PUSH] presence update skipped: {e}")


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
