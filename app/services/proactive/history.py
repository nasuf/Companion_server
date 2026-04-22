"""主动分享服务。

AI主动发起消息：事件路由 + 频率控制(≤3/日) + 记忆触发。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.db import db
from app.redis_client import get_redis
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

MAX_DAILY_PROACTIVE = 3
MAX_2DAY_PROACTIVE = 4  # n≤4: 2天内最多4次主动(含首次)


def _proactive_count_key(agent_id: str, user_id: str) -> str:
    return f"proactive_count:{agent_id}:{user_id}:{datetime.now(UTC).strftime('%Y%m%d')}"


def _proactive_2day_count_key(agent_id: str, user_id: str) -> str:
    return f"proactive_2day:{agent_id}:{user_id}"


async def can_send_proactive(agent_id: str, user_id: str) -> bool:
    """检查今日是否还能发送主动消息。"""
    redis = await get_redis()
    count = await redis.get(_proactive_count_key(agent_id, user_id))
    return int(count or 0) < MAX_DAILY_PROACTIVE


async def can_send_proactive_2day(agent_id: str, user_id: str) -> bool:
    """检查2天滑动窗口内是否还能发送主动消息（≤4次含首次）。"""
    redis = await get_redis()
    count = await redis.get(_proactive_2day_count_key(agent_id, user_id))
    return int(count or 0) < MAX_2DAY_PROACTIVE


async def increment_proactive_count(agent_id: str, user_id: str) -> None:
    """增加今日主动消息计数。"""
    redis = await get_redis()
    key = _proactive_count_key(agent_id, user_id)
    await redis.incr(key)
    await redis.expire(key, 86400)


async def increment_proactive_2day_count(agent_id: str, user_id: str) -> None:
    """增加2天滑动窗口主动消息计数。"""
    redis = await get_redis()
    key = _proactive_2day_count_key(agent_id, user_id)
    await redis.incr(key)
    await redis.expire(key, 172800)  # 48小时


async def get_proactive_history(
    agent_id: str,
    user_id: str,
    limit: int = 10,
    workspace_id: str | None = None,
) -> list[dict]:
    """获取主动消息历史。"""
    effective_workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    where = {"workspaceId": effective_workspace_id} if effective_workspace_id else {"agentId": agent_id, "userId": user_id}
    logs = await db.proactivechatlog.find_many(
        where=where,
        order={"createdAt": "desc"},
        take=limit,
    )
    return [
        {"content": log.message, "trigger_type": log.eventType, "created_at": str(log.createdAt)}
        for log in logs
    ]
