"""主动分享服务。

AI 主动消息频率控制 (spec §9: 沉默+记忆合计 ≤3 次/日).

计数存储: Redis 是 primary (快), proactive_counters 表是 snapshot (持久化).
Redis 挂时降级读 DB; 写时 Redis incr + fire_background 异步 upsert DB.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.db import db
from app.redis_client import get_redis
from app.services.runtime.tasks import fire_background
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

MAX_DAILY_PROACTIVE = 3
_DAILY_TTL_SEC = 86400


def _today_key() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def _proactive_count_key(agent_id: str, user_id: str) -> str:
    return f"proactive_count:{agent_id}:{user_id}:{_today_key()}"


async def _daily_count_from_db(agent_id: str, user_id: str, date: str) -> int:
    try:
        row = await db.proactivecounter.find_unique(
            where={
                "agentId_userId_date": {
                    "agentId": agent_id, "userId": user_id, "date": date,
                },
            },
        )
        return row.count if row else 0
    except Exception as e:
        logger.warning(f"[PROACTIVE-COUNT] DB read failed date={date}: {e}")
        return 0


async def can_send_proactive(agent_id: str, user_id: str) -> bool:
    """检查今日是否还能发送主动消息 (Redis 优先, 挂则 DB fallback)."""
    redis = await get_redis()
    try:
        count = await redis.get(_proactive_count_key(agent_id, user_id))
        if count is not None:
            return int(count) < MAX_DAILY_PROACTIVE
    except Exception as e:
        logger.warning(f"[PROACTIVE-COUNT] Redis read failed: {e}")

    # Redis miss 或异常 → DB
    total = await _daily_count_from_db(agent_id, user_id, _today_key())
    # 回填 Redis (24h TTL), 失败无所谓
    try:
        await redis.set(_proactive_count_key(agent_id, user_id), total, ex=_DAILY_TTL_SEC)
    except Exception:
        pass
    return total < MAX_DAILY_PROACTIVE


async def _upsert_counter(agent_id: str, user_id: str, date: str) -> None:
    """原子 upsert: 存在则 count+1, 否则 count=1. 失败仅 warning (Redis 仍是 primary)."""
    try:
        await db.proactivecounter.upsert(
            where={
                "agentId_userId_date": {
                    "agentId": agent_id, "userId": user_id, "date": date,
                },
            },
            data={
                "create": {
                    "agentId": agent_id, "userId": user_id,
                    "date": date, "count": 1,
                },
                "update": {"count": {"increment": 1}},
            },
        )
    except Exception as e:
        logger.warning(
            f"[PROACTIVE-COUNT] DB upsert failed "
            f"agent={agent_id} user={user_id} date={date}: {e}"
        )


async def increment_proactive_count(agent_id: str, user_id: str) -> None:
    """日计数 +1: Redis incr (主路径) + 异步 DB upsert (持久化)."""
    redis = await get_redis()
    key = _proactive_count_key(agent_id, user_id)
    try:
        await redis.incr(key)
        await redis.expire(key, _DAILY_TTL_SEC)
    except Exception as e:
        logger.warning(f"[PROACTIVE-COUNT] Redis incr failed: {e}")
    # fire-and-forget DB upsert, 不阻塞 caller
    fire_background(_upsert_counter(agent_id, user_id, _today_key()))


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
