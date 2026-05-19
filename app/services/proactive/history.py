"""主动分享服务。

AI 主动消息频率控制 (spec §9: 沉默+记忆合计 ≤3 次/日).

计数存储: Redis 是 primary (快), proactive_counters 表是 snapshot (持久化).
Redis 挂时降级读 DB; 写时 Redis incr + fire_background 异步 upsert DB.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from app.db import db
from app.redis_client import get_redis
from app.services.runtime.tasks import fire_background
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

MAX_DAILY_PROACTIVE = 3
PROACTIVE_FATIGUE_BLOCK_THRESHOLD = 0.85
_DAILY_TTL_SEC = 86400
_RHYTHM_TZ = "Asia/Shanghai"


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


async def get_proactive_fatigue_score(
    agent_id: str,
    user_id: str,
    *,
    workspace_id: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Return a user-level proactive fatigue score in [0, 1].

    The fixed daily cap remains the hard guard. This softer score catches cases
    where the user is near the cap, has received several proactive messages over
    a few days, or repeatedly does not respond.
    """
    now_ts = (now or datetime.now(UTC)).astimezone(UTC)
    today_count = await _daily_count_from_db(agent_id, user_id, _today_key())
    sent_72h = 0
    reply_timeout_72h = 0
    skipped_24h = 0
    try:
        workspace_clause = "AND workspace_id = $4" if workspace_id else ""
        params = [
            agent_id,
            user_id,
            (now_ts - timedelta(hours=72)).replace(tzinfo=None).isoformat(),
        ]
        if workspace_id:
            params.append(workspace_id)
        skipped_since_idx = len(params) + 1
        rows = await db.query_raw(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE event_type = 'message_sent')::int AS sent_72h,
                COUNT(*) FILTER (WHERE event_type = 'reply_timeout')::int AS reply_timeout_72h,
                COUNT(*) FILTER (
                    WHERE event_type IN ('send_skipped', 'window_deferred')
                      AND created_at >= ${skipped_since_idx}::timestamp
                )::int AS skipped_24h
            FROM proactive_event_logs
            WHERE agent_id = $1
              AND user_id = $2
              AND created_at >= $3::timestamp
              {workspace_clause}
            """,
            *params,
            (now_ts - timedelta(hours=24)).replace(tzinfo=None).isoformat(),
        )
        row = rows[0] if rows else {}
        sent_72h = int(row.get("sent_72h") or 0)
        reply_timeout_72h = int(row.get("reply_timeout_72h") or 0)
        skipped_24h = int(row.get("skipped_24h") or 0)
    except Exception as e:
        logger.warning(f"[PROACTIVE-FATIGUE] DB read failed: {e}")

    components = {
        "today_count": today_count,
        "sent_72h": sent_72h,
        "reply_timeout_72h": reply_timeout_72h,
        "skipped_24h": skipped_24h,
    }
    score = min(
        1.0,
        (today_count / MAX_DAILY_PROACTIVE) * 0.45
        + min(sent_72h / 6.0, 1.0) * 0.25
        + min(reply_timeout_72h / 2.0, 1.0) * 0.20
        + min(skipped_24h / 4.0, 1.0) * 0.10,
    )
    return {
        "score": round(score, 3),
        "threshold": PROACTIVE_FATIGUE_BLOCK_THRESHOLD,
        "block": score >= PROACTIVE_FATIGUE_BLOCK_THRESHOLD,
        "components": components,
    }


async def get_proactive_rhythm_adjustment(
    agent_id: str,
    user_id: str,
    *,
    workspace_id: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Learn a small send-probability adjustment from recent user rhythm.

    This is intentionally conservative: with little evidence it returns 1.0.
    Same-local-hour reply events are positive signals; reply timeouts and recent
    fatigue skips are negative signals.
    """
    now_ts = (now or datetime.now(UTC)).astimezone(UTC)
    local_hour = now_ts.astimezone().hour
    try:
        from app.services.schedule_domain.time_service import _TZ
        local_hour = now_ts.astimezone(_TZ).hour
    except Exception:
        pass

    workspace_clause = "AND workspace_id = $5" if workspace_id else ""
    params = [
        agent_id,
        user_id,
        (now_ts - timedelta(days=30)).replace(tzinfo=None).isoformat(),
        local_hour,
    ]
    if workspace_id:
        params.append(workspace_id)

    sent_same_hour = 0
    replied_same_hour = 0
    timeout_same_hour = 0
    skipped_same_hour = 0
    try:
        rows = await db.query_raw(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE event_type = 'message_sent')::int AS sent_same_hour,
                COUNT(*) FILTER (WHERE event_type = 'user_replied')::int AS replied_same_hour,
                COUNT(*) FILTER (WHERE event_type = 'reply_timeout')::int AS timeout_same_hour,
                COUNT(*) FILTER (
                    WHERE event_type = 'send_skipped'
                      AND payload->>'reason' = 'fatigue_score'
                )::int AS skipped_same_hour
            FROM proactive_event_logs
            WHERE agent_id = $1
              AND user_id = $2
              AND created_at >= $3::timestamp
              AND EXTRACT(HOUR FROM created_at AT TIME ZONE '{_RHYTHM_TZ}')::int = $4
              {workspace_clause}
            """,
            *params,
        )
        row = rows[0] if rows else {}
        sent_same_hour = int(row.get("sent_same_hour") or 0)
        replied_same_hour = int(row.get("replied_same_hour") or 0)
        timeout_same_hour = int(row.get("timeout_same_hour") or 0)
        skipped_same_hour = int(row.get("skipped_same_hour") or 0)
    except Exception as e:
        logger.warning(f"[PROACTIVE-RHYTHM] DB read failed: {e}")

    signal_count = sent_same_hour + replied_same_hour + timeout_same_hour + skipped_same_hour
    if signal_count < 3:
        multiplier = 1.0
    else:
        positive = min(replied_same_hour / 3.0, 1.0) * 0.25
        negative = min((timeout_same_hour + skipped_same_hour) / 3.0, 1.0) * 0.35
        no_reply_penalty = 0.15 if sent_same_hour >= 3 and replied_same_hour == 0 else 0.0
        multiplier = max(0.55, min(1.25, 1.0 + positive - negative - no_reply_penalty))

    return {
        "multiplier": round(multiplier, 3),
        "local_hour": local_hour,
        "components": {
            "sent_same_hour": sent_same_hour,
            "replied_same_hour": replied_same_hour,
            "timeout_same_hour": timeout_same_hour,
            "skipped_same_hour": skipped_same_hour,
        },
    }


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
