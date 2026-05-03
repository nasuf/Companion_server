"""Reminder 列表 endpoint — inspector "提醒" tab 用.

返回 user 的 timetrigger 中 actionType=reminder 的所有条目, 按 status 过滤:
- active   : isActive=true (未来计划要响的, 含 retry pending)
- fired    : isActive=false AND lastFired IS NOT NULL (已响过)
- cancelled: isActive=false AND lastFired IS NULL (用户取消 / archive 时被
             deactivate, 但没真的响过)
- dlq      : 从 Redis ZSET reminder:dlq 读, 是 emit 失败 retry 耗尽的死信

支持 limit + offset 分页, 默认 limit=50 (跟 memories 对齐).
"""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.ownership import require_user_self
from app.db import db
from app.redis_client import get_redis
from app.services.reminder.scheduling import REMINDER_ACTION_TYPE
from app.services.proactive.triggers import _DLQ_KEY

router = APIRouter(prefix="/reminders", tags=["reminders"])


class ReminderItem(BaseModel):
    """单条提醒条目 (active / fired / cancelled 共用 shape)."""
    id: str  # trigger.id
    memory_id: str | None
    summary: str
    trigger_time: str  # ISO
    last_fired: str | None  # ISO
    recurrence: str  # once | daily | weekly | monthly | yearly
    status: Literal["active", "fired", "cancelled"]
    retry_count: int  # 0 if never retried
    agent_id: str
    created_at: str  # ISO


class DlqItem(BaseModel):
    """DLQ 死信条目 (单独 shape, 来自 Redis ZSET 而非 DB)."""
    trigger_id: str
    memory_id: str
    summary: str
    recurrence: str
    error: str
    kind: str  # exhausted | reactivate_failed | periodic_lost_one
    attempt: int
    failed_at: str  # ISO


class RemindersResponse(BaseModel):
    """返 items + 总数 (分页用) + dlq 条数 (单独显示在 tab title 上)."""
    items: list[ReminderItem]
    total: int
    dlq_count: int


def _classify_status(trigger) -> Literal["active", "fired", "cancelled"]:
    if trigger.isActive:
        return "active"
    if trigger.lastFired is not None:
        return "fired"
    return "cancelled"


def _to_item(trigger) -> ReminderItem:
    data = trigger.actionData or {}
    return ReminderItem(
        id=trigger.id,
        memory_id=data.get("memory_id") or None,
        summary=str(data.get("summary") or "")[:200],
        trigger_time=trigger.triggerTime.isoformat(),
        last_fired=trigger.lastFired.isoformat() if trigger.lastFired else None,
        recurrence=str(data.get("recurrence") or "once"),
        status=_classify_status(trigger),
        retry_count=int(data.get("retry_count") or 0),
        agent_id=trigger.aiAgentId,
        created_at=trigger.createdAt.isoformat(),
    )


@router.get("", response_model=RemindersResponse)
async def list_reminders(
    user_id: str,
    agent_id: str | None = None,
    status: Literal["active", "fired", "cancelled", "all"] = "all",
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    _user=Depends(require_user_self),
):
    """user 的提醒列表. status='all' 默认返三种状态混合, 按 triggerTime 倒序
    (active 在前因为时间最新, fired/cancelled 按 lastFired/createdAt 排).

    DB 侧: time_triggers WHERE user_id=? AND actionType='reminder' [+ agent_id]
    [+ status filter]. 索引 (user_id, action_type, is_active) 命中.
    DLQ 侧: 仅返 dlq_count 让 tab 上显示徽标; 详细列表走 /reminders/dlq.
    """
    where: dict = {
        "userId": user_id,
        "actionType": REMINDER_ACTION_TYPE,
    }
    if agent_id:
        where["aiAgentId"] = agent_id
    if status == "active":
        where["isActive"] = True
    elif status == "fired":
        where["isActive"] = False
        where["lastFired"] = {"not": None}
    elif status == "cancelled":
        where["isActive"] = False
        where["lastFired"] = None
    # status == "all": 不加 isActive/lastFired 过滤

    total = await db.timetrigger.count(where=where)
    rows = await db.timetrigger.find_many(
        where=where,
        order={"triggerTime": "desc"},
        take=limit,
        skip=offset,
    )

    # DLQ count — Redis ZSET cardinality. 失败不冒泡 (DLQ 是观察性数据).
    dlq_count = 0
    try:
        redis = await get_redis()
        dlq_count = int(await redis.zcard(_DLQ_KEY) or 0)
    except Exception:
        pass

    return RemindersResponse(
        items=[_to_item(r) for r in rows],
        total=total,
        dlq_count=dlq_count,
    )


@router.get("/dlq", response_model=list[DlqItem])
async def list_reminder_dlq(
    user_id: str,
    limit: int = Query(default=100, le=500),
    _user=Depends(require_user_self),
):
    """DLQ 死信列表 (跨用户共享 Redis ZSET, 但前端按 user_id 过滤展示).

    DLQ 当前不按 user 分桶 (一个 ZSET 收所有失败), 这里读全量然后 Python 侧
    过滤. 量级 <=1000 (cap), 性能不是问题. 长远如果需要按 user 分桶, 改 ZSET
    key 为 reminder:dlq:{user_id} 即可.
    """
    items: list[DlqItem] = []
    try:
        redis = await get_redis()
        # ZREVRANGE 倒序 (最新 failure 在前)
        raw_entries = await redis.zrevrange(_DLQ_KEY, 0, limit - 1)
    except Exception:
        return items

    # 拉所有 user 的 trigger 一次, 用来按 trigger_id 反查 user_id 过滤
    # (Python 侧, DLQ 量小可接受)
    candidate_ids = []
    parsed: list[dict] = []
    for raw in raw_entries:
        try:
            entry = json.loads(raw if isinstance(raw, str) else raw.decode())
        except Exception:
            continue
        candidate_ids.append(entry.get("trigger_id", ""))
        parsed.append(entry)

    user_trigger_ids: set[str] = set()
    if candidate_ids:
        try:
            triggers = await db.timetrigger.find_many(
                where={"id": {"in": candidate_ids}, "userId": user_id},
            )
            user_trigger_ids = {t.id for t in triggers}
        except Exception:
            pass

    for entry in parsed:
        tid = entry.get("trigger_id", "")
        if tid and tid not in user_trigger_ids:
            continue
        items.append(DlqItem(
            trigger_id=tid,
            memory_id=str(entry.get("memory_id") or ""),
            summary=str(entry.get("summary") or "")[:200],
            recurrence=str(entry.get("recurrence") or "once"),
            error=str(entry.get("error") or "")[:200],
            kind=str(entry.get("kind") or "unknown"),
            attempt=int(entry.get("attempt") or 0),
            failed_at=str(entry.get("failed_at") or ""),
        ))
    return items
