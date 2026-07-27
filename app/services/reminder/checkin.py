"""Reminder/check-in service used by the public reminders API.

返回 user 的 timetrigger 中 actionType=reminder 的所有条目, 按 status 过滤:
- active   : isActive=true (未来计划要响的, 含 retry pending)
- fired    : isActive=false AND lastFired IS NOT NULL (已响过)
- cancelled: isActive=false AND lastFired IS NULL (用户取消 / archive 时被
             deactivate, 但没真的响过)
- dlq      : 从 Redis ZSET reminder:dlq 读, 是 emit 失败 retry 耗尽的死信

支持 limit + offset 分页, 默认 limit=50 (跟 memories 对齐).
status=open 给打卡页用: active trigger + 已响过但用户尚未完成/删除的一次性提醒.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field
from prisma import Json

from app.db import db
from app.redis_client import get_redis
from app.services.reminder.scheduling import (
    REMINDER_ACTION_TYPE,
    RecurrenceKind,
    archive_reminder_memory,
    create_user_reminder,
    notify_reminder_changed,
)
from app.services.proactive.triggers import _DLQ_KEY


class ReminderItem(BaseModel):
    """单条提醒条目 (active / fired / cancelled 共用 shape)."""
    id: str  # trigger.id
    memory_id: str | None
    summary: str
    trigger_time: str  # ISO
    last_fired: str | None  # ISO
    completed_at: str | None = None
    recurrence: str  # once | daily | weekly | monthly | yearly
    status: Literal["active", "fired", "cancelled"]
    retry_count: int  # 0 if never retried
    pinned: bool = False
    habit_weekdays: list[int] = Field(default_factory=list)
    completed_dates: list[str] = Field(default_factory=list)
    sent_to_ai: bool = False
    agent_id: str
    created_at: str  # ISO


class ReminderCreate(BaseModel):
    agent_id: str
    workspace_id: str | None = None
    summary: str
    trigger_time: str
    recurrence: RecurrenceKind = "once"
    habit_weekdays: list[int] | None = None
    sent_to_ai: bool = False
    conversation_id: str | None = None


class ReminderUpdate(BaseModel):
    summary: str | None = None
    trigger_time: str | None = None
    recurrence: RecurrenceKind | None = None
    habit_weekdays: list[int] | None = None
    pinned: bool | None = None
    sent_to_ai: bool | None = None
    conversation_id: str | None = None


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


def _is_open_checkin_item(trigger) -> bool:
    data = trigger.actionData or {}
    if data.get("deleted_at"):
        return False
    if trigger.isActive:
        return True
    recurrence = str(data.get("recurrence") or "once")
    return recurrence == "once" and trigger.lastFired is not None


def _is_deleted_checkin_item(trigger) -> bool:
    data = trigger.actionData or {}
    return bool(data.get("deleted_at"))


def _is_completed_once_checkin_item(trigger) -> bool:
    data = trigger.actionData or {}
    recurrence = str(data.get("recurrence") or "once")
    return recurrence == "once" and bool(data.get("completed_at"))


def _to_item(trigger) -> ReminderItem:
    data = trigger.actionData or {}
    habit_weekdays = _normalize_weekdays(data.get("habit_weekdays"), allow_empty=True)
    completed_dates = [
        str(item)
        for item in (data.get("completed_dates") or [])
        if isinstance(item, str)
    ]
    return ReminderItem(
        id=trigger.id,
        memory_id=data.get("memory_id") or None,
        summary=str(data.get("summary") or "")[:200],
        trigger_time=trigger.triggerTime.isoformat(),
        last_fired=trigger.lastFired.isoformat() if trigger.lastFired else None,
        completed_at=data.get("completed_at") or None,
        recurrence=str(data.get("recurrence") or "once"),
        status=_classify_status(trigger),
        retry_count=int(data.get("retry_count") or 0),
        pinned=bool(data.get("pinned") or False),
        habit_weekdays=habit_weekdays,
        completed_dates=completed_dates,
        sent_to_ai=bool(data.get("sent_to_ai") or False),
        agent_id=trigger.aiAgentId,
        created_at=trigger.createdAt.isoformat(),
    )


def _parse_datetime(value: str, field: str) -> datetime:
    raw = (value or "").strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _normalize_weekdays(value, *, allow_empty: bool = False) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        if allow_empty:
            return []
        raise HTTPException(status_code=400, detail="habit_weekdays must be a list")
    weekdays: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 1 or item > 7:
            if allow_empty:
                continue
            raise HTTPException(
                status_code=400,
                detail="habit_weekdays must contain integers from 1 to 7",
            )
        if item not in weekdays:
            weekdays.append(item)
    weekdays.sort()
    if not weekdays and not allow_empty:
        raise HTTPException(status_code=400, detail="habit_weekdays cannot be empty")
    return weekdays


def _normalize_date_key(value: str | None) -> str:
    if not value:
        return datetime.now(UTC).date().isoformat()
    raw = value.strip()
    try:
        return datetime.fromisoformat(raw).date().isoformat()
    except ValueError:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid occurrence_date") from exc


async def _ensure_agent_scope(
    *,
    agent_id: str,
    workspace_id: str | None,
    user: dict,
):
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") == "archived":
        raise HTTPException(status_code=404, detail="Agent not found")
    if user.get("role") != "admin" and agent.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your agent")
    if workspace_id:
        workspace = await db.chatworkspace.find_unique(where={"id": workspace_id})
        if (
            not workspace
            or workspace.userId != agent.userId
            or workspace.agentId != agent_id
        ):
            raise HTTPException(status_code=400, detail="Workspace does not match agent")
    return agent


async def _require_trigger_owner(trigger_id: str, user: dict):
    trigger = await db.timetrigger.find_unique(where={"id": trigger_id})
    if not trigger or trigger.actionType != REMINDER_ACTION_TYPE:
        raise HTTPException(status_code=404, detail="Reminder not found")
    if user.get("role") != "admin" and trigger.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your reminder")
    return trigger


async def _fetch_trigger_for_memory(memory_id: str, user_id: str, agent_id: str):
    rows = await db.timetrigger.find_many(
        where={
            "userId": user_id,
            "aiAgentId": agent_id,
            "actionType": REMINDER_ACTION_TYPE,
            "isActive": True,
        },
        order={"createdAt": "desc"},
        take=20,
    )
    return next(
        (row for row in rows if (row.actionData or {}).get("memory_id") == memory_id),
        None,
    )


async def _patch_reminder_memory(
    memory_id: str | None,
    *,
    summary: str | None = None,
    trigger_time: datetime | None = None,
    recurrence: RecurrenceKind | None = None,
) -> None:
    if not memory_id:
        return
    from app.services.memory.storage import repo as memory_repo

    data = {}
    if summary is not None:
        data["content"] = summary
    if trigger_time is not None:
        data["occurTime"] = trigger_time
    if recurrence is not None:
        data["recurrence"] = recurrence
    if data:
        await memory_repo.update(memory_id, source="user", **data)


async def list_reminders_for_user(
    user_id: str,
    agent_id: str | None = None,
    status: Literal["active", "fired", "cancelled", "all", "open"] = "all",
    limit: int = 50,
    offset: int = 0,
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
    # status == "all" / "open": 不加 isActive/lastFired 过滤.
    # open 需要看 actionData.completed_at/deleted_at，Prisma JSON 条件表达有限，
    # 这里按 limit+offset 之前的小量列表在 Python 侧过滤。
    if status == "open":
        rows = []
        total = 0
        skip = 0
        batch_size = 500
        while True:
            batch = await db.timetrigger.find_many(
                where=where,
                order={"triggerTime": "desc"},
                take=batch_size,
                skip=skip,
            )
            if not batch:
                break
            for row in batch:
                if not _is_open_checkin_item(row):
                    continue
                if offset <= total < offset + limit:
                    rows.append(row)
                total += 1
            if len(batch) < batch_size:
                break
            skip += batch_size
    else:
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


async def create_reminder_for_user(
    data: ReminderCreate,
    user: dict,
):
    agent = await _ensure_agent_scope(
        agent_id=data.agent_id,
        workspace_id=data.workspace_id,
        user=user,
    )
    summary = data.summary.strip()
    if not summary:
        raise HTTPException(status_code=400, detail="summary is required")
    trigger_time = _parse_datetime(data.trigger_time, "trigger_time")
    habit_weekdays = _normalize_weekdays(
        data.habit_weekdays,
        allow_empty=data.habit_weekdays is None,
    )
    if habit_weekdays and data.recurrence != "weekly":
        raise HTTPException(status_code=400, detail="habit_weekdays requires weekly recurrence")
    if trigger_time <= datetime.now(UTC):
        raise HTTPException(status_code=400, detail="提醒时间必须在未来")

    memory_id = await create_user_reminder(
        user_id=agent.userId,
        agent_id=data.agent_id,
        workspace_id=data.workspace_id,
        summary=summary[:200],
        occur_time=trigger_time,
        statement_time=datetime.now(UTC),
        recurrence=data.recurrence,
    )
    if not memory_id:
        raise HTTPException(status_code=500, detail="Reminder create failed")
    trigger = await _fetch_trigger_for_memory(memory_id, agent.userId, data.agent_id)
    if not trigger:
        raise HTTPException(status_code=500, detail="Reminder trigger missing")
    action_data = dict(trigger.actionData or {})
    if habit_weekdays:
        action_data["habit_weekdays"] = habit_weekdays
    action_data["sent_to_ai"] = bool(data.sent_to_ai)
    action_data["conversation_id"] = data.conversation_id
    trigger = await db.timetrigger.update(
        where={"id": trigger.id},
        data={"actionData": Json(action_data)},
    )
    await notify_reminder_changed(data.conversation_id, kind="created", trigger_id=trigger.id)
    return _to_item(trigger)


async def update_reminder_for_user(
    trigger_id: str,
    data: ReminderUpdate,
    user: dict,
):
    trigger = await _require_trigger_owner(trigger_id, user)
    if _is_deleted_checkin_item(trigger):
        raise HTTPException(status_code=409, detail="提醒已删除")
    immutable_update = (
        data.summary is not None
        or data.trigger_time is not None
        or data.recurrence is not None
        or data.habit_weekdays is not None
        or data.sent_to_ai is not None
    )
    if _is_completed_once_checkin_item(trigger) and immutable_update:
        raise HTTPException(status_code=409, detail="已完成提醒不允许编辑")
    action_data = dict(trigger.actionData or {})
    update_data: dict = {}

    if data.summary is not None:
        summary = data.summary.strip()
        if not summary:
            raise HTTPException(status_code=400, detail="summary cannot be empty")
        action_data["summary"] = summary[:200]
    if data.recurrence is not None:
        action_data["recurrence"] = data.recurrence
        if data.recurrence != "weekly":
            action_data.pop("habit_weekdays", None)
    if data.habit_weekdays is not None:
        habit_weekdays = _normalize_weekdays(data.habit_weekdays)
        recurrence = data.recurrence or action_data.get("recurrence") or "once"
        if recurrence != "weekly":
            raise HTTPException(status_code=400, detail="habit_weekdays requires weekly recurrence")
        action_data["habit_weekdays"] = habit_weekdays
    if data.pinned is not None:
        action_data["pinned"] = data.pinned
    if data.sent_to_ai is not None:
        action_data["sent_to_ai"] = bool(data.sent_to_ai)
    if data.conversation_id is not None:
        action_data["conversation_id"] = data.conversation_id
    if data.trigger_time is not None:
        trigger_time = _parse_datetime(data.trigger_time, "trigger_time")
        if trigger_time <= datetime.now(UTC):
            raise HTTPException(status_code=400, detail="提醒时间必须在未来")
        update_data["triggerTime"] = trigger_time

    update_data["actionData"] = Json(action_data)
    updated = await db.timetrigger.update(where={"id": trigger_id}, data=update_data)
    await _patch_reminder_memory(
        action_data.get("memory_id"),
        summary=action_data.get("summary") if data.summary is not None else None,
        trigger_time=update_data.get("triggerTime"),
        recurrence=data.recurrence,
    )
    await notify_reminder_changed(data.conversation_id, kind="rescheduled", trigger_id=trigger_id)
    return _to_item(updated)


async def complete_reminder_for_user(
    trigger_id: str,
    *,
    conversation_id: str | None = None,
    occurrence_date: str | None = None,
    user: dict,
):
    trigger = await _require_trigger_owner(trigger_id, user)
    if _is_deleted_checkin_item(trigger):
        raise HTTPException(status_code=409, detail="提醒已删除")
    data = dict(trigger.actionData or {})
    recurrence = str(data.get("recurrence") or "once")
    if recurrence != "once":
        date_key = _normalize_date_key(occurrence_date)
        completed_dates = [
            str(item)
            for item in (data.get("completed_dates") or [])
            if isinstance(item, str)
        ]
        if date_key not in completed_dates:
            completed_dates.append(date_key)
            completed_dates.sort()
        data["completed_dates"] = completed_dates
        updated = await db.timetrigger.update(
            where={"id": trigger_id},
            data={"actionData": Json(data)},
        )
        await notify_reminder_changed(conversation_id, kind="archived", trigger_id=trigger_id)
        return _to_item(updated)

    if data.get("completed_at"):
        return _to_item(trigger)

    data["completed_at"] = datetime.now(UTC).isoformat()
    updated = await db.timetrigger.update(
        where={"id": trigger_id},
        data={
            "isActive": False,
            "lastFired": datetime.now(UTC),
            "actionData": Json(data),
        },
    )
    memory_id = data.get("memory_id")
    if memory_id:
        await archive_reminder_memory(
            memory_id=memory_id,
            side=data.get("memory_side") or "user",
            reason="completed_from_checkin",
        )
    await notify_reminder_changed(conversation_id, kind="archived", trigger_id=trigger_id)
    return _to_item(updated)


async def delete_reminder_for_user(
    trigger_id: str,
    *,
    conversation_id: str | None = None,
    user: dict,
):
    trigger = await _require_trigger_owner(trigger_id, user)
    data = dict(trigger.actionData or {})
    if data.get("deleted_at"):
        return None
    data["deleted_at"] = datetime.now(UTC).isoformat()
    await db.timetrigger.update(
        where={"id": trigger_id},
        data={"isActive": False, "actionData": Json(data)},
    )
    memory_id = data.get("memory_id")
    if memory_id:
        await archive_reminder_memory(
            memory_id=memory_id,
            side=data.get("memory_side") or "user",
            reason="deleted_from_checkin",
        )
    await notify_reminder_changed(conversation_id, kind="cancelled", trigger_id=trigger_id)
    return None


async def list_reminder_dlq_for_user(
    user_id: str,
    limit: int = 100,
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
