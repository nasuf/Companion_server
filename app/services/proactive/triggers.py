"""时间触发引擎。

PRD §9.5: 在合适的时间触发AI主动行为，遵循作息规则。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.db import db
from app.redis_client import get_redis
from app.services.proactive.sender import send_manual_or_triggered_proactive
from app.services.proactive.special_dates import (
    Occasion,
    send_special_date_proactive,
)
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status
from app.services.schedule_domain.time_service import _TZ
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

# 每日主动问候上限
MAX_DAILY_TRIGGERS = 3
# 两次主动间隔最小时间（秒）
MIN_TRIGGER_INTERVAL = 7200  # 2小时
# 交流状态判定窗口（秒）
CHAT_ACTIVE_WINDOW = 1800  # 30分钟


async def scan_triggers() -> None:
    """每分钟扫描到期触发器，检查规则后执行。"""
    now = datetime.now(_TZ)
    window_start = now - timedelta(minutes=1)
    window_end = now + timedelta(minutes=1)

    triggers = await db.timetrigger.find_many(
        where={
            "isActive": True,
            "triggerTime": {"gte": window_start, "lt": window_end},
        },
    )

    for trigger in triggers:
        try:
            await _execute_trigger(trigger, now)
        except Exception as e:
            logger.warning(f"Trigger {trigger.id} execution failed: {e}")


async def _defer_special_date_trigger(
    trigger,
    schedule: list[dict] | None,
    now: datetime,
    reason: str,
) -> None:
    """spec §10/§5.2: special_date 触发被互斥阻塞时, 顺延到 "下一个空闲状态" 开始时刻.

    顺延算法:
    1. 优先找作息表中 status=idle 且 start > now 的最早时段
    2. 找不到 (今日无后续 idle) → 顺延到 +30 min, 让下一次 scan 重新尝试
    3. 仍跨过当日 22:00 → 取消本次 trigger (避免越过 spec §1.2 主动时段)
    """
    next_time: datetime | None = None
    if schedule:
        cur_hm = now.strftime("%H:%M")
        for slot in schedule:
            start_hm = str(slot.get("start") or "")
            slot_type = slot.get("type") or "leisure"
            if start_hm <= cur_hm:
                continue
            if slot_type in ("leisure",) or slot.get("status") == "idle":
                h, m = start_hm.split(":")
                cand = datetime(now.year, now.month, now.day, int(h), int(m), tzinfo=now.tzinfo)
                if cand > now:
                    next_time = cand
                    break
    if next_time is None:
        next_time = now + timedelta(minutes=30)

    # spec §1.2: 22:00-8:00 不发送, 越过则取消
    local_h = next_time.astimezone(_TZ).hour
    if local_h >= 22 or local_h < 8:
        logger.info(
            f"Trigger {trigger.id} (special_date) cancelled after defer: "
            f"next_time={next_time} fell outside active hours"
        )
        await db.timetrigger.update(
            where={"id": trigger.id},
            data={"isActive": False},
        )
        return

    await db.timetrigger.update(
        where={"id": trigger.id},
        data={"triggerTime": next_time},
    )
    logger.info(
        f"Trigger {trigger.id} (special_date) deferred: "
        f"reason={reason} new_time={next_time}"
    )


async def _execute_trigger(trigger, now: datetime) -> None:
    """执行单个触发器，检查所有规则。"""
    agent_id = trigger.aiAgentId
    user_id = trigger.userId
    is_special_date = trigger.actionType == "special_date"

    # 规则1: 非交流中（最后消息间隔 > 30min）
    if await _is_in_chat(agent_id, user_id):
        if is_special_date:
            schedule = await get_cached_schedule(agent_id)
            await _defer_special_date_trigger(trigger, schedule, now, reason="user_in_chat")
            return
        logger.debug(f"Trigger {trigger.id} skipped: user in active chat")
        return

    # 规则2: AI非睡眠状态
    schedule = await get_cached_schedule(agent_id)
    if schedule:
        status = get_current_status(schedule, now)
        if status.get("status") == "sleep":
            if is_special_date:
                await _defer_special_date_trigger(trigger, schedule, now, reason="ai_sleep")
                return
            logger.debug(f"Trigger {trigger.id} skipped: AI is sleeping")
            return

    # 规则3: 每日次数限制
    redis = await get_redis()
    daily_key = f"trigger_count:{agent_id}:{user_id}:{now.strftime('%Y%m%d')}"
    count = int(await redis.get(daily_key) or 0)
    if count >= MAX_DAILY_TRIGGERS:
        logger.debug(f"Trigger {trigger.id} skipped: daily limit reached ({count})")
        return

    # 规则4: 间隔限制
    last_key = f"trigger_last:{agent_id}:{user_id}"
    last_fired_str = await redis.get(last_key)
    if last_fired_str:
        last_ts = float(last_fired_str)
        if now.timestamp() - last_ts < MIN_TRIGGER_INTERVAL:
            logger.debug(f"Trigger {trigger.id} skipped: interval too short")
            return

    # 执行触发动作
    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    if not workspace_id:
        logger.debug(f"Trigger {trigger.id} skipped: workspace not found")
        return

    # spec §10 特殊日期走独立发送路径 (带 skip_post_process + 合并/单日期 prompt)
    if trigger.actionType == "special_date":
        raw_occasions = (trigger.actionData or {}).get("occasions") or []
        occasions = [
            Occasion(
                type=o.get("type", "holiday"),
                name=o.get("name", ""),
                owner=o.get("owner", "user"),
            )
            for o in raw_occasions
        ]
        sent = await send_special_date_proactive(
            agent_id=agent_id,
            user_id=user_id,
            occasions=occasions,
            now=now,
        )
        if not sent:
            logger.debug(f"Trigger {trigger.id}: special_date send failed/skipped")
            return
        message = "[special_date sent]"
    else:
        result = await send_manual_or_triggered_proactive(
            workspace_id=workspace_id,
            trigger_type=f"trigger:{trigger.actionType}",
            now=now,
        )
        message = result.get("message")
        if not result.get("ok") or not message:
            logger.debug(f"Trigger {trigger.id}: proactive message generation returned empty")
            return

    # 记录日志
    await redis.incr(daily_key)
    await redis.expire(daily_key, 86400)
    await redis.set(last_key, str(now.timestamp()), ex=MIN_TRIGGER_INTERVAL)

    # 更新触发器
    update_data: dict = {"lastFired": now}
    if trigger.repeatRule is None:
        update_data["isActive"] = False  # 一次性触发器执行后停用
    await db.timetrigger.update(where={"id": trigger.id}, data=update_data)

    logger.info(f"Trigger {trigger.id} fired: {trigger.actionType} for agent {agent_id}")


async def _is_in_chat(agent_id: str, user_id: str) -> bool:
    """判断用户是否在交流状态（最后消息间隔 < 30min）。"""
    try:
        last_msg = await db.message.find_first(
            where={
                "conversation": {
                    "agentId": agent_id,
                    "userId": user_id,
                }
            },
            order={"createdAt": "desc"},
        )
        if last_msg and last_msg.createdAt:
            msg_time = last_msg.createdAt
            if msg_time.tzinfo is None:
                msg_time = msg_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - msg_time).total_seconds() < CHAT_ACTIVE_WINDOW
    except Exception as e:
        logger.warning(f"Chat active check failed: {e}")
    return False


async def seed_default_triggers(ai_agent_id: str, user_id: str) -> None:
    """Agent创建时种入默认触发器：早安/午安/晚安。"""
    now = datetime.now(_TZ)
    today = now.date()

    defaults = [
        ("08:00", "greeting", "morning"),
        ("12:30", "greeting", "noon"),
        ("22:00", "greeting", "night"),
    ]

    for time_str, action_type, period in defaults:
        h, m = time_str.split(":")
        trigger_time = datetime(today.year, today.month, today.day, int(h), int(m), tzinfo=_TZ)
        await db.timetrigger.create(data={
            "agent": {"connect": {"id": ai_agent_id}},
            "user": {"connect": {"id": user_id}},
            "triggerTime": trigger_time,
            "repeatRule": f"0 {h.lstrip('0') or '0'} * * *",  # 每天
            "actionType": action_type,
            "actionData": {"period": period},
        })

# 已移除 (Part 5 §4.3 统一扫描代替):
# - create_reminder_trigger    → 改走 memory taxonomy 提醒子类 + scan_special_dates_today
# - scan_birthday_memories     → 整合进 scan_special_dates_today
# - create_holiday_triggers    → 整合进 scan_special_dates_today
