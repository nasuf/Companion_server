"""时间触发引擎。

PRD §9.5: 在合适的时间触发AI主动行为，遵循作息规则。
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from app.db import db
from app.redis_client import get_redis
from app.services.proactive.sender import send_manual_or_triggered_proactive
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


async def _execute_trigger(trigger, now: datetime) -> None:
    """执行单个触发器，检查所有规则。"""
    agent_id = trigger.aiAgentId
    user_id = trigger.userId

    # 规则1: 非交流中（最后消息间隔 > 30min）
    if await _is_in_chat(agent_id, user_id):
        logger.debug(f"Trigger {trigger.id} skipped: user in active chat")
        return

    # 规则2: AI非睡眠状态
    schedule = await get_cached_schedule(agent_id)
    if schedule:
        status = get_current_status(schedule, now)
        if status.get("status") == "sleep":
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


async def create_reminder_trigger(
    ai_agent_id: str,
    user_id: str,
    trigger_time: datetime,
    content: str,
) -> str:
    """创建一次性提醒触发器。

    PRD §9.5.3: 用户说"明天提醒我开会" → 创建一次性触发器。
    """
    trigger = await db.timetrigger.create(data={
        "agent": {"connect": {"id": ai_agent_id}},
        "user": {"connect": {"id": user_id}},
        "triggerTime": trigger_time,
        "actionType": "reminder",
        "actionData": {"content": content},
    })
    logger.info(f"Created reminder trigger {trigger.id} for {trigger_time}")
    return trigger.id


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


async def scan_birthday_memories() -> None:
    """每周扫描用户记忆，识别含"生日"关键词的记忆，创建生日触发器。

    PRD §9.6.3: 用户自定义日期（如生日）自动生成周期性触发器。
    """
    import re
    from app.services.memory import memory_repo

    agents = await db.aiagent.find_many(where={"status": "active"})
    birthday_pattern = re.compile(r"(\d{1,2})月(\d{1,2})[日号].*生日|生日.*(\d{1,2})月(\d{1,2})[日号]")

    for agent in agents:
        try:
            memories = await memory_repo.find_many(
                source="user",
                where={"userId": agent.userId, "isArchived": False},
            )
            for mem in memories:
                text = mem.content or ""
                if "生日" not in text:
                    continue
                m = birthday_pattern.search(text)
                if not m:
                    continue
                month = int(m.group(1) or m.group(3))
                day = int(m.group(2) or m.group(4))

                # 检查是否已有该用户的生日触发器
                existing = await db.timetrigger.find_first(
                    where={
                        "aiAgentId": agent.id,
                        "userId": agent.userId,
                        "actionType": "birthday",
                    }
                )
                if existing:
                    continue

                # 创建当年的生日触发器
                now = datetime.now(_TZ)
                year = now.year
                try:
                    birthday = date(year, month, day)
                    if birthday < now.date():
                        birthday = date(year + 1, month, day)
                    trigger_time = datetime(birthday.year, birthday.month, birthday.day, 8, 0, tzinfo=_TZ)
                    await db.timetrigger.create(data={
                        "agent": {"connect": {"id": agent.id}},
                        "user": {"connect": {"id": agent.userId}},
                        "triggerTime": trigger_time,
                        "actionType": "birthday",
                        "actionData": {"month": month, "day": day},
                    })
                    logger.info(f"Birthday trigger created for agent {agent.id}: {month}月{day}日")
                except ValueError:
                    pass
        except Exception as e:
            logger.warning(f"Birthday scan failed for agent {agent.id}: {e}")


async def create_holiday_triggers(date_str: str, holiday_name: str) -> None:
    """为所有活跃agent创建节假日祝福触发器。

    PRD §9.6.3: 节日当天早上8:00触发。
    """
    d = date.fromisoformat(date_str)
    trigger_time = datetime(d.year, d.month, d.day, 8, 0, tzinfo=_TZ)

    # 查找所有活跃的 agent-user 组合
    agents = await db.aiagent.find_many(where={"status": "active"})
    for agent in agents:
        try:
            await db.timetrigger.create(data={
                "agent": {"connect": {"id": agent.id}},
                "user": {"connect": {"id": agent.userId}},
                "triggerTime": trigger_time,
                "actionType": "holiday",
                "actionData": {"holiday_name": holiday_name},
            })
        except Exception as e:
            logger.warning(f"Failed to create holiday trigger for agent {agent.id}: {e}")
