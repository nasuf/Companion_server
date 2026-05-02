"""统一提醒调度服务 — Reminder 功能的单一入口.

Round-2 review 找出的架构问题:
- intent_handlers.py 的 _direct_create_reminder 持 6 个跨模块依赖 (memory_repo / store_memory /
  generate_embedding / find_duplicate_id / _create_reminder_timetrigger / resolve_workspace_id),
  做 parse + store + dedup + update + 建 trigger 5 件事
- _cancel_active_reminders / _create_reminder_timetrigger / apply_reschedule 都做"按 user_id +
  active 找 reminder triggers"的 find_many + Python 侧 filter, 三份重复
- _archive_reminder_memory (triggers.py) vs execute_confirmed_deletion (deletion.py) 概念重叠

收口到本模块. handler / pipeline / preflight / triggers 都调本模块的 helper.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

from app.db import db

logger = logging.getLogger(__name__)

# 类型别名 — single source of truth for stringly-typed reminder fields.
# 使用 Literal 让 mypy 在 typo 时报错; 其他模块 import 这些别名而非硬编码字符串.
ReminderSide = Literal["user", "ai"]
RecurrenceKind = Literal["once", "daily", "weekly", "monthly", "yearly"]
ReminderState = Literal["completed", "cancelled", "rescheduled", "needed"]

# TimeTrigger.actionType for reminders. 散落 5+ 处的硬编码 "reminder" 字符串都该用这个常量.
REMINDER_ACTION_TYPE = "reminder"


# ═══════════════════════════════════════════════════════════════════
# Trigger lookup — 唯一入口, 替代散落 3 处的 find_many + Python filter
# ═══════════════════════════════════════════════════════════════════


async def find_active_reminder_triggers(
    *,
    user_id: str,
    agent_id: str | None = None,
    memory_id: str | None = None,
) -> list[Any]:
    """查 user (可选 agent) 的所有 active reminder timetrigger, 可按 memory_id 进一步筛.

    `agent_id` 强烈建议传入 — 多 agent 用户场景下不传会跨 agent 拉到无关 trigger,
    历史 bug 的根因 (cancel 跨 agent 误删).
    `memory_id` 传入 → 仅返该 memory 对应的 triggers (Python 侧筛, prisma 不支持
    嵌套 JSON 等值过滤).
    """
    where: dict = {
        "userId": user_id,
        "actionType": REMINDER_ACTION_TYPE,
        "isActive": True,
    }
    if agent_id:
        where["aiAgentId"] = agent_id
    try:
        rows = await db.timetrigger.find_many(where=where)
    except Exception as e:
        logger.warning(f"[REMINDER] find_active_reminder_triggers failed: {e}")
        return []

    if memory_id is None:
        return rows
    return [
        t for t in rows
        if (t.actionData or {}).get("memory_id") == memory_id
    ]


# ═══════════════════════════════════════════════════════════════════
# Trigger lifecycle — 建/续期/取消, 统一入口
# ═══════════════════════════════════════════════════════════════════


async def upsert_reminder_trigger(
    *,
    user_id: str,
    agent_id: str,
    memory_id: str,
    summary: str,
    trigger_time: datetime,
    recurrence: RecurrenceKind,
    side: ReminderSide,
) -> str | None:
    """按 (agent_id, memory_id) 幂等 upsert reminder timetrigger.

    - 已存在 active trigger for memory_id → update triggerTime + reset lastFired
      (重设语义: 用户重发同一提醒, 时间应跟最新一次)
    - 不存在 → create 新 trigger

    返回 trigger.id (成功) 或 None (失败).
    """
    from prisma import Json

    existing = await find_active_reminder_triggers(
        user_id=user_id, agent_id=agent_id, memory_id=memory_id,
    )

    if existing:
        # 重设 — update 第一个匹配的 trigger (memory_id 应该全局唯一, 本来就最多 1 个)
        target = existing[0]
        try:
            await db.timetrigger.update(
                where={"id": target.id},
                data={
                    "triggerTime": trigger_time,
                    # reset lastFired 让 idempotency 守门 (lastFired<2min) 不误拦
                    "lastFired": None,
                },
            )
            logger.info(
                f"[REMINDER] trigger UPDATED memory={memory_id[:8]} "
                f"trigger={target.id[:8]} new_time={trigger_time} recurrence={recurrence}"
            )
            return target.id
        except Exception as e:
            logger.warning(
                f"[REMINDER] trigger update failed ({e}); falling through to create"
            )

    try:
        created = await db.timetrigger.create(data={
            "aiAgentId": agent_id,
            "userId": user_id,
            "triggerTime": trigger_time,
            "actionType": REMINDER_ACTION_TYPE,
            "actionData": Json({
                "memory_id": memory_id,
                "summary": summary[:200],
                "recurrence": recurrence,
                "memory_side": side,
            }),
        })
        logger.info(
            f"[REMINDER] trigger CREATED memory={memory_id[:8]} "
            f"trigger={created.id[:8]} agent={agent_id[:8]} at={trigger_time} "
            f"recurrence={recurrence}"
        )
        return created.id
    except Exception as e:
        logger.warning(f"[REMINDER] trigger create failed: {e}")
        return None


async def renew_periodic_trigger(
    *,
    user_id: str,
    agent_id: str,
    next_trigger_time: datetime,
    action_data: dict,
) -> str | None:
    """周期性 reminder 触发后建下一周期 trigger. action_data 复用旧 trigger 的, 仅
    triggerTime 变. 显式 lastFired=None (新 row 应该是 fresh).
    """
    from prisma import Json
    try:
        created = await db.timetrigger.create(data={
            "aiAgentId": agent_id,
            "userId": user_id,
            "triggerTime": next_trigger_time,
            "actionType": REMINDER_ACTION_TYPE,
            "actionData": Json(dict(action_data)),
            "lastFired": None,  # fresh row, lastFired 守门 (2min 窗口) 不应误拦
        })
        return created.id
    except Exception as e:
        logger.warning(f"[REMINDER] renewal create failed: {e}")
        return None


async def deactivate_reminder_triggers(
    *,
    user_id: str,
    agent_id: str | None = None,
    memory_id: str | None = None,
) -> int:
    """deactivate 匹配的 reminder triggers, 返回 deactivate 的条数.

    - agent_id=None: 跨 agent 全部 (危险, 仅 admin 场景)
    - memory_id=None: 该 (user, agent) 的所有 active reminder
    - 都传: 仅该 memory 对应的 trigger

    用户路径必须传 agent_id (cancel 不该跨 agent 误删, 这是 round-2 review 的 bug #2).

    实现选择: memory_id 路径走 find + 按 JSON filter (Prisma 不支持 nested JSON
    等值过滤); 其他路径走 update_many 单 SQL (避免 N+1 update).
    """
    if memory_id is not None:
        rows = await find_active_reminder_triggers(
            user_id=user_id, agent_id=agent_id, memory_id=memory_id,
        )
        if not rows:
            return 0
        try:
            result = await db.timetrigger.update_many(
                where={"id": {"in": [t.id for t in rows]}},
                data={"isActive": False},
            )
            return int(result) if result is not None else len(rows)
        except Exception as e:
            logger.warning(f"[REMINDER] deactivate (memory_id path) failed: {e}")
            return 0

    where: dict = {
        "userId": user_id,
        "actionType": REMINDER_ACTION_TYPE,
        "isActive": True,
    }
    if agent_id:
        where["aiAgentId"] = agent_id
    try:
        result = await db.timetrigger.update_many(
            where=where, data={"isActive": False},
        )
        return int(result) if result is not None else 0
    except Exception as e:
        logger.warning(f"[REMINDER] deactivate failed: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════
# Memory lifecycle — 软删 reminder memory (cancelled/completed)
# ═══════════════════════════════════════════════════════════════════


async def archive_reminder_memory(
    *, memory_id: str, side: ReminderSide, reason: str,
) -> bool:
    """软删 reminder memory (isArchived=True) + 写 changelog. 返 True/False
    供测试断言; 生产 caller 通常忽略返回值 (失败已 logged).

    替代之前 triggers.py:_archive_reminder_memory + deletion.py:execute_confirmed_deletion
    的部分重复. 用于 reminder pre-check 判 completed/cancelled 时归档原 memory.
    """
    from app.services.memory.storage import repo as memory_repo
    from app.services.memory.storage.persistence import log_memory_changelog

    try:
        record = await memory_repo.find_unique(memory_id)
    except Exception as e:
        logger.warning(f"[REMINDER] archive: find_unique {memory_id[:8]} failed: {e}")
        return False
    if not record:
        return False

    try:
        await memory_repo.update(memory_id, source=side, isArchived=True)
    except Exception as e:
        logger.warning(f"[REMINDER] archive: update {memory_id[:8]} failed: {e}")
        return False

    try:
        await log_memory_changelog(
            record.userId, memory_id, "reminder_archived",
            new_value=reason,
            workspace_id=record.workspaceId,
        )
    except Exception:
        pass  # changelog 失败不影响主流程
    return True


# ═══════════════════════════════════════════════════════════════════
# 端到端落库 — 给定 (user, agent, summary, occur_time) 一步建好 memory + trigger
# ═══════════════════════════════════════════════════════════════════


async def create_user_reminder(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    summary: str,
    occur_time: datetime,
    statement_time: datetime,
    recurrence: RecurrenceKind = "once",
) -> str | None:
    """端到端建 1 条 user-side reminder: store_memory + dedup + upsert trigger.
    返回 memory_id (失败 None).

    抽出原因: 之前 _direct_create_reminder (intent_handlers) 内联了 ~40 行
    "落 memory + dedup fallback + 建 trigger" 逻辑. preflight 第二轮拿到时间后
    需要复用同一逻辑落库 — 抽到这里, 两侧都调.

    dedup 命中时 (相同 summary 已存在): 不 silently skip, 而是 update 旧 memory
    的 occurTime + statementTime 到新值 (用户重设语义), 用 existing memory_id
    继续建/重设 trigger.
    """
    from app.services.memory.storage import repo as memory_repo
    from app.services.memory.storage.embedding import generate_embedding
    from app.services.memory.storage.persistence import find_duplicate_id, store_memory

    try:
        memory_id = await store_memory(
            user_id=user_id,
            content=summary,
            summary=summary,
            level=3,
            importance=0.45,  # 落 L3 (pipeline clamp 也是 [0.4, 0.49])
            memory_type="life",
            main_category="生活",
            sub_category="提醒",
            occur_time=occur_time,
            statement_time=statement_time,
            workspace_id=workspace_id,
            source="user",
            recurrence=recurrence,
        )
    except Exception as e:
        logger.warning(f"[REMINDER] create_user_reminder: store_memory failed: {e}")
        return None

    if not memory_id:
        # dedup 命中 → 复用 existing memory_id, 更新 occurTime 到新时刻 (重设语义)
        try:
            embedding = await generate_embedding(summary)
            memory_id = await find_duplicate_id(
                user_id, summary, embedding, workspace_id=workspace_id,
            )
            if memory_id:
                await memory_repo.update(
                    memory_id, source="user",
                    occurTime=occur_time, statementTime=statement_time,
                )
                logger.info(
                    f"[REMINDER] reused deduped memory={memory_id[:8]} "
                    f"updated occurTime={occur_time} recurrence={recurrence}"
                )
        except Exception as e:
            logger.warning(f"[REMINDER] dedup fallback failed: {e}")
            return None

    if not memory_id:
        logger.warning("[REMINDER] both store_memory and dedup lookup failed; aborting")
        return None

    await upsert_reminder_trigger(
        user_id=user_id,
        agent_id=agent_id,
        memory_id=memory_id,
        summary=summary,
        trigger_time=occur_time,
        recurrence=recurrence,
        side="user",
    )
    return memory_id
