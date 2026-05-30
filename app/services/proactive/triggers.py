"""时间触发引擎。

PRD §9.5: 在合适的时间触发AI主动行为，遵循作息规则。

Phase 4: actionType="reminder" 走 `_handle_reminder_trigger` 独立路径
(pre-check LLM + 续期 + 隐式完成/取消/改期处理). 详见 CLAUDE.md §6 偏离表.
"""

from __future__ import annotations

import asyncio
import json
import logging
from calendar import monthrange
from datetime import datetime, timedelta, timezone

from app.db import db
from app.observability import bind_context
from app.observability.events import (
    EVT_DB_LOOKUP_FAIL,
    EVT_REMINDER_DLQ,
    EVT_REMINDER_HANDLED,
    EVT_REMINDER_PRECHECK,
    EVT_REMINDER_RENEWED,
    EVT_REMINDER_RESCHEDULED,
)
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
# Reminder pre-check 上下文窗口
REMINDER_PRECHECK_WINDOW = timedelta(minutes=30)

# scan_triggers 单次启的最大 LLM 并发数. 防一波同时到点的 reminder 各自走
# 5-10s LLM (pre-check + message generation) 启 N 个 dashscope 调用打爆
# rate limit (默认 QPS≈60) 让 circuit breaker 触发, 进而**所有** reminder
# 都 fallback 到 needed/template — 影响半径放大. 8 是经验值: 单 trigger
# 平均 ~6s 链路, 8 并发 ≈ 1.3 QPS reminder 占用, 留余量给其他主动消息.
_TRIGGER_LLM_CONCURRENCY = 8
_trigger_semaphore: asyncio.Semaphore | None = None

# Reminder emit 失败重试上限. 超过则进 DLQ (Redis ZSET) 不再重试.
# 3 次 + 每次推后 30s = 总重试窗 90s, 跟 5s pre-check timeout 配套, 临时
# rate limit / 网络抖动应该能 cover.
MAX_REMINDER_RETRY = 3
REMINDER_RETRY_DELAY = timedelta(seconds=30)
# DLQ Redis ZSET 大小上限 (按时间戳排序, 超出则丢最老的)
_DLQ_KEY = "reminder:dlq"
_DLQ_MAX_SIZE = 1000


def _get_trigger_semaphore() -> asyncio.Semaphore:
    """Lazy init 让 Semaphore 绑到当前 event loop. 测试场景每个 test loop
    独立, 模块级初始化会出"attached to different loop" 错."""
    global _trigger_semaphore
    if _trigger_semaphore is None:
        _trigger_semaphore = asyncio.Semaphore(_TRIGGER_LLM_CONCURRENCY)
    return _trigger_semaphore


# ── 周期性提醒续期算法 (Phase 4.6) ────────────────────────────────────

def _add_months(dt: datetime, months: int) -> datetime:
    """加 N 个月并处理月底边界 (1月31日 +1月 → 2月最后一天).

    stdlib-only 替代 dateutil.relativedelta — 避免新增依赖.
    """
    new_month = dt.month + months
    new_year = dt.year + (new_month - 1) // 12
    new_month = (new_month - 1) % 12 + 1
    last_day = monthrange(new_year, new_month)[1]
    new_day = min(dt.day, last_day)
    return dt.replace(year=new_year, month=new_month, day=new_day)


def _next_occurrence(current: datetime, recurrence: str) -> datetime | None:
    """下一周期 trigger 时刻. once 返 None (不续期)."""
    if recurrence == "yearly":
        # 闰年 2-29 → 平年 2-28 自动 clamp
        target_month = current.month
        target_day = current.day
        new_year = current.year + 1
        last = monthrange(new_year, target_month)[1]
        return current.replace(year=new_year, day=min(target_day, last))
    if recurrence == "monthly":
        return _add_months(current, 1)
    if recurrence == "weekly":
        return current + timedelta(weeks=1)
    if recurrence == "daily":
        return current + timedelta(days=1)
    return None  # once / unknown


def _normalize_habit_weekdays(value: object) -> list[int]:
    """Return sorted ISO weekdays (Mon=1..Sun=7) from reminder actionData."""
    if not isinstance(value, list):
        return []
    days: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        if 1 <= item <= 7 and item not in days:
            days.append(item)
    return sorted(days)


def _next_habit_weekday_occurrence(
    current: datetime,
    habit_weekdays: list[int],
) -> datetime | None:
    """Next selected weekday after current, preserving the original time."""
    if not habit_weekdays:
        return None
    for offset in range(1, 8):
        candidate = current + timedelta(days=offset)
        if candidate.isoweekday() in habit_weekdays:
            return candidate
    return current + timedelta(weeks=1)


async def scan_triggers() -> None:
    """每 15 秒扫描到期触发器，检查规则后执行。"""
    now = datetime.now(_TZ)
    window_start = now - timedelta(minutes=1)
    window_end = now + timedelta(minutes=1)

    triggers = await db.timetrigger.find_many(
        where={
            "isActive": True,
            "triggerTime": {"gte": window_start, "lt": window_end},
        },
    )

    if not triggers:
        return

    # 诊断: 命中时打 INFO log. 之前生产观察到 "trigger 建好了但没 fire", 没有
    # log 完全摸不到方向 — 至少先确认 scan 是否选中了 trigger.
    logger.info(
        f"[SCAN] {len(triggers)} active trigger(s) in window "
        f"[{window_start.strftime('%H:%M:%S')}, {window_end.strftime('%H:%M:%S')}]: "
        f"{[(t.id[:8], t.actionType, t.triggerTime.isoformat()) for t in triggers]}"
    )

    # 批量预拉 agent 名 — 之前每个 trigger 都 find_unique 一次, 50 reminder 一波
    # 就 50 round-trips. IN 查询 1 次拿全部, ~10ms vs ~250ms.
    # workspace_id 仍在 _bounded_execute 单查 (resolve_workspace_id 内部有 Redis
    # 缓存, 真实命中率高; 跨 user_id 维度 batch 反而复杂).
    agent_ids = list({t.aiAgentId for t in triggers})
    agent_name_by_id: dict[str, str | None] = {}
    try:
        agents = await db.aiagent.find_many(where={"id": {"in": agent_ids}})
        agent_name_by_id = {a.id: getattr(a, "name", None) for a in agents}
    except Exception as e:
        logger.warning(
            f"agent batch lookup failed for log enrichment: {e}",
            extra={"event": EVT_DB_LOOKUP_FAIL, "table": "aiagent",
                   "stage": "trigger_scan_batch", "n_ids": len(agent_ids),
                   "error_type": type(e).__name__},
        )

    # Round-2 review #10: 串行 → asyncio.gather 并发 (N×10s → max(10s) 总耗时).
    # Semaphore(8) 限流: gather 启全部 task 但同时只 8 个进 LLM 段, 防 N 个
    # reminder 同时到点把 dashscope rate limit (QPS=60) 打爆触发 circuit breaker.
    sem = _get_trigger_semaphore()

    async def _bounded_execute(trig):
        async with sem:
            # 在 LLM 段开始前绑定 log 上下文 — 内层 _execute_trigger / _handle_reminder_trigger
            # 派生的 fire_background / asyncio.create_task 自动继承 (各 task 独立 context).
            # workspace_id 失败仅 ID 不带字段, 不阻塞主流程 (比让一波 reminder 全 fail 好).
            agent_name = agent_name_by_id.get(trig.aiAgentId)
            ws_id = None
            try:
                ws_id = await resolve_workspace_id(
                    user_id=trig.userId, agent_id=trig.aiAgentId,
                )
            except Exception as e:
                logger.warning(
                    f"workspace lookup failed for log enrichment: {e}",
                    extra={"event": EVT_DB_LOOKUP_FAIL, "table": "workspace",
                           "stage": "trigger_log_enrichment",
                           "error_type": type(e).__name__},
                )
            with bind_context(
                agent_id=trig.aiAgentId,
                agent_name=agent_name,
                user_id=trig.userId,
                workspace_id=ws_id,
            ):
                await _execute_trigger(trig, now)

    results = await asyncio.gather(
        *(_bounded_execute(t) for t in triggers),
        return_exceptions=True,
    )
    for trigger, result in zip(triggers, results):
        if isinstance(result, Exception):
            logger.warning(f"Trigger {trigger.id} execution failed: {result}")


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

    # spec §1.2: 22:00-8:00 不发送, 越过则取消.
    # 注: spec §10.2 字面没让"取消", 仅说"顺延至下一个空闲状态". 选择 cancel
    # 是为避免凌晨打扰用户 (假设次日 8:00 重发祝福语义错位 — 比如生日已过).
    # 若以后 spec 明确要求次日补发, 在此扩展为推迟到次日起床后第一个空闲段.
    # CLAUDE.md §6 spec 偏离表已登记此条.
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


async def _fetch_recent_messages_for_reminder(
    agent_id: str, user_id: str, now: datetime,
) -> str:
    """拉最近 30min 对话作为 pre-check LLM 上下文. 取所有相关 conversation 的消息."""
    try:
        cutoff = now - REMINDER_PRECHECK_WINDOW
        msgs = await db.message.find_many(
            where={
                "conversation": {"agentId": agent_id, "userId": user_id},
                "createdAt": {"gte": cutoff},
            },
            order={"createdAt": "asc"},
            take=40,
        )
    except Exception as e:
        logger.warning(f"reminder pre-check: fetch recent messages failed: {e}")
        return ""
    lines = []
    for m in msgs:
        role = "用户" if m.role == "user" else "AI"
        text = (m.content or "").strip().replace("\n", " ")
        if text:
            lines.append(f"{role}: {text[:120]}")
    return "\n".join(lines) if lines else ""


async def _archive_reminder_memory(memory_id: str, side: str, reason: str) -> None:
    """软删提醒 memory + 写 changelog. 薄 wrapper, 真实现在
    `services/reminder/scheduling.archive_reminder_memory` 收口 (round-2 review
    #8: 跟 deletion 流程的重复部分集中)."""
    from app.services.reminder.scheduling import archive_reminder_memory
    if side not in ("user", "ai"):
        side = "user"
    await archive_reminder_memory(
        memory_id=memory_id, side=side, reason=reason,  # type: ignore[arg-type]
    )


# ── Reminder emit 失败重试 + DLQ (无 schema 改动) ────────────────────────


async def _push_reminder_dlq(
    trigger, error: str, *, kind: str, attempt: int,
) -> None:
    """写一条 DLQ 记录到 Redis ZSET (按 timestamp 排序). 超过 _DLQ_MAX_SIZE
    自动丢最老的. admin 可读 ZRANGE 看最近失败. DLQ 写本身失败不冒泡 — 不能
    让 DLQ 错误掩盖真正失败原因."""
    data = trigger.actionData or {}
    entry = {
        "trigger_id": trigger.id,
        "memory_id": data.get("memory_id", ""),
        "summary": (data.get("summary") or "")[:120],
        "recurrence": data.get("recurrence", "once"),
        "error": error[:200],
        "kind": kind,
        "attempt": attempt,
        "failed_at": datetime.now(_TZ).isoformat(),
    }
    try:
        redis = await get_redis()
        score = datetime.now(_TZ).timestamp()
        await redis.zadd(
            _DLQ_KEY,
            {json.dumps(entry, ensure_ascii=False): score},
        )
        # cap 大小: 删除排序最低 (最老) 的, 只留最近 _DLQ_MAX_SIZE 条
        await redis.zremrangebyrank(_DLQ_KEY, 0, -(_DLQ_MAX_SIZE + 1))
    except Exception as e:
        logger.warning(f"[REMINDER-DLQ] write failed for {trigger.id}: {e}")


async def _handle_emit_failure(
    trigger, recurrence: str, exc: BaseException,
) -> None:
    """reminder emit 链路抛异常时的统一处理: once → 重试上限内重新激活,
    超限进 DLQ; periodic → 不重试 (续期已建, 下周期照常), 仅写 DLQ 留痕.

    重试机制 (仅 once):
    - actionData.retry_count 记次数 (无 schema 改动)
    - 每次 +30s 推迟 triggerTime, isActive=True, lastFired=None → 下次 scan 重试
    - 超过 MAX_REMINDER_RETRY → DLQ + 永久死信
    """
    data = dict(trigger.actionData or {})
    error_str = f"{type(exc).__name__}: {str(exc)[:150]}"

    if recurrence != "once":
        # 周期性: 续期 (next_occur) 已经在 _handle_reminder_trigger 里建好,
        # 这次 emit 丢了下个周期照常 — 是可接受退化, 仅 DLQ 留痕方便 admin 排查.
        await _push_reminder_dlq(
            trigger, error_str, kind="periodic_lost_one",
            attempt=int(data.get("retry_count") or 0),
        )
        logger.warning(
            f"[REMINDER-EMIT] {trigger.id[:8]} (periodic={recurrence}) emit "
            f"failed, next period intact: {error_str}",
            extra={
                "event": EVT_REMINDER_DLQ,
                "kind": "periodic_lost_one",
                "trigger_id": trigger.id,
                "recurrence": recurrence,
                "error_type": type(exc).__name__,
            },
        )
        return

    retry_count = int(data.get("retry_count") or 0)
    if retry_count >= MAX_REMINDER_RETRY:
        await _push_reminder_dlq(
            trigger, error_str, kind="exhausted", attempt=retry_count,
        )
        logger.warning(
            f"[REMINDER-EMIT] {trigger.id[:8]} retries exhausted "
            f"({retry_count}/{MAX_REMINDER_RETRY}), DEAD: {error_str}",
            extra={
                "event": EVT_REMINDER_DLQ,
                "kind": "exhausted",
                "trigger_id": trigger.id,
                "attempt": retry_count,
                "error_type": type(exc).__name__,
            },
        )
        return

    # 重试: reactivate trigger, 推迟 30s 让下次 scan 重新尝试.
    # 必须 import Json (Prisma 显式包装 JSON 字段) — 直接 dict 在某些 client 版本拒绝.
    from prisma import Json
    new_data = {**data, "retry_count": retry_count + 1}
    new_time = datetime.now(_TZ) + REMINDER_RETRY_DELAY
    try:
        await db.timetrigger.update(
            where={"id": trigger.id},
            data={
                "isActive": True,
                "lastFired": None,
                "triggerTime": new_time,
                "actionData": Json(new_data),
            },
        )
        logger.info(
            f"[REMINDER-EMIT] {trigger.id[:8]} retry scheduled at {new_time} "
            f"attempt={retry_count + 1}/{MAX_REMINDER_RETRY}: {error_str}"
        )
    except Exception as reactivate_err:
        # 重新激活也失败 → 直接进 DLQ, 否则会陷入"既不响也不进 DLQ"幽灵态
        logger.warning(
            f"[REMINDER-EMIT] {trigger.id[:8]} reactivation failed: {reactivate_err}"
        )
        await _push_reminder_dlq(
            trigger, f"reactivate_failed: {reactivate_err}",
            kind="reactivate_failed", attempt=retry_count,
        )


async def _handle_reminder_trigger(trigger, now: datetime) -> None:
    """统一提醒触发: pre-check LLM + 隐式状态判定 + 续期 + 发送.

    pre-check 状态机 (见 prompt `proactive.reminder_pre_check`):
    - completed/cancelled: 软删 memory + trigger 失活, 不发消息
    - rescheduled: 更新 trigger.triggerTime + memory.occurTime, 不发, 等新时刻
    - needed: 周期性提醒触发前先续期下一周期 (抗崩溃) → 发提醒消息

    豁免 30min in_chat 阻塞 (用户主动设置), 但守 AI 睡眠 + 22:00-08:00 主动时段.
    """
    from app.services.chat.intent_replies import reminder_message, reminder_pre_check

    agent_id = trigger.aiAgentId
    user_id = trigger.userId
    data = trigger.actionData or {}
    memory_id = data.get("memory_id")
    summary = (data.get("summary") or "").strip()
    recurrence = data.get("recurrence") or "once"
    sent_to_ai = bool(data.get("sent_to_ai", True))
    side = data.get("memory_side") or "user"

    if not summary:
        logger.warning(f"reminder trigger {trigger.id} missing summary; deactivating")
        await db.timetrigger.update(where={"id": trigger.id}, data={"isActive": False})
        return

    # 诊断 INFO log (调升自 debug): 帮 admin 排查"trigger 建好了但没响"问题.
    # 频率不高 (用户主动设置才有), 提到 INFO 不会刷屏.
    logger.info(
        f"[REMINDER-TRIGGER] {trigger.id[:8]} entered handler: "
        f"now={now.isoformat()} scheduled={trigger.triggerTime.isoformat()} "
        f"lastFired={trigger.lastFired} summary={summary[:40]!r}"
    )

    # 幂等守门: scan_triggers ±1min 窗口跨两次扫描会重复选中同一 trigger.
    # lastFired 在 2 min 内说明上一次扫描已处理过, 直接 return 避免双发.
    if trigger.lastFired:
        last = trigger.lastFired
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if (now - last) < timedelta(minutes=2):
            logger.info(
                f"[REMINDER-TRIGGER] {trigger.id[:8]} SKIP: lastFired {last} "
                f"within 2min idempotency window"
            )
            return

    # 守门 0: scan_triggers 用 ±1min 窗口 (window_end = now + 1min) 是为兼容
    # special_date / greeting 这类粒度宽的场景, 但提醒类用户期望"准点". 之前
    # "2 分钟后提醒" 实际 ~1 分 27 秒就响 (early by 33s). 改为只在
    # triggerTime 已到达时 fire — 推迟到下次 scan (~1 min 后) 比早响 1 min
    # 体感好得多. scan 间隔 15s, 最大延迟 ~15s.
    trigger_time_aware = trigger.triggerTime
    if trigger_time_aware.tzinfo is None:
        trigger_time_aware = trigger_time_aware.replace(tzinfo=timezone.utc)
    if now < trigger_time_aware:
        logger.info(
            f"[REMINDER-TRIGGER] {trigger.id[:8]} DEFER: not yet due "
            f"(diff={(trigger_time_aware - now).total_seconds():.1f}s)"
        )
        return

    # 注: reminder **豁免** spec §1.2 主动消息所有 gate (daily_limit / quiet hours /
    # AI sleep). spec §1.2 这些限制是为了"AI 不要主动找用户太频繁/太晚/不在线时
    # 还出击" — 但 reminder 是用户**主动**让 AI 帮忙记的事, 用户自己说"该睡觉
    # 了"/"明早 7 点叫我" 就明确希望在那个时刻响. 用 quiet hours 拦下
    # "23:01 该睡觉" 提醒就是 AI 反客为主决定该不该响, 用户体感: "我设了你不响"
    # 比"凌晨被打扰"更糟. 设错时间用户可以改/取消, 不该 AI 替用户做决定.
    #
    # 生产 bug 复现 (2026-05-02 23:01:31): 用户 22:58 说"待会儿提醒我该睡觉啦",
    # 落库 23:01:20, 4 次 DEFER 都对, 第 5 次 scan 进 quiet hours guard SKIP,
    # 提醒永远不响. 现在统一豁免.
    #
    # 仍然 redis 取 daily_key 用于 emit 后递增计数 (供其他主动路径参考统计).
    redis = await get_redis()
    daily_key = f"trigger_count:{agent_id}:{user_id}:{now.strftime('%Y%m%d')}"

    if sent_to_ai:
        # pre-check: 看最近 30min 对话, 判别隐式完成 / 取消 / 改期 / 仍需要
        recent = await _fetch_recent_messages_for_reminder(agent_id, user_id, now)
        state_info = await reminder_pre_check(
            summary=summary,
            trigger_time=trigger.triggerTime.isoformat(),
            recent_messages=recent,
        )
        state = state_info.get("state", "needed")

        logger.info(
            f"[REMINDER-PRECHECK] {trigger.id[:8]} → {state}",
            extra={
                "event": EVT_REMINDER_PRECHECK,
                "trigger_id": trigger.id,
                "precheck_state": state,
                "reason": state_info.get("reason"),
                "recurrence": recurrence,
            },
        )
    else:
        # 打卡页创建但未“发聊天”的事项只属于系统打卡提醒, 不让 AI
        # 读取上下文/生成聊天, 避免用户没授权就被 AI 主动介入。
        state_info = {"state": "needed", "reason": "system_only_checkin"}
        state = "needed"

    if state in ("completed", "cancelled"):
        if memory_id:
            await _archive_reminder_memory(
                memory_id, side, reason=f"{state}:{state_info.get('reason', '')}",
            )
        await db.timetrigger.update(
            where={"id": trigger.id},
            data={"isActive": False, "lastFired": now},
        )
        logger.info(
            f"reminder {trigger.id} archived as {state} "
            f"memory={memory_id and memory_id[:8]} reason={state_info.get('reason')}",
            extra={
                "event": EVT_REMINDER_HANDLED,
                "outcome": state,  # completed / cancelled
                "trigger_id": trigger.id,
                "memory_id": memory_id,
                "recurrence": recurrence,
            },
        )
        return

    if state == "rescheduled":
        new_iso = state_info.get("new_time")
        if not new_iso:
            logger.warning(f"reminder {trigger.id} rescheduled but no new_time; falling through to send")
        else:
            try:
                new_time = datetime.fromisoformat(new_iso)
            except ValueError:
                logger.warning(f"reminder {trigger.id}: invalid rescheduled new_time={new_iso!r}")
                new_time = None
            if new_time is not None:
                await db.timetrigger.update(
                    where={"id": trigger.id}, data={"triggerTime": new_time},
                )
                if memory_id:
                    from app.services.memory.storage import repo as memory_repo
                    try:
                        await memory_repo.update(
                            memory_id, source=side,  # type: ignore[arg-type]
                            occurTime=new_time,
                        )
                    except Exception as e:
                        logger.warning(f"reminder reschedule: memory occurTime update failed: {e}")
                logger.info(
                    f"reminder {trigger.id} rescheduled to {new_time} "
                    f"reason={state_info.get('reason')}",
                    extra={
                        "event": EVT_REMINDER_RESCHEDULED,
                        "trigger_id": trigger.id,
                        "memory_id": memory_id,
                        "new_trigger_time": new_time.isoformat(),
                        "reason": state_info.get("reason"),
                    },
                )
                return

    # state == "needed" (or fall-through from rescheduled-without-time): 发提醒.
    #
    # ⚠️ 关键顺序: 先 claim 原 trigger, 再 create 续期, 最后 emit.
    # 之前的"续期先于 claim"顺序在 claim 抛异常时会双写续期 (round-2 bug2):
    #   原 trigger 仍 active + 续期已存在 → 下次 scan 又进 _handle_reminder_trigger
    #   → 续期 create 再跑一次 → 用户的下一周期 trigger 重复.
    # 现在顺序: claim → create 续期 → emit.
    # - claim 失败 (抛异常): scan 外层 try 捕获, 原 trigger 仍 active. 但
    #   idempotency 守门 (lastFired) 还是 None → 下次 scan 重入再尝试 claim,
    #   合理重试. 续期还没建, 不会重复.
    # - 续期失败 (after claim 成功): 原 trigger 已死, 续期丢失. 用户失去 1
    #   个周期 — 是可接受的"退化", 比双重周期 trigger 好.
    # - emit 失败 (after claim+续期): 原死、续期在、消息丢. 周期性下周期照样
    #   提醒用户; 一次性提醒丢失 (见 WARNING 日志).
    await db.timetrigger.update(
        where={"id": trigger.id},
        data={"isActive": False, "lastFired": now},
    )

    if recurrence != "once":
        habit_weekdays = _normalize_habit_weekdays(data.get("habit_weekdays"))
        next_occur = (
            _next_habit_weekday_occurrence(trigger.triggerTime, habit_weekdays)
            if recurrence == "weekly" and habit_weekdays
            else _next_occurrence(trigger.triggerTime, recurrence)
        )
        if next_occur:
            try:
                from app.services.reminder.scheduling import renew_periodic_trigger
                await renew_periodic_trigger(
                    user_id=user_id,
                    agent_id=agent_id,
                    next_trigger_time=next_occur,
                    action_data=data,
                )
                logger.info(
                    f"reminder {trigger.id} renewed: next={next_occur} "
                    f"recurrence={recurrence}",
                    extra={
                        "event": EVT_REMINDER_RENEWED,
                        "trigger_id": trigger.id,
                        "memory_id": memory_id,
                        "recurrence": recurrence,
                        "next_trigger_time": next_occur.isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"reminder {trigger.id} renewal failed: {e}")

    if not sent_to_ai:
        conversation_id = str(data.get("conversation_id") or "")
        if conversation_id:
            from app.services.reminder.scheduling import notify_reminder_changed
            await notify_reminder_changed(
                conversation_id, kind="fired", trigger_id=trigger.id,
            )
        logger.info(
            f"reminder {trigger.id} handled as system-only check-in "
            f"memory={memory_id and memory_id[:8]} recurrence={recurrence}",
            extra={
                "event": EVT_REMINDER_HANDLED,
                "outcome": "system_notification_only",
                "trigger_id": trigger.id,
                "memory_id": memory_id,
                "recurrence": recurrence,
                "summary_preview": summary[:40],
            },
        )
        return

    # emit 段统一 try/except → 失败走 _handle_emit_failure (重试 / DLQ).
    # 之前任何中途 return (workspace 缺失 / agent 没了 / message 空 /
    # emit_proactive_message 抛) 都是 silent log + dead, 一次性提醒永久丢失.
    try:
        workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
        if not workspace_id:
            raise RuntimeError("workspace not found")

        from app.services.llm.usage_tracker import traced_usage_session
        from app.services.proactive.emit import emit_proactive_message
        from app.services.proactive.sender import _build_personality_brief
        from app.services.proactive.state import get_active_workspace_context

        workspace_context = await get_active_workspace_context(workspace_id)
        if not workspace_context:
            raise RuntimeError("no workspace context")
        conversation_id = str(workspace_context.get("conversation_id") or "")
        if not conversation_id:
            raise RuntimeError("no conversation_id")
        agent = await db.aiagent.find_unique(where={"id": agent_id})
        if not agent:
            raise RuntimeError(f"agent {agent_id[:8]} gone")

        # LangSmith trace + usage session 包住 message 生成 + emit, 让 admin 在
        # LangSmith 看板能看到 reminder 触发的 LLM 调用 + 前端 Trace 按钮可点
        # (生产观察 2026-05-03 14:00: reminder bubble 没 Trace 按钮, 因为之前
        # 没传 trace_id 给 emit. 跟 sender.py 主动消息路径对齐).
        async with traced_usage_session(
            name=f"[reminder:{trigger.id[:8]}] {summary[:60]}",
            scope="proactive",
            conversation_id=conversation_id,
            agent_id=agent_id,
            user_id=user_id,
        ) as tracer:
            message = await reminder_message(
                summary=summary,
                personality_brief=_build_personality_brief(agent),
            )
            if not message or len(message) < 4:
                raise RuntimeError("message generation empty/short")

            await emit_proactive_message(
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                message=message,
                trigger_type="reminder",
                skip_post_process=True,  # spec §10.4 类比: 提醒消息不走 emoji/拆句加工
                extra_metadata={
                    "reminder_memory_id": memory_id or "",
                    "reminder_recurrence": recurrence,
                    "reminder_summary": summary[:120],
                },
                trace_id=tracer.safe_trace_id,
            )

        await redis.incr(daily_key)
        await redis.expire(daily_key, 86400)
        # 注: trigger.isActive=False + lastFired 已在 emit 前 claim, 不重复 update.
        logger.info(
            f"reminder {trigger.id} fired memory={memory_id and memory_id[:8]} "
            f"recurrence={recurrence}",
            extra={
                "event": EVT_REMINDER_HANDLED,
                "outcome": "fired",
                "trigger_id": trigger.id,
                "memory_id": memory_id,
                "recurrence": recurrence,
                "summary_preview": summary[:40],
            },
        )
        # 通知 inspector 提醒 tab 实时刷新 (status: active → fired)
        from app.services.reminder.scheduling import notify_reminder_changed
        await notify_reminder_changed(
            conversation_id, kind="fired", trigger_id=trigger.id,
        )
    except Exception as exc:
        await _handle_emit_failure(trigger, recurrence, exc)


async def _execute_trigger(trigger, now: datetime) -> None:
    """执行单个触发器，检查所有规则。"""
    agent_id = trigger.aiAgentId
    user_id = trigger.userId
    is_special_date = trigger.actionType == "special_date"

    # Phase 4 reminder: 用户主动设置的事项, 豁免 30min in_chat 阻塞 (用户说
    # "半小时后" 必然在 30min 内活跃). 仍守 AI 睡眠 + spec §1.2 主动时段 gates,
    # 由 _handle_reminder_trigger 内部判定.
    if trigger.actionType == "reminder":
        await _handle_reminder_trigger(trigger, now)
        return

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
