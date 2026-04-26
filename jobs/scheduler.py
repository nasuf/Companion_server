"""Job scheduler for periodic tasks.

Uses APScheduler for:
- Daily: L2 动态分数调整 (spec §1.5.2), reflection, 记忆衰减
- Weekly: weekly reflection, portrait update
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.services.memory.lifecycle.l2_dynamics import run_l2_adjustment
from app.services.reflection import run_weekly_reflection
from app.services.portrait import update_portrait_weekly
from app.services.schedule_domain.schedule import (
    generate_and_save_life_overview, generate_daily_schedule, get_cached_schedule,
    get_current_status, get_life_overview, review_daily_schedule,
)
from app.services.mbti import get_mbti
from app.services.interaction.boundary import recover_patience_hourly
from app.services.relationship.intimacy import compute_growth_intimacy, compute_topic_intimacy
from app.services.proactive.orchestrator import scan_proactive_states
from app.services.interaction.aggregation import scan_expired
from app.services.interaction.delayed_queue import (
    enqueue_delayed_message, scan_due_delayed_messages, merge_delayed_payloads,
    try_lock_conversation, unlock_conversation
)
from app.services.proactive.triggers import scan_triggers
from app.services.proactive.special_dates import scan_special_dates_today

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def _run_for_all_agents(
    fn: Callable, concurrency: int = 3, task_name: str = "task"
) -> None:
    """Run an async function for all agents with concurrency control."""
    from app.db import db
    agents = await db.aiagent.find_many()
    sem = asyncio.Semaphore(concurrency)

    async def _process(agent):
        async with sem:
            try:
                await fn(agent)
            except Exception as e:
                logger.warning(f"{task_name} failed for agent {agent.id}: {e}")

    await asyncio.gather(*[_process(a) for a in agents])


def setup_scheduler():
    """Configure and start the job scheduler."""
    # Daily growth intimacy at 2 AM
    scheduler.add_job(
        _run_daily_intimacy,
        "cron",
        hour=2,
        minute=0,
        id="daily_intimacy",
        replace_existing=True,
    )

    # Weekly topic intimacy on Sunday at 2 AM
    scheduler.add_job(
        _run_weekly_topic_intimacy,
        "cron",
        day_of_week="sun",
        hour=2,
        minute=30,
        id="weekly_topic_intimacy",
        replace_existing=True,
    )

    # Daily L2 memory scoring adjustment at 2:30 AM (spec §1.5.2)
    scheduler.add_job(
        _run_l2_adjustment,
        "cron",
        hour=2,
        minute=30,
        id="l2_adjustment",
        replace_existing=True,
    )

    # Weekly reflection on Sunday at 4 AM
    scheduler.add_job(
        run_weekly_reflection,
        "cron",
        day_of_week="sun",
        hour=4,
        minute=0,
        id="weekly_reflection",
        replace_existing=True,
    )

    # Weekly portrait update on Sunday at 3:45 AM (staggered from daily reflection)
    scheduler.add_job(
        _run_weekly_portraits,
        "cron",
        day_of_week="sun",
        hour=3,
        minute=45,
        id="weekly_portrait",
        replace_existing=True,
    )

    # Daily schedule generation at 3:30 AM
    scheduler.add_job(
        _run_daily_schedules,
        "cron",
        hour=3,
        minute=30,
        id="daily_schedule",
        replace_existing=True,
    )

    # Monthly life overview refresh on 1st at 5:30 AM
    scheduler.add_job(
        _run_monthly_overview_refresh,
        "cron",
        day=1,
        hour=5,
        minute=30,
        id="monthly_overview",
        replace_existing=True,
    )

    # Daily schedule review at 4 AM
    scheduler.add_job(
        _run_schedule_review,
        "cron",
        hour=4,
        minute=0,
        id="schedule_review",
        replace_existing=True,
    )

    scheduler.add_job(
        _run_proactive_orchestrator_scan,
        "interval",
        minutes=1,
        id="proactive_orchestrator_scan",
        replace_existing=True,
    )

    # Patience recovery every hour
    scheduler.add_job(
        _run_patience_recovery,
        "interval",
        hours=1,
        id="patience_recovery",
        replace_existing=True,
    )

    # Redis health recheck: flip app-level readonly mode as Redis recovers/fails
    scheduler.add_job(
        _run_redis_health_recheck,
        "interval",
        seconds=30,
        id="redis_health_recheck",
        replace_existing=True,
    )

    # spec §1.4: 后台定时任务每秒扫描延迟队列
    scheduler.add_job(
        _run_aggregation_scan,
        "interval",
        seconds=1,
        id="aggregation_scan",
        replace_existing=True,
        max_instances=1,  # prevent "max instances reached" warning
    )

    # §9.5: Time trigger scan every minute
    scheduler.add_job(
        _run_trigger_scan,
        "interval",
        minutes=1,
        id="trigger_scan",
        replace_existing=True,
    )

    # Part 4 §10 + Part 5 §4.3: Unified special date scan daily at 3:30 AM.
    # 收集当日 用户+AI 7 类特殊日期 (春节/元旦/生日/提醒), 合并为一条 trigger,
    # actionType=special_date, triggerTime=作息表"起床"事件后第一个空闲段.
    # 此 job 取代了原来的 _run_holiday_check + _run_birthday_scan
    # (二者保留为可手动调用的函数, 但不再自动定时, 否则会导致同日多发).
    scheduler.add_job(
        _run_special_dates_scan,
        "cron",
        hour=3,
        minute=30,
        id="special_dates_scan",
        replace_existing=True,
    )

    # Part 5 §2.1: NTP 校准每 6 小时跑一次
    scheduler.add_job(
        _run_ntp_calibration,
        "cron",
        hour="*/6",
        minute=15,
        id="ntp_calibration",
        replace_existing=True,
    )

    # 节假日 DB 不走定时 cron, 也不走后端批量 refresh: 年度变化 (国务院
    # 11-12 月发布次年安排), 运营需要时在 admin UI "查询外部源" 拉候选挑
    # 选保存即可 — preview + bulk_save 工作流覆盖所有使用场景.

    scheduler.start()
    logger.info("Job scheduler started")


async def _run_weekly_portraits():
    await _run_for_all_agents(
        lambda a: update_portrait_weekly(a.userId, a.id),
        concurrency=3, task_name="Portrait update",
    )


async def _run_daily_schedules():
    async def _gen(agent):
        overview = await get_life_overview(agent.id)
        mbti = get_mbti(agent)
        await generate_daily_schedule(
            agent.id, agent.name, mbti,
            life_overview=overview, user_id=agent.userId,
        )

    await _run_for_all_agents(_gen, concurrency=3, task_name="Daily schedule")


async def _run_monthly_overview_refresh():
    async def _refresh(agent):
        await generate_and_save_life_overview(agent)

    await _run_for_all_agents(_refresh, concurrency=2, task_name="Monthly overview")


async def _run_schedule_review():
    await _run_for_all_agents(
        lambda a: review_daily_schedule(a.id, a.userId, a.name),
        concurrency=3, task_name="Schedule review",
    )


async def _run_proactive_orchestrator_scan():
    """扫描主动状态机。第一阶段仅做区间推进和互斥检查。"""
    try:
        await scan_proactive_states()
    except Exception as e:
        logger.warning(f"Proactive orchestrator scan failed: {e}")


async def _run_l2_adjustment():
    """Spec §1.5.2: recalculate L2 scores, promote/demote."""
    try:
        stats = await run_l2_adjustment()
        if stats.get("promoted") or stats.get("demoted"):
            logger.info(f"L2 adjustment: {stats}")
    except Exception as e:
        logger.warning(f"L2 adjustment failed: {e}")


async def _run_daily_intimacy():
    await _run_for_all_agents(
        lambda a: compute_growth_intimacy(a.id, a.userId, a.createdAt),
        concurrency=3, task_name="Growth intimacy",
    )


async def _run_weekly_topic_intimacy():
    await _run_for_all_agents(
        lambda a: compute_topic_intimacy(a.id, a.userId, a.createdAt),
        concurrency=3, task_name="Topic intimacy",
    )


async def _run_patience_recovery():
    await _run_for_all_agents(
        lambda a: recover_patience_hourly(a.id, a.userId),
        concurrency=5, task_name="Patience recovery",
    )


async def _run_trigger_scan():
    """§9.5: 扫描到期的时间触发器。"""
    try:
        await scan_triggers()
    except Exception as e:
        logger.warning(f"Trigger scan failed: {e}")


async def _run_redis_health_recheck():
    """30s 周期 ping Redis 并更新 _redis_healthy flag. 允许 Redis 故障后自愈
    (修好后下次 tick flip 回 healthy, 写 endpoints 自动重开).
    """
    try:
        from app.redis_client import recheck_redis_health
        await recheck_redis_health()
    except Exception as e:
        logger.warning(f"Redis health recheck failed: {e}")


async def _run_special_dates_scan():
    """Part 4 §10 + Part 5 §4.3: 每日统一扫描特殊日期.

    收集当日 用户+AI 共 4 类 (春节/元旦/生日/提醒), 汇总成一条 trigger
    actionType=special_date, triggerTime=作息表"起床"事件后第一个空闲段.
    多命中走合并消息 prompt; 跳过 reply_post_process.
    """
    try:
        await scan_special_dates_today()
    except Exception as e:
        logger.warning(f"Special dates scan failed: {e}")


async def _run_ntp_calibration():
    """Part 5 §2.1: NTP 校准, 漂移 > 阈值时告警."""
    import asyncio
    from app.services.schedule_domain.time_service import calibrate_against_ntp
    try:
        # ntplib 是同步阻塞调用, 放线程池
        drift = await asyncio.to_thread(calibrate_against_ntp)
        if drift is None:
            logger.warning("NTP calibration failed (network or lib unavailable)")
            return
        if abs(drift) > 1.0:
            logger.warning(f"NTP drift {drift:+.3f}s exceeds 1s threshold; investigate clock source")
        else:
            logger.info(f"NTP drift {drift:+.3f}s (within threshold)")
    except Exception as e:
        logger.warning(f"NTP calibration job failed: {e}")




async def _already_covered(conversation_id: str, user_msg_id: str) -> bool:
    """检查是否已有 assistant 回复在 prompt 中显式覆盖了这条 user 消息.

    依赖 orchestrator 主路径在 save_replies 时写入的 metadata.covered_until_user_ts
    (LLM 数据拉取时刻能看到的最新 user 消息时间). user_msg.createdAt 早于或等于
    任一 assistant 的 covered_until_user_ts → 视为已被覆盖, 跳过避免双发。

    短路/边界回复不写此字段 → 不会误判, 这类回复对应的 user 消息仍按原路处理。
    """
    from app.db import db

    user_msg = await db.message.find_unique(where={"id": user_msg_id})
    if not user_msg or user_msg.createdAt is None:
        return False

    # 拉取此消息之后的所有 assistant 消息 (上限 10 条防长会话扫描过大).
    later_ai = await db.message.find_many(
        where={
            "conversationId": conversation_id,
            "role": "assistant",
            "createdAt": {"gt": user_msg.createdAt},
        },
        order={"createdAt": "asc"},
        take=10,
    )
    for ai_msg in later_ai:
        md = getattr(ai_msg, "metadata", None) or {}
        covered = md.get("covered_until_user_ts") if isinstance(md, dict) else None
        if not covered:
            continue
        try:
            cutoff = datetime.fromisoformat(covered) if isinstance(covered, str) else None
        except ValueError:
            cutoff = None
        if cutoff is None:
            continue
        # 比较时统一带 tz, prisma 默认返回 aware datetime; isoformat 也带 tz.
        if cutoff >= user_msg.createdAt:
            return True
    return False


async def _run_aggregation_scan():
    """Scan aggregation windows and due delayed replies, then deliver asynchronously."""
    from app.services.chat.orchestrator import stream_chat_response
    from app.services.runtime.ws_manager import manager
    from app.api.realtime.ws import stream_to_ws
    from app.db import db

    try:
        expired = await scan_expired()
        for agent_id, user_id, combined_text, conv_id, reply_context, latest_message_id in expired:
            delay_seconds = float((reply_context or {}).get("delay_seconds", 0.0) or 0.0)
            await enqueue_delayed_message(
                conv_id,
                {
                    "conversation_id": conv_id,
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "message": combined_text,
                    "message_id": latest_message_id,
                    "reply_context": reply_context,
                },
                delay_seconds,
            )
            # 12E: Update frontend after aggregation window ends.
            # send_event 跨进程 routing: scheduler 与 WS holder 不同 worker 时 publish.
            if delay_seconds > 5:
                await manager.send_event(conv_id, "delay", {"duration": delay_seconds})
            await manager.send_event(conv_id, "pending", {"status": "queued", "delay": delay_seconds})

        due_conversations = await scan_due_delayed_messages()
        for conv_id, payloads in due_conversations:
            # Prevent concurrent processing of the same conversation
            if not await try_lock_conversation(conv_id, ttl=120):
                logger.debug(f"Conversation {conv_id[:8]} is locked, skipping this scan")
                continue

            try:
                merged = merge_delayed_payloads(payloads)
                if not merged:
                    continue

                # 去重 gate: ws.py 已用 enqueue_or_append_delayed 关闭主要 race;
                # 这里兜底 "msg1 已被 flush 出队但仍在 LLM 中, msg2 才到达" 窗口:
                # 上一轮 LLM 数据拉取若已隐式包含本 user_msg(写到 reply 的
                # metadata.covered_until_user_ts), 则跳过避免重复回复。
                # 仅看 metadata 显式字段 → 不会因短路/边界 reply 误伤未覆盖消息。
                user_msg_id = merged.get("user_message_id")
                if user_msg_id and await _already_covered(conv_id, user_msg_id):
                    logger.info(
                        f"[DEDUP-GATE] skip conv={conv_id[:8]} "
                        f"user_msg={user_msg_id[:8]} already covered by prior reply"
                    )
                    continue

                conv = await db.conversation.find_unique(
                    where={"id": conv_id},
                    include={"agent": True},
                )
                if not conv or not conv.agent:
                    continue

                gen = stream_chat_response(
                    conversation_id=conv_id,
                    user_message=merged["user_message"],
                    agent=conv.agent,
                    user_id=merged["user_id"],
                    reply_context=merged.get("reply_context"),
                    save_user_message=False,
                    user_message_id=merged.get("user_message_id"),
                    delivered_from_queue=True,
                )

                # stream_to_ws 内部每条 chunk 走 manager.send_event,
                # fast path 本地命中或 slow path publish 跨 worker, 无需手工查 WS.
                # 离线用户 (无 WS / 跨进程 publish 也无人订阅): 仍 await 消费完
                # generator 触发 LLM + 持久化, 避免漏存回复.
                await stream_to_ws(gen, conv_id)
                logger.debug(f"Delayed reply pushed for conv={conv_id[:8]}")
            finally:
                await unlock_conversation(conv_id)
    except Exception as e:
        logger.warning(f"Aggregation scan failed: {e}")


def shutdown_scheduler():
    """Shutdown the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Job scheduler stopped")
