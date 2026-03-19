"""Job scheduler for periodic tasks.

Uses APScheduler for:
- Daily: consolidation, importance decay
- Weekly: reflection, memory compression
- Monthly: memory compression (L2 -> L1)
"""

import asyncio
import logging
from collections.abc import Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.services.reflection import run_daily_reflection, run_weekly_reflection
from app.services.memory.compression import compress_weekly, compress_monthly
from app.services.portrait import update_portrait_weekly
from app.services.memory.self_memory import generate_daily_self_memories
from app.services.emotion import decay_emotion_toward_baseline
from app.services.schedule import (
    generate_daily_schedule, generate_life_overview, get_cached_schedule,
    get_current_status, get_life_overview, review_daily_schedule, save_life_overview,
)
from app.services.trait_model import get_seven_dim
from app.services.boundary import recover_patience_hourly, scan_blacklist_expiry
from app.services.intimacy import compute_growth_intimacy, compute_topic_intimacy
from app.services.proactive import generate_proactive_message
from app.services.aggregation import scan_expired
from app.services.delayed_queue import enqueue_delayed_message, scan_due_delayed_messages, merge_delayed_payloads
from app.services.trigger_engine import scan_triggers, create_holiday_triggers, scan_birthday_memories
from app.services.time_service import is_holiday

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

    # Daily reflection at 3 AM
    scheduler.add_job(
        run_daily_reflection,
        "cron",
        hour=3,
        minute=0,
        id="daily_reflection",
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

    # Weekly memory compression on Sunday at 5 AM
    scheduler.add_job(
        compress_weekly,
        "cron",
        day_of_week="sun",
        hour=5,
        minute=0,
        id="weekly_compression",
        replace_existing=True,
    )

    # Monthly memory compression on 1st at 5 AM
    scheduler.add_job(
        compress_monthly,
        "cron",
        day=1,
        hour=5,
        minute=0,
        id="monthly_compression",
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

    # Daily self-memory generation at 3:15 AM
    scheduler.add_job(
        _run_daily_self_memories,
        "cron",
        hour=3,
        minute=15,
        id="daily_self_memory",
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

    # Proactive chat scan every hour at :30
    scheduler.add_job(
        _run_proactive_scan,
        "interval",
        hours=1,
        id="proactive_scan",
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

    # 5B.3: Blacklist expiry scan every 5 minutes
    scheduler.add_job(
        _run_blacklist_scan,
        "interval",
        minutes=5,
        id="blacklist_scan",
        replace_existing=True,
    )

    # Emotion decay every 5 minutes
    scheduler.add_job(
        _run_emotion_decay,
        "interval",
        minutes=5,
        id="emotion_decay",
        replace_existing=True,
    )

    # 12E + PRD §6.2.2: aggregation + delayed reply delivery scan every second
    scheduler.add_job(
        _run_aggregation_scan,
        "interval",
        seconds=1,
        id="aggregation_scan",
        replace_existing=True,
    )

    # §9.5: Time trigger scan every minute
    scheduler.add_job(
        _run_trigger_scan,
        "interval",
        minutes=1,
        id="trigger_scan",
        replace_existing=True,
    )

    # §9.6: Holiday trigger check daily at 3:00 AM
    scheduler.add_job(
        _run_holiday_check,
        "cron",
        hour=3,
        minute=0,
        id="holiday_check",
        replace_existing=True,
    )

    # §9.6: Birthday memory scan weekly on Sunday at 4:30 AM
    scheduler.add_job(
        _run_birthday_scan,
        "cron",
        day_of_week="sun",
        hour=4,
        minute=30,
        id="birthday_scan",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Job scheduler started")


async def _run_weekly_portraits():
    await _run_for_all_agents(
        lambda a: update_portrait_weekly(a.userId, a.id),
        concurrency=3, task_name="Portrait update",
    )


async def _run_daily_self_memories():
    await _run_for_all_agents(
        lambda a: generate_daily_self_memories(agent_id=a.id, user_id=a.userId, dialogue_summary=None),
        concurrency=3, task_name="Self-memory generation",
    )


async def _run_daily_schedules():
    async def _gen(agent):
        overview = await get_life_overview(agent.id)
        seven_dim = get_seven_dim(agent)
        await generate_daily_schedule(
            agent.id, agent.name, seven_dim,
            life_overview=overview, user_id=agent.userId,
        )

    await _run_for_all_agents(_gen, concurrency=3, task_name="Daily schedule")


async def _run_monthly_overview_refresh():
    async def _refresh(agent):
        seven_dim = get_seven_dim(agent)
        overview = await generate_life_overview(agent.name, seven_dim)
        await save_life_overview(agent.id, overview)

    await _run_for_all_agents(_refresh, concurrency=2, task_name="Monthly overview")


async def _run_schedule_review():
    await _run_for_all_agents(
        lambda a: review_daily_schedule(a.id, a.userId, a.name),
        concurrency=3, task_name="Schedule review",
    )


async def _run_proactive_scan():
    """扫描所有Agent-用户对，尝试发送主动消息。如有WS连接则直接推送。"""
    from app.services.ws_manager import manager

    async def _try_proactive(agent):
        # 跳过睡眠状态
        schedule = await get_cached_schedule(agent.id)
        if schedule:
            status = get_current_status(schedule)
            if status.get("status") == "sleep":
                return

        msg = await generate_proactive_message(agent.userId, agent.id)
        if msg:
            # 尝试通过 WS 推送主动消息
            sent = await manager.send_to_user(
                agent.userId, "proactive",
                {"text": msg, "agent_id": agent.id},
            )
            if sent:
                logger.info(f"Proactive message pushed via WS for agent {agent.id}")
            else:
                logger.info(f"Proactive message saved (no WS) for agent {agent.id}")

    await _run_for_all_agents(_try_proactive, concurrency=3, task_name="Proactive scan")


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


async def _run_blacklist_scan():
    """5B.3: 扫描已过期的拉黑计时器并自动解除。"""
    try:
        count = await scan_blacklist_expiry()
        if count > 0:
            logger.info(f"Blacklist scan: lifted {count} blacklists")
    except Exception as e:
        logger.warning(f"Blacklist scan failed: {e}")


async def _run_emotion_decay():
    from app.services.trait_model import get_seven_dim

    async def _decay(agent):
        seven_dim = get_seven_dim(agent)
        await decay_emotion_toward_baseline(agent.id, agent.personality or {}, seven_dim)

    await _run_for_all_agents(
        _decay, concurrency=5, task_name="Emotion decay",
    )


async def _run_trigger_scan():
    """§9.5: 扫描到期的时间触发器。"""
    try:
        await scan_triggers()
    except Exception as e:
        logger.warning(f"Trigger scan failed: {e}")


async def _run_holiday_check():
    """§9.6: 检查今天是否节假日，若是则创建祝福触发器。"""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from app.config import settings

    try:
        now = datetime.now(ZoneInfo(settings.schedule_timezone))
        holiday = is_holiday(now.date())
        if holiday:
            await create_holiday_triggers(now.date().isoformat(), holiday.name)
            logger.info(f"Holiday triggers created for {holiday.name}")
    except Exception as e:
        logger.warning(f"Holiday check failed: {e}")


async def _run_birthday_scan():
    """§9.6: 每周扫描用户记忆中的生日信息，创建生日触发器。"""
    try:
        await scan_birthday_memories()
    except Exception as e:
        logger.warning(f"Birthday scan failed: {e}")


async def _run_aggregation_scan():
    """Scan aggregation windows and due delayed replies, then deliver asynchronously."""
    from app.services.chat_service import stream_chat_response
    from app.services.ws_manager import manager
    from app.api.ws import stream_to_ws
    from app.db import db

    try:
        expired = await scan_expired()
        for user_id, combined_text, conv_id, reply_context, latest_message_id in expired:
            delay_seconds = float((reply_context or {}).get("delay_seconds", 0.0) or 0.0)
            await enqueue_delayed_message(
                conv_id,
                {
                    "conversation_id": conv_id,
                    "agent_id": None,
                    "user_id": user_id,
                    "message": combined_text,
                    "message_id": latest_message_id,
                    "reply_context": reply_context,
                },
                delay_seconds,
            )
            # 12E: Update frontend after aggregation window ends
            ws = manager.get(conv_id)
            if ws:
                if delay_seconds > 5:
                    await ws.send_json({"type": "delay", "data": {"duration": delay_seconds}})
                await ws.send_json({"type": "pending", "data": {"status": "queued", "delay": delay_seconds}})

        due_conversations = await scan_due_delayed_messages()
        for conv_id, payloads in due_conversations:
            merged = merge_delayed_payloads(payloads)
            if not merged:
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

            ws = manager.get(conv_id)
            if ws:
                await stream_to_ws(ws, gen)
                logger.debug(f"Delayed reply pushed via WS for conv={conv_id[:8]}")
            else:
                async for _ in gen:
                    pass
                logger.debug(f"Delayed reply consumed silently for conv={conv_id[:8]}")
    except Exception as e:
        logger.warning(f"Aggregation scan failed: {e}")


def shutdown_scheduler():
    """Shutdown the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Job scheduler stopped")
