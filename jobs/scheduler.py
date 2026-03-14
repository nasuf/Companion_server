"""Job scheduler for periodic tasks.

Uses APScheduler for:
- Daily: consolidation, importance decay
- Weekly: reflection, memory compression
- Monthly: memory compression (L2 -> L1)
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.services.reflection import run_daily_reflection, run_weekly_reflection
from app.services.memory.compression import compress_weekly, compress_monthly
from app.services.portrait import update_portrait_weekly
from app.services.memory.self_memory import generate_daily_self_memories
from app.services.emotion import decay_emotion_toward_baseline

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

    await asyncio.gather(*[_process(a) for a in agents], return_exceptions=True)


def setup_scheduler():
    """Configure and start the job scheduler."""
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

    # Weekly portrait update on Sunday at 3 AM
    scheduler.add_job(
        _run_weekly_portraits,
        "cron",
        day_of_week="sun",
        hour=3,
        minute=0,
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

    # Emotion decay every 5 minutes
    scheduler.add_job(
        _run_emotion_decay,
        "interval",
        minutes=5,
        id="emotion_decay",
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


async def _run_emotion_decay():
    await _run_for_all_agents(
        lambda a: decay_emotion_toward_baseline(a.id, a.personality or {}),
        concurrency=5, task_name="Emotion decay",
    )


def shutdown_scheduler():
    """Shutdown the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Job scheduler stopped")
