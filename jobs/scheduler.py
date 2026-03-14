"""Job scheduler for periodic tasks.

Uses APScheduler for:
- Daily: consolidation, importance decay
- Weekly: reflection, memory compression
- Monthly: memory compression (L2 -> L1)
"""

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.services.reflection import run_daily_reflection, run_weekly_reflection
from app.services.memory.compression import compress_weekly, compress_monthly
from app.services.portrait import update_portrait_weekly
from app.services.memory.self_memory import generate_daily_self_memories
from app.services.emotion import decay_emotion_toward_baseline

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


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
    """Update portraits for all active user-agent pairs."""
    from app.db import db  # deferred to avoid circular import at scheduler setup
    agents = await db.aiagent.find_many()
    sem = asyncio.Semaphore(3)

    async def _process(agent):
        async with sem:
            try:
                await update_portrait_weekly(agent.userId, agent.id)
            except Exception as e:
                logger.warning(f"Portrait update failed for agent {agent.id}: {e}")

    await asyncio.gather(*[_process(a) for a in agents], return_exceptions=True)


async def _run_daily_self_memories():
    """Generate daily self-memories for all active agents."""
    from app.db import db  # deferred to avoid circular import at scheduler setup
    agents = await db.aiagent.find_many()
    sem = asyncio.Semaphore(3)

    async def _process(agent):
        async with sem:
            try:
                await generate_daily_self_memories(
                    agent_id=agent.id,
                    user_id=agent.userId,
                    dialogue_summary=None,
                )
            except Exception as e:
                logger.warning(f"Self-memory generation failed for agent {agent.id}: {e}")

    await asyncio.gather(*[_process(a) for a in agents], return_exceptions=True)


async def _run_emotion_decay():
    """Decay all agents' emotions toward their personality baseline."""
    from app.db import db
    agents = await db.aiagent.find_many()
    sem = asyncio.Semaphore(5)

    async def _process(agent):
        async with sem:
            try:
                await decay_emotion_toward_baseline(agent.id, agent.personality or {})
            except Exception as e:
                logger.warning(f"Emotion decay failed for agent {agent.id}: {e}")

    await asyncio.gather(*[_process(a) for a in agents], return_exceptions=True)


def shutdown_scheduler():
    """Shutdown the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Job scheduler stopped")
