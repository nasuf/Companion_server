"""Job scheduler for periodic tasks.

Uses APScheduler for:
- Daily: consolidation, importance decay
- Weekly: reflection, memory compression
- Monthly: memory compression (L2 -> L1)
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.services.reflection import run_daily_reflection, run_weekly_reflection
from app.services.memory.compression import compress_weekly, compress_monthly

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

    scheduler.start()
    logger.info("Job scheduler started")


def shutdown_scheduler():
    """Shutdown the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Job scheduler stopped")
