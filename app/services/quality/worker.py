"""Single cluster executor, independent background task with a renewable lease."""

import asyncio
import logging
from datetime import timedelta
from app.config import settings
from app.services.quality import jobs, triggers, daily, regression

logger = logging.getLogger(__name__)
_task = None


async def execute(job):
    async def heartbeat():
        while True:
            await asyncio.sleep(30)
            await jobs.renew(job["id"], job["lease_token"])

    async def work():
        if job["kind"] == "daily":
            await daily.run(job)
        else:
            await regression.run(job)

    heart = asyncio.create_task(heartbeat())
    run = asyncio.create_task(work())
    try:
        done, _ = await asyncio.wait([heart, run], return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except BaseException as exc:
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        try:
            await jobs.finish(
                job,
                "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed",
                {"error_type": type(exc).__name__},
            )
        except jobs.LostLease:
            pass
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.warning("Quality worker stopped: %s", type(exc).__name__)
    finally:
        heart.cancel()
        await asyncio.gather(heart, return_exceptions=True)


async def tick():
    global _task
    if not settings.quality_enabled:
        return
    if _task is not None and not _task.done():
        return
    # Polling is leased separately so two API workers do not race the cursor.
    from app.db import db

    async with db.tx(timeout=timedelta(seconds=30)) as tx:
        won = await tx.query_raw("SELECT pg_try_advisory_xact_lock(731291409) AS won")
        if won[0]["won"]:
            await triggers.poll()
    job = await jobs.claim()
    if job:
        _task = asyncio.create_task(execute(job))


async def stop():
    if _task and not _task.done():
        _task.cancel()
        await asyncio.gather(_task, return_exceptions=True)
