"""Independent scheduler/background ASGI entry, with internal health only.

Run one process: python -m uvicorn jobs.runtime:app --host 0.0.0.0 --port 8000.
Choose APP_RUNTIME_ROLE=scheduler|background and set total LLM_PROCESS_COUNT.
No SQL consumers are registered here: business ingress remains gated separately.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import db, connect_db, disconnect_db
from app.redis_client import get_redis, close_redis, mark_redis_healthy
from app.middleware import configure_logging, configure_langsmith
from app.services.runtime.handler_registry import register_runtime_handlers
from app.services.runtime.role_health import RoleHealth, probe_dependencies, cancel_and_drain
from app.services.runtime.roles import require_role, validate_process_budget
from app.services.runtime.sql_job_contracts import WorkerStopRequired

logger = logging.getLogger(__name__)


def terminate_worker() -> None:
    # Exit the sole process in its container, including uncooperative handlers.
    # Container restart policy, rather than another in-process loop, owns recovery.
    os._exit(70)


async def monitor_dependencies(health, redis):
    while True:
        if health.role == "scheduler":
            from jobs.scheduler import scheduler
            if not scheduler.running:
                raise RuntimeError("Business scheduler stopped unexpectedly")
        await probe_dependencies(health, db, redis)
        mark_redis_healthy(health.redis)
        await asyncio.sleep(10)


async def consume_background_jobs(health):
    from app.services.runtime.job_queue import process_runtime_jobs
    from jobs.scheduler import _run_local_job

    while True:
        if health.ready():
            try:
                await _run_local_job("runtime_job_queue", lambda: process_runtime_jobs(max_jobs=20))
            except WorkerStopRequired:
                raise
            except Exception as error:
                logger.warning("Background scan failed (%s)", type(error).__name__)
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app):
    role = require_role(settings.app_runtime_role, api=False)
    validate_process_budget(settings)
    settings.validate_security_config()
    configure_logging()
    configure_langsmith()
    health = app.state.role_health = RoleHealth(role.name)
    tasks = []
    scheduler_started = False

    def supervise(task):
        if health.closing:
            return
        health.fatal = True
        error = None if task.cancelled() else task.exception()
        logger.critical("Runtime loop stopped (%s)", type(error).__name__ if error else "unexpected_exit")
        terminate_worker()

    try:
        await connect_db()
        redis = await get_redis()
        await probe_dependencies(health, db, redis)
        if not health.postgres or not health.redis:
            raise RuntimeError("Independent runtime dependencies unavailable")
        from app.services.runtime_config import load_caches, refresh_worker_config
        from app.services.schedule_domain.holiday_cache import reload as reload_holiday_cache

        await load_caches()
        await reload_holiday_cache()
        if role.consumes_redis_jobs:
            register_runtime_handlers()
        if role.starts_scheduler:
            from jobs.scheduler import setup_scheduler
            setup_scheduler(role="scheduler")
            scheduler_started = True
        health.initialized = True
        tasks.extend((
            asyncio.create_task(monitor_dependencies(health, redis), name="dependency-monitor"),
            asyncio.create_task(refresh_worker_config(), name="configuration-refresh"),
        ))
        if role.consumes_redis_jobs:
            tasks.append(asyncio.create_task(consume_background_jobs(health), name="background-consumer"))
        for task in tasks:
            task.add_done_callback(supervise)
        logger.info("Runtime role ready: %s", role.name)
        yield
    finally:
        health.closing = True
        from app.services.runtime.tasks import pending_background_tasks
        owned = set(tasks) | set(pending_background_tasks())
        if scheduler_started:
            from jobs.scheduler import shutdown_scheduler, active_scheduler_tasks
            owned.update(active_scheduler_tasks())
            shutdown_scheduler()
        try:
            await cancel_and_drain(owned)
        except WorkerStopRequired:
            logger.critical("Runtime shutdown deadline exceeded")
            terminate_worker()
        try:
            await asyncio.wait_for(disconnect_db(), timeout=5)
        finally:
            await asyncio.wait_for(close_redis(), timeout=5)


app = FastAPI(title="Companion internal runtime", lifespan=lifespan)


@app.get("/health")
async def health_check():
    health = getattr(app.state, "role_health", None)
    payload = health.snapshot() if health else {"ready": False}
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)
