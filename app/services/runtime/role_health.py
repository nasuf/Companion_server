"""Bounded local process health, separate from business task success."""

import asyncio
from dataclasses import dataclass, field
import time


@dataclass
class RoleHealth:
    role: str
    initialized: bool = False
    closing: bool = False
    postgres: bool = False
    redis: bool = False
    fatal: bool = False
    last_probe: float = field(default=0)

    def ready(self) -> bool:
        return (
            self.initialized and not self.closing and not self.fatal
            and self.postgres and self.redis
            and time.monotonic() - self.last_probe < 35
        )

    def snapshot(self) -> dict:
        return {
            "role": self.role, "ready": self.ready(), "postgres": self.postgres,
            "redis": self.redis, "closing": self.closing, "fatal": self.fatal,
        }


async def probe_dependencies(health, postgres, redis):
    """A hanging query must stop readiness from being refreshed indefinitely."""
    async def check_db():
        try:
            await asyncio.wait_for(postgres.query_raw("SELECT 1"), timeout=4)
            return True
        except Exception:
            return False

    async def check_redis():
        try:
            return bool(await asyncio.wait_for(redis.ping(), timeout=4))
        except Exception:
            return False

    health.postgres, health.redis = await asyncio.gather(check_db(), check_redis())
    health.last_probe = time.monotonic()


async def cancel_and_drain(tasks, *, timeout=10):
    """Preserve leases on shutdown; never await an uncooperative handler forever."""
    for task in tasks:
        task.cancel()
    if not tasks:
        return
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    for task in done:
        if not task.cancelled():
            task.exception()
    if pending:
        from app.services.runtime.sql_job_contracts import WorkerStopRequired
        raise WorkerStopRequired("Runtime tasks did not stop within the shutdown deadline")
