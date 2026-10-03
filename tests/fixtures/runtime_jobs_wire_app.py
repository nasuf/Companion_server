"""Local synthetic runtime API, mounted into a candidate image for wire checks.

This fixture is excluded from the production Docker image. It creates no
database/model traffic and requires a disposable Redis service named test-redis.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin.runtime_jobs import router
from app.redis_client import get_redis
from app.services.auth import create_jwt
from app.services.runtime import job_queue as q

if (
    os.environ.get("APP_ENV") != "test"
    or urlsplit(os.environ.get("REDIS_URL", "")).hostname != "test-redis"
):
    raise RuntimeError("Synthetic wire fixture requires test mode and isolated test-redis")

for field in (
    "_READY_KEY", "_DELAYED_KEY", "_RUNNING_KEY", "_DLQ_KEY",
    "_SUCCEEDED_KEY", "_JOB_KEY_PREFIX", "_IDEMP_KEY_PREFIX",
):
    setattr(q, field, "n04-wire:" + getattr(q, field))

ids: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = await get_redis()

    async def handler(payload: dict) -> None:
        await redis.incr("n04-wire:synthetic-calls")

    q.register_job_handler("synthetic.job", handler)
    for name, status in (
        ("retryable", "dead_letter"), ("resolvable", "dead_letter"),
        ("completed", "succeeded"), ("active", "running"),
    ):
        jid = await q.enqueue_runtime_job("synthetic.job", {"synthetic": True, "name": name})
        ids[name] = jid
        await redis.lrem(q._READY_KEY, 0, jid)
        await redis.hset(q._job_key(jid), mapping={"status": status, "attempts": "1"})
        if status == "dead_letter":
            await redis.lpush(q._DLQ_KEY, jid)
        elif status == "succeeded":
            await redis.lpush(q._SUCCEEDED_KEY, jid)
        elif status == "running":
            await redis.zadd(q._RUNNING_KEY, {jid: int(time.time())})
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5176"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(router)


@app.get("/__test__/config")
async def config():
    return {"token": create_jwt("synthetic-admin", "admin"), "ids": ids}


@app.get("/__test__/state")
async def state():
    redis = await get_redis()
    return {
        "jobs": {name: await q.inspect_runtime_job(jid) for name, jid in ids.items()},
        "calls": int(await redis.get("n04-wire:synthetic-calls") or 0),
        "ready": await redis.lrange(q._READY_KEY, 0, -1),
    }


@app.post("/__test__/run")
async def run():
    await q.process_runtime_jobs(max_jobs=20)
    redis = await get_redis()
    # Inject stale duplicates of a new success and an existing success.
    await redis.lpush(q._READY_KEY, ids["retryable"], ids["completed"])
    await q.process_runtime_jobs(max_jobs=20)
    return await state()
