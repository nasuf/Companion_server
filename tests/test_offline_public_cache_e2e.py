"""Two independent Python workers → real Redis → exactly one public load."""

import asyncio
import multiprocessing
import os
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from redis.asyncio import Redis


def worker(redis_url, namespace, queue):
    async def run():
        from app.services.offline.public_cache import public_cached

        redis = Redis.from_url(redis_url, decode_responses=True)

        async def redis_factory():
            return redis

        async def load():
            calls = await redis.incr(namespace + ":calls")
            while not await redis.get(namespace + ":release"):
                await asyncio.sleep(0.02)
            return {"places": [{"name": "小岛咖啡", "load": calls}]}

        try:
            await redis.incr(namespace + ":ready")
            result = await public_cached(
                namespace,
                ["镇江市", "咖啡"],
                load,
                valid=lambda v: isinstance(v, dict),
                timeout=8,
                redis_factory=redis_factory,
            )
            queue.put(result)
        finally:
            await redis.aclose()

    asyncio.run(run())


async def test_independent_workers_share_one_redis_lease():
    redis_url = os.getenv("PROACTIVE_E2E_REDIS_URL")
    if not redis_url:
        pytest.skip("isolated Redis URL required")
    assert urlsplit(redis_url).hostname in {"localhost", "127.0.0.1"}
    redis = Redis.from_url(redis_url, decode_responses=True)
    namespace = "e2e-public-workers-" + uuid4().hex
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=worker, args=(redis_url, namespace, queue)) for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        async with asyncio.timeout(15):
            while await redis.get(namespace + ":ready") != "2":
                await asyncio.sleep(0.05)
        await redis.set(namespace + ":release", "1")
        for process in processes:
            await asyncio.to_thread(process.join, 10)
            assert process.exitcode == 0
        results = [await asyncio.to_thread(queue.get, True, 1) for _ in range(2)]
        assert results == [{"places": [{"name": "小岛咖啡", "load": 1}]}] * 2
        assert await redis.get(namespace + ":calls") == "1"
        assert not [
            key
            async for key in redis.scan_iter(
                match="offline:public:v1:" + namespace + ":*:lock"
            )
        ]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, 3)
        keys = [key async for key in redis.scan_iter(match="*" + namespace + "*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
        queue.close()
