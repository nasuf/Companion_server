from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services.runtime import job_queue


class FakeJobRedis:
    def __init__(self):
        self.kv: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.lists: dict[str, list[str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, **_kwargs):
        if _kwargs.get("nx") and key in self.kv:
            return False
        self.kv[key] = str(value)
        return True

    async def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update({str(k): str(v) for k, v in mapping.items()})
        return 1

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def expire(self, *_args):
        return True

    async def lpush(self, key, value):
        values = self.lists.setdefault(key, [])
        values.insert(0, str(value))
        return len(values)

    async def rpop(self, key):
        values = self.lists.setdefault(key, [])
        if not values:
            return None
        return values.pop()

    async def zadd(self, key, mapping):
        zset = self.zsets.setdefault(key, {})
        for member, score in mapping.items():
            zset[str(member)] = float(score)
        return len(mapping)

    async def zrangebyscore(self, key, min_score, max_score):
        zset = self.zsets.setdefault(key, {})
        return [
            member
            for member, score in zset.items()
            if float(min_score) <= score <= float(max_score)
        ]

    async def zrem(self, key, *members):
        zset = self.zsets.setdefault(key, {})
        removed = 0
        for member in members:
            removed += 1 if zset.pop(str(member), None) is not None else 0
        return removed


@pytest.mark.asyncio
async def test_runtime_job_queue_runs_registered_handler():
    redis = FakeJobRedis()
    calls: list[dict] = []

    async def handler(payload):
        calls.append(payload)

    job_queue.register_job_handler("test.job", handler)
    with (
        patch.object(job_queue, "get_redis", AsyncMock(return_value=redis)),
        patch.object(job_queue, "distributed_lock", _lock_acquired),
    ):
        job_id = await job_queue.enqueue_runtime_job("test.job", {"x": 1})
        processed = await job_queue.process_runtime_jobs(max_jobs=1)

    assert processed == 1
    assert calls == [{"x": 1}]
    assert redis.hashes[job_queue._job_key(job_id)]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_runtime_job_queue_retries_then_dead_letters():
    redis = FakeJobRedis()

    async def handler(_payload):
        raise RuntimeError("boom")

    job_queue.register_job_handler("test.fail", handler)
    with (
        patch.object(job_queue, "get_redis", AsyncMock(return_value=redis)),
        patch.object(job_queue, "distributed_lock", _lock_acquired),
    ):
        job_id = await job_queue.enqueue_runtime_job("test.fail", {"x": 1}, max_attempts=1)
        await job_queue.process_runtime_jobs(max_jobs=1)

    record = redis.hashes[job_queue._job_key(job_id)]
    assert record["status"] == "dead_letter"
    assert "boom" in record["last_error"]
    assert redis.lists[job_queue._DLQ_KEY][0] == job_id


@pytest.mark.asyncio
async def test_runtime_job_queue_idempotency_returns_existing_job():
    redis = FakeJobRedis()

    with patch.object(job_queue, "get_redis", AsyncMock(return_value=redis)):
        first = await job_queue.enqueue_runtime_job(
            "test.idem",
            {"x": 1},
            idempotency_key="same",
        )
        second = await job_queue.enqueue_runtime_job(
            "test.idem",
            {"x": 2},
            idempotency_key="same",
        )

    assert second == first
    payloads = [
        json.loads(record["payload"])
        for record in redis.hashes.values()
    ]
    assert payloads == [{"x": 1}]


class _lock_acquired:
    def __init__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        return True

    async def __aexit__(self, *_args):
        return False
