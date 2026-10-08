from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.services.runtime.distributed_lock import DistributedLockNotAcquired
from jobs import scheduler as scheduler_mod


@asynccontextmanager
async def _lock_acquired(*_args, **_kwargs):
    yield True


@asynccontextmanager
async def _lock_busy(*_args, **_kwargs):
    raise DistributedLockNotAcquired("busy")
    yield


@pytest.mark.asyncio
async def test_run_distributed_job_executes_body_when_lock_acquired():
    body = AsyncMock()

    with patch.object(scheduler_mod, "distributed_lock", _lock_acquired):
        await scheduler_mod._run_distributed_job("job-a", 30, body)

    body.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_distributed_job_skips_when_lock_busy():
    body = AsyncMock()

    with patch.object(scheduler_mod, "distributed_lock", _lock_busy):
        await scheduler_mod._run_distributed_job("job-a", 30, body)

    body.assert_not_called()


def test_weekly_reflection_is_registered_through_distributed_wrapper():
    jobs = {job.id: job for job in scheduler_mod.scheduler_job_definitions()}
    assert jobs["weekly_reflection"].func is scheduler_mod._run_weekly_reflection


def test_runtime_job_queue_is_registered():
    jobs = {job.id: job for job in scheduler_mod.scheduler_job_definitions()}
    assert jobs["runtime_job_queue"].func is scheduler_mod._run_runtime_job_queue
