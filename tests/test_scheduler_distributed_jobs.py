from __future__ import annotations

from contextlib import asynccontextmanager
import inspect
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
    source = inspect.getsource(scheduler_mod.setup_scheduler)

    assert "scheduler.add_job(\n        run_weekly_reflection," not in source
    assert "_run_weekly_reflection," in source
