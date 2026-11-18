"""遗忘后画像重建: 时间比较与调度注册 (无数据库)。"""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.services import portrait


def test_after_handles_missing_and_naive_times():
    now = datetime.now(UTC)
    naive = now.replace(tzinfo=None)
    assert portrait._after(None, now) is False
    assert portrait._after(now, None) is False
    assert portrait._after(naive + timedelta(seconds=1), now) is True
    assert portrait._after(now, naive + timedelta(seconds=1)) is False


def test_rebuild_job_is_registered_daily():
    from jobs import scheduler

    jobs = {j.id: j for j in scheduler.scheduler_job_definitions()}
    trigger = str(jobs["portrait_forget_rebuild"].trigger)
    assert "hour='4'" in trigger and "minute='40'" in trigger


@pytest.mark.asyncio
async def test_rebuild_job_runs_under_the_distributed_lock():
    from jobs import scheduler

    async def _run(name, ttl, fn, **_):
        assert name == "portrait_forget_rebuild"
        await fn()

    with patch.object(scheduler, "_run_distributed_job", AsyncMock(side_effect=_run)), \
         patch("app.services.portrait.rebuild_portraits_after_forget", AsyncMock()) as rebuild:
        await scheduler._run_portrait_forget_rebuild()
    rebuild.assert_awaited_once()
