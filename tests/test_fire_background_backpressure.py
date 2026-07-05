"""fire_background tracks in-flight tasks and warns (never drops) on overflow."""

from __future__ import annotations

import asyncio

import pytest

from app.services.runtime import tasks as tasks_mod


@pytest.mark.asyncio
async def test_inflight_tracked_and_drained():
    ev = asyncio.Event()

    async def _work():
        await ev.wait()

    start = tasks_mod.background_inflight_count()
    t1 = tasks_mod.fire_background(_work())
    t2 = tasks_mod.fire_background(_work())
    assert tasks_mod.background_inflight_count() == start + 2

    ev.set()
    await asyncio.gather(t1, t2)
    # done callback discards from the in-flight set
    assert tasks_mod.background_inflight_count() == start


@pytest.mark.asyncio
async def test_overflow_warns_but_runs_all(monkeypatch, caplog):
    from app.config import settings
    monkeypatch.setattr(settings, "background_task_max_concurrency", 2)
    # reset latch so the warning can fire in this test
    tasks_mod._overflow_warned = False

    ev = asyncio.Event()

    async def _work():
        await ev.wait()

    tasks: list[asyncio.Task] = []
    import logging
    with caplog.at_level(logging.WARNING, logger=tasks_mod.logger.name):
        for _ in range(5):  # exceed the hwm=2
            tasks.append(tasks_mod.fire_background(_work()))

    assert any("backlog high" in r.message for r in caplog.records)

    ev.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # all ran to completion (none dropped)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_failed_task_logged_and_removed_from_inflight():
    async def _boom():
        raise ValueError("boom")

    start = tasks_mod.background_inflight_count()
    t = tasks_mod.fire_background(_boom())
    with pytest.raises(ValueError):
        await t
    # allow the done callback to run
    await asyncio.sleep(0)
    assert tasks_mod.background_inflight_count() == start
