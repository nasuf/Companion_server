"""Trace 写入有上限: DB 慢或挂掉时不无限堆积任务, 超出部分丢弃并计数; 退出前等在途写入."""

from __future__ import annotations

import asyncio

import pytest

from app.services.chat import local_tracer as lt


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(lt, "_pending_writes", set())
    monkeypatch.setattr(lt, "_dropped_unreported", 0)
    monkeypatch.setattr(lt, "_drop_report", None)


def _capture_counts(monkeypatch):
    counts: list[tuple[str, int]] = []

    async def note(kind, error, count=1):
        counts.append((kind, count))

    monkeypatch.setattr(lt, "_note_write_failure", note)
    return counts


@pytest.mark.asyncio
async def test_writes_past_the_cap_are_dropped_and_counted_once(monkeypatch):
    counts = _capture_counts(monkeypatch)
    monkeypatch.setattr(lt, "MAX_PENDING_WRITES", 2)
    gate = asyncio.Event()
    ran: list[int] = []

    async def write(i):
        await gate.wait()
        ran.append(i)

    tasks = [lt._fire(write(i)) for i in range(5)]
    assert [t is not None for t in tasks] == [True, True, False, False, False]
    await asyncio.sleep(0)
    assert counts == [("dropped", 3)]
    gate.set()
    await lt.drain_trace_writes(timeout=1)
    assert sorted(ran) == [0, 1]
    assert not lt._pending_writes


@pytest.mark.asyncio
async def test_drain_cancels_writes_that_outlive_the_timeout(monkeypatch):
    cancelled = asyncio.Event()

    async def stuck():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    lt._fire(stuck())
    await lt.drain_trace_writes(timeout=0.05)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_drain_without_pending_writes_returns_immediately():
    await lt.drain_trace_writes(timeout=0.01)


def test_fire_without_loop_closes_the_coroutine():
    async def write():
        raise AssertionError("must not run")

    coro = write()
    assert lt._fire(coro) is None
    assert coro.cr_frame is None  # closed, so no "never awaited" warning


@pytest.mark.asyncio
async def test_drop_counter_failure_is_swallowed(monkeypatch):
    class _Redis:
        def pipeline(self):
            raise RuntimeError("redis down")

    async def get_redis():
        return _Redis()

    monkeypatch.setattr("app.redis_client.get_redis", get_redis)
    await lt._note_write_failure(lt.TRACE_WRITE_DROPPED_KIND, None, 4)


@pytest.mark.asyncio
async def test_drain_waits_for_an_in_flight_drop_report(monkeypatch):
    reported = asyncio.Event()

    async def slow_note(kind, error, count=1):
        await asyncio.sleep(0.05)
        reported.set()

    monkeypatch.setattr(lt, "_note_write_failure", slow_note)
    monkeypatch.setattr(lt, "MAX_PENDING_WRITES", 0)

    async def write():
        raise AssertionError("must not run")

    assert lt._fire(write()) is None
    await lt.drain_trace_writes(timeout=1)
    assert reported.is_set()
