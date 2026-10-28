"""UXM01.08: real asyncio cancellation, without external provider calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.llm import usage_repo, usage_tracker, resilience
from app.services.chat import local_tracer


async def wait_writes():
    if usage_repo._pending_writes:
        await asyncio.gather(*tuple(usage_repo._pending_writes))


@pytest.mark.parametrize("phase", ["during_call", "during_write"])
async def test_cancellation_preserves_one_usage_row(monkeypatch, phase):
    started, release, calling = asyncio.Event(), asyncio.Event(), asyncio.Event()
    rows = []

    async def write(**kw):
        started.set()
        await release.wait()
        rows.append(kw)

    monkeypatch.setattr(usage_repo, "write_usage_row", write)

    async def turn():
        async with usage_tracker.usage_session(
            scope="chat",
            conversation_id="c",
            agent_id="a",
            user_id="u",
            trace_id_provider=lambda: "trace-123456",
        ):
            usage_tracker.record("synthetic", 100, 20)
            calling.set()
            if phase == "during_call":
                await asyncio.Event().wait()

    task = asyncio.create_task(turn())
    await calling.wait()
    if phase == "during_call":
        task.cancel()
        await started.wait()
    else:
        await started.wait()
    task.cancel()  # repeated cancellation during finally must not abort INSERT
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await wait_writes()
    assert len(rows) == 1
    assert rows[0]["trace_id"] == "trace-123456"
    assert rows[0]["summary"]["input_tokens"] == 100


async def test_write_error_does_not_change_reply_or_leak_error(monkeypatch, caplog):
    monkeypatch.setattr(
        usage_repo, "write_usage_row", AsyncMock(side_effect=RuntimeError("SECRET"))
    )
    replies = []
    async with usage_tracker.usage_session(
        scope="chat",
        conversation_id="c",
        agent_id=None,
        user_id="u",
        trace_id_provider=lambda: "12345678-private",
    ):
        usage_tracker.record("synthetic", 10, 2)
        replies.append("unchanged reply")
    assert replies == ["unchanged reply"]
    assert "scope=chat trace=12345678 type=RuntimeError" in caplog.text
    assert "SECRET" not in caplog.text and "private" not in caplog.text


async def test_write_timeout_is_bounded(monkeypatch, caplog):
    cancelled = asyncio.Event()

    async def write(**kw):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(usage_repo, "write_usage_row", write)
    monkeypatch.setattr(usage_repo, "WRITE_TIMEOUT_SECONDS", 0.01)
    await usage_repo.persist_usage(scope="post_process", trace_id="t")
    assert cancelled.is_set()
    assert "TimeoutError" in caplog.text
    await asyncio.sleep(0)
    assert not usage_repo._pending_writes


@pytest.mark.parametrize(
    "first_usage", [None, {"input_tokens": 100, "output_tokens": 2}]
)
@pytest.mark.parametrize("before_first", [False, True])
async def test_cancelled_stream_records_metering_uncertainty(
    monkeypatch, first_usage, before_first
):
    resilience.reset_breakers_for_testing()
    entered, closed = asyncio.Event(), asyncio.Event()
    rows = []

    async def factory():
        try:
            entered.set()
            if before_first:
                await asyncio.Event().wait()
            yield SimpleNamespace(content="first", usage_metadata=first_usage)
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def write(**kw):
        rows.append(kw)

    monkeypatch.setattr(usage_repo, "write_usage_row", write)

    async def turn():
        async with usage_tracker.usage_session(
            scope="chat", conversation_id="c", agent_id=None, user_id="u"
        ):
            async for _ in resilience._stream_provider(
                factory,
                provider="synthetic",
                model_name="synthetic/model",
                profile=resilience.CallProfile(
                    timeout_s=10,
                    max_retries=0,
                    retry_backoff_s=(),
                    first_chunk_timeout_s=10,
                ),
                op="test",
            ):
                pass

    task = asyncio.create_task(turn())
    await asyncio.wait_for(entered.wait(), 2)
    # Allow first __anext__ result to reach the consumer before cancelling.
    for _ in range(10):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_writes()
    assert closed.is_set() and len(rows) == 1
    summary = rows[0]["summary"]
    bucket = summary["tokens_by_model"]["synthetic/model"]
    unknown = before_first or first_usage is None
    assert bucket.get("unmetered_calls", 0) == int(unknown)
    assert summary["input_tokens"] == (0 if unknown else 100)
    assert summary["call_count"] == 1


async def test_background_usage_has_parent_trace_without_double_counting(monkeypatch):
    from app.services.chat import post_process

    rows = []

    async def write(**kw):
        rows.append(kw)

    async def process(*args, **kw):
        usage_tracker.record("synthetic", 40, 3)

    monkeypatch.setattr(usage_repo, "write_usage_row", write)
    monkeypatch.setattr(post_process, "_bg_user_emotion", process)
    async with usage_tracker.usage_session(
        scope="chat",
        conversation_id="c",
        agent_id=None,
        user_id="u",
        trace_id_provider=lambda: "same-trace",
    ):
        usage_tracker.record("synthetic", 100, 5)
        await post_process.run_post_process(
            user_id="u",
            agent_id=None,
            conversation_id="c",
            user_message="",
            user_message_id=None,
            full_response="",
            messages_dicts=[],
            skip_memory=True,
            trace_id="same-trace",
        )
    assert [r["scope"] for r in rows] == ["post_process", "chat"]
    assert {r["trace_id"] for r in rows} == {"same-trace"}
    assert sum(r["summary"]["input_tokens"] for r in rows) == 140
    assert sum(r["summary"]["output_tokens"] for r in rows) == 8


@pytest.mark.parametrize(
    "error,status",
    [
        (None, "success"),
        ("CancelledError()", "cancelled"),
        ("GeneratorExit()", "cancelled"),
        ("ValueError()", "error"),
    ],
)
async def test_run_end_status_matches_real_terminal_state(monkeypatch, error, status):
    import app.db

    fake = SimpleNamespace(tracerun=SimpleNamespace(upsert=AsyncMock()))
    monkeypatch.setattr(app.db, "db", fake)
    handler = local_tracer.LocalTraceHandler("trace", "root")
    run = SimpleNamespace(id="run", error=error, outputs=None)
    await local_tracer._write_run_end(handler, run)
    assert fake.tracerun.upsert.call_args.kwargs["data"]["update"]["status"] == status


async def test_shutdown_drains_owned_write_after_caller_cancel(monkeypatch):
    entered, release, complete = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def write(**kwargs):
        entered.set()
        await release.wait()
        complete.set()

    monkeypatch.setattr(usage_repo, "write_usage_row", write)
    caller = asyncio.create_task(
        usage_repo.persist_usage(scope="chat", trace_id="owned")
    )
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    shutdown = asyncio.create_task(usage_repo.drain_usage_writes())
    await asyncio.sleep(0)
    assert not shutdown.done()
    release.set()
    await shutdown
    assert complete.is_set()
    await asyncio.sleep(0)
    assert not usage_repo._pending_writes
    await usage_repo.drain_usage_writes()  # empty is harmless


async def test_shutdown_cancels_stuck_owned_writes_at_deadline(monkeypatch, caplog):
    cancelled = asyncio.Event()

    async def stuck():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(stuck())
    monkeypatch.setattr(usage_repo, "_pending_writes", {task})
    monkeypatch.setattr(usage_repo, "WRITE_TIMEOUT_SECONDS", 0.01)
    await usage_repo.drain_usage_writes()
    assert task.cancelled() and cancelled.is_set()
    assert "shutdown writes timed out count=1" in caplog.text
