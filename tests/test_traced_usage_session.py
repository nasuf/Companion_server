"""Real scoped trace/usage lifecycle with synthetic storage and no provider calls."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import settings
from app.services.chat import local_tracer, tracing
from app.services.llm import usage_tracker, usage_repo


@pytest.fixture
def storage(monkeypatch):
    import app.db
    fake = MagicMock()
    fake.tracerun.create = AsyncMock()
    fake.tracerun.upsert = AsyncMock()
    fake.tracerun.update = AsyncMock()
    monkeypatch.setattr(app.db, "db", fake)
    monkeypatch.setattr(settings, "trace_backend", "local")
    write_usage = AsyncMock()
    monkeypatch.setattr(usage_repo, "write_usage_row", write_usage)
    return fake, write_usage


async def settle(fake):
    # Wait for the fire-and-forget completion, not a fixed timing assumption.
    for _ in range(100):
        if fake.tracerun.update.await_count:
            return
        await asyncio.sleep(.001)
    raise AssertionError("Trace completion did not persist")


def metadata(fake):
    return fake.tracerun.update.await_args.kwargs["data"]["extraJson"].data["metadata"]


@pytest.mark.parametrize("scope", ["proactive", "post_process", "agent_creation",
                                    "schedule_cron", "offline", "music", "chat_media", "chat"])
async def test_scoped_operation_records_identity_before_usage_flush(storage, scope):
    fake, write_usage = storage
    async with usage_tracker.traced_usage_session(
        name="synthetic-operation", scope=scope, conversation_id="c-1",
        agent_id="a-1", user_id="u-1",
    ) as tracer:
        assert usage_tracker.current_scope() == scope
        usage_tracker.record("synthetic", 10, 2)
    await settle(fake)
    assert metadata(fake) == {"usage_scope": scope, "usage_expected": True}
    write_usage.assert_awaited_once()
    assert write_usage.await_args.kwargs["trace_id"] == tracer.trace_id
    assert write_usage.await_args.kwargs["scope"] == scope
    assert not usage_tracker.has_session()
    assert usage_tracker.current_scope() == ""
    assert local_tracer._local_trace_handler.get() is None


async def test_zero_model_operation_is_explicit(storage):
    fake, write_usage = storage
    async with usage_tracker.traced_usage_session(
        name="rule-only", scope="proactive", conversation_id="c-1",
        agent_id="a-1", user_id="u-1",
    ):
        pass
    await settle(fake)
    assert metadata(fake) == {"usage_scope": "proactive", "usage_expected": False}
    write_usage.assert_not_awaited()


async def test_real_langchain_callback_stays_under_the_scoped_root(storage):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    fake, _ = storage
    async with usage_tracker.traced_usage_session(
        name="topic-judge", scope="proactive", conversation_id="c-1",
        agent_id="a-1", user_id="u-1",
    ) as tracer:
        result = await FakeListChatModel(responses=["已完结"]).ainvoke("synthetic prompt")
        assert result.content == "已完结"
        usage_tracker.record("synthetic", 10, 2)
    await settle(fake)
    rows = [call.kwargs["data"] for call in fake.tracerun.create.await_args_list]
    roots = [row for row in rows if row["name"] == "chat_request"]
    models = [row for row in rows if row["runType"] == "llm"]
    assert len(roots) == len(models) == 1
    assert roots[0]["extraJson"].data["metadata"] == {"usage_scope": "proactive"}
    assert models[0]["traceId"] == models[0]["parentId"] == tracer.trace_id


@pytest.mark.parametrize("error", [RuntimeError("PRIVATE_SECRET"), asyncio.CancelledError()])
async def test_failure_and_cancellation_do_not_become_successful_roots(storage, error):
    fake, write_usage = storage
    with pytest.raises(type(error)):
        async with usage_tracker.traced_usage_session(
            name="failed-operation", scope="proactive", conversation_id="c-1",
            agent_id="a-1", user_id="u-1",
        ):
            usage_tracker.record_runtime_event(result="timeout", latency_ms=10)
            raise error
    await settle(fake)
    data = fake.tracerun.update.await_args.kwargs["data"]
    assert data["status"] == "error"
    assert data["error"] == type(error).__name__
    assert "PRIVATE_SECRET" not in str(data)
    assert metadata(fake)["usage_expected"] is True
    write_usage.assert_awaited_once()
    assert not usage_tracker.has_session()
    assert local_tracer._local_trace_handler.get() is None


async def test_auxiliary_scope_restores_parent_chat_accumulator_and_trace(storage):
    fake, write_usage = storage
    token = usage_tracker.start_session()
    parent = tracing.create_tracer("chat", "c-1", executor="langgraph", graph_version="chat-g01-v1").enter()
    parent_handler = local_tracer._local_trace_handler.get()
    try:
        usage_tracker.record("chat-model", 100, 20)
        async with usage_tracker.traced_usage_session(
            name="child-operation", scope="proactive", conversation_id="c-1",
            agent_id="a-1", user_id="u-1",
        ) as child:
            usage_tracker.record("auxiliary-model", 10, 2)
        assert child.trace_id != parent.trace_id
        assert local_tracer._local_trace_handler.get() is parent_handler
        summary = usage_tracker.flush_session(token)
        token = None
        assert summary["call_count"] == 1
        assert summary["tokens_by_model"] == {"chat-model": {"input": 100, "output": 20, "cached_input": 0}}
        assert write_usage.await_args.kwargs["summary"]["input_tokens"] == 10
    finally:
        if token is not None:
            usage_tracker.flush_session(token)
        parent.close()
    await settle(fake)


def test_scope_metadata_is_identical_for_langsmith_backend(monkeypatch):
    monkeypatch.setattr(settings, "trace_backend", "langsmith")
    tracer = tracing.create_tracer("operation", "c-1", usage_scope="proactive")
    assert tracer._execution_metadata == {"usage_scope": "proactive"}
    main = tracing.create_tracer("chat", "c-1", executor="legacy")
    assert main._execution_metadata == {"usage_scope": "chat", "executor": "legacy", "checkpoint_enabled": False}
