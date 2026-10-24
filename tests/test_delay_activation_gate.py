"""U8 release gate: real Redis, no models or production traffic."""
import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import pytest
from redis.asyncio import Redis

from app.services.interaction import delayed_queue as q


@pytest.fixture
async def isolated(monkeypatch, unit_media_scope):
    url = os.environ.get("RUNTIME_JOB_TEST_REDIS_URL", "")
    if not url:
        pytest.skip("explicit isolated Redis required")
    assert urlsplit(url).hostname in {"127.0.0.1", "localhost"}
    assert os.environ.get("APP_ENV") != "production"
    redis = Redis.from_url(url, decode_responses=True)
    prefix = "test:delay-gate:" + uuid.uuid4().hex + ":"
    for field in ("_DELAYED_LIST_KEY", "_DELAYED_ZSET_KEY", "_LOCK_KEY", "_INFLIGHT_KEY"):
        monkeypatch.setattr(q, field, prefix + getattr(q, field))
    monkeypatch.setattr(q, "get_redis", AsyncMock(return_value=redis))
    try:
        yield redis
    finally:
        keys = [key async for key in redis.scan_iter(match=prefix + "*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


def payload(text="第一句", ident="msg-1"):
    return {"conversation_id": "conv", "user_id": "user", "agent_id": "agent",
            "message": text, "message_id": ident, "reply_context": {"delay_seconds": 0}}


async def test_locked_conversation_must_remain_queued(isolated, monkeypatch):
    from jobs import scheduler
    monkeypatch.setattr(scheduler, "scan_due_user_turns", AsyncMock(return_value=[]))
    await q.enqueue_or_append_delayed("conv", payload(), 0)
    token = await q.try_lock_conversation("conv")
    assert token
    try:
        await scheduler._run_aggregation_scan_body()
        assert await isolated.zcard(q._DELAYED_LIST_KEY.format(cid="conv")) == 1
        assert await q.has_pending_delayed_messages("conv")
    finally:
        await q.unlock_conversation("conv", token)


async def test_enqueue_error_must_not_send_queued(monkeypatch, unit_media_scope):
    from app.api.realtime import ws
    broken = AsyncMock()
    broken.eval.side_effect = RuntimeError("isolated write rejected")
    monkeypatch.setattr(q, "get_redis", AsyncMock(return_value=broken))
    socket = AsyncMock()
    await ws._queue_reply_or_error(socket, conversation_id="conv", agent=SimpleNamespace(id="agent"),
        user_id="user", user_message="hello", user_message_id="msg", reply_context={"delay_seconds": 3})
    events = [c.args[0] for c in socket.send_json.await_args_list]
    assert any(e["type"] == "error" for e in events)
    assert not any(e["type"] == "pending" for e in events)


async def test_rapid_messages_keep_arrival_order_and_merge_once(isolated):
    await q.enqueue_or_append_delayed("conv", payload("甲", "a"), 0)
    await q.enqueue_or_append_delayed("conv", payload("乙", "b"), 0)
    first, second = await asyncio.gather(q.scan_due_delayed_messages(), q.scan_due_delayed_messages())
    all_batches = first + second
    assert len(all_batches) == 1
    merged = q.merge_delayed_payloads(all_batches[0][1])
    assert merged["queued_messages"] == ["甲", "乙"]
    assert merged["reply_context"]["turn_message_ids"] == ["a", "b"]
