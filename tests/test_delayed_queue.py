"""Tests for delayed reply queue helpers."""

from unittest.mock import AsyncMock

import pytest

from app.services import delayed_queue
from app.services.delayed_queue import merge_delayed_payloads


def test_merge_delayed_payloads_combines_all_messages_and_keeps_first_context():
    merged = merge_delayed_payloads([
        {
            "conversation_id": "conv-1",
            "agent_id": "agent-1",
            "user_id": "user-1",
            "message": "第一句",
            "message_id": "msg-1",
            "reply_context": {
                "received_at": "2026-03-19T00:00:00+00:00",
                "received_status": {"activity": "工作", "status": "busy"},
                "delay_reason": "schedule_busy",
            },
        },
        {
            "conversation_id": "conv-1",
            "agent_id": "agent-1",
            "user_id": "user-1",
            "message": "第二句",
            "message_id": "msg-2",
            "reply_context": {
                "received_at": "2026-03-19T00:00:10+00:00",
                "received_status": {"activity": "自由时间", "status": "idle"},
                "delay_reason": "conversation_mode",
            },
        },
    ])

    assert merged is not None
    assert merged["user_message"] == "第一句 第二句"
    assert merged["user_message_id"] == "msg-2"
    assert merged["queued_messages"] == ["第一句", "第二句"]
    assert merged["reply_context"]["received_status"]["activity"] == "工作"
    assert merged["reply_context"]["latest_received_at"] == "2026-03-19T00:00:10+00:00"


def test_merge_delayed_payloads_falls_back_to_last_non_empty_text():
    merged = merge_delayed_payloads([
        {
            "conversation_id": "conv-1",
            "agent_id": "agent-1",
            "user_id": "user-1",
            "message": "片段一 片段二",
            "message_id": "msg-1",
            "reply_context": None,
        },
        {
            "conversation_id": "conv-1",
            "agent_id": "agent-1",
            "user_id": "user-1",
            "message": "",
            "message_id": "msg-2",
            "reply_context": None,
        },
    ])

    assert merged is not None
    assert merged["user_message"] == "片段一 片段二"


@pytest.mark.asyncio
async def test_flush_due_delayed_messages_keeps_future_payloads(monkeypatch):
    redis = AsyncMock()
    redis.zrangebyscore.return_value = [
        '{"message":"先发","message_id":"msg-1","due_at":10}',
    ]
    redis.zrange.return_value = [("future-payload", 20.0)]
    monkeypatch.setattr(delayed_queue, "get_redis", AsyncMock(return_value=redis))

    payloads = await delayed_queue.flush_due_delayed_messages("conv-1", now=15.0)

    assert payloads == [{"message": "先发", "message_id": "msg-1", "due_at": 10}]
    redis.zrem.assert_awaited_once_with("delayed:msgs:conv-1", '{"message":"先发","message_id":"msg-1","due_at":10}')
    redis.zadd.assert_awaited_once_with("delayed:due", {"conv-1": 20.0})
    redis.delete.assert_not_called()


@pytest.mark.asyncio
async def test_flush_due_delayed_messages_clears_conversation_when_empty(monkeypatch):
    redis = AsyncMock()
    redis.zrangebyscore.return_value = [
        '{"message":"先发","message_id":"msg-1","due_at":10}',
    ]
    redis.zrange.return_value = []
    monkeypatch.setattr(delayed_queue, "get_redis", AsyncMock(return_value=redis))

    payloads = await delayed_queue.flush_due_delayed_messages("conv-1", now=15.0)

    assert payloads == [{"message": "先发", "message_id": "msg-1", "due_at": 10}]
    redis.delete.assert_awaited_once_with("delayed:msgs:conv-1")
    redis.zrem.assert_any_await("delayed:due", "conv-1")
