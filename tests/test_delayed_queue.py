"""Tests for delayed reply queue helpers."""

from unittest.mock import AsyncMock

import pytest

from app.services.runtime import delayed_queue
from app.services.interaction.delayed_queue import merge_delayed_payloads


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
async def test_flush_due_delayed_messages_returns_due_payloads(monkeypatch):
    """Lua-based flush returns due payloads atomically."""
    redis = AsyncMock()
    redis.eval.return_value = [
        '{"message":"先发","message_id":"msg-1","due_at":10}',
    ]
    monkeypatch.setattr(delayed_queue, "get_redis", AsyncMock(return_value=redis))

    payloads = await delayed_queue.flush_due_delayed_messages("conv-1", now=15.0)

    assert payloads == [{"message": "先发", "message_id": "msg-1", "due_at": 10}]
    redis.eval.assert_awaited_once()
    # Verify Lua script receives correct keys and args
    call_args = redis.eval.call_args
    assert call_args[0][1] == 2  # 2 keys
    assert call_args[0][4] == "conv-1"  # ARGV[1] = conversation_id


@pytest.mark.asyncio
async def test_flush_due_delayed_messages_returns_empty_when_nothing_due(monkeypatch):
    """Lua returns nil when no items are due."""
    redis = AsyncMock()
    redis.eval.return_value = None
    monkeypatch.setattr(delayed_queue, "get_redis", AsyncMock(return_value=redis))

    payloads = await delayed_queue.flush_due_delayed_messages("conv-1", now=15.0)

    assert payloads == []
