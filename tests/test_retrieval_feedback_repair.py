"""Tests for routing retrieval correction feedback into memory repair."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@dataclass
class _Memory:
    id: str = "m1"
    userId: str = "u1"
    workspaceId: str | None = "w1"
    source: str = "user"
    level: int = 1
    content: str = "用户喜欢芒果"
    summary: str | None = "用户喜欢芒果"
    importance: float = 0.9
    mentionCount: int = 1
    isArchived: bool = False
    occurTime: datetime | None = None
    createdAt: datetime = datetime(2026, 5, 1)
    updatedAt: datetime = datetime(2026, 5, 2)


def _previous_assistant(metadata: dict):
    return SimpleNamespace(
        id="a1",
        role="assistant",
        content="我记得你喜欢芒果。",
        metadata=metadata,
    )


def _metadata() -> dict:
    return {
        "memory_retrieval_analysis": {
            "likely_used_count": 1,
            "items": [{"id": "m1", "text": "用户喜欢芒果", "likely_used": True}],
        },
        "memory_retrievals": [
            {"selected": [{"id": "m1", "text": "用户喜欢芒果"}]},
        ],
    }


@pytest.mark.asyncio
async def test_build_retrieval_feedback_conflict_uses_retrieved_memory(monkeypatch):
    from app.services.memory.interaction import retrieval_feedback

    monkeypatch.setattr(
        retrieval_feedback.memory_repo,
        "find_unique",
        AsyncMock(return_value=_Memory()),
    )

    result = await retrieval_feedback.build_retrieval_feedback_conflict(
        user_message="你记错了，我从来没说过我喜欢芒果。",
        previous_assistant=_previous_assistant(_metadata()),
        user_id="u1",
        workspace_id="w1",
    )

    assert result is not None
    conflict, feedback = result
    assert conflict["source"] == "retrieval_feedback"
    assert conflict["conflicting_memory_id"] == "m1"
    assert conflict["old_content"] == "用户喜欢芒果"
    assert feedback["repair_action"]["type"] == "confirmation_requested"


@pytest.mark.asyncio
async def test_build_retrieval_feedback_conflict_rejects_cross_workspace(monkeypatch):
    from app.services.memory.interaction import retrieval_feedback

    monkeypatch.setattr(
        retrieval_feedback.memory_repo,
        "find_unique",
        AsyncMock(return_value=_Memory(workspaceId="other")),
    )

    result = await retrieval_feedback.build_retrieval_feedback_conflict(
        user_message="你记错了，我从来没说过我喜欢芒果。",
        previous_assistant=_previous_assistant(_metadata()),
        user_id="u1",
        workspace_id="w1",
    )

    assert result is None


@pytest.mark.asyncio
async def test_build_retrieval_feedback_conflict_rejects_ai_memory(monkeypatch):
    from app.services.memory.interaction import retrieval_feedback

    monkeypatch.setattr(
        retrieval_feedback.memory_repo,
        "find_unique",
        AsyncMock(return_value=_Memory(
            source="ai",
            content="29岁生日前夜突然失眠",
            summary="29岁生日前夜突然失眠",
        )),
    )

    result = await retrieval_feedback.build_retrieval_feedback_conflict(
        user_message="不对啊，我到底多大你不记得了吗，我跟你说过的",
        previous_assistant=_previous_assistant(_metadata()),
        user_id="u1",
        workspace_id="w1",
    )

    assert result is None


@pytest.mark.asyncio
async def test_resolve_retrieval_feedback_correction_short_circuits(monkeypatch):
    from app.services.memory.interaction import retrieval_feedback

    events = [{"type": "reply", "text": "我可能记错了，确认一下？"}]

    async def fake_short_circuit(*args, **kwargs):
        return events

    ctx = SimpleNamespace(
        conversation_id="c1",
        agent_id="agent1",
        user_id="u1",
        agent=SimpleNamespace(name="Hillow"),
        tracer=SimpleNamespace(safe_trace_id="trace1", close=lambda: None),
        short_circuit_fn=fake_short_circuit,
        stopped=False,
        last_short_circuit_reply=None,
    )
    monkeypatch.setattr(
        retrieval_feedback,
        "build_retrieval_feedback_conflict",
        AsyncMock(return_value=(
            {"conflicting_memory_id": "m1", "old_content": "用户喜欢芒果"},
            {"confidence": 0.92, "repair_action": {"type": "confirmation_requested"}},
        )),
    )
    monkeypatch.setattr(retrieval_feedback, "_patch_previous_feedback", AsyncMock())
    monkeypatch.setattr(retrieval_feedback, "save_pending_contradiction", AsyncMock())
    monkeypatch.setattr(
        retrieval_feedback,
        "generate_contradiction_inquiry",
        AsyncMock(return_value=events[0]["text"]),
    )

    got = [
        evt async for evt in retrieval_feedback.resolve_retrieval_feedback_correction(
            user_message="你记错了。",
            previous_assistant=_previous_assistant(_metadata()),
            ctx=ctx,
            workspace_id="w1",
        )
    ]

    assert got == events
    assert ctx.stopped is True
    assert ctx.last_short_circuit_reply == events[0]["text"]
    retrieval_feedback.save_pending_contradiction.assert_awaited_once()
