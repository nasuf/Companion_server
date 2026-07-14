from __future__ import annotations

import pytest

from app.services.offline import memory_hooks


@pytest.mark.asyncio
async def test_shared_game_experience_stores_both_sides_without_prefilter(monkeypatch):
    calls = []

    async def store_memory(**data):
        calls.append(data)
        return f"{data['source']}-memory"

    monkeypatch.setattr(memory_hooks, "store_memory", store_memory)

    result = await memory_hooks.remember_shared_game_experience(
        user_id="user-1",
        workspace_id="workspace-1",
        user_text="我和小伴下了一局五子棋，我赢了。",
        ai_text="我和用户下了一局五子棋，用户赢了。",
        agent_name="小伴",
        game_title="五子棋",
    )

    assert result == {
        "status": "stored",
        "user_memory_id": "user-memory",
        "ai_memory_id": "ai-memory",
        "failed_sides": [],
    }
    assert [call["source"] for call in calls] == ["user", "ai"]
    assert calls[0]["level"] == 2
    assert calls[0]["sub_category"] == "其他特殊事件"
    assert calls[1]["level"] == 2
    assert calls[1]["importance"] == 0.80
    assert calls[1]["sub_category"] == "交互"


@pytest.mark.asyncio
async def test_shared_game_experience_reports_partial_failure(monkeypatch):
    attempts = {"user": 0, "ai": 0}

    async def store_memory(**data):
        side = data["source"]
        attempts[side] += 1
        if side == "user":
            raise RuntimeError("embedding unavailable")
        return "ai-memory"

    monkeypatch.setattr(memory_hooks, "store_memory", store_memory)

    result = await memory_hooks.remember_shared_game_experience(
        user_id="user-1",
        workspace_id="workspace-1",
        user_text="用户侧经历",
        ai_text="AI 侧经历",
        agent_name="小伴",
        game_title="五子棋",
    )

    assert result["status"] == "partial"
    assert result["failed_sides"] == ["user"]
    assert result["ai_memory_id"] == "ai-memory"
    assert attempts == {"user": 1, "ai": 1}


@pytest.mark.asyncio
async def test_shared_game_experience_can_retry_only_one_side(monkeypatch):
    calls = []

    async def store_memory(**data):
        calls.append(data["source"])
        return "ai-memory"

    monkeypatch.setattr(memory_hooks, "store_memory", store_memory)

    result = await memory_hooks.remember_shared_game_experience(
        user_id="user-1",
        workspace_id="workspace-1",
        user_text="用户侧经历",
        ai_text="AI 侧经历",
        agent_name="小伴",
        game_title="围棋",
        sides=("ai",),
    )

    assert calls == ["ai"]
    assert result["status"] == "stored"
    assert result["user_memory_id"] is None
    assert result["ai_memory_id"] == "ai-memory"
