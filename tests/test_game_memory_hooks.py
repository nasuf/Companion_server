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
    # L3 而不是 L2: 游戏记忆该像真人一样快速淡出。原来 level=2 / importance 0.80
    # 已经逼近 L1 阈值 (0.85), 而记的是"走了97步，4分钟"这类流水 —— 近 30 天有
    # 745 局, 照那个写法光游戏就能产出上千条 L2 把重要的事挤出检索。
    assert calls[0]["level"] == 3
    assert calls[0]["sub_category"] == "其他特殊事件"
    assert calls[1]["level"] == 3
    assert calls[1]["importance"] == 0.45
    assert calls[1]["sub_category"] == "交互"
    # importance 落在 L3 区间 (0.10-0.50), 不会被 level_for_importance 拉回 L2
    assert all(0.10 <= c["importance"] < 0.50 for c in calls)


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
