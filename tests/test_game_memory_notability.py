"""游戏记忆只在这一局确实特别时才写.

改之前是**每局都写**双侧 L2 (importance 0.74/0.80)。那个分数已经逼近 L1 阈值,
而记的内容是"走了97步，4分钟"。近 30 天有 745 局 —— 照那个写法光游戏就能产出
上千条 L2, 把真正重要的事挤出检索。实测这批记忆两两相似度中位 0.710 (普通记忆
0.361), 已经在向量空间里挤成一坨。

真朋友一起下二十盘棋, 隔天想得起的就一两盘, 而且想起的是"那次你连跳七格反超",
不是统计量。所以单局流水不进记忆库 (它在 game_sessions 表里), 只留稀有的那几局。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.games import native
from app.services.games.rarity import GameRarity


def _session(game_key: str = "gomoku"):
    return SimpleNamespace(
        id="s1", user_id="u1", workspace_id="ws1", game_key=game_key,
        ai_player=SimpleNamespace(nick_name="小伴"),
    )


def _result(*, outcome="win", moves=20, duration=120, moments=None):
    return {
        "game_key": "gomoku",
        "user_outcome": outcome,
        "duration_seconds": duration,
        "gomoku": {"move_count": moves, "key_moments": moments or []},
        "process": {"gomoku": {"action_count": moves, "key_moments": moments or []}},
    }


@pytest.mark.asyncio
class TestNotabilityGate:
    async def test_ordinary_game_writes_nothing(self, monkeypatch):
        """绝大多数局就该被忘掉 —— 这是整个改动的要点."""
        monkeypatch.setattr(
            native, "compute_rarity", AsyncMock(return_value=GameRarity()),
        )
        store = AsyncMock()
        monkeypatch.setattr(native, "remember_shared_game_experience", store)

        out = await native._remember_shared_experience(_session(), _result())
        assert out["status"] == "skipped"
        assert out["reason"] == "not_notable"
        store.assert_not_awaited()

    async def test_rare_game_is_remembered(self, monkeypatch):
        monkeypatch.setattr(
            native, "compute_rarity",
            AsyncMock(return_value=GameRarity(notes=["这是用户第一次在《五子棋》赢"])),
        )
        captured: list = []

        async def fake_store(**kw):
            captured.append(kw)
            return {"status": "stored", "user_memory_id": "m1", "ai_memory_id": "m2",
                    "failed_sides": []}

        monkeypatch.setattr(native, "remember_shared_game_experience", fake_store)
        out = await native._remember_shared_experience(_session(), _result())
        assert out["status"] == "stored"
        assert "第一次" in captured[0]["user_text"]

    async def test_key_moment_alone_is_enough(self, monkeypatch):
        """引擎标出的高光时刻也算值得留 —— 稀有性和精彩度是两条独立的路."""
        monkeypatch.setattr(
            native, "compute_rarity", AsyncMock(return_value=GameRarity()),
        )
        store = AsyncMock(return_value={"status": "stored", "user_memory_id": "m",
                                        "ai_memory_id": "m", "failed_sides": []})
        monkeypatch.setattr(native, "remember_shared_game_experience", store)
        monkeypatch.setattr(native, "_memory_moment", lambda *a: "有一次双三绝杀。")

        out = await native._remember_shared_experience(_session(), _result())
        assert out["status"] == "stored"

    async def test_zero_action_game_still_short_circuits_first(self, monkeypatch):
        """没走过一步的局连稀有性都不用算 —— 省一次 DB 查询."""
        rarity = AsyncMock(return_value=GameRarity())
        monkeypatch.setattr(native, "compute_rarity", rarity)
        out = await native._remember_shared_experience(_session(), _result(moves=0))
        assert out["status"] == "skipped"
        rarity.assert_not_awaited()


@pytest.mark.asyncio
class TestTemplateRemoval:
    async def test_fixed_tail_is_gone(self, monkeypatch):
        """「这是我们共同经历的一局游戏。」每条都挂 —— 那是记忆挤成一坨的原因之一."""
        monkeypatch.setattr(
            native, "compute_rarity",
            AsyncMock(return_value=GameRarity(notes=["这是你们第一次一起玩《五子棋》"])),
        )
        captured: list = []

        async def fake_store(**kw):
            captured.append(kw)
            return {"status": "stored", "user_memory_id": "m", "ai_memory_id": "m",
                    "failed_sides": []}

        monkeypatch.setattr(native, "remember_shared_game_experience", fake_store)
        await native._remember_shared_experience(_session(), _result())
        assert "共同经历" not in captured[0]["ai_text"]

    async def test_rarity_appears_in_both_sides(self, monkeypatch):
        monkeypatch.setattr(
            native, "compute_rarity",
            AsyncMock(return_value=GameRarity(notes=["用户已经连赢 3 局"])),
        )
        captured: list = []

        async def fake_store(**kw):
            captured.append(kw)
            return {"status": "stored", "user_memory_id": "m", "ai_memory_id": "m",
                    "failed_sides": []}

        monkeypatch.setattr(native, "remember_shared_game_experience", fake_store)
        await native._remember_shared_experience(_session(), _result())
        assert "连赢 3" in captured[0]["user_text"]
        assert "连赢 3" in captured[0]["ai_text"]
