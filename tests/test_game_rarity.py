"""一局游戏在用户历史里的稀有性.

"值不值得记住"是相对的 —— 用户第一次赢你值得记, 第二十次不值得。LLM 只看得到这
一局, 算不出这个; SQL 一句话就能算准。这里只回答**可以被计算**的问题, 主观的
"精彩不精彩"留给 LLM。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.games import rarity as R


def _prior(outcome: str, actions: int = 20, dur: int = 60) -> dict:
    return {"user_outcome": outcome, "action_count": actions,
            "duration_seconds": dur, "ended_at": "2026-08-01T00:00:00+00:00"}


async def _compute(monkeypatch, prior_rows: list[dict], **kw):
    monkeypatch.setattr(R.db, "query_raw", AsyncMock(return_value=prior_rows))
    defaults = dict(
        workspace_id="ws1", game_key="gomoku", game_title="五子棋",
        session_id="s-now", user_outcome="win", action_count=20,
        duration_seconds=60,
    )
    defaults.update(kw)
    return await R.compute_rarity(**defaults)


@pytest.mark.asyncio
class TestFirsts:
    async def test_first_game_ever(self, monkeypatch):
        r = await _compute(monkeypatch, [])
        assert r.is_first_ever is True
        assert any("第一次一起玩" in n for n in r.notes)

    async def test_first_win_after_losses(self, monkeypatch):
        """输了五局终于赢一次 —— 这是最该被记住的时刻之一."""
        r = await _compute(monkeypatch, [_prior("lose")] * 5)
        assert r.is_first_win is True
        assert r.is_first_ever is False
        assert any("第一次" in n and "赢" in n for n in r.notes)

    async def test_not_first_win_when_won_before(self, monkeypatch):
        r = await _compute(monkeypatch, [_prior("win"), _prior("lose")])
        assert r.is_first_win is False


@pytest.mark.asyncio
class TestStreak:
    async def test_win_streak_counts_current_game(self, monkeypatch):
        r = await _compute(monkeypatch, [_prior("win"), _prior("win"), _prior("lose")])
        assert r.streak == 3
        assert any("连赢 3" in n for n in r.notes)

    async def test_lose_streak_is_negative(self, monkeypatch):
        r = await _compute(
            monkeypatch, [_prior("lose")] * 3, user_outcome="lose",
        )
        assert r.streak == -4
        assert any("连输 4" in n for n in r.notes)

    async def test_draw_does_not_start_a_streak(self, monkeypatch):
        r = await _compute(monkeypatch, [_prior("win")] * 3, user_outcome="draw")
        assert r.streak == 0

    async def test_a_draw_in_between_breaks_the_streak(self, monkeypatch):
        """「连赢三局」里夹一局平局就不该再算连赢."""
        r = await _compute(monkeypatch, [_prior("win"), _prior("draw"), _prior("win")])
        assert r.streak == 2

    async def test_short_streak_is_not_worth_mentioning(self, monkeypatch):
        """赢两局就喊"连赢"会让 agent 显得很聒噪."""
        r = await _compute(monkeypatch, [_prior("win"), _prior("lose")])
        assert r.streak == 2
        assert not any("连赢" in n for n in r.notes)


@pytest.mark.asyncio
class TestRecords:
    async def test_fewest_moves_needs_enough_history(self, monkeypatch):
        """只玩过一局就说"步数最少"是荒谬的 —— 那是唯一一局."""
        r = await _compute(monkeypatch, [_prior("lose", actions=50)], action_count=10)
        assert r.is_fewest_moves is True  # 事实成立
        assert not any("步数最少" in n for n in r.notes)  # 但不值得说

    async def test_fewest_moves_is_mentioned_with_history(self, monkeypatch):
        r = await _compute(
            monkeypatch, [_prior("lose", actions=50)] * 4, action_count=10,
        )
        assert any("步数最少" in n for n in r.notes)

    async def test_longest_game(self, monkeypatch):
        r = await _compute(
            monkeypatch, [_prior("win", dur=30)] * 4, duration_seconds=600,
        )
        assert r.is_longest is True
        assert any("最久" in n for n in r.notes)


@pytest.mark.asyncio
class TestQueryShape:
    """这个查询跑在每局结束的热路径上, 而 game_sessions 没有 workspace_id 索引."""

    async def test_scan_is_bounded(self, monkeypatch):
        captured: list = []

        async def fake(sql, *args):
            captured.append(sql)
            return []

        monkeypatch.setattr(R.db, "query_raw", fake)
        await R.compute_rarity(
            workspace_id="ws1", game_key="go", game_title="围棋", session_id="s",
            user_outcome="win", action_count=10, duration_seconds=60,
        )
        assert "LIMIT" in captured[0], "无上限的扫表会随游戏量线性变慢"

    async def test_excludes_the_current_session(self, monkeypatch):
        """把本局算进历史会让"首次"永远为假、连胜多算一局."""
        captured: list = []

        async def fake(sql, *args):
            captured.append((sql, args))
            return []

        monkeypatch.setattr(R.db, "query_raw", fake)
        await R.compute_rarity(
            workspace_id="ws1", game_key="go", game_title="围棋", session_id="s-now",
            user_outcome="win", action_count=10, duration_seconds=60,
        )
        sql, args = captured[0]
        assert "id <> $3" in sql
        assert args[2] == "s-now"


@pytest.mark.asyncio
class TestRobustness:
    async def test_ordinary_game_has_no_notes(self, monkeypatch):
        """绝大多数局就该是"没什么特别的" —— 否则又变成模板量产."""
        r = await _compute(monkeypatch, [_prior("win"), _prior("lose")] * 5)
        assert r.notes == []
        assert r.is_notable is False

    async def test_db_failure_degrades_quietly(self, monkeypatch):
        """稀有性是锦上添花, 算不出来不该让游戏结束流程失败."""
        monkeypatch.setattr(R.db, "query_raw", AsyncMock(side_effect=RuntimeError("db down")))
        r = await R.compute_rarity(
            workspace_id="ws1", game_key="go", game_title="围棋", session_id="s",
            user_outcome="win", action_count=10, duration_seconds=60,
        )
        assert r.notes == []

    async def test_missing_workspace_is_safe(self, monkeypatch):
        r = await R.compute_rarity(
            workspace_id=None, game_key="go", game_title="围棋", session_id="s",
            user_outcome="win", action_count=10, duration_seconds=60,
        )
        assert r.is_notable is False

    async def test_aborted_game_has_no_streak(self, monkeypatch):
        """中途退出不算输也不算赢, 不该打断也不该延续连胜."""
        r = await _compute(monkeypatch, [_prior("win")] * 3, user_outcome="aborted")
        assert r.streak == 0
