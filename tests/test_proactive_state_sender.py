from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.proactive.sender import send_manual_or_triggered_proactive
from app.services.proactive.state import (
    ProactiveStateRecord,
    advance_to_next_window,
    claim_due_proactive_state,
    claim_waiting_timeout_state,
    has_recent_user_activity,
)


UTC = timezone.utc


def _state(**overrides):
    base = dict(
        id="state-1",
        workspace_id="ws-1",
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        status="idle",
        stage="warming",
        silence_level_n=0,
        followup_plan_type="normal",
        remaining_forced_triggers=None,
        current_window_index=1,
        window_due_at=None,
        response_deadline_at=None,
        t0_at=None,
        last_proactive_at=None,
        last_user_reply_at=None,
        last_assistant_reply_at=None,
        last_attempt_at=None,
        daily_scene_triggered_at=None,
        stop_reason=None,
        metadata=None,
    )
    base.update(overrides)
    return ProactiveStateRecord(**base)


@pytest.mark.asyncio
async def test_claim_due_proactive_state_returns_claimed_row():
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    row = {
        "id": "state-1",
        "workspace_id": "ws-1",
        "user_id": "user-1",
        "agent_id": "agent-1",
        "conversation_id": "conv-1",
        "status": "processing",
        "stage": "warming",
        "silence_level_n": 0,
        "followup_plan_type": "normal",
        "remaining_forced_triggers": None,
        "current_window_index": 1,
        "window_due_at": now.isoformat(),
        "response_deadline_at": None,
        "t0_at": now.isoformat(),
        "last_proactive_at": None,
        "last_user_reply_at": None,
        "last_assistant_reply_at": now.isoformat(),
        "last_attempt_at": now.isoformat(),
        "daily_scene_triggered_at": None,
        "stop_reason": None,
        "metadata": None,
    }
    mock_db = SimpleNamespace(query_raw=AsyncMock(return_value=[row]))
    with patch("app.services.proactive.state.db", new=mock_db):
        claimed = await claim_due_proactive_state("state-1", now=now)

    assert claimed is not None
    assert claimed.status == "processing"
    mock_db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_waiting_timeout_state_returns_none_when_already_claimed():
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    mock_db = SimpleNamespace(query_raw=AsyncMock(return_value=[]))
    with patch("app.services.proactive.state.db", new=mock_db):
        claimed = await claim_waiting_timeout_state("state-1", now=now)

    assert claimed is None


@pytest.mark.asyncio
async def test_send_manual_or_triggered_proactive_blocks_waiting_state():
    state = _state(status="waiting_user")
    with (
        patch("app.services.proactive.sender.ensure_proactive_state_for_workspace", new_callable=AsyncMock, return_value=state),
        patch("app.services.proactive.sender.log_proactive_event", new_callable=AsyncMock) as mock_log,
        patch("app.services.proactive.sender.generate_and_send_proactive", new_callable=AsyncMock) as mock_send,
    ):
        result = await send_manual_or_triggered_proactive(
            workspace_id="ws-1",
            trigger_type="trigger:greeting",
        )

    assert result["ok"] is False
    assert result["reason"] == "state_not_sendable:waiting_user"
    mock_send.assert_not_awaited()
    mock_log.assert_awaited_once()
    assert mock_log.await_args.kwargs["payload"]["status"] == "waiting_user"


@pytest.mark.asyncio
async def test_advance_window_loops_back_to_1_at_cycle_end():
    """spec §1.2 step 4: 走完 4-6h 区间未命中 → 重启 0-6h 循环 (回到 window 1).

    历史 bug: next_index >= 5 直接 _escalate (n+1), 配合 off_hours 命中也走
    advance_to_next_window, 用户夜间衰减比 spec 快 ~2x.
    """
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    state = _state(current_window_index=4, t0_at=now)
    mock_db = SimpleNamespace(execute_raw=AsyncMock())
    with (
        patch("app.services.proactive.state.db", new=mock_db),
        patch("app.services.proactive.state.log_proactive_event", new_callable=AsyncMock) as mock_log,
        patch("app.services.proactive.state._escalate_silence_level", new_callable=AsyncMock) as mock_escalate,
    ):
        await advance_to_next_window(state, now=now)

    # 关键: escalate 不能被调用 (n+1 只属于 spec §8 用户回复超时)
    mock_escalate.assert_not_awaited()
    mock_db.execute_raw.assert_awaited_once()
    args = mock_db.execute_raw.await_args.args
    assert args[2] == 1  # next_index 回到 1
    # 事件类型默认升级为 cycle_restarted, payload 含标记
    mock_log.assert_awaited_once()
    log_kwargs = mock_log.await_args.kwargs
    assert log_kwargs["event_type"] == "cycle_restarted"
    assert log_kwargs["window_index"] == 1
    assert log_kwargs["payload"].get("cycle_restarted") is True


@pytest.mark.asyncio
async def test_advance_window_normal_increments_next_index():
    """常规推进: next_index < 5 → 递增, 不重启不 escalate."""
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    state = _state(current_window_index=2, t0_at=now)
    mock_db = SimpleNamespace(execute_raw=AsyncMock())
    with (
        patch("app.services.proactive.state.db", new=mock_db),
        patch("app.services.proactive.state.log_proactive_event", new_callable=AsyncMock),
        patch("app.services.proactive.state._escalate_silence_level", new_callable=AsyncMock) as mock_escalate,
    ):
        await advance_to_next_window(state, now=now)

    mock_escalate.assert_not_awaited()
    args = mock_db.execute_raw.await_args.args
    assert args[2] == 3


class TestPresenceIncludesGames:
    """一起玩游戏也算"用户在场".

    游戏全程只写 assistant 消息 (状态播报 + 完局伴聊), 用户落子不产生 user 消息。
    只看 messages 的话, 一局 20 分钟的棋在主动交流眼里是"沉默了 20 分钟" ——
    实测 161 局超过 5 分钟且期间用户零消息。主动窗口落在里面, AI 就会一边陪用户
    下棋一边发「好久没聊了」。
    """

    @staticmethod
    def _rows_for(*, message_hit: bool, game_hit: bool):
        """按调用顺序返回: 第一次查 messages, 第二次查 game_sessions."""
        calls = {"n": 0}

        async def _q(sql, *args):
            calls["n"] += 1
            if "FROM messages" in sql:
                return [{"?column?": 1}] if message_hit else []
            return [{"?column?": 1}] if game_hit else []

        return _q, calls

    @pytest.mark.asyncio
    async def test_an_active_game_counts_as_presence(self, monkeypatch):
        q, _ = self._rows_for(message_hit=False, game_hit=True)
        monkeypatch.setattr("app.services.proactive.state.db.query_raw", q)
        assert await has_recent_user_activity("ws") is True

    @pytest.mark.asyncio
    async def test_silence_with_no_game_is_still_silence(self, monkeypatch):
        q, _ = self._rows_for(message_hit=False, game_hit=False)
        monkeypatch.setattr("app.services.proactive.state.db.query_raw", q)
        assert await has_recent_user_activity("ws") is False

    @pytest.mark.asyncio
    async def test_a_recent_message_short_circuits_the_game_lookup(self, monkeypatch):
        """消息命中就不必再查一遍对局 —— 这是每分钟跑的热路径."""
        q, calls = self._rows_for(message_hit=True, game_hit=True)
        monkeypatch.setattr("app.services.proactive.state.db.query_raw", q)
        assert await has_recent_user_activity("ws") is True
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_games_in_progress_are_not_filtered_out_by_status(self):
        """实测 status 只有 created/settled/aborted, 而 created 就是"正在下".

        按 status 过滤会漏掉最该保护的场景, 所以查询里刻意不带 status 条件。
        """
        import inspect

        from app.services.proactive import state as mod

        src = inspect.getsource(mod.has_recent_user_activity)
        game_query = src[src.index("FROM game_sessions"):]
        assert "status" not in game_query, "按 status 过滤会漏掉进行中的对局"

    @pytest.mark.asyncio
    async def test_lookup_failure_does_not_block_proactive(self, monkeypatch):
        """查不到就当没在场 —— 宁可多发一条也不能让主动交流永久停摆."""
        async def _boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr("app.services.proactive.state.db.query_raw", _boom)
        assert await has_recent_user_activity("ws") is False
