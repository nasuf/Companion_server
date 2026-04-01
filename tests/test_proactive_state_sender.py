from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.proactive_sender import send_manual_or_triggered_proactive
from app.services.proactive_state import (
    ProactiveStateRecord,
    claim_due_proactive_state,
    claim_waiting_timeout_state,
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
    with patch("app.services.proactive_state.db", new=mock_db):
        claimed = await claim_due_proactive_state("state-1", now=now)

    assert claimed is not None
    assert claimed.status == "processing"
    mock_db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_waiting_timeout_state_returns_none_when_already_claimed():
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    mock_db = SimpleNamespace(query_raw=AsyncMock(return_value=[]))
    with patch("app.services.proactive_state.db", new=mock_db):
        claimed = await claim_waiting_timeout_state("state-1", now=now)

    assert claimed is None


@pytest.mark.asyncio
async def test_send_manual_or_triggered_proactive_blocks_waiting_state():
    state = _state(status="waiting_user")
    with (
        patch("app.services.proactive_sender.ensure_proactive_state_for_workspace", new_callable=AsyncMock, return_value=state),
        patch("app.services.proactive_sender.log_proactive_event", new_callable=AsyncMock) as mock_log,
        patch("app.services.proactive_sender.generate_and_send_proactive", new_callable=AsyncMock) as mock_send,
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
