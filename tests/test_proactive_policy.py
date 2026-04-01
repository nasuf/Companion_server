from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.proactive_policy import (
    scene_candidate_available,
    select_trigger_type,
    should_hit_window,
)
from app.services.proactive_state import ProactiveStateRecord, escalate_waiting_state


UTC = timezone.utc


def _state(**overrides):
    base = dict(
        id="state-1",
        workspace_id="ws-1",
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        status="running",
        stage="warming",
        silence_level_n=0,
        followup_plan_type="normal",
        remaining_forced_triggers=None,
        current_window_index=4,
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


def test_should_hit_window_zero_rate_never_hits():
    hit, final_rate = should_hit_window(_state(current_window_index=0))
    assert hit is False
    assert final_rate == 0.0


def test_scene_candidate_unavailable_when_sleeping():
    assert (
        scene_candidate_available(
            _state(),
            {"status": "sleep"},
            now=datetime(2026, 4, 1, 10, 0, tzinfo=UTC),
        )
        is False
    )


def test_scene_candidate_unavailable_when_already_triggered_today():
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    assert (
        scene_candidate_available(
            _state(daily_scene_triggered_at=now),
            {"status": "idle"},
            now=now,
        )
        is False
    )


def test_select_trigger_type_without_scene_never_returns_scene():
    selections = {
        select_trigger_type(_state(stage="cold_start"), scene_available=False)
        for _ in range(50)
    }
    assert "scheduled_scene" not in selections


@pytest.mark.asyncio
async def test_escalate_waiting_state_resumes_normal_plan_for_n_le_4():
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    due_at = now + timedelta(minutes=45)
    mock_db = SimpleNamespace(execute_raw=AsyncMock())
    with (
        patch("app.services.proactive_state.db", new=mock_db),
        patch("app.services.proactive_state.log_proactive_event", new_callable=AsyncMock) as mock_log,
        patch("app.services.proactive_state._pick_random_due_at", return_value=due_at),
    ):
        await escalate_waiting_state(_state(silence_level_n=3), now=now)

    mock_db.execute_raw.assert_awaited_once()
    args = mock_db.execute_raw.await_args.args
    assert "UPDATE proactive_states" in args[0]
    assert args[2] == 4
    assert args[3] == due_at
    mock_log.assert_awaited_once()
    payload = mock_log.await_args.kwargs["payload"]
    assert payload["n"] == 4
    assert payload["plan"] == "normal"


@pytest.mark.asyncio
async def test_escalate_waiting_state_enters_seven_day_sparse_plan_at_n_5():
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    due_at = now + timedelta(days=2)
    with (
        patch("app.services.proactive_state._resume_forced_plan", new_callable=AsyncMock) as mock_resume,
        patch("app.services.proactive_state._pick_random_future_due_at", return_value=due_at),
    ):
        await escalate_waiting_state(_state(silence_level_n=4), now=now)

    mock_resume.assert_awaited_once()
    kwargs = mock_resume.await_args.kwargs
    assert kwargs["followup_plan_type"] == "seven_day_sparse"
    assert kwargs["remaining_forced_triggers"] == 2
    assert kwargs["silence_level_n"] == 5
    assert kwargs["payload"]["plan"] == "seven_day_sparse"


@pytest.mark.asyncio
async def test_escalate_waiting_state_resumes_existing_forced_plan_when_remaining():
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    due_at = now + timedelta(days=1)
    window_end = (now + timedelta(days=7)).isoformat()
    with (
        patch("app.services.proactive_state._resume_forced_plan", new_callable=AsyncMock) as mock_resume,
        patch("app.services.proactive_state._pick_random_future_due_at", return_value=due_at),
    ):
        await escalate_waiting_state(
            _state(
                silence_level_n=5,
                followup_plan_type="seven_day_sparse",
                remaining_forced_triggers=1,
                metadata={"plan_window_end_at": window_end},
            ),
            now=now,
        )

    mock_resume.assert_awaited_once()
    kwargs = mock_resume.await_args.kwargs
    assert kwargs["followup_plan_type"] == "seven_day_sparse"
    assert kwargs["remaining_forced_triggers"] == 1
    assert kwargs["event_type"] == "silence_plan_resumed"


@pytest.mark.asyncio
async def test_escalate_waiting_state_stops_after_final_no_reply():
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    with (
        patch("app.services.proactive_state.stop_proactive_state", new_callable=AsyncMock) as mock_stop,
        patch("app.services.proactive_state.log_proactive_event", new_callable=AsyncMock) as mock_log,
    ):
        await escalate_waiting_state(
            _state(
                silence_level_n=6,
                followup_plan_type="thirty_day_final",
                remaining_forced_triggers=0,
            ),
            now=now,
        )

    mock_stop.assert_awaited_once()
    assert mock_stop.await_args.kwargs["reason"] == "final_no_reply"
    mock_log.assert_awaited_once()
    assert mock_log.await_args.kwargs["event_type"] == "permanently_stopped"
