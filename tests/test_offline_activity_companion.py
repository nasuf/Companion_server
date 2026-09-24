import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.offline import activity_companion as companion
from app.services.offline import fragment_pregen, recognition, shooting_conditions
from app.services.offline import repository as offline_repo
from app.services.offline.guidance import (
    contains_hidden_target,
    normalize_guidance_profile,
    safe_guidance,
)
from app.services.proactive import emit as proactive_emit
from app.services.proactive import orchestrator as proactive_orchestrator
from app.services.prompting import defaults
from app.services.prompting.registry import PROMPT_DEFINITION_MAP


def _activity(*, state=None):
    return {
        "id": "activity-1",
        "user_id": "user-1",
        "agent_id": "agent-1",
        "workspace_id": "workspace-1",
        "conversation_id": "conversation-1",
        "title": "城市散步",
        "location_name": "河滨公园",
        "arrival_confirmed_at": (datetime.now(UTC) - timedelta(minutes=30)).isoformat(),
        "focus_condition_id": "condition-1",
        "companion_claim_token": "claim-1",
        "companion_state": state or {},
    }


def _condition(condition_id="condition-1"):
    return {
        "id": condition_id,
        "short_name": "花",
        "category": "植物",
        "criteria": "主体清楚呈现花朵",
        "guidance_profile": normalize_guidance_profile(
            short_name="花",
            category="植物",
            raw_profile={},
        ),
    }


def _install_common(monkeypatch, *, messages, idle=True, last_ai_at=None):
    monkeypatch.setattr(
        companion.repo,
        "resolve_user_context",
        AsyncMock(
            return_value={
                "conversation_id": "conversation-1",
                "agent_id": "agent-1",
            }
        ),
    )
    monkeypatch.setattr(
        companion,
        "_recent_messages",
        AsyncMock(return_value=messages),
    )
    monkeypatch.setattr(
        companion,
        "_new_user_message_arrived",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        companion,
        "_ai_is_idle",
        AsyncMock(return_value=idle),
    )
    monkeypatch.setattr(
        companion,
        "_latest_assistant_at",
        AsyncMock(return_value=last_ai_at),
    )
    monkeypatch.setattr(
        companion.repo,
        "save_companion_decision",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        companion,
        "_generate_companion_message",
        AsyncMock(return_value="慢慢逛，我陪着你。"),
    )


def _open_window(**overrides):
    now = datetime.now(UTC)
    state = {
        "phase": "opening",
        "segment_sent": 0,
        "activity_sent": 0,
        "window_index": 2,
        "anchor_at": (now - timedelta(minutes=3)).isoformat(),
        "window_start": (now - timedelta(seconds=40)).isoformat(),
        "window_end": (now + timedelta(minutes=1)).isoformat(),
        "due_at": (now - timedelta(seconds=5)).isoformat(),
        "probability": 1.0,
        "recent_modes": [],
        "awaiting_passive_reply": False,
    }
    state.update(overrides)
    return state


class _Rng:
    def __init__(self, unit=0.0, wait=1):
        self.unit = unit
        self.wait = wait

    def random(self):
        return self.unit

    def randint(self, low, high):
        return self.wait


def test_opening_tables_match_the_timing_document():
    assert companion.OPENING_WINDOWS[0][0] == (0, 1, 0.80)
    assert companion.OPENING_WINDOWS[0][2] == (2, 3, 0.50)
    assert companion.OPENING_WINDOWS[2] == (
        (0, 1, 0.40),
        (1, 2, 0.28),
        (2, 3, 0.18),
        (3, 7, 0.12),
        (7, 12, 0.06),
    )
    assert companion.FOLLOWUP_WINDOWS[0][0] == (0, 2, 0.55)
    assert companion.FOLLOWUP_WINDOWS[1][-1] == (8, 15, 0.12)
    assert companion.ACTIVITY_CAP == 5
    assert companion.OPENING_CAP == 3
    assert companion.FOLLOWUP_CAP == 2


def test_expired_window_advances_without_counting():
    now = datetime.now(UTC)
    anchor = now - timedelta(minutes=2)
    state = _open_window(
        window_index=0,
        anchor_at=anchor.isoformat(),
        window_start=anchor.isoformat(),
        window_end=(anchor + timedelta(minutes=1)).isoformat(),
        due_at=(anchor + timedelta(seconds=20)).isoformat(),
        probability=0.80,
    )

    decision = companion.evaluate_tick(
        state,
        now,
        ai_idle=True,
        last_ai_at=None,
        rng=_Rng(),
    )

    assert decision.reason == "window_expired"
    assert decision.state["activity_sent"] == 0
    assert decision.state["window_index"] == 1
    assert decision.state["probability"] == 0.65


def test_probability_miss_does_not_count():
    now = datetime.now(UTC)
    decision = companion.evaluate_tick(
        _open_window(probability=0.0, window_index=2),
        now,
        ai_idle=True,
        last_ai_at=None,
        rng=_Rng(unit=0.99),
    )

    assert decision.reason == "probability_miss"
    assert decision.state["activity_sent"] == 0
    assert decision.state["window_index"] == 3
    assert decision.state["probability"] == 0.35


def test_two_minute_gap_skips_the_window_without_counting():
    now = datetime.now(UTC)
    anchor = now - timedelta(seconds=30)
    state = _open_window(
        window_index=0,
        anchor_at=anchor.isoformat(),
        window_start=anchor.isoformat(),
        window_end=(anchor + timedelta(minutes=1)).isoformat(),
        due_at=(anchor + timedelta(seconds=10)).isoformat(),
        probability=0.80,
    )

    decision = companion.evaluate_tick(
        state,
        now,
        ai_idle=True,
        last_ai_at=anchor,
        rng=_Rng(),
    )

    assert decision.reason == "ai_gap"
    assert decision.state["activity_sent"] == 0
    assert decision.state["window_index"] == 1
    assert decision.outcome != "send"


def test_gap_leaves_the_two_to_three_minute_window_rollable():
    now = datetime.now(UTC)
    anchor = now - timedelta(minutes=2, seconds=30)
    state = _open_window(
        window_index=2,
        anchor_at=anchor.isoformat(),
        window_start=(anchor + timedelta(minutes=2)).isoformat(),
        window_end=(anchor + timedelta(minutes=3)).isoformat(),
        due_at=(anchor + timedelta(minutes=2, seconds=10)).isoformat(),
        probability=0.50,
    )

    decision = companion.evaluate_tick(
        state,
        now,
        ai_idle=True,
        last_ai_at=anchor,
        rng=_Rng(unit=0.1),
    )

    assert decision.outcome == "send"
    assert decision.state["activity_sent"] == 0


def test_busy_waits_inside_the_window_and_skips_after_it():
    now = datetime.now(UTC)
    open_end = now + timedelta(minutes=2)
    waiting = companion.evaluate_tick(
        _open_window(window_end=open_end.isoformat()),
        now,
        ai_idle=False,
        last_ai_at=None,
        rng=_Rng(),
    )
    assert waiting.reason == "ai_busy"
    assert waiting.state["activity_sent"] == 0
    assert waiting.state["window_index"] == 2
    assert waiting.next_at is not None
    assert now < waiting.next_at < open_end

    expired = companion.evaluate_tick(
        _open_window(window_end=(now + timedelta(milliseconds=200)).isoformat()),
        now,
        ai_idle=False,
        last_ai_at=None,
        rng=_Rng(),
    )
    assert expired.reason == "ai_busy"
    assert expired.state["activity_sent"] == 0
    assert expired.state["window_index"] == 3


def test_five_missed_opening_windows_end_the_segment():
    now = datetime.now(UTC)
    anchor = now - timedelta(minutes=20)
    state = _open_window(
        window_index=4,
        anchor_at=anchor.isoformat(),
        window_start=(anchor + timedelta(minutes=7)).isoformat(),
        window_end=(anchor + timedelta(minutes=12)).isoformat(),
        due_at=(anchor + timedelta(minutes=8)).isoformat(),
        probability=0.18,
    )

    decision = companion.evaluate_tick(
        state,
        now,
        ai_idle=True,
        last_ai_at=None,
        rng=_Rng(),
    )

    assert decision.outcome == "stop"
    assert decision.reason == "segment_done"
    assert decision.next_at is None
    assert decision.state["activity_sent"] == 0
    assert decision.state["phase"] == "stopped"


def test_activity_cap_stops_without_another_send():
    decision = companion.evaluate_tick(
        _open_window(activity_sent=5),
        datetime.now(UTC),
        ai_idle=True,
        last_ai_at=None,
    )

    assert decision.outcome == "stop"
    assert decision.next_at is None
    assert decision.state["phase"] == "stopped"


def test_followup_wait_is_one_to_three_minutes_and_keeps_the_activity_count():
    reply_at = datetime(2026, 9, 24, 6, 0, tzinfo=UTC)
    state, due = companion.begin_followup(
        reply_at,
        activity_sent=2,
        recent_modes=["social"],
        rng=_Rng(unit=0.0, wait=3),
    )

    assert state["phase"] == "followup"
    assert state["segment_sent"] == 0
    assert state["activity_sent"] == 2
    assert state["probability"] == 0.55
    assert due == reply_at + timedelta(minutes=3)
    assert state["awaiting_passive_reply"] is False


def test_hint_is_skipped_when_it_was_used_in_the_last_three_sends():
    assert (
        companion.choose_companion_action(
            ["social", "gentle_hint"],
            has_focus=True,
        )
        == "ambient"
    )
    assert (
        companion.choose_companion_action(
            ["social", "ambient", "care"],
            has_focus=True,
        )
        == "gentle_hint"
    )
    assert (
        companion.choose_companion_action(
            ["social", "ambient", "care"],
            has_focus=False,
        )
        == "social"
    )


async def test_two_minute_gap_does_not_generate_a_message(monkeypatch):
    now = datetime.now(UTC)
    anchor = now - timedelta(seconds=20)
    _install_common(
        monkeypatch,
        messages=[],
        last_ai_at=anchor,
    )
    generate = companion._generate_companion_message

    sent = await companion._process_activity(
        _activity(
            state=_open_window(
                window_index=0,
                anchor_at=anchor.isoformat(),
                window_start=anchor.isoformat(),
                window_end=(anchor + timedelta(minutes=1)).isoformat(),
                due_at=(anchor + timedelta(seconds=5)).isoformat(),
                probability=0.80,
            )
        )
    )

    assert sent is False
    generate.assert_not_awaited()
    saved = companion.repo.save_companion_decision.await_args.kwargs
    assert saved["state"]["activity_sent"] == 0
    assert saved["state"]["window_index"] == 1
    assert saved["next_companion_at"] is not None


async def test_non_idle_defers_without_counting(monkeypatch):
    _install_common(monkeypatch, messages=[], idle=False)
    generate = companion._generate_companion_message

    assert await companion._process_activity(_activity(state=_open_window())) is False
    generate.assert_not_awaited()
    saved = companion.repo.save_companion_decision.await_args.kwargs
    assert saved["state"]["activity_sent"] == 0
    assert saved["state"]["window_index"] == 2


async def test_social_companion_message_is_sent_and_counted(monkeypatch):
    _install_common(monkeypatch, messages=[])
    monkeypatch.setattr(
        companion.repo,
        "list_untriggered_conditions",
        AsyncMock(return_value=[_condition()]),
    )
    monkeypatch.setattr(
        recognition,
        "_guard_visible_message",
        AsyncMock(return_value="慢慢逛，舒服就多待一会儿。"),
    )
    reserve = AsyncMock(return_value=True)
    monkeypatch.setattr(companion.repo, "reserve_companion_send", reserve)
    monkeypatch.setattr(companion.repo, "mark_companion_sent", AsyncMock())
    emit = AsyncMock(return_value="message-1")
    monkeypatch.setattr(companion.chat_emit, "emit_assistant", emit)

    assert await companion._process_activity(_activity(state=_open_window())) is True
    emit.assert_awaited_once()
    kwargs = reserve.await_args.kwargs
    assert kwargs["state"]["activity_sent"] == 0
    assert kwargs["state"]["segment_sent"] == 0
    assert kwargs["next_companion_at"] > datetime.now(UTC) + timedelta(minutes=10)
    assert (
        emit.await_args.kwargs["extra_metadata"]["offline_companion_delivery_key"]
        == kwargs["delivery_key"]
    )
    marked = companion.repo.mark_companion_sent.await_args.args
    assert marked[2]["activity_sent"] == 1
    assert marked[2]["segment_sent"] == 1
    assert marked[2]["recent_modes"] == ["social"]


async def test_activity_cap_does_not_send(monkeypatch):
    _install_common(monkeypatch, messages=[])
    generate = companion._generate_companion_message

    sent = await companion._process_activity(
        _activity(state=_open_window(activity_sent=5))
    )

    assert sent is False
    generate.assert_not_awaited()
    saved = companion.repo.save_companion_decision.await_args.kwargs
    assert saved["next_companion_at"] is None
    assert saved["state"]["phase"] == "stopped"


async def test_claim_fence_blocks_send_after_user_interaction(monkeypatch):
    _install_common(monkeypatch, messages=[])
    monkeypatch.setattr(
        companion.repo,
        "list_untriggered_conditions",
        AsyncMock(return_value=[_condition()]),
    )
    monkeypatch.setattr(
        recognition,
        "_guard_visible_message",
        AsyncMock(return_value="走累了就歇一会儿。"),
    )
    monkeypatch.setattr(
        companion.repo,
        "reserve_companion_send",
        AsyncMock(return_value=False),
    )
    emit = AsyncMock()
    monkeypatch.setattr(companion.chat_emit, "emit_assistant", emit)

    assert await companion._process_activity(_activity(state=_open_window())) is False
    emit.assert_not_awaited()


async def test_emit_exception_does_not_reschedule_when_message_was_persisted(
    monkeypatch,
):
    _install_common(monkeypatch, messages=[])
    monkeypatch.setattr(
        companion.repo,
        "list_untriggered_conditions",
        AsyncMock(return_value=[_condition()]),
    )
    monkeypatch.setattr(
        recognition,
        "_guard_visible_message",
        AsyncMock(return_value="慢慢逛，我陪着你。"),
    )
    monkeypatch.setattr(
        companion.repo,
        "reserve_companion_send",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        companion.chat_emit,
        "emit_assistant",
        AsyncMock(side_effect=RuntimeError("ws failed after insert")),
    )
    monkeypatch.setattr(
        companion.repo,
        "companion_message_exists",
        AsyncMock(return_value=True),
    )
    restore = AsyncMock()
    monkeypatch.setattr(companion.repo, "restore_companion_schedule", restore)

    with pytest.raises(RuntimeError):
        await companion._process_activity(_activity(state=_open_window()))

    restore.assert_not_awaited()


async def test_failed_emit_restores_the_uncounted_window(monkeypatch):
    _install_common(monkeypatch, messages=[])
    monkeypatch.setattr(
        companion.repo,
        "list_untriggered_conditions",
        AsyncMock(return_value=[_condition()]),
    )
    monkeypatch.setattr(
        recognition,
        "_guard_visible_message",
        AsyncMock(return_value="慢慢逛，我陪着你。"),
    )
    monkeypatch.setattr(
        companion.repo,
        "reserve_companion_send",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        companion.chat_emit,
        "emit_assistant",
        AsyncMock(return_value=""),
    )
    restore = AsyncMock()
    monkeypatch.setattr(companion.repo, "restore_companion_schedule", restore)

    assert await companion._process_activity(_activity(state=_open_window())) is False
    restored = restore.await_args.kwargs["state"]
    assert restored["activity_sent"] == 0
    assert restored["segment_sent"] == 0


async def test_guarded_emit_drops_message_when_user_replied_after_reservation(
    monkeypatch,
):
    query = AsyncMock(return_value=[])
    monkeypatch.setattr(proactive_emit.db, "query_raw", query)
    send = AsyncMock()
    monkeypatch.setattr(proactive_emit.manager, "send_to_workspace", send)

    result = await proactive_emit.emit_proactive_message(
        conversation_id="conversation-1",
        user_id="user-1",
        agent_id="agent-1",
        workspace_id="workspace-1",
        message="慢慢逛，我陪着你。",
        trigger_type="offline_activity_companion_social",
        extra_metadata={
            "offline_companion_delivery_key": "delivery-1",
        },
        guard_activity_id="activity-1",
        guard_delivery_key="delivery-1",
    )

    assert result == ""
    send.assert_not_awaited()
    sql = query.await_args.args[0]
    assert "user_message.created_at" in sql
    assert "activity.status = 'accepted'" in sql


async def test_companion_provider_failure_is_not_counted_as_normal_silence(
    monkeypatch,
):
    monkeypatch.setattr(
        companion,
        "get_prompt_text",
        AsyncMock(return_value=defaults.OFFLINE_ACTIVITY_COMPANION_MESSAGE_PROMPT),
    )
    monkeypatch.setattr(
        companion,
        "invoke_text",
        AsyncMock(side_effect=RuntimeError("provider down")),
    )

    with pytest.raises(companion.CompanionDecisionGenerationError):
        await companion._generate_companion_message(
            action="social",
            activity=_activity(),
            ctx={},
            messages=[],
            safe_hint="",
        )


async def test_generation_failure_retries_inside_the_window(monkeypatch):
    _install_common(monkeypatch, messages=[])
    monkeypatch.setattr(
        companion.repo,
        "list_untriggered_conditions",
        AsyncMock(return_value=[_condition()]),
    )
    monkeypatch.setattr(
        companion,
        "_generate_companion_message",
        AsyncMock(side_effect=RuntimeError("provider down")),
    )

    assert await companion._process_activity(_activity(state=_open_window())) is False
    saved = companion.repo.save_companion_decision.await_args.kwargs
    assert saved["state"]["activity_sent"] == 0
    assert saved["state"]["window_index"] == 2
    assert saved["next_companion_at"] is not None


async def test_arrival_guide_starts_opening_once(monkeypatch):
    monkeypatch.setattr(
        companion.repo,
        "get_activity",
        AsyncMock(return_value={"reached": True, "companion_state": {}}),
    )
    schedule = AsyncMock()
    monkeypatch.setattr(companion.repo, "schedule_opening_unless_replied", schedule)

    await companion.start_opening_segment("activity-1", "user-1")

    state = schedule.await_args.args[1]
    due = schedule.await_args.args[2]
    assert state["phase"] == "opening"
    assert state["segment_sent"] == 0
    assert state["activity_sent"] == 0
    assert due > datetime.now(UTC) - timedelta(seconds=1)
    assert due < datetime.now(UTC) + timedelta(minutes=1)

    schedule.reset_mock()
    monkeypatch.setattr(
        companion.repo,
        "get_activity",
        AsyncMock(
            return_value={
                "reached": True,
                "companion_state": {"phase": "opening", "awaiting_passive_reply": True},
            }
        ),
    )
    await companion.start_opening_segment("activity-1", "user-1")
    schedule.assert_not_awaited()


async def test_user_reply_cancels_pending_companion_without_starting_followup(
    monkeypatch,
):
    monkeypatch.setattr(
        companion.repo,
        "get_current_reached_activity",
        AsyncMock(
            return_value={
                "id": "activity-1",
                "companion_state": _open_window(activity_sent=1, segment_sent=1),
            }
        ),
    )
    cancel = AsyncMock()
    monkeypatch.setattr(companion.repo, "cancel_pending_companion", cancel)

    await companion.note_user_interaction("user-1", "workspace-1")

    cancel.assert_awaited_once()
    assert cancel.await_args.args[0] == "activity-1"
    assert cancel.await_args.args[1]


async def test_passive_reply_starts_followup_after_the_reply_is_stored(monkeypatch):
    monkeypatch.setattr(
        companion.db,
        "conversation",
        SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    userId="user-1",
                    workspaceId="workspace-1",
                )
            )
        ),
    )
    monkeypatch.setattr(
        companion.repo,
        "get_current_reached_activity",
        AsyncMock(
            return_value={
                "id": "activity-1",
                "companion_state": {
                    "phase": "opening",
                    "activity_sent": 1,
                    "segment_sent": 1,
                    "recent_modes": ["social"],
                    "awaiting_passive_reply": True,
                    "reply_token": "token-user-1",
                },
            }
        ),
    )
    schedule = AsyncMock()
    monkeypatch.setattr(companion.repo, "schedule_followup_if_token", schedule)
    monkeypatch.setattr(companion, "_wait_minutes", lambda rng=None: 2)
    monkeypatch.setattr(companion, "_unit", lambda rng=None: 0.0)

    before = datetime.now(UTC)
    await companion.note_passive_reply("conversation-1")

    state = schedule.await_args.args[1]
    due = schedule.await_args.args[2]
    assert state["phase"] == "followup"
    assert state["segment_sent"] == 0
    assert state["activity_sent"] == 1
    assert state["awaiting_passive_reply"] is False
    assert schedule.await_args.args[3] == "token-user-1"
    assert due >= before + timedelta(minutes=2)


async def test_passive_reply_ignores_a_turn_that_did_not_cancel_companion(
    monkeypatch,
):
    monkeypatch.setattr(
        companion.db,
        "conversation",
        SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(
                    userId="user-1",
                    workspaceId="workspace-1",
                )
            )
        ),
    )
    monkeypatch.setattr(
        companion.repo,
        "get_current_reached_activity",
        AsyncMock(
            return_value={
                "id": "activity-1",
                "companion_state": _open_window(),
            }
        ),
    )
    schedule = AsyncMock()
    monkeypatch.setattr(companion.repo, "schedule_followup_if_token", schedule)

    await companion.note_passive_reply("conversation-1")

    schedule.assert_not_awaited()


def test_companion_clock_updates_are_fenced_in_sql():
    cancel_sql = inspect.getsource(offline_repo.cancel_pending_companion)
    opening_sql = inspect.getsource(offline_repo.schedule_opening_unless_replied)
    followup_sql = inspect.getsource(offline_repo.schedule_followup_if_token)
    restore_sql = inspect.getsource(offline_repo.restore_companion_schedule)
    mark_sql = inspect.getsource(offline_repo.mark_companion_sent)

    assert "jsonb_build_object" in cancel_sql
    assert "reply_token" in cancel_sql
    assert "companion_state = $2::jsonb" not in cancel_sql
    assert "awaiting_passive_reply" in opening_sql
    assert "reply_token" in followup_sql
    assert "companion_state->>'activity_sent'" in followup_sql
    assert "awaiting_passive_reply" in restore_sql
    assert "reply_token" in restore_sql
    assert "delivery_key" in mark_sql
    assert "awaiting_passive_reply" in mark_sql
    assert "reply_token" in mark_sql


async def test_disabled_companion_does_not_take_over_proactive_channel(monkeypatch):
    monkeypatch.setattr(
        companion.settings,
        "offline_activity_companion_enabled",
        False,
    )
    current = AsyncMock()
    monkeypatch.setattr(companion.repo, "get_current_reached_activity", current)

    assert (
        await companion.can_take_over_proactive_channel("user-1", "workspace-1")
        is False
    )
    current.assert_not_awaited()


async def test_ready_activity_takes_over_proactive_channel(monkeypatch):
    monkeypatch.setattr(
        companion.settings,
        "offline_activity_companion_enabled",
        True,
    )
    monkeypatch.setattr(
        companion,
        "is_activity_enabled",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        companion.repo,
        "get_current_reached_activity",
        AsyncMock(return_value={"conditions_ready_at": "2026-09-22T08:00:00Z"}),
    )

    assert (
        await companion.can_take_over_proactive_channel(
            "user-1",
            "workspace-1",
        )
        is True
    )


async def test_condition_recovery_runs_when_companion_messages_are_disabled(
    monkeypatch,
):
    monkeypatch.setattr(
        companion.settings,
        "offline_activity_companion_enabled",
        False,
    )
    monkeypatch.setattr(
        companion,
        "is_activity_enabled",
        AsyncMock(return_value=True),
    )
    recovery = AsyncMock(
        return_value={"scanned": 1, "recovered": 1, "failed": 0}
    )
    monkeypatch.setattr(
        shooting_conditions,
        "recover_unready_reached_activities",
        recovery,
    )
    monkeypatch.setattr(
        shooting_conditions,
        "recover_missing_prewritten_fragments",
        AsyncMock(return_value={"scanned": 0, "repaired": 0, "failed": 0}),
    )
    claim = AsyncMock()
    monkeypatch.setattr(companion.repo, "claim_due_companion_activities", claim)

    result = await companion.scan_activity_companions()

    assert result["recovered"] == 1
    recovery.assert_awaited_once()
    claim.assert_not_awaited()


async def test_condition_generation_uses_atomic_replace(monkeypatch):
    generated = [
        {"short_name": "花", "category": "植物"},
        {"short_name": "天空", "category": "天空"},
        {"short_name": "建筑", "category": "建筑"},
    ]
    created = [
        {**item, "id": f"condition-{index}", "guidance_profile": {}}
        for index, item in enumerate(generated)
    ]
    monkeypatch.setattr(
        shooting_conditions,
        "_generate_items",
        AsyncMock(return_value=generated),
    )
    replace = AsyncMock(return_value=(created, True))
    monkeypatch.setattr(
        shooting_conditions.repo,
        "replace_shooting_items_and_initialize",
        replace,
    )
    monkeypatch.setattr(
        shooting_conditions,
        "fire_background",
        lambda coroutine: coroutine.close(),
    )

    await shooting_conditions.generate_items_for_activity(
        {
            "id": "activity-1",
            "user_id": "user-1",
            "status": "accepted",
            "reached": True,
            "conditions_ready_at": None,
        }
    )

    replace.assert_awaited_once()
    assert len(replace.await_args.kwargs["items"]) == 3


async def test_condition_recovery_continues_after_one_failure(monkeypatch):
    activities = [{"id": "a1"}, {"id": "a2"}]
    monkeypatch.setattr(
        shooting_conditions.repo,
        "list_unready_reached_activities",
        AsyncMock(return_value=activities),
    )
    generate = AsyncMock(side_effect=[RuntimeError("boom"), None])
    monkeypatch.setattr(
        shooting_conditions,
        "generate_items_for_activity",
        generate,
    )

    result = await shooting_conditions.recover_unready_reached_activities()

    assert result == {"scanned": 2, "recovered": 1, "failed": 1}


async def test_fragment_pregen_repairs_only_missing_tiers(monkeypatch):
    monkeypatch.setattr(
        fragment_pregen.repo,
        "list_prewritten_condition_tiers",
        AsyncMock(return_value={("condition-1", "rare")}),
    )
    monkeypatch.setattr(
        fragment_pregen.repo,
        "ai_memory_brief",
        AsyncMock(return_value="一段经历"),
    )
    prewrite = AsyncMock(return_value="预生成内容")
    monkeypatch.setattr(fragment_pregen, "_prewrite", prewrite)
    create = AsyncMock()
    monkeypatch.setattr(
        fragment_pregen.repo,
        "create_prewritten_fragments",
        create,
    )

    await fragment_pregen.pregenerate_for_activity(
        {"id": "activity-1", "user_id": "user-1", "title": "散步"},
        [{"id": "condition-1", "short_name": "花", "category": "植物"}],
    )

    assert prewrite.await_count == 2
    tiers = create.await_args.args[2]
    assert set(tiers) == {"epic", "legendary"}


async def test_missing_prewritten_recovery_is_persistent(monkeypatch):
    monkeypatch.setattr(
        shooting_conditions.repo,
        "list_ready_activities_with_missing_prewritten",
        AsyncMock(return_value=[{"id": "activity-1"}]),
    )
    monkeypatch.setattr(
        shooting_conditions.repo,
        "list_all_conditions",
        AsyncMock(return_value=[{"id": "condition-1"}]),
    )
    pregenerate = AsyncMock(return_value={"generated": 2, "missing": 0})
    monkeypatch.setattr(
        shooting_conditions.fragment_pregen,
        "pregenerate_for_activity",
        pregenerate,
    )

    result = await shooting_conditions.recover_missing_prewritten_fragments()

    assert result == {"scanned": 1, "repaired": 1, "failed": 0}
    pregenerate.assert_awaited_once()


def test_single_reached_activity_is_database_backed():
    mark_arrived_source = inspect.getsource(offline_repo.mark_arrived)
    migration = (
        Path(__file__).parents[1]
        / "prisma/migrations/20260922170000_offline_activity_companion/migration.sql"
    ).read_text()

    assert "pg_advisory_xact_lock" in mark_arrived_source
    assert "offline_activity_one_reached_workspace_key" in migration
    assert "offline_activity_one_reached_legacy_user_key" in migration


def test_guidance_profile_removes_hidden_target_words():
    profile = normalize_guidance_profile(
        short_name="花",
        category="植物",
        raw_profile={
            "aliases": ["花朵"],
            "guidance": {
                "weak": "留意一下附近的植物",
                "medium": "留意鲜艳颜色",
                "strong": "拍一朵花",
            },
        },
    )
    condition = {
        "short_name": "花",
        "category": "植物",
        "guidance_profile": profile,
    }

    assert not contains_hidden_target(profile["guidance"]["weak"], [condition])
    assert not contains_hidden_target(profile["guidance"]["medium"], [condition])
    assert not contains_hidden_target(profile["guidance"]["strong"], [condition])
    assert "植物" not in profile["guidance"]["weak"]
    assert safe_guidance(condition, "weak")


def test_all_new_companion_prompts_are_managed_by_defaults_registry():
    expected = {
        "offline.photo_match": defaults.OFFLINE_PHOTO_MATCH_PROMPT,
        "offline.photo_followup_transition": (
            defaults.OFFLINE_PHOTO_FOLLOWUP_TRANSITION_PROMPT
        ),
        "offline.photo_followup_near": (
            defaults.OFFLINE_PHOTO_FOLLOWUP_NEAR_PROMPT
        ),
        "offline.photo_followup_free_roam": (
            defaults.OFFLINE_PHOTO_FOLLOWUP_FREE_ROAM_PROMPT
        ),
        "offline.activity_companion_decision": (
            defaults.OFFLINE_ACTIVITY_COMPANION_DECISION_PROMPT
        ),
        "offline.activity_companion_message": (
            defaults.OFFLINE_ACTIVITY_COMPANION_MESSAGE_PROMPT
        ),
        "offline.safe_rewrite": defaults.OFFLINE_SAFE_REWRITE_PROMPT,
    }

    for key, default_text in expected.items():
        assert PROMPT_DEFINITION_MAP[key].default_text == default_text


def test_user_visible_prompts_do_not_accept_hidden_target_placeholders():
    visible = (
        defaults.OFFLINE_MISS_HINT_PROMPT
        + defaults.OFFLINE_PHOTO_FOLLOWUP_TRANSITION_PROMPT
        + defaults.OFFLINE_PHOTO_FOLLOWUP_NEAR_PROMPT
        + defaults.OFFLINE_PHOTO_FOLLOWUP_FREE_ROAM_PROMPT
        + defaults.OFFLINE_ACTIVITY_COMPANION_MESSAGE_PROMPT
    )

    assert "{hintable_items}" not in visible
    assert "{matched_item}" not in visible
    assert "{short_name}" not in visible
    assert "{completed_count}" not in visible
    assert "{total_count}" not in visible
    assert "{completed}" not in visible
    assert "{switched}" not in visible
    assert "{has_next}" not in visible
    assert "{miss_count}" not in visible
    assert "{hint_count}" not in visible
    assert "{activity_phase}" not in visible
    assert "{unanswered_count}" not in visible


def test_criteria_synonym_is_part_of_hidden_target_guard():
    condition = {
        "short_name": "长椅",
        "category": "设施",
        "criteria": "画面主体清楚呈现座椅或长凳",
        "guidance_profile": {"aliases": []},
    }

    assert contains_hidden_target("留意一下附近的座椅", [condition])
    assert contains_hidden_target("找找长凳周围的细节", [condition])


async def test_generic_proactive_is_deferred_during_arrived_activity(monkeypatch):
    monkeypatch.setattr(
        proactive_orchestrator.db,
        "query_raw",
        AsyncMock(return_value=[{"status": "active"}]),
    )
    monkeypatch.setattr(
        companion,
        "can_take_over_proactive_channel",
        AsyncMock(return_value=True),
    )
    advance = AsyncMock()
    monkeypatch.setattr(proactive_orchestrator, "advance_to_next_window", advance)
    state = SimpleNamespace(
        workspace_id="workspace-1",
        user_id="user-1",
        stage="P3",
    )

    await proactive_orchestrator._process_due_state(state)

    advance.assert_awaited_once()
    assert advance.await_args.kwargs["payload"]["reason"] == "offline_activity_active"
