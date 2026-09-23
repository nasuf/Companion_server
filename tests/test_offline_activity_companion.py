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


def _install_common(monkeypatch, *, messages):
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
        companion.repo,
        "save_companion_decision",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        companion,
        "_generate_companion_message",
        AsyncMock(return_value="慢慢逛，我陪着你。"),
    )


async def test_active_conversation_suppresses_extra_message(monkeypatch):
    now = datetime.now(UTC)
    _install_common(
        monkeypatch,
        messages=[{"role": "user", "content": "我刚走到河边", "created_at": now}],
    )
    generate = AsyncMock()
    monkeypatch.setattr(companion, "_generate_decision", generate)

    assert await companion._process_activity(_activity()) is False
    generate.assert_not_awaited()
    companion.repo.save_companion_decision.assert_awaited_once()


async def test_two_unanswered_messages_pause_until_user_returns(monkeypatch):
    old = datetime.now(UTC) - timedelta(minutes=30)
    _install_common(
        monkeypatch,
        messages=[{"role": "assistant", "content": "逛得怎么样", "created_at": old}],
    )

    activity = _activity(state={"unanswered_count": 2})
    activity["last_companion_at"] = (
        datetime.now(UTC) - timedelta(minutes=10)
    ).isoformat()
    assert await companion._process_activity(activity) is False
    kwargs = companion.repo.save_companion_decision.await_args.kwargs
    assert kwargs["next_companion_at"] is None
    assert kwargs["state"]["mode"] == "paused_unanswered"


async def test_hint_ratio_guard_prevents_tasky_consecutive_push(monkeypatch):
    old = datetime.now(UTC) - timedelta(minutes=30)
    _install_common(
        monkeypatch,
        messages=[{"role": "user", "content": "继续走走", "created_at": old}],
    )
    monkeypatch.setattr(
        companion.repo,
        "list_untriggered_conditions",
        AsyncMock(return_value=[_condition()]),
    )
    monkeypatch.setattr(
        companion,
        "_generate_decision",
        AsyncMock(
            return_value={
                "action": "gentle_hint",
                "text": "换个方向看看",
                "next_delay_minutes": 12,
                "reason": "model_hint",
            }
        ),
    )
    emit = AsyncMock()
    monkeypatch.setattr(companion.chat_emit, "emit_assistant", emit)

    sent = await companion._process_activity(
        _activity(
            state={
                "recent_modes": ["social", "gentle_hint"],
                "unanswered_count": 0,
            }
        )
    )

    assert sent is False
    emit.assert_not_awaited()


async def test_social_companion_message_is_sent_and_counted(monkeypatch):
    old = datetime.now(UTC) - timedelta(minutes=30)
    _install_common(
        monkeypatch,
        messages=[{"role": "user", "content": "这里还挺舒服", "created_at": old}],
    )
    monkeypatch.setattr(
        companion.repo,
        "list_untriggered_conditions",
        AsyncMock(return_value=[_condition()]),
    )
    monkeypatch.setattr(
        companion,
        "_generate_decision",
        AsyncMock(
            return_value={
                "action": "social",
                "text": "慢慢逛，舒服就多待一会儿。",
                "next_delay_minutes": 15,
                "reason": "quiet_gap",
            }
        ),
    )
    monkeypatch.setattr(
        recognition,
        "_guard_visible_message",
        AsyncMock(return_value="慢慢逛，舒服就多待一会儿。"),
    )
    reserve = AsyncMock(return_value=True)
    monkeypatch.setattr(companion.repo, "reserve_companion_send", reserve)
    emit = AsyncMock(return_value="message-1")
    monkeypatch.setattr(companion.chat_emit, "emit_assistant", emit)

    assert await companion._process_activity(_activity()) is True
    emit.assert_awaited_once()
    kwargs = reserve.await_args.kwargs
    assert kwargs["state"]["unanswered_count"] == 1
    assert kwargs["state"]["recent_modes"] == ["social"]
    assert (
        emit.await_args.kwargs["extra_metadata"]["offline_companion_delivery_key"]
        == kwargs["delivery_key"]
    )


async def test_claim_fence_blocks_send_after_user_interaction(monkeypatch):
    old = datetime.now(UTC) - timedelta(minutes=30)
    _install_common(
        monkeypatch,
        messages=[{"role": "user", "content": "先随便走走", "created_at": old}],
    )
    monkeypatch.setattr(
        companion.repo,
        "list_untriggered_conditions",
        AsyncMock(return_value=[_condition()]),
    )
    monkeypatch.setattr(
        companion,
        "_generate_decision",
        AsyncMock(
            return_value={
                "action": "social",
                "text": "走累了就歇一会儿。",
                "next_delay_minutes": 15,
            }
        ),
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

    assert await companion._process_activity(_activity()) is False
    emit.assert_not_awaited()


async def test_emit_exception_does_not_reschedule_when_message_was_persisted(
    monkeypatch,
):
    old = datetime.now(UTC) - timedelta(minutes=30)
    _install_common(
        monkeypatch,
        messages=[{"role": "user", "content": "我在慢慢逛", "created_at": old}],
    )
    monkeypatch.setattr(
        companion.repo,
        "list_untriggered_conditions",
        AsyncMock(return_value=[_condition()]),
    )
    monkeypatch.setattr(
        companion,
        "_generate_decision",
        AsyncMock(
            return_value={
                "action": "social",
                "text": "慢慢逛，我陪着你。",
                "next_delay_minutes": 15,
            }
        ),
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
    reschedule = AsyncMock()
    monkeypatch.setattr(
        companion.repo,
        "reschedule_failed_companion_delivery",
        reschedule,
    )

    with pytest.raises(RuntimeError):
        await companion._process_activity(_activity())

    reschedule.assert_not_awaited()


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
        AsyncMock(
            return_value=defaults.OFFLINE_ACTIVITY_COMPANION_DECISION_PROMPT
        ),
    )
    monkeypatch.setattr(
        companion,
        "invoke_text",
        AsyncMock(side_effect=RuntimeError("provider down")),
    )

    with pytest.raises(companion.CompanionDecisionGenerationError):
        await companion._generate_decision(
            activity=_activity(),
            messages=[],
            safe_hint="留意颜色与明暗",
            activity_phase="guided",
            recent_modes=[],
            unanswered=0,
        )


async def test_user_interaction_resumes_paused_companion(monkeypatch):
    monkeypatch.setattr(
        companion.repo,
        "get_current_reached_activity",
        AsyncMock(
            return_value={
                "id": "activity-1",
                "status": "accepted",
                "reached": True,
            }
        ),
    )
    touch = AsyncMock()
    monkeypatch.setattr(companion.repo, "touch_activity_interaction", touch)

    await companion.note_user_interaction("user-1", "workspace-1")

    touch.assert_awaited_once()
    assert touch.await_args.kwargs["next_companion_at"] > datetime.now(UTC)


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
