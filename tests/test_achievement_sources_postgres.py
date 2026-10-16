"""Source/clock and duplicate/failure contracts against disposable PostgreSQL."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prisma import Json
import pytest

from app.api.admin import agents as admin
from app.services.auth import create_jwt
from app.services.achievements import engine, repository, source
from app.services.achievements.events import (
    UserMessageAchievementEvent,
    AssistantMessageAchievementEvent,
    AssistantTurnAchievementEvent,
    AggregationAchievementEvent,
    IntentAchievementEvent,
)
from app.services.achievements.rules import user_message_rules, assistant_message_rules
from app.services import interaction_streak
from tests.test_memory_evidence_postgres import origins

AT = datetime(2026, 10, 15, 15, 59, 59, tzinfo=timezone.utc)


@pytest.fixture
async def achieved(origins, monkeypatch):
    db, uid, wid, _, message, _, _, users, agents, spaces, convs = origins
    for module in [
        source,
        repository,
        admin,
        user_message_rules,
        assistant_message_rules,
        interaction_streak,
    ]:
        monkeypatch.setattr(module, "db", db)
    for module in [engine, repository]:
        monkeypatch.setattr(
            module, "achievement_evaluation_enabled", AsyncMock(return_value=True)
        )
    monkeypatch.setattr(
        repository, "achievement_alerts_enabled", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(repository, "_is_unlock_cached", AsyncMock(return_value=False))
    monkeypatch.setattr(repository, "_cache_unlocked_achievements", AsyncMock())
    await db.message.update(
        where={"id": message.id}, data={"content": "哈哈", "createdAt": AT}
    )
    event = UserMessageAchievementEvent(
        uid,
        agents[0],
        wid,
        convs[0],
        message.id,
        "伪造长回调",
        occurred_at=datetime.now(timezone.utc),
    )
    return db, event, users, agents, spaces, convs


async def record_bound(event, **kwargs):
    event, identity = await source.resolve_message_event(event)
    token = source.current_source.set(identity)
    try:
        return await repository.record_event(
            user_id=identity.user_id,
            agent_id=identity.agent_id,
            workspace_id=identity.workspace_id,
            conversation_id=identity.conversation_id,
            event_type="user_message",
            source_id=identity.message_id,
            **kwargs,
        )
    finally:
        source.current_source.reset(token)


@pytest.mark.asyncio
async def test_duplicate_concurrent_replay_uses_original_clock_and_counts_once(
    achieved,
):
    db, event, *_ = achieved
    results = await asyncio.gather(
        *(record_bound(event, value_int=2) for _ in range(4))
    )
    assert sorted(results) == [False, False, False, True]
    rows = await db.query_raw(
        "SELECT * FROM achievement_events WHERE source_id=$1", event.message_id
    )
    assert len(rows) == 1 and source._aware(rows[0]["occurred_at"]) == AT
    assert rows[0]["metadata"] == {
        "source_contract": source.CONTRACT,
        "source_message_id": event.message_id,
        "source_role": "user",
    }
    assert (
        await repository._event_count(event.user_id, event.agent_id, "user_message")
        == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["deleted", "archived", "wrong_role", "human"])
async def test_source_becoming_unavailable_blocks_writes_and_unlocks(
    achieved, mutation
):
    db, event, users, agents, spaces, convs = achieved
    _, identity = await source.resolve_message_event(event)
    if mutation == "deleted":
        await db.conversation.update(where={"id": convs[0]}, data={"isDeleted": True})
    elif mutation == "archived":
        await db.chatworkspace.update(
            where={"id": spaces[0]}, data={"status": "archived"}
        )
    elif mutation == "wrong_role":
        await db.message.update(
            where={"id": event.message_id}, data={"role": "assistant"}
        )
    else:
        await db.message.update(
            where={"id": event.message_id},
            data={"metadata": Json({"human_takeover": True})},
        )
    with pytest.raises(source.AchievementSourceUnavailable):
        await source.resolve_message_event(event)
    token = source.current_source.set(identity)
    try:
        assert not await repository.record_event(
            user_id=event.user_id,
            agent_id=event.agent_id,
            workspace_id=event.workspace_id,
            conversation_id=event.conversation_id,
            event_type="user_message",
            source_id=event.message_id,
        )
        assert not await repository.unlock_achievement(
            user_id=event.user_id,
            agent_id=event.agent_id,
            workspace_id=event.workspace_id,
            conversation_id=event.conversation_id,
            achievement_id=1,
        )
    finally:
        source.current_source.reset(token)
    assert (
        await db.query_raw(
            "SELECT id FROM achievement_events WHERE user_id=$1", event.user_id
        )
        == []
    )
    repository._cache_unlocked_achievements.assert_not_awaited()


@pytest.mark.asyncio
async def test_cross_user_agent_conversation_cannot_supply_the_source(achieved):
    db, event, users, agents, spaces, convs = achieved
    for changes in [
        dict(user_id=users[1]),
        dict(agent_id=agents[1]),
        dict(workspace_id=spaces[2]),
        dict(conversation_id=convs[2]),
        dict(message_id="missing"),
    ]:
        with pytest.raises(source.AchievementSourceUnavailable):
            await source.resolve_message_event(replace(event, **changes))


@pytest.mark.asyncio
async def test_partial_rule_failure_can_retry_without_duplicate_event_or_unlock(
    achieved, monkeypatch
):
    db, event, *_ = achieved
    attempts = 0

    async def rule(e):
        nonlocal attempts
        await repository.record_event(
            user_id=e.user_id,
            agent_id=e.agent_id,
            workspace_id=e.workspace_id,
            conversation_id=e.conversation_id,
            event_type="user_message",
            source_id=e.message_id,
            value_int=2,
        )
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic failure after counter")
        await repository.unlock_achievement(
            user_id=e.user_id,
            agent_id=e.agent_id,
            workspace_id=e.workspace_id,
            conversation_id=e.conversation_id,
            achievement_id=1,
        )

    monkeypatch.setattr(engine, "evaluate_user_message", rule)
    with pytest.raises(RuntimeError, match="after counter"):
        await engine.handle_achievement_event(event)
    await engine.handle_achievement_event(event)
    await engine.handle_achievement_event(event)
    assert (
        len(
            await db.query_raw(
                "SELECT id FROM achievement_events WHERE user_id=$1", event.user_id
            )
        )
        == 1
    )
    assert (
        len(
            await db.query_raw(
                "SELECT id FROM achievement_unlocks WHERE user_id=$1", event.user_id
            )
        )
        == 1
    )


@pytest.mark.asyncio
async def test_real_user_rule_uses_persisted_text_and_records_derivative_time(
    achieved, monkeypatch
):
    db, event, *_ = achieved
    monkeypatch.setattr(
        user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        user_message_rules, "get_intimacy_data", AsyncMock(return_value={})
    )
    await engine.handle_achievement_event(event)
    rows = await db.query_raw(
        "SELECT achievement_id FROM achievement_unlocks WHERE user_id=$1", event.user_id
    )
    assert {r["achievement_id"] for r in rows} >= {1, 9}
    events = await db.query_raw(
        "SELECT occurred_at,value_int FROM achievement_events WHERE source_id=$1",
        event.message_id,
    )
    assert events and all(source._aware(r["occurred_at"]) == AT for r in events)
    assert any(r["value_int"] == 2 for r in events)


@pytest.mark.asyncio
async def test_assistant_turn_is_bounded_and_cannot_borrow_other_conversation(achieved):
    db, event, users, agents, spaces, convs = achieved
    metadata = Json({"achievement_turn_id": "turn"})
    assistant = await db.message.create(
        data={
            "conversationId": convs[0],
            "role": "assistant",
            "content": "好" * 98,
            "metadata": metadata,
        }
    )
    foreign_conv = await db.conversation.create(
        data={
            "userId": event.user_id,
            "agentId": event.agent_id,
            "workspaceId": None,
        }
    )
    convs.append(foreign_conv.id)
    foreign = await db.message.create(
        data={"conversationId": foreign_conv.id, "role": "user", "content": "你好"}
    )
    # Same user/agent and same turn tag in another conversation must not contribute.
    await db.message.create(
        data={
            "conversationId": foreign_conv.id,
            "role": "assistant",
            "content": "坏" * 20,
            "metadata": metadata,
        }
    )
    turn = AssistantTurnAchievementEvent(
        convs[0], assistant.id, ["fake"], [event.message_id], "turn"
    )
    await engine.handle_achievement_event(turn)
    assert (
        await db.query_raw(
            "SELECT achievement_id FROM achievement_unlocks WHERE user_id=$1",
            event.user_id,
        )
    ) == [{"achievement_id": 81}]
    for change in [
        dict(user_message_ids=[foreign.id]),
        dict(user_message_ids=[]),
        dict(turn_id="wrong"),
        dict(user_message_ids=[str(i) for i in range(101)]),
    ]:
        with pytest.raises(source.AchievementSourceUnavailable):
            await source.resolve_message_event(replace(turn, **change))


@pytest.mark.asyncio
async def test_streak_replay_after_midnight_never_marks_callback_day(achieved):
    db, event, *_ = achieved
    await asyncio.gather(
        *(
            interaction_streak.record_user_message_day_for_conversation(
                event.conversation_id,
                message_id=event.message_id,
                workspace_id=event.workspace_id,
                user_id=event.user_id,
            )
            for _ in range(4)
        )
    )
    rows = await db.query_raw(
        "SELECT local_date FROM workspace_interaction_days WHERE workspace_id=$1",
        event.workspace_id,
    )
    assert len(rows) == 1 and str(rows[0]["local_date"])[:10] == "2026-10-15"
    await interaction_streak.record_user_message_day_for_conversation(
        event.conversation_id, message_id="missing"
    )
    assert (
        len(
            await db.query_raw(
                "SELECT id FROM workspace_interaction_days WHERE workspace_id=$1",
                event.workspace_id,
            )
        )
        == 1
    )


@pytest.mark.asyncio
async def test_source_api_is_bounded_authorized_and_hides_content(achieved):
    db, event, users, agents, *_ = achieved
    await record_bound(event)
    app = FastAPI()
    app.include_router(admin.router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = f"/admin-api/agents/{event.agent_id}/achievement-sources"
        admin_headers = {
            "Authorization": "Bearer " + create_jwt(event.user_id, role="admin")
        }
        assert (await client.get(url)).status_code == 401
        assert (
            await client.get(
                url,
                headers={
                    "Authorization": "Bearer " + create_jwt(event.user_id, role="user")
                },
            )
        ).status_code == 403
        assert (
            await client.get(url + "?limit=51", headers=admin_headers)
        ).status_code == 422
        response = await client.get(url, headers=admin_headers)
        assert response.status_code == 200, response.text
        data = response.json()
        assert (
            data["count"] == 1
            and data["items"][0]["source_state"] == "persisted_message"
        )
        assert "哈哈" not in response.text and event.message_id not in response.text
        await db.message.delete(where={"id": event.message_id})
        assert (await client.get(url, headers=admin_headers)).json()["items"][0][
            "source_state"
        ] == "source_missing"
        assert (
            await repository._event_count(event.user_id, event.agent_id, "user_message")
            == 0
        )
        assert (
            await client.get(
                url.replace(event.agent_id, agents[2]), headers=admin_headers
            )
        ).json()["items"] == []
        assert (
            await client.get(
                url.replace(event.agent_id, "missing"), headers=admin_headers
            )
        ).status_code == 404
    await repository.record_event(
        user_id=event.user_id,
        agent_id=event.agent_id,
        event_type="user_message",
        source_id="legacy",
    )
    result = await source.recent_message_sources(event.user_id, event.agent_id, 1)
    assert (
        result["count"] == 1
        and result["items"][0]["source_state"] == "legacy_unverified"
    )


@pytest.mark.asyncio
async def test_actual_assistant_intent_and_aggregation_sources(achieved):
    db, event, *_ = achieved
    assistant = await db.message.create(
        data={
            "conversationId": event.conversation_id,
            "role": "assistant",
            "content": "好呀",
            "createdAt": AT,
        }
    )
    await engine.handle_achievement_event(
        AssistantMessageAchievementEvent(
            event.conversation_id, assistant.id, "fake", {"sticker_url": "fake"}
        )
    )
    await engine.handle_achievement_event(
        IntentAchievementEvent(
            "schedule_adjust",
            event.user_id,
            event.agent_id,
            event.workspace_id,
            event.conversation_id,
            event.message_id,
        )
    )
    await engine.handle_achievement_event(
        AggregationAchievementEvent(
            event.user_id, event.agent_id, event.conversation_id, event.message_id, 2
        )
    )
    rows = await db.query_raw(
        "SELECT event_type,occurred_at,workspace_id FROM achievement_events WHERE user_id=$1",
        event.user_id,
    )
    types = {r["event_type"] for r in rows}
    assert types == {
        "assistant_message",
        "assistant_short_reply",
        "schedule_adjust_request",
        "aggregation_window_completed",
    }
    assert all(
        source._aware(r["occurred_at"]) == AT
        and r["workspace_id"] == event.workspace_id
        for r in rows
    )
    # Missing role/scope cannot make an interaction day for a current user.
    await interaction_streak.record_user_message_day_for_conversation(
        event.conversation_id, message_id=assistant.id
    )
    await interaction_streak.record_user_message_day_for_conversation(
        event.conversation_id, message_id=event.message_id, user_id="other"
    )
    assert (
        await db.query_raw(
            "SELECT id FROM workspace_interaction_days WHERE workspace_id=$1",
            event.workspace_id,
        )
        == []
    )
