"""Achievement notification/history parity against isolated PostgreSQL and HTTP."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from app.api import ownership
from app.api.public import achievements, conversations
from app.services.achievements import engine, repository
from app.services.achievements.events import MemoryChangelogAchievementEvent
from app.services.achievements.rules import memory_rules
from app.services.auth import create_jwt
from app.services.notifications import service as notifications
from tests.test_memory_evidence_postgres import origins

AT = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)


@pytest.fixture
async def timeline(origins, monkeypatch):
    db, uid, wid, _, message, _, _, users, agents, spaces, convs = origins
    for module in (repository, memory_rules, conversations, achievements, ownership):
        monkeypatch.setattr(module, "db", db)
    for module in (engine, repository):
        monkeypatch.setattr(module, "achievement_evaluation_enabled", AsyncMock(return_value=True))
    for module in (repository, conversations):
        monkeypatch.setattr(module, "achievement_alerts_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(achievements, "achievement_display_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(repository, "_is_unlock_cached", AsyncMock(return_value=False))
    monkeypatch.setattr(repository, "_cache_unlocked_achievements", AsyncMock())
    monkeypatch.setattr(notifications, "notify_achievement_unlocked", AsyncMock())
    monkeypatch.setattr(repository.manager, "send_event", AsyncMock(return_value=True))
    monkeypatch.setattr(repository.manager, "send_to_workspace", AsyncMock(return_value=1))
    for conv in convs:
        await db.conversation.update(where={"id": conv}, data={"createdAt": AT})
    await db.message.update(where={"id": message.id}, data={"createdAt": AT + timedelta(minutes=1)})
    app = FastAPI()
    app.include_router(conversations.router)
    app.include_router(achievements.router)
    return db, uid, wid, message, users, agents, spaces, convs, app


async def history(app, uid, conversation_id, **params):
    # Each call is a fresh HTTP client, without the live WS/local message state.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://isolated.test") as client:
        return await client.get(
            f"/conversations/{conversation_id}/messages",
            headers={"Authorization": "Bearer " + create_jwt(uid, role="user")},
            params={"include_achievements": "true", **params},
        )


async def legacy_unlock(db, uid, aid, wid, achievement_id, at, conversation_id=None):
    await db.execute_raw(
        """INSERT INTO achievement_unlocks
           (user_id, agent_id, workspace_id, conversation_id, achievement_id, unlocked_at)
           VALUES ($1, $2, $3, $4, $5, $6::timestamp)""",
        uid, aid, wid, conversation_id, achievement_id, at,
    )


@pytest.mark.asyncio
async def test_memory_and_chat_unlocks_both_survive_cold_history_load(timeline):
    db, uid, wid, message, users, agents, spaces, convs, app = timeline
    for i in range(20):
        await db.usermemory.create(data={
            "userId": uid, "workspaceId": wid, "content": f"合成喜好{i}",
            "mainCategory": "偏好", "subCategory": "兴趣偏好", "level": 2,
        })
    preference = await db.usermemory.find_first(where={"userId": uid, "mainCategory": "偏好"})
    await engine.handle_achievement_event(MemoryChangelogAchievementEvent(uid, preference.id, "create", wid))
    assert await repository.unlock_achievement(
        user_id=uid, agent_id=agents[0], workspace_id=wid,
        conversation_id=convs[0], achievement_id=59,
    )
    rows = await db.query_raw(
        "SELECT achievement_id, conversation_id FROM achievement_unlocks WHERE user_id=$1", uid
    )
    assert {r["achievement_id"] for r in rows} == {59, 65}
    assert {r["conversation_id"] for r in rows} == {convs[0]}
    assert {call.args[2]["achievement_id"] for call in repository.manager.send_event.await_args_list} == {59, 65}
    assert {call.args[0] for call in repository.manager.send_event.await_args_list} == {convs[0]}
    repository.manager.send_to_workspace.assert_not_awaited()
    for _ in range(2):
        response = await history(app, uid, convs[0])
        assert response.status_code == 200
        notices = [r for r in response.json() if r["role"] == "achievement"]
        assert {r["metadata"]["achievement"]["achievement_id"] for r in notices} == {59, 65}
        assert len(notices) == len({r["id"] for r in notices}) == 2
    assert repository.manager.send_event.await_count == 2  # Loading history never re-notifies.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://isolated.test") as client:
        response = await client.get(
            "/achievements", params={"agent_id": agents[0]},
            headers={"Authorization": "Bearer " + create_jwt(uid, role="user")},
        )
    assert response.status_code == 200
    listing = response.json()
    assert listing["unlocked"] == 2 and listing["score"] == 460


@pytest.mark.asyncio
async def test_legacy_workspace_unlock_restored_alongside_same_time_chat_unlock(timeline):
    db, uid, wid, message, users, agents, spaces, convs, app = timeline
    at = AT + timedelta(minutes=2)
    await legacy_unlock(db, uid, agents[0], wid, 65, at)
    await legacy_unlock(db, uid, agents[0], wid, 59, at, convs[0])
    first = await history(app, uid, convs[0], limit=1)
    second = await history(app, uid, convs[0], limit=1)
    assert first.status_code == second.status_code == 200
    notices = [r for r in first.json() if r["role"] == "achievement"]
    assert {r["content"] for r in notices} == {"喜好记录大师", "表情包大户"}
    assert len({r["id"] for r in notices}) == 2
    assert [r["id"] for r in first.json()] == [r["id"] for r in second.json()]
    assert all(r["metadata"]["achievement"]["unlocked"] for r in notices)
    # Read-only compatibility: no backdated DB changes or repeated notifications.
    assert (await db.query_raw("SELECT conversation_id FROM achievement_unlocks WHERE user_id=$1 AND achievement_id=65", uid)) == [{"conversation_id": None}]
    repository.manager.send_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_history_scope_and_visibility_guards(timeline, monkeypatch):
    db, uid, wid, message, users, agents, spaces, convs, app = timeline
    at = AT + timedelta(minutes=2)
    await legacy_unlock(db, uid, agents[0], wid, 65, at)
    await legacy_unlock(db, uid, agents[0], spaces[1], 59, at)  # Other workspace/agent.
    await legacy_unlock(db, users[-1], agents[-1], spaces[-1], 49, at)
    await legacy_unlock(db, uid, agents[0], None, 66, at)  # No workspace evidence.
    await legacy_unlock(db, uid, agents[0], wid, 3, at)  # Disabled definition.
    await legacy_unlock(db, uid, agents[0], wid, 29, AT - timedelta(seconds=1))  # Predates chat.
    response = await history(app, uid, convs[0])
    assert response.status_code == 200
    assert [r["content"] for r in response.json() if r["role"] == "achievement"] == ["喜好记录大师"]
    assert (await history(app, users[-1], convs[0])).status_code == 403
    compact = await history(app, uid, convs[0], include_metadata="false")
    assert [r["metadata"] for r in compact.json()] == [None, None]
    hidden = await history(app, uid, convs[0], include_achievements="false")
    assert not any(r["role"] == "achievement" for r in hidden.json())
    monkeypatch.setattr(conversations, "achievement_alerts_enabled", AsyncMock(return_value=False))
    assert not any(r["role"] == "achievement" for r in (await history(app, uid, convs[0])).json())


@pytest.mark.asyncio
async def test_old_legacy_unlock_does_not_move_to_recreated_chat(timeline):
    db, uid, wid, message, users, agents, spaces, convs, app = timeline
    await legacy_unlock(db, uid, agents[0], wid, 65, AT + timedelta(minutes=2))
    await db.conversation.update(where={"id": convs[0]}, data={"isDeleted": True, "archivedAt": AT + timedelta(minutes=3)})
    replacement = await db.conversation.create(data={
        "userId": uid, "agentId": agents[0], "workspaceId": wid,
        "createdAt": AT + timedelta(minutes=4),
    })
    convs.append(replacement.id)
    await legacy_unlock(db, uid, agents[0], wid, 59, AT + timedelta(minutes=5))
    response = await history(app, uid, replacement.id)
    assert response.status_code == 200
    assert [r["content"] for r in response.json()] == ["表情包大户"]
    assert (await history(app, uid, convs[0])).status_code == 404
    assert (await repository.list_achievements(uid, agents[0]))["unlocked"] == 2


@pytest.mark.asyncio
async def test_workspace_binding_is_atomic_and_duplicate_attempts_stay_idempotent(timeline):
    db, uid, wid, message, users, agents, spaces, convs, app = timeline
    async def unlock():
        return await repository.unlock_achievement(
            user_id=uid, agent_id=agents[0], workspace_id=wid,
            achievement_id=65, notify=False,
        )
    assert sorted(await asyncio.gather(unlock(), unlock(), unlock())) == [False, False, True]
    assert await db.query_raw(
        "SELECT conversation_id FROM achievement_unlocks WHERE user_id=$1", uid,
    ) == [{"conversation_id": convs[0]}]


@pytest.mark.asyncio
async def test_workspace_fallback_never_binds_unrelated_or_deleted_conversation(timeline):
    db, uid, wid, message, users, agents, spaces, convs, app = timeline
    await db.conversation.update(where={"id": convs[0]}, data={"isDeleted": True})
    for aid, workspace_id, achievement_id in (
        (agents[0], wid, 65),  # No live conversation.
        (agents[0], spaces[1], 59),  # Foreign agent's workspace.
        (agents[0], spaces[-1], 49),  # Other user's workspace.
    ):
        assert await repository.unlock_achievement(
            user_id=uid, agent_id=aid, workspace_id=workspace_id,
            achievement_id=achievement_id, notify=False,
        )
    assert {r["conversation_id"] for r in await db.query_raw(
        "SELECT conversation_id FROM achievement_unlocks WHERE user_id=$1", uid,
    )} == {None}
