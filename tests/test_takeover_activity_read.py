"""Read-only takeover activity endpoints and unchanged user review behavior."""
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from app.api.admin import takeovers
from app.api.public import offline
from app.services.admin_takeover import activities
from app.services.offline import activity_service


def row(status="pending", **overrides):
    return {
        "id": "activity", "user_id": "subject", "workspace_id": "workspace",
        "conversation_id": "conversation", "status": status,
        "title": "公园走走", "summary": "沿湖慢慢走", "description": "活动介绍",
        "created_at": "2026-10-13T00:00:00Z", "updated_at": "2026-10-13T00:00:00Z",
        **overrides,
    }


@pytest.fixture
def reads(monkeypatch):
    query = AsyncMock(return_value=[row()])
    monkeypatch.setattr(activities.db, "query_raw", query)
    monkeypatch.setattr(activities.repo, "get_activity", AsyncMock(side_effect=AssertionError("mutating read")))
    monkeypatch.setattr(activities.repo, "expire_inactive_events", AsyncMock(side_effect=AssertionError("expiry write")))
    monkeypatch.setattr(activities.repo, "get_activity_completion_feedback", AsyncMock(return_value=None))
    monkeypatch.setattr(activities.repo, "list_fragments", AsyncMock(return_value=[]))
    monkeypatch.setattr(activities.repo, "list_gallery_media", AsyncMock(return_value=[]))
    monkeypatch.setattr(activities.repo, "find_arrival_card_message_id", AsyncMock(return_value="arrival"))
    monkeypatch.setattr(activities.repo, "journey_evidence", AsyncMock(return_value={"photo_count": 1}))
    monkeypatch.setattr(activity_service, "_ensure_journey_note", AsyncMock(side_effect=AssertionError("note generation")))
    return query


@pytest.mark.parametrize("status", ["pending", "ignored", "accepted", "completed", "expired", "cancelled"])
async def test_scoped_detail_never_refreshes_or_mutates_event(reads, status):
    reads.return_value = [row(status, discovery_metadata={"kind": "event"})]
    result = await activities.read_takeover_activity("case", "admin", "activity")
    assert result.status == status
    reads.assert_awaited_once()
    assert reads.await_args.args[1:] == ("case", "admin", "activity")


@pytest.mark.parametrize("note", [None, "已有旅途故事"])
async def test_review_uses_saved_note_without_model_or_write(reads, note):
    reads.return_value = [row("completed", travel_note=note)]
    result = await activities.read_takeover_activity_review("case", "admin", "activity")
    assert result.story == (note or "这次旅途已收好，还没有生成旅途故事。可以查看已有素材与思绪碎片。")
    assert result.arrival_message_id == "arrival"
    assert not result.can_generate_memory_note
    activity_service._ensure_journey_note.assert_not_awaited()


async def test_review_still_requires_completed_activity(reads):
    with pytest.raises(HTTPException) as error:
        await activities.read_takeover_activity_review("case", "admin", "activity")
    assert error.value.status_code == 409


async def test_user_review_preserves_note_generation(monkeypatch):
    monkeypatch.setattr(activity_service.repo, "get_activity", AsyncMock(return_value=row("completed")))
    monkeypatch.setattr(activity_service.repo, "list_fragments", AsyncMock(return_value=[]))
    monkeypatch.setattr(activity_service.repo, "list_gallery_media", AsyncMock(return_value=[]))
    monkeypatch.setattr(activity_service.repo, "find_arrival_card_message_id", AsyncMock(return_value=None))
    monkeypatch.setattr(activity_service.repo, "journey_evidence", AsyncMock(return_value={"photo_count": 1}))
    note = AsyncMock(return_value="用户原有故事生成")
    monkeypatch.setattr(activity_service, "_ensure_journey_note", note)
    result = await activity_service.get_review("subject", "activity")
    assert result.story == "用户原有故事生成"
    note.assert_awaited_once()


@pytest.mark.parametrize("suffix", ["", "/review"])
async def test_api_is_read_only_scoped_and_uncached(reads, suffix):
    app = FastAPI()
    app.include_router(takeovers.router)
    app.dependency_overrides[takeovers.require_takeover_admin] = lambda: {"sub": "admin"}
    reads.return_value = [row("completed")]
    url = "/admin-api/takeovers/case/activities/activity" + suffix
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(url)
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["id"] == "activity"
        for method in ["POST", "PATCH", "DELETE"]:
            assert (await client.request(method, url)).status_code == 405
        reads.return_value = []
        response = await client.get(url)
        assert response.status_code == 404
        assert response.json()["detail"] == "活动不存在或已无权查看"


@pytest.mark.parametrize("suffix", ["", "/review"])
async def test_api_requires_current_active_admin(monkeypatch, suffix):
    from tests.conftest import make_auth_header
    app = FastAPI()
    app.include_router(takeovers.router)
    query = AsyncMock(return_value=[])
    monkeypatch.setattr(activities.db, "query_raw", query)
    url = "/admin-api/takeovers/case/activities/activity" + suffix
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get(url)).status_code == 401
        assert (await client.get(url, headers=make_auth_header("subject", "user"))).status_code == 403
        assert (await client.get(url, headers=make_auth_header("admin", "admin"))).status_code == 403
    assert query.await_count == 1


async def test_public_user_detail_keeps_token_subject(monkeypatch):
    from app.api.jwt_auth import require_user
    app = FastAPI()
    app.include_router(offline.router)
    app.dependency_overrides[require_user] = lambda: {"sub": "admin", "role": "admin"}
    fetch = AsyncMock(side_effect=HTTPException(404, "Activity not found"))
    monkeypatch.setattr(offline.activity_service, "get_activity", fetch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/offline/activities/activity")).status_code == 404
    fetch.assert_awaited_once_with("admin", "activity")


@pytest.mark.parametrize("variant", ["owned", "other_admin", "inactive", "other_user", "other_conversation", "other_workspace", "null_workspace", "both_null", "missing"])
async def test_scope_join_enforces_resource_boundaries(monkeypatch, variant):
    """Execute the actual bounded SELECT with synthetic rows, including null scopes."""
    import sqlite3
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE human_takeovers (id TEXT, user_id TEXT, owner_admin_id TEXT,
          status TEXT, conversation_id TEXT, workspace_id TEXT);
        CREATE TABLE offline_activity_recommendations (id TEXT, user_id TEXT,
          conversation_id TEXT, workspace_id TEXT, status TEXT, title TEXT,
          summary TEXT, description TEXT, created_at TEXT, updated_at TEXT);
    """)
    owner = "other" if variant == "other_admin" else "admin"
    status = "pending" if variant == "inactive" else "active"
    claim_workspace = None if variant == "both_null" else "workspace"
    connection.execute("INSERT INTO human_takeovers VALUES (?,?,?,?,?,?)",
                       ("case", "subject", owner, status, "conversation", claim_workspace))
    data = row()
    if variant == "other_user": data["user_id"] = "other"
    if variant == "other_conversation": data["conversation_id"] = "other"
    if variant == "other_workspace": data["workspace_id"] = "other"
    if variant in {"null_workspace", "both_null"}: data["workspace_id"] = None
    if variant != "missing":
        columns = ",".join(data)
        connection.execute(f"INSERT INTO offline_activity_recommendations ({columns}) VALUES ({','.join('?' for _ in data)})", tuple(data.values()))
    async def query(sql, *params):
        return [dict(item) for item in connection.execute(sql, {str(index): value for index, value in enumerate(params, 1)})]
    monkeypatch.setattr(activities.db, "query_raw", query)
    try:
        if variant in {"owned", "both_null"}:
            assert (await activities._read_owned_activity("case", "admin", "activity"))["id"] == "activity"
        else:
            with pytest.raises(HTTPException) as error:
                await activities._read_owned_activity("case", "admin", "activity")
            assert error.value.status_code == 404
    finally:
        connection.close()
