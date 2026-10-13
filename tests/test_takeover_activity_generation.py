"""Takeover generation must use the complete user activity pipeline and recipient."""
from copy import deepcopy
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from app.api.admin import takeovers
from app.api.public import offline
from app.api.jwt_auth import require_user
from app.services.admin_takeover import activities
from app.services.offline import activity_service, module_settings

SUBJECT = {"userId": "subject", "workspaceId": "workspace", "conversationId": "conversation", "agentId": "agent"}
CTX = {
    "user_id": "subject", "agent_id": "agent", "workspace_id": "workspace",
    "conversation_id": "conversation", "user_location_city": "镇江市",
    "user_location_latitude": 32.2, "user_location_longitude": 119.4,
}
CARD = {
    "title": "小岛咖啡", "summary": "散步时可以停留的一站",
    "description": "完整地点介绍、开放时间参考与出行提示。" * 12,
    "location_name": "小岛咖啡", "address": "伯先路12号", "city": "镇江市",
    "category": "咖啡", "image_urls": ["/offline/media/one.jpg", "/offline/media/two.jpg", "/offline/media/three.jpg"],
    "place_lat": 32.2, "place_lng": 119.4, "official_url": "https://example.test/place",
    "search_sources": [{"url": "https://example.test/source"}],
    "discovery_metadata": {
        "coordinate_source": "native_poi", "poi_id": "poi", "session_key": "poi:poi",
        "recommendation_message": "你说过喜欢在咖啡馆看书，这家店可以作为下次散步的一站。",
        "recommendation_message_status": "verified", "user_relevance_count": 2,
        "recommendation_evidence_status": {"memory": "collected", "dialogue": "collected", "preference": "collected"},
    },
}


@pytest.fixture
def pipeline(monkeypatch):
    events = []
    owned = AsyncMock(return_value=SUBJECT)
    monkeypatch.setattr(activities, "_owned_subject", owned)
    enabled = AsyncMock(return_value=True)
    monkeypatch.setattr(module_settings, "is_activity_enabled", enabled)
    monkeypatch.setattr(offline, "is_activity_enabled", enabled)
    context = AsyncMock(return_value=CTX)
    monkeypatch.setattr(activity_service.repo, "resolve_user_context", context)
    async def generate(**kwargs):
        events.append("generate")
        return deepcopy(CARD)
    generated = AsyncMock(side_effect=generate)
    monkeypatch.setattr(activity_service, "generate_activity_card", generated)
    stored = {}
    async def persist(data):
        events.append("persist")
        stored.update(deepcopy(data))
        return {**data, "id": "activity", "status": "pending",
                "recommendation_message": data["discovery_metadata"]["recommendation_message"],
                "created_at": "2026-10-13T00:00:00Z", "updated_at": "2026-10-13T00:00:00Z"}
    create = AsyncMock(side_effect=persist)
    monkeypatch.setattr(activity_service.repo, "create_activity", create)
    async def invite(**kwargs):
        events.append("invite")
        return "要不要了解一下这家咖啡馆？"
    monkeypatch.setattr(activity_service, "generate_activity_invite_message", AsyncMock(side_effect=invite))
    async def text(**kwargs): events.append("text")
    async def card(**kwargs): events.append("card")
    async def schedule(*args): events.append("schedule")
    emit = AsyncMock(side_effect=text)
    emit_card = AsyncMock(side_effect=card)
    monkeypatch.setattr(activity_service, "emit_assistant", emit)
    monkeypatch.setattr(activity_service, "emit_activity_card", emit_card)
    monkeypatch.setattr(activity_service.repo, "update_next_activity_due", AsyncMock(side_effect=schedule))
    return {"owned": owned, "enabled": enabled, "context": context, "generated": generated,
            "create": create, "stored": stored, "emit": emit, "emit_card": emit_card, "events": events}


def app():
    value = FastAPI()
    value.include_router(takeovers.router)
    value.include_router(offline.router)
    value.dependency_overrides[takeovers.require_takeover_admin] = lambda: {"sub": "admin", "role": "admin"}
    value.dependency_overrides[require_user] = lambda: {"sub": "subject", "role": "user"}
    return value


async def test_admin_and_user_generation_have_identical_content_and_publication(pipeline):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url="http://test") as client:
        admin = await client.post("/admin-api/takeovers/case/activities/recommend")
        assert admin.status_code == 200, admin.text
        assert admin.headers["cache-control"] == "no-store"
        # The API retains every public detail, while hiding raw personalization evidence.
        public = admin.json()
        assert public["recommendation_message"] == CARD["discovery_metadata"]["recommendation_message"]
        assert public["description"] == CARD["description"]
        assert public["image_urls"] == CARD["image_urls"]
        assert "discovery_metadata" not in public and "user_relevance" not in public
        assert pipeline["events"] == ["generate", "invite", "persist", "text", "card", "schedule"]
        assert pipeline["stored"]["search_sources"] == CARD["search_sources"]
        assert pipeline["stored"]["discovery_metadata"] == CARD["discovery_metadata"]
        assert pipeline["stored"]["place_key"] == "poi:poi"
        pipeline["context"].assert_awaited_with("subject", "workspace")
        assert pipeline["generated"].await_args.kwargs["conversation_id"] == "conversation"
        for send in (pipeline["emit"], pipeline["emit_card"]):
            args = send.await_args.kwargs
            assert (args["user_id"], args["agent_id"], args["workspace_id"], args["conversation_id"]) == ("subject", "agent", "workspace", "conversation")
        assert pipeline["emit"].await_args.kwargs["extra_metadata"]["recommendation_message_status"] == "verified"
        user = await client.post("/offline/activities/recommend?workspace_id=workspace")
        assert user.status_code == 200, user.text
        assert user.json() == public
        assert pipeline["events"][6:] == ["generate", "persist", "invite", "text", "card", "schedule"]


@pytest.mark.parametrize("failure,status", [("not_owner", 403), ("disabled", 403), ("conversation", 409), ("agent", 409), ("workspace", 409), ("location", 422), ("empty", 503), ("revoked_during_generation", 403), ("revoked_during_invite", 403)])
async def test_failure_never_sends_to_the_wrong_chat(pipeline, failure, status):
    if failure == "not_owner": pipeline["owned"].return_value = None
    if failure == "disabled": pipeline["enabled"].return_value = False
    if failure == "conversation": pipeline["context"].return_value = {**CTX, "conversation_id": "other"}
    if failure == "agent": pipeline["context"].return_value = {**CTX, "agent_id": "other"}
    if failure == "workspace": pipeline["context"].return_value = {**CTX, "workspace_id": "other"}
    if failure == "location": pipeline["context"].return_value = {**CTX, "user_location_city": ""}
    if failure == "empty": pipeline["generated"].side_effect = None; pipeline["generated"].return_value = None
    if failure == "revoked_during_generation": pipeline["owned"].side_effect = [SUBJECT, SUBJECT, None]
    if failure == "revoked_during_invite": pipeline["owned"].side_effect = [SUBJECT, SUBJECT, SUBJECT, None]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url="http://test") as client:
        response = await client.post("/admin-api/takeovers/case/activities/recommend")
        assert response.status_code == status, response.text
    pipeline["emit"].assert_not_awaited()
    pipeline["emit_card"].assert_not_awaited()
    pipeline["create"].assert_not_awaited()


async def test_recommendation_requires_current_active_admin(pipeline, monkeypatch):
    from tests.conftest import make_auth_header
    value = FastAPI()
    value.include_router(takeovers.router)
    monkeypatch.setattr(activities.db, "query_raw", AsyncMock(return_value=[]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=value), base_url="http://test") as client:
        url = "/admin-api/takeovers/case/activities/recommend"
        assert (await client.post(url)).status_code == 401
        assert (await client.post(url, headers=make_auth_header("subject", "user"))).status_code == 403
        assert (await client.post(url, headers=make_auth_header("admin", "admin"))).status_code == 403
    pipeline["generated"].assert_not_awaited()
    pipeline["create"].assert_not_awaited()
