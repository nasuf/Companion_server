from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.admin.proactive import router as admin_proactive_router
from app.api.jwt_auth import require_admin_jwt

app = FastAPI()
app.include_router(admin_proactive_router)


@pytest.fixture
def admin_client():
    app.dependency_overrides[require_admin_jwt] = lambda: {"sub": "admin-user-1", "role": "admin"}
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_admin_trigger_proactive_success(admin_client):
    with (
        patch(
            "app.api.admin.proactive._resolve_workspace_and_agent",
            new_callable=AsyncMock,
            return_value=("ws-1", "agent-1"),
        ),
        patch(
            "app.api.admin.proactive.send_manual_or_triggered_proactive",
            new_callable=AsyncMock,
            return_value={
                "ok": True,
                "reason": None,
                "message": "嗨，在吗？",
                "web_search_used": True,
                "link_card_used": False,
            },
        ) as mock_send,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/admin-api/proactive/trigger",
                json={
                    "workspace_id": "ws-1",
                    "trigger_type": "silence_wakeup",
                    "skip_limits": True,
                    "use_web_search": True,
                    "use_link_card": False,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["trigger_type"] == "silence_wakeup"
    assert body["message"] == "嗨，在吗？"
    assert body["web_search_used"] is True
    assert body["link_card_used"] is False
    admin_opts = mock_send.await_args.kwargs["admin_test_options"]
    assert admin_opts.use_web_search is True
    assert admin_opts.use_link_card is False


@pytest.mark.asyncio
async def test_admin_trigger_proactive_rejects_unknown_type(admin_client):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/admin-api/proactive/trigger",
            json={"workspace_id": "ws-1", "trigger_type": "hot_news"},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_admin_trigger_proactive_resolves_workspace_from_agent(admin_client):
    with (
        patch(
            "app.api.admin.proactive._resolve_workspace_and_agent",
            new_callable=AsyncMock,
            return_value=("ws-resolved", "agent-1"),
        ) as mock_resolve,
        patch(
            "app.api.admin.proactive.send_manual_or_triggered_proactive",
            new_callable=AsyncMock,
            return_value={"ok": False, "reason": "generation_or_limit_blocked", "message": None},
        ) as mock_send,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/admin-api/proactive/trigger",
                json={"agent_id": "agent-1", "trigger_type": "scheduled_scene"},
            )

    assert response.status_code == 200
    mock_resolve.assert_awaited_once_with(
        user_id="admin-user-1",
        workspace_id=None,
        agent_id="agent-1",
    )
    mock_send.assert_awaited_once()
    assert mock_send.await_args.kwargs["workspace_id"] == "ws-resolved"
    assert mock_send.await_args.kwargs["trigger_type"] == "scheduled_scene"
    assert mock_send.await_args.kwargs["skip_limits"] is True


@pytest.mark.asyncio
async def test_admin_trigger_special_date(admin_client):
    with (
        patch(
            "app.api.admin.proactive._resolve_workspace_and_agent",
            new_callable=AsyncMock,
            return_value=("ws-1", "agent-1"),
        ),
        patch(
            "app.api.admin.proactive.send_special_date_proactive",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_send,
        patch(
            "app.api.admin.proactive.db.query_raw",
            new_callable=AsyncMock,
            return_value=[{"message": "节日快乐"}],
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/admin-api/proactive/trigger",
                json={"workspace_id": "ws-1", "trigger_type": "special_date"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["message"] == "节日快乐"
    mock_send.assert_awaited_once()
    assert mock_send.await_args.kwargs["workspace_id"] == "ws-1"
    assert mock_send.await_args.kwargs["skip_limits"] is True
