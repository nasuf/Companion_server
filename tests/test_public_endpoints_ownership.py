"""End-to-end ownership wiring tests for public endpoints.

For each endpoint family verify:
- 401 when no Bearer token
- 403 when token belongs to a different user
- 200 when token belongs to the correct owner

Wiring tests (mocked DB layer) — does not duplicate dependency logic
already covered by test_ownership_dependencies.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with (
        patch("app.db.connect_db", new_callable=AsyncMock),
        patch("app.db.disconnect_db", new_callable=AsyncMock),
        patch("app.redis_client.get_redis", new_callable=AsyncMock),
        patch("app.redis_client.close_redis", new_callable=AsyncMock),
        patch("jobs.scheduler.setup_scheduler"),
        patch("jobs.scheduler.shutdown_scheduler"),
        patch("app.middleware.configure_logging"),
        patch("app.middleware.configure_langsmith"),
    ):
        from app.main import app
        yield TestClient(app)


def _token(user_id: str, role: str = "user") -> str:
    from app.services.auth import create_jwt
    return create_jwt(user_id, role=role)


def _hdr(user_id: str, role: str = "user") -> dict:
    return {"Authorization": f"Bearer {_token(user_id, role)}"}


# ───────────────────── /intimacy/{agent_id}/{user_id} ─────────────────────

class TestIntimacyOwnership:
    def test_no_token_401(self, client):
        r = client.get("/intimacy/a1/u1")
        assert r.status_code == 401

    def test_wrong_user_path_403(self, client):
        # JWT u1, path user_id u2 → require_user_self 403
        agent = SimpleNamespace(id="a1", userId="u1", status="active")
        with patch("app.api.ownership.db") as db_mock:
            db_mock.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/intimacy/a1/u2", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_wrong_agent_owner_403(self, client):
        agent = SimpleNamespace(id="a1", userId="other-user", status="active")
        with patch("app.api.ownership.db") as db_mock:
            db_mock.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/intimacy/a1/u1", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_owner_200(self, client):
        agent = SimpleNamespace(id="a1", userId="u1", status="active")
        with (
            patch("app.api.ownership.db") as db_mock,
            patch(
                "app.api.public.intimacy.get_intimacy_data",
                new_callable=AsyncMock,
                return_value={"score": 50, "stage": "L1"},
            ),
        ):
            db_mock.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/intimacy/a1/u1", headers=_hdr("u1"))
        assert r.status_code == 200
        assert r.json()["score"] == 50


# ───────────────────── /boundary/{agent_id}/{user_id} ─────────────────────

class TestBoundaryOwnership:
    def test_no_token_401(self, client):
        r = client.get("/boundary/a1/u1")
        assert r.status_code == 401

    def test_wrong_user_path_403(self, client):
        agent = SimpleNamespace(id="a1", userId="u1", status="active")
        with patch("app.api.ownership.db") as db_mock:
            db_mock.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/boundary/a1/u2", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_owner_200(self, client):
        agent = SimpleNamespace(id="a1", userId="u1", status="active")
        with (
            patch("app.api.ownership.db") as db_mock,
            patch("app.api.public.boundary.get_patience", new_callable=AsyncMock, return_value=85),
            patch("app.api.public.boundary.get_patience_zone", return_value="normal"),
        ):
            db_mock.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/boundary/a1/u1", headers=_hdr("u1"))
        assert r.status_code == 200
        assert r.json() == {"patience": 85, "zone": "normal"}


# ───────────────────── /conversations/{conversation_id} ─────────────────────

class TestConversationsOwnership:
    def test_get_no_token_401(self, client):
        r = client.get("/conversations/c1")
        assert r.status_code == 401

    def test_get_wrong_owner_403(self, client):
        conv = SimpleNamespace(id="c1", userId="other-user", isDeleted=False)
        with patch("app.api.ownership.db") as db_mock:
            db_mock.conversation.find_unique = AsyncMock(return_value=conv)
            r = client.get("/conversations/c1", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_get_owner_200(self, client):
        conv = SimpleNamespace(
            id="c1", userId="u1", agentId="a1", workspaceId="w1",
            title="t", isDeleted=False,
            createdAt="2026-04-25T00:00:00", updatedAt="2026-04-25T00:00:00",
        )
        with patch("app.api.ownership.db") as db_mock:
            db_mock.conversation.find_unique = AsyncMock(return_value=conv)
            r = client.get("/conversations/c1", headers=_hdr("u1"))
        assert r.status_code == 200
        assert r.json()["id"] == "c1"

    def test_list_wrong_user_403(self, client):
        r = client.get("/conversations?user_id=u2", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_list_owner_200(self, client):
        with patch("app.api.public.conversations.db") as db_mock:
            db_mock.conversation.find_many = AsyncMock(return_value=[])
            r = client.get("/conversations?user_id=u1", headers=_hdr("u1"))
        assert r.status_code == 200

    def test_messages_wrong_owner_403(self, client):
        conv = SimpleNamespace(id="c1", userId="other-user", isDeleted=False)
        with patch("app.api.ownership.db") as db_mock:
            db_mock.conversation.find_unique = AsyncMock(return_value=conv)
            r = client.get("/conversations/c1/messages", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_create_wrong_user_id_403(self, client):
        """POST body 携带的 user_id 不是自己的, 拒绝."""
        r = client.post(
            "/conversations",
            headers=_hdr("u1"),
            json={"user_id": "other-user", "agent_id": "a1"},
        )
        assert r.status_code == 403


# ───────────────────── /agents/{agent_id} ─────────────────────

class TestAgentsOwnership:
    def test_get_no_token_401(self, client):
        r = client.get("/agents/a1")
        assert r.status_code == 401

    def test_get_wrong_owner_403(self, client):
        agent = SimpleNamespace(id="a1", userId="other-user", status="active")
        with patch("app.api.ownership.db") as db_mock:
            db_mock.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/agents/a1", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_list_wrong_user_403(self, client):
        r = client.get("/agents?user_id=u2", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_list_owner_200(self, client):
        with patch("app.api.public.agents.db") as db_mock:
            db_mock.aiagent.find_many = AsyncMock(return_value=[])
            r = client.get("/agents?user_id=u1", headers=_hdr("u1"))
        assert r.status_code == 200

    def test_provision_status_wrong_owner_403(self, client):
        """Provision status 用手工 ownership 校验 (status=provisioning, 走不了 require_agent_owner)."""
        agent = SimpleNamespace(id="a1", userId="other-user", status="provisioning")
        with patch("app.api.public.agents.db") as db_mock:
            db_mock.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/agents/a1/provision-status", headers=_hdr("u1"))
        assert r.status_code == 403


# ───────────────────── /memories/* ─────────────────────

class TestMemoriesOwnership:
    def test_list_no_token_401(self, client):
        r = client.get("/memories?user_id=u1")
        assert r.status_code == 401

    def test_list_wrong_user_403(self, client):
        r = client.get("/memories?user_id=u2", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_list_owner_200(self, client):
        with patch(
            "app.api.public.memories.memory_repo.find_many",
            new_callable=AsyncMock, return_value=[],
        ):
            r = client.get("/memories?user_id=u1", headers=_hdr("u1"))
        assert r.status_code == 200

    def test_get_memory_wrong_owner_403(self, client):
        m = SimpleNamespace(id="m1", userId="other-user")
        with patch(
            "app.services.memory.storage.repo.find_unique",
            new_callable=AsyncMock, return_value=m,
        ):
            r = client.get("/memories/m1", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_search_wrong_user_403(self, client):
        r = client.post(
            "/memories/search?user_id=u2",
            headers=_hdr("u1"),
            json={"query": "x", "top_k": 5},
        )
        assert r.status_code == 403
