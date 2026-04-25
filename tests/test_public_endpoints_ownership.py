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
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_auth_header as _hdr  # noqa: F401 — shared helper


@pytest.fixture
def client(api_client):
    """Alias to shared api_client fixture for existing test code."""
    return api_client


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
        """provision-status 用 require_agent_owner_any_status (能在 status=provisioning 阶段访问)."""
        agent = SimpleNamespace(id="a1", userId="other-user", status="provisioning")
        with patch("app.api.ownership.db") as db_mock:
            db_mock.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/agents/a1/provision-status", headers=_hdr("u1"))
        assert r.status_code == 403

    def test_get_agent_returns_200_during_provisioning(self, client):
        """Regression for "agent 创建完成后跳回创建页" bug:
        get_agent 老用 require_agent_owner (active-only), 期间 GET 返 404 → 前端
        视为 agent 不存在 → 跳创建页. 改用 _any_status 后 provisioning 阶段也能查."""
        agent = SimpleNamespace(
            id="a-prov", userId="u1", status="provisioning",
            name="x", mbti=None, currentMbti=None, background=None,
            values=None, gender="male", lifeOverview=None,
            createdAt=__import__("datetime").datetime.now(),
        )
        workspace = SimpleNamespace(id="ws-1", status="active")
        with (
            patch("app.api.ownership.db") as db_owner,
            patch("app.api.public.agents.get_active_workspace",
                  new_callable=AsyncMock, return_value=workspace),
        ):
            db_owner.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/agents/a-prov", headers=_hdr("u1"))
        assert r.status_code == 200, r.text

    def test_get_agent_archived_returns_404(self, client):
        """archived agent 即使 owner 也应 404 (软删除语义)."""
        agent = SimpleNamespace(id="a-arch", userId="u1", status="archived")
        with patch("app.api.ownership.db") as db_owner:
            db_owner.aiagent.find_unique = AsyncMock(return_value=agent)
            r = client.get("/agents/a-arch", headers=_hdr("u1"))
        assert r.status_code == 404


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
