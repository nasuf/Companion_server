"""memories API must reject an explicitly-supplied workspace the caller does
not own (defense-in-depth on top of the existing user_id scoping)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_auth_header as _hdr


@pytest.fixture
def client(api_client):
    return api_client


def test_list_foreign_workspace_403(client):
    foreign_ws = SimpleNamespace(id="w-other", userId="other-user")
    with patch("app.api.public.memories.db") as db_mock:
        db_mock.chatworkspace.find_unique = AsyncMock(return_value=foreign_ws)
        r = client.get("/memories?user_id=u1&workspace_id=w-other", headers=_hdr("u1"))
    assert r.status_code == 403


def test_list_own_workspace_ok(client):
    own_ws = SimpleNamespace(id="w1", userId="u1")
    with (
        patch("app.api.public.memories.db") as db_mock,
        patch("app.api.public.memories.memory_repo.find_many", new_callable=AsyncMock, return_value=[]),
    ):
        db_mock.chatworkspace.find_unique = AsyncMock(return_value=own_ws)
        r = client.get("/memories?user_id=u1&workspace_id=w1", headers=_hdr("u1"))
    assert r.status_code == 200


def test_list_no_workspace_still_ok(client):
    """未传 workspace_id → 保持既有聚合行为 (同用户数据), 不回归."""
    with patch("app.api.public.memories.memory_repo.find_many", new_callable=AsyncMock, return_value=[]):
        r = client.get("/memories?user_id=u1", headers=_hdr("u1"))
    assert r.status_code == 200


def test_search_foreign_workspace_403(client):
    foreign_ws = SimpleNamespace(id="w-other", userId="other-user")
    with patch("app.api.public.memories.db") as db_mock:
        db_mock.chatworkspace.find_unique = AsyncMock(return_value=foreign_ws)
        r = client.post(
            "/memories/search?user_id=u1",
            headers=_hdr("u1"),
            json={"query": "x", "top_k": 5, "workspace_id": "w-other"},
        )
    assert r.status_code == 403
