from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_auth_header as _hdr


@pytest.fixture
def client(api_client):
    return api_client


def _memory(memory_id="m1", *, user_id="u1", source="user", workspace_id="ws1", archived=False):
    return SimpleNamespace(
        id=memory_id,
        userId=user_id,
        type="life",
        mainCategory="生活",
        subCategory="工作",
        source=source,
        level=2,
        content="旧内容",
        summary="旧摘要",
        importance=0.7,
        mentionCount=0,
        isArchived=archived,
        occurTime=None,
        createdAt=datetime(2026, 5, 1, tzinfo=timezone.utc),
        updatedAt=datetime(2026, 5, 1, tzinfo=timezone.utc),
        workspaceId=workspace_id,
    )


def test_export_memories_returns_owned_workspace_memories(client):
    memories = [_memory("m1"), _memory("m2", source="ai")]
    with (
        patch("app.api.public.memories.resolve_workspace_id", AsyncMock(return_value="ws1")),
        patch("app.api.public.memories.memory_repo.find_many", AsyncMock(return_value=memories)) as find_many,
    ):
        response = client.get("/memories/export?user_id=u1", headers=_hdr("u1"))

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["workspace_id"] == "ws1"
    find_many.assert_awaited_once()


def test_list_memories_can_include_quality_signals(client):
    memories = [_memory("m1")]
    quality = SimpleNamespace(
        confidence=0.82,
        evidence_message_ids=["msg-1"],
        last_verified_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
        contradiction_state="none",
        user_corrected_count=0,
        access_count=3,
        signals=["has_evidence_messages"],
    )
    with (
        patch("app.api.public.memories.memory_repo.find_many", AsyncMock(return_value=memories)),
        patch("app.api.public.memories.derive_memory_quality", AsyncMock(return_value={"m1": quality})),
    ):
        response = client.get(
            "/memories?user_id=u1&include_quality=true",
            headers=_hdr("u1"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body[0]["quality"]["confidence"] == 0.82
    assert body[0]["quality"]["evidence_message_ids"] == ["msg-1"]


def test_update_memory_edits_row_embedding_and_changelog(client):
    old = _memory("m1")
    updated = _memory("m1")
    updated.content = "新内容"
    updated.summary = "新摘要"

    with (
        patch("app.services.memory.storage.repo.find_unique", AsyncMock(return_value=old)),
        patch("app.api.public.memories.generate_embedding", AsyncMock(return_value=[0.1, 0.2])),
        patch("app.api.public.memories.store_embedding", AsyncMock()) as store_embedding,
        patch("app.api.public.memories.memory_repo.update", AsyncMock()) as update,
        patch("app.api.public.memories.memory_repo.find_unique", AsyncMock(return_value=updated)),
        patch("app.api.public.memories.log_memory_changelog", AsyncMock()) as changelog,
    ):
        response = client.patch(
            "/memories/m1",
            headers=_hdr("u1"),
            json={"content": "新内容", "summary": "新摘要"},
        )

    assert response.status_code == 200
    assert response.json()["content"] == "新内容"
    store_embedding.assert_awaited_once()
    update.assert_awaited_once()
    assert update.await_args.kwargs["content"] == "新内容"
    changelog.assert_awaited_once()


def test_update_memory_rejects_taxonomy_and_level_edits(client):
    with patch("app.services.memory.storage.repo.find_unique", AsyncMock(return_value=_memory("m1"))):
        response = client.patch(
            "/memories/m1",
            headers=_hdr("u1"),
            json={"importance": 0.9},
        )

    assert response.status_code == 422


def test_bulk_delete_archives_only_owned_memories(client):
    owned = _memory("m1", user_id="u1")
    other = _memory("m2", user_id="other")

    async def _find(memory_id):
        return {"m1": owned, "m2": other}.get(memory_id)

    with (
        patch("app.api.public.memories.memory_repo.find_unique", AsyncMock(side_effect=_find)),
        patch("app.api.public.memories.memory_repo.update", AsyncMock()) as update,
        patch("app.api.public.memories.log_memory_changelog", AsyncMock()) as changelog,
    ):
        response = client.post(
            "/memories/bulk-delete?user_id=u1",
            headers=_hdr("u1"),
            json={"memory_ids": ["m1", "m2", "ghost"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["archived"] == 1
    assert body["missing_or_forbidden"] == ["m2", "ghost"]
    update.assert_awaited_once()
    assert update.await_args.kwargs["isArchived"] is True
    changelog.assert_awaited_once()


def test_workspace_wipe_archives_user_and_ai_memories(client):
    workspace = SimpleNamespace(id="ws1", userId="u1")
    user_records = [_memory("u-m1", source="user")]
    ai_records = [_memory("a-m1", source="ai")]
    fake_db = MagicMock()
    fake_db.chatworkspace.find_unique = AsyncMock(return_value=workspace)

    with (
        patch("app.api.public.memories.db", fake_db),
        patch(
            "app.api.public.memories.memory_repo.find_many",
            AsyncMock(side_effect=[user_records, ai_records]),
        ),
        patch(
            "app.api.public.memories.memory_repo.update_many",
            AsyncMock(side_effect=[1, 1]),
        ) as update_many,
        patch("app.api.public.memories.log_memory_changelog", AsyncMock()) as changelog,
    ):
        response = client.post(
            "/memories/workspace-wipe?user_id=u1",
            headers=_hdr("u1"),
            json={"workspace_id": "ws1", "include_ai": True, "include_user": True},
        )

    assert response.status_code == 200
    assert response.json() == {
        "workspace_id": "ws1",
        "archived_user": 1,
        "archived_ai": 1,
    }
    assert update_many.await_count == 2
    assert changelog.await_count == 2
