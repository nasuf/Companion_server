"""Early manual edit guards and post-commit optional hooks."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.memory import manual_edit
from app.services.memory.storage import persistence


def record():
    return SimpleNamespace(id="m", userId="u", workspaceId="w", source="user", content="旧值",
        updatedAt=datetime(2026, 10, 12, tzinfo=timezone.utc), level=2, importance=.6)


@pytest.mark.parametrize("actor,extra,status,detail", [
    ({}, {}, 403, "memory_edit_actor_invalid"),
    ({"sub": "u", "role": "unknown"}, {}, 403, "memory_edit_actor_invalid"),
    ({"sub": "another", "role": "user"}, {}, 403, "Not your memory"),
    ({"sub": "u", "role": "user"}, {"repair_item": {}}, 403, "Admin access required"),
    ({"sub": "u", "role": "user"}, {"level": 4}, 400, "invalid_memory_level"),
    ({"sub": "u", "role": "user"}, {"importance": float("nan")}, 400, "invalid_memory_importance"),
    ({"sub": "u", "role": "user"}, {"importance": 0}, 400, "invalid_memory_importance"),
])
async def test_invalid_edit_rejects_before_embedding_or_database(monkeypatch, actor, extra, status, detail):
    database = SimpleNamespace(query_raw=AsyncMock())
    embedding = AsyncMock()
    monkeypatch.setattr(manual_edit, "db", database)
    monkeypatch.setattr(manual_edit, "generate_embedding", embedding)
    with pytest.raises(manual_edit.MemoryEditError) as exc:
        await manual_edit.edit_memory_with_evidence(record=record(), content="新值", actor=actor, **extra)
    assert (exc.value.status_code, exc.value.detail) == (status, detail)
    database.query_raw.assert_not_awaited()
    embedding.assert_not_awaited()


@pytest.mark.parametrize("source,workspace", [("invalid", "w"), ("user", None)])
async def test_legacy_missing_scope_is_not_silently_resolved(source, workspace):
    item = record()
    item.source, item.workspaceId = source, workspace
    with pytest.raises(manual_edit.MemoryEditError, match="memory_scope_changed_or_inactive"):
        await manual_edit.edit_memory_with_evidence(record=item, content="新值", actor={"sub": "u", "role": "user"})


async def test_unavailable_scope_fails_before_embedding(monkeypatch):
    database = SimpleNamespace(query_raw=AsyncMock(return_value=[]))
    embedding = AsyncMock()
    monkeypatch.setattr(manual_edit, "db", database)
    monkeypatch.setattr(manual_edit, "generate_embedding", embedding)
    with pytest.raises(manual_edit.MemoryEditError, match="memory_scope_changed_or_inactive"):
        await manual_edit.edit_memory_with_evidence(record=record(), content="新值", actor={"sub": "u", "role": "user"})
    embedding.assert_not_awaited()


async def test_post_commit_derived_hooks_are_independently_best_effort(monkeypatch):
    quality = AsyncMock(side_effect=RuntimeError("synthetic quality failure"))
    achievement = AsyncMock(side_effect=RuntimeError("synthetic achievement failure"))
    monkeypatch.setattr("app.services.memory.lifecycle.quality_state.refresh_quality_state_for_changelog", quality)
    monkeypatch.setattr("app.services.achievements.service.handle_memory_changelog_event", achievement)
    await persistence.after_memory_changelog(user_id="u", memory_id="m", operation="repair_edit", workspace_id="w")
    quality.assert_awaited_once_with("m")
    achievement.assert_awaited_once_with(user_id="u", memory_id="m", operation="repair_edit", workspace_id="w")
    quality.reset_mock(); achievement.reset_mock()
    await persistence.after_memory_changelog(user_id="u", memory_id="m", operation="access", workspace_id="w")
    quality.assert_not_awaited()
    achievement.assert_not_awaited()


async def test_knowledge_cache_invalidation_is_scoped_and_failure_is_bounded_by_ttl(monkeypatch):
    from app.services.memory.retrieval import knowledge_hits
    redis = SimpleNamespace(delete=AsyncMock())
    monkeypatch.setattr(knowledge_hits, "get_redis", AsyncMock(return_value=redis))
    await knowledge_hits.invalidate_knowledge_rows("workspace-1")
    redis.delete.assert_awaited_once_with("knowledge_rows:workspace-1")
    redis.delete.side_effect = RuntimeError("synthetic unavailable Redis")
    await knowledge_hits.invalidate_knowledge_rows("workspace-1")
    assert redis.delete.await_count == 2
