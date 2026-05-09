"""Scheduled memory hygiene tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.lifecycle.hygiene import run_memory_hygiene
from app.services.memory.storage.reconciliation import ReconciliationDecision
from app.services.memory.storage.repo import MemoryRecord


def _record(
    *,
    id: str,
    content: str,
    source: str = "ai",
    main: str = "身份",
    sub: str = "宠物",
    level: int = 1,
    importance: float = 0.85,
) -> MemoryRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return MemoryRecord(
        id=id,
        userId="u1",
        type="identity",
        source=source,  # type: ignore[arg-type]
        level=level,
        content=content,
        summary=content,
        importance=importance,
        mentionCount=0,
        isArchived=False,
        occurTime=None,
        createdAt=now,
        updatedAt=now,
        mainCategory=main,
        subCategory=sub,
        workspaceId="ws1",
    )


@pytest.mark.asyncio
async def test_memory_hygiene_archives_duplicate_record():
    current = _record(id="short", content="我养了一只叫“芝麻”的黑猫")
    existing = _record(
        id="rich",
        content="养了一只叫“芝麻”的黑猫，是从灵隐寺附近的草丛里捡来的流浪猫，当时它只有巴掌大。",
    )
    decision = ReconciliationDecision(
        action="drop_duplicate",
        existing_id="rich",
        existing_record=existing,
    )
    P = "app.services.memory.lifecycle.hygiene"
    with (
        patch(f"{P}._active_scopes", new_callable=AsyncMock, return_value=[("ai", "u1", "ws1")]),
        patch(f"{P}._scope_memories", new_callable=AsyncMock, return_value=[current]),
        patch(f"{P}.generate_embedding", new_callable=AsyncMock, return_value=[0.1]),
        patch(f"{P}.resolve_memory_write", new_callable=AsyncMock, return_value=decision) as mock_resolve,
        patch(f"{P}.memory_repo.update", new_callable=AsyncMock) as mock_update,
        patch(f"{P}.log_memory_changelog", new_callable=AsyncMock) as mock_changelog,
    ):
        stats = await run_memory_hygiene()

    assert stats["checked"] == 1
    assert stats["archived"] == 1
    assert stats["errors"] == 0
    assert stats["changes"][0]["action"] == "archived_duplicate"
    assert stats["changes"][0]["removed"]["id"] == "short"
    assert stats["changes"][0]["kept"]["id"] == "rich"
    mock_resolve.assert_awaited_once()
    assert mock_resolve.await_args.kwargs["exclude_id"] == "short"
    assert mock_resolve.await_args.kwargs["main_category"] == "身份"
    mock_update.assert_awaited_once()
    assert mock_update.await_args.args[0] == "short"
    assert mock_update.await_args.kwargs["isArchived"] is True
    mock_changelog.assert_awaited_once()
    assert mock_changelog.await_args.args[2] == "hygiene_archived_duplicate"


@pytest.mark.asyncio
async def test_memory_hygiene_merges_then_archives_absorbed_record():
    current = _record(
        id="new-rich",
        content="用户喜欢研究咖啡豆，尤其关注浅烘埃塞豆",
        source="user",
        main="偏好",
        sub="饮食喜好",
        level=2,
        importance=0.8,
    )
    existing = _record(
        id="old-generic",
        content="用户喜欢咖啡",
        source="user",
        main="偏好",
        sub="饮食喜好",
        level=2,
        importance=0.7,
    )
    decision = ReconciliationDecision(
        action="merge_existing",
        existing_id="old-generic",
        existing_record=existing,
        merged_summary="用户喜欢咖啡，也喜欢研究浅烘埃塞咖啡豆",
        merged_content="用户喜欢咖啡，也喜欢研究浅烘埃塞咖啡豆",
    )
    P = "app.services.memory.lifecycle.hygiene"
    with (
        patch(f"{P}._active_scopes", new_callable=AsyncMock, return_value=[("user", "u1", "ws1")]),
        patch(f"{P}._scope_memories", new_callable=AsyncMock, return_value=[current]),
        patch(f"{P}.generate_embedding", new_callable=AsyncMock, side_effect=[[0.1], [0.2]]),
        patch(f"{P}.resolve_memory_write", new_callable=AsyncMock, return_value=decision),
        patch(f"{P}.store_embedding", new_callable=AsyncMock) as mock_store_embedding,
        patch(f"{P}.memory_repo.update", new_callable=AsyncMock) as mock_update,
        patch(f"{P}.log_memory_changelog", new_callable=AsyncMock) as mock_changelog,
    ):
        stats = await run_memory_hygiene()

    assert stats["checked"] == 1
    assert stats["archived"] == 1
    assert stats["merged"] == 1
    assert stats["errors"] == 0
    assert stats["changes"][0]["action"] == "merged"
    assert stats["changes"][0]["kept"]["id"] == "old-generic"
    assert stats["changes"][0]["removed"]["id"] == "new-rich"
    assert stats["changes"][0]["after"] == "用户喜欢咖啡，也喜欢研究浅烘埃塞咖啡豆"
    mock_store_embedding.assert_awaited_once_with("old-generic", [0.2])
    assert mock_update.await_count == 2
    first_update = mock_update.await_args_list[0]
    second_update = mock_update.await_args_list[1]
    assert first_update.args[0] == "old-generic"
    assert first_update.kwargs["content"] == "用户喜欢咖啡，也喜欢研究浅烘埃塞咖啡豆"
    assert second_update.args[0] == "new-rich"
    assert second_update.kwargs["isArchived"] is True
    assert [call.args[2] for call in mock_changelog.await_args_list] == [
        "hygiene_merge",
        "hygiene_absorbed",
    ]


def test_scheduler_registers_memory_hygiene_job():
    import inspect

    from jobs import scheduler as scheduler_mod

    source = inspect.getsource(scheduler_mod.setup_scheduler)
    assert 'id="memory_hygiene"' in source
    assert "_run_memory_hygiene" in source


@pytest.mark.asyncio
async def test_memory_hygiene_endpoint_scopes_to_user_workspace():
    from app.api.public.memories import run_memory_hygiene_now
    from app.models.memory import MemoryHygieneRequest

    P = "app.api.public.memories"
    report = {
        "scopes": 1,
        "checked": 2,
        "archived": 1,
        "merged": 0,
        "updated": 0,
        "errors": 0,
        "changes": [],
    }
    with (
        patch(f"{P}.resolve_workspace_id", new_callable=AsyncMock, return_value="ws1"),
        patch(f"{P}.run_memory_hygiene", new_callable=AsyncMock, return_value=report) as mock_run,
    ):
        result = await run_memory_hygiene_now(
            MemoryHygieneRequest(workspace_id=None, allow_llm=False, max_memories_per_scope=123),
            user_id="u1",
            _user={"sub": "u1"},
        )

    assert result == report
    mock_run.assert_awaited_once_with(
        user_id="u1",
        workspace_id="ws1",
        allow_llm=False,
        max_scopes=2,
        max_memories_per_scope=123,
    )
