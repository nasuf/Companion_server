"""P4 24h 自动归档扫描（spec §3.6/§6）。"""

from unittest.mock import AsyncMock

from app.services.offline import activity_service


async def test_auto_archive_archives_all_due(monkeypatch):
    due = [
        {"id": "a1", "user_id": "u1", "workspace_id": "w1", "title": "X"},
        {"id": "a2", "user_id": "u1", "workspace_id": "w1", "title": "Y"},
    ]
    monkeypatch.setattr(
        activity_service.repo, "list_due_for_auto_archive",
        AsyncMock(return_value=due),
    )
    monkeypatch.setattr(
        activity_service.repo, "mark_archived",
        AsyncMock(side_effect=lambda aid, uid, *, auto: {"id": aid, "status": "completed"}),
    )
    monkeypatch.setattr(
        activity_service.repo, "resolve_user_context",
        AsyncMock(return_value={"conversation_id": "c1", "agent_id": "ag1", "workspace_id": "w1"}),
    )
    emit_card = AsyncMock()
    emit_assist = AsyncMock()
    monkeypatch.setattr(activity_service, "emit_activity_card", emit_card)
    monkeypatch.setattr(activity_service, "emit_assistant", emit_assist)

    result = await activity_service.auto_archive_due_activities()

    assert result == {"scanned": 2, "archived": 2}
    assert emit_card.await_count == 2
    assert emit_assist.await_count == 2
    # 自动归档必须带 auto=True
    for call in activity_service.repo.mark_archived.await_args_list:
        assert call.kwargs["auto"] is True


async def test_auto_archive_skips_failed_and_continues(monkeypatch):
    due = [
        {"id": "a1", "user_id": "u1", "title": "X"},
        {"id": "a2", "user_id": "u1", "title": "Y"},
    ]
    monkeypatch.setattr(
        activity_service.repo, "list_due_for_auto_archive",
        AsyncMock(return_value=due),
    )

    async def mark(aid, uid, *, auto):
        if aid == "a1":
            raise RuntimeError("db blip")  # spec §6: 单条失败跳过
        return {"id": aid, "status": "completed"}

    monkeypatch.setattr(activity_service.repo, "mark_archived", mark)
    monkeypatch.setattr(
        activity_service.repo, "resolve_user_context",
        AsyncMock(return_value=None),  # 无 ctx → 不推卡，仍算归档
    )

    result = await activity_service.auto_archive_due_activities()

    assert result == {"scanned": 2, "archived": 1}


async def test_auto_archive_empty(monkeypatch):
    monkeypatch.setattr(
        activity_service.repo, "list_due_for_auto_archive",
        AsyncMock(return_value=[]),
    )
    result = await activity_service.auto_archive_due_activities()
    assert result == {"scanned": 0, "archived": 0}
