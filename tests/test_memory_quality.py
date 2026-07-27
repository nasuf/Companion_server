from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.lifecycle.quality import derive_memory_quality
from app.services.memory.storage.repo import MemoryRecord


def _record(memory_id="m1") -> MemoryRecord:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    return MemoryRecord(
        id=memory_id,
        userId="u1",
        type="life",
        source="user",
        level=2,
        content="用户喜欢手冲咖啡",
        importance=0.7,
        mentionCount=0,
        isArchived=False,
        occurTime=None,
        createdAt=now,
        updatedAt=now,
        mainCategory="生活",
        subCategory="兴趣",
        workspaceId="ws1",
    )


@pytest.mark.asyncio
async def test_derive_memory_quality_uses_evidence_access_and_corrections():
    rows = [
        {
            "memory_id": "m1",
            "operation": "evidence_linked",
            "new_value": '{"message_ids":["msg-1","msg-2"]}',
            "created_at": datetime(2026, 5, 2, tzinfo=timezone.utc),
        },
        {
            "memory_id": "m1",
            "operation": "access",
            "new_value": None,
            "created_at": datetime(2026, 5, 3, tzinfo=timezone.utc),
        },
        {
            "memory_id": "m1",
            "operation": "user_edit",
            "new_value": "用户纠正了内容",
            "created_at": datetime(2026, 5, 4, tzinfo=timezone.utc),
        },
    ]
    with patch("app.services.memory.lifecycle.quality.db") as fake_db:
        fake_db.query_raw = AsyncMock(return_value=rows)
        result = await derive_memory_quality([_record("m1")])

    q = result["m1"]
    assert q.evidence_message_ids == ["msg-1", "msg-2"]
    assert q.access_count == 1
    assert q.user_corrected_count == 1
    assert q.contradiction_state == "corrected"
    assert q.last_verified_at == datetime(2026, 5, 4, tzinfo=timezone.utc)
    assert "has_evidence_messages" in q.signals


@pytest.mark.asyncio
async def test_derive_memory_quality_caps_archived_confidence():
    record = _record("m1")
    record.isArchived = True
    record.importance = 0.95

    with patch("app.services.memory.lifecycle.quality.db") as fake_db:
        fake_db.query_raw = AsyncMock(return_value=[])
        result = await derive_memory_quality([record])

    assert result["m1"].confidence <= 0.2
    assert "archived" in result["m1"].signals


@pytest.mark.asyncio
async def test_refresh_memory_quality_state_materializes_row(monkeypatch):
    from app.services.memory.lifecycle import quality_state

    record = _record("m1")
    monkeypatch.setattr(quality_state.memory_repo, "find_unique", AsyncMock(return_value=record))
    monkeypatch.setattr(
        quality_state,
        "derive_memory_quality",
        AsyncMock(return_value={
            "m1": type("Q", (), {
                "confidence": 0.82,
                "evidence_message_ids": ["msg-1"],
                "last_verified_at": datetime(2026, 5, 4, tzinfo=timezone.utc),
                "contradiction_state": "none",
                "user_corrected_count": 0,
                "access_count": 3,
                "signals": ["has_evidence_messages"],
            })(),
        }),
    )
    fake_db = type("DB", (), {})()
    fake_db.query_raw = AsyncMock(side_effect=[
        [],
        [{
            "memory_id": "m1",
            "memory_source": "user",
            "user_id": "u1",
            "workspace_id": "ws1",
            "confidence": 0.82,
            "evidence_message_ids": ["msg-1"],
            "last_verified_at": datetime(2026, 5, 4, tzinfo=timezone.utc),
            "verified_by": None,
            "contradiction_state": "none",
            "user_corrected_count": 0,
            "admin_repaired_count": 0,
            "access_count": 3,
            "last_repair_item_id": None,
            "superseded_by_memory_id": None,
            "signals": {"signals": ["has_evidence_messages"]},
            "source_updated_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2026, 5, 5, tzinfo=timezone.utc),
        }],
    ])
    monkeypatch.setattr(quality_state, "db", fake_db)

    state = await quality_state.refresh_memory_quality_state("m1")

    assert state is not None
    assert state["confidence"] == 0.82
    assert state["evidence_message_ids"] == ["msg-1"]
    assert fake_db.query_raw.await_count == 2
