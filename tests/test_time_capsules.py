from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.public import time_capsules
from app.models.time_capsule import TimeCapsuleUpdate


def _capsule_row(*, content: str, media=None):
    now = datetime(2026, 5, 27, tzinfo=UTC)
    return SimpleNamespace(
        id="capsule-id",
        userId="user-id",
        agentId="agent-id",
        workspaceId="workspace-id",
        title=content[:18],
        content=content,
        media=media,
        skin="paper",
        openDate=None,
        status="draft",
        sealedAt=None,
        createdAt=now,
        updatedAt=now,
    )


@pytest.mark.asyncio
async def test_update_capsule_clears_media_with_raw_sql(monkeypatch):
    existing = _capsule_row(
        content="旧内容",
        media={"images": [], "audio": {"duration_seconds": 3, "base64": "AAAA"}},
    )
    updated_without_media = _capsule_row(content="新内容", media=None)
    db = SimpleNamespace(
        timecapsule=SimpleNamespace(
            find_unique=AsyncMock(side_effect=[existing, updated_without_media]),
            update=AsyncMock(return_value=_capsule_row(content="新内容", media=existing.media)),
        ),
        execute_raw=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(time_capsules, "db", db)

    response = await time_capsules.update_capsule(
        "capsule-id",
        TimeCapsuleUpdate(content="新内容", status="draft", media=None),
        user={"sub": "user-id", "role": "user"},
    )

    db.timecapsule.update.assert_awaited_once()
    update_data = db.timecapsule.update.await_args.kwargs["data"]
    assert update_data["content"] == "新内容"
    assert "media" not in update_data
    db.execute_raw.assert_awaited_once()
    assert response.content == "新内容"
    assert response.media is None
