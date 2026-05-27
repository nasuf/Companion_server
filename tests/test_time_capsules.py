from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

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
            find_unique=AsyncMock(return_value=existing),
        ),
        execute_raw=AsyncMock(return_value=1),
        query_raw=AsyncMock(return_value=[updated_without_media.__dict__]),
    )
    monkeypatch.setattr(time_capsules, "db", db)

    response = await time_capsules.update_capsule(
        "capsule-id",
        TimeCapsuleUpdate(content="新内容", status="draft", media=None),
        user={"sub": "user-id", "role": "user"},
    )

    db.execute_raw.assert_awaited_once()
    sql = db.execute_raw.await_args.args[0]
    args = db.execute_raw.await_args.args[1:]
    assert "media = NULL" in sql
    assert "content = $1" in sql
    assert args[0] == "新内容"
    db.query_raw.assert_awaited_once()
    assert response.content == "新内容"
    assert response.media is None


@pytest.mark.asyncio
async def test_update_capsule_casts_timestamp_fields(monkeypatch):
    existing = _capsule_row(content="旧内容", media=None)
    updated = _capsule_row(content="旧内容", media=None)
    updated.status = "sealed"
    updated.openDate = datetime(2026, 5, 28, tzinfo=UTC)
    updated.sealedAt = datetime(2026, 5, 27, tzinfo=UTC)
    db = SimpleNamespace(
        timecapsule=SimpleNamespace(find_unique=AsyncMock(return_value=existing)),
        execute_raw=AsyncMock(return_value=1),
        query_raw=AsyncMock(return_value=[updated.__dict__]),
    )
    monkeypatch.setattr(time_capsules, "db", db)

    response = await time_capsules.update_capsule(
        "capsule-id",
        TimeCapsuleUpdate(status="sealed", open_date=datetime(2026, 5, 28, tzinfo=UTC).date()),
        user={"sub": "user-id", "role": "user"},
    )

    sql = db.execute_raw.await_args.args[0]
    assert "open_date = $1::timestamp" in sql
    assert "sealed_at = $3::timestamp" in sql
    assert response.status == "sealed"


@pytest.mark.asyncio
async def test_update_capsule_removes_replaced_storage_files(monkeypatch, tmp_path):
    monkeypatch.setattr(time_capsules, "_MEDIA_DIR", tmp_path)
    removed = tmp_path / "user-id_removed.jpg"
    kept = tmp_path / "user-id_kept.m4a"
    removed.write_bytes(b"image")
    kept.write_bytes(b"audio")
    existing = _capsule_row(
        content="旧内容",
        media={
            "images": [{"storage_key": removed.name, "size": len(b"image")}],
            "audio": {
                "storage_key": kept.name,
                "size": len(b"audio"),
                "duration_seconds": 3,
            },
        },
    )
    updated = _capsule_row(
        content="旧内容",
        media={
            "images": [],
            "audio": {
                "storage_key": kept.name,
                "size": len(b"audio"),
                "duration_seconds": 3,
            },
        },
    )
    db = SimpleNamespace(
        timecapsule=SimpleNamespace(find_unique=AsyncMock(return_value=existing)),
        execute_raw=AsyncMock(return_value=1),
        query_raw=AsyncMock(return_value=[updated.__dict__]),
    )
    monkeypatch.setattr(time_capsules, "db", db)

    await time_capsules.update_capsule(
        "capsule-id",
        TimeCapsuleUpdate(media=updated.media),
        user={"sub": "user-id", "role": "user"},
    )

    assert not removed.exists()
    assert kept.exists()


@pytest.mark.asyncio
async def test_delete_capsule_removes_storage_files(monkeypatch, tmp_path):
    monkeypatch.setattr(time_capsules, "_MEDIA_DIR", tmp_path)
    image = tmp_path / "user-id_image.jpg"
    audio = tmp_path / "user-id_audio.m4a"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    existing = _capsule_row(
        content="旧内容",
        media={
            "images": [{"storage_key": image.name}],
            "audio": {"storage_key": audio.name},
        },
    )
    db = SimpleNamespace(
        timecapsule=SimpleNamespace(
            find_unique=AsyncMock(return_value=existing),
            delete=AsyncMock(return_value=existing),
        ),
    )
    monkeypatch.setattr(time_capsules, "db", db)

    await time_capsules.delete_capsule(
        "capsule-id",
        user={"sub": "user-id", "role": "user"},
    )

    assert not image.exists()
    assert not audio.exists()


def test_normalize_media_rejects_foreign_storage_key(monkeypatch, tmp_path):
    monkeypatch.setattr(time_capsules, "_MEDIA_DIR", tmp_path)
    foreign = tmp_path / "other-user_image.jpg"
    foreign.write_bytes(b"image")

    with pytest.raises(HTTPException) as exc_info:
        time_capsules._normalize_media(
            {"images": [{"storage_key": foreign.name, "size": 5}]},
            user_id="user-id",
        )

    assert exc_info.value.status_code == 403
