from datetime import UTC, datetime
import base64
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api.public import time_capsules
from app.models.time_capsule import TimeCapsuleCreate, TimeCapsuleUpdate


def _capsule_row(
    *,
    content: str,
    media=None,
    status: str = "draft",
    open_date=None,
    opened_at=None,
    agent_id="agent-id",
):
    now = datetime(2026, 5, 27, tzinfo=UTC)
    return SimpleNamespace(
        id="capsule-id",
        userId="user-id",
        agentId=agent_id,
        workspaceId="workspace-id",
        title=content[:18],
        content=content,
        media=media,
        skin="paper",
        openDate=open_date,
        status=status,
        sealedAt=None,
        openedAt=opened_at,
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
async def test_update_capsule_rejects_opened_capsule(monkeypatch):
    opened = _capsule_row(
        content="旧内容",
        status="sealed",
        opened_at=datetime(2026, 5, 28, tzinfo=UTC),
    )
    db = SimpleNamespace(
        timecapsule=SimpleNamespace(find_unique=AsyncMock(return_value=opened)),
    )
    monkeypatch.setattr(time_capsules, "db", db)

    with pytest.raises(HTTPException) as exc_info:
        await time_capsules.update_capsule(
            "capsule-id",
            TimeCapsuleUpdate(content="新内容"),
            user={"sub": "user-id", "role": "user"},
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_get_capsule_rejects_unopened_locked_content(monkeypatch):
    pending = _capsule_row(
        content="秘密内容",
        status="sealed",
        open_date=datetime(2026, 6, 1, tzinfo=UTC),
    )
    monkeypatch.setattr(
        time_capsules,
        "_fetch_capsule_full",
        AsyncMock(return_value=pending),
    )

    with pytest.raises(HTTPException) as exc_info:
        await time_capsules.get_capsule(
            "capsule-id",
            user={"sub": "user-id", "role": "user"},
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_list_capsules_redacts_locked_content_and_does_not_filter_agent(monkeypatch):
    pending = _capsule_row(
        content="秘密内容",
        status="sealed",
        open_date=datetime(2026, 6, 1, tzinfo=UTC),
        agent_id="old-agent-id",
    )
    db = SimpleNamespace(query_raw=AsyncMock(return_value=[pending.__dict__]))
    monkeypatch.setattr(time_capsules, "db", db)

    response = await time_capsules.list_capsules(
        agent_id="new-agent-id",
        workspace_id="new-workspace-id",
        state="pending",
        user={"sub": "user-id", "role": "user"},
    )

    sql = db.query_raw.await_args.args[0]
    assert "agent_id =" not in sql
    assert "workspace_id =" not in sql
    assert db.query_raw.await_args.args[1] == "user-id"
    assert "open_date::date >" in sql
    assert response[0].content == ""
    assert response[0].title is None


@pytest.mark.asyncio
async def test_create_capsule_allows_user_owned_capsule_without_agent(monkeypatch):
    inserted = {}

    async def fake_insert(**kwargs):
        inserted.update(kwargs)

    created = _capsule_row(content="给未来的我", agent_id=None)
    monkeypatch.setattr(time_capsules, "_ensure_capsule_context_scope", AsyncMock())
    monkeypatch.setattr(time_capsules, "_insert_capsule_raw", fake_insert)
    monkeypatch.setattr(
        time_capsules,
        "_fetch_capsule_light",
        AsyncMock(return_value=created),
    )

    response = await time_capsules.create_capsule(
        TimeCapsuleCreate(content="给未来的我", status="draft"),
        user={"sub": "user-id", "role": "user"},
    )

    assert inserted["agent_id"] is None
    assert response.agent_id is None


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


@pytest.mark.asyncio
async def test_delete_capsule_keeps_row_when_media_delete_fails(monkeypatch):
    def fail_media_delete(*_args, **_kwargs):
        raise RuntimeError("capsule disk unavailable")

    existing = _capsule_row(
        content="旧内容",
        media={"audio": {"storage_key": "user-id_audio.m4a"}},
    )
    delete_row = AsyncMock(return_value=existing)
    db = SimpleNamespace(
        timecapsule=SimpleNamespace(
            find_unique=AsyncMock(return_value=existing),
            delete=delete_row,
        ),
    )
    monkeypatch.setattr(time_capsules, "db", db)
    monkeypatch.setattr(
        time_capsules,
        "_delete_media_files",
        fail_media_delete,
    )

    with pytest.raises(RuntimeError, match="capsule disk unavailable"):
        await time_capsules.delete_capsule(
            "capsule-id",
            user={"sub": "user-id", "role": "user"},
        )

    delete_row.assert_not_awaited()


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


def test_capsule_image_limit_is_10mb():
    assert time_capsules._MAX_IMAGE_BYTES == 10 * 1024 * 1024


def test_normalize_media_checks_decoded_image_size(monkeypatch):
    monkeypatch.setattr(time_capsules, "_MAX_IMAGE_BYTES", 3)

    with pytest.raises(HTTPException) as exc_info:
        time_capsules._normalize_media(
            {
                "images": [
                    {
                        "base64": base64.b64encode(b"image").decode("ascii"),
                        "size": 1,
                    }
                ]
            },
            user_id="user-id",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Image must be under 10MB"


@pytest.mark.asyncio
async def test_cleanup_unreferenced_media_keeps_referenced_files(monkeypatch, tmp_path):
    monkeypatch.setattr(time_capsules, "_MEDIA_DIR", tmp_path)
    referenced = tmp_path / "user-id_keep.jpg"
    orphan = tmp_path / "user-id_orphan.m4a"
    fresh = tmp_path / "user-id_fresh.jpg"
    referenced.write_bytes(b"image")
    orphan.write_bytes(b"audio")
    fresh.write_bytes(b"new")
    old_time = 1_700_000_000
    now = old_time + 100_000
    os.utime(referenced, (old_time, old_time))
    os.utime(orphan, (old_time, old_time))
    os.utime(fresh, (now, now))
    monkeypatch.setattr(time_capsules.time_module, "time", lambda: now)
    db = SimpleNamespace(
        query_raw=AsyncMock(
            return_value=[
                {
                    "media": {
                        "images": [{"storage_key": referenced.name}],
                    }
                }
            ]
        )
    )
    monkeypatch.setattr(time_capsules, "db", db)

    await time_capsules._cleanup_unreferenced_media(
        "user-id",
        max_age_seconds=3600,
    )

    assert referenced.exists()
    assert not orphan.exists()
    assert fresh.exists()
