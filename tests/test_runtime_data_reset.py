from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.runtime import data_reset


class _Delegate:
    def __init__(self, *, find_many_result=None, delete_result=0):
        self.find_many = AsyncMock(return_value=find_many_result or [])
        self.delete_many = AsyncMock(return_value=delete_result)
        self.delete = AsyncMock(return_value=None)


@pytest.mark.asyncio
async def test_hard_delete_agent_data_does_not_require_removed_emotion_delegate(monkeypatch):
    """ai_emotion_states was dropped; hard delete must not access its old delegate."""
    fake_db = SimpleNamespace(
        chatworkspace=_Delegate(),
        conversation=_Delegate(),
        usermemory=_Delegate(),
        aimemory=_Delegate(),
        userprofile=_Delegate(),
        memorychangelog=_Delegate(),
        message=_Delegate(),
        intimacy=_Delegate(),
        aidailyschedule=_Delegate(),
        traitfeedbacklog=_Delegate(),
        proactivechatlog=_Delegate(),
        proactivecounter=_Delegate(),
        timetrigger=_Delegate(),
        userportrait=_Delegate(),
        aiagent=_Delegate(),
        execute_raw=AsyncMock(return_value=0),
        query_raw=AsyncMock(return_value=[]),
    )
    fake_redis = SimpleNamespace(
        delete=AsyncMock(return_value=0),
        zrem=AsyncMock(return_value=0),
        scan=AsyncMock(return_value=(0, [])),
    )

    monkeypatch.setattr(data_reset, "db", fake_db)
    monkeypatch.setattr(data_reset, "get_redis", AsyncMock(return_value=fake_redis))

    stats = await data_reset.hard_delete_agent_data("agent-1", "user-1")

    assert stats["agent"] == 1
    fake_db.aiagent.delete.assert_awaited_once_with(where={"id": "agent-1"})
    assert not hasattr(fake_db, "aiemotionstate")


@pytest.mark.asyncio
async def test_hard_delete_agent_data_removes_chat_media_files(monkeypatch, tmp_path):
    scoped_prefix = data_reset.chat_media_storage.conversation_storage_prefix(
        "user-1",
        "conv-1",
    )
    scoped_orphan_keys = [f"{scoped_prefix}orphan.m4a", f"{scoped_prefix}orphan.jpg"]
    media_keys = [
        "user-1_photo.jpg",
        "user-1_voice.m4a",
        "user-1_cover.jpg",
        *scoped_orphan_keys,
    ]
    for key in media_keys:
        (tmp_path / key).write_bytes(b"media")

    async def query_raw(sql, *args):
        if "SELECT id, storage_key" in sql and "FROM chat_message_attachments" in sql:
            assert args == ("user-1", ["conv-1"])
            return [
                {"id": "attachment-image", "storage_key": "user-1_photo.jpg"},
                {"id": "attachment-audio", "storage_key": "user-1_voice.m4a"},
            ]
        if "DELETE FROM chat_message_attachments" in sql:
            assert args == ("user-1", ["conv-1"])
            return [{"id": "attachment-image"}, {"id": "attachment-audio"}]
        if "FROM chat_link_cards" in sql:
            assert args == ("user-1", ["conv-1"])
            return [{"storage_key": "user-1_cover.jpg"}]
        return []

    fake_db = SimpleNamespace(
        chatworkspace=_Delegate(),
        conversation=_Delegate(find_many_result=[SimpleNamespace(id="conv-1")]),
        usermemory=_Delegate(),
        aimemory=_Delegate(),
        userprofile=_Delegate(),
        memorychangelog=_Delegate(),
        message=_Delegate(),
        intimacy=_Delegate(),
        aidailyschedule=_Delegate(),
        traitfeedbacklog=_Delegate(),
        proactivechatlog=_Delegate(),
        proactivecounter=_Delegate(),
        timetrigger=_Delegate(),
        userportrait=_Delegate(),
        aiagent=_Delegate(),
        execute_raw=AsyncMock(return_value=0),
        query_raw=AsyncMock(side_effect=query_raw),
    )
    fake_redis = SimpleNamespace(
        delete=AsyncMock(return_value=0),
        zrem=AsyncMock(return_value=0),
        scan=AsyncMock(return_value=(0, [])),
    )

    monkeypatch.setattr(data_reset, "db", fake_db)
    monkeypatch.setattr(data_reset, "get_redis", AsyncMock(return_value=fake_redis))
    monkeypatch.setattr(data_reset.chat_media_storage, "_MEDIA_DIR", tmp_path)

    stats = await data_reset.hard_delete_agent_data("agent-1", "user-1")

    assert stats["chat_attachments"] == 2
    assert stats["chat_link_cover_media"] == 1
    assert stats["chat_media_files"] == 3
    assert stats["chat_conversation_orphan_media_files"] == 2
    for key in media_keys:
        assert not (tmp_path / key).exists()

    fake_db.message.delete_many.assert_awaited_once_with(
        where={"conversationId": {"in": ["conv-1"]}},
    )


@pytest.mark.asyncio
async def test_chat_media_disk_failure_keeps_attachment_rows_for_retry(
    monkeypatch,
    tmp_path,
):
    statements: list[str] = []
    (tmp_path / "user-1_voice.m4a").write_bytes(b"audio")

    def fail_disk_delete(_storage_key):
        raise PermissionError("read-only disk")

    async def query_raw(sql, *args):
        statements.append(sql)
        assert args == ("user-1", ["conv-1"])
        if "FROM chat_message_attachments" in sql:
            return [{"id": "attachment-audio", "storage_key": "user-1_voice.m4a"}]
        if "FROM chat_link_cards" in sql:
            return []
        raise AssertionError("attachment DELETE must not run after a disk failure")

    fake_db = SimpleNamespace(query_raw=AsyncMock(side_effect=query_raw))
    monkeypatch.setattr(data_reset, "db", fake_db)
    monkeypatch.setattr(data_reset.chat_media_storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(
        data_reset.chat_media_storage,
        "delete_media_file",
        fail_disk_delete,
    )

    with pytest.raises(RuntimeError, match="user-1_voice.m4a"):
        await data_reset._delete_chat_media_for_conversations(
            user_id="user-1",
            conversation_ids=["conv-1"],
        )

    assert not any("DELETE FROM chat_message_attachments" in sql for sql in statements)


def test_chat_media_missing_mount_is_not_treated_as_deleted(monkeypatch, tmp_path):
    missing_media_dir = tmp_path / "missing-chat-mount"
    monkeypatch.setattr(data_reset.chat_media_storage, "_MEDIA_DIR", missing_media_dir)

    with pytest.raises(RuntimeError, match="directory is unavailable"):
        data_reset._delete_chat_media_files(["user-1_photo.jpg"])


@pytest.mark.asyncio
async def test_delete_remaining_user_chat_data_removes_tracked_and_orphan_audio(
    monkeypatch,
    tmp_path,
):
    tracked_key = "user-1_tracked.m4a"
    orphan_key = "user-1_orphan.m4a"
    (tmp_path / tracked_key).write_bytes(b"tracked")
    (tmp_path / orphan_key).write_bytes(b"orphan")

    async def query_raw(sql, *args):
        assert args == ("user-1",)
        if "SELECT id, storage_key" in sql and "FROM chat_message_attachments" in sql:
            return [{"id": "attachment-audio", "storage_key": tracked_key}]
        if "FROM chat_link_cards" in sql:
            return []
        if "DELETE FROM chat_message_attachments" in sql:
            return [{"id": "attachment-audio"}]
        return []

    fake_db = SimpleNamespace(
        conversation=_Delegate(),
        chatworkspace=_Delegate(),
        query_raw=AsyncMock(side_effect=query_raw),
    )
    monkeypatch.setattr(data_reset, "db", fake_db)
    monkeypatch.setattr(data_reset.chat_media_storage, "_MEDIA_DIR", tmp_path)

    stats = await data_reset._delete_remaining_user_chat_data("user-1")

    assert stats["chat_attachments"] == 1
    assert stats["chat_media_files"] == 1
    assert stats["chat_user_media_files"] == 1
    assert not (tmp_path / tracked_key).exists()
    assert not (tmp_path / orphan_key).exists()


@pytest.mark.asyncio
async def test_hard_delete_user_keeps_user_row_when_chat_media_cleanup_fails(monkeypatch):
    fake_db = SimpleNamespace(
        aiagent=_Delegate(),
        user=_Delegate(),
    )
    cleanup_error = RuntimeError("chat media disk unavailable")
    monkeypatch.setattr(data_reset, "db", fake_db)
    monkeypatch.setattr(
        data_reset,
        "_delete_remaining_user_chat_data",
        AsyncMock(side_effect=cleanup_error),
    )

    with pytest.raises(RuntimeError, match="chat media disk unavailable"):
        await data_reset.hard_delete_user_data("user-1")

    fake_db.user.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_hard_delete_user_data_reuses_agent_delete_and_removes_user(monkeypatch):
    fake_db = SimpleNamespace(
        aiagent=_Delegate(find_many_result=[SimpleNamespace(id="agent-1")], delete_result=0),
        user=_Delegate(),
    )

    agent_delete = AsyncMock(return_value={"agent": 1, "redis": 2})
    monkeypatch.setattr(data_reset, "db", fake_db)
    monkeypatch.setattr(data_reset, "hard_delete_agent_data", agent_delete)
    monkeypatch.setattr(
        data_reset,
        "_delete_remaining_user_chat_data",
        AsyncMock(return_value={"conversations": 1}),
    )
    monkeypatch.setattr(
        data_reset,
        "_delete_remaining_user_memory_data",
        AsyncMock(return_value={"user_memories": 2}),
    )
    monkeypatch.setattr(
        data_reset,
        "_delete_remaining_user_side_tables",
        AsyncMock(return_value={"time_capsules": 1, "last_wills": 1}),
    )
    monkeypatch.setattr(data_reset, "_clear_user_redis", AsyncMock(return_value=3))

    stats = await data_reset.hard_delete_user_data("user-1")

    agent_delete.assert_awaited_once_with("agent-1", "user-1")
    fake_db.aiagent.delete_many.assert_awaited_once_with(where={"userId": "user-1"})
    fake_db.user.delete.assert_awaited_once_with(where={"id": "user-1"})
    assert stats["agents_found"] == 1
    assert stats["agent"] == 1
    assert stats["time_capsules"] == 1
    assert stats["last_wills"] == 1
    assert stats["redis"] == 5
    assert stats["user"] == 1


@pytest.mark.asyncio
async def test_delete_remaining_user_side_tables_clears_user_owned_capsules_and_wills(monkeypatch):
    executed_sql: list[str] = []

    async def execute_raw(sql, *args):
        executed_sql.append(sql)
        assert args == ("user-1",)
        return 1

    fake_db = SimpleNamespace(
        query_raw=AsyncMock(return_value=[]),
        execute_raw=AsyncMock(side_effect=execute_raw),
    )

    monkeypatch.setattr(data_reset, "db", fake_db)
    monkeypatch.setattr(
        data_reset.activity_media_storage,
        "delete_user_media_files",
        lambda user_id: 0,
    )

    stats = await data_reset._delete_remaining_user_side_tables("user-1")

    sql_text = "\n".join(executed_sql)
    assert "DELETE FROM time_capsules WHERE user_id = $1" in sql_text
    assert "DELETE FROM last_wills WHERE user_id = $1" in sql_text
    assert "DELETE FROM last_will_deliveries" in sql_text
    assert stats["time_capsules"] == 1
    assert stats["last_wills"] == 1


@pytest.mark.asyncio
async def test_delete_remaining_user_side_tables_removes_capsule_media_files(monkeypatch, tmp_path):
    media_keys = ["user-1_capsule.jpg", "user-1_capsule.m4a", "user-1_orphan.webp"]
    for key in media_keys:
        (tmp_path / key).write_bytes(b"media")

    async def query_raw(sql, *args):
        if "FROM time_capsules" in sql:
            assert args == ("user-1",)
            return [
                {
                    "media": {
                        "images": [{"storage_key": "user-1_capsule.jpg"}],
                        "audio": {"storage_key": "user-1_capsule.m4a"},
                    }
                }
            ]
        if "DELETE FROM offline_activity_media" in sql:
            return []
        return []

    fake_db = SimpleNamespace(
        query_raw=AsyncMock(side_effect=query_raw),
        execute_raw=AsyncMock(return_value=0),
    )

    monkeypatch.setenv("CAPSULE_MEDIA_DIR", str(tmp_path))
    monkeypatch.setattr(data_reset, "db", fake_db)
    monkeypatch.setattr(
        data_reset.activity_media_storage,
        "delete_user_media_files",
        lambda user_id: 0,
    )

    stats = await data_reset._delete_remaining_user_side_tables("user-1")

    assert stats["capsule_media_files"] == 2
    assert stats["capsule_user_media_files"] == 1
    for key in media_keys:
        assert not (tmp_path / key).exists()


def test_capsule_media_disk_failure_is_not_silently_ignored(monkeypatch, tmp_path):
    media_path = tmp_path / "user-1_capsule.m4a"
    media_path.write_bytes(b"audio")
    original_unlink = data_reset.Path.unlink

    def fail_capsule_unlink(path, *args, **kwargs):
        if path == media_path:
            raise PermissionError("read-only disk")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setenv("CAPSULE_MEDIA_DIR", str(tmp_path))
    monkeypatch.setattr(data_reset.Path, "unlink", fail_capsule_unlink)

    with pytest.raises(RuntimeError, match="user-1_capsule.m4a"):
        data_reset._delete_capsule_media_files([media_path.name])

    assert media_path.exists()


def test_capsule_media_missing_mount_is_not_treated_as_deleted(monkeypatch, tmp_path):
    missing_media_dir = tmp_path / "missing-capsule-mount"
    monkeypatch.setenv("CAPSULE_MEDIA_DIR", str(missing_media_dir))

    with pytest.raises(RuntimeError, match="directory is unavailable"):
        data_reset._delete_capsule_media_files(["user-1_capsule.jpg"])


@pytest.mark.asyncio
async def test_hard_delete_agent_data_removes_offline_image_url_files(monkeypatch, tmp_path):
    media_key = "user-1_image_activity.jpg"
    (tmp_path / media_key).write_bytes(b"image")

    async def query_raw(sql, *args):
        if "FROM offline_activity_recommendations" in sql and "SELECT image_urls" in sql:
            assert args == ("user-1", "agent-1")
            return [{"image_urls": [f"/offline/media/{media_key}"]}]
        if "DELETE FROM offline_activity_media" in sql:
            return []
        return []

    fake_db = SimpleNamespace(
        chatworkspace=_Delegate(),
        conversation=_Delegate(),
        usermemory=_Delegate(),
        aimemory=_Delegate(),
        userprofile=_Delegate(),
        memorychangelog=_Delegate(),
        message=_Delegate(),
        intimacy=_Delegate(),
        aidailyschedule=_Delegate(),
        traitfeedbacklog=_Delegate(),
        proactivechatlog=_Delegate(),
        proactivecounter=_Delegate(),
        timetrigger=_Delegate(),
        userportrait=_Delegate(),
        aiagent=_Delegate(),
        execute_raw=AsyncMock(return_value=0),
        query_raw=AsyncMock(side_effect=query_raw),
    )
    fake_redis = SimpleNamespace(
        delete=AsyncMock(return_value=0),
        zrem=AsyncMock(return_value=0),
        scan=AsyncMock(return_value=(0, [])),
    )

    monkeypatch.setattr(data_reset, "db", fake_db)
    monkeypatch.setattr(data_reset, "get_redis", AsyncMock(return_value=fake_redis))
    monkeypatch.setattr(data_reset.activity_media_storage, "_MEDIA_DIR", tmp_path)

    stats = await data_reset.hard_delete_agent_data("agent-1", "user-1")

    assert stats["offline_activity_media_files"] == 1
    assert not (tmp_path / media_key).exists()
