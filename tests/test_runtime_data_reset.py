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
    media_keys = [
        "user-1_photo.jpg",
        "user-1_second.jpg",
        "user-1_cover.jpg",
    ]
    for key in media_keys:
        (tmp_path / key).write_bytes(b"image")

    async def query_raw(sql, *args):
        if "DELETE FROM chat_message_attachments" in sql:
            assert args == ("user-1", ["conv-1"])
            return [
                {"storage_key": "user-1_photo.jpg"},
                {"storage_key": "user-1_second.jpg"},
            ]
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
    for key in media_keys:
        assert not (tmp_path / key).exists()

    fake_db.message.delete_many.assert_awaited_once_with(
        where={"conversationId": {"in": ["conv-1"]}},
    )


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
