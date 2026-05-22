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
