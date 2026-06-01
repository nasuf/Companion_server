from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements import memory_events
from app.services.achievements.user_messages import _is_schedule_adjust_request


def test_schedule_adjust_request_avoids_plain_user_routine_statement():
    assert _is_schedule_adjust_request("调整作息吧")
    assert _is_schedule_adjust_request("你早点睡吧")
    assert _is_schedule_adjust_request("帮你晚点睡也可以")

    assert not _is_schedule_adjust_request("你知道我今天早起了吗")
    assert not _is_schedule_adjust_request("我以后要早点睡")
    assert not _is_schedule_adjust_request("明天记得叫醒我")


@pytest.mark.asyncio
async def test_memory_changelog_does_not_unlock_for_mismatched_user():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        return_value=[
            {
                "source": "user",
                "user_id": "other-user",
                "workspace_id": "ws1",
                "main_category": "身份",
                "sub_category": "年龄",
                "content": "用户 18 岁",
            }
        ]
    )
    with (
        patch.object(memory_events, "db", fake_db),
        patch.object(memory_events, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await memory_events.process_memory_changelog("user-1", "mem-1", "create", "ws1")

    unlock.assert_not_awaited()
    fake_db.query_raw.assert_awaited_once()
