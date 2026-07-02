from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.profile_stats import get_profile_stats_for_workspace


@pytest.mark.asyncio
async def test_profile_stats_uses_workspace_messages_and_intimacy_stage():
    workspace = SimpleNamespace(
        id="workspace-1",
        agentId="agent-1",
        createdAt=datetime.now(UTC) - timedelta(days=125),
    )
    agent = SimpleNamespace(
        gender="female",
        currentMbti={"type": "enfp"},
        mbti=None,
    )
    fake_db = SimpleNamespace(
        aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent)),
        query_raw=AsyncMock(
            side_effect=[
                [
                    {
                        "message_count": 3284,
                        "active_chat_hours": 48,
                        "active_chat_minutes": 2912,
                        "recent_7d_message_count": 213,
                    }
                ],
                [{"count": 5}],
            ]
        ),
    )

    with patch("app.services.profile_stats.db", fake_db), patch(
        "app.services.profile_stats.get_intimacy_data",
        new_callable=AsyncMock,
        return_value={"topic_intimacy": 72},
    ):
        stats = await get_profile_stats_for_workspace(
            user_id="user-1",
            workspace=workspace,
        )

    assert stats.workspace_id == "workspace-1"
    assert stats.intimacy_stage == "P4"
    assert stats.intimacy_stage_label == "稳定陪伴"
    assert stats.companion_days == 126
    assert stats.companion_started_on is not None
    assert stats.chat_hours == 48
    assert stats.chat_minutes == 2912
    assert stats.chat_duration_label == "48h32m"
    assert stats.chat_duration_subtitle == "≈ 一起看了26场电影"
    assert stats.message_count == 3284
    assert stats.recent_7d_message_count == 213
    assert stats.recent_7d_message_label == "近7天 +213条"
    assert stats.companion_summary == "唯一伴生对象 · 女 · ENFP"
    assert stats.backpack_count == 5
    assert fake_db.query_raw.await_count == 2
    assert fake_db.query_raw.await_args_list[0].args[1:3] == (
        "workspace-1",
        "user-1",
    )
    assert fake_db.query_raw.await_args_list[1].args[1:] == ("user-1",)
