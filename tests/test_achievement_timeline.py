from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.public import conversations


@pytest.mark.asyncio
async def test_achievement_unlocks_render_as_conversation_timeline_items():
    message = SimpleNamespace(createdAt=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    unlocked_at = datetime(2026, 6, 1, 12, 5, tzinfo=timezone.utc)
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        return_value=[
            {
                "id": "unlock-1",
                "achievement_id": 1,
                "unlocked_at": unlocked_at,
            }
        ]
    )

    with patch.object(conversations, "db", fake_db):
        items = await conversations._achievement_timeline_items(
            conversation_id="c1",
            user_id="u1",
            agent_id="a1",
            messages=[message],
            is_latest_page=True,
            include_metadata=True,
        )

    assert len(items) == 1
    item = items[0]
    assert item.id == "achievement-unlock-1"
    assert item.role == "achievement"
    assert item.content == "初次开口"
    assert item.metadata is not None
    assert item.metadata["achievement"]["achievement_id"] == 1
    assert item.metadata["achievement"]["unlocked"] is True


def test_timeline_datetime_parser_handles_message_and_unlock_formats():
    message_time = conversations._parse_timeline_at("2026-06-01 12:04:00+00:00")
    unlock_time = conversations._parse_timeline_at("2026-06-01T12:05:00+00:00")

    assert unlock_time > message_time


@pytest.mark.asyncio
async def test_empty_older_message_page_does_not_return_achievements():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock()

    with patch.object(conversations, "db", fake_db):
        items = await conversations._achievement_timeline_items(
            conversation_id="c1",
            user_id="u1",
            agent_id="a1",
            messages=[],
            is_latest_page=False,
            include_metadata=True,
        )

    assert items == []
    fake_db.query_raw.assert_not_awaited()
