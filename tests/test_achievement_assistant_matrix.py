from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements.events import AssistantMessageAchievementEvent
from app.services.achievements.rules import assistant_message_rules


LOCAL_TZ = timezone(timedelta(hours=8))


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 6, 1, hour, minute, tzinfo=LOCAL_TZ).astimezone(UTC)


async def _evaluate(
    *,
    text: str = "普通回复",
    metadata: dict | None = None,
    at: datetime | None = None,
    event_count: int = 0,
    birthday: tuple[int, int] | None = None,
) -> set[int]:
    fake_db = MagicMock()
    fake_db.conversation.find_unique = AsyncMock(
        return_value=MagicMock(userId="u1", agentId="a1", workspaceId="w1")
    )
    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(
            assistant_message_rules,
            "_event_count",
            AsyncMock(return_value=event_count),
        ),
        patch.object(
            assistant_message_rules,
            "_birthday_mmdd",
            AsyncMock(return_value=birthday),
        ),
        patch.object(
            assistant_message_rules,
            "_check_slow_assistant_reply",
            AsyncMock(),
        ),
        patch.object(assistant_message_rules, "record_event", AsyncMock()),
        patch.object(
            assistant_message_rules,
            "unlock_achievement",
            AsyncMock(),
        ) as unlock,
    ):
        await assistant_message_rules.evaluate_assistant_message(
            AssistantMessageAchievementEvent(
                conversation_id="c1",
                message_id="a1",
                text=text,
                metadata=metadata,
                occurred_at=at or _at(12),
            )
        )
    return {call.kwargs["achievement_id"] for call in unlock.await_args_list}


@pytest.mark.asyncio
async def test_first_user_message_achievement_1_never_unlocks_from_ai_message():
    assert 1 not in await _evaluate(text="AI消息")


@pytest.mark.asyncio
async def test_memory_proactive_achievement_55_positive_and_negative():
    positive = await _evaluate(
        metadata={"proactive": True, "trigger_type": "memory_proactive"},
    )
    negative = await _evaluate(
        metadata={"proactive": True, "trigger_type": "silence_wakeup"},
    )

    assert 55 in positive
    assert 55 not in negative


@pytest.mark.asyncio
@pytest.mark.parametrize(("count", "expected"), [(99, False), (100, True)])
async def test_sticker_achievement_59_threshold(count: int, expected: bool):
    unlocked = await _evaluate(
        text="带表情包的回复",
        metadata={"sticker_url": "https://example.com/sticker.png"},
        event_count=count,
    )
    assert (59 in unlocked) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("count", "expected"), [(499, False), (500, True)])
async def test_short_assistant_reply_achievement_79_threshold(
    count: int,
    expected: bool,
):
    unlocked = await _evaluate(text="好呀", event_count=count)
    assert (79 in unlocked) is expected


@pytest.mark.asyncio
async def test_proactive_1314_achievement_84_exact_minute_only():
    positive = await _evaluate(
        metadata={"proactive": True},
        at=_at(13, 14),
    )
    negative = await _evaluate(
        metadata={"proactive": True},
        at=_at(13, 15),
    )

    assert 84 in positive
    assert 84 not in negative


@pytest.mark.asyncio
async def test_user_birthday_clock_achievement_96_exact_minute_only():
    positive = await _evaluate(
        metadata={"proactive": True},
        at=_at(6, 11),
        birthday=(6, 11),
    )
    negative = await _evaluate(
        metadata={"proactive": True},
        at=_at(6, 12),
        birthday=(6, 11),
    )

    assert 96 in positive
    assert 96 not in negative


@pytest.mark.asyncio
@pytest.mark.parametrize(("delay_seconds", "expected"), [(1799, False), (1800, True)])
async def test_slow_reply_achievement_30_boundary(
    delay_seconds: int,
    expected: bool,
):
    at = _at(12)
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        return_value=[{"created_at": at - timedelta(seconds=delay_seconds)}]
    )
    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(
            assistant_message_rules,
            "unlock_achievement",
            AsyncMock(),
        ) as unlock,
    ):
        await assistant_message_rules._check_slow_assistant_reply(
            "u1",
            "a1",
            "w1",
            "c1",
            at,
        )

    unlocked = {call.kwargs["achievement_id"] for call in unlock.await_args_list}
    assert (30 in unlocked) is expected
    assert fake_db.query_raw.await_args.args[3] == "c1"
