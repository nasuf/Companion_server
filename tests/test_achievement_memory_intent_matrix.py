from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements.events import IntentAchievementEvent, MemoryChangelogAchievementEvent
from app.services.achievements.rules import intent_rules, memory_rules


MEMORY_MAPPING_CASES = [
    ("身份", "姓名", 29),
    ("情绪", "悲伤", 23),
    ("思维", "理想与目标", 49),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("main", "sub", "achievement_id"), MEMORY_MAPPING_CASES)
async def test_memory_achievement_mapping_positive_cases(
    main: str,
    sub: str,
    achievement_id: int,
):
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[
        [{
            "source": "user",
            "user_id": "u1",
            "workspace_id": "w1",
            "main_category": main,
            "sub_category": sub,
            "content": "记忆内容",
        }],
        [{"agent_id": "a1"}],
    ])
    with (
        patch.object(memory_rules, "db", fake_db),
        patch.object(memory_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await memory_rules.evaluate_memory_changelog(
            MemoryChangelogAchievementEvent(
                user_id="u1",
                memory_id="mem1",
                operation="create",
                workspace_id="w1",
            )
        )

    assert achievement_id in {
        call.kwargs["achievement_id"] for call in unlock.await_args_list
    }


@pytest.mark.asyncio
async def test_memory_mapping_rejects_ai_owned_memory():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[{
        "source": "ai",
        "user_id": "u1",
        "workspace_id": "w1",
        "main_category": "情绪",
        "sub_category": "悲伤",
        "content": "AI记忆",
    }])
    with (
        patch.object(memory_rules, "db", fake_db),
        patch.object(memory_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await memory_rules.evaluate_memory_changelog(
            MemoryChangelogAchievementEvent(
                user_id="u1",
                memory_id="mem1",
                operation="create",
                workspace_id="w1",
            )
        )

    unlock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("main", "sub", "count", "achievement_id", "expected"),
    [
        ("偏好", "兴趣偏好", 19, 65, False),
        ("偏好", "兴趣偏好", 20, 65, True),
        ("情绪", "恐惧", 9, 66, False),
        ("情绪", "恐惧", 10, 66, True),
    ],
)
async def test_memory_threshold_achievement_boundaries(
    main: str,
    sub: str,
    count: int,
    achievement_id: int,
    expected: bool,
):
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[
        [{
            "source": "user",
            "user_id": "u1",
            "workspace_id": "w1",
            "main_category": main,
            "sub_category": sub,
            "content": "记忆内容",
        }],
        [{"agent_id": "a1"}],
    ])
    with (
        patch.object(memory_rules, "db", fake_db),
        patch.object(memory_rules, "_memory_count", AsyncMock(return_value=count)),
        patch.object(memory_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await memory_rules.evaluate_memory_changelog(
            MemoryChangelogAchievementEvent(
                user_id="u1",
                memory_id="mem1",
                operation="create",
                workspace_id="w1",
            )
        )

    unlocked = {
        call.kwargs["achievement_id"] for call in unlock.await_args_list
    }
    assert (achievement_id in unlocked) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "expected_ids"),
    [
        (49, {20}),
        (50, {20, 87}),
    ],
)
async def test_schedule_adjustment_threshold_boundaries(
    count: int,
    expected_ids: set[int],
):
    with (
        patch.object(intent_rules, "record_event", AsyncMock()),
        patch.object(intent_rules, "_event_count", AsyncMock(return_value=count)),
        patch.object(intent_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await intent_rules.evaluate_intent(
            IntentAchievementEvent(
                intent="schedule_adjust",
                user_id="u1",
                agent_id="a1",
                workspace_id="w1",
                conversation_id="c1",
                message_id="m1",
            )
        )

    assert {
        call.kwargs["achievement_id"] for call in unlock.await_args_list
    } == expected_ids


@pytest.mark.asyncio
async def test_non_schedule_intent_does_not_unlock_schedule_achievements():
    with (
        patch.object(intent_rules, "record_event", AsyncMock()) as record,
        patch.object(intent_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await intent_rules.evaluate_intent(
            IntentAchievementEvent(
                intent="schedule_query",
                user_id="u1",
                agent_id="a1",
                workspace_id="w1",
                conversation_id="c1",
                message_id="m1",
            )
        )

    record.assert_not_awaited()
    unlock.assert_not_awaited()
