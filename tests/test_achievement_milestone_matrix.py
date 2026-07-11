from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements.rules import user_message_rules


DAY_MILESTONES = [
    (35, 7),
    (54, 15),
    (77, 30),
    (91, 90),
    (97, 180),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("achievement_id", "threshold"), DAY_MILESTONES)
async def test_chat_day_milestone_positive_boundary(
    achievement_id: int,
    threshold: int,
):
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[{"count": threshold}])
    with (
        patch.object(user_message_rules, "db", fake_db),
        patch.object(
            user_message_rules,
            "unlock_achievement",
            AsyncMock(),
        ) as unlock,
    ):
        await user_message_rules._check_daily_chat_day_milestones(
            "u1",
            "a1",
            "w1",
            "c1",
        )

    assert achievement_id in {
        call.kwargs["achievement_id"] for call in unlock.await_args_list
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("achievement_id", "threshold"), DAY_MILESTONES)
async def test_chat_day_milestone_negative_boundary(
    achievement_id: int,
    threshold: int,
):
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[{"count": threshold - 1}])
    with (
        patch.object(user_message_rules, "db", fake_db),
        patch.object(
            user_message_rules,
            "unlock_achievement",
            AsyncMock(),
        ) as unlock,
    ):
        await user_message_rules._check_daily_chat_day_milestones(
            "u1",
            "a1",
            "w1",
            "c1",
        )

    assert achievement_id not in {
        call.kwargs["achievement_id"] for call in unlock.await_args_list
    }


INTIMACY_MILESTONES = [
    (39, 401),
    (67, 601),
    (75, 801),
    (88, 1000),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("achievement_id", "threshold"), INTIMACY_MILESTONES)
async def test_intimacy_milestone_positive_boundary(
    achievement_id: int,
    threshold: int,
):
    with (
        patch.object(
            user_message_rules,
            "get_intimacy_data",
            AsyncMock(return_value={"growth_intimacy": threshold}),
        ),
        patch.object(
            user_message_rules,
            "unlock_achievement",
            AsyncMock(),
        ) as unlock,
    ):
        await user_message_rules._check_intimacy("u1", "a1", "w1", "c1")

    assert achievement_id in {
        call.kwargs["achievement_id"] for call in unlock.await_args_list
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("achievement_id", "threshold"), INTIMACY_MILESTONES)
async def test_intimacy_milestone_negative_boundary(
    achievement_id: int,
    threshold: int,
):
    with (
        patch.object(
            user_message_rules,
            "get_intimacy_data",
            AsyncMock(return_value={"growth_intimacy": threshold - 1}),
        ),
        patch.object(
            user_message_rules,
            "unlock_achievement",
            AsyncMock(),
        ) as unlock,
    ):
        await user_message_rules._check_intimacy("u1", "a1", "w1", "c1")

    assert achievement_id not in {
        call.kwargs["achievement_id"] for call in unlock.await_args_list
    }
