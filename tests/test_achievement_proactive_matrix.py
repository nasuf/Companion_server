from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements.rules import user_message_rules


NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


async def _evaluate_proactive_response(
    *,
    metadata: dict,
    response_delay_seconds: int = 60,
    proactive_count: int = 1,
    all_quick: bool = False,
    intervening_user_count: int = 0,
) -> tuple[set[int], MagicMock]:
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[
        [{
            "id": "proactive1",
            "created_at": NOW - timedelta(seconds=response_delay_seconds),
            "metadata": {"proactive": True, **metadata},
        }],
        [{"count": intervening_user_count}],
    ])
    with (
        patch.object(user_message_rules, "db", fake_db),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(
            user_message_rules,
            "_event_count",
            AsyncMock(return_value=proactive_count),
        ),
        patch.object(
            user_message_rules,
            "_all_proactive_messages_replied_quickly",
            AsyncMock(return_value=all_quick),
        ),
        patch.object(
            user_message_rules,
            "unlock_achievement",
            AsyncMock(),
        ) as unlock,
    ):
        await user_message_rules._check_proactive_response(
            "u1",
            "a1",
            "w1",
            "c1",
            "user-message1",
            NOW,
        )

    return {
        call.kwargs["achievement_id"] for call in unlock.await_args_list
    }, fake_db


@pytest.mark.asyncio
async def test_first_proactive_response_achievement_25_and_conversation_scope():
    unlocked, fake_db = await _evaluate_proactive_response(metadata={})

    assert 25 in unlocked
    first_query_args = fake_db.query_raw.await_args_list[0].args
    second_query_args = fake_db.query_raw.await_args_list[1].args
    assert first_query_args[3] == "c1"
    assert second_query_args[3] == "c1"


@pytest.mark.asyncio
async def test_intervening_user_message_rejects_proactive_response():
    unlocked, _ = await _evaluate_proactive_response(
        metadata={},
        intervening_user_count=1,
    )

    assert unlocked == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("occasions", "expected_ids"),
    [
        ([{"type": "holiday", "name": "元旦", "owner": "user"}], {61}),
        ([{"type": "birthday", "name": "用户生日", "owner": "user"}], {68}),
        (
            [
                {"type": "holiday", "name": "元旦", "owner": "user"},
                {"type": "birthday", "name": "用户生日", "owner": "user"},
            ],
            {61, 68, 78},
        ),
        ([{"type": "birthday", "name": "AI生日", "owner": "ai"}], set()),
    ],
)
async def test_special_date_response_achievements_use_occasions_metadata(
    occasions: list[dict],
    expected_ids: set[int],
):
    unlocked, _ = await _evaluate_proactive_response(
        metadata={"trigger_type": "special_date", "occasions": occasions},
    )

    assert unlocked & {61, 68, 78} == expected_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy_trigger",
    ["special_holiday", "special_birthday", "special_combined"],
)
async def test_legacy_special_trigger_name_alone_cannot_unlock(
    legacy_trigger: str,
):
    unlocked, _ = await _evaluate_proactive_response(
        metadata={"trigger_type": legacy_trigger},
    )

    assert unlocked.isdisjoint({61, 68, 78})


@pytest.mark.asyncio
async def test_legacy_special_trigger_name_without_occasions_does_not_unlock():
    unlocked, _ = await _evaluate_proactive_response(
        metadata={"trigger_type": "special_combined"},
    )

    assert unlocked.isdisjoint({61, 68, 78})


@pytest.mark.asyncio
@pytest.mark.parametrize(("count", "expected"), [(99, False), (100, True)])
async def test_proactive_response_achievement_89_threshold(
    count: int,
    expected: bool,
):
    unlocked, _ = await _evaluate_proactive_response(
        metadata={},
        proactive_count=count,
    )
    assert (89 in unlocked) is expected


@pytest.mark.asyncio
async def test_all_quick_proactive_achievement_92_positive_and_negative():
    positive, _ = await _evaluate_proactive_response(
        metadata={},
        response_delay_seconds=180,
        proactive_count=100,
        all_quick=True,
    )
    slow, _ = await _evaluate_proactive_response(
        metadata={},
        response_delay_seconds=181,
        proactive_count=100,
        all_quick=True,
    )
    incomplete, _ = await _evaluate_proactive_response(
        metadata={},
        response_delay_seconds=180,
        proactive_count=100,
        all_quick=False,
    )

    assert 92 in positive
    assert 92 not in slow
    assert 92 not in incomplete
