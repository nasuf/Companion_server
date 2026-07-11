from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements import repository, schedule_status
from app.services.achievements.rules import daily_rollup_rules, user_message_rules


@pytest.mark.asyncio
async def test_event_count_excludes_soft_deleted_conversations():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[{"count": 4}])
    with patch.object(repository, "db", fake_db):
        count = await repository._event_count("u1", "a1", "assistant_sticker")

    assert count == 4
    sql = fake_db.query_raw.await_args.args[0]
    assert "LEFT JOIN conversations" in sql
    assert "c.is_deleted = FALSE" in sql


@pytest.mark.asyncio
async def test_memory_count_filters_workspace_and_archived_rows():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[{"count": 20}])
    with patch.object(repository, "db", fake_db):
        count = await repository._memory_count(
            "u1",
            "w1",
            "偏好",
            None,
        )

    assert count == 20
    sql, user_id, workspace_id, main, sub = fake_db.query_raw.await_args.args
    assert "is_archived = FALSE" in sql
    assert (user_id, workspace_id, main, sub) == ("u1", "w1", "偏好", None)


@pytest.mark.asyncio
async def test_schedule_status_streak_checks_every_day_and_deleted_filter():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[[{"count": 3}], [{"count": 3}], [{"count": 3}]]
    )
    with patch.object(schedule_status, "db", fake_db):
        result = await schedule_status.has_schedule_status_streak(
            user_id="u1",
            agent_id="a1",
            local_day=datetime(2026, 6, 3, tzinfo=UTC),
            days=3,
        )

    assert result is True
    assert fake_db.query_raw.await_count == 3
    assert all(
        "c.is_deleted = FALSE" in call.args[0]
        for call in fake_db.query_raw.await_args_list
    )


@pytest.mark.asyncio
async def test_schedule_status_streak_stops_when_one_day_lacks_a_state():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[[{"count": 3}], [{"count": 2}], [{"count": 3}]]
    )
    with patch.object(schedule_status, "db", fake_db):
        result = await schedule_status.has_schedule_status_streak(
            user_id="u1",
            agent_id="a1",
            local_day=datetime(2026, 6, 3, tzinfo=UTC),
            days=3,
        )

    assert result is False
    assert fake_db.query_raw.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("proactive_count", "quick_count", "expected"),
    [(100, 100, True), (100, 99, False), (99, 99, False)],
)
async def test_all_proactive_messages_quick_predicate(
    proactive_count: int,
    quick_count: int,
    expected: bool,
):
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[
        [{"count": proactive_count}],
        [{"count": quick_count}],
    ])
    with patch.object(user_message_rules, "db", fake_db):
        result = await user_message_rules._all_proactive_messages_replied_quickly(
            "u1",
            "a1",
            datetime(2026, 6, 1, tzinfo=UTC),
            required=100,
        )

    assert result is expected
    assert "c.is_deleted = FALSE" in fake_db.query_raw.await_args_list[1].args[0]


@pytest.mark.asyncio
async def test_daily_milestone_query_requires_thirty_messages_per_distinct_day():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[{"count": 7}])
    with (
        patch.object(user_message_rules, "db", fake_db),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()),
    ):
        await user_message_rules._check_daily_chat_day_milestones(
            "u1",
            "a1",
            "w1",
            "c1",
        )

    sql = fake_db.query_raw.await_args.args[0]
    assert "GROUP BY d" in sql
    assert "HAVING COUNT(*) >= 30" in sql


@pytest.mark.asyncio
async def test_consecutive_day_flags_exclude_deleted_conversations():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[{"exists": 1}])
    with patch.object(daily_rollup_rules, "db", fake_db):
        result = await daily_rollup_rules._has_consecutive_day_flags(
            "u1",
            "a1",
            "clean_chat_day",
            datetime(2026, 6, 2, tzinfo=UTC),
            2,
        )

    assert result is True
    assert all(
        "c.is_deleted = FALSE" in call.args[0]
        for call in fake_db.query_raw.await_args_list
    )
