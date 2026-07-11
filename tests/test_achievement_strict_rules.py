from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements.events import AggregationAchievementEvent
from app.services.achievements.rules import aggregation_rules
from app.services.achievements.rules import daily_rollup_rules
from app.services.achievements.rules import user_message_rules
from app.services.achievements import schedule_status


LOCAL_TZ = timezone(timedelta(hours=8))


def _local_at(day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(
        2026,
        6,
        day,
        hour,
        minute,
        second,
        tzinfo=LOCAL_TZ,
    ).astimezone(UTC)


@pytest.mark.asyncio
async def test_unique_48h_uses_exact_raw_message_equality():
    rows = [
        {"content": "你好", "created_at": _local_at(1, 9)},
        {"content": "你好！", "created_at": _local_at(2, 9)},
    ]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=rows)

    with patch.object(daily_rollup_rules, "db", fake_db):
        result = await daily_rollup_rules._has_complete_unique_48h_window(
            "u1",
            "a1",
            datetime(2026, 6, 2, tzinfo=LOCAL_TZ),
        )

    assert result is True
    query_args = fake_db.query_raw.await_args.args
    assert query_args[4] - query_args[3] == timedelta(hours=48)


@pytest.mark.asyncio
async def test_unique_48h_rejects_an_exact_duplicate_across_days():
    rows = [
        {"content": "完全相同", "created_at": _local_at(1, 9)},
        {"content": "完全相同", "created_at": _local_at(2, 9)},
    ]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=rows)

    with patch.object(daily_rollup_rules, "db", fake_db):
        result = await daily_rollup_rules._has_complete_unique_48h_window(
            "u1",
            "a1",
            datetime(2026, 6, 2, tzinfo=LOCAL_TZ),
        )

    assert result is False


def _quick_reply_rows(
    *,
    count: int,
    slow_index: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    base = _local_at(1, 9)
    for index in range(count):
        assistant_at = base + timedelta(minutes=index)
        delay = 11 if index == slow_index else 10
        rows.extend([
            {
                "conversation_id": "c1",
                "role": "assistant",
                "created_at": assistant_at,
            },
            {
                "conversation_id": "c1",
                "role": "user",
                "created_at": assistant_at + timedelta(seconds=delay),
            },
        ])
    return rows


@pytest.mark.asyncio
async def test_quick_reply_day_accepts_twenty_replies_at_ten_second_boundary():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=_quick_reply_rows(count=20))

    with patch.object(daily_rollup_rules, "db", fake_db):
        result = await daily_rollup_rules._day_has_all_quick_replies(
            "u1",
            "a1",
            datetime(2026, 6, 1, tzinfo=LOCAL_TZ),
            required=20,
        )

    assert result is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [
        _quick_reply_rows(count=19),
        _quick_reply_rows(count=20, slow_index=7),
    ],
)
async def test_quick_reply_day_rejects_below_threshold_or_any_slow_reply(rows):
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=rows)

    with patch.object(daily_rollup_rules, "db", fake_db):
        result = await daily_rollup_rules._day_has_all_quick_replies(
            "u1",
            "a1",
            datetime(2026, 6, 1, tzinfo=LOCAL_TZ),
            required=20,
        )

    assert result is False


@pytest.mark.asyncio
async def test_cross_midnight_sleep_messages_share_one_sleep_period_key():
    sleep_schedule = [
        {
            "start": "23:00",
            "end": "07:00",
            "event": "睡觉",
            "status": "sleep",
        }
    ]
    with (
        patch.object(
            schedule_status,
            "get_cached_schedule",
            AsyncMock(return_value=sleep_schedule),
        ),
        patch.object(schedule_status, "record_event", AsyncMock()),
    ):
        before_midnight = await schedule_status.record_schedule_status_chat(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m1",
            occurred_at=_local_at(1, 23, 30),
        )
        after_midnight = await schedule_status.record_schedule_status_chat(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m2",
            occurred_at=_local_at(2, 0, 30),
        )

    assert before_midnight is not None
    assert after_midnight is not None
    assert before_midnight.bucket == "sleep"
    assert before_midnight.period_key == after_midnight.period_key


@pytest.mark.asyncio
@pytest.mark.parametrize("days", [7, 30])
async def test_schedule_status_streak_requires_all_three_states_every_day(days: int):
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[[{"count": 3}]] * days)
    with patch.object(schedule_status, "db", fake_db):
        assert await schedule_status.has_schedule_status_streak(
            user_id="u1",
            agent_id="a1",
            local_day=datetime(2026, 6, 30, tzinfo=LOCAL_TZ),
            days=days,
        )

    counts = [[{"count": 3}]] * days
    counts[days // 2] = [{"count": 2}]
    fake_db.query_raw = AsyncMock(side_effect=counts)
    with patch.object(schedule_status, "db", fake_db):
        assert not await schedule_status.has_schedule_status_streak(
            user_id="u1",
            agent_id="a1",
            local_day=datetime(2026, 6, 30, tzinfo=LOCAL_TZ),
            days=days,
        )


@pytest.mark.asyncio
async def test_echo_streak_requires_each_user_message_to_immediately_follow_ai():
    rows = [
        {"role": "assistant", "content": "甲乙"},
        {"role": "user", "content": "一二"},
        {"role": "assistant", "content": "丙丁"},
        {"role": "user", "content": "三四"},
        {"role": "assistant", "content": "戊己"},
        {"role": "user", "content": "五六"},
    ]
    with patch.object(
        user_message_rules,
        "_day_messages_until",
        AsyncMock(return_value=rows),
    ):
        assert await user_message_rules._has_echo_same_len_streak(
            "u1",
            "a1",
            "c1",
            _local_at(1, 12),
            required=3,
        )

    rows.insert(4, {"role": "user", "content": "打断"})
    with patch.object(
        user_message_rules,
        "_day_messages_until",
        AsyncMock(return_value=rows),
    ):
        assert not await user_message_rules._has_echo_same_len_streak(
            "u1",
            "a1",
            "c1",
            _local_at(1, 12),
            required=3,
        )


@pytest.mark.asyncio
async def test_aggregation_achievement_counts_completed_multi_fragment_window():
    event = AggregationAchievementEvent(
        user_id="u1",
        agent_id="a1",
        conversation_id="c1",
        source_id="m2",
        part_count=2,
    )
    with (
        patch.object(aggregation_rules, "record_event", AsyncMock()) as record,
        patch.object(aggregation_rules, "_event_count", AsyncMock(return_value=50)),
        patch.object(aggregation_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await aggregation_rules.evaluate_aggregation(event)

    record.assert_awaited_once()
    assert record.await_args.kwargs["event_type"] == "aggregation_window_completed"
    assert unlock.await_args.kwargs["achievement_id"] == 86


@pytest.mark.asyncio
async def test_aggregation_achievement_ignores_single_fragment_window():
    event = AggregationAchievementEvent(
        user_id="u1",
        agent_id="a1",
        conversation_id="c1",
        source_id="m1",
        part_count=1,
    )
    with (
        patch.object(aggregation_rules, "record_event", AsyncMock()) as record,
        patch.object(aggregation_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await aggregation_rules.evaluate_aggregation(event)

    record.assert_not_awaited()
    unlock.assert_not_awaited()
