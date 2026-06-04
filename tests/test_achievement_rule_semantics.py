from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements import engine
from app.services.achievements import service
from app.services.achievements import repository
from app.services.achievements.rules import assistant_message_rules
from app.services.achievements.rules import daily_rollup_rules
from app.services.achievements.rules import intent_rules
from app.services.achievements.rules import memory_rules
from app.services.achievements.rules import user_message_rules
from app.services.achievements.utils import count_chars


def test_public_service_exports_event_entrypoints_not_rule_helpers():
    assert "handle_intent_event" in service.__all__
    assert "handle_user_message_event" in service.__all__
    assert "record_event" not in service.__all__
    assert "unlock_achievement" not in service.__all__
    assert not hasattr(service, "record_event")
    assert not hasattr(service, "unlock_achievement")


def test_achievement_count_chars_includes_punctuation_and_emoji():
    assert count_chars("哈！😊 ~") == 4


@pytest.mark.asyncio
async def test_schedule_adjust_achievement_uses_resolved_intent():
    with (
        patch.object(intent_rules, "record_event", AsyncMock()) as record,
        patch.object(intent_rules, "unlock_achievement", AsyncMock()) as unlock,
        patch.object(intent_rules, "_event_count", AsyncMock(return_value=49)),
    ):
        await engine.handle_intent_event(
            intent="schedule_adjust",
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m1",
            metadata={"source": "chat_intent"},
        )

    record.assert_awaited_once()
    assert record.await_args.kwargs["event_type"] == "schedule_adjust_request"
    assert record.await_args.kwargs["source_id"] == "m1"
    unlock.assert_awaited_once()
    assert unlock.await_args.kwargs["achievement_id"] == 20


@pytest.mark.asyncio
async def test_schedule_adjust_achievement_unlocks_after_50_intents():
    with (
        patch.object(intent_rules, "record_event", AsyncMock()),
        patch.object(intent_rules, "unlock_achievement", AsyncMock()) as unlock,
        patch.object(intent_rules, "_event_count", AsyncMock(return_value=50)),
    ):
        await engine.handle_intent_event(
            intent="schedule_adjust",
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m50",
        )

    assert [call.kwargs["achievement_id"] for call in unlock.await_args_list] == [20, 87]


@pytest.mark.asyncio
async def test_unlock_achievement_redis_hit_skips_db_insert():
    fake_redis = MagicMock()
    fake_redis.sismember = AsyncMock(return_value=True)
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock()

    with (
        patch.object(repository, "get_redis", AsyncMock(return_value=fake_redis)),
        patch.object(repository, "db", fake_db),
    ):
        unlocked = await repository.unlock_achievement(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            achievement_id=27,
        )

    assert unlocked is False
    fake_redis.sismember.assert_awaited_once_with("achievements:unlocked:u1:a1", "27")
    fake_db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_unlock_achievement_db_conflict_backfills_redis_cache():
    fake_redis = MagicMock()
    fake_redis.sismember = AsyncMock(return_value=False)
    fake_redis.sadd = AsyncMock()
    fake_redis.expire = AsyncMock()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[])

    with (
        patch.object(repository, "get_redis", AsyncMock(return_value=fake_redis)),
        patch.object(repository, "db", fake_db),
    ):
        unlocked = await repository.unlock_achievement(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            achievement_id=27,
        )

    assert unlocked is False
    fake_db.query_raw.assert_awaited_once()
    fake_redis.sadd.assert_awaited_once_with("achievements:unlocked:u1:a1", "27")
    fake_redis.expire.assert_awaited_once()


@pytest.mark.asyncio
async def test_unlock_achievement_redis_down_falls_back_to_db():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[{"id": "unlock-1", "unlocked_at": datetime(2026, 6, 1, tzinfo=timezone.utc)}])

    with (
        patch.object(repository, "get_redis", AsyncMock(side_effect=Exception("redis down"))),
        patch.object(repository, "db", fake_db),
    ):
        unlocked = await repository.unlock_achievement(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            achievement_id=27,
            notify=False,
        )

    assert unlocked is True
    fake_db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_achievements_backfills_user_agent_redis_cache():
    fake_redis = MagicMock()
    fake_redis.sadd = AsyncMock()
    fake_redis.expire = AsyncMock()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        return_value=[
            {"achievement_id": 27, "unlocked_at": datetime(2026, 6, 1, tzinfo=timezone.utc)},
            {"achievement_id": 64, "unlocked_at": datetime(2026, 6, 1, tzinfo=timezone.utc)},
        ]
    )

    with (
        patch.object(repository, "get_redis", AsyncMock(return_value=fake_redis)),
        patch.object(repository, "db", fake_db),
    ):
        result = await repository.list_achievements(user_id="u1", agent_id="a1")

    assert result["unlocked"] == 2
    fake_redis.sadd.assert_awaited_once()
    assert fake_redis.sadd.await_args.args[0] == "achievements:unlocked:u1:a1"
    assert set(fake_redis.sadd.await_args.args[1:]) == {"27", "64"}
    fake_redis.expire.assert_awaited_once()


@pytest.mark.asyncio
async def test_day_user_messages_uses_local_natural_day_bounds():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[])
    at = datetime(2026, 6, 1, 20, 30, tzinfo=timezone.utc)

    with patch.object(repository, "db", fake_db):
        await repository._day_user_messages("u1", "a1", at)

    args = fake_db.query_raw.await_args.args
    assert args[3] == datetime(2026, 6, 1, 16, 0, tzinfo=timezone.utc)
    assert args[4] == datetime(2026, 6, 2, 16, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_question_achievement_31_unlocks_on_fifth_natural_day_question():
    occurred_at = datetime(2026, 6, 1, 4, 30, tzinfo=timezone.utc)
    rows = [
        {"content": "你在吗？"},
        {"content": "今天忙吗?"},
        {"content": "现在方便吗？"},
        {"content": "要不要聊会儿？"},
        {"content": "你觉得呢?"},
    ]

    with (
        patch.object(user_message_rules, "_message_created_at", AsyncMock(return_value=occurred_at)),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_day_user_messages", AsyncMock(return_value=rows)) as day_messages,
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 0))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_event_count", AsyncMock(return_value=0)),
        patch.object(user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()),
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m-question-5",
            text="你觉得呢?",
        )

    day_messages.assert_awaited_once_with("u1", "a1", occurred_at)
    assert any(call.kwargs["achievement_id"] == 31 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_question_achievement_31_waits_for_five_natural_day_questions():
    occurred_at = datetime(2026, 6, 1, 4, 30, tzinfo=timezone.utc)
    rows = [
        {"content": "你在吗？"},
        {"content": "今天忙吗?"},
        {"content": "现在方便吗？"},
        {"content": "要不要聊会儿？"},
        {"content": "这个不算？！"},
        {"content": "这个也不算？😊"},
    ]

    with (
        patch.object(user_message_rules, "_message_created_at", AsyncMock(return_value=occurred_at)),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 0))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_event_count", AsyncMock(return_value=0)),
        patch.object(user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()),
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m-question-4",
            text="这个也不算？😊",
        )

    assert not any(call.kwargs["achievement_id"] == 31 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_single_haha_achievement_9_unlocks_only_for_exact_user_haha():
    occurred_at = datetime(2026, 6, 1, 4, 30, tzinfo=timezone.utc)

    with (
        patch.object(user_message_rules, "_message_created_at", AsyncMock(return_value=occurred_at)),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_day_user_messages", AsyncMock(return_value=[])),
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 0))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_event_count", AsyncMock(return_value=0)),
        patch.object(user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()),
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m-haha",
            text="哈哈",
        )

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert unlocked_ids.count(9) == 1


@pytest.mark.asyncio
async def test_single_haha_achievement_9_rejects_any_decorated_or_extended_form():
    occurred_at = datetime(2026, 6, 1, 4, 30, tzinfo=timezone.utc)
    rejected_texts = ["  哈哈  ", "哈哈！", "哈哈～", "哈哈😊", "哈 哈", "哈，哈", "哈哈哈", "我哈哈", "哈哈1"]

    with (
        patch.object(user_message_rules, "_message_created_at", AsyncMock(return_value=occurred_at)),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_day_user_messages", AsyncMock(return_value=[])),
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 0))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_event_count", AsyncMock(return_value=0)),
        patch.object(user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()),
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        for index, text in enumerate(rejected_texts, start=1):
            await engine.handle_user_message_event(
                user_id="u1",
                agent_id="a1",
                workspace_id="w1",
                conversation_id="c1",
                message_id=f"m-reject-haha-{index}",
                text=text,
            )

    assert not any(call.kwargs["achievement_id"] == 9 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_single_haha_achievement_9_does_not_unlock_from_assistant_message():
    fake_db = MagicMock()
    fake_db.conversation.find_unique = AsyncMock(
        return_value=MagicMock(userId="u1", agentId="a1", workspaceId="w1")
    )

    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(assistant_message_rules, "record_event", AsyncMock()),
        patch.object(assistant_message_rules, "_event_count", AsyncMock(return_value=0)),
        patch.object(assistant_message_rules, "_check_slow_assistant_reply", AsyncMock()),
        patch.object(assistant_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_assistant_message_event(
            conversation_id="c1",
            message_id="m-assistant-haha",
            text="哈哈",
            metadata={},
            occurred_at=datetime(2026, 6, 1, 4, 30, tzinfo=timezone.utc),
        )

    assert not any(call.kwargs["achievement_id"] == 9 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_user_message_achievement_60_counts_only_user_chars_realtime():
    occurred_at = datetime(2026, 6, 1, 4, 30, tzinfo=timezone.utc)

    with (
        patch.object(user_message_rules, "_message_created_at", AsyncMock(return_value=occurred_at)),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_day_user_messages", AsyncMock(return_value=[])),
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(10000, 0))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_event_count", AsyncMock(return_value=0)),
        patch.object(user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()),
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m-long-user",
            text="长聊",
        )

    assert any(call.kwargs["achievement_id"] == 60 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_user_message_achievement_60_ignores_assistant_chars_realtime():
    occurred_at = datetime(2026, 6, 1, 4, 30, tzinfo=timezone.utc)

    with (
        patch.object(user_message_rules, "_message_created_at", AsyncMock(return_value=occurred_at)),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_day_user_messages", AsyncMock(return_value=[])),
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(9999, 99999))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_event_count", AsyncMock(return_value=0)),
        patch.object(user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()),
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m-ai-long",
            text="还差一点",
        )

    assert not any(call.kwargs["achievement_id"] == 60 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_daily_rollup_single_message_does_not_unlock_first_last_length_match():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": "你好", "created_at": day}]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(2, 0))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 2 in unlocked_ids
    assert 43 not in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_unlocks_first_last_length_match_with_two_messages():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [
        {"content": "哈！", "created_at": day},
        {"content": "嗯😊", "created_at": day + timedelta(hours=1)},
    ]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(4, 0))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 43 in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_achievement_60_counts_only_user_chars():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": "用户长聊", "created_at": day}]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(10000, 0))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 60 in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_achievement_60_ignores_assistant_chars():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": "用户短聊", "created_at": day}]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(9999, 99999))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 60 not in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_unlocks_clean_chat_after_twenty_symbol_free_messages():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": f"纯文字{i}", "created_at": day + timedelta(minutes=i)} for i in range(20)]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(80, 0))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_consecutive_day_flags", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 36 in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_clean_chat_requires_twenty_user_messages():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": f"纯文字{i}", "created_at": day + timedelta(minutes=i)} for i in range(19)]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(76, 0))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 36 not in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_clean_chat_rejects_punctuation_and_emoji_symbols():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": f"纯文字{i}", "created_at": day + timedelta(minutes=i)} for i in range(18)]
    rows.extend([
        {"content": "带标点！", "created_at": day + timedelta(minutes=18)},
        {"content": "带表情😊", "created_at": day + timedelta(minutes=19)},
    ])
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(80, 0))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 36 not in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_unlocks_odd_length_user_messages_only():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": f"奇数{i}", "created_at": day + timedelta(minutes=i)} for i in range(10)]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(50, 999))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 46 in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_odd_length_rejects_any_even_user_message():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": f"奇数{i}", "created_at": day + timedelta(minutes=i)} for i in range(9)]
    rows.append({"content": "偶数字数", "created_at": day + timedelta(minutes=9)})
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(49, 999))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 46 not in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_odd_length_ignores_assistant_message_lengths():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": f"奇数{i}", "created_at": day + timedelta(minutes=i)} for i in range(10)]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(50, 998))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 46 in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_time_mirror_requires_two_user_messages():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [{"content": "只有一句", "created_at": datetime(2026, 6, 1, 4, 21, tzinfo=timezone.utc)}]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(4, 0))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 90 not in unlocked_ids


@pytest.mark.asyncio
async def test_daily_rollup_time_mirror_unlocks_from_first_and_last_user_messages():
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pair = {"user_id": "u1", "agent_id": "a1", "workspace_id": "w1", "conversation_id": "c1"}
    rows = [
        {"content": "第一句", "created_at": datetime(2026, 5, 31, 17, 20, tzinfo=timezone.utc)},
        {"content": "中间句", "created_at": datetime(2026, 5, 31, 17, 40, tzinfo=timezone.utc)},
        {"content": "最后句", "created_at": datetime(2026, 5, 31, 18, 10, tzinfo=timezone.utc)},
    ]
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[pair])

    with (
        patch.object(daily_rollup_rules, "db", fake_db),
        patch.object(daily_rollup_rules, "_day_user_messages", AsyncMock(return_value=rows)),
        patch.object(daily_rollup_rules, "_day_role_char_counts", AsyncMock(return_value=(9, 0))),
        patch.object(daily_rollup_rules, "has_schedule_status_streak", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "_has_complete_unique_48h_window", AsyncMock(return_value=False)),
        patch.object(daily_rollup_rules, "record_event", AsyncMock()),
        patch.object(daily_rollup_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await daily_rollup_rules.run_daily_rollup(day)

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 90 in unlocked_ids


@pytest.mark.asyncio
async def test_assistant_turn_pair_100_counts_whole_user_and_ai_turn():
    fake_db = MagicMock()
    fake_db.conversation.find_unique = AsyncMock(
        return_value=MagicMock(userId="u1", agentId="a1", workspaceId="w1")
    )
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [{"content": "a" * 30}, {"content": "b" * 30}],
            [{"content": "我" * 20}, {"content": "你" * 20}],
        ]
    )

    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(assistant_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_assistant_turn_event(
            conversation_id="c1",
            message_id="a-msg-1",
            assistant_texts=["a" * 30, "b" * 30],
            user_message_ids=["u-msg-1", "u-msg-2"],
            turn_id="user-turn:u-msg-1,u-msg-2",
        )

    unlock.assert_awaited_once()
    assert unlock.await_args.kwargs["achievement_id"] == 81


@pytest.mark.asyncio
async def test_assistant_turn_pair_100_does_not_unlock_from_partial_ai_bubble():
    fake_db = MagicMock()
    fake_db.conversation.find_unique = AsyncMock(
        return_value=MagicMock(userId="u1", agentId="a1", workspaceId="w1")
    )
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [{"content": "a" * 30}, {"content": "b" * 10}],
            [{"content": "我" * 70}],
        ]
    )

    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(assistant_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_assistant_turn_event(
            conversation_id="c1",
            message_id="a-msg-1",
            assistant_texts=["a" * 30],
            user_message_ids=["u-msg-1"],
            turn_id="user-turn:u-msg-1",
        )

    unlock.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeat_message_achievement_71_requires_exact_user_text_match():
    rows = [{"content": "你好"} for _ in range(9)]
    rows.append({"content": "你好！"})

    with patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock:
        await user_message_rules._check_sequences("u1", "a1", "w1", "c1", rows)

    assert not any(call.kwargs["achievement_id"] == 71 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_repeat_message_achievement_71_unlocks_after_ten_exact_user_texts():
    rows = [{"content": "你好！"} for _ in range(10)]

    with patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock:
        await user_message_rules._check_sequences("u1", "a1", "w1", "c1", rows)

    assert any(call.kwargs["achievement_id"] == 71 for call in unlock.await_args_list)


def test_scene_experience_achievement_63_detects_three_daily_windows():
    rows = [
        {"created_at": datetime(2026, 6, 1, 1, 30, tzinfo=timezone.utc)},
        {"created_at": datetime(2026, 6, 1, 11, 15, tzinfo=timezone.utc)},
        {"created_at": datetime(2026, 6, 1, 15, 5, tzinfo=timezone.utc)},
    ]

    assert user_message_rules._has_scene_experience_windows(rows)


def test_scene_experience_achievement_63_requires_all_three_windows():
    rows = [
        {"created_at": datetime(2026, 6, 1, 1, 30, tzinfo=timezone.utc)},
        {"created_at": datetime(2026, 6, 1, 11, 15, tzinfo=timezone.utc)},
    ]

    assert not user_message_rules._has_scene_experience_windows(rows)


@pytest.mark.asyncio
async def test_non_schedule_adjust_intent_does_not_count_schedule_adjustment():
    with (
        patch.object(intent_rules, "record_event", AsyncMock()) as record,
        patch.object(intent_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_intent_event(
            intent="none",
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m1",
        )

    record.assert_not_awaited()
    unlock.assert_not_awaited()


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
        patch.object(memory_rules, "db", fake_db),
        patch.object(memory_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_memory_changelog_event("user-1", "mem-1", "create", "ws1")

    unlock.assert_not_awaited()
    fake_db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_assistant_emoji_counts_for_achievement_59():
    fake_db = MagicMock()
    fake_db.conversation.find_unique = AsyncMock(
        return_value=MagicMock(userId="u1", agentId="a1", workspaceId="w1")
    )
    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(assistant_message_rules, "_check_slow_assistant_reply", AsyncMock()),
        patch.object(assistant_message_rules, "record_event", AsyncMock()) as record,
        patch.object(assistant_message_rules, "_event_count", AsyncMock(return_value=100)),
        patch.object(assistant_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_assistant_message_event(
            conversation_id="c1",
            message_id="a-msg-1",
            text="好呀😊",
            metadata={"reply_index": 0},
        )

    event_types = [call.kwargs["event_type"] for call in record.await_args_list]
    assert "assistant_emoji" in event_types
    assert any(call.kwargs["achievement_id"] == 59 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_plain_assistant_message_does_not_count_for_achievement_59():
    fake_db = MagicMock()
    fake_db.conversation.find_unique = AsyncMock(
        return_value=MagicMock(userId="u1", agentId="a1", workspaceId="w1")
    )
    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(assistant_message_rules, "_check_slow_assistant_reply", AsyncMock()),
        patch.object(assistant_message_rules, "record_event", AsyncMock()) as record,
        patch.object(assistant_message_rules, "_event_count", AsyncMock(return_value=100)),
        patch.object(assistant_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_assistant_message_event(
            conversation_id="c1",
            message_id="a-msg-2",
            text="好呀",
            metadata={"reply_index": 0},
        )

    event_types = [call.kwargs["event_type"] for call in record.await_args_list]
    assert "assistant_emoji" not in event_types
    assert not any(call.kwargs["achievement_id"] == 59 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_assistant_message_does_not_unlock_user_char_achievement_60():
    fake_db = MagicMock()
    fake_db.conversation.find_unique = AsyncMock(
        return_value=MagicMock(userId="u1", agentId="a1", workspaceId="w1")
    )
    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(assistant_message_rules, "_check_slow_assistant_reply", AsyncMock()),
        patch.object(assistant_message_rules, "record_event", AsyncMock()),
        patch.object(assistant_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_assistant_message_event(
            conversation_id="c1",
            message_id="a-msg-long",
            text="我" * 10000,
            metadata={"reply_index": 0},
        )

    assert not any(call.kwargs["achievement_id"] == 60 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_reduplicated_words_do_not_unlock_disabled_achievement_17():
    with (
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_day_user_messages", AsyncMock(return_value=[])),
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 0))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_event_count", AsyncMock(return_value=0)),
        patch.object(user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()),
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m1",
            text="哈哈，好好休息",
        )

    unlocked_ids = [call.kwargs["achievement_id"] for call in unlock.await_args_list]
    assert 17 not in unlocked_ids


@pytest.mark.asyncio
async def test_disabled_user_message_achievements_do_not_unlock_from_legacy_triggers():
    disabled_ids = {3, 4, 14, 15, 22, 32, 34}
    occurred_at = datetime(2026, 6, 1, 2, 9, tzinfo=timezone.utc)
    legacy_trigger_texts = [
        ("m-short-name", "小芜"),
        ("m-emoji-number-state", "小芜123😊你现在在干嘛"),
        ("m-high-emotion", "我超级激动开心到爆炸！！！"),
    ]

    with (
        patch.object(user_message_rules, "_message_created_at", AsyncMock(return_value=occurred_at)),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(
            user_message_rules,
            "_day_user_messages",
            AsyncMock(return_value=[{"content": "测试消息", "created_at": occurred_at}]),
        ),
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 0))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_event_count", AsyncMock(return_value=0)),
        patch.object(user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()),
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        for message_id, text in legacy_trigger_texts:
            await engine.handle_user_message_event(
                user_id="u1",
                agent_id="a1",
                workspace_id="w1",
                conversation_id="c1",
                message_id=message_id,
                text=text,
                agent_name="小芜",
                aggregation_route="fragment_window",
            )

    unlocked_ids = {call.kwargs["achievement_id"] for call in unlock.await_args_list}
    assert unlocked_ids.isdisjoint(disabled_ids)


@pytest.mark.asyncio
async def test_delay_explanation_metadata_does_not_unlock_disabled_achievement_24():
    fake_db = MagicMock()
    fake_db.conversation.find_unique = AsyncMock(
        return_value=MagicMock(userId="u1", agentId="a1", workspaceId="w1")
    )

    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(assistant_message_rules, "_check_slow_assistant_reply", AsyncMock()),
        patch.object(assistant_message_rules, "record_event", AsyncMock()),
        patch.object(assistant_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_assistant_message_event(
            conversation_id="c1",
            message_id="a-delay",
            text="刚才有点事，现在回来啦",
            metadata={"delay_explanation": True},
        )

    assert not any(call.kwargs["achievement_id"] == 24 for call in unlock.await_args_list)


@pytest.mark.asyncio
async def test_disabled_memory_category_achievements_do_not_unlock():
    disabled_cases = [
        ("身份", "年龄", 10),
        ("身份", "性别", 11),
        ("身份", "现居地", 12),
        ("身份", "出生地", 12),
        ("身份", "成长地", 12),
        ("身份", "职业/与经济", 13),
        ("情绪", "高兴", 16),
    ]

    fake_db = MagicMock()
    with (
        patch.object(memory_rules, "db", fake_db),
        patch.object(memory_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        for index, (main, sub, _achievement_id) in enumerate(disabled_cases, start=1):
            fake_db.query_raw = AsyncMock(
                side_effect=[
                    [
                        {
                            "source": "user",
                            "user_id": "u1",
                            "workspace_id": "ws1",
                            "main_category": main,
                            "sub_category": sub,
                            "content": f"{main}/{sub}",
                        }
                    ],
                    [{"agent_id": "a1"}],
                ]
            )
            await engine.handle_memory_changelog_event("u1", f"mem-{index}", "create", "ws1")

    unlocked_ids = {call.kwargs["achievement_id"] for call in unlock.await_args_list}
    assert unlocked_ids.isdisjoint({10, 11, 12, 13, 16})


@pytest.mark.asyncio
async def test_user_message_achievements_use_persisted_message_time_for_reply_pairing():
    persisted_at = datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)
    delayed_worker_at = persisted_at + timedelta(seconds=30)
    with (
        patch.object(user_message_rules, "_message_created_at", AsyncMock(return_value=persisted_at)),
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_day_user_messages", AsyncMock(return_value=[])),
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 0))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "record_schedule_status_chat", AsyncMock(return_value=None)),
        patch.object(user_message_rules, "_check_sequences", AsyncMock()),
        patch.object(user_message_rules, "_check_reply_timing_and_echo", AsyncMock()) as check_reply,
        patch.object(user_message_rules, "_check_proactive_response", AsyncMock()),
        patch.object(user_message_rules, "_check_daily_chat_day_milestones", AsyncMock()),
        patch.object(user_message_rules, "_check_intimacy", AsyncMock()),
        patch.object(user_message_rules, "unlock_achievement", AsyncMock()),
    ):
        await engine.handle_user_message_event(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            message_id="m1",
            text="你好",
            occurred_at=delayed_worker_at,
        )

    check_reply.assert_awaited_once()
    assert check_reply.await_args.args[6] == persisted_at


@pytest.mark.asyncio
async def test_assistant_sticker_metadata_does_not_count_for_achievement_59():
    fake_db = MagicMock()
    fake_db.conversation.find_unique = AsyncMock(
        return_value=MagicMock(userId="u1", agentId="a1", workspaceId="w1")
    )
    with (
        patch.object(assistant_message_rules, "db", fake_db),
        patch.object(assistant_message_rules, "_check_slow_assistant_reply", AsyncMock()),
        patch.object(assistant_message_rules, "record_event", AsyncMock()) as record,
        patch.object(assistant_message_rules, "_event_count", AsyncMock(return_value=100)),
        patch.object(assistant_message_rules, "unlock_achievement", AsyncMock()) as unlock,
    ):
        await engine.handle_assistant_message_event(
            conversation_id="c1",
            message_id="a-msg-3",
            text="好呀",
            metadata={"reply_index": 0, "sticker_url": "https://example.com/s.png"},
        )

    event_types = [call.kwargs["event_type"] for call in record.await_args_list]
    assert "assistant_emoji" not in event_types
    assert not any(call.kwargs["achievement_id"] == 59 for call in unlock.await_args_list)
