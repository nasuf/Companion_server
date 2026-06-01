from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements import engine
from app.services.achievements import service
from app.services.achievements import repository
from app.services.achievements.rules import assistant_message_rules
from app.services.achievements.rules import intent_rules
from app.services.achievements.rules import memory_rules
from app.services.achievements.rules import user_message_rules


def test_public_service_exports_event_entrypoints_not_rule_helpers():
    assert "handle_intent_event" in service.__all__
    assert "handle_user_message_event" in service.__all__
    assert "record_event" not in service.__all__
    assert "unlock_achievement" not in service.__all__
    assert not hasattr(service, "record_event")
    assert not hasattr(service, "unlock_achievement")


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
        patch.object(assistant_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 1))),
        patch.object(assistant_message_rules, "_check_pair_100", AsyncMock()),
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
        patch.object(assistant_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 1))),
        patch.object(assistant_message_rules, "_check_pair_100", AsyncMock()),
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
async def test_reduplicated_words_do_not_unlock_disabled_achievement_17():
    with (
        patch.object(user_message_rules, "record_event", AsyncMock()),
        patch.object(user_message_rules, "_day_user_messages", AsyncMock(return_value=[])),
        patch.object(user_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 0))),
        patch.object(user_message_rules, "_birthday_mmdd", AsyncMock(return_value=None)),
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
        patch.object(assistant_message_rules, "_day_role_char_counts", AsyncMock(return_value=(0, 1))),
        patch.object(assistant_message_rules, "_check_pair_100", AsyncMock()),
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
