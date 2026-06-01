from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements import engine
from app.services.achievements import service
from app.services.achievements.rules import assistant_message_rules
from app.services.achievements.rules import intent_rules
from app.services.achievements.rules import memory_rules


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
