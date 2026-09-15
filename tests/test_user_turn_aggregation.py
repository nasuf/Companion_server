"""Unified user-turn aggregation planner tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services.interaction.aggregation import is_short_message, push_pending, push_turn_pending
from app.services import runtime_config
from app.services.interaction.user_turn_aggregation import plan_user_message_aggregation


@pytest.mark.asyncio
async def test_plan_aggregation_disabled_routes_immediate(monkeypatch):
    monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
    monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {
        "userMessageAggregationEnabled": False,
    })
    plan = await plan_user_message_aggregation(
        agent_id="agent-A",
        user_id="u1",
        conversation_id="conv-A",
        text="吗",
        reply_context={"delay_seconds": 0},
    )
    assert plan.route == "immediate"
    assert plan.metadata == {"aggregation_disabled": True}


@pytest.mark.asyncio
async def test_plan_short_fragment_starts_fragment_window(fake_aggregation_redis):
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="吗",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "fragment_window"
    assert plan.metadata == {"fragment": True}


@pytest.mark.asyncio
async def test_plan_short_fragment_joins_open_turn_window(fake_aggregation_redis):
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        await push_turn_pending(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="你看过",
        )
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="吗",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "turn_window"
    assert plan.metadata == {"queued": True}


@pytest.mark.asyncio
async def test_plan_non_fragment_flushes_existing_fragment(fake_aggregation_redis):
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        await push_pending(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="你",
            reply_context={"received_at": "2026-05-10T00:00:00+00:00", "delay_seconds": 0},
        )
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="好",
            reply_context={"received_at": "2026-05-10T00:00:01+00:00", "delay_seconds": 0},
        )

    assert plan.route == "immediate"
    assert plan.final_message == "你好"
    assert plan.final_context["received_at"] == "2026-05-10T00:00:00+00:00"
    assert plan.final_context["latest_received_at"] == "2026-05-10T00:00:01+00:00"


@pytest.mark.asyncio
async def test_plan_normal_message_uses_turn_window(fake_aggregation_redis):
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="我最近在看一部美剧",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "turn_window"
    assert plan.metadata == {"queued": True}


@pytest.mark.asyncio
async def test_plan_current_state_message_uses_turn_window(fake_aggregation_redis):
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="你现在在干嘛",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "turn_window"
    assert plan.metadata == {"queued": True}


@pytest.mark.asyncio
async def test_plan_schedule_query_message_uses_turn_window(fake_aggregation_redis):
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="你明天忙吗",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "turn_window"
    assert plan.metadata == {"queued": True}


@pytest.mark.asyncio
async def test_plan_appends_readonly_message_to_existing_delayed_reply(fake_aggregation_redis):
    redis = fake_aggregation_redis
    with (
        patch("app.services.interaction.aggregation.get_redis", return_value=redis),
        patch(
            "app.services.interaction.user_turn_aggregation.has_pending_delayed_messages",
            return_value=True,
        ),
    ):
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="忙啥呢",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "immediate"
    assert plan.metadata == {"queued": True, "append_delayed": True}


@pytest.mark.asyncio
async def test_plan_coalesces_message_arriving_during_inflight_reply(
    fake_aggregation_redis,
):
    """A normal message that lands while a reply is mid-generation must append
    to the delayed queue (coalesce), not spawn an independent parallel turn."""
    redis = fake_aggregation_redis
    with (
        patch("app.services.interaction.aggregation.get_redis", return_value=redis),
        patch(
            "app.services.interaction.user_turn_aggregation.has_pending_delayed_messages",
            return_value=False,
        ),
        patch(
            "app.services.interaction.user_turn_aggregation.is_reply_inflight",
            return_value=True,
        ),
    ):
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="我准备去周游全国",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "immediate"
    assert plan.metadata == {"queued": True, "append_delayed": True}


@pytest.mark.asyncio
async def test_plan_normal_message_uses_turn_window_when_not_inflight(
    fake_aggregation_redis,
):
    """Sanity guard: with no queued reply and nothing in flight, a normal
    message still opens a fresh turn window (no false coalescing)."""
    redis = fake_aggregation_redis
    with (
        patch("app.services.interaction.aggregation.get_redis", return_value=redis),
        patch(
            "app.services.interaction.user_turn_aggregation.has_pending_delayed_messages",
            return_value=False,
        ),
        patch(
            "app.services.interaction.user_turn_aggregation.is_reply_inflight",
            return_value=False,
        ),
    ):
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="我准备去周游全国",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "turn_window"
    assert plan.metadata == {"queued": True}


@pytest.mark.asyncio
async def test_plan_record_request_bypasses_turn_window(fake_aggregation_redis):
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="明天提醒我交报告",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "immediate"


def test_cjk_fragment_is_short_message():
    assert is_short_message("吗") is True
    assert is_short_message("我") is True
    assert is_short_message("好") is False  # COMMON_RESPONSES


def test_emoji_only_is_not_a_fragment():
    assert is_short_message("😂") is False
    assert is_short_message("😂😅") is False
    assert is_short_message("❤️") is False
    assert is_short_message("☀️") is False


@pytest.mark.asyncio
async def test_plan_single_emoji_uses_turn_window_not_fragment(fake_aggregation_redis):
    """A lone emoji is a complete reaction. It must not sit in the 5s fragment
    window (that is how 😂 vanished behind aggregation_scan + the drop-older
    TypeError). Same quiet window as a normal sentence is correct.
    """
    redis = fake_aggregation_redis
    with patch("app.services.interaction.aggregation.get_redis", return_value=redis):
        plan = await plan_user_message_aggregation(
            agent_id="agent-A",
            user_id="u1",
            conversation_id="conv-A",
            text="😂",
            reply_context={"delay_seconds": 0},
        )

    assert plan.route == "turn_window"
