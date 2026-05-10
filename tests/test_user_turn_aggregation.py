"""Unified user-turn aggregation planner tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services.interaction.aggregation import push_pending, push_turn_pending
from app.services.interaction.user_turn_aggregation import plan_user_message_aggregation


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
