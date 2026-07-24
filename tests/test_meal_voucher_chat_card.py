"""Meal-voucher chat card trigger, payload, and once-only state tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.services import meal_voucher_chat as meal_card


def _memory(text: str):
    return SimpleNamespace(text=text)


def test_card_payload_points_to_profile_and_native_expired_state():
    card = meal_card.build_meal_voucher_component_card()

    assert card["type"] == "meal_voucher"
    assert card["payload"]["target_tab"] == "profile"
    assert card["payload"]["target_section"] == "meal_voucher"
    assert card["payload"]["native_status"] == "ended"
    assert card["payload"]["campaign_ends_at"].startswith("2026-08-24")
    assert "活动已结束" in card["payload"]["native_message"]


def test_campaign_stops_new_cards_after_event_end():
    tz = ZoneInfo("Asia/Shanghai")

    assert meal_card.is_meal_voucher_campaign_active(
        datetime(2026, 8, 23, 23, 59, tzinfo=tz)
    )
    assert not meal_card.is_meal_voucher_campaign_active(
        datetime(2026, 8, 24, 0, 0, tzinfo=tz)
    )


def test_turn_requires_question_terms_and_retrieved_meal_knowledge():
    memories = [_memory("我们公司的活动领取方式：霸王餐券需要现场通关")]

    assert meal_card.is_meal_voucher_turn("听说有霸王餐？", memories)
    assert meal_card.is_meal_voucher_turn("那券码怎么激活？", memories)
    assert not meal_card.is_meal_voucher_turn("今天吃什么？", memories)
    assert not meal_card.is_meal_voucher_turn(
        "听说有霸王餐？",
        [_memory("我喜欢吃小锅米线")],
    )


@pytest.mark.asyncio
async def test_existing_history_returns_repeat_without_redis_claim():
    with (
        patch.object(
            meal_card,
            "_history_has_card",
            AsyncMock(return_value=True),
        ),
        patch.object(meal_card, "get_redis", AsyncMock()) as get_redis,
        patch.object(
            meal_card,
            "is_meal_voucher_campaign_active",
            return_value=True,
        ),
    ):
        result = await meal_card.prepare_meal_voucher_card(
            conversation_id="conv-1",
            user_message="霸王餐券怎么用？",
            classified_memories=[_memory("霸王餐券激活后给商家扫码")],
        )

    assert result.state == "repeat"
    assert result.component_card is None
    get_redis.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_turn_claims_and_returns_card():
    redis = MagicMock()
    redis.set = AsyncMock(return_value=True)
    with (
        patch.object(
            meal_card,
            "_history_has_card",
            AsyncMock(return_value=False),
        ),
        patch.object(
            meal_card,
            "get_redis",
            AsyncMock(return_value=redis),
        ),
        patch.object(
            meal_card,
            "is_meal_voucher_campaign_active",
            return_value=True,
        ),
    ):
        result = await meal_card.prepare_meal_voucher_card(
            conversation_id="conv-1",
            user_message="听说有霸王餐？",
            classified_memories=[_memory("我们公司的活动名称：霸王餐")],
        )

    assert result.state == "first"
    assert result.should_send
    assert result.component_card["type"] == "meal_voucher"
    assert result.claim_token
    redis.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_claim_returns_repeat():
    redis = MagicMock()
    redis.set = AsyncMock(return_value=False)
    with (
        patch.object(
            meal_card,
            "_history_has_card",
            AsyncMock(return_value=False),
        ),
        patch.object(
            meal_card,
            "get_redis",
            AsyncMock(return_value=redis),
        ),
        patch.object(
            meal_card,
            "is_meal_voucher_campaign_active",
            return_value=True,
        ),
    ):
        result = await meal_card.prepare_meal_voucher_card(
            conversation_id="conv-1",
            user_message="霸王餐怎么参加？",
            classified_memories=[_memory("霸王餐需要完成线下游戏")],
        )

    assert result.state == "repeat"
    assert not result.should_send


@pytest.mark.asyncio
async def test_finalize_marks_sent_or_releases_pending_claim():
    redis = MagicMock()
    redis.set = AsyncMock()
    redis.eval = AsyncMock()
    decision = meal_card.MealVoucherCardDecision(
        state="first",
        component_card=meal_card.build_meal_voucher_component_card(),
        claim_token="token-1",
    )
    with patch.object(
        meal_card,
        "get_redis",
        AsyncMock(return_value=redis),
    ):
        await meal_card.finalize_meal_voucher_card(
            conversation_id="conv-1",
            decision=decision,
            emitted=True,
        )
        await meal_card.finalize_meal_voucher_card(
            conversation_id="conv-2",
            decision=decision,
            emitted=False,
        )

    redis.set.assert_awaited_once()
    redis.eval.assert_awaited_once()
