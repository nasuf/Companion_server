from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.offerings_followup import (
    build_followup_memory_texts,
    detect_spending_followup,
    extract_followup_detail,
    maybe_record_offering_followup,
)


def _red_packet_offering(**overrides):
    base = {
        "id": "off-rp-1",
        "kind": "red_packet",
        "ticket_amount": 500,
        "agent_value_yuan": 500,
        "agent_name": "小芜",
        "status": "received",
        "workspace_id": "ws-1",
        "user_id": "u1",
        "conversation_id": "c1",
    }
    base.update(overrides)
    return base


def _gift_offering(**overrides):
    base = {
        "id": "off-g-1",
        "kind": "gift",
        "ticket_amount": 95,
        "agent_value_yuan": 95,
        "agent_name": "小芜",
        "product_title": "挂耳咖啡",
        "product_subcategory": "饮品",
        "status": "received",
        "workspace_id": "ws-1",
        "user_id": "u1",
        "conversation_id": "c1",
    }
    base.update(overrides)
    return base


def test_detect_spending_followup_on_purchase_reply():
    assert detect_spending_followup("买好啦！挑到了喜欢的咖啡罐和小挂饰。")
    assert detect_spending_followup("", "买东西了吗？买好了吗？")


def test_detect_spending_followup_ignores_unrelated():
    assert not detect_spending_followup("今天天气不错呀。")


def test_extract_followup_detail_from_ai_reply():
    detail = extract_followup_detail(
        "买好啦！挑到了喜欢的咖啡罐和小挂饰。",
        "",
        _red_packet_offering(),
    )
    assert "咖啡" in detail


def test_build_followup_memory_texts_red_packet():
    user_text, ai_text = build_followup_memory_texts(
        _red_packet_offering(),
        "挂耳咖啡",
    )
    assert "500元红包" in user_text
    assert "挂耳咖啡" in user_text
    assert "小芜用我发的" in user_text
    assert "我用用户发的" in ai_text


def test_build_followup_memory_texts_gift():
    user_text, ai_text = build_followup_memory_texts(
        _gift_offering(),
        "挂耳咖啡",
    )
    assert "挂耳咖啡" in user_text
    assert "用了我送的" in user_text
    assert "我用了用户送的" in ai_text


@pytest.mark.asyncio
async def test_maybe_record_offering_followup_writes_once():
    offering = _red_packet_offering()
    with patch(
        "app.services.offerings_followup.find_recent_received_offering",
        AsyncMock(return_value=offering),
    ), patch(
        "app.services.offerings_followup._followup_already_recorded",
        AsyncMock(return_value=False),
    ), patch(
        "app.services.offerings_followup.store_memory",
        AsyncMock(return_value="mem-1"),
    ) as store, patch(
        "app.services.offerings_followup._mark_followup_recorded",
        AsyncMock(),
    ) as mark:
        ok = await maybe_record_offering_followup(
            user_id="u1",
            agent_id="a1",
            conversation_id="c1",
            workspace_id="ws-1",
            user_message="买好了吗？",
            ai_response="买好啦！挑到了喜欢的咖啡罐。",
        )
    assert ok is True
    assert store.await_count == 2
    mark.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_record_offering_followup_skips_when_no_recent_offering():
    with patch(
        "app.services.offerings_followup.find_recent_received_offering",
        AsyncMock(return_value=None),
    ), patch(
        "app.services.offerings_followup.store_memory",
        AsyncMock(),
    ) as store:
        ok = await maybe_record_offering_followup(
            user_id="u1",
            agent_id="a1",
            conversation_id="c1",
            workspace_id="ws-1",
            user_message="",
            ai_response="买好啦！",
        )
    assert ok is False
    store.assert_not_called()
