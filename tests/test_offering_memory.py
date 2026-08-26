from __future__ import annotations

from app.services.chat_media.prompt import render_message_content_for_prompt
from app.services.offerings_memory_text import (
    build_offering_history_text,
    build_offering_memory_texts,
    offering_recall_search_query,
    render_component_card_line,
)


def _red_packet_offering(**overrides):
    base = {
        "id": "off-1",
        "kind": "red_packet",
        "ticket_amount": 500,
        "agent_value_yuan": 500,
        "agent_name": "小芜",
        "blessing": "给你的一点心意",
        "status": "sent",
        "created_at": "2026-08-23T03:06:44+00:00",
    }
    base.update(overrides)
    return base


def _gift_offering(**overrides):
    base = {
        "id": "off-g1",
        "kind": "gift",
        "ticket_amount": 95,
        "agent_value_yuan": 95,
        "agent_name": "小芜",
        "product_title": "挂耳咖啡",
        "product_subcategory": "饮品",
        "status": "received",
        "created_at": "2026-08-23T11:06:00+00:00",
        "received_at": "2026-08-23T11:06:10+00:00",
    }
    base.update(overrides)
    return base


def test_red_packet_memory_text_includes_yuan_and_tickets():
    user_text, ai_text = build_offering_memory_texts(_red_packet_offering(), event="sent")
    assert "500元红包" in user_text
    assert "小芜" in user_text
    assert "500元红包" in ai_text
    assert "给你的一点心意" in user_text


def test_red_packet_received_memory_text():
    offering = _red_packet_offering(status="received")
    user_text, ai_text = build_offering_memory_texts(offering, event="received")
    assert "领取" in user_text
    assert "领取" in ai_text
    assert "500元" in user_text


def test_gift_memory_text_includes_product_title():
    user_text, ai_text = build_offering_memory_texts(_gift_offering(), event="sent")
    assert "挂耳咖啡" in user_text
    assert "挂耳咖啡" in ai_text


def test_history_text_marks_received_gift():
    text = build_offering_history_text(_gift_offering())
    assert "挂耳咖啡" in text
    assert "收下" in text


def test_render_component_card_red_packet_for_prompt():
    card = {
        "type": "red_packet",
        "body": "给你的一点心意",
        "payload": {
            "ticket_amount": 500,
            "agent_value_yuan": 500,
            "status_label": "已领取",
        },
    }
    rendered = render_component_card_line("", card)
    assert "500元" in rendered
    assert "红包" in rendered
    assert "已领取" in rendered


def test_render_message_content_for_prompt_renders_offering_card():
    metadata = {
        "component_card": {
            "type": "gift",
            "title": "挂耳咖啡",
            "payload": {
                "product_title": "挂耳咖啡",
                "product_subcategory": "饮品",
                "status_label": "已接收",
            },
        }
    }
    rendered = render_message_content_for_prompt("", metadata)
    assert "挂耳咖啡" in rendered
    assert "礼物" in rendered


def test_offering_recall_search_query_for_money_and_coffee():
    q = offering_recall_search_query("还记得我上次给你钱，你买了挂耳咖啡吗")
    assert q is not None
    assert "红包" in q
    assert "咖啡" in q
    assert "买了" in q


def test_offering_recall_search_query_ignores_unrelated_chat():
    assert offering_recall_search_query("今天天气怎么样") is None
