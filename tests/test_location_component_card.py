from __future__ import annotations

from app.services.chat_media.prompt import render_message_content_for_prompt
from app.services.offerings_memory_text import render_component_card_line
from app.api.realtime import ws as ws_module


def _location_card(**payload_overrides):
    payload = {
        "latitude": 31.2304,
        "longitude": 121.4737,
        "address": "上海市黄浦区中山东一路",
        "city": "上海市",
        "region": "黄浦区",
        "country": "中国",
        "source": "device",
    }
    payload.update(payload_overrides)
    return {
        "type": "location",
        "title": "上海市",
        "subtitle": payload["address"],
        "body": payload["region"],
        "footer": "刚刚",
        "accent": "#22C66B",
        "payload": payload,
    }


def test_render_component_card_line_location():
    card = _location_card()
    rendered = render_component_card_line("", card)
    assert "用户分享了当前位置" in rendered
    assert "上海市黄浦区中山东一路" in rendered
    assert "31.23040" in rendered


def test_render_message_content_for_prompt_location_card():
    card = _location_card()
    rendered = render_message_content_for_prompt("", {"component_card": card})
    assert "用户分享了当前位置" in rendered


def test_sanitize_component_card_location():
    sanitized = ws_module._sanitize_component_card(_location_card())
    assert sanitized is not None
    assert sanitized["type"] == "location"
    assert sanitized["payload"]["latitude"] == 31.2304
    assert sanitized["payload"]["longitude"] == 121.4737


def test_sanitize_component_card_location_rejects_invalid_coords():
    bad = _location_card(latitude=999, longitude=121.4737)
    assert ws_module._sanitize_component_card(bad) is None


def test_component_card_reply_message_location():
    message = ws_module._component_card_reply_message("", _location_card())
    assert message is not None
    assert "用户分享了当前位置" in message
    assert "不要重复索要定位权限" in message
