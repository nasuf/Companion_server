from types import SimpleNamespace

from app.api.admin.runtime_config import (
    ConfigPayload,
    _payload_to_data,
    _payload_to_update_data,
    _row_to_payload,
)
from app.config import settings
from app.services import runtime_config


def _loaded_caches(monkeypatch):
    monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
    monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {})


def test_chat_management_defaults_to_env(monkeypatch):
    _loaded_caches(monkeypatch)
    resolved = runtime_config.resolve_config_sync(agent_id=None)
    assert resolved.reply_delay_enabled is settings.reply_delay_enabled
    assert resolved.reply_delay_max_seconds == settings.reply_delay_max_seconds
    assert (
        resolved.user_message_aggregation_enabled
        is settings.user_message_aggregation_enabled
    )


def test_chat_management_via_system_config(monkeypatch):
    _loaded_caches(monkeypatch)
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {
        "replyDelayEnabled": True,
        "replyDelayMaxSeconds": 45,
        "userMessageAggregationEnabled": False,
    })
    resolved = runtime_config.resolve_config_sync(agent_id=None)
    assert resolved.reply_delay_enabled is True
    assert resolved.reply_delay_max_seconds == 45
    assert resolved.user_message_aggregation_enabled is False


def test_payload_to_update_data_chat_management_partial():
    payload = ConfigPayload(reply_delay_enabled=True, reply_delay_max_seconds=90)
    data = _payload_to_update_data(payload, include_global_only=True)
    assert data == {
        "replyDelayEnabled": True,
        "replyDelayMaxSeconds": 90,
    }


def test_payload_to_data_chat_management_global_only():
    payload = ConfigPayload(
        reply_delay_enabled=False,
        reply_delay_max_seconds=120,
        user_message_aggregation_enabled=True,
    )
    global_data = _payload_to_data(payload, include_global_only=True)
    assert global_data["replyDelayEnabled"] is False
    assert global_data["replyDelayMaxSeconds"] == 120
    assert global_data["userMessageAggregationEnabled"] is True
    agent_data = _payload_to_data(payload)
    assert "replyDelayEnabled" not in agent_data


def test_row_to_payload_chat_management_getattr_safe():
    row = SimpleNamespace(
        onlineModel=None,
        remoteProvider=None,
        remoteChatProvider=None,
        remoteSmallProvider=None,
        localChatModel=None,
        localSmallModel=None,
        remoteChatModel=None,
        remoteSmallModel=None,
        replyDelayEnabled=True,
        replyDelayMaxSeconds=180,
        userMessageAggregationEnabled=False,
    )
    payload = _row_to_payload(row)
    assert payload["reply_delay_enabled"] is True
    assert payload["reply_delay_max_seconds"] == 180
    assert payload["user_message_aggregation_enabled"] is False
