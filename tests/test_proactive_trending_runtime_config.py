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


def test_proactive_trending_defaults_to_env(monkeypatch):
    _loaded_caches(monkeypatch)
    resolved = runtime_config.resolve_config_sync(agent_id=None)
    assert resolved.proactive_trending_enabled is settings.proactive_trending_enabled
    assert resolved.proactive_trending_probability == settings.proactive_trending_probability
    assert (
        resolved.proactive_trending_link_probability
        == settings.proactive_trending_link_probability
    )
    assert resolved.proactive_trending_cache_ttl_s == settings.proactive_trending_cache_ttl_s


def test_proactive_trending_via_system_config(monkeypatch):
    _loaded_caches(monkeypatch)
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {
        "proactiveTrendingEnabled": True,
        "proactiveTrendingProbability": 0.42,
        "proactiveTrendingLinkProbability": 0.18,
        "proactiveTrendingCacheTtlS": 7200,
    })
    resolved = runtime_config.resolve_config_sync(agent_id=None)
    assert resolved.proactive_trending_enabled is True
    assert resolved.proactive_trending_probability == 0.42
    assert resolved.proactive_trending_link_probability == 0.18
    assert resolved.proactive_trending_cache_ttl_s == 7200


def test_payload_to_update_data_partial_fields_only():
    payload = ConfigPayload(proactive_trending_enabled=True)
    data = _payload_to_update_data(payload, include_global_only=True)
    assert data == {"proactiveTrendingEnabled": True}
    assert "webSearchEnabled" not in data
    assert "onlineModel" not in data


def test_payload_to_data_proactive_trending_global_only():
    payload = ConfigPayload(
        proactive_trending_enabled=True,
        proactive_trending_probability=0.25,
        proactive_trending_link_probability=0.15,
        proactive_trending_cache_ttl_s=1800,
    )
    global_data = _payload_to_data(payload, include_global_only=True)
    assert global_data["proactiveTrendingEnabled"] is True
    assert global_data["proactiveTrendingProbability"] == 0.25
    agent_data = _payload_to_data(payload)
    assert "proactiveTrendingEnabled" not in agent_data


def test_row_to_payload_proactive_trending_getattr_safe():
    row = SimpleNamespace(
        onlineModel=None,
        remoteProvider=None,
        remoteChatProvider=None,
        remoteSmallProvider=None,
        localChatModel=None,
        localSmallModel=None,
        remoteChatModel=None,
        remoteSmallModel=None,
        proactiveTrendingEnabled=True,
        proactiveTrendingProbability=0.3,
        proactiveTrendingLinkProbability=0.2,
        proactiveTrendingCacheTtlS=3600,
    )
    payload = _row_to_payload(row)
    assert payload["proactive_trending_enabled"] is True
    assert payload["proactive_trending_probability"] == 0.3
    assert payload["proactive_trending_link_probability"] == 0.2
    assert payload["proactive_trending_cache_ttl_s"] == 3600
