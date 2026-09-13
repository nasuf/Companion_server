from app.services import runtime_config
from app.services.proactive.trending_gate import (
    TRENDING_ELIGIBLE_TRIGGER_TYPES,
    is_trending_eligible_trigger,
    should_attach_trending,
    should_attach_trending_link_card,
)


def _loaded_caches(monkeypatch):
    monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
    monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {})


def test_trending_eligible_trigger_types():
    assert TRENDING_ELIGIBLE_TRIGGER_TYPES == frozenset({
        "silence_wakeup",
        "scheduled_scene",
        "special_date",
    })
    assert is_trending_eligible_trigger("silence_wakeup")
    assert is_trending_eligible_trigger("scheduled_scene")
    assert is_trending_eligible_trigger("special_date")
    assert not is_trending_eligible_trigger("memory_proactive")


def test_should_attach_trending_disabled_by_default(monkeypatch):
    _loaded_caches(monkeypatch)
    assert should_attach_trending("silence_wakeup", random_value=0.0) is False


def test_should_attach_trending_probability(monkeypatch):
    _loaded_caches(monkeypatch)
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {
        "proactiveTrendingEnabled": True,
        "proactiveTrendingProbability": 0.30,
    })
    assert should_attach_trending("memory_proactive", random_value=0.0) is False
    assert should_attach_trending("silence_wakeup", random_value=0.29) is True
    assert should_attach_trending("silence_wakeup", random_value=0.31) is False


def test_should_attach_trending_link_card_conditional(monkeypatch):
    _loaded_caches(monkeypatch)
    monkeypatch.setattr(
        "app.services.proactive.trending_gate.settings.proactive_link_recommendation_enabled",
        True,
    )
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {
        "proactiveTrendingLinkProbability": 0.30,
    })
    assert should_attach_trending_link_card(trending_attached=False, random_value=0.0) is False
    assert should_attach_trending_link_card(trending_attached=True, random_value=0.29) is True
    assert should_attach_trending_link_card(trending_attached=True, random_value=0.31) is False
