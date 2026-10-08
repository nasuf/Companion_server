"""Activity routing reuses configured providers without coupling to chat balance."""

from unittest.mock import Mock

from app.config import settings
from app.services.offline import llm


def test_activity_model_route_is_explicit_and_content_addressed(monkeypatch):
    llm._model.cache_clear()
    build = Mock(side_effect=lambda provider, name: (provider, name))
    monkeypatch.setattr(llm, "build_chat_model", build)
    monkeypatch.setattr(settings, "offline_model_provider", "dashscope")
    monkeypatch.setattr(settings, "offline_chat_model", "qwen3.5-plus")
    monkeypatch.setattr(settings, "offline_small_model", "qwen3.5-flash")
    assert llm.get_offline_chat_model() == ("dashscope", "qwen3.5-plus")
    assert llm.get_offline_chat_model() == ("dashscope", "qwen3.5-plus")
    assert llm.get_offline_small_model() == ("dashscope", "qwen3.5-flash")
    assert build.call_count == 2
    monkeypatch.setattr(settings, "offline_chat_model", "another-model")
    assert llm.get_offline_chat_model() == ("dashscope", "another-model")
    llm._model.cache_clear()


def test_runtime_route_remains_an_explicit_option(monkeypatch):
    monkeypatch.setattr(settings, "offline_model_provider", "runtime")
    monkeypatch.setattr(llm, "get_chat_model", lambda: "agent-chat-model")
    monkeypatch.setattr(llm, "get_utility_model", lambda: "agent-small-model")
    assert llm.get_offline_chat_model() == "agent-chat-model"
    assert llm.get_offline_small_model() == "agent-small-model"
