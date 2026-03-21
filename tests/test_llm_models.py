from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.services.llm import models


def _reset_caches() -> None:
    models.get_chat_model.cache_clear()
    models.get_summarizer_model.cache_clear()
    models.get_utility_model.cache_clear()
    models.get_embedding_model.cache_clear()


def test_dashscope_chat_and_embedding_provider(monkeypatch):
    _reset_caches()
    monkeypatch.setattr(models.settings, "llm_provider", "ollama")
    monkeypatch.setattr(models.settings, "chat_provider", "dashscope")
    monkeypatch.setattr(models.settings, "embedding_provider", "dashscope")
    monkeypatch.setattr(models.settings, "dashscope_api_key", "test-key")
    monkeypatch.setattr(
        models.settings,
        "dashscope_base_url",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setattr(models.settings, "chat_model", "qwen3.5-plus")
    monkeypatch.setattr(models.settings, "embedding_model", "text-embedding-v3")
    monkeypatch.setattr(models.settings, "embedding_dimensions", 768)

    chat_model = models.get_chat_model()
    embedding_model = models.get_embedding_model()

    assert isinstance(chat_model, ChatOpenAI)
    assert isinstance(embedding_model, OpenAIEmbeddings)


def test_ollama_provider_still_supported(monkeypatch):
    _reset_caches()
    monkeypatch.setattr(models.settings, "llm_provider", "ollama")
    monkeypatch.setattr(models.settings, "chat_provider", "")
    monkeypatch.setattr(models.settings, "embedding_provider", "")
    monkeypatch.setattr(models.settings, "chat_model", "qwen2.5:14b")
    monkeypatch.setattr(models.settings, "embedding_model", "nomic-embed-text")

    chat_model = models.get_chat_model()
    embedding_model = models.get_embedding_model()

    assert isinstance(chat_model, ChatOllama)
    assert isinstance(embedding_model, OllamaEmbeddings)
