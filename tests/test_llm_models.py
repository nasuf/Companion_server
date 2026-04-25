from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.services.llm import models


def _reset_caches() -> None:
    models.get_chat_model.cache_clear()
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


# ── _extract_json + 截断救援 ──

class TestExtractJson:
    def test_plain_json(self):
        assert models._extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        text = '```json\n{"a": 1, "b": "x"}\n```'
        assert models._extract_json(text) == {"a": 1, "b": "x"}

    def test_object_in_prose(self):
        text = '一段话 {"k": [1,2,3]} 后面还有'
        assert models._extract_json(text) == {"k": [1, 2, 3]}

    def test_truncated_inside_string_recovers_prior_keys(self):
        """LLM 满 max_tokens 在字符串中间停: 救出之前完整的顶层字段."""
        truncated = (
            '```json\n{"identity": {"name": "x", "age": 24}, '
            '"appearance": {"height": "170cm"}, '
            '"career": {"title": "占卜师", "duties": "在专属塔罗占卜室接待咨询者，')
        result = models._extract_json(truncated)
        assert "identity" in result
        assert result["identity"]["name"] == "x"
        assert "appearance" in result
        assert result["appearance"]["height"] == "170cm"
        # career 在中途被截断, 不保留
        assert "career" not in result

    def test_truncated_inside_nested_object(self):
        """嵌套对象内部截断: 整个顶层 key 丢, 之前的保住."""
        truncated = (
            '{"a": 1, "b": {"x": "ok"}, "c": {"d": "incomplete')
        result = models._extract_json(truncated)
        assert result == {"a": 1, "b": {"x": "ok"}}

    def test_completely_garbled_raises(self):
        """既不是 JSON 也救不了 (无 `{` 或顶层逗号), 抛 ValueError."""
        import pytest
        with pytest.raises(ValueError):
            models._extract_json("just plain text no json here")

    def test_truncated_only_first_field_no_comma_yet(self):
        """第一个字段都没写完且没出现顶层逗号 — 没东西可救, 抛 ValueError."""
        import pytest
        truncated = '{"identity": {"name": "x", "incomplete'
        with pytest.raises(ValueError):
            models._extract_json(truncated)
