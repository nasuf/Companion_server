import pytest
from types import SimpleNamespace
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.services.llm import models, resilience
from app.services.llm.providers import build_chat_model, public_provider_options


def _reset_caches() -> None:
    models.get_chat_model.cache_clear()
    models.get_utility_model.cache_clear()
    models.get_embedding_model.cache_clear()
    models.get_fallback_chat_model.cache_clear()


@pytest.fixture(autouse=True)
def _reset_profiles_cache():
    """Defense-in-depth: 即便测试直接动了 _PROFILES_CACHE, 每个测试结束也复位."""
    yield
    resilience.reset_profiles_cache_for_testing()


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


def test_deepseek_chat_provider_uses_direct_openai_compatible_endpoint(monkeypatch):
    _reset_caches()
    monkeypatch.setattr(models.settings, "llm_provider", "ollama")
    monkeypatch.setattr(models.settings, "chat_provider", "deepseek")
    monkeypatch.setattr(models.settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(models.settings, "deepseek_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(models.settings, "chat_model", "deepseek-v4-pro")

    chat_model = models.get_chat_model()

    assert isinstance(chat_model, ChatOpenAI)
    assert resilience.provider_name(chat_model) == "deepseek"
    assert models._resolve_usage_model_key(chat_model) == "deepseek/deepseek-v4-pro"


@pytest.mark.parametrize(
    ("provider", "key_setting", "base_setting", "base_url", "model_name"),
    [
        (
            "ark", "ark_api_key", "ark_base_url", "https://ark.test/api/v3",
            "doubao-seed-character-260628",
        ),
        ("minimax", "minimax_api_key", "minimax_base_url", "https://minimax.test/v1", "M2-her"),
    ],
)
def test_openai_compatible_character_providers(
    monkeypatch, provider, key_setting, base_setting, base_url, model_name,
):
    monkeypatch.setattr(models.settings, key_setting, "test-key")
    monkeypatch.setattr(models.settings, base_setting, base_url)

    chat_model = build_chat_model(provider, model_name)

    assert isinstance(chat_model, ChatOpenAI)
    assert resilience.provider_name(chat_model) == provider
    assert str(chat_model.openai_api_base) == base_url
    if provider == "minimax":
        assert chat_model.temperature == 1.0
        assert chat_model.top_p == 0.95
        assert chat_model.max_tokens == 2048


def test_qwen_character_uses_workspace_endpoint_without_thinking_extension(monkeypatch):
    monkeypatch.setattr(models.settings, "dashscope_api_key", "test-key")
    monkeypatch.setattr(
        models.settings,
        "dashscope_character_base_url",
        "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setattr(models.settings, "dashscope_enable_thinking", True)

    chat_model = build_chat_model("dashscope", "qwen-plus-character")

    assert isinstance(chat_model, ChatOpenAI)
    assert str(chat_model.openai_api_base).startswith("https://workspace.")
    assert chat_model.model_kwargs == {}
    assert chat_model.max_tokens == 4096


def test_provider_options_are_dynamic_and_secret_safe(monkeypatch):
    monkeypatch.setattr(models.settings, "minimax_api_key", "super-secret")

    options = public_provider_options()
    minimax = next(item for item in options if item["id"] == "minimax")
    ark = next(item for item in options if item["id"] == "ark")

    assert {item["id"] for item in options} == {
        "ollama", "dashscope", "deepseek", "ark", "minimax",
    }
    assert minimax["configured"] is True
    assert minimax["credential_env"] == "MINIMAX_API_KEY"
    assert ark["preferred_chat_models"] == ["doubao-seed-character-260628"]
    assert "super-secret" not in repr(options)


def test_runtime_config_can_split_chat_and_utility_providers(monkeypatch):
    monkeypatch.setattr(models.settings, "chat_provider", "")
    monkeypatch.setattr(models.settings, "utility_provider", "")
    monkeypatch.setattr(models.settings, "llm_provider", "")
    monkeypatch.setattr(models, "_resolved", lambda: SimpleNamespace(
        online_model=True,
        remote_chat_provider="minimax",
        remote_small_provider="dashscope",
    ))

    assert models._provider_for("chat") == "minimax"
    assert models._provider_for("utility") == "dashscope"


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


# ── invoke_json with stream_mode (background profile) ──

class TestInvokeJsonStreaming:
    """character.generation 走 streaming 路径的端到端集成测试.

    验证 stream_mode=True 的 profile 让 invoke_json 内部走 astream + 累积 +
    _extract_json, API 输出与 unary 等价. 见 plan-1-bug-2-squishy-elephant.md.
    """

    @staticmethod
    def _fake_chunks(parts):
        from langchain_core.messages import AIMessageChunk

        async def _gen():
            for p in parts:
                yield AIMessageChunk(content=p)
        return _gen

    @staticmethod
    def _fake_model_with_astream(astream_factory):
        """构造一个 mock LangChain 模型, .astream(messages, **kwargs) → factory()."""
        from langchain_ollama import ChatOllama

        class _MockModel(ChatOllama):
            def astream(self, _messages, **_kwargs):
                return astream_factory()

        return _MockModel.model_construct(model="mock", base_url="http://fake")

    import pytest as _pytest

    @_pytest.mark.asyncio
    async def test_stream_mode_accumulates_chunks_and_parses_json(self):
        """stream_mode=True 下应累积所有 chunk → _extract_json 解析."""
        # 用公共 helper 注入 profile, 避免直接动私有 _PROFILES_CACHE
        resilience.set_profiles_for_testing({
            "test_stream": resilience.CallProfile(
                timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                first_chunk_timeout_s=1.0, idle_timeout_s=1.0,
                allow_ollama_fallback=False, stream_mode=True,
            ),
        })

        astream_factory = self._fake_chunks(['{"name": ', '"小雨", "age"', ': 25}'])
        model = self._fake_model_with_astream(astream_factory)

        result = await models.invoke_json(model, "fake-prompt", profile="test_stream")
        assert result == {"name": "小雨", "age": 25}

    @_pytest.mark.asyncio
    async def test_stream_mode_extracts_json_from_truncated_output(self):
        """stream 累积后被 _salvage_truncated_json_object 抢救."""
        resilience.set_profiles_for_testing({
            "test_stream_trunc": resilience.CallProfile(
                timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                first_chunk_timeout_s=1.0, idle_timeout_s=1.0,
                allow_ollama_fallback=False, stream_mode=True,
            ),
        })

        # 模拟 LLM 在中途断流 (常见于满 max_tokens)
        astream_factory = self._fake_chunks([
            '{"identity": {"name": "x"}, ',
            '"life_events": {"travel": ["incomplet',
        ])
        model = self._fake_model_with_astream(astream_factory)

        result = await models.invoke_json(model, "fake-prompt", profile="test_stream_trunc")
        # 截断救援保住 identity, life_events 整个 key 丢失
        assert result["identity"]["name"] == "x"
        assert "life_events" not in result

    @_pytest.mark.asyncio
    async def test_stream_mode_off_uses_unary_path(self):
        """profile.stream_mode=False (默认) 走 ainvoke 而非 astream."""
        from langchain_core.messages import AIMessage
        from langchain_ollama import ChatOllama

        resilience.set_profiles_for_testing({
            "test_unary": resilience.CallProfile(
                timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                stream_mode=False,
            ),
        })

        astream_called = {"flag": False}
        ainvoke_called = {"flag": False}

        class _MockModel(ChatOllama):
            def astream(self, _messages, **_kwargs):
                astream_called["flag"] = True
                async def _g():
                    yield AIMessage(content="should-not-be-used")
                return _g()

            async def ainvoke(self, _messages, **_kwargs):
                ainvoke_called["flag"] = True
                return AIMessage(content='{"x": 1}')

        model = _MockModel.model_construct(model="mock", base_url="http://fake")
        result = await models.invoke_json(model, "p", profile="test_unary")
        assert result == {"x": 1}
        assert ainvoke_called["flag"] is True
        assert astream_called["flag"] is False


# ── _provider_and_profile override 形参 ──
# 防 silent failure regression — 生产 trace 019e0004 (2026-05-07): extraction 误传
# CallProfile 对象给 invoke_json profile=, 嵌套 get_profile(get_profile("...")) →
# KeyError(CallProfile) → silent. 现在 _provider_and_profile 接受 str | CallProfile
# | None, 错误类型直接 TypeError 让调用方一眼看到, 不再静默吃掉.

class TestProviderAndProfile:
    def test_string_override_resolves_via_get_profile(self):
        from unittest.mock import MagicMock
        provider, profile = models._provider_and_profile(MagicMock(), "chat_extract")
        assert profile is resilience.get_profile("chat_extract")

    def test_callprofile_override_passed_through(self):
        """传 CallProfile 对象直接用 (动态构造 / 测试注入), 不再被嵌套 get_profile 撞死."""
        from unittest.mock import MagicMock
        from app.services.llm.resilience import CallProfile
        custom = CallProfile(timeout_s=99.0, max_retries=0, retry_backoff_s=())
        _provider, profile = models._provider_and_profile(MagicMock(), custom)
        assert profile is custom

    def test_none_override_uses_default(self):
        """None override 走默认 (utility_fast for non-chat-model)."""
        from unittest.mock import MagicMock
        _provider, profile = models._provider_and_profile(MagicMock(), None)
        assert profile is resilience.get_profile("utility_fast")

    def test_wrong_type_raises_typeerror_not_keyerror(self):
        """关键: 错误类型 → TypeError loud failure, 不再 KeyError silent (019e0004 regression)."""
        from unittest.mock import MagicMock
        with pytest.raises(TypeError, match="profile override must be"):
            models._provider_and_profile(MagicMock(), 12345)  # type: ignore[arg-type]
