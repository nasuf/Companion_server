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
    resilience.reset_breakers_for_testing()
    resilience.reset_slots_for_testing()
    yield
    resilience.reset_profiles_cache_for_testing()
    resilience.reset_breakers_for_testing()
    resilience.reset_slots_for_testing()


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
        result = models._extract_json(truncated, allow_truncated=True)
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
        result = models._extract_json(truncated, allow_truncated=True)
        assert result == {"a": 1, "b": {"x": "ok"}}

    def test_completely_garbled_raises(self):
        """既不是 JSON 也救不了 (无 `{` 或顶层逗号), 抛 ValueError."""
        import pytest
        with pytest.raises(ValueError):
            models._extract_json("just plain text no json here")

    def test_truncated_output_is_rejected_unless_caller_opts_in(self):
        """A partial object must not look like success to write paths."""
        import pytest
        with pytest.raises(models.TruncatedJSONError):
            models._extract_json('{"a": 1, "b": "cut')
        assert models._parse_json('{"a": 1, "b": "cut', allow_truncated=True) == ({"a": 1}, True)
        assert models._parse_json('{"a": 1}') == ({"a": 1}, False)

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

        meta: dict = {}
        result = await models.invoke_json(
            model, "fake-prompt", profile="test_stream_trunc", allow_truncated=True, invocation_metadata=meta,
        )
        assert meta["truncated"] is True
        with pytest.raises(models.TruncatedJSONError):
            await models.invoke_json(model, "fake-prompt", profile="test_stream_trunc")
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


# ── Ollama 兜底的采样参数 ──

class TestOllamaSamplingKwargs:
    """调用方按 OpenAI 习惯传 temperature=0; ChatOllama 把 ainvoke 的额外 kwargs
    平铺进 AsyncClient.chat(**params), 于是兜底每次都 TypeError (生产 2026-10-16:
    地点核验主 provider 失败后, ollama 兜底三次全报
    "AsyncClient.chat() got an unexpected keyword argument 'temperature'")。"""

    @staticmethod
    def _accepted_by_ollama_client(model, kwargs) -> None:
        import inspect

        import ollama
        from langchain_core.messages import HumanMessage

        params = model._chat_params([HumanMessage(content="hi")], **kwargs)
        inspect.signature(ollama.AsyncClient.chat).bind(object(), **params)

    def test_sampling_kwargs_move_onto_the_ollama_model(self):
        base = ChatOllama(model="qwen2.5:7b", base_url="http://127.0.0.1:9", num_ctx=4096)
        copy, rest = models._for_ollama(
            base, {"temperature": 0, "max_tokens": 64, "top_p": None, "format": "json"},
        )
        assert (copy.temperature, copy.num_predict, copy.num_ctx) == (0, 64, 4096)
        assert copy.top_p is None and base.temperature is None
        assert rest == {"format": "json"}
        self._accepted_by_ollama_client(copy, rest)

    def test_unmoved_kwargs_still_break_the_client(self):
        """反向守卫: 证明上面那条断言确实在测参数形状, 而不是恒真。"""
        base = ChatOllama(model="qwen2.5:7b", base_url="http://127.0.0.1:9")
        with pytest.raises(TypeError, match="temperature"):
            self._accepted_by_ollama_client(base, {"temperature": 0})

    def test_non_ollama_model_keeps_kwargs_as_request_params(self):
        model = object()
        kwargs = {"temperature": 0}
        same, rest = models._for_ollama(model, kwargs)
        assert same is model and rest == kwargs and rest is not kwargs

    @staticmethod
    def _failing_primary(stream: bool = False):
        class _Primary(ChatOpenAI):
            async def ainvoke(self, _messages, **_kwargs):
                raise ConnectionError("primary down")

            def astream(self, _messages, **_kwargs):
                async def _g():
                    raise ConnectionError("primary down")
                    yield  # pragma: no cover
                return _g()

        return _Primary.model_construct(model_name="qwen3.5-plus")

    def _recording_fallback(self, seen: dict):
        from langchain_core.messages import AIMessage, AIMessageChunk

        check = self._accepted_by_ollama_client

        class _Fallback(ChatOllama):
            async def ainvoke(self, _messages, **kwargs):
                check(self, kwargs)
                seen.update(temperature=self.temperature, kwargs=kwargs)
                return AIMessage(content='{"ok": 1}')

            def astream(self, _messages, **kwargs):
                check(self, kwargs)
                seen.update(temperature=self.temperature, kwargs=kwargs)

                async def _g():
                    yield AIMessageChunk(content='{"ok": 1}')
                return _g()

        return _Fallback.model_construct(model="qwen2.5:7b", base_url="http://fake")

    @pytest.mark.parametrize("stream", [False, True])
    async def test_fallback_runs_with_temperature(self, monkeypatch, stream):
        resilience.set_profiles_for_testing({
            "test_fallback": resilience.CallProfile(
                timeout_s=2.0, max_retries=0, retry_backoff_s=(),
                first_chunk_timeout_s=1.0, idle_timeout_s=1.0,
                allow_ollama_fallback=True, stream_mode=stream,
            ),
        })
        seen: dict = {}
        monkeypatch.setattr(models, "get_fallback_chat_model", lambda: self._recording_fallback(seen))
        result = await models.invoke_json(
            self._failing_primary(), "prompt", temperature=0, profile="test_fallback",
        )
        assert result == {"ok": 1}
        assert seen["temperature"] == 0
        assert "temperature" not in seen["kwargs"] and seen["kwargs"]["format"] == "json"



# ── Q01.06: unknown usage, structured output with one repair ──

class _Unary:
    """Model stub returning queued contents via ainvoke, optionally with usage."""

    @staticmethod
    def build(contents, usage=None):
        from langchain_core.messages import AIMessage
        from langchain_ollama import ChatOllama

        calls = []

        class _M(ChatOllama):
            async def ainvoke(self, messages, **_kwargs):
                calls.append(messages)
                return AIMessage(content=contents[len(calls) - 1], usage_metadata=usage)

        return _M.model_construct(model="stub", base_url="http://fake"), calls


@pytest.fixture
def unary_profile():
    resilience.set_profiles_for_testing({
        "test_q0106": resilience.CallProfile(
            timeout_s=2.0, max_retries=0, retry_backoff_s=(), allow_ollama_fallback=False,
        ),
    })
    yield "test_q0106"
    resilience.reset_profiles_cache_for_testing()


from contextlib import contextmanager


@contextmanager
def _usage():
    from app.services.llm import usage_tracker

    token = usage_tracker.start_session()
    try:
        yield usage_tracker._current.get()
    finally:
        usage_tracker._current.reset(token)


class _Answer(__import__("pydantic").BaseModel):
    kind: str
    count: int


@pytest.mark.asyncio
async def test_missing_usage_is_unmetered_not_zero(unary_profile):
    model, _ = _Unary.build(['{"a": 1}'])
    with _usage() as summary:
        await models.invoke_json(model, "p", profile=unary_profile)
    bucket = summary["tokens_by_model"]["ollama/stub"]
    assert bucket["unmetered_calls"] == 1 and bucket["input"] == 0
    assert summary["call_count"] == 1 and summary["input_tokens"] == 0

    metered, _ = _Unary.build(['{"a": 1}'], usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
    with _usage() as summary:
        await models.invoke_json(metered, "p", profile=unary_profile)
    assert summary["input_tokens"] == 7 and "unmetered_calls" not in summary["tokens_by_model"]["ollama/stub"]


def test_streamed_usage_is_not_counted_twice():
    from langchain_core.messages import AIMessage

    with _usage() as summary:
        assert models._record_usage_from_response(None, AIMessage(content="x"), already_recorded=True) == {}
    assert summary["call_count"] == 0


@pytest.mark.asyncio
async def test_structured_output_validates_and_repairs_at_most_once(unary_profile, monkeypatch):
    from app.services.prompting import store
    from app.services.prompting.defaults import LLM_FORMAT_REPAIR_PROMPT

    async def repair_template(key):
        assert key == "llm.format_repair"
        return LLM_FORMAT_REPAIR_PROMPT

    monkeypatch.setattr(store, "get_prompt_text_or_default", repair_template)
    model, calls = _Unary.build(['{"kind": "a", "count": 2}'])
    assert (await models.invoke_structured(model, "p", _Answer, profile=unary_profile)).count == 2
    assert len(calls) == 1

    meta: dict = {}
    model, calls = _Unary.build(['{"kind": "a"', '{"kind": "a", "count": 3}'])
    result = await models.invoke_structured(model, "p", _Answer, profile=unary_profile, invocation_metadata=meta)
    assert result.count == 3 and meta["format_repair"] == "attempted" and len(calls) == 2
    repair = calls[1]
    assert [type(m).__name__ for m in repair] == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert '"count"' in repair[2].content  # schema shown to the model

    model, calls = _Unary.build(['{"kind": "a"}', '{"kind": "a", "count": "many"}'])
    with pytest.raises(models.StructuredOutputError):
        await models.invoke_structured(model, "p", _Answer, profile=unary_profile, invocation_metadata=meta)
    assert len(calls) == 2 and meta["format_repair"] == "failed"


@pytest.mark.asyncio
async def test_contradiction_analysis_failure_is_not_a_new_fact(monkeypatch):
    from app.services.memory.interaction import contradiction

    async def broken(*_a, **_k):
        raise models.StructuredOutputError("ContradictionAnalysis")

    async def template(_key):
        return "{user_reply}"

    monkeypatch.setattr(contradiction, "invoke_structured", broken)
    monkeypatch.setattr(contradiction, "get_prompt_text", template)
    result = await contradiction.analyze_contradiction_response("嗯", {"old_content": "x"})
    assert result == {"change_type": None, "reason": "analysis_failed"}
    from app.services.memory.interaction.contradiction_write import replacement_text
    from app.services.memory.manual_write import MemoryWriteError
    with pytest.raises(MemoryWriteError):
        replacement_text(result)


@pytest.mark.asyncio
async def test_contradiction_analysis_returns_validated_verdict(monkeypatch):
    from app.services.memory.interaction import contradiction

    async def structured(_model, _prompt, schema, **_k):
        return schema.model_validate({"change_type": "变化", "new_memory": "住杭州", "extra": "x"})

    async def template(_key):
        return "{user_reply}"

    monkeypatch.setattr(contradiction, "invoke_structured", structured)
    monkeypatch.setattr(contradiction, "get_prompt_text", template)
    result = await contradiction.analyze_contradiction_response("搬家了", {"old_content": "住苏州"})
    assert result["change_type"] == "变化" and result["new_memory"] == "住杭州" and "extra" not in result


@pytest.mark.asyncio
async def test_memory_extraction_marks_salvaged_output(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from app.services.memory.recording import extraction

    async def salvaged(_model, _prompt, **kwargs):
        assert kwargs["allow_truncated"] is True
        kwargs["invocation_metadata"].update(actual_model="ark/m", truncated=True)
        return {"memories": []}

    monkeypatch.setattr(extraction, "invoke_json", salvaged)
    monkeypatch.setattr(extraction, "get_chat_model", MagicMock())
    monkeypatch.setattr(extraction, "get_prompt_text", AsyncMock(
        return_value="{new_conversation} {context_conversation} {current_time} {taxonomy_list}"))
    result = await extraction.extract_memories("用户：你好")
    assert result["_provenance"]["truncated"] is True and "_extraction_error" not in result
