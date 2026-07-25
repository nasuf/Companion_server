"""Web search main-reply integration tests.

Covers:
- runtime config: web_search_enabled resolve chain + admin payload mapping
  (global-only, agent endpoints must not carry the column)
- ark_web_search service: payload shape, output extraction, usage recording,
  fail-open on every error class
- reply_generate branch: gate conditions + short-circuit before the
  streaming path, fallback to streaming when search yields nothing
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.admin.runtime_config import ConfigPayload, _payload_to_data, _row_to_payload
from app.config import settings
from app.services import runtime_config
from app.services.chat import reply_generate
from app.services.llm import ark_web_search


@pytest.fixture()
def loaded_caches(monkeypatch):
    monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
    monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {})
    return runtime_config


# ─── runtime config chain ─────────────────────────────────────────────────


def test_web_search_defaults_to_env_false(loaded_caches):
    resolved = runtime_config.resolve_config_sync(agent_id=None)
    assert resolved.web_search_enabled is settings.web_search_enabled is False


def test_web_search_enabled_via_system_config(monkeypatch, loaded_caches):
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {"webSearchEnabled": True})
    assert runtime_config.resolve_config_sync(agent_id=None).web_search_enabled is True


def test_payload_to_data_web_search_global_only():
    payload = ConfigPayload(web_search_enabled=True)
    global_data = _payload_to_data(payload, include_global_only=True)
    assert global_data["webSearchEnabled"] is True
    agent_data = _payload_to_data(payload)
    assert "webSearchEnabled" not in agent_data


def test_row_to_payload_web_search_getattr_safe():
    agent_row = SimpleNamespace(
        onlineModel=None, remoteProvider=None, remoteChatProvider=None,
        remoteSmallProvider=None, localChatModel=None, localSmallModel=None,
        remoteChatModel=None, remoteSmallModel=None,
    )
    assert _row_to_payload(agent_row)["web_search_enabled"] is None
    assert _row_to_payload(None)["web_search_enabled"] is None


# ─── ark_web_search service ───────────────────────────────────────────────


def test_to_responses_input_filters_bad_roles():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "dropped"},
        {"role": "user", "content": None},
    ]
    out = ark_web_search._to_responses_input(messages)
    assert [m["role"] for m in out] == ["system", "assistant", "user"]


def test_extract_output_text_counts_search_calls():
    payload = {
        "output": [
            {"type": "web_search_call", "status": "completed"},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "北京今天31℃"},
                ],
            },
        ],
    }
    text, calls = ark_web_search._extract_output_text(payload)
    assert text == "北京今天31℃"
    assert calls == 1


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "err"

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.captured: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, endpoint, headers=None, json=None):
        self.captured["endpoint"] = endpoint
        self.captured["json"] = json
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_generate_with_web_search_success(monkeypatch):
    payload = {
        "output": [
            {"type": "web_search_call"},
            {"type": "message", "content": [{"type": "output_text", "text": "回复"}]},
        ],
        "usage": {
            "input_tokens": 6000, "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 500},
        },
    }
    client = _FakeClient(response=_FakeResponse(200, payload))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "test-key")
    recorded: dict = {}
    monkeypatch.setattr(
        ark_web_search.usage_tracker, "record",
        lambda model, i, o, cached_input_tokens=0: recorded.update(
            {"model": model, "in": i, "out": o, "cached": cached_input_tokens},
        ),
    )

    text = await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "明天天气"}], model="doubao-seed-character-260628",
    )

    assert text == "回复"
    body = client.captured["json"]
    assert body["model"] == "doubao-seed-character-260628"
    assert body["tools"] == [{"type": "web_search"}]
    assert body["stream"] is False
    assert client.captured["endpoint"].endswith("/responses")
    assert recorded == {
        "model": "ark/doubao-seed-character-260628",
        "in": 6000, "out": 100, "cached": 500,
    }


@pytest.mark.asyncio
async def test_generate_with_web_search_records_trace_step(monkeypatch):
    """Raw HTTP calls bypass langchain, so the step must be recorded manually
    or the main reply disappears from the trace tree."""
    payload = {
        "output": [
            {"type": "web_search_call"},
            {"type": "message", "content": [{"type": "output_text", "text": "北京31℃"}]},
        ],
        "usage": {
            "input_tokens": 4830, "output_tokens": 29,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }
    client = _FakeClient(response=_FakeResponse(200, payload))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "test-key")
    monkeypatch.setattr(ark_web_search.usage_tracker, "record", lambda *a, **k: None)
    recorded: dict = {}
    monkeypatch.setattr(
        ark_web_search, "record_manual_llm_run",
        lambda **kwargs: recorded.update(kwargs),
    )

    messages = [
        {"role": "system", "content": "人设 prompt"},
        {"role": "user", "content": "北京天气"},
    ]
    await ark_web_search.generate_with_web_search(messages, model="doubao-x")

    assert recorded["model_name"] == "doubao-x"
    assert recorded["provider"] == "ark"
    assert recorded["messages"] == messages
    assert recorded["output_text"] == "北京31℃"
    assert recorded["input_tokens"] == 4830
    assert recorded["output_tokens"] == 29
    assert recorded["cached_input_tokens"] == 0
    assert recorded["metadata"] == {"web_search_calls": 1}
    assert recorded["ended_at"] >= recorded["started_at"]


@pytest.mark.asyncio
async def test_generate_with_web_search_fail_open(monkeypatch):
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "test-key")

    # HTTP error status → None
    client = _FakeClient(response=_FakeResponse(404))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="m",
    ) is None

    # Transport exception → None
    client = _FakeClient(error=RuntimeError("timeout"))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="m",
    ) is None

    # Empty output → None
    client = _FakeClient(response=_FakeResponse(200, {"output": []}))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="m",
    ) is None


@pytest.mark.asyncio
async def test_generate_with_web_search_requires_key_and_model(monkeypatch):
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "")
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="m",
    ) is None
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "k")
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="",
    ) is None


# ─── reply_generate branch ────────────────────────────────────────────────


def _resolved(web_search=True, online=True, provider="ark"):
    return runtime_config.ResolvedConfig(
        online_model=online,
        remote_provider=provider,
        remote_chat_provider=provider,
        remote_small_provider=provider,
        local_chat_model="qwen2.5:14b",
        local_small_model="qwen2.5:7b",
        remote_chat_model="doubao-seed-character-260628",
        remote_small_model="doubao-seed-character-260628",
        vision_model="v",
        asr_model="a",
        web_search_enabled=web_search,
    )


@pytest.mark.asyncio
async def test_try_web_search_reply_gates(monkeypatch):
    calls: list = []

    async def fake_generate(messages, *, model):
        calls.append(model)
        return "搜到了"

    monkeypatch.setattr(
        "app.services.llm.ark_web_search.generate_with_web_search", fake_generate,
    )

    # Enabled + online + ark → generate called
    monkeypatch.setattr(
        "app.services.runtime_config.resolve_for_current", lambda: _resolved(),
    )
    assert await reply_generate._try_web_search_reply([{"role": "user", "content": "x"}]) == "搜到了"
    assert calls == ["doubao-seed-character-260628"]

    # Disabled → skipped
    monkeypatch.setattr(
        "app.services.runtime_config.resolve_for_current",
        lambda: _resolved(web_search=False),
    )
    assert await reply_generate._try_web_search_reply([]) is None

    # Non-ark chat provider → skipped
    monkeypatch.setattr(
        "app.services.runtime_config.resolve_for_current",
        lambda: _resolved(provider="deepseek"),
    )
    assert await reply_generate._try_web_search_reply([]) is None

    # Offline (ollama) mode → skipped
    monkeypatch.setattr(
        "app.services.runtime_config.resolve_for_current",
        lambda: _resolved(online=False),
    )
    assert await reply_generate._try_web_search_reply([]) is None
    assert calls == ["doubao-seed-character-260628"]  # only the first case called


@pytest.mark.asyncio
async def test_run_main_llm_short_circuits_on_web_search(monkeypatch):
    monkeypatch.setattr(
        reply_generate, "_try_web_search_reply",
        AsyncMock(return_value="联网回复[EMO:高兴/60]"),
    )

    def _boom():
        raise AssertionError("streaming path must not run when web search hits")

    monkeypatch.setattr(reply_generate, "get_chat_model", _boom)

    text, is_fallback = await reply_generate._run_main_llm(
        [{"role": "user", "content": "明天天气"}],
    )
    assert text == "联网回复[EMO:高兴/60]"
    assert is_fallback is False


@pytest.mark.asyncio
async def test_run_main_llm_falls_back_to_stream_when_search_empty(monkeypatch):
    monkeypatch.setattr(
        reply_generate, "_try_web_search_reply", AsyncMock(return_value=None),
    )
    fake_model = SimpleNamespace(astream=lambda msgs: None)
    monkeypatch.setattr(reply_generate, "get_chat_model", lambda: fake_model)
    monkeypatch.setattr(reply_generate, "get_fallback_chat_model", lambda: fake_model)
    monkeypatch.setattr(reply_generate, "provider_name", lambda m: "ark")
    monkeypatch.setattr(
        "app.services.llm.models._resolve_usage_model_key", lambda m: "ark/x",
    )

    async def fake_collect(*args, **kwargs):
        return "流式回复"

    monkeypatch.setattr(reply_generate, "collect_stream", fake_collect)

    text, is_fallback = await reply_generate._run_main_llm(
        [{"role": "user", "content": "hi"}],
    )
    assert text == "流式回复"
    assert is_fallback is False
