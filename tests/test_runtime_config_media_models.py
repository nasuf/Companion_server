"""Vision / ASR model runtime-config tests (admin "系统设置 → 模型配置").

Covers:
- resolve chain: SystemConfig override → env fallback (vision_model / asr_model)
- _payload_to_data: media fields only written for the global SystemConfig path
  (AgentConfigOverride has no such columns)
- _row_to_payload: agent rows without the new attrs stay safe (getattr)
- consumers: fun_asr.transcribe_audio / vision._call_doubao_vision pick up the
  admin override instead of env
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.admin.runtime_config import (
    ConfigPayload, _payload_to_data, _row_to_payload,
)
from app.services import runtime_config
from app.config import settings


@pytest.fixture()
def loaded_caches(monkeypatch):
    """Pretend load_caches already ran; tests mutate _GLOBAL_CACHE directly."""
    monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
    monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {})
    return runtime_config


# ─── resolve chain ────────────────────────────────────────────────────────


def test_resolve_falls_back_to_env_when_unset(loaded_caches):
    resolved = runtime_config.resolve_config_sync(agent_id=None)
    assert resolved.vision_model == settings.doubao_vision_model
    assert resolved.asr_model == settings.dashscope_asr_model
    assert resolved.tts_model == settings.dashscope_tts_model
    assert resolved.tts_output_probability == settings.tts_output_probability


def test_resolve_prefers_system_config_override(monkeypatch, loaded_caches):
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {
        "visionModel": "doubao-vision-next",
        "asrModel": "fun-asr-next",
        "ttsModel": "qwen-tts-next",
        "ttsOutputProbability": 73,
    })
    resolved = runtime_config.resolve_config_sync(agent_id=None)
    assert resolved.vision_model == "doubao-vision-next"
    assert resolved.asr_model == "fun-asr-next"
    assert resolved.tts_model == "qwen-tts-next"
    assert resolved.tts_output_probability == 73


@pytest.mark.asyncio
async def test_effective_getters_use_resolve_chain(monkeypatch, loaded_caches):
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {
        "visionModel": "vision-x", "asrModel": "asr-x",
    })
    assert await runtime_config.get_effective_vision_model() == "vision-x"
    assert await runtime_config.get_effective_asr_model() == "asr-x"


def test_row_to_dict_reads_media_columns_from_system_row():
    sys_row = SimpleNamespace(
        onlineModel=True, remoteProvider=None, remoteChatProvider=None,
        remoteSmallProvider=None, localChatModel=None, localSmallModel=None,
        remoteChatModel=None, remoteSmallModel=None,
        visionModel="v-1", asrModel="a-1",
        ttsModel="t-1", ttsOutputProbability=42,
    )
    out = runtime_config._row_to_dict(sys_row)
    assert out["visionModel"] == "v-1"
    assert out["asrModel"] == "a-1"
    assert out["ttsModel"] == "t-1"
    assert out["ttsOutputProbability"] == 42
    # Agent override rows lack the attrs entirely → keys absent, not None.
    agent_row = SimpleNamespace(onlineModel=None, remoteChatModel="m")
    agent_out = runtime_config._row_to_dict(agent_row)
    assert "visionModel" not in agent_out
    assert "asrModel" not in agent_out


# ─── admin API payload mapping ────────────────────────────────────────────


def test_payload_to_data_includes_media_only_for_global_path():
    payload = ConfigPayload(
        vision_model=" doubao-v ",
        asr_model="fun-a",
        tts_model="qwen-tts",
        tts_output_probability=65,
    )
    global_data = _payload_to_data(payload, include_global_only=True)
    assert global_data["visionModel"] == "doubao-v"  # stripped
    assert global_data["asrModel"] == "fun-a"
    assert global_data["ttsModel"] == "qwen-tts"
    assert global_data["ttsOutputProbability"] == 65
    # Agent path must not carry the columns at all (prisma unknown column).
    agent_data = _payload_to_data(payload)
    assert "visionModel" not in agent_data
    assert "asrModel" not in agent_data
    assert "ttsModel" not in agent_data
    assert "ttsOutputProbability" not in agent_data


def test_payload_to_data_empty_string_clears_override():
    payload = ConfigPayload(vision_model="  ", asr_model="")
    data = _payload_to_data(payload, include_global_only=True)
    assert data["visionModel"] is None
    assert data["asrModel"] is None


def test_row_to_payload_handles_rows_without_media_attrs():
    agent_row = SimpleNamespace(
        onlineModel=None, remoteProvider=None, remoteChatProvider=None,
        remoteSmallProvider=None, localChatModel=None, localSmallModel=None,
        remoteChatModel="qwen-plus", remoteSmallModel=None,
    )
    out = _row_to_payload(agent_row)
    assert out["vision_model"] is None
    assert out["asr_model"] is None
    assert out["remote_chat_model"] == "qwen-plus"


def test_row_to_payload_none_row_contains_media_keys():
    out = _row_to_payload(None)
    assert out["vision_model"] is None
    assert out["asr_model"] is None


@pytest.mark.asyncio
async def test_options_group_multimodal_models_by_kind_and_provider(monkeypatch):
    from app.api.admin import runtime_config as api_rc

    rows = [
        SimpleNamespace(
            identifier="doubao-chat", provider="ark", modelKind="llm",
        ),
        SimpleNamespace(
            identifier="doubao-vision", provider="ark", modelKind="vision",
        ),
        SimpleNamespace(
            identifier="fun-asr", provider="dashscope", modelKind="asr",
        ),
        SimpleNamespace(
            identifier="qwen-tts", provider="dashscope", modelKind="tts",
        ),
    ]
    monkeypatch.setattr(
        api_rc.db,
        "modelregistry",
        SimpleNamespace(find_many=AsyncMock(return_value=rows)),
        raising=False,
    )
    monkeypatch.setattr(api_rc.settings, "ark_api_key", "ark-key")
    monkeypatch.setattr(api_rc.settings, "dashscope_api_key", "dash-key")
    monkeypatch.setattr(api_rc.settings, "dashscope_tts_api_key", "")

    options = await api_rc.list_options()

    assert options["by_provider"]["ark"] == ["doubao-chat"]
    assert options["media"]["vision"]["by_provider"]["ark"] == [
        "doubao-vision",
    ]
    assert options["media"]["asr"]["by_provider"]["dashscope"] == ["fun-asr"]
    assert options["media"]["tts"]["by_provider"]["dashscope"] == ["qwen-tts"]
    assert options["tts"] == ["qwen-tts"]
    assert options["media"]["vision"]["providers"][0]["configured"] is True
    assert options["media"]["tts"]["providers"][0]["configured"] is False
    assert (
        options["media"]["tts"]["providers"][0]["credential_env"]
        == "DASHSCOPE_TTS_API_KEY"
    )


@pytest.mark.asyncio
async def test_media_validation_normalizes_values_and_allows_clear(monkeypatch):
    from app.api.admin import runtime_config as api_rc

    chat_model_exists = AsyncMock(return_value=False)
    monkeypatch.setattr(
        api_rc,
        "_model_exists_for_provider",
        chat_model_exists,
    )
    media_exists = AsyncMock(return_value=True)
    monkeypatch.setattr(api_rc, "_media_model_exists", media_exists)

    await api_rc._validate_payload_models(
        ConfigPayload(vision_model="  doubao-vision  ", asr_model="   "),
        fallback_remote_chat_provider="dashscope",
        fallback_remote_small_provider="dashscope",
        fallback_remote_chat_model="chat",
        fallback_remote_small_model="small",
    )

    media_exists.assert_awaited_once_with("doubao-vision", "vision")
    chat_model_exists.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_model_lookup_requires_enabled_llm_kind(monkeypatch):
    from app.api.admin import runtime_config as api_rc

    find_first = AsyncMock(return_value=SimpleNamespace(id="model"))
    monkeypatch.setattr(
        api_rc.db,
        "modelregistry",
        SimpleNamespace(find_first=find_first),
        raising=False,
    )

    assert await api_rc._model_exists_for_provider("chat", "ark") is True
    assert find_first.await_args.kwargs["where"] == {
        "identifier": "chat",
        "provider": "ark",
        "modelKind": "llm",
        "enabled": True,
    }


# ─── consumers pick up override ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_transcribe_audio_uses_effective_asr_model(monkeypatch, loaded_caches):
    from app.services.speech_to_text import fun_asr

    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {"asrModel": "fun-asr-admin"})
    monkeypatch.setattr(fun_asr.settings, "dashscope_api_key", "key")
    monkeypatch.setattr(
        fun_asr.settings, "dashscope_asr_endpoint", "https://asr.example/api",
    )

    captured: dict = {}

    class FakeResponse:
        status_code = 200
        headers = {"x-request-id": "req-1"}

        def json(self):
            return {"output": {"text": "你好"}, "request_id": "req-1"}

    class FakeClient:
        async def post(self, endpoint, headers=None, json=None):
            captured["json"] = json
            return FakeResponse()

    result = await fun_asr.transcribe_audio(
        audio=b"pcm-bytes", mime="audio/wav", audio_format="wav",
        client=FakeClient(),
    )
    assert captured["json"]["model"] == "fun-asr-admin"
    assert result.model == "fun-asr-admin"


@pytest.mark.asyncio
async def test_vision_call_uses_effective_vision_model(monkeypatch, loaded_caches):
    from app.services.chat_media import vision

    monkeypatch.setattr(
        runtime_config, "_GLOBAL_CACHE", {"visionModel": "doubao-vision-admin"},
    )

    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "一张图片"}}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, endpoint, headers=None, json=None):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(vision.httpx, "AsyncClient", FakeAsyncClient)
    content = await vision._call_doubao_vision(data_url="data:image/png;base64,x", user_text="看看")
    assert content == "一张图片"
    assert captured["json"]["model"] == "doubao-vision-admin"


# ─── PUT endpoint wiring (global writes media columns) ────────────────────


@pytest.mark.asyncio
async def test_put_system_config_persists_media_fields(monkeypatch):
    from app.api.admin import runtime_config as api_rc

    captured: dict = {}

    async def fake_upsert(where=None, data=None):
        captured["data"] = data
        row = SimpleNamespace(
            onlineModel=True, remoteProvider=None, remoteChatProvider=None,
            remoteSmallProvider=None, localChatModel=None, localSmallModel=None,
            remoteChatModel=None, remoteSmallModel=None,
            visionModel=data["update"]["visionModel"],
            asrModel=data["update"]["asrModel"],
        )
        return row

    monkeypatch.setattr(api_rc, "_validate_payload_models", AsyncMock())
    monkeypatch.setattr(api_rc, "load_caches", AsyncMock())
    monkeypatch.setattr(api_rc, "invalidate_caches", lambda: None)
    monkeypatch.setattr(
        api_rc.db, "systemconfig",
        SimpleNamespace(upsert=fake_upsert), raising=False,
    )

    resp = await api_rc.put_system_config(
        ConfigPayload(vision_model="doubao-v3", asr_model="fun-asr-v3"),
    )
    assert captured["data"]["update"]["visionModel"] == "doubao-v3"
    assert captured["data"]["update"]["asrModel"] == "fun-asr-v3"
    assert resp["config"]["vision_model"] == "doubao-v3"
    assert resp["config"]["asr_model"] == "fun-asr-v3"
    assert "vision_model" in resp["resolved"]
    assert "asr_model" in resp["resolved"]
