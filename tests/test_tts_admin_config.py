from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.api.admin import model_registry, runtime_config


def _system_row(probability: int):
    return SimpleNamespace(
        onlineModel=True,
        remoteProvider="dashscope",
        remoteChatProvider="dashscope",
        remoteSmallProvider="dashscope",
        localChatModel="qwen2.5:14b",
        localSmallModel="qwen2.5:7b",
        remoteChatModel="qwen3.5-plus",
        remoteSmallModel="qwen3.5-flash",
        visionModel="vision",
        asrModel="asr",
        ttsModel="qwen-tts",
        ttsOutputProbability=probability,
        webSearchEnabled=False,
    )


def test_tts_probability_payload_rejects_out_of_range():
    with pytest.raises(ValidationError):
        runtime_config.TtsProbabilityPayload(probability=101)
    with pytest.raises(ValidationError):
        runtime_config.ConfigPayload(tts_output_probability=-1)


@pytest.mark.asyncio
async def test_probability_endpoint_only_updates_tts_field(monkeypatch):
    captured = {}

    async def fake_upsert(where=None, data=None):
        captured["data"] = data
        return _system_row(data["update"]["ttsOutputProbability"])

    monkeypatch.setattr(
        runtime_config.db,
        "systemconfig",
        SimpleNamespace(upsert=fake_upsert),
        raising=False,
    )
    monkeypatch.setattr(runtime_config, "load_caches", AsyncMock())
    monkeypatch.setattr(runtime_config, "invalidate_caches", lambda: None)
    monkeypatch.setattr(runtime_config, "_sync_tts_probability", AsyncMock())
    monkeypatch.setattr(
        runtime_config,
        "resolve_config_sync",
        lambda agent_id=None: SimpleNamespace(
            online_model=True,
            remote_provider="dashscope",
            remote_chat_provider="dashscope",
            remote_small_provider="dashscope",
            local_chat_model="qwen2.5:14b",
            local_small_model="qwen2.5:7b",
            remote_chat_model="qwen3.5-plus",
            remote_small_model="qwen3.5-flash",
            vision_model="vision",
            asr_model="asr",
            tts_model="qwen-tts",
            tts_output_probability=37,
            web_search_enabled=False,
        ),
    )

    result = await runtime_config.put_tts_output_probability(
        runtime_config.TtsProbabilityPayload(probability=37),
    )

    assert captured["data"]["update"] == {"ttsOutputProbability": 37}
    assert result["probability"] == 37
    assert result["resolved"]["tts_output_probability"] == 37


def test_tts_registry_requires_character_billing():
    with pytest.raises(Exception) as exc:
        model_registry._validate_model_metadata("tts", "per_million_tokens")
    assert getattr(exc.value, "status_code", None) == 400
    model_registry._validate_model_metadata("tts", "per_10k_characters")
