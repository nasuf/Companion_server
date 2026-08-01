from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.api.admin import model_registry, runtime_config, tts
from app.services.speech_output import voice_enrollment


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


def test_agent_tts_payload_rejects_provider_parameter_overflow():
    with pytest.raises(ValidationError):
        tts.AgentTtsPayload(
            voice_profile_id="voice-1",
            rate=2.1,
            pitch=1,
            volume=50,
            seed=0,
            emotion_scale=1,
        )
    with pytest.raises(ValidationError):
        tts.AgentTtsPayload(
            voice_profile_id="voice-1",
            rate=1,
            pitch=1,
            volume=101,
            seed=0,
            emotion_scale=1,
        )


def test_instruction_uses_dashscope_weighted_limit():
    assert tts._validate_instruction("自然一点") == "自然一点"
    with pytest.raises(Exception) as exc:
        tts._validate_instruction("真" * 51)
    assert getattr(exc.value, "status_code", None) == 422


def test_signed_enrollment_url_expires_and_rejects_tampering(monkeypatch):
    monkeypatch.setattr(voice_enrollment.settings, "jwt_secret", "x" * 40)
    monkeypatch.setattr(
        voice_enrollment.settings,
        "tts_voice_enrollment_public_base_url",
        "https://banshengcomp.com/api",
    )
    url = voice_enrollment.signed_enrollment_url(
        storage_key="tts_enroll_sample.wav",
        request_base_url="http://localhost:8000",
    )
    query = url.split("?", 1)[1]
    values = dict(part.split("=", 1) for part in query.split("&"))
    expires = int(values["expires"])
    signature = values["signature"]

    assert url.startswith(
        "https://banshengcomp.com/api/admin-api/tts/enrollment-audio/",
    )
    assert voice_enrollment.verify_signed_enrollment_url(
        "tts_enroll_sample.wav",
        expires,
        signature,
    )
    assert not voice_enrollment.verify_signed_enrollment_url(
        "tts_enroll_other.wav",
        expires,
        signature,
    )


def test_signed_enrollment_url_restores_public_https_behind_proxy(monkeypatch):
    monkeypatch.setattr(voice_enrollment.settings, "jwt_secret", "x" * 40)
    monkeypatch.setattr(voice_enrollment.settings, "app_env", "production")
    monkeypatch.setattr(
        voice_enrollment.settings,
        "tts_voice_enrollment_public_base_url",
        "",
    )

    url = voice_enrollment.signed_enrollment_url(
        storage_key="tts_enroll_sample.wav",
        request_base_url="http://api.example.com/",
    )

    assert url.startswith(
        "https://api.example.com/admin-api/tts/enrollment-audio/",
    )


@pytest.mark.asyncio
async def test_agent_tts_update_persists_every_runtime_parameter(monkeypatch):
    fake_db = SimpleNamespace(
        query_raw=AsyncMock(
            side_effect=[
                [
                    {
                        "id": "profile-1",
                        "voice_id": "longanlingxin",
                        "model": "qwen-audio-3.0-tts-plus",
                        "enabled": True,
                    }
                ],
                [
                    {
                        "id": "agent-1",
                        "name": "小伴",
                        "gender": "female",
                        "user_id": "user-1",
                        "tts_voice_id": "longanlingxin",
                        "tts_rate": 1.2,
                        "tts_pitch": 0.9,
                        "tts_volume": 60,
                        "tts_seed": 42,
                        "tts_instruction": "自然一点",
                        "tts_auto_emotion": False,
                        "tts_emotion_scale": 0.8,
                    }
                ],
                [{"id": "profile-1"}],
            ],
        ),
        execute_raw=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(tts, "db", fake_db)

    result = await tts.update_agent_tts_config(
        "agent-1",
        tts.AgentTtsPayload(
            voice_profile_id="profile-1",
            rate=1.2,
            pitch=0.9,
            volume=60,
            seed=42,
            instruction="自然一点",
            auto_emotion=False,
            emotion_scale=0.8,
        ),
    )

    update_args = fake_db.execute_raw.await_args.args
    assert update_args[1:9] == (
        "longanlingxin",
        1.2,
        0.9,
        60,
        42,
        "自然一点",
        False,
        0.8,
    )
    assert result["voice_profile_id"] == "profile-1"
    assert result["auto_emotion"] is False
