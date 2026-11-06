from __future__ import annotations

import io
import struct
import wave
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
    # 用真的 ResolvedConfig dataclass, 不是 SimpleNamespace: dataclass 缺字段会在
    # 构造时炸出清晰的 missing kwarg, 而 SimpleNamespace 只会在下游 attribute access
    # 时炸, 每次 ResolvedConfig 新加字段这里都会静默漂 (本文件的原始失败就是这样
    # 来的 —— ba6dd42 加了 7 个新字段, 测试直到运行时 AttributeError 才发现).
    monkeypatch.setattr(
        runtime_config,
        "resolve_config_sync",
        lambda agent_id=None: runtime_config.ResolvedConfig(
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
            proactive_trending_enabled=False,
            proactive_trending_probability=0.30,
            proactive_trending_link_probability=0.30,
            proactive_trending_cache_ttl_s=3600,
            reply_delay_enabled=False,
            reply_delay_max_seconds=300,
            user_message_aggregation_enabled=True,
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


def test_multimodal_registry_rejects_unsupported_providers():
    model_registry._validate_model_metadata(
        "vision",
        "per_million_tokens",
        provider="ark",
    )
    model_registry._validate_model_metadata(
        "asr",
        "per_million_tokens",
        provider="dashscope",
    )
    with pytest.raises(Exception) as exc:
        model_registry._validate_model_metadata(
            "vision",
            "per_million_tokens",
            provider="dashscope",
        )
    assert getattr(exc.value, "status_code", None) == 400


@pytest.mark.asyncio
async def test_model_registry_exposes_multimodal_capabilities(monkeypatch):
    monkeypatch.setattr(
        model_registry.db,
        "modelregistry",
        SimpleNamespace(find_many=AsyncMock(return_value=[])),
        raising=False,
    )

    result = await model_registry.list_models()

    assert set(result["model_kinds"]) == {"llm", "vision", "asr", "tts", "realtime"}
    assert result["model_kind_providers"]["realtime"] == ["dashscope"]
    assert result["model_kind_providers"]["vision"] == ["ark"]
    assert result["model_kind_providers"]["asr"] == ["dashscope"]
    assert result["model_kind_providers"]["tts"] == ["dashscope"]
    assert all(p["id"] != "stepfun" for p in result["providers"])


def test_voice_models_are_dashscope_only():
    model_registry._validate_model_metadata("realtime", "per_million_tokens", provider="dashscope")
    for kind, unit in (("realtime", "per_million_tokens"), ("tts", "per_10k_characters")):
        with pytest.raises(Exception) as exc:
            model_registry._validate_model_metadata(kind, unit, provider="stepfun")
        assert getattr(exc.value, "status_code", None) == 400
    with pytest.raises(Exception) as exc:
        model_registry._validate_provider("stepfun")
    assert getattr(exc.value, "status_code", None) == 400


def test_agent_tts_payload_rejects_provider_parameter_overflow():
    with pytest.raises(ValidationError):
        tts.AgentTtsPayload(
            voice_profile_id="voice-1",
            rate=2.1,
            volume=50,
            emotion_scale=1,
        )
    with pytest.raises(ValidationError):
        tts.AgentTtsPayload(
            voice_profile_id="voice-1",
            rate=1,
            volume=101,
            emotion_scale=1,
        )
    # Older admin clients still send DashScope-only fields; they are ignored.
    legacy = tts.AgentTtsPayload(
        voice_profile_id="voice-1", rate=1, volume=50, emotion_scale=1,
        pitch=0.9, seed=42,
    )
    assert not hasattr(legacy, "pitch") and not hasattr(legacy, "seed")


def test_instruction_uses_weighted_limit():
    assert tts._validate_instruction("自然一点") == "自然一点"
    with pytest.raises(Exception) as exc:
        tts._validate_instruction("真" * 51)
    assert getattr(exc.value, "status_code", None) == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds,kept", [(5, 5), (15, 15), (25, 20), (60, 20)])
async def test_enrollment_audio_is_normalized_and_capped_at_twenty_seconds(
    monkeypatch, seconds, kept,
):
    pcm = struct.pack("<h", 5000) * (24_000 * seconds)
    monkeypatch.setattr(
        voice_enrollment, "_decode_enrollment_pcm", AsyncMock(return_value=pcm),
    )

    audio = await voice_enrollment.prepare_enrollment_audio(
        blob=b"source-m4a", mime="audio/mp4", filename="voice.m4a",
    )

    with wave.open(io.BytesIO(audio), "rb") as output:
        assert output.getframerate() == 24_000
        assert output.getnchannels() == 1
        assert output.getsampwidth() == 2
        assert output.getnframes() == 24_000 * kept


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("seconds", "active_ms", "error"),
    [(4, 4_000, "10–20 秒"), (61, 61_000, "10–20 秒"), (12, 2_999, "人声太少")],
)
async def test_enrollment_rejects_bad_length_or_mostly_silent_samples(
    monkeypatch, seconds, active_ms, error,
):
    monkeypatch.setattr(
        voice_enrollment,
        "_decode_enrollment_pcm",
        AsyncMock(return_value=b"\x00\x00" * (24_000 * seconds)),
    )
    monkeypatch.setattr(
        voice_enrollment,
        "analyze_pcm16_activity",
        lambda *args, **kwargs: SimpleNamespace(active_milliseconds=active_ms),
    )

    with pytest.raises(ValueError, match=error):
        await voice_enrollment.prepare_enrollment_audio(
            blob=b"source-m4a", mime="audio/mp4", filename="voice.m4a",
        )


@pytest.mark.asyncio
async def test_enrollment_rejects_empty_or_oversized_uploads(monkeypatch):
    with pytest.raises(ValueError, match="empty"):
        await voice_enrollment.prepare_enrollment_audio(blob=b"", mime="audio/mp4", filename="a.m4a")
    monkeypatch.setattr(voice_enrollment.settings, "tts_voice_enrollment_max_bytes", 4)
    with pytest.raises(ValueError, match="too large"):
        await voice_enrollment.prepare_enrollment_audio(blob=b"12345", mime="audio/mp4", filename="a.m4a")


@pytest.mark.asyncio
async def test_agent_tts_update_persists_every_runtime_parameter(monkeypatch):
    fake_db = SimpleNamespace(
        query_raw=AsyncMock(
            side_effect=[
                [
                    {
                        "id": "profile-1",
                        "voice_id": "Serena",
                        "provider": "dashscope",
                        "model": "qwen3.8-omni-flash-realtime",
                        "enabled": True,
                    }
                ],
                [
                    {
                        "id": "agent-1",
                        "name": "小伴",
                        "gender": "female",
                        "user_id": "user-1",
                        "tts_voice_id": "Serena",
                        "tts_rate": 1.2,
                        "tts_volume": 60,
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
            volume=60,
            instruction="自然一点",
            auto_emotion=False,
            emotion_scale=0.8,
        ),
    )

    update_args = fake_db.execute_raw.await_args.args
    assert update_args[1:8] == (
        "Serena",
        1.2,
        60,
        "自然一点",
        False,
        0.8,
        "agent-1",
    )
    assert "tts_pitch" not in update_args[0] and "tts_seed" not in update_args[0]
    assert result["voice_profile_id"] == "profile-1"
    assert result["auto_emotion"] is False


@pytest.mark.asyncio
async def test_voice_profile_from_retired_provider_is_rejected(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(tts, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[{
        "id": "old", "provider": "dashscope", "model": "qwen-audio-3.0-tts-plus",
        "voice_id": "qwen-voice", "enabled": True,
    }])))
    with pytest.raises(HTTPException) as exc:
        await tts._voice_profile("old", require_enabled=True)
    assert exc.value.status_code == 422


def _clone_db():
    return SimpleNamespace(
        query_raw=AsyncMock(side_effect=[
            [{"id": "p1", "provider": "dashscope", "model": "qwen3.8-omni-flash-realtime",
              "voice_id": "qwen-omni-vc-companion-voice-1", "source": "cloned", "enabled": False}],
            [{"count": 0}],
        ]),
        execute_raw=AsyncMock(return_value=1),
    )


@pytest.mark.asyncio
async def test_removing_a_clone_deletes_it_at_dashscope_then_the_row(monkeypatch):
    database = _clone_db()
    remove = AsyncMock()
    monkeypatch.setattr(tts, "db", database)
    monkeypatch.setattr(tts, "delete_cloned_voice", remove)

    assert await tts.delete_voice_profile("p1") == {"status": "ok"}
    remove.assert_awaited_once_with("qwen-omni-vc-companion-voice-1")
    database.execute_raw.assert_awaited_once_with(
        "DELETE FROM tts_voice_profiles WHERE id = $1", "p1",
    )


@pytest.mark.asyncio
async def test_a_failed_provider_deletion_keeps_the_row(monkeypatch):
    from fastapi import HTTPException

    from app.services.speech_output.client import SpeechSynthesisError

    database = _clone_db()
    monkeypatch.setattr(tts, "db", database)
    monkeypatch.setattr(tts, "delete_cloned_voice", AsyncMock(side_effect=SpeechSynthesisError("Throttling")))

    with pytest.raises(HTTPException) as exc:
        await tts.delete_voice_profile("p1")
    assert exc.value.status_code == 502
    database.execute_raw.assert_not_awaited()
