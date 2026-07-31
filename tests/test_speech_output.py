from __future__ import annotations

import io
import json
import struct
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.speech_output import client as tts_client
from app.services.speech_output import voices
from app.services.speech_output.policy import VoiceContext, should_generate_voice
from app.services.speech_output.style import (
    decorate_text_with_emotion,
    instruction_billable_characters,
    resolve_style_instruction,
)
from app.services.speech_output.voices import SYSTEM_VOICE_BY_GENDER


def _wav_bytes(*, seconds: float = 0.25, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x00\x00" * round(seconds * rate))
    return buffer.getvalue()


def test_billable_characters_follow_dashscope_rules():
    assert tts_client.count_billable_characters("你好 A1。") == 2 + 2 + 1 + 1 + 1 + 1


def test_wav_duration_uses_actual_frames():
    assert tts_client.wav_duration_milliseconds(_wav_bytes(seconds=0.25)) == 250


def test_wav_duration_handles_streaming_length_sentinel():
    audio = bytearray(_wav_bytes(seconds=0.5))
    struct.pack_into("<I", audio, 4, 0xFFFFFFFF)
    data_offset = audio.index(b"data")
    struct.pack_into("<I", audio, data_offset + 4, 0xFFFFFFFF)

    assert tts_client.wav_duration_milliseconds(bytes(audio)) == 500


@pytest.mark.parametrize(
    ("probability", "random_number", "expected"),
    [(0, 0.0, False), (100, 0.999, True), (40, 0.39, True), (40, 0.4, False)],
)
@pytest.mark.asyncio
async def test_probability_boundaries(probability, random_number, expected):
    assert await should_generate_voice(
        context=VoiceContext.NORMAL_CHAT,
        client_supports_voice=True,
        probability=probability,
        random_value=lambda: random_number,
    ) is expected


@pytest.mark.asyncio
async def test_ineligible_or_unsupported_client_is_always_text():
    assert not await should_generate_voice(
        context=VoiceContext.CRISIS,
        client_supports_voice=True,
        probability=100,
    )
    assert not await should_generate_voice(
        context=VoiceContext.NORMAL_CHAT,
        client_supports_voice=False,
        probability=100,
    )


def test_voice_assignment_uses_plus_gender_fallbacks():
    assert SYSTEM_VOICE_BY_GENDER["female"] == "longanlingxin"
    assert SYSTEM_VOICE_BY_GENDER["male"] == "longanlufeng"


@pytest.mark.asyncio
async def test_stylized_legacy_voice_is_migrated(monkeypatch):
    fake_db = SimpleNamespace(
        query_raw=AsyncMock(
            side_effect=[
                [{"tts_voice_id": "Momo", "gender": "female"}],
                [],
                [{"voice_id": "longanlingxin"}],
            ],
        ),
        execute_raw=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(voices, "db", fake_db)
    agent = SimpleNamespace(
        id="agent-1",
        gender="female",
        currentMbti={"type": "ENFP"},
        ttsVoiceId="Momo",
    )

    assert await voices.ensure_agent_voice(agent) == "longanlingxin"
    assert agent.ttsVoiceId == "longanlingxin"
    assert fake_db.execute_raw.await_args.args[-2:] == (
        "longanlingxin",
        "agent-1",
    )


@pytest.mark.asyncio
async def test_agent_tts_settings_are_loaded_from_latest_db_row(monkeypatch):
    fake_db = SimpleNamespace(
        query_raw=AsyncMock(
            side_effect=[
                [
                    {
                        "tts_voice_id": "longanlingxin",
                        "gender": "female",
                        "tts_rate": 1.25,
                        "tts_pitch": 0.9,
                        "tts_volume": 63,
                        "tts_seed": 12,
                        "tts_instruction": "轻松自然",
                        "tts_auto_emotion": False,
                        "tts_emotion_scale": 0.7,
                    }
                ],
                [{"exists": 1}],
            ],
        ),
    )
    monkeypatch.setattr(voices, "db", fake_db)

    settings = await voices.get_agent_tts_settings("agent-1")

    assert settings.voice_id == "longanlingxin"
    assert settings.rate == 1.25
    assert settings.pitch == 0.9
    assert settings.volume == 63
    assert settings.seed == 12
    assert settings.instruction == "轻松自然"
    assert settings.auto_emotion is False
    assert settings.emotion_scale == 0.7


def test_style_instruction_prioritizes_natural_conversation():
    instruction = resolve_style_instruction(None)
    assert "熟人聊天" in instruction
    assert instruction_billable_characters(instruction) <= 100
    assert (
        decorate_text_with_emotion(
            "别这样。",
            "愤怒",
            80,
            enabled=True,
            scale=1.0,
        )
        == "[angry]别这样。"
    )


@pytest.mark.asyncio
async def test_synthesize_uses_dedicated_key_and_returns_metering(monkeypatch):
    audio = _wav_bytes(seconds=0.5)
    captured = {}

    class FakeResponse:
        status_code = 200
        headers = {
            "x-request-id": "tts-request-1",
            "content-type": "text/event-stream",
        }
        content = b""
        text = "data: " + json.dumps(
            {
                "request_id": "tts-request-1",
                "output": {
                    "finish_reason": "stop",
                    "audio": {
                        "url": "http://dashscope-test.oss-cn-beijing.aliyuncs.com/out.wav",
                    },
                },
                "usage": {"characters": 7},
            },
        )

    class FakeAudioResponse:
        status_code = 200
        headers = {"content-type": "audio/wav"}
        content = audio

    class FakeClient:
        async def post(self, endpoint, headers=None, json=None):
            captured["endpoint"] = endpoint
            captured["authorization"] = headers["Authorization"]
            captured["payload"] = json
            return FakeResponse()

        async def get(self, url):
            captured["audio_url"] = url
            return FakeAudioResponse()

    monkeypatch.setattr(tts_client.settings, "dashscope_tts_api_key", "tts-key")
    monkeypatch.setattr(tts_client.settings, "dashscope_tts_endpoint", "https://tts.example/api")
    monkeypatch.setattr(tts_client.settings, "dashscope_tts_max_bytes", 1024 * 1024)

    async def fake_model():
        return "qwen-tts-test"

    monkeypatch.setattr(tts_client, "get_effective_tts_model", fake_model)
    monkeypatch.setattr(
        tts_client,
        "get_tts_pricing",
        lambda model: {
            "unit_price_cny": 0.8,
            "billing_unit": "per_10k_characters",
        },
    )
    result = await tts_client.synthesize_speech(
        text="你好",
        voice_id="longanlingxin",
        instruction="自然表达",
        rate=1.2,
        pitch=0.9,
        volume=60,
        seed=42,
        client=FakeClient(),
    )

    assert captured["authorization"] == "Bearer tts-key"
    assert captured["payload"]["model"] == "qwen-tts-test"
    assert captured["payload"]["input"]["voice"] == "longanlingxin"
    assert captured["payload"]["input"]["rate"] == 1.2
    assert captured["payload"]["input"]["pitch"] == 0.9
    assert captured["payload"]["input"]["volume"] == 60
    assert captured["payload"]["input"]["seed"] == 42
    assert captured["payload"]["input"]["enable_aigc_tag"] is True
    assert "optimize_instructions" not in captured["payload"]["input"]
    assert captured["audio_url"].startswith("https://dashscope-test.")
    assert result.duration_milliseconds == 500
    assert result.request_id == "tts-request-1"
    assert result.billable_characters == 7
    assert result.cost_cny == pytest.approx(7 * 0.8 / 10_000)


@pytest.mark.asyncio
async def test_short_circuit_voice_is_saved_bound_and_emitted(monkeypatch):
    from app.services.chat import multi_intent
    from app.services.speech_output import delivery, policy

    prepared = SimpleNamespace(
        transcript="晚安，早点休息。",
        metadata={
            "id": "attachment-1",
            "kind": "audio",
            "mime": "audio/wav",
            "url": "/chat/media/voice.wav",
        },
    )
    save_calls = []
    bind_calls = []

    async def fake_prepare(**kwargs):
        return prepared

    async def fake_bind(value, *, message_id):
        bind_calls.append((value, message_id))

    async def fake_save(*args, **kwargs):
        save_calls.append((args, kwargs))
        return "assistant-message-1"

    async def always_voice(**kwargs):
        return True

    monkeypatch.setattr(policy, "should_generate_voice", always_voice)
    monkeypatch.setattr(delivery, "prepare_voice_output", fake_prepare)
    monkeypatch.setattr(delivery, "bind_prepared_voice_output", fake_bind)
    monkeypatch.setattr(
        multi_intent,
        "_fire_background",
        lambda coroutine: coroutine.close(),
    )
    monkeypatch.setattr(
        multi_intent,
        "save_last_reply_timestamp",
        AsyncMock(),
    )

    events = await multi_intent.short_circuit_reply(
        "晚安，早点休息。",
        "conversation-1",
        "agent-1",
        "user-1",
        fake_save,
        agent=SimpleNamespace(id="agent-1"),
        reply_context={"client_supports_voice": True},
        voice_context=VoiceContext.NORMAL_CHAT,
    )

    payload = json.loads(events[0]["data"])
    assert payload["display_mode"] == "voice"
    assert payload["assistant_message_id"] == "assistant-message-1"
    assert payload["attachments"][0]["id"] == "attachment-1"
    assert len(save_calls) == 1
    assert bind_calls == [(prepared, "assistant-message-1")]
