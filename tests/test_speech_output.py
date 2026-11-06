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
    build_speech_plan,
    instruction_billable_characters,
    resolve_style_instruction,
)


def _wav_bytes(*, seconds: float = 0.25, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x00\x00" * round(seconds * rate))
    return buffer.getvalue()


def test_billable_characters_count_han_once_and_others_half():
    assert tts_client.count_billable_characters("你好 A1。") == 2 + 2
    assert tts_client.count_billable_characters("你好A") == 2 + 1
    assert tts_client.count_billable_characters("") == 0


@pytest.mark.parametrize(("rate", "expected"), [
    (0.5, "语速偏慢。"), (0.9, "语速稍慢。"), (1.0, ""), (1.1, "语速稍快。"), (1.5, "语速偏快。"),
])
def test_rate_becomes_part_of_the_reading_instruction(rate, expected):
    assert tts_client.rate_instruction(rate) == expected


def test_volume_scales_pcm_and_clips():
    pcm = struct.pack("<3h", 1000, -1000, 30000)
    assert tts_client.apply_volume(pcm, 50) == pcm
    assert struct.unpack("<3h", tts_client.apply_volume(pcm, 100)) == (2000, -2000, 32767)
    assert struct.unpack("<3h", tts_client.apply_volume(pcm, 0)) == (0, 0, 0)
    assert tts_client.apply_volume(b"", 80) == b""


def test_wav_duration_and_framing():
    assert tts_client.wav_duration_milliseconds(_wav_bytes(seconds=0.25)) == 250
    wav = tts_client.wav_bytes(b"\x00\x00" * 12_000)
    assert tts_client.wav_duration_milliseconds(wav) == 500
    with pytest.raises(tts_client.SpeechSynthesisError):
        tts_client.wav_duration_milliseconds(b"not a wav")
    with pytest.raises(tts_client.SpeechSynthesisError, match="empty"):
        tts_client.wav_duration_milliseconds(tts_client.wav_bytes(b""))


@pytest.mark.parametrize(("spoken", "text", "ok"), [
    ("刚下班到家，累瘫了。", "刚下班到家，累瘫了", True),
    ("3点的会改到4点半了。", "3点的会改到4点半了，你记得别迟到。", False),
    ("我挺好的，就是有点想你了。", "刚下班到家，累瘫了。你今天怎么样？", False),
    ("", "你好", False),
])
def test_read_back_must_match_the_text(spoken, text, ok):
    assert tts_client.read_back_matches(spoken, text) is ok


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
    assert fake_db.execute_raw.await_args.args[1:3] == (
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
                        "tts_volume": 63,
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
    assert settings.volume == 63
    assert settings.instruction == "轻松自然"
    assert settings.auto_emotion is False
    assert settings.emotion_scale == 0.7


def test_style_instruction_prioritizes_natural_conversation():
    instruction = resolve_style_instruction(None)
    assert "微信语音" in instruction
    assert "自然连读" in instruction
    assert "少重音" in instruction
    assert instruction_billable_characters(instruction) <= 100
    plan = build_speech_plan("（生气地）别这样。", "愤怒", 80, instruction=None, enabled=True, scale=1.0)
    # Cues travel in the instruction: the reader would speak them aloud.
    assert plan.text == "别这样。"
    assert plan.instruction.endswith("本句明显生气严肃。")


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
        "finish_assistant_turn",
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
