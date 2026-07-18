import base64
import math
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from app.api.public import speech
from app.models.speech import ChatAudioTranscriptionRequest
from app.services.speech_to_text import fun_asr
from app.services.speech_to_text.audio import (
    AudioActivity,
    analyze_pcm16_activity,
    audio_format_for_mime,
    decode_audio_base64,
    normalize_audio_mime,
    validate_audio,
    validate_chat_m4a_duration,
)


def _active_audio() -> AudioActivity:
    return AudioActivity(
        active_milliseconds=1_000,
        total_milliseconds=3_000,
        peak_dbfs=-12.0,
    )


def _test_m4a(duration_seconds: float = 3) -> bytes:
    mvhd_payload = (
        b"\x00\x00\x00\x00"
        + b"\x00" * 8
        + struct.pack(">II", 1000, int(duration_seconds * 1000))
    )
    mvhd = struct.pack(">I4s", 8 + len(mvhd_payload), b"mvhd") + mvhd_payload
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    ftyp = struct.pack(">I4s4s", 12, b"ftyp", b"M4A ")
    return ftyp + moov


def test_chat_audio_request_accepts_flutter_base64_data_alias():
    encoded = base64.b64encode(b"audio").decode("ascii")

    request = ChatAudioTranscriptionRequest.model_validate(
        {
            "conversation_id": "conv-1",
            "duration_seconds": 3,
            "base64Data": encoded,
        }
    )

    assert request.base64 == encoded


def test_audio_validation_normalizes_flutter_m4a_and_checks_size(monkeypatch):
    monkeypatch.setattr(fun_asr.settings, "chat_voice_max_bytes", 6)
    mime = normalize_audio_mime("audio/mp4; codecs=mp4a.40.2", "voice.m4a")
    audio = decode_audio_base64(base64.b64encode(b"audio").decode("ascii"))

    assert mime == "audio/mp4"
    assert audio_format_for_mime(mime) == "m4a"
    validate_audio(audio, declared_size=5, duration_seconds=3)

    with pytest.raises(HTTPException) as error:
        validate_audio(b"1234567", declared_size=7, duration_seconds=3)
    assert error.value.status_code == 413


def test_audio_validation_rejects_invalid_base64_and_unsupported_mime():
    with pytest.raises(HTTPException) as invalid_base64:
        decode_audio_base64("not-base64")
    assert invalid_base64.value.status_code == 400

    with pytest.raises(HTTPException) as unsupported:
        normalize_audio_mime("video/mp4", "voice.bin")
    assert unsupported.value.status_code == 415


def test_chat_m4a_validation_uses_container_duration(monkeypatch):
    monkeypatch.setattr(fun_asr.settings, "chat_voice_min_seconds", 0.5)
    monkeypatch.setattr(fun_asr.settings, "chat_voice_max_seconds", 60)

    assert (
        validate_chat_m4a_duration(
            _test_m4a(3),
            mime="audio/mp4",
            declared_duration_seconds=3,
        )
        == 3
    )

    with pytest.raises(HTTPException) as forged_duration:
        validate_chat_m4a_duration(
            _test_m4a(20),
            mime="audio/mp4",
            declared_duration_seconds=3,
        )
    assert forged_duration.value.status_code == 422

    with pytest.raises(HTTPException) as too_short:
        validate_chat_m4a_duration(
            _test_m4a(0.192),
            mime="audio/mp4",
            declared_duration_seconds=1,
        )
    assert too_short.value.status_code == 422
    assert too_short.value.detail == "语音时间太短，请重新录制"


def test_build_request_payload_contains_only_current_audio():
    payload = fun_asr.build_request_payload(
        audio=b"audio",
        mime="audio/mp4",
        audio_format="m4a",
        model="fun-asr-test",
    )

    messages = payload["input"]["messages"]
    assert payload["model"] == "fun-asr-test"
    assert len(messages) == 1
    audio_data = messages[0]["content"][0]["input_audio"]["data"]
    assert audio_data == (
        "data:audio/mp4;base64," + base64.b64encode(b"audio").decode("ascii")
    )
    assert payload["parameters"] == {
        "format": "m4a",
        "sample_rate": "16000",
        "vad_enabled": True,
    }


def test_pcm_activity_rejects_silence_and_accepts_speech_level_audio():
    sample_rate = 16_000
    duration_seconds = 1
    silence = b"\x00\x00" * sample_rate * duration_seconds
    tone = b"".join(
        struct.pack(
            "<h",
            round(8_000 * math.sin(2 * math.pi * 220 * index / sample_rate)),
        )
        for index in range(sample_rate * duration_seconds)
    )

    silent_activity = analyze_pcm16_activity(silence)
    speech_activity = analyze_pcm16_activity(tone)

    assert silent_activity.active_milliseconds == 0
    assert silent_activity.peak_dbfs == float("-inf")
    assert speech_activity.active_milliseconds == 1_000
    assert speech_activity.peak_dbfs > -20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            {"output": {"sentence": {"text": "嵌套结果"}}, "text": "备用结果"},
            "嵌套结果",
        ),
        ({"text": "顶层结果"}, "顶层结果"),
    ],
)
async def test_fun_asr_calls_native_endpoint_and_parses_both_response_paths(
    monkeypatch,
    output,
    expected,
):
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"output": output, "request_id": "req-1"},
        )

    monkeypatch.setattr(fun_asr.settings, "dashscope_api_key", "dash-key")
    monkeypatch.setattr(
        fun_asr.settings,
        "dashscope_asr_endpoint",
        "https://dashscope.example/asr",
    )
    monkeypatch.setattr(fun_asr.settings, "dashscope_asr_model", "fun-asr-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await fun_asr.transcribe_audio(
            audio=b"audio",
            mime="audio/mp4",
            audio_format="m4a",
            client=client,
        )

    assert result.text == expected
    assert result.request_id == "req-1"
    assert captured == {
        "authorization": "Bearer dash-key",
        "url": "https://dashscope.example/asr",
    }


@pytest.mark.asyncio
async def test_fun_asr_maps_provider_rate_limit(monkeypatch):
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="too many requests")

    monkeypatch.setattr(fun_asr.settings, "dashscope_api_key", "dash-key")
    monkeypatch.setattr(
        fun_asr.settings,
        "dashscope_asr_endpoint",
        "https://dashscope.example/asr",
    )
    monkeypatch.setattr(fun_asr.settings, "dashscope_asr_model", "fun-asr-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(fun_asr.SpeechTranscriptionRateLimited):
            await fun_asr.transcribe_audio(
                audio=b"audio",
                mime="audio/mp4",
                audio_format="m4a",
                client=client,
            )


def test_speech_route_precedes_chat_conversation_fallback():
    from app.main import app

    paths = [
        getattr(route, "path", "")
        for route in app.routes
        if "POST" in getattr(route, "methods", set())
    ]

    assert paths.index("/chat/transcribe") < paths.index("/chat/{conversation_id}")


@pytest.mark.asyncio
async def test_voice_send_persists_audio_and_returns_attachment(
    monkeypatch,
    tmp_path,
):
    def close_background(coro):
        coro.close()

    monkeypatch.setattr(speech, "fire_background", close_background)
    monkeypatch.setattr(speech, "_enforce_rate_limit", AsyncMock())
    monkeypatch.setattr(
        speech.chat_media_repo,
        "conversation_belongs_to_user",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        speech, "analyze_audio_activity", AsyncMock(return_value=_active_audio())
    )
    transcribe = AsyncMock(
        return_value=fun_asr.TranscriptionResult(
            text="今天天气怎么样？",
            request_id="req-1",
            model="fun-asr-test",
        )
    )
    monkeypatch.setattr(speech, "transcribe_audio", transcribe)
    monkeypatch.setattr(speech.chat_media_storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(
        speech.chat_media_storage,
        "storage_key_for",
        lambda _user_id, _mime, **_kwargs: "user-1_voice.m4a",
    )
    audio = _test_m4a(3)
    attachment = speech.chat_media_repo.ChatAttachment(
        id="att-audio-1",
        user_id="user-1",
        conversation_id="conv-1",
        message_id=None,
        kind="audio",
        name="voice.m4a",
        mime="audio/mp4",
        size=len(audio),
        width=None,
        height=None,
        storage_key="user-1_voice.m4a",
        url="/chat/media/user-1_voice.m4a",
        vision_status="skipped",
        vision_summary=None,
        vision_error=None,
        duration_seconds=3,
        transcription_status="ready",
        transcription_text="今天天气怎么样？",
        transcription_model="fun-asr-test",
        transcription_request_id="req-1",
    )
    create_audio_attachment = AsyncMock(return_value=attachment)
    monkeypatch.setattr(
        speech.chat_media_repo,
        "create_audio_attachment",
        create_audio_attachment,
    )
    encoded = base64.b64encode(audio).decode("ascii")

    response = await speech.transcribe_chat_audio(
        ChatAudioTranscriptionRequest(
            conversation_id="conv-1",
            name="voice.m4a",
            mime="audio/mp4",
            size=len(audio),
            duration_seconds=3,
            base64=encoded,
        ),
        user={"sub": "user-1"},
    )

    assert response.text == "今天天气怎么样？"
    assert response.model == "fun-asr-test"
    assert response.request_id == "req-1"
    assert response.attachment is not None
    assert response.attachment.id == "att-audio-1"
    assert response.attachment.transcription_text == "今天天气怎么样？"
    assert (tmp_path / "user-1_voice.m4a").read_bytes() == audio
    create_audio_attachment.assert_awaited_once()
    assert "context" not in transcribe.await_args.kwargs


@pytest.mark.asyncio
async def test_silent_chat_audio_is_rejected_before_transcription(monkeypatch):
    monkeypatch.setattr(speech, "_enforce_rate_limit", AsyncMock())
    monkeypatch.setattr(
        speech.chat_media_repo,
        "conversation_belongs_to_user",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        speech,
        "analyze_audio_activity",
        AsyncMock(
            return_value=AudioActivity(
                active_milliseconds=0,
                total_milliseconds=3_000,
                peak_dbfs=float("-inf"),
            )
        ),
    )
    transcribe = AsyncMock()
    monkeypatch.setattr(speech, "transcribe_audio", transcribe)
    audio = _test_m4a(3)

    with pytest.raises(HTTPException) as error:
        await speech.transcribe_chat_audio(
            ChatAudioTranscriptionRequest(
                conversation_id="conv-1",
                name="voice.m4a",
                mime="audio/mp4",
                size=len(audio),
                duration_seconds=3,
                base64=base64.b64encode(audio).decode("ascii"),
            ),
            user={"sub": "user-1"},
        )

    assert error.value.status_code == 422
    assert error.value.detail == "没有检测到清晰的语音，请重新录制"
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_unreadable_chat_audio_does_not_fall_through_to_provider(monkeypatch):
    monkeypatch.setattr(speech, "_enforce_rate_limit", AsyncMock())
    monkeypatch.setattr(
        speech.chat_media_repo,
        "conversation_belongs_to_user",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        speech,
        "analyze_audio_activity",
        AsyncMock(return_value=None),
    )
    transcribe = AsyncMock()
    monkeypatch.setattr(speech, "transcribe_audio", transcribe)
    audio = _test_m4a(3)

    with pytest.raises(HTTPException) as error:
        await speech.transcribe_chat_audio(
            ChatAudioTranscriptionRequest(
                conversation_id="conv-1",
                name="voice.m4a",
                mime="audio/mp4",
                size=len(audio),
                duration_seconds=3,
                base64=base64.b64encode(audio).decode("ascii"),
            ),
            user={"sub": "user-1"},
        )

    assert error.value.status_code == 422
    assert error.value.detail == "语音文件无法解析，请重新录制"
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_transcribe_chat_audio_deletes_saved_file_when_db_insert_fails(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(speech, "_enforce_rate_limit", AsyncMock())
    monkeypatch.setattr(
        speech.chat_media_repo,
        "conversation_belongs_to_user",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        speech,
        "analyze_audio_activity",
        AsyncMock(return_value=_active_audio()),
    )
    monkeypatch.setattr(
        speech,
        "transcribe_audio",
        AsyncMock(
            return_value=fun_asr.TranscriptionResult(
                text="测试语音",
                request_id="req-2",
                model="fun-asr-test",
            )
        ),
    )
    monkeypatch.setattr(speech.chat_media_storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(
        speech.chat_media_storage,
        "storage_key_for",
        lambda _user_id, _mime, **_kwargs: "user-1_failed.m4a",
    )
    monkeypatch.setattr(
        speech.chat_media_repo,
        "create_audio_attachment",
        AsyncMock(side_effect=RuntimeError("database unavailable")),
    )

    audio = _test_m4a(3)
    with pytest.raises(RuntimeError, match="database unavailable"):
        await speech.transcribe_chat_audio(
            ChatAudioTranscriptionRequest(
                conversation_id="conv-1",
                name="voice.m4a",
                mime="audio/mp4",
                size=len(audio),
                duration_seconds=3,
                display_mode="voice",
                base64=base64.b64encode(audio).decode("ascii"),
            ),
            user={"sub": "user-1"},
        )

    assert not (tmp_path / "user-1_failed.m4a").exists()


@pytest.mark.asyncio
async def test_text_send_transcribes_without_persisting_audio(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(speech, "_enforce_rate_limit", AsyncMock())
    monkeypatch.setattr(
        speech.chat_media_repo,
        "conversation_belongs_to_user",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        speech,
        "analyze_audio_activity",
        AsyncMock(return_value=_active_audio()),
    )
    monkeypatch.setattr(
        speech,
        "transcribe_audio",
        AsyncMock(
            return_value=fun_asr.TranscriptionResult(
                text="只发送文字",
                request_id="req-text",
                model="fun-asr-test",
            )
        ),
    )
    save_audio = AsyncMock()
    create_attachment = AsyncMock()
    monkeypatch.setattr(speech.chat_media_storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(speech.chat_media_storage, "save_audio_blob", save_audio)
    monkeypatch.setattr(
        speech.chat_media_repo,
        "create_audio_attachment",
        create_attachment,
    )

    audio = _test_m4a(3)
    response = await speech.transcribe_chat_audio(
        ChatAudioTranscriptionRequest(
            conversation_id="conv-1",
            name="voice.m4a",
            mime="audio/mp4",
            size=len(audio),
            duration_seconds=3,
            display_mode="text",
            base64=base64.b64encode(audio).decode("ascii"),
        ),
        user={"sub": "user-1"},
    )

    assert response.text == "只发送文字"
    assert response.attachment is None
    save_audio.assert_not_awaited()
    create_attachment.assert_not_awaited()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_chat_audio_rate_limit_blocks_excess_requests(monkeypatch):
    redis = SimpleNamespace(
        incr=AsyncMock(return_value=21),
        expire=AsyncMock(),
    )
    monkeypatch.setattr(speech, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(
        speech.settings,
        "chat_voice_max_requests_per_minute",
        20,
    )

    with pytest.raises(HTTPException) as error:
        await speech._enforce_rate_limit("user-1")

    assert error.value.status_code == 429
