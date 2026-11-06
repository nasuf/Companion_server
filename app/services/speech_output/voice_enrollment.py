from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from dataclasses import dataclass

import httpx

from app.config import settings
from app.services.speech_to_text.audio import (
    analyze_pcm16_activity,
    normalize_audio_mime,
)
from app.services.speech_output.client import SpeechSynthesisError
from app.services.speech_output.voices import TTS_MODEL


_ENROLLMENT_SAMPLE_RATE = 24_000
# DashScope clones from 10–20 s of speech (60 s at most) and needs at least
# 3 s of continuous reading. Longer uploads keep their first 20 s.
_MIN_SAMPLE_SECONDS = 5
_MAX_SAMPLE_SECONDS = 20
_MAX_UPLOAD_SECONDS = 60.5
_MIN_ACTIVE_MILLISECONDS = 3_000
_CUSTOMIZATION_PATH = "/api/v1/services/audio/tts/customization"
# preferred_name: letters, digits and underscores, at most 16 characters.
_PREFERRED_NAME = "companion"


@dataclass(frozen=True)
class EnrollmentResult:
    voice_id: str
    request_id: str | None


async def prepare_enrollment_audio(
    *,
    blob: bytes,
    mime: str | None,
    filename: str | None,
) -> bytes:
    """Decode any supported upload to a 24 kHz mono WAV of at most 20 s."""
    if not blob:
        raise ValueError("Enrollment audio is empty")
    if len(blob) > settings.tts_voice_enrollment_max_bytes:
        raise ValueError("Enrollment audio is too large")
    normalize_audio_mime(mime, filename)
    pcm = await _decode_enrollment_pcm(blob)
    total_seconds = len(pcm) / 2 / _ENROLLMENT_SAMPLE_RATE
    if total_seconds <= 0:
        raise ValueError("Enrollment audio could not be decoded")
    if total_seconds < _MIN_SAMPLE_SECONDS or total_seconds > _MAX_UPLOAD_SECONDS:
        raise ValueError("请录制 10–20 秒清晰人声（5–60 秒可用）")
    pcm = pcm[: _ENROLLMENT_SAMPLE_RATE * 2 * _MAX_SAMPLE_SECONDS]
    activity = analyze_pcm16_activity(pcm, sample_rate=_ENROLLMENT_SAMPLE_RATE)
    if activity.active_milliseconds < _MIN_ACTIVE_MILLISECONDS:
        raise ValueError("录音里的人声太少，请减少静音后重新录制")

    wav = io.BytesIO()
    with wave.open(wav, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(_ENROLLMENT_SAMPLE_RATE)
        output.writeframes(pcm)
    return wav.getvalue()


async def _decode_enrollment_pcm(blob: bytes) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_ENROLLMENT_SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("Voice enrollment requires ffmpeg") from exc
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(blob),
            timeout=max(15.0, settings.chat_voice_analysis_timeout_s),
        )
    except TimeoutError as exc:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.communicate()
        raise ValueError("Enrollment audio decoding timed out") from exc
    if process.returncode != 0 or not stdout:
        raise ValueError("Enrollment audio could not be decoded")
    return stdout


def _provider_error(action: str, response: httpx.Response) -> SpeechSynthesisError:
    code = f"http_{response.status_code}"
    message = "request failed"
    try:
        body = response.json()
        if isinstance(body, dict):
            code = str(body.get("code") or code)
            message = str(body.get("message") or message)
    except (ValueError, json.JSONDecodeError):
        pass
    return SpeechSynthesisError(f"DashScope {action} {code}: {message[:200]}")


def _json_object(action: str, response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError as exc:
        raise SpeechSynthesisError(f"DashScope {action} returned invalid JSON") from exc
    if not isinstance(body, dict):
        raise SpeechSynthesisError(f"DashScope {action} returned invalid payload")
    return body


async def _customization(action: str, payload: dict, client: httpx.AsyncClient | None) -> httpx.Response:
    from app.services.realtime_voice.provider import http_base_url

    api_key = settings.dashscope_api_key.strip()
    if not api_key:
        raise SpeechSynthesisError("DashScope voice enrollment is not configured")
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=settings.tts_timeout_s, trust_env=False)
    try:
        return await http.post(
            http_base_url() + _CUSTOMIZATION_PATH,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "qwen-voice-enrollment", **payload},
        )
    except httpx.HTTPError as exc:
        raise SpeechSynthesisError(f"DashScope {action} network request failed") from exc
    finally:
        if owns_client:
            await http.aclose()


async def create_cloned_voice(
    *,
    audio: bytes,
    client: httpx.AsyncClient | None = None,
) -> EnrollmentResult:
    """Clone the reference WAV into a voice for the realtime model.

    Calls and voice messages both run on that model, so one clone serves
    both. Clones can be deleted (delete_cloned_voice).
    """
    data_uri = "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")
    response = await _customization("voice enrollment", {
        "input": {
            "action": "create",
            "target_model": TTS_MODEL,
            "preferred_name": _PREFERRED_NAME,
            "audio": {"data": data_uri},
        },
        "parameters": {"voice_clone_mode": "normal"},
    }, client)
    if response.status_code != 200:
        raise _provider_error("voice enrollment", response)
    body = _json_object("voice enrollment", response)
    output = body.get("output") if isinstance(body.get("output"), dict) else {}
    voice_id = output.get("voice")
    if not voice_id:
        raise SpeechSynthesisError("DashScope voice enrollment did not return a voice")
    request_id = body.get("request_id")
    return EnrollmentResult(
        voice_id=str(voice_id),
        request_id=str(request_id) if request_id else None,
    )


async def delete_cloned_voice(
    voice_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Delete a clone at DashScope. A voice already gone counts as deleted."""
    response = await _customization("voice deletion", {
        "input": {"action": "delete", "voice": voice_id},
    }, client)
    if response.status_code == 200:
        return
    if response.status_code == 400:
        try:
            if _json_object("voice deletion", response).get("code") == "BadRequest.VoiceNotFound":
                return
        except SpeechSynthesisError:
            pass
    raise _provider_error("voice deletion", response)
