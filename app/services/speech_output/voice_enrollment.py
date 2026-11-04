from __future__ import annotations

import asyncio
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
# StepFun clones from a 5–10 s reference. Longer uploads keep their first
# 10 s, so a recorder that stops a moment late is not rejected.
_MIN_SAMPLE_SECONDS = 5
_MAX_SAMPLE_SECONDS = 10
_MAX_UPLOAD_SECONDS = 30.5
_MIN_ACTIVE_MILLISECONDS = 3_000


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
    """Decode any supported upload to a 24 kHz mono WAV of at most 10 s."""
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
        raise ValueError("请录制 5–10 秒清晰人声")
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
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            code = str(error.get("type") or code)
            message = str(error.get("message") or message)
    except (ValueError, json.JSONDecodeError):
        pass
    return SpeechSynthesisError(f"StepFun {action} {code}: {message[:200]}")


def _json_object(action: str, response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError as exc:
        raise SpeechSynthesisError(f"StepFun {action} returned invalid JSON") from exc
    if not isinstance(body, dict):
        raise SpeechSynthesisError(f"StepFun {action} returned invalid payload")
    return body


async def create_cloned_voice(
    *,
    audio: bytes,
    client: httpx.AsyncClient | None = None,
) -> EnrollmentResult:
    """Upload the reference WAV, then clone it into the account voice library.

    StepFun has no delete-voice API, so a clone cannot be rolled back. The
    uploaded reference file stays on StepFun alongside the voice it produced.
    """
    api_key = settings.stepfun_api_key.strip()
    files_endpoint = settings.stepfun_files_endpoint.strip()
    voices_endpoint = settings.stepfun_voices_endpoint.strip()
    if not api_key or not files_endpoint or not voices_endpoint:
        raise SpeechSynthesisError("StepFun voice enrollment is not configured")
    headers = {"Authorization": f"Bearer {api_key}"}
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=settings.stepfun_tts_timeout_s)
    try:
        upload = await http.post(
            files_endpoint,
            headers=headers,
            data={"purpose": "storage"},
            files={"file": ("voice_sample.wav", audio, "audio/wav")},
        )
        if upload.status_code != 200:
            raise _provider_error("file upload", upload)
        file_id = _json_object("file upload", upload).get("id")
        if not file_id:
            raise SpeechSynthesisError("StepFun file upload did not return a file id")

        response = await http.post(
            voices_endpoint,
            headers=headers,
            json={"model": TTS_MODEL, "file_id": file_id},
        )
        if response.status_code != 200:
            raise _provider_error("voice enrollment", response)
        voice_id = _json_object("voice enrollment", response).get("id")
        if not voice_id:
            raise SpeechSynthesisError("StepFun voice enrollment did not return a voice id")
        request_id = response.headers.get("x-trace-id")
        return EnrollmentResult(
            voice_id=str(voice_id),
            request_id=str(request_id) if request_id else None,
        )
    except httpx.HTTPError as exc:
        raise SpeechSynthesisError(
            "StepFun voice enrollment network request failed"
        ) from exc
    finally:
        if owns_client:
            await http.aclose()
