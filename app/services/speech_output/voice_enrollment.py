from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
import uuid
import wave

import httpx

from app.config import settings
from app.services.chat_media import storage as media_storage
from app.services.speech_to_text.audio import (
    analyze_pcm16_activity,
    normalize_audio_mime,
)
from app.services.speech_output.client import SpeechSynthesisError
from app.services.speech_output.voices import QWEN_AUDIO_TTS_MODEL


_PREFIX_RE = re.compile(r"^[A-Za-z0-9]{1,10}$")
_SIGNED_URL_TTL_SECONDS = 10 * 60
_ORPHAN_RETENTION_SECONDS = 20 * 60
_DELAYED_DELETE_SECONDS = 12 * 60
_ENROLLMENT_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class EnrollmentResult:
    voice_id: str
    request_id: str | None


def _signature(storage_key: str, expires_at: int) -> str:
    secret = (
        settings.jwt_secret.strip()
        or "companion-development-tts-enrollment-signing-key"
    )
    payload = f"{storage_key}:{expires_at}".encode("utf-8")
    return hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()


def verify_signed_enrollment_url(
    storage_key: str,
    expires_at: int,
    signature: str,
) -> bool:
    if expires_at < int(time.time()):
        return False
    return hmac.compare_digest(
        _signature(storage_key, expires_at),
        signature,
    )


def enrollment_storage_path(storage_key: str) -> Path:
    if not storage_key.startswith("tts_enroll_"):
        raise ValueError("Invalid enrollment storage key")
    return media_storage.storage_path(storage_key)


async def save_enrollment_audio(
    *,
    blob: bytes,
    mime: str | None,
    filename: str | None,
) -> tuple[str, str]:
    cleanup_expired_enrollment_audio()
    if not blob:
        raise ValueError("Enrollment audio is empty")
    if len(blob) > settings.tts_voice_enrollment_max_bytes:
        raise ValueError("Enrollment audio is too large")
    normalize_audio_mime(mime, filename)
    pcm = await _decode_enrollment_pcm(blob)
    activity = analyze_pcm16_activity(
        pcm,
        sample_rate=_ENROLLMENT_SAMPLE_RATE,
    )
    if activity.total_milliseconds <= 0:
        raise ValueError("Enrollment audio could not be decoded")
    duration_seconds = activity.total_milliseconds / 1000
    if duration_seconds < 3 or duration_seconds > 30.5:
        raise ValueError("Enrollment audio must be between 3 and 30 seconds")
    if activity.active_milliseconds < 1_500:
        raise ValueError("Enrollment audio does not contain enough speech")

    wav = io.BytesIO()
    with wave.open(wav, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(_ENROLLMENT_SAMPLE_RATE)
        output.writeframes(pcm)
    normalized_blob = wav.getvalue()
    storage_key = f"tts_enroll_{uuid.uuid4().hex}.wav"
    path = enrollment_storage_path(storage_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(normalized_blob)
    return storage_key, "audio/wav"


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


def cleanup_expired_enrollment_audio() -> None:
    directory = media_storage.storage_path("tts_enroll_probe").parent
    if not directory.exists():
        return
    cutoff = time.time() - _ORPHAN_RETENTION_SECONDS
    for path in directory.glob("tts_enroll_*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def signed_enrollment_url(
    *,
    storage_key: str,
    request_base_url: str,
) -> str:
    base = (
        settings.tts_voice_enrollment_public_base_url.strip()
        or request_base_url.strip()
    ).rstrip("/")
    if not base.startswith(("https://", "http://")):
        raise ValueError("A public API base URL is required for voice enrollment")
    if settings.is_production and base.startswith("http://"):
        # TLS commonly terminates at the reverse proxy, so FastAPI may observe
        # the internal HTTP hop even though the public origin is HTTPS.
        base = f"https://{base.removeprefix('http://')}"
    expires_at = int(time.time()) + _SIGNED_URL_TTL_SECONDS
    signature = _signature(storage_key, expires_at)
    return (
        f"{base}/admin-api/tts/enrollment-audio/{quote(storage_key)}"
        f"?expires={expires_at}&signature={signature}"
    )


def delete_enrollment_audio(storage_key: str) -> None:
    try:
        enrollment_storage_path(storage_key).unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


async def delete_enrollment_audio_later(storage_key: str) -> None:
    """Keep the signed sample available for provider-side asynchronous fetches."""
    await asyncio.sleep(_DELAYED_DELETE_SECONDS)
    delete_enrollment_audio(storage_key)


def normalize_voice_prefix(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9]", "", value or "")[:10]
    if not _PREFIX_RE.fullmatch(prefix):
        raise ValueError("Voice prefix must contain 1-10 letters or digits")
    return prefix


async def create_cloned_voice(
    *,
    prefix: str,
    audio_url: str,
    client: httpx.AsyncClient | None = None,
) -> EnrollmentResult:
    api_key = settings.dashscope_tts_api_key.strip()
    endpoint = settings.dashscope_tts_voice_enrollment_endpoint.strip()
    if not api_key or not endpoint:
        raise SpeechSynthesisError("DashScope voice enrollment is not configured")
    if not api_key or not endpoint:
        raise SpeechSynthesisError("DashScope voice enrollment is not configured")
    payload = {
        "model": "voice-enrollment",
        "input": {
            "action": "create_voice",
            "target_model": QWEN_AUDIO_TTS_MODEL,
            "prefix": normalize_voice_prefix(prefix),
            "url": audio_url,
            "language_hints": ["zh"],
            "max_prompt_audio_length": 30.0,
            "enable_preprocess": True,
        },
    }
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=settings.dashscope_tts_timeout_s)
    try:
        response = await http.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if response.status_code != 200:
            code = f"http_{response.status_code}"
            message = "request failed"
            try:
                error_body = response.json()
                if isinstance(error_body, dict):
                    code = str(error_body.get("code") or code)
                    message = str(error_body.get("message") or message)
            except ValueError:
                pass
            raise SpeechSynthesisError(
                f"DashScope voice enrollment {code}: {message[:200]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise SpeechSynthesisError(
                "DashScope voice enrollment returned invalid JSON"
            ) from exc
        output = body.get("output") if isinstance(body, dict) else None
        voice_id = (
            output.get("voice_id")
            if isinstance(output, dict)
            else None
        ) or (body.get("voice_id") if isinstance(body, dict) else None)
        if not voice_id:
            code = str(body.get("code") or "missing_voice_id")
            message = str(body.get("message") or "response did not include voice id")
            raise SpeechSynthesisError(
                f"DashScope voice enrollment {code}: {message[:200]}"
            )
        request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("x-dashscope-request-id")
            or (body.get("request_id") if isinstance(body, dict) else None)
        )
        return EnrollmentResult(
            voice_id=str(voice_id),
            request_id=str(request_id) if request_id else None,
        )
    except httpx.HTTPError as exc:
        raise SpeechSynthesisError(
            "DashScope voice enrollment network request failed"
        ) from exc
    finally:
        if owns_client:
            await http.aclose()


async def delete_cloned_voice(
    voice_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> None:
    api_key = settings.dashscope_tts_api_key.strip()
    endpoint = settings.dashscope_tts_voice_enrollment_endpoint.strip()
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=settings.dashscope_tts_timeout_s)
    try:
        response = await http.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "voice-enrollment",
                "input": {
                    "action": "delete_voice",
                    "voice_id": voice_id,
                },
            },
        )
        if response.status_code != 200:
            raise SpeechSynthesisError(
                f"DashScope voice deletion failed: http_{response.status_code}"
            )
    except httpx.HTTPError as exc:
        raise SpeechSynthesisError(
            "DashScope voice deletion network request failed"
        ) from exc
    finally:
        if owns_client:
            await http.aclose()
