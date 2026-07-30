from __future__ import annotations

import asyncio
import io
import json
import wave
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import httpx

from app.config import settings
from app.services.runtime_config import (
    get_effective_tts_model,
    get_tts_pricing,
)


class SpeechSynthesisError(RuntimeError):
    pass


_TTS_SEMAPHORE: asyncio.Semaphore | None = None


def _tts_semaphore() -> asyncio.Semaphore:
    global _TTS_SEMAPHORE
    if _TTS_SEMAPHORE is None:
        _TTS_SEMAPHORE = asyncio.Semaphore(
            max(1, int(settings.dashscope_tts_max_concurrency)),
        )
    return _TTS_SEMAPHORE


@dataclass(frozen=True)
class SynthesizedSpeech:
    audio: bytes
    mime: str
    duration_milliseconds: int
    request_id: str | None
    model: str
    voice_id: str
    raw_characters: int
    billable_characters: int
    unit_price_cny: float
    cost_cny: float


def _is_double_billed_han(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x323AF
    )


def count_billable_characters(text: str) -> int:
    """Apply DashScope's TTS rule: Han characters count as two, others one."""
    return sum(2 if _is_double_billed_han(char) else 1 for char in text)


def wav_duration_milliseconds(audio: bytes) -> int:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
    except (wave.Error, EOFError) as exc:
        raise SpeechSynthesisError("DashScope TTS returned invalid WAV audio") from exc
    if rate <= 0 or frames <= 0:
        raise SpeechSynthesisError("DashScope TTS returned empty WAV audio")
    return max(1, round(frames * 1000 / rate))


def _response_error(response: httpx.Response) -> SpeechSynthesisError:
    code = f"http_{response.status_code}"
    message = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            code = str(body.get("code") or code)
            message = str(body.get("message") or "")
    except ValueError:
        pass
    safe_message = message[:160] or "request failed"
    return SpeechSynthesisError(f"DashScope TTS {code}: {safe_message}")


async def synthesize_speech(
    *,
    text: str,
    voice_id: str,
    instruction: str,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> SynthesizedSpeech:
    clean_text = " ".join((text or "").split())
    if not clean_text:
        raise SpeechSynthesisError("TTS text is empty")
    api_key = settings.dashscope_tts_api_key.strip()
    endpoint = settings.dashscope_tts_endpoint.strip()
    effective_model = (model or await get_effective_tts_model()).strip()
    if not api_key or not endpoint or not effective_model:
        raise SpeechSynthesisError("DashScope TTS is not configured")

    payload = {
        "model": effective_model,
        "input": {
            "text": clean_text,
            "voice": voice_id,
            "language_type": "Chinese",
            "instructions": instruction,
            "optimize_instructions": True,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=settings.dashscope_tts_timeout_s)
    semaphore = _tts_semaphore()
    await semaphore.acquire()
    try:
        response = await http.post(endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            raise _response_error(response)
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise SpeechSynthesisError("DashScope TTS returned invalid JSON") from exc
        output = body.get("output") if isinstance(body, dict) else None
        audio_meta = output.get("audio") if isinstance(output, dict) else None
        audio_url = audio_meta.get("url") if isinstance(audio_meta, dict) else None
        if not isinstance(audio_url, str) or not audio_url:
            raise SpeechSynthesisError("DashScope TTS response did not include audio")
        parsed = urlparse(audio_url)
        if (
            parsed.scheme == "http"
            and parsed.netloc.endswith(".aliyuncs.com")
        ):
            # DashScope currently returns an OSS HTTP URL. Upgrade the trusted
            # Alibaba host before downloading so audio never crosses plaintext.
            parsed = parsed._replace(scheme="https")
            audio_url = urlunparse(parsed)
        if parsed.scheme != "https" or not parsed.netloc:
            raise SpeechSynthesisError("DashScope TTS returned an unsafe audio URL")
        audio_response = await http.get(audio_url)
        if audio_response.status_code != 200:
            raise SpeechSynthesisError(
                f"DashScope TTS audio download failed: http_{audio_response.status_code}"
            )
        audio = audio_response.content
        if not audio or len(audio) > settings.dashscope_tts_max_bytes:
            raise SpeechSynthesisError("DashScope TTS audio exceeded the allowed size")
        duration_ms = wav_duration_milliseconds(audio)
        request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("x-dashscope-request-id")
            or (body.get("request_id") if isinstance(body, dict) else None)
        )
        raw_characters = len(clean_text)
        billable = count_billable_characters(clean_text)
        pricing = get_tts_pricing(effective_model) or {}
        unit_price = float(
            pricing.get("unit_price_cny")
            or settings.tts_price_cny_per_10k_chars
        )
        cost = billable * unit_price / 10_000
        return SynthesizedSpeech(
            audio=audio,
            mime="audio/wav",
            duration_milliseconds=duration_ms,
            request_id=str(request_id) if request_id else None,
            model=effective_model,
            voice_id=voice_id,
            raw_characters=raw_characters,
            billable_characters=billable,
            unit_price_cny=unit_price,
            cost_cny=cost,
        )
    except httpx.TimeoutException as exc:
        raise SpeechSynthesisError("DashScope TTS request timed out") from exc
    except httpx.HTTPError as exc:
        raise SpeechSynthesisError("DashScope TTS network request failed") from exc
    finally:
        semaphore.release()
        if owns_client:
            await http.aclose()
