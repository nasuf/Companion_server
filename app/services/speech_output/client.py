from __future__ import annotations

import asyncio
import io
import json
import struct
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
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
    except (wave.Error, EOFError) as exc:
        raise SpeechSynthesisError("DashScope TTS returned invalid WAV audio") from exc
    data_bytes = _wav_data_bytes(audio)
    frame_width = channels * sample_width
    actual_frames = data_bytes // frame_width if frame_width > 0 else 0
    # Streaming WAV writers commonly leave the RIFF/data length at 0xFFFFFFFF.
    # Python's wave module interprets that sentinel as billions of frames, so
    # prefer the bytes physically present whenever the header is implausible.
    if actual_frames > 0 and (
        frames <= 0 or frames > actual_frames + max(1, rate // 10)
    ):
        frames = actual_frames
    if rate <= 0 or frames <= 0:
        raise SpeechSynthesisError("DashScope TTS returned empty WAV audio")
    return max(1, round(frames * 1000 / rate))


def _wav_data_bytes(audio: bytes) -> int:
    if len(audio) < 20 or audio[8:12] != b"WAVE":
        return 0
    offset = 12
    while offset + 8 <= len(audio):
        chunk_id = audio[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", audio, offset + 4)[0]
        payload_start = offset + 8
        if chunk_id == b"data":
            available = max(0, len(audio) - payload_start)
            return available if chunk_size == 0xFFFFFFFF else min(
                chunk_size,
                available,
            )
        if chunk_size == 0xFFFFFFFF:
            return 0
        offset = payload_start + chunk_size + (chunk_size % 2)
    return 0


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


def _parse_synthesis_response(response: httpx.Response) -> dict:
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise SpeechSynthesisError(
                "DashScope TTS returned invalid JSON"
            ) from exc
        if not isinstance(body, dict):
            raise SpeechSynthesisError("DashScope TTS returned invalid payload")
        return body

    final_payload: dict | None = None
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        final_payload = event
        output = event.get("output")
        if isinstance(output, dict) and output.get("finish_reason") == "stop":
            return event
    if final_payload is None:
        raise SpeechSynthesisError("DashScope TTS stream returned no events")
    return final_payload


async def synthesize_speech(
    *,
    text: str,
    voice_id: str,
    instruction: str,
    rate: float = 1.0,
    pitch: float = 1.0,
    volume: int = 50,
    seed: int = 0,
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
            "format": "wav",
            "sample_rate": 24_000,
            "volume": max(0, min(100, int(volume))),
            "rate": max(0.5, min(2.0, float(rate))),
            "pitch": max(0.5, min(2.0, float(pitch))),
            "seed": max(0, min(65_535, int(seed))),
            "language_hints": ["zh"],
            "instruction": instruction,
            "enable_aigc_tag": True,
            "enable_ssml": False,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "enable",
    }
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=settings.dashscope_tts_timeout_s)
    semaphore = _tts_semaphore()
    await semaphore.acquire()
    try:
        response = await http.post(endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            raise _response_error(response)
        body = _parse_synthesis_response(response)
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
        usage = body.get("usage") if isinstance(body, dict) else None
        provider_billable = (
            usage.get("characters") if isinstance(usage, dict) else None
        )
        billable = (
            int(provider_billable)
            if provider_billable is not None
            else count_billable_characters(clean_text)
        )
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
