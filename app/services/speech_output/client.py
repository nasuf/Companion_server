from __future__ import annotations

import asyncio
import io
import json
import math
import struct
import wave
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import httpx

from app.config import settings
from app.services.runtime_config import (
    get_effective_tts_model,
    get_tts_pricing,
)


class SpeechSynthesisError(RuntimeError):
    pass


TTS_PROVIDER = "stepfun"
# StepFun rejects input of 1000 characters or more.
MAX_INPUT_CHARACTERS = 999


@dataclass
class _Transport:
    semaphore: asyncio.Semaphore
    client: httpx.AsyncClient | None = None


_TRANSPORTS: WeakKeyDictionary[asyncio.AbstractEventLoop, _Transport] = WeakKeyDictionary()


def _tts_transport() -> _Transport:
    loop = asyncio.get_running_loop()
    if loop not in _TRANSPORTS:
        _TRANSPORTS[loop] = _Transport(
            asyncio.Semaphore(max(1, int(settings.stepfun_tts_max_concurrency)))
        )
    return _TRANSPORTS[loop]


async def close_tts_client() -> None:
    transport = _TRANSPORTS.pop(asyncio.get_running_loop(), None)
    if transport is not None and transport.client is not None:
        await transport.client.aclose()


async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
    data = bytearray()
    async for chunk in response.aiter_bytes(chunk_size=16_384):
        if len(data) + len(chunk) > limit:
            raise SpeechSynthesisError("StepFun TTS response exceeded the allowed size")
        data.extend(chunk)
    return bytes(data)


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


def _is_han(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x323AF
    )


def count_billable_characters(text: str) -> int:
    """StepFun's stepaudio-2.5-tts rule: a Han character counts as one, every
    other character as half (two letters or two punctuation marks = one)."""
    han = sum(1 for char in text if _is_han(char))
    return han + math.ceil((len(text) - han) / 2)


def wav_duration_milliseconds(audio: bytes) -> int:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
    except (wave.Error, EOFError) as exc:
        raise SpeechSynthesisError("StepFun TTS returned invalid WAV audio") from exc
    data_bytes = _wav_data_bytes(audio)
    frame_width = channels * sample_width
    if frame_width <= 0 or data_bytes % frame_width:
        raise SpeechSynthesisError("StepFun TTS returned incomplete WAV frames")
    actual_frames = data_bytes // frame_width if frame_width > 0 else 0
    # Streaming WAV writers commonly leave the RIFF/data length at 0xFFFFFFFF.
    # Python's wave module interprets that sentinel as billions of frames, so
    # prefer the bytes physically present whenever the header is implausible.
    if actual_frames > 0 and (
        frames <= 0 or frames > actual_frames + max(1, rate // 10)
    ):
        frames = actual_frames
    if rate <= 0 or frames <= 0:
        raise SpeechSynthesisError("StepFun TTS returned empty WAV audio")
    return max(1, round(frames * 1000 / rate))


def _wav_data_chunk(audio: bytes) -> tuple[int, int] | None:
    if len(audio) < 20 or audio[8:12] != b"WAVE":
        return None
    offset = 12
    while offset + 8 <= len(audio):
        chunk_id = audio[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", audio, offset + 4)[0]
        payload_start = offset + 8
        if chunk_id == b"data":
            return offset, chunk_size
        if chunk_size == 0xFFFFFFFF:
            return None
        offset = payload_start + chunk_size + (chunk_size % 2)
    return None


def _wav_data_bytes(audio: bytes) -> int:
    chunk = _wav_data_chunk(audio)
    if chunk is None:
        return 0
    offset, size = chunk
    available = max(0, len(audio) - offset - 8)
    if size == 0xFFFFFFFF:
        return available
    if size > available:
        raise SpeechSynthesisError("StepFun TTS returned truncated WAV audio")
    return size


def _finalize_wav(audio: bytes) -> bytes:
    """Replace a streaming length placeholder with the real lengths."""
    chunk = _wav_data_chunk(audio)
    if chunk is None or chunk[1] != 0xFFFFFFFF:
        return audio
    offset, _ = chunk
    finalized = bytearray(audio)
    struct.pack_into("<I", finalized, 4, len(audio) - 8)
    struct.pack_into("<I", finalized, offset + 4, len(audio) - offset - 8)
    return bytes(finalized)


def _response_error(status_code: int, body: bytes) -> SpeechSynthesisError:
    code = f"http_{status_code}"
    message = ""
    try:
        payload = json.loads(body)
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            code = str(error.get("type") or code)
            message = str(error.get("message") or "")
    except (ValueError, UnicodeError):
        pass
    return SpeechSynthesisError(f"StepFun TTS {code}: {message[:160] or 'request failed'}")


def provider_volume(volume: int) -> float:
    """Map the stored 0–100 volume (50 = normal) onto StepFun's 0.1–2.0."""
    return round(max(0.1, min(2.0, int(volume) / 50)), 2)


async def synthesize_speech(
    *,
    text: str,
    voice_id: str,
    instruction: str,
    rate: float = 1.0,
    volume: int = 50,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> SynthesizedSpeech:
    clean_text = " ".join((text or "").split())
    if not clean_text:
        raise SpeechSynthesisError("TTS text is empty")
    if len(clean_text) > MAX_INPUT_CHARACTERS:
        raise SpeechSynthesisError("TTS text exceeds the provider input limit")
    api_key = settings.stepfun_api_key.strip()
    endpoint = settings.stepfun_tts_endpoint.strip()
    effective_model = (model or await get_effective_tts_model()).strip()
    if not api_key or not endpoint or not effective_model:
        raise SpeechSynthesisError("StepFun TTS is not configured")

    payload = {
        "model": effective_model,
        "input": clean_text,
        "voice": voice_id,
        "response_format": "wav",
        "sample_rate": 24_000,
        "speed": max(0.5, min(2.0, float(rate))),
        "volume": provider_volume(volume),
    }
    if instruction:
        payload["instruction"] = instruction
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    transport = _tts_transport()
    await transport.semaphore.acquire()
    try:
        # Create resources after acquisition: cancellation while queued leaks
        # neither sockets nor permits. Share connections only within this loop.
        if client is None and transport.client is None:
            concurrency = max(1, int(settings.stepfun_tts_max_concurrency))
            transport.client = httpx.AsyncClient(
                timeout=settings.stepfun_tts_timeout_s,
                limits=httpx.Limits(
                    max_connections=concurrency,
                    max_keepalive_connections=concurrency,
                ),
            )
        http = client if client is not None else transport.client
        assert http is not None
        async with http.stream(
            "POST", endpoint, headers=headers, json=payload,
            timeout=settings.stepfun_tts_timeout_s,
        ) as response:
            if response.status_code != 200:
                raise _response_error(
                    response.status_code, await _read_bounded(response, 16_384),
                )
            audio = await _read_bounded(response, settings.stepfun_tts_max_bytes)
            request_id = response.headers.get("x-trace-id")
        if not audio:
            raise SpeechSynthesisError("StepFun TTS response did not include audio")
        audio = _finalize_wav(audio)
        duration_ms = wav_duration_milliseconds(audio)
        # StepFun returns no usage block; apply its published counting rule.
        billable = count_billable_characters(clean_text)
        pricing = get_tts_pricing(effective_model) or {}
        configured_price = pricing.get("unit_price_cny")
        unit_price = float(
            configured_price if configured_price is not None
            else settings.tts_price_cny_per_10k_chars
        )
        return SynthesizedSpeech(
            audio=audio,
            mime="audio/wav",
            duration_milliseconds=duration_ms,
            request_id=str(request_id) if request_id else None,
            model=effective_model,
            voice_id=voice_id,
            raw_characters=len(clean_text),
            billable_characters=billable,
            unit_price_cny=unit_price,
            cost_cny=billable * unit_price / 10_000,
        )
    except httpx.TimeoutException as exc:
        raise SpeechSynthesisError("StepFun TTS request timed out") from exc
    except httpx.HTTPError as exc:
        raise SpeechSynthesisError("StepFun TTS network request failed") from exc
    finally:
        transport.semaphore.release()
