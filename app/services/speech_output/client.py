from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import struct
import wave
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse
from weakref import WeakKeyDictionary

import httpx

from app.config import settings
from app.services.runtime_config import (
    get_effective_tts_model,
    get_tts_pricing,
)


class SpeechSynthesisError(RuntimeError):
    pass


@dataclass
class _Transport:
    semaphore: asyncio.Semaphore
    client: httpx.AsyncClient | None = None


_TRANSPORTS: WeakKeyDictionary[asyncio.AbstractEventLoop, _Transport] = WeakKeyDictionary()


def _tts_transport() -> _Transport:
    loop = asyncio.get_running_loop()
    if loop not in _TRANSPORTS:
        _TRANSPORTS[loop] = _Transport(
            asyncio.Semaphore(max(1, int(settings.dashscope_tts_max_concurrency)))
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
            raise SpeechSynthesisError("DashScope TTS response exceeded the allowed size")
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
    if frame_width <= 0 or data_bytes % frame_width:
        raise SpeechSynthesisError("DashScope TTS returned incomplete WAV frames")
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


def _streaming_wav_length(audio: bytes, offset: int, size: int) -> bool:
    # Observed DashScope SSE headers use INT_MAX-64 for RIFF and subtract the
    # preceding chunks for data (including a variable-length AIGC chunk).
    riff_size = struct.unpack_from("<I", audio, 4)[0]
    return size == 0xFFFFFFFF or (
        riff_size == 0x7FFFFFBF and size == riff_size - offset
    )


def _wav_data_bytes(audio: bytes) -> int:
    chunk = _wav_data_chunk(audio)
    if chunk is None:
        return 0
    offset, size = chunk
    available = max(0, len(audio) - offset - 8)
    if _streaming_wav_length(audio, offset, size):
        return available
    if size > available:
        raise SpeechSynthesisError("DashScope TTS returned truncated WAV audio")
    return size


def _finalize_wav(audio: bytes) -> bytes:
    """Fix only recognized streaming placeholders after provider completion."""
    chunk = _wav_data_chunk(audio)
    if chunk is None or not _streaming_wav_length(audio, *chunk):
        return audio
    offset, _ = chunk
    finalized = bytearray(audio)
    struct.pack_into("<I", finalized, 4, len(audio) - 8)
    struct.pack_into("<I", finalized, offset + 4, len(audio) - offset - 8)
    return bytes(finalized)


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


def _provider_error(body: dict) -> SpeechSynthesisError:
    message = str(body.get("message") or "request failed")[:160]
    return SpeechSynthesisError(f"DashScope TTS {body['code']}: {message}")


async def _read_synthesis_response(response: httpx.Response) -> tuple[dict, bytes]:
    content_type = response.headers.get("content-type", "").lower()
    max_audio = settings.dashscope_tts_max_bytes
    if "text/event-stream" not in content_type:
        try:
            body = json.loads(await _read_bounded(response, 1_048_576))
        except (ValueError, UnicodeError) as exc:
            raise SpeechSynthesisError(
                "DashScope TTS returned invalid JSON"
            ) from exc
        if not isinstance(body, dict):
            raise SpeechSynthesisError("DashScope TTS returned invalid payload")
        if body.get("code"):
            raise _provider_error(body)
        return body, b""

    audio = bytearray()
    usage: dict = {}
    request_id = None
    # Bound both decoded audio and encoded events, including a missing newline.
    # aiter_lines alone would accumulate an unbounded line before our checks.
    buffer = bytearray()
    event_lines: list[bytes] = []
    received = 0

    def parse_event(lines: list[bytes]) -> dict | None:
        nonlocal usage, request_id
        raw = b"\n".join(lines).strip()
        if not raw or raw == b"[DONE]":
            return None
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise SpeechSynthesisError("DashScope TTS stream returned invalid JSON") from exc
        if not isinstance(event, dict):
            raise SpeechSynthesisError("DashScope TTS stream returned invalid payload")
        if event.get("code"):
            raise _provider_error(event)
        request_id = event.get("request_id") or request_id
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        output = event.get("output")
        meta = output.get("audio") if isinstance(output, dict) else None
        encoded = meta.get("data") if isinstance(meta, dict) else None
        if encoded:
            try:
                chunk = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError, binascii.Error) as exc:
                raise SpeechSynthesisError("DashScope TTS stream returned invalid audio data") from exc
            if len(audio) + len(chunk) > max_audio:
                raise SpeechSynthesisError("DashScope TTS audio exceeded the allowed size")
            audio.extend(chunk)
        if isinstance(output, dict) and output.get("finish_reason") == "stop":
            return {**event, "usage": usage, "request_id": request_id}
        return None

    async for chunk in response.aiter_bytes(chunk_size=16_384):
        received += len(chunk)
        if received > max_audio * 2 + 1_048_576:
            raise SpeechSynthesisError("DashScope TTS stream exceeded the allowed size")
        buffer.extend(chunk)
        while b"\n" in buffer:
            line, _, rest = buffer.partition(b"\n")
            buffer = bytearray(rest)
            line = line.rstrip(b"\r")
            if not line:
                final = parse_event(event_lines)
                event_lines.clear()
                if final is not None:
                    return final, bytes(audio)
            elif line.startswith(b"data:"):
                event_lines.append(line[5:].lstrip())
    if buffer.startswith(b"data:"):
        event_lines.append(buffer[5:].strip())
    final = parse_event(event_lines)
    if final is not None:
        return final, bytes(audio)
    raise SpeechSynthesisError("DashScope TTS stream ended before completion")


async def _download_audio(http: httpx.AsyncClient, audio_url: str) -> bytes:
    parsed = urlparse(audio_url)
    if parsed.scheme == "http" and (parsed.hostname or "").endswith(".aliyuncs.com"):
        parsed = parsed._replace(scheme="https")
        audio_url = urlunparse(parsed)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise SpeechSynthesisError("DashScope TTS returned an unsafe audio URL")
    async with http.stream(
        "GET", audio_url, timeout=settings.dashscope_tts_timeout_s,
    ) as response:
        if response.status_code != 200:
            raise SpeechSynthesisError(f"DashScope TTS audio download failed: http_{response.status_code}")
        return await _read_bounded(response, settings.dashscope_tts_max_bytes)


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
    transport = _tts_transport()
    await transport.semaphore.acquire()
    try:
        # Create resources after acquisition: cancellation while queued leaks
        # neither sockets nor permits. Share connections only within this loop.
        if client is None and transport.client is None:
            concurrency = max(1, int(settings.dashscope_tts_max_concurrency))
            transport.client = httpx.AsyncClient(
                timeout=settings.dashscope_tts_timeout_s,
                limits=httpx.Limits(
                    max_connections=concurrency,
                    max_keepalive_connections=concurrency,
                ),
            )
        http = client if client is not None else transport.client
        assert http is not None
        async with http.stream(
            "POST", endpoint, headers=headers, json=payload,
            timeout=settings.dashscope_tts_timeout_s,
        ) as response:
            if response.status_code != 200:
                error_bytes = await _read_bounded(response, 16_384)
                raise _response_error(httpx.Response(response.status_code, content=error_bytes))
            body, audio = await _read_synthesis_response(response)
            response_request_id = (
                response.headers.get("x-request-id")
                or response.headers.get("x-dashscope-request-id")
            )
        output = body.get("output") if isinstance(body, dict) else None
        audio_meta = output.get("audio") if isinstance(output, dict) else None
        audio_url = audio_meta.get("url") if isinstance(audio_meta, dict) else None
        if not audio:
            if not isinstance(audio_url, str) or not audio_url:
                raise SpeechSynthesisError("DashScope TTS response did not include audio")
            audio = await _download_audio(http, audio_url)
        if not audio or len(audio) > settings.dashscope_tts_max_bytes:
            raise SpeechSynthesisError("DashScope TTS audio exceeded the allowed size")
        audio = _finalize_wav(audio)
        duration_ms = wav_duration_milliseconds(audio)
        request_id = (
            response_request_id
            or (body.get("request_id") if isinstance(body, dict) else None)
        )
        raw_characters = len(clean_text)
        usage = body.get("usage") if isinstance(body, dict) else None
        provider_billable = (
            usage.get("characters") if isinstance(usage, dict) else None
        )
        try:
            billable = (
                int(provider_billable) if provider_billable is not None
                else count_billable_characters(clean_text)
            )
            if billable < 0:
                raise ValueError("negative usage")
        except (TypeError, ValueError, OverflowError) as exc:
            raise SpeechSynthesisError("DashScope TTS returned invalid usage") from exc
        pricing = get_tts_pricing(effective_model) or {}
        configured_price = pricing.get("unit_price_cny")
        unit_price = float(
            configured_price if configured_price is not None
            else settings.tts_price_cny_per_10k_chars
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
        transport.semaphore.release()
