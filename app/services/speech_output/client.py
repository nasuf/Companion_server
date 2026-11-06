"""Voice messages: the call model reads the reply aloud.

One realtime session per message, so voice messages and calls share every
voice id, clones included (a DashScope clone is bound to one model). In 78
read-aloud trials the spoken text matched the reply every time it finished
(2026-11-06); the transcript is still checked, with one retry.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import math
import struct
import time
import wave
from array import array
from dataclasses import dataclass
from difflib import SequenceMatcher
from weakref import WeakKeyDictionary

from app.config import settings
from app.services.runtime_config import get_effective_realtime_model


class SpeechSynthesisError(RuntimeError):
    pass


TTS_PROVIDER = "dashscope"
MAX_INPUT_CHARACTERS = 999
SAMPLE_RATE = 24_000
# The spoken text may differ from the reply only in how it was transcribed
# (e.g. digits). A model that answered instead of reading falls far below.
_READ_MATCH = 0.85
_READER_INSTRUCTIONS = (
    "你是语音朗读器，只负责把用户给的台词原样读出来，"
    "绝不回答、不评论、不增删任何字。"
)


@dataclass
class _Transport:
    semaphore: asyncio.Semaphore


_TRANSPORTS: WeakKeyDictionary[asyncio.AbstractEventLoop, _Transport] = WeakKeyDictionary()


def _transport() -> _Transport:
    loop = asyncio.get_running_loop()
    if loop not in _TRANSPORTS:
        _TRANSPORTS[loop] = _Transport(asyncio.Semaphore(max(1, int(settings.tts_max_concurrency))))
    return _TRANSPORTS[loop]


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
    """Length for usage statistics: a Han character is one, others half.

    Billing itself is per token (cost_cny); this keeps tts_usage comparable
    across providers.
    """
    han = sum(1 for char in text if _is_han(char))
    return han + math.ceil((len(text) - han) / 2)


def rate_instruction(rate: float) -> str:
    """The model has no speed parameter; speed is part of the instruction."""
    if rate < 0.85:
        return "语速偏慢。"
    if rate < 0.97:
        return "语速稍慢。"
    if rate > 1.15:
        return "语速偏快。"
    if rate > 1.03:
        return "语速稍快。"
    return ""


def apply_volume(pcm: bytes, volume: int) -> bytes:
    """Scale PCM16 by the stored 0–100 volume (50 = unchanged)."""
    gain = max(0, min(100, int(volume))) / 50
    if gain == 1 or not pcm:
        return pcm
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    for index, value in enumerate(samples):
        samples[index] = max(-32768, min(32767, int(value * gain)))
    return samples.tobytes()


def wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


def _spoken_key(text: str) -> str:
    from app.services.realtime_voice.instructions import spoken_key

    return spoken_key(text)


def read_back_matches(spoken: str, text: str) -> bool:
    heard, wanted = _spoken_key(spoken), _spoken_key(text)
    if not heard or not wanted:
        return False
    return heard == wanted or SequenceMatcher(None, heard, wanted).ratio() >= _READ_MATCH


async def _read_once(*, text: str, voice_id: str, instructions: str, model: str) -> tuple[bytes, str, dict, str | None]:
    """One realtime session: send the line, collect audio and its transcript."""
    from app.services.realtime_voice.protocol import response_create_event
    from app.services.realtime_voice.provider import (
        RealtimeSocket,
        connect_realtime,
        error_message,
        realtime_url,
    )

    api_key = settings.dashscope_api_key.strip()
    if not api_key:
        raise SpeechSynthesisError("DashScope speech is not configured")
    pcm = bytearray()
    spoken = ""
    usage: dict = {}
    response_id: str | None = None
    async with connect_realtime(realtime_url(model), api_key) as raw:
        sock = RealtimeSocket(raw)
        await sock.send_event({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": instructions,
                "voice": voice_id,
                "turn_detection": None,
                "temperature": 0.01,
                "audio": {"output": {"format": {"type": "pcm", "sample_rate": SAMPLE_RATE}}},
            },
        })
        await sock.send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"请原样朗读引号里的台词，只读台词本身：「{text}」"}],
            },
        })
        await sock.send_event(response_create_event())
        async for event in sock:
            kind = event.get("type")
            if kind == "response.audio.delta":
                try:
                    pcm += base64.b64decode(str(event.get("delta") or ""), validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise SpeechSynthesisError("DashScope speech returned invalid audio") from exc
                if len(pcm) > settings.tts_max_bytes:
                    raise SpeechSynthesisError("DashScope speech response exceeded the allowed size")
            elif kind == "response.audio_transcript.done":
                spoken = str(event.get("transcript") or "")
            elif kind == "response.done":
                response = event.get("response") if isinstance(event.get("response"), dict) else {}
                usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
                response_id = str(response.get("id") or "") or None
                break
            elif kind == "error":
                raise SpeechSynthesisError(f"DashScope speech: {error_message(event)}")
    return bytes(pcm), spoken, usage, response_id


async def synthesize_speech(
    *,
    text: str,
    voice_id: str,
    instruction: str | None,
    rate: float = 1.0,
    volume: int = 50,
    model: str | None = None,
) -> SynthesizedSpeech:
    """Read ``text`` aloud in ``voice_id``. ``instruction`` sets the delivery."""
    from app.services.realtime_voice.billing import cost_cny, split_usage

    clean_text = " ".join((text or "").split())
    if not clean_text:
        raise SpeechSynthesisError("TTS text is empty")
    if len(clean_text) > MAX_INPUT_CHARACTERS:
        raise SpeechSynthesisError("TTS text exceeds the provider input limit")
    effective_model = (model or await get_effective_realtime_model() or settings.dashscope_realtime_model).strip()
    style = " ".join(part for part in ((instruction or "").strip(), rate_instruction(rate)) if part)
    instructions = _READER_INSTRUCTIONS + (f"\n朗读要求：{style}" if style else "")

    transport = _transport()
    await transport.semaphore.acquire()
    try:
        total: dict[str, int] = {"input_text": 0, "input_audio": 0, "output_text": 0, "output_audio": 0}
        pcm = b""
        response_id = None
        for attempt in range(2):
            try:
                async with asyncio.timeout(settings.tts_timeout_s):
                    pcm, spoken, usage, response_id = await _read_once(
                        text=clean_text, voice_id=voice_id,
                        instructions=instructions, model=effective_model,
                    )
            except TimeoutError as exc:
                raise SpeechSynthesisError("DashScope speech request timed out") from exc
            except SpeechSynthesisError:
                raise
            except Exception as exc:
                raise SpeechSynthesisError(f"DashScope speech failed: {type(exc).__name__}") from exc
            for key, value in split_usage(usage).items():
                total[key] += value
            if pcm and read_back_matches(spoken, clean_text):
                break
            if attempt == 1:
                raise SpeechSynthesisError("DashScope speech did not read the text verbatim")
    finally:
        transport.semaphore.release()

    # Per-sample Python loop: keep long messages off the event loop.
    pcm = await asyncio.to_thread(apply_volume, pcm, volume)
    duration_ms = max(1, round(len(pcm) * 1000 / (SAMPLE_RATE * 2)))
    billable = count_billable_characters(clean_text)
    cost = cost_cny(effective_model, total)
    return SynthesizedSpeech(
        audio=wav_bytes(pcm),
        mime="audio/wav",
        duration_milliseconds=duration_ms,
        request_id=response_id or f"tts_{time.time_ns()}",
        model=effective_model,
        voice_id=voice_id,
        raw_characters=len(clean_text),
        billable_characters=billable,
        # Per-token billing; the per-10k-character figure is derived for
        # comparison in the voice cost view.
        unit_price_cny=round(cost * 10_000 / billable, 6) if billable else 0.0,
        cost_cny=cost,
    )


def wav_duration_milliseconds(audio: bytes) -> int:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
    except (wave.Error, EOFError, struct.error) as exc:
        raise SpeechSynthesisError("invalid WAV audio") from exc
    if rate <= 0 or frames <= 0:
        raise SpeechSynthesisError("empty WAV audio")
    return max(1, round(frames * 1000 / rate))
