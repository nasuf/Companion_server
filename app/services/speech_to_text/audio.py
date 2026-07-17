from __future__ import annotations

import base64
import binascii
import math
from pathlib import Path
import struct

from fastapi import HTTPException

from app.config import settings


_MIME_FORMATS = {
    "audio/aac": "aac",
    "audio/amr": "amr",
    "audio/flac": "flac",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
}

_EXTENSION_MIMES = {
    ".aac": "audio/aac",
    ".amr": "audio/amr",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


def normalize_audio_mime(mime: str | None, name: str | None = None) -> str:
    normalized = (mime or "").split(";", 1)[0].strip().lower()
    if normalized in _MIME_FORMATS:
        return normalized
    extension = Path(name or "").suffix.lower()
    inferred = _EXTENSION_MIMES.get(extension)
    if inferred:
        return inferred
    raise HTTPException(status_code=415, detail="不支持该语音格式")


def audio_format_for_mime(mime: str) -> str:
    try:
        return _MIME_FORMATS[mime]
    except KeyError as exc:
        raise HTTPException(status_code=415, detail="不支持该语音格式") from exc


def decode_audio_base64(value: str) -> bytes:
    payload = value.strip()
    if payload.startswith("data:"):
        header, separator, encoded = payload.partition(",")
        if not separator or ";base64" not in header.lower():
            raise HTTPException(status_code=400, detail="语音数据格式不正确")
        payload = encoded
    try:
        blob = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="语音数据不是有效的 Base64") from exc
    if not blob:
        raise HTTPException(status_code=400, detail="语音内容为空")
    return blob


def validate_audio(
    blob: bytes,
    *,
    declared_size: int | None,
    duration_seconds: int,
) -> None:
    if declared_size is not None and declared_size != len(blob):
        raise HTTPException(status_code=400, detail="语音文件大小校验失败")
    if len(blob) > settings.chat_voice_max_bytes:
        raise HTTPException(status_code=413, detail="语音文件过大，请录制更短的内容")
    if duration_seconds > settings.chat_voice_max_seconds:
        raise HTTPException(
            status_code=422,
            detail=f"单条语音最长 {settings.chat_voice_max_seconds} 秒",
        )


def validate_chat_m4a_duration(
    blob: bytes,
    *,
    mime: str,
    declared_duration_seconds: int,
) -> int:
    """Validate duration from the M4A container instead of trusting the client."""
    if audio_format_for_mime(mime) != "m4a":
        raise HTTPException(status_code=415, detail="聊天语音仅支持 M4A 格式")
    actual = _m4a_duration_seconds(blob)
    if actual is None or actual <= 0:
        raise HTTPException(status_code=422, detail="语音文件无法解析，请重新录制")
    # Encoders and the UI timer can differ around a one-second boundary. A two
    # second tolerance catches forged metadata without rejecting normal clips.
    if abs(actual - declared_duration_seconds) > 2:
        raise HTTPException(status_code=422, detail="语音时长校验失败")
    if actual > settings.chat_voice_max_seconds + 1:
        raise HTTPException(
            status_code=422,
            detail=f"单条语音最长 {settings.chat_voice_max_seconds} 秒",
        )
    return min(settings.chat_voice_max_seconds, max(1, math.ceil(actual)))


def _m4a_duration_seconds(blob: bytes) -> float | None:
    for box_type, payload_start, box_end in _mp4_boxes(blob, 0, len(blob)):
        if box_type != b"moov":
            continue
        for child_type, child_start, child_end in _mp4_boxes(
            blob,
            payload_start,
            box_end,
        ):
            if child_type == b"mvhd":
                return _mvhd_duration(blob[child_start:child_end])
    return None


def _mp4_boxes(blob: bytes, start: int, end: int):
    offset = start
    while offset + 8 <= end:
        size = struct.unpack_from(">I", blob, offset)[0]
        box_type = blob[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if offset + 16 > end:
                return
            size = struct.unpack_from(">Q", blob, offset + 8)[0]
            header_size = 16
        elif size == 0:
            size = end - offset
        if size < header_size or offset + size > end:
            return
        box_end = offset + size
        yield box_type, offset + header_size, box_end
        offset = box_end


def _mvhd_duration(payload: bytes) -> float | None:
    if len(payload) < 20:
        return None
    version = payload[0]
    if version == 0:
        timescale = struct.unpack_from(">I", payload, 12)[0]
        duration = struct.unpack_from(">I", payload, 16)[0]
    elif version == 1:
        if len(payload) < 32:
            return None
        timescale = struct.unpack_from(">I", payload, 20)[0]
        duration = struct.unpack_from(">Q", payload, 24)[0]
    else:
        return None
    if timescale <= 0:
        return None
    return duration / timescale
