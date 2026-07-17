from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
from pathlib import Path
import uuid

from fastapi import HTTPException
from fastapi.responses import FileResponse

_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MEDIA_DIR = Path(os.getenv("CHAT_MEDIA_DIR", "var/chat_media"))
_MEDIA_PUBLIC_PREFIX = (
    os.getenv("CHAT_MEDIA_PUBLIC_PREFIX", "/chat/media").strip().rstrip("/")
    or "/chat/media"
)
_ALLOWED_IMAGE_MIMES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_AUDIO_EXTENSIONS = {
    "audio/aac": ".aac",
    "audio/amr": ".amr",
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
}


def strip_base64_prefix(value: str) -> str:
    raw = value.strip()
    comma = raw.find(",")
    return raw[comma + 1 :] if comma >= 0 else raw


def decode_image_base64(value: str) -> bytes:
    try:
        return base64.b64decode(strip_base64_prefix(value), validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image base64") from exc


def normalize_image_mime(mime: str | None) -> str:
    normalized = (mime or "image/jpeg").strip().lower()
    if normalized == "image/jpg":
        return "image/jpeg"
    if normalized not in _ALLOWED_IMAGE_MIMES:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    return normalized


def validate_image_size(blob: bytes) -> None:
    if not blob:
        raise HTTPException(status_code=400, detail="Image is empty")
    if len(blob) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image must be under 10MB")


def conversation_storage_prefix(user_id: str, conversation_id: str) -> str:
    """Return a deterministic, non-overlapping prefix for chat-owned media."""
    conversation_scope = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:24]
    return f"{user_id}_c{conversation_scope}_"


def storage_key_for(
    user_id: str,
    mime: str,
    *,
    conversation_id: str | None = None,
) -> str:
    ext = _ALLOWED_IMAGE_MIMES.get(mime) or _AUDIO_EXTENSIONS.get(mime)
    if ext is None:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    prefix = (
        conversation_storage_prefix(user_id, conversation_id)
        if conversation_id
        else f"{user_id}_"
    )
    return f"{prefix}{uuid.uuid4().hex}{ext}"


def storage_path(storage_key: str) -> Path:
    if "/" in storage_key or "\\" in storage_key or ".." in storage_key:
        raise HTTPException(status_code=400, detail="Invalid media storage key")
    return _MEDIA_DIR / storage_key


def media_url(storage_key: str) -> str:
    return f"{_MEDIA_PUBLIC_PREFIX}/{storage_key}"


def save_image_blob(
    *,
    user_id: str,
    blob: bytes,
    mime: str,
    conversation_id: str | None = None,
) -> str:
    validate_image_size(blob)
    _MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    key = storage_key_for(user_id, mime, conversation_id=conversation_id)
    storage_path(key).write_bytes(blob)
    return key


def save_audio_blob(
    *,
    user_id: str,
    conversation_id: str,
    blob: bytes,
    mime: str,
) -> str:
    if not blob:
        raise HTTPException(status_code=400, detail="语音内容为空")
    if mime not in _AUDIO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="不支持该语音格式")
    _MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    key = storage_key_for(user_id, mime, conversation_id=conversation_id)
    storage_path(key).write_bytes(blob)
    return key


def read_image_base64(storage_key: str, mime: str) -> str:
    path = storage_path(storage_key)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def delete_media_file(storage_key: str | None) -> None:
    if not storage_key:
        return
    path = storage_path(storage_key)
    if path.exists() and path.is_file():
        path.unlink()


def serve_media(
    storage_key: str,
    *,
    user_id: str,
    is_admin: bool = False,
) -> FileResponse:
    path = storage_path(storage_key)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    if not is_admin and not storage_key.startswith(f"{user_id}_"):
        raise HTTPException(status_code=403, detail="Not your media")
    explicit_type = "audio/mp4" if path.suffix.lower() == ".m4a" else None
    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path,
        media_type=explicit_type or media_type or "application/octet-stream",
    )
