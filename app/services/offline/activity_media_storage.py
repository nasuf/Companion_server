from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
import uuid

from fastapi import HTTPException, Response

_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_AUDIO_BYTES = 5 * 1024 * 1024
_MEDIA_DIR = Path(os.getenv("OFFLINE_MEDIA_DIR", "var/offline_media"))
_MEDIA_PUBLIC_PREFIX = (
    os.getenv("OFFLINE_MEDIA_PUBLIC_PREFIX", "/offline/media").strip().rstrip("/")
    or "/offline/media"
)
_ALLOWED_IMAGE_MIMES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_ALLOWED_AUDIO_MIMES = {
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/mpeg": ".mp3",
}


def _decode_base64(value: str, *, label: str) -> bytes:
    raw = value.strip()
    comma = raw.find(",")
    if comma >= 0:
        raw = raw[comma + 1 :]
    try:
        return base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label} base64") from exc


def decode_image_base64(value: str) -> bytes:
    return _decode_base64(value, label="image")


def decode_audio_base64(value: str) -> bytes:
    return _decode_base64(value, label="audio")


def normalize_image_mime(mime: str | None) -> str:
    normalized = (mime or "image/jpeg").strip().lower()
    if normalized == "image/jpg":
        return "image/jpeg"
    if normalized not in _ALLOWED_IMAGE_MIMES:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    return normalized


def normalize_audio_mime(mime: str | None) -> str:
    normalized = (mime or "audio/mp4").strip().lower()
    if normalized in {"audio/x-m4a", "audio/m4a"}:
        return "audio/mp4"
    if normalized not in _ALLOWED_AUDIO_MIMES:
        raise HTTPException(status_code=400, detail="Unsupported audio type")
    return normalized


def validate_image_size(blob: bytes) -> None:
    if not blob:
        raise HTTPException(status_code=400, detail="Image is empty")
    if len(blob) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image must be under 10MB")


def validate_audio_size(blob: bytes) -> None:
    if not blob:
        raise HTTPException(status_code=400, detail="Audio is empty")
    if len(blob) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="Audio must be under 5MB")


def storage_key_for(user_id: str, mime: str, *, kind: str = "image") -> str:
    if kind == "audio":
        ext = _ALLOWED_AUDIO_MIMES.get(mime, ".m4a")
        return f"{user_id}_voice_{uuid.uuid4().hex}{ext}"
    ext = _ALLOWED_IMAGE_MIMES.get(mime, ".jpg")
    return f"{user_id}_image_{uuid.uuid4().hex}{ext}"


def storage_path(storage_key: str) -> Path:
    if "/" in storage_key or "\\" in storage_key or ".." in storage_key:
        raise HTTPException(status_code=400, detail="Invalid media storage key")
    return _MEDIA_DIR / storage_key


def media_url(storage_key: str) -> str:
    return f"{_MEDIA_PUBLIC_PREFIX}/{storage_key}"


def save_image_blob(*, user_id: str, blob: bytes, mime: str) -> str:
    validate_image_size(blob)
    _MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    key = storage_key_for(user_id, mime, kind="image")
    storage_path(key).write_bytes(blob)
    return key


def save_audio_blob(*, user_id: str, blob: bytes, mime: str) -> str:
    validate_audio_size(blob)
    _MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    key = storage_key_for(user_id, mime, kind="audio")
    storage_path(key).write_bytes(blob)
    return key


def delete_media_file(storage_key: str | None) -> None:
    if not storage_key:
        return
    path = storage_path(storage_key)
    if path.exists() and path.is_file():
        path.unlink()


def delete_user_media_files(user_id: str) -> int:
    if not _MEDIA_DIR.exists() or not _MEDIA_DIR.is_dir():
        return 0
    prefix = f"{user_id}_"
    deleted = 0
    for path in _MEDIA_DIR.iterdir():
        if path.is_file() and path.name.startswith(prefix):
            path.unlink()
            deleted += 1
    return deleted


def serve_media(storage_key: str, *, user_id: str, is_admin: bool = False) -> Response:
    path = storage_path(storage_key)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    if not is_admin and not storage_key.startswith(f"{user_id}_"):
        raise HTTPException(status_code=403, detail="Not your media")
    media_type, _ = mimetypes.guess_type(path.name)
    return Response(
        content=path.read_bytes(),
        media_type=media_type or "application/octet-stream",
    )
