from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
import uuid

from fastapi import HTTPException, Response

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


def storage_key_for(user_id: str, mime: str) -> str:
    ext = _ALLOWED_IMAGE_MIMES.get(mime, ".jpg")
    return f"{user_id}_{uuid.uuid4().hex}{ext}"


def storage_path(storage_key: str) -> Path:
    if "/" in storage_key or "\\" in storage_key or ".." in storage_key:
        raise HTTPException(status_code=400, detail="Invalid media storage key")
    return _MEDIA_DIR / storage_key


def media_url(storage_key: str) -> str:
    return f"{_MEDIA_PUBLIC_PREFIX}/{storage_key}"


def save_image_blob(*, user_id: str, blob: bytes, mime: str) -> str:
    validate_image_size(blob)
    _MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    key = storage_key_for(user_id, mime)
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
