"""Local storage for user feedback screenshot attachments."""

from __future__ import annotations

import io
import os
import re
import uuid
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_IMAGES_PER_FEEDBACK = 3
_FEEDBACK_DIR = Path(os.getenv("FEEDBACK_MEDIA_DIR", "var/feedback_media"))
_PUBLIC_PREFIX = (
    os.getenv("FEEDBACK_MEDIA_PUBLIC_PREFIX", "/users/me/feedback/media").strip().rstrip("/")
    or "/users/me/feedback/media"
)
_ADMIN_PREFIX = "/admin-api/user-feedback/media"
_INGEST_MAX_EDGE = 2048
_INGEST_JPEG_QUALITY = 85
_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,160}\.jpg$")
_ALLOWED_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def max_images_per_feedback() -> int:
    return _MAX_IMAGES_PER_FEEDBACK


def build_media_url(key: str, *, admin: bool = False) -> str:
    prefix = _ADMIN_PREFIX if admin else _PUBLIC_PREFIX
    return f"{prefix}/{key}"


def media_path(key: str) -> Path:
    if not _KEY_RE.fullmatch(key.strip()):
        raise HTTPException(status_code=404, detail="Feedback media not found")
    return _FEEDBACK_DIR / key.strip()


def normalize_image_mime(mime: str | None) -> str:
    normalized = (mime or "image/jpeg").strip().lower()
    if normalized == "image/jpg":
        return "image/jpeg"
    if normalized not in _ALLOWED_MIMES:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    return normalized


def validate_image_size(blob: bytes) -> None:
    if len(blob) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image too large")


def save_feedback_image(*, user_id: str, blob: bytes, mime: str | None) -> str:
    if not blob:
        raise HTTPException(status_code=400, detail="Empty image")
    validate_image_size(blob)
    normalized_mime = normalize_image_mime(mime)
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(blob)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc
    image = image.convert("RGB")
    image.thumbnail((_INGEST_MAX_EDGE, _INGEST_MAX_EDGE), Image.Resampling.LANCZOS)
    _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{user_id}_{uuid.uuid4().hex}.jpg"
    out = _FEEDBACK_DIR / key
    image.save(out, format="JPEG", quality=_INGEST_JPEG_QUALITY, optimize=True)
    return key


def delete_feedback_image(key: str) -> None:
    try:
        path = media_path(key)
    except HTTPException:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def serve_media(key: str) -> FileResponse:
    path = media_path(key)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Feedback media not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )
