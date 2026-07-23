from __future__ import annotations

import base64
import hashlib
import io
import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
import uuid

from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MEDIA_DIR = Path(os.getenv("CHAT_MEDIA_DIR", "var/chat_media"))
_MEDIA_PUBLIC_PREFIX = (
    os.getenv("CHAT_MEDIA_PUBLIC_PREFIX", "/chat/media").strip().rstrip("/")
    or "/chat/media"
)
# Chat display never needs more than ~2K pixels; H5/mini clients historically
# uploaded raw camera files (4-8MB observed in production), so oversized
# originals are normalized at ingest. Vision/LLM flows read the stored file
# and work fine on the normalized variant.
_INGEST_MAX_EDGE = 2048
_INGEST_JPEG_QUALITY = 85
_INGEST_REENCODE_MIN_BYTES = 1_500_000
# Bubble-sized thumbnail variant, served via GET /chat/media/{key}?v=thumb.
_THUMB_MAX_EDGE = 480
_THUMB_JPEG_QUALITY = 72
_THUMB_KEY_SUFFIX = "_t.jpg"
# Media keys embed an immutable uuid4 hex, so responses can be cached forever.
_IMMUTABLE_CACHE_CONTROL = "private, max-age=31536000, immutable"
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


@dataclass(frozen=True)
class ProcessedImage:
    """An ingest-normalized image ready to be stored."""

    blob: bytes
    mime: str
    width: int | None
    height: int | None


def process_image_upload(blob: bytes, mime: str) -> ProcessedImage:
    """Normalize an uploaded image for storage.

    - applies EXIF orientation (photos otherwise render rotated after strip);
    - caps oversized originals to <=_INGEST_MAX_EDGE / re-encodes heavyweight
      files (chat display needs nothing bigger);
    - always reports the real pixel dimensions (H5 uploads used to claim 0x0).

    Undecodable payloads fall back to storing the raw bytes untouched, keeping
    the historical lenient behaviour.
    """
    try:
        image = Image.open(io.BytesIO(blob))
        image.load()
    except Exception:
        logger.warning("[chat-media] image decode failed; storing raw blob")
        return ProcessedImage(blob=blob, mime=mime, width=None, height=None)
    oriented = ImageOps.exif_transpose(image) or image
    width, height = oriented.size
    needs_resize = max(width, height) > _INGEST_MAX_EDGE
    needs_reencode = needs_resize or len(blob) > _INGEST_REENCODE_MIN_BYTES
    if not needs_reencode:
        return ProcessedImage(blob=blob, mime=mime, width=width, height=height)
    if needs_resize:
        oriented.thumbnail(
            (_INGEST_MAX_EDGE, _INGEST_MAX_EDGE), Image.Resampling.LANCZOS
        )
    keep_alpha = mime == "image/png" and "A" in oriented.getbands()
    buffer = io.BytesIO()
    try:
        if keep_alpha:
            oriented.save(buffer, format="PNG", optimize=True)
            out_mime = "image/png"
        else:
            oriented.convert("RGB").save(
                buffer,
                format="JPEG",
                quality=_INGEST_JPEG_QUALITY,
                optimize=True,
            )
            out_mime = "image/jpeg"
    except Exception:
        logger.warning("[chat-media] image re-encode failed; storing raw blob")
        return ProcessedImage(blob=blob, mime=mime, width=width, height=height)
    out = buffer.getvalue()
    # Re-encoding a small-but-heavy file can occasionally grow it; keep the
    # smaller representation.
    if not needs_resize and len(out) >= len(blob):
        return ProcessedImage(blob=blob, mime=mime, width=width, height=height)
    final = Image.open(io.BytesIO(out))
    return ProcessedImage(
        blob=out, mime=out_mime, width=final.size[0], height=final.size[1]
    )


def generate_thumbnail_blob(blob: bytes) -> bytes | None:
    """Bubble-sized JPEG thumbnail (<=_THUMB_MAX_EDGE). None when not an image."""
    try:
        image = Image.open(io.BytesIO(blob))
        image.load()
        oriented = ImageOps.exif_transpose(image) or image
        oriented.thumbnail(
            (_THUMB_MAX_EDGE, _THUMB_MAX_EDGE), Image.Resampling.LANCZOS
        )
        if "A" in oriented.getbands():
            # Flatten transparency onto white: thumbnails are always JPEG.
            background = Image.new("RGB", oriented.size, (255, 255, 255))
            background.paste(oriented, mask=oriented.getchannel("A"))
            oriented = background
        else:
            oriented = oriented.convert("RGB")
        buffer = io.BytesIO()
        oriented.save(
            buffer, format="JPEG", quality=_THUMB_JPEG_QUALITY, optimize=True
        )
        return buffer.getvalue()
    except Exception:
        logger.warning("[chat-media] thumbnail generation failed", exc_info=True)
        return None


def thumb_storage_key(storage_key: str) -> str:
    """Sibling thumbnail key: `{stem}_t.jpg` (keeps the `{user_id}_` auth prefix;
    original keys end in a 32-char uuid hex so the suffix cannot collide)."""
    return f"{Path(storage_key).stem}{_THUMB_KEY_SUFFIX}"


def save_image_with_thumbnail(
    *,
    user_id: str,
    conversation_id: str,
    blob: bytes,
    mime: str,
) -> tuple[str, ProcessedImage]:
    """Chat-image ingest: normalize, store the original, store a thumbnail."""
    processed = process_image_upload(blob, mime)
    validate_image_size(processed.blob)
    _MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    key = storage_key_for(user_id, processed.mime, conversation_id=conversation_id)
    storage_path(key).write_bytes(processed.blob)
    thumb = generate_thumbnail_blob(processed.blob)
    if thumb is not None:
        storage_path(thumb_storage_key(key)).write_bytes(thumb)
    return key, processed


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
    # Remove the thumbnail sibling too (no-op for audio / legacy files).
    thumb_path = storage_path(thumb_storage_key(storage_key))
    if thumb_path.exists() and thumb_path.is_file():
        thumb_path.unlink()


def serve_media(
    storage_key: str,
    *,
    user_id: str,
    is_admin: bool = False,
    variant: str | None = None,
) -> FileResponse:
    """Serve a stored media file.

    `variant="thumb"` serves the bubble-sized thumbnail when one exists and
    silently falls back to the original (audio files, media uploaded before
    thumbnails existed and not yet backfilled). Authorization is always checked
    against the ORIGINAL key's `{user_id}_` prefix, which the thumbnail key
    shares by construction.
    """
    if not is_admin and not storage_key.startswith(f"{user_id}_"):
        raise HTTPException(status_code=403, detail="Not your media")
    path = storage_path(storage_key)
    if variant == "thumb":
        thumb_path = storage_path(thumb_storage_key(storage_key))
        if thumb_path.exists() and thumb_path.is_file():
            path = thumb_path
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    explicit_type = "audio/mp4" if path.suffix.lower() == ".m4a" else None
    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path,
        media_type=explicit_type or media_type or "application/octet-stream",
        # Keys embed an immutable uuid: safe to cache client-side forever.
        # This is what lets Flutter's disk cache and the browser HTTP cache
        # skip re-downloads across app/page restarts.
        headers={"Cache-Control": _IMMUTABLE_CACHE_CONTROL},
    )
