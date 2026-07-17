from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from urllib.parse import urlparse

import httpx

from app.services.chat_links.extraction import LinkMetadata
from app.services.chat_media import storage

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(8.0, connect=4.0, read=6.0, write=4.0, pool=4.0)
_MAX_COVER_BYTES = 10 * 1024 * 1024
_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Companion/0.1"
)


@dataclass(frozen=True)
class CachedCoverResult:
    metadata: LinkMetadata
    extra_metadata: dict[str, str]


async def cache_link_cover(
    *,
    user_id: str,
    conversation_id: str | None = None,
    metadata: LinkMetadata,
) -> CachedCoverResult:
    """Cache a remote link cover through authenticated chat media storage.

    External platforms often use hotlink-protected image URLs. Keeping a local
    copy makes chat cards and the Daily Share link tab render consistently.
    Failures are non-fatal: callers still keep the remote image URL.
    """
    remote_url = (metadata.image_url or "").strip()
    if not _should_cache(remote_url):
        return CachedCoverResult(metadata=metadata, extra_metadata={})
    try:
        blob, mime = await _download_image(remote_url, referer_url=metadata.final_url or metadata.source_url)
        storage_key = storage.save_image_blob(
            user_id=user_id,
            conversation_id=conversation_id,
            blob=blob,
            mime=mime,
        )
        cached_url = storage.media_url(storage_key)
    except Exception as exc:
        logger.info("[chat-links] cover cache skipped url=%s error=%s", remote_url[:180], exc)
        return CachedCoverResult(metadata=metadata, extra_metadata={"remote_image_url": remote_url})
    return CachedCoverResult(
        metadata=replace(metadata, image_url=cached_url),
        extra_metadata={
            "remote_image_url": remote_url,
            "cover_storage_key": storage_key,
            "cover_cached_url": cached_url,
        },
    )


def _should_cache(raw_url: str) -> bool:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return True


async def _download_image(url: str, referer_url: str | None = None) -> tuple[bytes, str]:
    headers = {
        "user-agent": _USER_AGENT,
        "accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "referer": _origin(referer_url or url),
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > _MAX_COVER_BYTES:
            raise ValueError("cover image is too large")
        blob = response.content
    storage.validate_image_size(blob)
    mime = storage.normalize_image_mime(_response_mime(response))
    return blob, mime


def _response_mime(response: httpx.Response) -> str:
    content_type = response.headers.get("content-type") or ""
    return content_type.split(";", 1)[0].strip().lower()


def _origin(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/"
