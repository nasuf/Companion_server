from __future__ import annotations

import logging

import httpx

from app.services.offline import activity_media_storage
from app.services.offline.providers.search import SearchResult, tavily_image_search

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_S = 8.0


def _image_queries(card: dict, city: str) -> list[str]:
    location = str(card.get("location_name") or card.get("address") or "").strip()
    title = str(card.get("title") or "").strip()
    category = str(card.get("category") or "").strip()
    queries: list[str] = []
    if location:
        queries.append(f"{city} {location} 现场 图片 官方")
        queries.append(f"{city} {location} 实景图")
    if title:
        queries.append(f"{city} {title} 图片")
    fallback = " ".join(part for part in [city, category or "活动", "现场 图片"] if part)
    if fallback:
        queries.append(fallback)
    deduped: list[str] = []
    for query in queries:
        if query and query not in deduped:
            deduped.append(query)
    return deduped[:4]


def _remote_image_candidates(
    card: dict,
    search_results: list[SearchResult],
) -> list[str]:
    candidates: list[str] = []
    for value in card.get("image_urls") or []:
        text = str(value).strip()
        if text.startswith(("http://", "https://")):
            candidates.append(text)
    for result in search_results:
        if result.image_url:
            candidates.append(result.image_url)
    deduped: list[str] = []
    for image_url in candidates:
        if image_url not in deduped:
            deduped.append(image_url)
    return deduped


async def persist_activity_images(
    *,
    user_id: str,
    card: dict,
    city: str,
    search_results: list[SearchResult],
    limit: int = 3,
) -> list[str]:
    candidates: list[str] = []
    for query in _image_queries(card, city):
        candidates.extend(await tavily_image_search(query, max_results=6))
        if len(candidates) >= limit * 2:
            break
    candidates.extend(_remote_image_candidates(card, search_results))

    persisted: list[str] = []
    seen: set[str] = set()
    for image_url in candidates:
        if image_url in seen:
            continue
        seen.add(image_url)
        local_url = await _download_and_store_image(user_id=user_id, url=image_url)
        if local_url:
            persisted.append(local_url)
        if len(persisted) >= limit:
            break
    return persisted


async def _download_and_store_image(*, user_id: str, url: str) -> str | None:
    try:
        async with httpx.AsyncClient(
            timeout=_DOWNLOAD_TIMEOUT_S,
            follow_redirects=True,
            trust_env=False,
        ) as client:
            response = await client.get(url, headers={"accept": "image/*"})
            response.raise_for_status()
    except Exception as exc:
        logger.debug("[offline] activity image download failed url=%s err=%s", url, exc)
        return None

    mime = response.headers.get("content-type", "").split(";")[0].strip().lower()
    try:
        mime = activity_media_storage.normalize_image_mime(mime)
    except Exception:
        return None
    blob = response.content
    try:
        storage_key = activity_media_storage.save_image_blob(
            user_id=user_id,
            blob=blob,
            mime=mime,
        )
    except Exception as exc:
        logger.debug("[offline] activity image store failed url=%s err=%s", url, exc)
        return None
    return activity_media_storage.media_url(storage_key)
