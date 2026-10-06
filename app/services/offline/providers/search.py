from __future__ import annotations

import logging
import json
import hashlib
from dataclasses import asdict, replace
from app.redis_client import get_redis
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str
    score: float | None = None
    image_url: str | None = None
    images: list[dict[str, str]] = field(default_factory=list)
    raw_content: str = ''
    # Query-level image search results are independent documents, not photos
    # belonging to this page. Keep them separate through filtering/cache reads.
    query_images: list[dict[str, str]] = field(default_factory=list)


def _result_from_item(item: Any) -> SearchResult | None:
    if not isinstance(item, dict):
        return None
    url = str(item.get("url") or "").strip()
    if not url:
        return None
    title = str(item.get("title") or item.get("name") or url).strip()
    content = str(item.get("content") or item.get("snippet") or item.get("description") or "").strip()
    raw_score = item.get("score")
    try:
        score = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError):
        score = None
    image_url = item.get("image_url") or item.get("thumbnail") or item.get("image")
    return SearchResult(
        title=title[:160],
        url=url,
        content=content[:800],
        score=score,
        image_url=str(image_url).strip() if image_url else None,
        raw_content=str(item.get('raw_content') or '')[:50000],
        images=[{"url": url, "source_url": str(item.get("url") or ""),
                 "source_title": title, "description": str(img.get("description") or "") if isinstance(img, dict) else "",
                 "description_source": str(img.get("description_source") or "") if isinstance(img, dict) else ""}
                for img in (item.get("images") or []) if (url := _image_url_from_item(img))],
    )


async def tavily_search(
    query: str,
    *,
    max_results: int = 6,
    include_domains: list[str] | None = None,
    timeout_s: float = 10.0,
    image_evidence: bool = False,
) -> list[SearchResult]:
    api_key = settings.tavily_api_key.strip()
    endpoint = settings.tavily_search_endpoint.strip()
    if not api_key or not endpoint:
        return []
    payload = {
        "query": " ".join(query.split())[:380],
        "max_results": max(1, min(max_results, 10)),
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": "markdown" if image_evidence else False,
        "include_image_descriptions": image_evidence,
        "include_images": True,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    cache_key = 'offline:search:v4:' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    try:
        cached = await (await get_redis()).get(cache_key)
        if cached:
            return [SearchResult(**item) for item in json.loads(cached)]
    except Exception:
        pass
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    try:
        async with httpx.AsyncClient(
            timeout=timeout_s,
            headers=headers,
            trust_env=False,
        ) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("[offline] tavily search failed: %s", exc)
        return []

    if not isinstance(data, dict):
        return []
    raw_results = data.get("results") or []
    if not isinstance(raw_results, list):
        return []
    results: list[SearchResult] = []
    for item in raw_results:
        result = _result_from_item(item)
        if result and result.url not in {r.url for r in results}:
            results.append(result)
    query_images = []
    raw_images = data.get('images') or []
    for item in (raw_images[:20] if isinstance(raw_images, list) else []):
        url = _image_url_from_item(item)
        if not url:
            continue
        meta = item if isinstance(item, dict) else {}
        query_images.append(dict(url=url, source_title=str(meta.get('title') or '')[:200],
                                 source_url=str(meta.get('source_url') or '')[:2048],
                                 description=str(meta.get('description') or '')[:1200], query=payload['query']))
    if query_images:
        results = [replace(r, query_images=query_images) for r in results]
        if not results and image_evidence:
            # Images-only responses remain usable for gallery refill. An empty
            # page URL cannot become a source-backed activity destination.
            results = [SearchResult(title='', url='', content='', query_images=query_images)]
    if results:
        try:
            await (await get_redis()).set(cache_key, json.dumps([asdict(r) for r in results]), ex=3600)
        except Exception:
            pass
    return results


def _image_url_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        value = item.strip()
        return value if value.startswith(("http://", "https://")) else None
    if not isinstance(item, dict):
        return None
    value = item.get("url") or item.get("image_url") or item.get("src")
    if not value:
        return None
    text = str(value).strip()
    return text if text.startswith(("http://", "https://")) else None


async def tavily_image_search(
    query: str,
    *,
    max_results: int = 8,
    timeout_s: float = 10.0,
) -> list[str]:
    api_key = settings.tavily_api_key.strip()
    endpoint = settings.tavily_search_endpoint.strip()
    if not api_key or not endpoint:
        return []
    payload = {
        "query": " ".join(query.split())[:380],
        "max_results": max(1, min(max_results, 10)),
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": True,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s, headers=headers, trust_env=False) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("[offline] tavily image search failed: %s", exc)
        return []

    candidates: list[str] = []
    if isinstance(data, dict):
        for item in data.get("images") or []:
            image_url = _image_url_from_item(item)
            if image_url:
                candidates.append(image_url)
        for item in data.get("results") or []:
            result = _result_from_item(item)
            if result and result.image_url:
                candidates.append(result.image_url)
    deduped: list[str] = []
    for image_url in candidates:
        if image_url not in deduped:
            deduped.append(image_url)
    return deduped[:max_results]


async def tavily_place_images(query: str) -> list[SearchResult]:
    """Return page photos and separate image-search documents with their evidence."""
    return await tavily_search(query, max_results=8, image_evidence=True, timeout_s=15)
