from __future__ import annotations

import logging
from dataclasses import dataclass
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
    )


async def tavily_search(
    query: str,
    *,
    max_results: int = 6,
    include_domains: list[str] | None = None,
    timeout_s: float = 10.0,
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
        "include_raw_content": False,
        "include_images": True,
    }
    if include_domains:
        payload["include_domains"] = include_domains
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
        logger.warning("[offline] tavily search failed: %s", exc)
        return []

    raw_results = data.get("results") if isinstance(data, dict) else []
    if not isinstance(raw_results, list):
        return []
    results: list[SearchResult] = []
    for item in raw_results:
        result = _result_from_item(item)
        if result and result.url not in {r.url for r in results}:
            results.append(result)
    return results
