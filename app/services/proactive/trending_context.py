"""Fetch and cache public trending snippets for proactive messages."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings
from app.observability.events import EVT_PROACTIVE_TRENDING
from app.services.offline.providers.search import SearchResult, tavily_search
from app.services.proactive.admin_test import AdminProactiveTestOptions
from app.services.proactive.trending_cache import (
    cache_key_for_topic,
    get_cached_trending,
    set_cached_trending,
)
from app.services.proactive.trending_gate import (
    is_trending_eligible_trigger,
    should_attach_trending,
)

logger = logging.getLogger(__name__)

_LIVE_QUERIES = (
    "微博热搜 今日",
    "今日热点新闻",
)
_FETCH_BUDGET_S = 8.0


@dataclass(frozen=True)
class TrendingLoadResult:
    text: str
    cache_hit: bool = False
    provider: str | None = None


def _format_search_results(results: list[SearchResult], *, limit: int = 4) -> str:
    lines: list[str] = []
    for item in results[:limit]:
        snippet = (item.content or item.title or "").strip()
        title = (item.title or "").strip()
        if not snippet and not title:
            continue
        if title and snippet and title not in snippet:
            lines.append(f"- {title}: {snippet[:140]}")
        else:
            lines.append(f"- {snippet[:160]}")
    return "\n".join(lines)


def _results_from_brave_payload(data: Any) -> list[SearchResult]:
    raw_results = data.get("results") if isinstance(data, dict) else data
    if not raw_results and isinstance(data, dict):
        web = data.get("web")
        if isinstance(web, dict):
            raw_results = web.get("results")
    if not isinstance(raw_results, list):
        return []

    parsed: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or "").strip()
        title = str(item.get("title") or item.get("name") or url).strip()
        content = str(
            item.get("description")
            or item.get("snippet")
            or item.get("extra_snippets")
            or ""
        ).strip()
        if not title and not content:
            continue
        parsed.append(SearchResult(title=title[:160], url=url, content=content[:800]))
    return parsed


async def _tavily_snippets(query: str) -> str:
    if not settings.tavily_api_key.strip():
        return ""
    timeout = float(getattr(settings, "chat_link_search_timeout_s", 8.0))
    try:
        results = await tavily_search(query, max_results=5, timeout_s=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[proactive-trending] tavily failed query=%s: %s", query, exc)
        return ""
    return _format_search_results(results)


async def _brave_snippets(query: str) -> str:
    api_key = settings.brave_search_api_key.strip()
    endpoint = settings.brave_search_endpoint.strip()
    if not api_key or not endpoint:
        return ""
    headers = {
        "accept": "application/json",
        "accept-encoding": "gzip",
        "x-subscription-token": api_key,
    }
    params = {
        "q": " ".join(query.split())[:160],
        "count": 6,
        "safesearch": "moderate",
    }
    timeout = float(getattr(settings, "chat_link_search_timeout_s", 8.0))
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, trust_env=False) as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[proactive-trending] brave failed query=%s: %s", query, exc)
        return ""
    return _format_search_results(_results_from_brave_payload(data))


async def _fetch_live_trending_snippets(*, topic: str | None = None) -> TrendingLoadResult:
    seed = (topic or "").strip()
    queries: list[str] = list(_LIVE_QUERIES)
    if seed and seed not in {"公共话题", "问候", "日常"}:
        queries.append(f"微博热搜 今日 {seed}")
        queries.append(f"今日热点新闻 {seed}")

    for query in queries:
        text = await _tavily_snippets(query)
        if text:
            return TrendingLoadResult(text=text, cache_hit=False, provider="tavily")
        text = await _brave_snippets(query)
        if text:
            return TrendingLoadResult(text=text, cache_hit=False, provider="brave")
    return TrendingLoadResult(text="", cache_hit=False, provider=None)


async def load_trending_context(
    *,
    topic: str | None = None,
    bypass_cache: bool = False,
) -> TrendingLoadResult:
    """Load trending bullets, preferring Redis cache for generic topics."""
    key = cache_key_for_topic(topic)
    if not bypass_cache:
        cached = await get_cached_trending(key)
        if cached:
            return TrendingLoadResult(text=cached, cache_hit=True, provider="cache")

    loaded = await _fetch_live_trending_snippets(topic=topic)
    if loaded.text and not bypass_cache:
        await set_cached_trending(key, loaded.text)
    return loaded


async def append_trending_section(prompt: str, trending_context: str) -> str:
    cleaned = (trending_context or "").strip()
    if not cleaned:
        return prompt
    try:
        from app.services.prompting.store import get_prompt_text

        tpl = await get_prompt_text("proactive.trending_section")
        section = tpl.format(trending=cleaned)
    except Exception as exc:  # noqa: BLE001 — disabled/missing template falls back inline
        logger.debug("[proactive-trending] prompt template unavailable: %s", exc)
        section = (
            f"\n\n【今日热点参考（联网检索）】\n{cleaned}\n"
            "如合适可自然带一句公共话题，不要像新闻播报，也不要说「我刚搜了下」。"
        )
    return f"{prompt.rstrip()}{section}"


async def resolve_trending_context(
    trigger_type: str,
    *,
    topic: str | None,
    admin_test_options: AdminProactiveTestOptions | None = None,
) -> tuple[str, bool, TrendingLoadResult | None]:
    """Decide whether to search and return (context_text, attached, load_meta)."""
    if not is_trending_eligible_trigger(trigger_type):
        return "", False, None

    if admin_test_options is not None:
        if not admin_test_options.use_web_search:
            return "", False, None
    elif not should_attach_trending(trigger_type):
        return "", False, None

    bypass_cache = admin_test_options is not None and admin_test_options.use_web_search
    try:
        loaded = await asyncio.wait_for(
            load_trending_context(topic=topic, bypass_cache=bypass_cache),
            timeout=_FETCH_BUDGET_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[proactive-trending] fetch budget exceeded trigger=%s topic=%s",
            trigger_type,
            topic or "",
        )
        return "", False, None
    attached = bool(loaded.text.strip())
    if attached:
        logger.info(
            "[proactive-trending] attached trigger=%s cache_hit=%s provider=%s",
            trigger_type,
            loaded.cache_hit,
            loaded.provider,
            extra={
                "event": EVT_PROACTIVE_TRENDING,
                "trigger_type": trigger_type,
                "cache_hit": loaded.cache_hit,
                "provider": loaded.provider or "",
                "snippet_lines": loaded.text.count("\n") + 1,
            },
        )
    return loaded.text, attached, loaded
