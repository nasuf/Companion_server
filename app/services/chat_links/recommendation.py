from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings
from app.services.chat_links.cards import component_card_for_link, metadata_for_link_card
from app.services.chat_links.extraction import (
    extract_first_url,
    extract_link_metadata,
    platform_for_url,
)
from app.services.chat_links.repo import ChatLinkCard, create_or_update_link_card

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ("小红书", "微博", "今日头条", "抖音", "知乎", "B站")
_SEARCH_DOMAINS = (
    "xhslink.com",
    "xiaohongshu.com",
    "weibo.com",
    "weibo.cn",
    "toutiao.com",
    "douyin.com",
    "zhihu.com",
    "bilibili.com",
    "b23.tv",
)


@dataclass(frozen=True)
class ProactiveLinkRecommendation:
    link: ChatLinkCard
    component_card: dict[str, Any]
    link_card_metadata: dict[str, Any]


@dataclass(frozen=True)
class _CandidateUrl:
    url: str
    source: str


def configured_candidate_urls(raw: str | None = None) -> list[str]:
    """Parse operator-provided candidate URLs without accepting arbitrary text."""
    source = settings.proactive_link_candidate_urls if raw is None else raw
    urls: list[str] = []
    for chunk in re.split(r"[\s,;，；]+", source or ""):
        url = extract_first_url(chunk) or chunk.strip()
        if not url:
            continue
        if platform_for_url(url) in SUPPORTED_PLATFORMS and url not in urls:
            urls.append(url)
    return urls


def configured_search_provider() -> str:
    provider = (settings.chat_link_search_provider or "custom").strip().lower()
    return provider or "custom"


def search_provider_configured() -> tuple[bool, str]:
    provider = configured_search_provider()
    if provider in {"custom", "endpoint"}:
        if settings.chat_link_search_endpoint.strip():
            return True, "custom endpoint configured"
        return False, "CHAT_LINK_SEARCH_ENDPOINT is not set"
    if provider == "tavily":
        if settings.tavily_api_key.strip():
            return True, "TAVILY_API_KEY configured"
        return False, "TAVILY_API_KEY is not set"
    if provider == "brave":
        if settings.brave_search_api_key.strip():
            return True, "BRAVE_SEARCH_API_KEY configured"
        return False, "BRAVE_SEARCH_API_KEY is not set"
    return False, f"unknown CHAT_LINK_SEARCH_PROVIDER={provider}"


def should_attempt_proactive_link(
    *,
    trigger_type: str,
    source: str,
    random_value: float | None = None,
) -> bool:
    if not settings.proactive_link_recommendation_enabled:
        return False
    if trigger_type not in {"silence_wakeup", "memory_proactive"}:
        return False
    if source == "music":
        return False
    probability = max(0.0, min(float(settings.proactive_link_recommendation_probability), 0.20))
    if probability <= 0:
        return False
    roll = random.random() if random_value is None else random_value
    return roll < probability


async def maybe_prepare_proactive_link_recommendation(
    *,
    user_id: str,
    conversation_id: str,
    trigger_type: str,
    source: str,
    topic: str | None,
    stage: str | None,
    message: str,
) -> ProactiveLinkRecommendation | None:
    """Return a real assistant link card when a configured provider yields one.

    The agent must never hallucinate a card URL. This helper only emits a card
    after a candidate URL has been found, parsed, and stored as role=assistant.
    """
    if not should_attempt_proactive_link(trigger_type=trigger_type, source=source):
        return None
    candidate = await _select_candidate_url(query=_query(topic=topic, message=message))
    if not candidate:
        return None
    try:
        metadata = await extract_link_metadata(url=candidate.url, shared_text=message)
        link = await create_or_update_link_card(
            user_id=user_id,
            conversation_id=conversation_id,
            metadata=metadata,
            role="assistant",
            source_app="proactive_link_recommendation",
            extra_metadata={
                "trigger_type": trigger_type,
                "topic": topic,
                "stage": stage,
                "candidate_source": candidate.source,
            },
        )
    except Exception as exc:
        logger.warning("[chat-links] proactive recommendation failed: %s", exc)
        return None
    return ProactiveLinkRecommendation(
        link=link,
        component_card=component_card_for_link(link),
        link_card_metadata=metadata_for_link_card(link),
    )


async def _select_candidate_url(*, query: str) -> _CandidateUrl | None:
    urls = await _search_endpoint_urls(query=query)
    if urls:
        return _CandidateUrl(url=random.choice(urls), source="search_endpoint")
    fallback_urls = configured_candidate_urls()
    if fallback_urls:
        return _CandidateUrl(url=random.choice(fallback_urls), source="configured_pool")
    return None


async def _search_endpoint_urls(*, query: str) -> list[str]:
    provider = configured_search_provider()
    if provider in {"", "custom", "endpoint"}:
        return await _custom_endpoint_urls(query=query)
    if provider == "tavily":
        return await _tavily_search_urls(query=query)
    if provider == "brave":
        return await _brave_search_urls(query=query)
    logger.warning("[chat-links] unknown search provider=%s", provider)
    return []


async def _custom_endpoint_urls(*, query: str) -> list[str]:
    endpoint = settings.chat_link_search_endpoint.strip()
    if not endpoint:
        return []
    headers = {"accept": "application/json"}
    if settings.chat_link_search_api_key.strip():
        headers["authorization"] = f"Bearer {settings.chat_link_search_api_key.strip()}"
    payload = {
        "query": query,
        "platforms": list(SUPPORTED_PLATFORMS),
        "limit": 5,
    }
    try:
        async with httpx.AsyncClient(
            timeout=settings.chat_link_search_timeout_s,
            headers=headers,
            trust_env=False,
        ) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("[chat-links] search endpoint failed: %s", exc)
        return []
    return _urls_from_search_response(data)


async def _tavily_search_urls(*, query: str) -> list[str]:
    api_key = settings.tavily_api_key.strip()
    endpoint = settings.tavily_search_endpoint.strip()
    if not api_key or not endpoint:
        return []
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    payload = {
        "query": _clean_search_query(query),
        "max_results": 8,
        "include_answer": False,
        "include_raw_content": False,
        "include_domains": list(_SEARCH_DOMAINS),
    }
    try:
        async with httpx.AsyncClient(
            timeout=settings.chat_link_search_timeout_s,
            headers=headers,
            trust_env=False,
        ) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("[chat-links] tavily search failed: %s", exc)
        return []
    return _urls_from_search_response(data)


async def _brave_search_urls(*, query: str) -> list[str]:
    api_key = settings.brave_search_api_key.strip()
    endpoint = settings.brave_search_endpoint.strip()
    if not api_key or not endpoint:
        return []
    headers = {
        "accept": "application/json",
        "accept-encoding": "gzip",
        "x-subscription-token": api_key,
    }
    params = {
        "q": _site_scoped_query(query),
        "count": 8,
        "safesearch": "moderate",
    }
    try:
        async with httpx.AsyncClient(
            timeout=settings.chat_link_search_timeout_s,
            headers=headers,
            trust_env=False,
        ) as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("[chat-links] brave search failed: %s", exc)
        return []
    return _urls_from_search_response(data)


def _urls_from_search_response(data: Any) -> list[str]:
    raw_results = data.get("results") if isinstance(data, dict) else data
    if not raw_results and isinstance(data, dict):
        web = data.get("web")
        if isinstance(web, dict):
            raw_results = web.get("results")
    if not isinstance(raw_results, list):
        return []
    urls: list[str] = []
    for item in raw_results:
        raw_url = ""
        if isinstance(item, dict):
            raw_url = str(item.get("url") or item.get("link") or item.get("source_url") or "")
        elif isinstance(item, str):
            raw_url = item
        url = extract_first_url(raw_url) or raw_url.strip()
        if platform_for_url(url) in SUPPORTED_PLATFORMS and url not in urls:
            urls.append(url)
    return urls


def _site_scoped_query(query: str) -> str:
    cleaned = _clean_search_query(query)
    site_clause = " OR ".join(f"site:{domain}" for domain in _SEARCH_DOMAINS)
    return f"{cleaned} ({site_clause})"


def _clean_search_query(query: str) -> str:
    return " ".join((query or "").split()).strip()[:160] or "日常分享"


def _query(*, topic: str | None, message: str) -> str:
    topic_text = (topic or "").strip()
    if topic_text:
        return topic_text[:80]
    return (message or "日常分享").strip()[:80] or "日常分享"
