from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline.providers.search import SearchResult, tavily_search
from app.services.offline import repository as repo
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


_FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee",
    "https://images.unsplash.com/photo-1492684223066-81342ee5ff30",
    "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
]

_LOCALIZED_CITY_ALIASES = {
    "zhenjiang": ("江苏 镇江", "镇江", "Zhenjiang"),
    "jiangsu": ("江苏", "Jiangsu"),
}

_UNRELIABLE_ACTIVITY_DOMAINS = {
    "tripadvisor.com",
    "calendar.yahoo.com",
}

_GENERIC_ACTIVITY_TITLE_RE = re.compile(
    r"\b(the best|things to do|free things|attractions|travel guide|calendar)\b",
    flags=re.I,
)


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _localized_city_terms(city: str) -> tuple[str, ...]:
    key = city.strip().lower()
    return _LOCALIZED_CITY_ALIASES.get(key, (city.strip(),))


def _search_query(city: str, tags: list[str]) -> str:
    tag_text = " ".join(tags[:5])
    anchor = " ".join(term for term in _localized_city_terms(city) if term)
    return (
        f"{anchor or city} 近期 本周 周末 官方 活动公告 免费 低成本 小众 "
        f"展览 市集 音乐 书店 咖啡 公园 {tag_text}"
    ).strip()


def _sources(results: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "title": result.title,
            "url": result.url,
            "content": result.content,
            "score": result.score,
            "image_url": result.image_url,
        }
        for result in results[:6]
    ]


def _domain(url: str | None) -> str:
    if not url:
        return ""
    host = urlparse(str(url)).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _source_is_usable(result: SearchResult, city: str) -> bool:
    domain = _domain(result.url)
    if any(
        domain == bad or domain.endswith(f".{bad}")
        for bad in _UNRELIABLE_ACTIVITY_DOMAINS
    ):
        return False
    title = result.title.strip()
    content = result.content.strip()
    combined = f"{title}\n{content}"
    if not title or _GENERIC_ACTIVITY_TITLE_RE.search(title):
        return False
    if "No information is available for this page" in content:
        return False
    city_terms = [term for term in _localized_city_terms(city) if term]
    if city_terms and not any(term.lower() in combined.lower() for term in city_terms):
        return False
    return True


def _usable_results(results: list[SearchResult], city: str) -> list[SearchResult]:
    return [result for result in results if _source_is_usable(result, city)]


def _card_is_source_backed(card: dict[str, Any], sources: list[dict[str, Any]]) -> bool:
    if not sources:
        return True
    source_domains = {_domain(source.get("url")) for source in sources}
    source_domains.discard("")
    official_domain = _domain(card.get("official_url"))
    if not official_domain:
        return False
    return any(
        official_domain == domain or official_domain.endswith(f".{domain}")
        for domain in source_domains
    )


def _fallback_card(city: str, tags: list[str], results: list[SearchResult]) -> dict[str, Any]:
    localized_terms = _localized_city_terms(city)
    city_label = localized_terms[1] if len(localized_terms) > 1 else city
    title = f"{city_label}轻松散步小计划"
    task = "到现场后，拍下一处你觉得有一点可爱的细节，发给我看看。"
    return {
        "title": title,
        "summary": "一个低压力、可以独立完成的小出门计划。",
        "description": (
            f"我帮你在{city_label}附近保留了一个轻量出门计划。"
            "不需要社交表现，也不用赶时间，就当给今天换一点空气。"
        ),
        "category": tags[0] if tags else "城市漫游",
        "location_name": city_label,
        "address": city_label,
        "starts_at": None,
        "ends_at": None,
        "official_url": results[0].url if results else None,
        "image_urls": [r.image_url for r in results if r.image_url][:3]
        or _FALLBACK_IMAGES[:2],
        "task_hint": "接受后解锁一个小彩蛋任务",
        "easter_egg_task": {
            "title": "秘密彩蛋任务",
            "body": task,
            "principle": "低社交压力、可独立完成、无安全风险",
        },
        "metadata": {"fallback": True},
    }


async def generate_activity_card(
    *,
    user_id: str,
    workspace_id: str | None,
    city: str,
    source: str,
    search_location: str | None = None,
) -> dict[str, Any]:
    tags = await repo.list_user_tags(user_id, workspace_id, limit=9)
    memory = await repo.memory_brief(user_id, workspace_id, limit=60)
    query = _search_query(search_location or city, tags)
    raw_results = await tavily_search(query, max_results=8)
    results = _usable_results(raw_results, city)[:6]
    sources = _sources(results)
    card: dict[str, Any] | None = None
    if sources:
        try:
            prompt_text = (await get_prompt_text("offline.activity_card")).format(
                city=city,
                search_anchor=search_location or city,
                tags=", ".join(tags) if tags else "暂无",
                memory=memory or "暂无足够记忆，使用城市热门和季节普适活动兜底。",
                sources_json=json.dumps(sources, ensure_ascii=False),
            )
            raw = await invoke_text(get_chat_model(), prompt_text)
            card = _json_object(raw)
        except Exception as exc:
            logger.warning("[offline] activity LLM generation failed: %s", exc)
    if card and not _card_is_source_backed(card, sources):
        logger.warning(
            "[offline] discarded unbacked activity card title=%r official_url=%r query=%r",
            card.get("title"),
            card.get("official_url"),
            query,
        )
        card = None
    if not card:
        card = _fallback_card(city, tags, results)

    now = datetime.now(UTC)
    image_urls = card.get("image_urls")
    if not isinstance(image_urls, list):
        image_urls = []
    search_images = [r.image_url for r in results if r.image_url]
    card["image_urls"] = (
        [str(x) for x in image_urls if str(x).strip()]
        or search_images[:3]
        or _FALLBACK_IMAGES[:2]
    )
    card["search_sources"] = sources
    card["city"] = city
    card["source"] = source
    card["expires_at"] = now + timedelta(days=14)
    return card
