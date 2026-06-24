from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline.activity_images import persist_activity_images
from app.services.offline.providers.search import SearchResult, tavily_search
from app.services.offline import repository as repo
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


_FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1200&q=80",
    "https://images.unsplash.com/photo-1492684223066-81342ee5ff30?auto=format&fit=crop&w=1200&q=80",
    "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=80",
]

_LOCALIZED_CITY_ALIASES = {
    "zhenjiang": ("江苏 镇江", "镇江", "Zhenjiang"),
    "jiangsu": ("江苏", "Jiangsu"),
}

_GENERIC_LOCATION_RE = re.compile(
    r"^(当前位置附近|当前位置|附近|本地|当前城市|城市附近|周边|附近区域|"
    r"local|nearby|current\s*location)$",
    flags=re.I,
)
_CONCRETE_PLACE_HINT_RE = re.compile(
    r"(博物馆|图书馆|美术馆|展览馆|文化馆|书店|咖啡|茶|公园|花园|景区|"
    r"街区|市集|广场|剧场|影院|音乐厅|中心|码头|古镇|寺|山|湖|江|馆|园|店)"
)

_UNRELIABLE_ACTIVITY_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "tripadvisor.com",
    "youtube.com",
    "youtu.be",
    "calendar.yahoo.com",
}

_GENERIC_ACTIVITY_TITLE_RE = re.compile(
    r"\b(the best|things to do|free things|attractions|travel guide|calendar)\b",
    flags=re.I,
)
_TOKEN_SPLIT_RE = re.compile(
    r"[\s,，。；;:：/\\|·「」『』《》()（）\[\]【】\"'“”‘’]+"
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


def _display_city(city: str) -> str:
    terms = [term for term in _localized_city_terms(city) if term]
    if len(terms) >= 2:
        return terms[1]
    return terms[0] if terms else city.strip()


def _search_query(city: str, tags: list[str]) -> str:
    tag_text = " ".join(tags[:5])
    anchor = " ".join(term for term in _localized_city_terms(city) if term)
    return (
        f"{anchor or city} 官方 文旅 场馆 公告 开放时间 免费 低成本 小众 "
        f"博物馆 图书馆 展览 市集 音乐 书店 咖啡 公园 {tag_text}"
    ).strip()


def _normalize_fingerprint(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _TOKEN_SPLIT_RE.sub("", text)
    return re.sub(r"(市|区|县|省|官方|活动|常设展|推荐)$", "", text)


def _significant_terms(*values: Any) -> list[str]:
    terms: list[str] = []
    for value in values:
        normalized = _normalize_fingerprint(value)
        if len(normalized) >= 3 and normalized not in terms:
            terms.append(normalized)
    return terms


def _avoid_terms(recent_activities: list[dict[str, str]]) -> list[str]:
    terms: list[str] = []
    for item in recent_activities:
        for term in _significant_terms(
            item.get("location_name"),
            item.get("address"),
            item.get("title"),
        ):
            if term not in terms:
                terms.append(term)
    return terms[:30]


def _avoid_text(recent_activities: list[dict[str, str]]) -> str:
    if not recent_activities:
        return "暂无"
    parts: list[str] = []
    for item in recent_activities[:12]:
        location = item.get("location_name") or item.get("address") or "未知地点"
        title = item.get("title") or "未知活动"
        parts.append(f"- {title} / {location}")
    return "\n".join(parts)


def _mentions_avoided(value: str, avoid_terms: list[str]) -> bool:
    normalized = _normalize_fingerprint(value)
    if not normalized:
        return False
    return any(term in normalized or normalized in term for term in avoid_terms)


def _filter_repeated_results(
    results: list[SearchResult],
    recent_activities: list[dict[str, str]],
) -> list[SearchResult]:
    avoid_terms = _avoid_terms(recent_activities)
    if not avoid_terms:
        return results
    filtered = [
        result
        for result in results
        if not _mentions_avoided(f"{result.title}\n{result.content}", avoid_terms)
    ]
    return filtered


def _card_repeats_history(
    card: dict[str, Any],
    recent_activities: list[dict[str, str]],
) -> bool:
    if not recent_activities:
        return False
    card_terms = _significant_terms(
        card.get("location_name"),
        card.get("address"),
        card.get("title"),
    )
    history_terms = _avoid_terms(recent_activities)
    return any(
        term in historical or historical in term
        for term in card_terms
        for historical in history_terms
    )


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


def _is_generic_location(value: Any, city: str | None = None) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if _GENERIC_LOCATION_RE.search(text):
        return True
    normalized = _normalize_fingerprint(text)
    if not normalized:
        return True
    if city:
        city_terms = [
            _normalize_fingerprint(term)
            for term in _localized_city_terms(city)
            if term
        ]
        if normalized in city_terms:
            return True
    return False


def _card_has_concrete_place(card: dict[str, Any], city: str) -> bool:
    title = str(card.get("title") or "").strip()
    location = str(card.get("location_name") or card.get("address") or "").strip()
    if not title or _is_generic_location(location, city):
        return False
    if len(_normalize_fingerprint(location)) < 2:
        return False
    combined = f"{title}\n{location}\n{card.get('address') or ''}"
    return bool(_CONCRETE_PLACE_HINT_RE.search(combined))


def _place_from_source(result: SearchResult, city: str) -> str | None:
    title = result.title.strip()
    for part in _TOKEN_SPLIT_RE.split(title):
        part = part.strip()
        if not part:
            continue
        if _is_generic_location(part, city):
            continue
        if _CONCRETE_PLACE_HINT_RE.search(part):
            return part
    if not _is_generic_location(title, city) and _CONCRETE_PLACE_HINT_RE.search(title):
        return title[:32]
    return None


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


def _fallback_card(
    city: str,
    tags: list[str],
    results: list[SearchResult],
) -> dict[str, Any] | None:
    source = next((item for item in results if _place_from_source(item, city)), None)
    if source is None:
        return None
    location = _place_from_source(source, city)
    if not location:
        return None
    city_label = _display_city(city)
    title = source.title.strip()[:36] or f"{location}轻量出门计划"
    task = f"到{location}后，拍下一处你觉得有一点可爱的细节，发给我看看。"
    return {
        "title": title,
        "summary": source.content.strip()[:80] or "一个低压力、可以独立完成的小出门计划。",
        "description": (
            f"我帮你在{city_label}找到一个可以独自慢慢看的地方：{location}。"
            "不需要社交表现，也不用赶时间，就当给今天换一点空气。"
        ),
        "category": tags[0] if tags else "城市漫游",
        "location_name": location,
        "address": location,
        "starts_at": None,
        "ends_at": None,
        "official_url": source.url,
        "image_urls": [source.image_url] if source.image_url else [],
        "task_hint": "接受后解锁一个小彩蛋任务",
        "easter_egg_task": {
            "title": "小彩蛋任务",
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
) -> dict[str, Any] | None:
    tags = await repo.list_user_tags(user_id, workspace_id, limit=9)
    memory = await repo.memory_brief(user_id, workspace_id, limit=60)
    query = _search_query(search_location or city, tags)
    raw_results = await tavily_search(query, max_results=8)
    recent_activities = await repo.list_recent_activity_fingerprints(
        user_id,
        workspace_id,
        limit=20,
    )
    results = _filter_repeated_results(
        _usable_results(raw_results, city),
        recent_activities,
    )[:6]
    sources = _sources(results)
    card: dict[str, Any] | None = None
    if sources:
        try:
            prompt_template = await get_prompt_text("offline.activity_card")
            prompt_text = prompt_template.format(
                city=city,
                search_anchor=search_location or city,
                tags=", ".join(tags) if tags else "暂无",
                memory=memory or "暂无足够记忆，使用城市热门和季节普适活动兜底。",
                avoid_text=_avoid_text(recent_activities),
                sources_json=json.dumps(sources, ensure_ascii=False),
            )
            if "{avoid_text}" not in prompt_template:
                prompt_text += (
                    "\n\n最近已推荐过的活动/地点，必须尽量避开：\n"
                    f"{_avoid_text(recent_activities)}\n"
                    "不要重复推荐同一地点、同一场馆或高度相似主题。"
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
    if card and not _card_has_concrete_place(card, city):
        logger.warning(
            "[offline] discarded generic activity card title=%r location=%r query=%r",
            card.get("title"),
            card.get("location_name") or card.get("address"),
            query,
        )
        card = None
    if card and _card_repeats_history(card, recent_activities):
        logger.warning(
            "[offline] discarded repeated activity card title=%r location=%r query=%r",
            card.get("title"),
            card.get("location_name"),
            query,
        )
        card = None
    if not card:
        card = _fallback_card(city, tags, results)
    if not card:
        logger.warning("[offline] no concrete activity card generated query=%r", query)
        return None

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
    persisted_images = await persist_activity_images(
        user_id=user_id,
        card=card,
        city=city,
        search_results=results,
        limit=3,
    )
    if persisted_images:
        card["image_urls"] = persisted_images
    card["search_sources"] = sources
    card["city"] = city
    card["source"] = source
    card["expires_at"] = now + timedelta(days=14)
    return card


async def generate_activity_invite_message(
    *,
    activity: dict[str, Any],
    user_id: str,
    workspace_id: str | None,
) -> str:
    tags = await repo.list_user_tags(user_id, workspace_id, limit=6)
    memory = await repo.memory_brief(user_id, workspace_id, limit=20)
    fallback = (
        f"我看到「{activity.get('title') or '这个地方'}」还挺适合你，"
        "不是很吵，也不用赶流程。要不要看看这张小卡？"
    )
    try:
        prompt_template = await get_prompt_text("offline.activity_invite_message")
        prompt_text = prompt_template.format(
            title=activity.get("title") or "线下活动",
            location=activity.get("location_name")
            or activity.get("address")
            or activity.get("city")
            or "附近",
            summary=activity.get("summary") or activity.get("description") or "",
            tags=", ".join(tags) if tags else "暂无",
            memory=memory or "暂无",
        )
        text = (await invoke_text(get_chat_model(), prompt_text)).strip()
        text = re.sub(r"^['\"“”]+|['\"“”]+$", "", text).strip()
        return text[:80] or fallback
    except Exception as exc:
        logger.warning("[offline] activity invite message generation failed: %s", exc)
        return fallback
