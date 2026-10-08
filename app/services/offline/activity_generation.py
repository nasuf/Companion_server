from __future__ import annotations

from app.services.offline.content import plain_text, canonical_url, place_source_matches, concrete_place_name, is_collection_title, normalized

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from app.config import settings
from app.services.llm.models import invoke_text, invoke_json
from app.services.offline.llm import get_offline_chat_model as get_chat_model, get_offline_small_model as get_utility_model
from app.services.offline.activity_images import persist_activity_images
from app.services.offline.image_evidence import indexed_image_evidence
from app.services.offline.prompt_fields import clip_text, filled, parse_text_field
from app.services.offline.recommendation_content import fallback_detail, fallback_summary, useful_detail
from app.services.offline.providers.search import SearchResult, tavily_search
from app.services.offline import repository as repo
from app.services.offline.activity_discovery import (
    ACTIVITY_PLACE_CATEGORIES, VENUE_RE,
    article_place_names, category_for, event_facts, ordered_categories,
    source_address, source_text,
)
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)
_DISCOVERY_BUDGET_S = 40


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
    r"(博物馆|图书馆|美术馆|展览馆|纪念馆|科技馆|文化馆|艺术馆|非遗馆|"
    r"书店|书房|书屋|书吧|咖啡|咖啡馆|茶馆|茶室|奶茶|甜品|烘焙|蛋糕|"
    r"手作|陶艺|花艺|画室|文创|工坊|小店|杂货|唱片|胶片|商铺|门店|百货|"
    r"商场|购物中心|创意园|园区|社区|市民中心|游客中心|"
    r"公园|花园|植物园|湿地|绿道|步道|滨江|江边|河边|湖边|海边|码头|渡口|"
    r"景区|古镇|古街|老街|街区|步行街|市集|夜市|广场|剧场|影院|音乐厅|"
    r"体育馆|运动公园|球场|菜场|菜市场|集市|"
    r"小吃|小吃街|大排档|路边摊|档口|苍蝇馆|小馆|饭馆|餐厅|面馆|"
    r"火锅|烧烤|轻食|早餐|面包|老字号|"
    r"活动|展会|展览|音乐|市集|开幕|"
    r"酒吧|小酒馆|livehouse|KTV|台球|桌游|"
    r"桥|寺|山|湖|江|河|海|馆|园|店|铺|坊|巷|里|弄|站|场|街|楼|中心)"
)

_TAVILY_LOCAL_DOMAINS = (
    "dianping.com",
    "xiaohongshu.com",
    "gov.cn",
    "12301.cn",
    "ctrip.com",
    "mafengwo.cn",
    "meituan.com",
    "douban.com",
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


@dataclass(frozen=True)
class SearchQuerySpec:
    query: str
    include_domains: tuple[str, ...] | None = None


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


def _expand_search_anchor(search_anchor: str) -> str:
    anchor = search_anchor.strip()
    if not anchor:
        return anchor
    direct = _localized_city_terms(anchor)
    if len(direct) > 1:
        return " ".join(term for term in direct if term)
    parts: list[str] = []
    for token in _TOKEN_SPLIT_RE.split(anchor):
        token = token.strip()
        if not token:
            continue
        for term in _localized_city_terms(token):
            if term and term not in parts:
                parts.append(term)
        if token not in parts:
            parts.append(token)
    return " ".join(parts) if parts else anchor


def _search_query(search_anchor: str, tags: list[str]) -> str:
    tag_text = " ".join(tags[:5])
    anchor = _expand_search_anchor(search_anchor)
    return (
        f"{anchor} 真实地点 推荐 周末 一个人 可以去 "
        f"小吃 餐厅 商铺 活动 展览 公园 市集 {tag_text}"
    ).strip()


def resolve_activity_search_context(
    *,
    city: str | None,
    region: str | None,
) -> tuple[str, str, tuple[str, ...]]:
    """Resolve display city, Tavily anchor, and location match terms for filtering."""
    display_city = (city or region or "").strip()
    match_terms: list[str] = []

    if display_city:
        for term in _localized_city_terms(display_city):
            if term and term not in match_terms:
                match_terms.append(term)
        if display_city not in match_terms:
            match_terms.append(display_city)
    if region and region not in match_terms:
        match_terms.append(region)

    search_anchor = _expand_search_anchor(display_city)
    if region and region.strip() not in search_anchor:
        search_anchor += " " + region.strip()
    return display_city, search_anchor.strip(), tuple(match_terms)


def _location_match_terms(city: str, location_terms: list[str] | None) -> list[str]:
    terms: list[str] = []
    for term in (location_terms or []) + list(_localized_city_terms(city)):
        text = str(term or "").strip()
        if text and text not in terms:
            terms.append(text)
    return terms


def _search_query_specs(
    search_anchor: str,
    tags: list[str],
    recent_activities: list[dict[str, str]] | None = None,
    diversity_seed: str = "",
    city: str = "",
) -> list[SearchQuerySpec]:
    recent = recent_activities or []
    base = _expand_search_anchor(search_anchor)
    ordered = ordered_categories(recent, tags, diversity_seed)
    # Every run explores multiple families, with an explicit current-event lane.
    batch = ordered[:8]
    event = next(c for c in ACTIVITY_PLACE_CATEGORIES if c.family == '限时')
    batch = [c for c in batch if c != event]
    batch.insert(3, event)
    specs = [SearchQuerySpec(_search_query(base, tags))]
    for category in batch:
        variant = category.keywords[len(recent) % len(category.keywords)]
        suffix = (datetime.now().strftime('%Y年%m月') + ' 活动时间 地点'
                  if category.family == '限时' else ' 地址 开放时间')
        specs.append(SearchQuerySpec(f"{base} {variant} {suffix}"))
    # County/district remains in primary queries; wider city searches are a
    # bounded recovery lane, with the actual address retained in the result.
    city_anchor = _expand_search_anchor(city) if city else re.sub(r'\s+[^\s]+[区县]$', '', base)
    recovery = [SearchQuerySpec(_search_query(city_anchor, tags)), SearchQuerySpec(
        f"{city_anchor} {ordered[0].keywords[0]} 地址", include_domains=_TAVILY_LOCAL_DOMAINS)]
    return specs[:6] + recovery + specs[6:]


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
        if not _mentions_avoided(
            f"{result.title}\n{_place_from_source(result, '') or ''}",
            avoid_terms,
        )
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
            "content": source_text(result)[:2200],
            "score": result.score,
            "image_url": result.image_url,
        }
        for result in results[:12]
    ]


async def _search_activity_candidates(
    search_anchor: str,
    tags: list[str],
    recent_activities: list[dict[str, str]],
    *,
    city: str = "",
    location_terms: list[str] | None = None,
    max_queries: int = 12,
    diversity_seed: str = "",
) -> tuple[list[SearchResult], list[SearchResult], str]:
    all_usable: list[SearchResult] = []
    filtered: list[SearchResult] = []
    seen_urls: set[str] = set()
    first_query = ""
    specs = _search_query_specs(search_anchor, tags, recent_activities, diversity_seed, city)[:max_queries]

    async def search(spec: SearchQuerySpec) -> list[SearchResult]:
        try:
            return await tavily_search(
                spec.query,
                max_results=8,
                image_evidence=True,
                include_domains=list(spec.include_domains) if spec.include_domains else None,
            )
        except Exception:
            logger.warning("[offline] discovery query failed; continuing other intents")
            return []
    async def collect(spec: SearchQuerySpec) -> None:
        nonlocal filtered
        raw_results = await search(spec)
        for result in _usable_results(raw_results, city, location_terms):
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                all_usable.append(result)
        filtered = _filter_repeated_results(all_usable, recent_activities)
    # Count distinct, usable destinations, not articles or multiple pages for
    # one place. Bound upstream latency while retaining completed discoveries.
    try:
        async with asyncio.timeout(_DISCOVERY_BUDGET_S):
            for offset in range(0, len(specs), 2):
                batch = specs[offset:offset + 2]
                first_query = first_query or batch[0].query
                await asyncio.gather(*(collect(spec) for spec in batch))
                places = {
                    _normalize_fingerprint(card['location_name'])
                    for result in filtered
                    if (card := _fallback_card(city, tags, [result]))
                    and not _card_repeats_history(card, recent_activities)
                }
                families = {category.family for name in places
                            if (category := category_for(name))}
                if offset >= 4 and len(places) >= 4 and len(families) >= 2:
                    break
    except TimeoutError:
        logger.info('[offline] discovery budget reached; retaining %s sources', len(filtered))
    return filtered, all_usable, first_query


def _domain(url: str | None) -> str:
    if not url:
        return ""
    host = urlparse(str(url)).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _source_matches_location(
    combined: str,
    *,
    city: str,
    location_terms: list[str] | None,
) -> bool:
    if not city.strip():
        return True
    combined_lower = combined.lower()
    city_terms = [term for term in _localized_city_terms(city) if term]
    if city.endswith('市') and len(city) > 2:
        city_terms.append(city.removesuffix('市'))
    if city.strip() not in city_terms:
        city_terms.append(city.strip())
    if any(len(term) >= 2 and term.lower() in combined_lower for term in city_terms):
        return True
    # region 补充：仅当 snippet 不含城市名但含用户 region（如区县级）时放行。
    extra = [
        term
        for term in (location_terms or [])
        if term.strip() and term not in city_terms and not term.endswith(("省", "Province"))
    ]
    return any(len(term) >= 2 and term.lower() in combined_lower for term in extra)


def _source_is_usable(
    result: SearchResult,
    city: str,
    location_terms: list[str] | None = None,
) -> bool:
    domain = _domain(result.url)
    if any(
        domain == bad or domain.endswith(f".{bad}")
        for bad in _UNRELIABLE_ACTIVITY_DOMAINS
    ):
        return False
    title = result.title.strip()
    content = result.content.strip()
    combined = f"{title}\n{source_text(result)}"
    if not title or _GENERIC_ACTIVITY_TITLE_RE.search(title):
        return False
    if "No information is available for this page" in content:
        return False
    if not _source_matches_location(
        combined, city=city, location_terms=location_terms
    ):
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


def _looks_like_named_place(text: str) -> bool:
    visible = re.sub(r"\s+", "", str(text or ""))
    if len(visible) < 4:
        return False
    if not re.search(r"[\u4e00-\u9fff]", visible):
        return False
    return not re.fullmatch(r"[0-9A-Za-z\-_.]+", visible)


def _card_has_concrete_place(card: dict[str, Any], city: str) -> bool:
    title = str(card.get("title") or "").strip()
    location = str(card.get("location_name") or card.get("address") or "").strip()
    if not title or _is_generic_location(location, city) or not concrete_place_name(location, _display_city(city)):
        return False
    if len(_normalize_fingerprint(location)) < 2:
        return False
    combined = f"{title}\n{location}\n{card.get('address') or ''}"
    if _CONCRETE_PLACE_HINT_RE.search(combined):
        return True
    return (_looks_like_named_place(location) or _looks_like_named_place(title)
            or bool(re.search(r"(?:路|街|巷|号|弄|楼|street|road)", str(card.get("address") or ""), re.I)))


def _place_from_source(result: SearchResult, city: str) -> str | None:
    title = result.title.strip()
    if is_collection_title(title) or _GENERIC_ACTIVITY_TITLE_RE.search(title):
        return None
    event = event_facts(result)
    if event is not None:
        return event.get('location_name')
    # Keep branch names. A verified address also supports brands such as
    # "Blue Bottle" or "阿婆生煎", which do not end in 店/馆/园.
    name = re.split(r"\s+[-–—|]\s*|[-–—|]\s+|[，。；;:：|_]", title, maxsplit=1)[0].strip()
    name = re.sub(r"(?:开放时间|开放信息|常设展|交通指南|门票信息).*$", "", name).strip()
    name = re.sub(r'(?:官方网站|官方主页|官网|首页)$', '', name).strip()
    labelled = VENUE_RE.search(source_text(result))
    if labelled and labelled.group(1).strip() in title:
        name = labelled.group(1).strip()
    suffix = re.search(r"(?:馆|园|店|铺|坊|巷|弄|站|场|街|楼|中心|咖啡|茶室|书房|书屋|书吧|码头|渡口|古镇|景区|绿道|步道|寺|山|湖|桥|KTV|livehouse|酒吧|市集|夜市)(?:[（(][^()（）]{1,20}[)）])?$", name, re.I)
    if (not _is_generic_location(name, city)
            and concrete_place_name(name, _display_city(city))
            and (suffix or (source_address(result) and (
                (labelled and labelled.group(1).strip().rstrip('* ') == name)
                or category_for(name)
                or re.fullmatch(r'[A-Za-z][A-Za-z0-9 &\u2019\u0027.-]{1,40}', name))))):
        return name
    return None


def _usable_results(
    results: list[SearchResult],
    city: str,
    location_terms: list[str] | None = None,
) -> list[SearchResult]:
    return [
        result
        for result in results
        if _source_is_usable(result, city, location_terms)
    ]


def _source_supports_place(card: dict[str, Any], result: SearchResult) -> bool:
    if place_source_matches(card, result.title, source_text(result)):
        return event_facts(result) != {}
    event = event_facts(result)
    return bool(event and event['location_name'] == card.get('location_name')
                and _source_matches_location(result.title + source_text(result), city=card.get('city') or '', location_terms=None))


def _card_is_source_backed(card: dict[str, Any], sources: list[dict[str, Any]]) -> bool:
    official = canonical_url(card.get("official_url"))
    return bool(official and any(
        canonical_url(source.get("url")) == official
        and _source_supports_place(card, SearchResult(
            title=str(source.get('title') or ''), url=official, content=str(source.get('content') or '')))
        for source in sources
    ))


def _fallback_card(city: str, tags: list[str], results: list[SearchResult]) -> dict[str, Any] | None:
    for item in results:
        location = _place_from_source(item, city)
        if location and _source_supports_place({'location_name': location, 'city': _display_city(city)}, item):
            source = item
            break
    else:
        return None
    event = event_facts(source)
    category = category_for(location) or category_for(source.title + ' ' + source.content)
    suitable = category.suitable if category else '随意逛逛、坐一会儿'
    category_name = '活动现场' if event else category.name if category else '城市漫游'
    return {
        'title': source.title[:60] if event else location,
        'summary': f'想换个节奏的话，可以去{location}，{suitable}，按自己的兴致来就好。',
        'description': f'{location}可以放进你这次出门的备选里。{suitable}都可以，不用安排得太满。具体开放安排以现场为准。',
        'category': category_name, 'vibe': '轻松、随意', 'suitable': suitable,
        'location_name': location, 'address': source_address(source) or location,
        'starts_at': event['starts_at'] if event else None,
        'ends_at': event['ends_at'] if event else None,
        'official_url': source.url, 'image_urls': [],
        'task_hint': '接受后解锁一个小彩蛋任务',
        'easter_egg_task': {'title': '小彩蛋任务',
                           'body': f'在{location}挑一个让你停下来看一眼的小细节，想分享的话就发给我。',
                           'principle': '自愿参与、可独立完成、无安全风险'},
        'metadata': {'fallback': True},
    }


def _candidate_rank(card: dict, recent: list[dict], tags: list[str], region: str) -> tuple:
    category = category_for(str(card.get('category') or '') + ' ' + str(card.get('location_name') or ''))
    family = category.family if category else ''
    recent_categories = [category_for(str(item.get('location_name') or '') + ' ' + str(item.get('title') or '')) for item in recent[:6]]
    repeated = sum(bool(c and c.name == (category.name if category else '')) for c in recent_categories)
    interest = ' '.join(tags)
    affinity = bool(category and any(k in interest for k in category.keywords))
    near = bool(region and region in str(card.get('address') or '') + str(card.get('location_name') or ''))
    return (not near, repeated, not affinity,
            -min(int(card.get('_image_evidence_count') or 0), 3),
            family == (recent_categories[0].family if recent_categories and recent_categories[0] else ''))


async def _verify_discovery_leads(proposals: list[dict], results: list[SearchResult], city: str,
                                  tags: list[str], location_terms: list[str] | None, recent: list[dict]) -> list[dict]:
    verified: list[dict] = []
    leads: list[str] = []
    for card in proposals[:8]:
        card['city'] = _display_city(city)
        name = str(card.get('location_name') or '')
        if not concrete_place_name(name, _display_city(city)) or _card_repeats_history(card, recent):
            continue
        linked = next((r for r in results if canonical_url(r.url) == canonical_url(card.get('official_url'))), None)
        if linked and event_facts(linked) == {}:
            continue
        matching = next((r for r in results if canonical_url(r.url) == canonical_url(card.get('official_url'))
                         and _source_supports_place(card, r)), None)
        if matching:
            facts = _fallback_card(city, tags, [matching])
            if facts and facts['location_name'] == name:
                # Facts always come from the verified page, including dates and
                # addresses. The published prompt owns the recommendation prose.
                for field in ('title', 'summary', 'description', 'vibe', 'suitable'):
                    value = card.get(field)
                    if isinstance(value, str) and value.strip():
                        if field == 'title' and _normalize_fingerprint(name).replace('市', '') not in _normalize_fingerprint(value).replace('市', ''):
                            continue
                        facts[field] = plain_text(value)[:500]
                task = card.get('easter_egg_task')
                if isinstance(task, dict) and all(isinstance(task.get(k), str) for k in ('title', 'body', 'principle')):
                    facts['easter_egg_task'] = {k: plain_text(task[k])[:500] for k in ('title', 'body', 'principle')}
                verified.append(facts)
                continue
        # A model cannot send arbitrary unmentioned names to a verification API.
        if any(event_facts(r) != {} and name in source_text(r) + r.title for r in results):
            leads.append(name)
    for result in results:
        if event_facts(result) != {} and not _place_from_source(result, city):
            leads.extend(article_place_names(result))
    leads = list(dict.fromkeys(name for name in leads if not _card_repeats_history({'location_name': name}, recent)))[:3]

    async def verify(name: str) -> tuple[list[SearchResult], dict | None]:
        try:
            found = await tavily_search(f'{_display_city(city)} {name} 地址', max_results=5, image_evidence=True, timeout_s=8)
            for result in _usable_results(found, city, location_terms):
                card = {'location_name': name, 'city': _display_city(city)}
                if _source_supports_place(card, result):
                    facts = _fallback_card(city, tags, [result])
                    if facts and facts['location_name'] == name:
                        return [result], facts
        except Exception:
            logger.warning('[offline] independent place verification failed')
        return [], None
    # Search verification shares a hard budget and retains completed successes.
    async def collect(name: str) -> None:
        found, card = await verify(name)
        if card:
            results.extend(r for r in found if r.url not in {item.url for item in results})
            verified.append(card)
    try:
        async with asyncio.timeout(10):
            await asyncio.gather(*(collect(name) for name in leads))
    except TimeoutError:
        logger.info('[offline] verification budget reached')
    return verified


async def generate_activity_card(
    *,
    user_id: str,
    workspace_id: str | None,
    city: str,
    source: str,
    search_location: str | None = None,
    location_terms: list[str] | None = None,
    center: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    tags = await repo.list_user_tags(user_id, workspace_id, limit=9)
    memory = await repo.memory_brief(user_id, workspace_id, limit=60)
    recent_activities = await repo.list_recent_activity_fingerprints(
        user_id,
        workspace_id,
        limit=20,
    )
    search_anchor = search_location or city
    if settings.offline_search_provider == 'cleversee':
        from app.services.offline.cleversee_discovery import discover
        native_city = _display_city(city)
        candidates = await discover(city=native_city, search_anchor=search_anchor, tags=tags,
                                    recent=recent_activities, center=center)
        if candidates:
            return await _native_card(candidates, user_id=user_id, city=native_city,
                search_anchor=search_anchor, tags=tags, memory=memory, recent=recent_activities)
        if not settings.offline_tavily_fallback:
            return None
    filtered_results, all_usable_results, query = await _search_activity_candidates(
        search_anchor,
        tags,
        recent_activities,
        city=city,
        location_terms=location_terms,
        diversity_seed=f"{user_id}:{workspace_id or ''}",
    )
    # Search ranking often puts listicles above actual POI pages. Keep usable
    # individual places in the model's bounded source window first.
    distinct, other = [], []
    source_places: set[str] = set()
    for item in filtered_results or all_usable_results:
        place = _normalize_fingerprint(_place_from_source(item, city) or '')
        if place and place not in source_places:
            distinct.append(item)
            source_places.add(place)
        else:
            other.append(item)
    results = (distinct + other)[:24]
    sources = _sources(results)
    proposals: list[dict] = []
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
            async with asyncio.timeout(18):
                raw = await invoke_text(get_chat_model(), prompt_text)
            parsed = _json_object(raw) or {}
            pool = parsed.get("candidates")
            proposals = [c for c in pool[:8] if isinstance(c, dict)] if isinstance(pool, list) else [parsed]
        except Exception as exc:
            logger.warning("[offline] activity LLM generation failed: %s", exc)
    verified = await _verify_discovery_leads(proposals, results, city, tags, location_terms, recent_activities)
    fallback_cards = [candidate for result in results
                      if (candidate := _fallback_card(city, tags, [result]))]
    # All verified candidates must remain in the fact window; model input alone
    # is bounded. Galleries read only pages bound to the chosen place.
    fact_sources = [dict(title=r.title, url=r.url, content=source_text(r)) for r in results]
    region = next((t for t in location_terms or [] if t != city and t.endswith(('区', '县', '镇', '街道', '市'))), '')
    for candidate in verified + fallback_cards:
        identity = {'location_name': candidate['location_name'], 'city': _display_city(city)}
        candidate['_image_evidence_count'] = sum(
            len(indexed_image_evidence(r.raw_content, candidate['location_name'], r.images, r.url))
            for r in results if place_source_matches(identity, r.title, source_text(r)))
    candidates = sorted(verified + fallback_cards,
                        key=lambda c: _candidate_rank(c, recent_activities, tags, region))
    card = None
    unillustrated = None
    now = datetime.now(UTC)
    seen_places: set[str] = set()
    # A bounded retry selects another real place if a gallery cannot be verified.
    # This shares the same validation gate for model and deterministic candidates.
    try:
        async with asyncio.timeout(55):
            for candidate in candidates:
                candidate["city"] = _display_city(city)
                identity = _normalize_fingerprint(candidate.get("location_name") or "")
                if identity in seen_places:
                    continue
                if (not _card_has_concrete_place(candidate, city)
                        or not _card_is_source_backed(candidate, fact_sources)
                        or _card_repeats_history(candidate, recent_activities)):
                    continue
                # Expired/undated events cannot become the image-free fallback.
                if candidate.get('starts_at') or candidate.get('ends_at'):
                    try:
                        end = datetime.fromisoformat(str(candidate.get('ends_at') or '').replace('Z', '+00:00'))
                        if end.tzinfo is None:
                            end = end.replace(tzinfo=UTC)
                        if end <= now:
                            continue
                        candidate['expires_at'] = min(end, now + timedelta(days=30))
                    except ValueError:
                        continue
                if len(seen_places) >= 3:
                    break
                seen_places.add(identity)
                # Image availability ranks valid destinations; it must not
                # erase them. Never retain model-proposed or raw source URLs.
                candidate['image_urls'] = []
                if unillustrated is None:
                    unillustrated = candidate
                candidate["image_urls"] = await persist_activity_images(
                    user_id=user_id, card=candidate, city=city, search_results=results, limit=3,
                )
                if candidate["image_urls"]:
                    if card is None or len(candidate['image_urls']) > len(card['image_urls']):
                        card = candidate
                    if len(card['image_urls']) >= 3:
                        break
                logger.info("[offline] retained image-free fallback place=%r", candidate.get("location_name"))
    except TimeoutError:
        logger.info("[offline] candidate/gallery budget exhausted")
    card = card or unillustrated
    if not card:
        logger.warning("[offline] no concrete activity card generated query=%r", query)
        return None

    for field in ("title", "summary", "description", "vibe", "suitable"):
        card[field] = plain_text(card.get(field))
    card.pop('_image_evidence_count', None)
    chosen_sources = [dict(title=r.title, url=r.url, content=source_text(r)[:2200]) for r in results
                      if _source_supports_place(card, r)]
    card["search_sources"] = chosen_sources + [{"kind": "image", **image} for image in card.pop("image_provenance", [])]
    card["city"] = _display_city(city)
    card["source"] = source
    card.setdefault("expires_at", now + timedelta(days=14))
    copy = await _recommendation_copy(card)
    if copy:
        card["description"] = plain_text(copy)
    if not useful_detail(card["description"]):
        card["description"] = fallback_detail(card)
    return card


async def _native_card(
    candidates: list[dict],
    *,
    user_id: str,
    city: str,
    search_anchor: str,
    tags: list[str],
    memory: str,
    recent: list[dict],
) -> dict:
    # Model output selects an ID and supplies copy, never place/session facts.
    facts = [{k: v for k, v in c.items() if k != "native_images"} for c in candidates]
    prompt = (await get_prompt_text("offline.activity_card")).format(
        city=city,
        search_anchor=search_anchor,
        tags="、".join(tags) or "暂无",
        memory=memory or "暂无",
        avoid_text=_avoid_text(recent),
        sources_json=json.dumps(facts, ensure_ascii=False),
    )
    selected, copy = candidates[0], {}
    try:
        async with asyncio.timeout(18):
            raw = await invoke_text(get_chat_model(), prompt)
        payload = _json_object(raw) or {}
        proposed = payload.get("candidates", [])
        by_id = {c["candidate_id"]: c for c in candidates}
        for item in proposed if isinstance(proposed, list) else []:
            if isinstance(item, dict) and item.get("candidate_id") in by_id:
                selected, copy = by_id[item["candidate_id"]], item
                break
    except Exception as exc:
        logger.info("[offline] candidate selection fallback (%s)", type(exc).__name__)
    card, bound_sources, source = None, [], None
    # A valid but unillustrated POI must not beat an equally relevant POI with
    # a verified album. Keep the selected event/session and user category intact.
    alternatives = [
        c
        for c in candidates
        if c is not selected
        and c.get("native_images")
        and c.get("category") == selected.get("category")
        and (c.get("discovery_metadata") or {}).get("kind") == "place"
    ]
    attempts = [selected] + (
        alternatives[:2]
        if selected["discovery_metadata"].get("kind") == "place"
        else []
    )
    try:
        async with asyncio.timeout(55):
            for candidate in attempts:
                current = {
                    **candidate,
                    "discovery_metadata": dict(candidate["discovery_metadata"]),
                }
                pages = await _native_context_sources(current, city)
                album = _native_album_source(current, city)
                current["image_urls"] = await persist_activity_images(
                    user_id=user_id,
                    card=current,
                    city=city,
                    search_results=[album] + pages,
                    limit=3,
                )
                if card is None or len(current["image_urls"]) > len(card["image_urls"]):
                    card, bound_sources, source = current, pages, album
                # A partial but real album stays with the intended destination;
                # fallback is for wholly empty galleries, not a three-photo quota.
                if current["image_urls"]:
                    break
    except TimeoutError:
        logger.info("[offline] native gallery selection deadline reached")
    if card is None:
        card = {
            **selected,
            "discovery_metadata": dict(selected["discovery_metadata"]),
            "image_urls": [],
        }
        source = _native_album_source(card, city)
    if card["candidate_id"] != selected["candidate_id"]:
        copy = {}  # Never attach the original place's reason to its replacement.
    card["summary"] = fallback_summary(card)
    notes = [
        dict(title=p.title, url=p.url, text=source_text(p)[:2200])
        for p in bound_sources
    ]
    facts_for_copy = {**card, "verified_source_text": notes}
    copy_context = {
        **card,
        "description": json.dumps(
            {
                "place_introduction": card["description"],
                "native_category": card["category"],
                "opening_hours": card["discovery_metadata"].get("opening_hours", ""),
                "average_spend_reference": card["discovery_metadata"].get(
                    "price_info", ""
                ),
                "verified_source_text": notes,
            },
            ensure_ascii=False,
        ),
    }
    card["search_sources"] = [
        dict(
            title=source.title,
            url=source.url,
            provider="cleversee",
            poi_id=card["native_poi_id"],
        )
    ] + card.get("image_provenance", [])
    card["search_sources"] += [
        dict(title=p.title, url=p.url, subject="place_context") for p in bound_sources
    ]
    if card["discovery_metadata"].get("kind") == "event":
        card["search_sources"].insert(
            0,
            dict(
                title=card["title"],
                url=card["official_url"],
                subject="event_announcement",
            ),
        )
    card["source"] = "cleversee"
    card["expires_at"] = (
        card.get("expires_at")
        or card.get("ends_at")
        or (datetime.now(UTC) + timedelta(days=14)).isoformat()
    )
    copy_text = await _recommendation_copy(copy_context)
    card["description"] = fallback_detail(card)
    copy_status = "fallback_unavailable"
    if useful_detail(copy_text):
        try:
            summary = plain_text(copy.get("summary"))[:120] or card["summary"]
            if normalized(summary) == normalized(f"可以去{card['location_name']}看看"):
                summary = card["summary"]
            vibe = plain_text(copy.get("vibe"))[:30]
            check_prompt = (
                await get_prompt_text("offline.recommendation_fact_check")
            ).format(
                facts_json=json.dumps(
                    {
                        k: v
                        for k, v in facts_for_copy.items()
                        if k not in {"native_images", "vibe", "image_provenance"}
                    },
                    ensure_ascii=False,
                ),
                copy_text="推荐摘要："
                + summary
                + "\n氛围："
                + vibe
                + "\n\n推荐详情："
                + copy_text,
            )
            async with asyncio.timeout(8):
                checked = await invoke_json(get_utility_model(), check_prompt)
            if isinstance(checked, dict) and checked.get("supported") is True:
                card["description"] = plain_text(copy_text)
                card["summary"] = summary
                if vibe:
                    card["vibe"] = vibe
                copy_status = "verified"
            else:
                copy_status = "fallback_rejected"
        except Exception as exc:
            logger.info("[offline] factual copy fallback (%s)", type(exc).__name__)
    elif copy_text:
        copy_status = "fallback_too_short"
    card["discovery_metadata"]["copy_status"] = copy_status
    logger.info(
        "[offline] native detail copy=%s photos=%s",
        copy_status,
        len(card["image_urls"]),
    )
    return card


async def _native_context_sources(card: dict, city: str) -> list[SearchResult]:
    if card["discovery_metadata"].get("kind") != "place":
        return []
    from app.services.offline.providers.cleversee import web_search

    try:
        async with asyncio.timeout(10):
            pages = await web_search(
                f"{city} {card['location_name']} 简介 开放时间",
                max_results=4,
            )
        return [
            p
            for p in pages
            if _source_is_usable(p, city)
            and place_source_matches(card, p.title, p.content + p.raw_content)
        ][:2]
    except Exception as exc:
        logger.info("[offline] place context unavailable (%s)", type(exc).__name__)
        return []


def _native_album_source(card: dict, city: str) -> SearchResult:
    items = [
        ({"url": item} if isinstance(item, str) else item)
        for item in card.get("native_images", [])
    ]
    return SearchResult(
        title=city + " " + card["location_name"],
        url="https://www.amap.com/detail/" + card["native_poi_id"],
        content=card["address"],
        images=[
            {**item, "poi_id": card["native_poi_id"], "evidence": "native_poi"}
            for item in items
            if isinstance(item, dict) and item.get("url")
        ],
    )


def _date_time_text(card: dict[str, Any]) -> str:
    from app.services.offline.event_schedule import schedule_label

    label = schedule_label((card.get("discovery_metadata") or {}).get("event") or {})
    if label:
        return label
    start = str(card.get("starts_at") or "").strip()
    end = str(card.get("ends_at") or "").strip()
    if start and end:
        return f"{start} 至 {end}"
    if start:
        return start
    hours = (card.get("discovery_metadata") or {}).get("opening_hours")
    return (
        "固定地点；开放时间：" + str(hours)
        if hours
        else "固定地点；开放时段未提供，不能推断全天或随时营业"
    )


async def _recommendation_copy(card: dict[str, Any]) -> str:
    """Detail-page seed copy. The caller supplies a readable factual fallback."""
    try:
        prompt = (await get_prompt_text("offline.activity_recommendation_copy")).format(
            activity_name=filled(card.get("title"), empty="这次外出"),
            date_time=_date_time_text(card),
            location=filled(
                " ".join(
                    part
                    for part in (
                        str(card.get("location_name") or "").strip(),
                        str(card.get("address") or "").strip(),
                    )
                    if part
                ),
                empty="（未提供）",
            ),
            category=filled(card.get("category"), empty="线下活动"),
            description=filled(card.get("description") or card.get("summary"), empty="（无）"),
            official_link=filled(card.get("official_url"), empty="（无）"),
            activity_summary=filled(card.get("summary"), empty="（无）"),
            recommendation_blurb=filled(card.get("summary"), empty="（无）"),
        )
        async with asyncio.timeout(15):
            return clip_text(parse_text_field(await invoke_text(get_chat_model(), prompt)), 500)
    except Exception as exc:
        logger.warning("[offline] recommendation copy failed: %s", exc)
        return ""


async def generate_activity_invite_message(
    *,
    activity: dict[str, Any],
    user_id: str,
    workspace_id: str | None,
) -> str:
    place = clip_text(str(activity.get("location_name") or activity.get("title") or "这个地方"), 30)
    fallback = f"「{place}」要不要了解一下？"
    try:
        from app.services.offline.activity_message_context import message_context

        tags = await repo.list_user_tags(user_id, workspace_id, limit=6)
        memory = await repo.memory_brief(user_id, workspace_id, limit=20, include_ai=False)
        ctx = await repo.resolve_user_context(user_id, workspace_id)
        context_fields = await message_context(ctx or {})
        prompt_template = await get_prompt_text("offline.activity_invite_message")
        prompt_text = prompt_template.format(
            **context_fields,
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
        return clip_text(text, 80) or fallback
    except Exception as exc:
        logger.warning("[offline] activity invite message generation failed: %s", exc)
        return fallback
