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
    # V3 (2026-09-14): 保留结构化 candidates 给 topic_source 分类器 + prompt.
    # V0 只需要 text (append_trending_section 尾追), V3 需要按候选粒度打分/挑选.
    # 缓存路径拿到的 text 无法反解回结构化, 这时候 candidates=() (V3 分类器会走 none).
    candidates: tuple[dict, ...] = ()


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


def _url_platform(url: str) -> str:
    """URL → 支持的社交平台名; 不在白名单则返回空串 (2026-09-14 平台白名单单一真源).

    白名单必须跟 chat_links.SUPPORTED_PLATFORMS 对齐 — 那才是**卡片渲染真的能画**
    的平台集合. 聚合站 (tophub.today / 今日热榜 / 36kr 热榜等) 不在这里因为 URL
    是聚合首页, 点进去用户看不到具体那条新闻; tavily 抓"微博热搜"经常返 tophub 类
    聚合站, 若不过滤会一路穿进 socially_hot 档, 卡片挂个"链接" (无平台兜底名)
    上去 —— 就是用户截图那种"什么都没说清楚"的坏体验.
    """
    u = (url or "").strip().lower()
    if not u:
        return ""
    # 顺序: 单一子串匹配, 优先度按域名唯一性排 (b23.tv / xhslink 是短链要单独列)
    if "weibo.com" in u or "s.weibo.cn" in u or "m.weibo" in u:
        return "微博"
    if "xiaohongshu.com" in u or "xhslink.com" in u:
        return "小红书"
    if "bilibili.com" in u or "b23.tv" in u:
        return "B站"
    if "zhihu.com" in u:  # 含 zhuanlan.zhihu.com
        return "知乎"
    if "douyin.com" in u or "iesdouyin.com" in u:
        return "抖音"
    if "toutiao.com" in u or "toutiaoimg.com" in u:
        return "头条"
    return ""


def _filter_to_supported_platforms(
    results: list[SearchResult],
) -> list[SearchResult]:
    """丢掉 URL 不在白名单里的结果 (tophub 类聚合站等). 见 _url_platform 说明."""
    out: list[SearchResult] = []
    for r in results:
        if _url_platform(r.url):
            out.append(r)
    if not out and results:
        # 全部被过滤: 观测点, 让运维知道 tavily 这次抓的全是杂鱼
        logger.info(
            "[proactive-trending] all %d results dropped by platform whitelist "
            "(likely tophub/aggregator noise) → proactive falls back to plain",
            len(results),
        )
    return out


def _search_results_to_candidates(results: list[SearchResult]) -> tuple[dict, ...]:
    """SearchResult 列表 → V3 分类器认识的 dict 形状 (title/snippet/url/platform).

    调用方应先过 _filter_to_supported_platforms; 但这里再保底判一次 platform 非空,
    确保 candidates 里没有 platform="" 的杂鱼 (卡片渲染兜底才不会显示"链接").
    """
    out: list[dict] = []
    for r in results:
        title = (r.title or "").strip()
        snippet = (r.content or "").strip()
        if not title and not snippet:
            continue
        url = (r.url or "").strip()
        platform = _url_platform(url)
        if not platform:
            continue  # 双保险: 白名单外一律不进 candidates
        out.append({
            "title": title[:120], "snippet": snippet[:400],
            "url": url, "platform": platform,
        })
    return tuple(out)


async def _tavily_snippets(query: str) -> tuple[str, tuple[dict, ...]]:
    """→ (formatted_text, structured_candidates). 空则 ("", ()).

    text 与 candidates 都基于**过滤后**的 results: 若白名单外的杂鱼被过滤光, text
    也一起空 —— 上游 trending_attached=False → 主动消息干净落回 silence_plain,
    比给 LLM 塞聚合站 UI 残余强.
    """
    if not settings.tavily_api_key.strip():
        return "", ()
    timeout = float(getattr(settings, "chat_link_search_timeout_s", 8.0))
    try:
        # max_results 从 5 提到 8: 平台过滤会砍掉一部分 (tavily 首页常返 tophub),
        # 多抓几条保证白名单内候选够
        results = await tavily_search(query, max_results=8, timeout_s=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[proactive-trending] tavily failed query=%s: %s", query, exc)
        return "", ()
    results = _filter_to_supported_platforms(results)
    return _format_search_results(results), _search_results_to_candidates(results)


async def _brave_snippets(query: str) -> tuple[str, tuple[dict, ...]]:
    """→ (formatted_text, structured_candidates). 空则 ("", ())."""
    api_key = settings.brave_search_api_key.strip()
    endpoint = settings.brave_search_endpoint.strip()
    if not api_key or not endpoint:
        return "", ()
    headers = {
        "accept": "application/json",
        "accept-encoding": "gzip",
        "x-subscription-token": api_key,
    }
    params = {
        # count 提到 10 让平台过滤后仍有候选 (brave 首页也常返聚合站)
        "q": " ".join(query.split())[:160],
        "count": 10,
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
        return "", ()
    results = _filter_to_supported_platforms(_results_from_brave_payload(data))
    return _format_search_results(results), _search_results_to_candidates(results)


async def _fetch_live_trending_snippets(*, topic: str | None = None) -> TrendingLoadResult:
    seed = (topic or "").strip()
    queries: list[str] = list(_LIVE_QUERIES)
    if seed and seed not in {"公共话题", "问候", "日常"}:
        queries.append(f"微博热搜 今日 {seed}")
        queries.append(f"今日热点新闻 {seed}")

    for query in queries:
        text, cands = await _tavily_snippets(query)
        if text:
            return TrendingLoadResult(
                text=text, cache_hit=False, provider="tavily", candidates=cands,
            )
        text, cands = await _brave_snippets(query)
        if text:
            return TrendingLoadResult(
                text=text, cache_hit=False, provider="brave", candidates=cands,
            )
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
