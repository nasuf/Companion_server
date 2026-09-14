"""Fetch and cache public trending snippets for proactive messages."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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

# URL 里出现的年份启发式过滤: tavily 常返 /2020/... /p/2021... 类老 SEO 文章.
# 例: "yg-hgt-p-2022" / "/article/6800000000/" 里的 4 位数字若是往年年份 → 老内容.
# 目前只匹配"完整 4 位年份出现在 path 里"这个强信号, 不搞太复杂避免误杀.
_URL_YEAR_PAT = re.compile(r"/(?:19|20)(\d{2})/|-(?:19|20)(\d{2})-|_(?:19|20)(\d{2})_")


def _url_looks_stale(url: str, max_age_days: int) -> bool:
    """URL path 里出现明确早于 max_age_days 天的年份 → 视为老内容.

    只是启发式 (URL 里的年份未必等于内容发布年份), 但对 tavily 拿到的"2021 SEO
    列表" 这类命名规范的旧文命中率高. 未匹配 (无年份 or 是当年) → 保留判断给上游.
    """
    if max_age_days <= 0:
        return False  # 关闭 filter
    m = _URL_YEAR_PAT.search(url or "")
    if not m:
        return False  # 无年份线索 → 不判定
    year_2digit = int(m.group(1) or m.group(2) or m.group(3))
    url_year = 2000 + year_2digit if year_2digit <= 99 else year_2digit
    cutoff_year = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).year
    return url_year < cutoff_year


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
    # 微博加 "s.weibo" 无 .cn 后缀匹配 (DailyHot 返的热搜 URL 是 s.weibo.com/weibo?q=..)
    if "weibo.com" in u or "s.weibo" in u or "m.weibo" in u:
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
    """丢掉 URL 不在白名单里的结果 + 明显是老 SEO 的 URL.

    两层过滤:
      A) _url_platform(url) 空 → 不是支持平台, 丢 (聚合站/杂鱼)
      B) _url_looks_stale(url, max_age_days) → URL path 里带明显的往年年份, 丢
         (tavily 常返 "50个热门话题 2021" 这类 SEO 老列表)
    """
    max_age = int(getattr(settings, "proactive_hot_max_age_days", 7))
    out: list[SearchResult] = []
    stale_dropped = 0
    for r in results:
        if not _url_platform(r.url):
            continue
        if _url_looks_stale(r.url, max_age):
            stale_dropped += 1
            continue
        out.append(r)
    if not out and results:
        logger.info(
            "[proactive-trending] all %d results dropped by filters "
            "(likely tophub/aggregator noise or stale SEO; %d stale) "
            "→ proactive falls back to plain",
            len(results), stale_dropped,
        )
    elif stale_dropped:
        logger.info(
            "[proactive-trending] dropped %d stale-URL results "
            "(URL path 带往年年份, 超 %d 天)",
            stale_dropped, max_age,
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


# DailyHot 每个平台一个 endpoint, 我们并发拉 6 个平台的 top-N 汇总.
# slug 是 DailyHot 的路径名, platform 是我们内部展示名 (跟 _url_platform 一致).
# 顺序保持稳定 (V3 classifier 内部按优先级挑, 不依赖顺序; 但日志读着舒服).
_DAILYHOT_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("weibo",    "微博"),
    ("zhihu",    "知乎"),
    ("bilibili", "B站"),
    ("xhs",      "小红书"),
    ("toutiao",  "头条"),
    ("douyin",   "抖音"),
)
_DAILYHOT_PER_PLATFORM_TAKE = 5  # 每个平台拉多少 top → 6*5=30 候选池给 V3 挑


async def _hot_api_snippets(query: str) -> tuple[str, tuple[dict, ...]]:
    """从 proactive_hot_api_url (DailyHot base URL) 并发拉 6 个平台的 top-N.

    ## 上游: DailyHot 开源热榜聚合 (github.com/imsyy/DailyHot)

    - 免费, MIT, 有官方托管实例 (e.g. api-hot.imsyy.top), 也可自建
    - 每个平台一个 endpoint: GET {base_url}/weibo, /zhihu, /bilibili, /xhs, /toutiao, /douyin
    - 响应: {"code":200, "data":[{"title","url","hot","mobileUrl","desc"?}, ...]}

    ## 使用

    admin 后台 env 设 proactive_hot_api_url = "https://api-hot.imsyy.top"
    (base URL, 不带 path, **不带斜杠结尾**). 未设则静默返空, tavily 顶上老路.

    query 参数保留为签名兼容, 不使用 —— 热榜按热度返 top-N, V3 分类器再从池子里
    按 user/AI 兴趣挑, 不按 query 收窄.

    ## 与 tavily 的关系

    - hot_api 优先: 拿到 candidates 直接返, 不调 tavily
    - hot_api 返空 (endpoint 未配 / 全部平台失败): 静默 fallback tavily
    - 两者都空: trending_attached=False → 主动消息落 silence_plain

    ## 失败容忍

    - 任一平台 endpoint 失败/超时不影响其它平台 (asyncio.gather return_exceptions)
    - 至少 1 个平台成功即算成功
    - 全部失败: 返 ("", ()), fallback 到 tavily
    """
    base_url = getattr(settings, "proactive_hot_api_url", "").strip().rstrip("/")
    if not base_url:
        return "", ()

    headers = {"accept": "application/json", "user-agent": "Mozilla/5.0 CompanionBot"}
    api_key = getattr(settings, "proactive_hot_api_key", "").strip()
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    timeout = float(getattr(settings, "chat_link_search_timeout_s", 8.0))

    async def _fetch_one(slug: str, platform: str) -> list[dict]:
        endpoint = f"{base_url}/{slug}"
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=headers,
                                          trust_env=False, follow_redirects=True) as client:
                resp = await client.get(endpoint)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001 - 单平台失败不阻塞其它
            # SSL / ConnectError 类异常 str() 会返空; 用 repr() 保证有信号
            msg = str(exc) or repr(exc)
            logger.info("[proactive-trending] dailyhot fetch failed platform=%s: %s",
                        platform, msg[:120])
            return []
        items = _extract_hot_items(data)[: _DAILYHOT_PER_PLATFORM_TAKE]
        # 给每条标 platform (来自 slug 映射, 100% 可信, 不依赖 URL 推断)
        for item in items:
            if isinstance(item, dict):
                item["__platform__"] = platform
        return items

    fetches = await asyncio.gather(
        *[_fetch_one(slug, platform) for slug, platform in _DAILYHOT_ENDPOINTS],
        return_exceptions=False,  # 内部已 catch, 不会真 raise
    )
    raw_items: list[dict] = []
    for platform_items in fetches:
        raw_items.extend(platform_items or [])

    if not raw_items:
        logger.info("[proactive-trending] dailyhot base=%s: all 6 platforms empty",
                    base_url[:60])
        return "", ()

    max_age = int(getattr(settings, "proactive_hot_max_age_days", 7))
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age) if max_age > 0 else None

    candidates: list[dict] = []
    lines: list[str] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or item.get("mobileUrl") or item.get("link") or "").strip()
        if not title or not url:
            continue
        # platform 优先用 slug 映射来的 (可信); 兜底回 URL 推 (处理泛用 API)
        platform = str(item.get("__platform__")
                       or item.get("platform") or "").strip() or _url_platform(url)
        if not platform:
            continue
        if _url_looks_stale(url, max_age):
            continue
        if cutoff is not None:
            pub = _parse_iso(item.get("published_at"))
            if pub is not None and pub < cutoff:
                continue
        snippet = str(item.get("snippet") or item.get("desc") or "").strip()
        candidates.append({
            "title": title[:120], "snippet": snippet[:400],
            "url": url, "platform": platform,
        })
        lines.append(f"- [{platform}] {title[:80]}"
                     + (f": {snippet[:140]}" if snippet else ""))

    if not candidates:
        logger.info(
            "[proactive-trending] dailyhot yielded 0 usable candidates "
            "(all filtered by freshness/quality)",
        )
        return "", ()

    logger.info(
        "[proactive-trending] dailyhot yielded %d candidates (from %d raw items across %d platforms)",
        len(candidates), len(raw_items),
        len([f for f in fetches if f]),
    )
    return "\n".join(lines), tuple(candidates)


def _extract_hot_items(data: Any) -> list:
    """兼容 3 种常见 JSON 形状 (见 _hot_api_snippets docstring)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "data", "list", "results"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    return []


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        # 时间戳 (秒或毫秒, 简单启发)
        v = float(value)
        if v > 1e12:  # ms
            v /= 1000
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


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
    # 优先: 结构化热榜 API (若配了 proactive_hot_api_url). 拿到就用, 不需要按 query.
    # 理由: 热榜数据源本身就是"今日 top-N", V3 分类器再按 user/AI 兴趣挑, 比 tavily
    # 按 query 猜好得多. 未配置时 (env 空) 静默跳过, tavily 顶上.
    text, cands = await _hot_api_snippets(topic or "")
    if text:
        return TrendingLoadResult(
            text=text, cache_hit=False, provider="hot_api", candidates=cands,
        )

    # Fallback: tavily / brave 通用搜索
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
