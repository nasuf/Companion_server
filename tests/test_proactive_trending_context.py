from unittest.mock import AsyncMock, patch

import pytest

from app.services.proactive.admin_test import AdminProactiveTestOptions
from app.services.proactive.trending_context import (
    TrendingLoadResult,
    append_trending_section,
    load_trending_context,
    resolve_trending_context,
)
from app.services.proactive.trending_cache import cache_key_for_topic


@pytest.mark.asyncio
async def test_append_trending_section():
    prompt = "【任务】问候用户"
    out = await append_trending_section(prompt, "- 热点A\n- 热点B")
    assert "热点A" in out
    assert out.startswith(prompt)


@pytest.mark.asyncio
async def test_resolve_trending_context_admin_force():
    loaded = TrendingLoadResult(text="- 测试热点", provider="tavily")
    with patch(
        "app.services.proactive.trending_context.load_trending_context",
        new_callable=AsyncMock,
        return_value=loaded,
    ) as mock_load:
        text, attached, meta = await resolve_trending_context(
            "silence_wakeup",
            topic="公共话题",
            admin_test_options=AdminProactiveTestOptions(
                use_web_search=True,
                use_link_card=False,
            ),
        )

    assert attached is True
    assert "测试热点" in text
    assert meta is loaded
    mock_load.assert_awaited_once_with(topic="公共话题", bypass_cache=True)


@pytest.mark.asyncio
async def test_resolve_trending_context_admin_off():
    with patch(
        "app.services.proactive.trending_context.load_trending_context",
        new_callable=AsyncMock,
    ) as mock_load:
        text, attached, meta = await resolve_trending_context(
            "silence_wakeup",
            topic="公共话题",
            admin_test_options=AdminProactiveTestOptions(
                use_web_search=False,
                use_link_card=True,
            ),
        )

    assert attached is False
    assert text == ""
    assert meta is None
    mock_load.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_trending_context_cache_hit():
    with patch(
        "app.services.proactive.trending_context.get_cached_trending",
        new_callable=AsyncMock,
        return_value="- cached line",
    ) as mock_get:
        with patch(
            "app.services.proactive.trending_context._fetch_live_trending_snippets",
            new_callable=AsyncMock,
        ) as mock_fetch:
            result = await load_trending_context(topic="公共话题")

    assert result.cache_hit is True
    assert result.provider == "cache"
    assert "cached line" in result.text
    mock_get.assert_awaited_once()
    mock_fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_trending_context_live_fetch_and_write():
    loaded = TrendingLoadResult(text="- live line", provider="brave")
    with patch(
        "app.services.proactive.trending_context.get_cached_trending",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with patch(
            "app.services.proactive.trending_context._fetch_live_trending_snippets",
            new_callable=AsyncMock,
            return_value=loaded,
        ):
            with patch(
                "app.services.proactive.trending_context.set_cached_trending",
                new_callable=AsyncMock,
            ) as mock_set:
                result = await load_trending_context(topic="公共话题")

    assert result.provider == "brave"
    mock_set.assert_awaited_once()


def test_cache_key_for_topic():
    assert cache_key_for_topic(None) == "proactive:trending:global:v1"
    assert cache_key_for_topic("公共话题") == "proactive:trending:global:v1"
    assert cache_key_for_topic("某具体话题").startswith("proactive:trending:topic:")


# ─── 平台白名单过滤 (2026-09-14): tophub 类聚合站不应进候选 ────────────────


class TestPlatformWhitelist:
    """URL 不在支持平台名单里的 → 直接丢, 不进 candidates.

    修的是: tavily 抓"微博热搜"经常返 tophub.today 等聚合站 URL, 之前 platform=""
    但仍进 candidates + LLM prompt + 卡片. 用户截图里"链接卡片是 tophub.today"就是
    这个漏洞. 现在白名单里没有的一律扔掉, 若整批 tavily 结果全是聚合站, 上游
    text 也会空 → trending_attached=False → 主动消息干净落 silence_plain.
    """

    def _sr(self, url, title="t", content="c"):
        from app.services.offline.providers.search import SearchResult
        return SearchResult(title=title, url=url, content=content)

    def test_url_platform_all_supported(self):
        from app.services.proactive.trending_context import _url_platform
        # 全 6 个支持平台各查一遍, 覆盖主域和常见短链
        assert _url_platform("https://weibo.com/1/2") == "微博"
        assert _url_platform("https://s.weibo.cn/foo") == "微博"
        assert _url_platform("https://m.weibo.cn/status/1") == "微博"
        assert _url_platform("https://www.xiaohongshu.com/explore/1") == "小红书"
        assert _url_platform("https://xhslink.com/abc") == "小红书"
        assert _url_platform("https://www.bilibili.com/video/BV1a") == "B站"
        assert _url_platform("https://b23.tv/abc") == "B站"
        assert _url_platform("https://www.zhihu.com/question/1") == "知乎"
        assert _url_platform("https://zhuanlan.zhihu.com/p/1") == "知乎"
        assert _url_platform("https://www.douyin.com/video/1") == "抖音"
        assert _url_platform("https://www.iesdouyin.com/share/video/1") == "抖音"
        assert _url_platform("https://www.toutiao.com/article/1") == "头条"

    def test_url_platform_case_insensitive(self):
        from app.services.proactive.trending_context import _url_platform
        assert _url_platform("HTTPS://WWW.BILIBILI.COM/x") == "B站"

    def test_url_platform_rejects_aggregators(self):
        # tophub 是典型爬取各站热榜的聚合站, 是我们要拦的主要目标
        from app.services.proactive.trending_context import _url_platform
        for aggregator in [
            "https://tophub.today/",
            "https://tophub.today/n/xxxxx",
            "https://36kr.com/hot/",
            "https://www.jiemian.com/lists/91.html",
            "https://news.sohu.com/",
            "https://www.example.com/some-article",
            "",
            "not a url",
        ]:
            assert _url_platform(aggregator) == "", f"{aggregator} 不该被识别为支持平台"

    def test_filter_drops_unsupported(self):
        from app.services.proactive.trending_context import _filter_to_supported_platforms
        got = _filter_to_supported_platforms([
            self._sr("https://tophub.today/n/1", title="中国足球小将西班牙捧杯"),
            self._sr("https://www.bilibili.com/video/BV1a", title="真视频"),
            self._sr("https://weibo.com/1/2", title="真微博"),
            self._sr("https://36kr.com/hot", title="聚合"),
        ])
        assert len(got) == 2
        assert all(("bilibili" in r.url or "weibo" in r.url) for r in got)

    def test_filter_empty_input_returns_empty(self):
        from app.services.proactive.trending_context import _filter_to_supported_platforms
        assert _filter_to_supported_platforms([]) == []

    def test_search_results_to_candidates_double_defense(self):
        """即使 caller 没跑过 _filter, _search_results_to_candidates 也不该
        把 platform="" 的杂鱼放进 candidates —— 兜底不被 tophub 污染."""
        from app.services.proactive.trending_context import (
            _search_results_to_candidates,
        )
        got = _search_results_to_candidates([
            self._sr("https://tophub.today/x", title="聚合"),
            self._sr("https://weibo.com/1/2", title="真微博", content="内容"),
        ])
        assert len(got) == 1
        assert got[0]["platform"] == "微博"
        assert "tophub" not in got[0]["url"]


# ─── freshness filter + hot API 数据源 (2026-09-14 task#6) ───────────────────


class TestStaleUrlFilter:
    """URL path 里带明显早年份 → 视为老 SEO 内容, 丢. 修 tavily 常返 2021 老列表.

    只是 URL 启发, 但对 "50个热门话题-2021.html" / "/article/6800000/2020/x" 这类
    命名规范的旧文命中率高. 未匹配到年份则不判定, 交上游.
    """

    def test_drops_url_with_old_year_in_path(self):
        from app.services.proactive.trending_context import _url_looks_stale
        assert _url_looks_stale("https://www.zhihu.com/p/2020/xxx", max_age_days=7)
        assert _url_looks_stale("https://article.example/2018/03/story", max_age_days=7)
        assert _url_looks_stale("https://sohu.com/a/list-2021-hot", max_age_days=7)

    def test_keeps_url_with_recent_year(self):
        from app.services.proactive.trending_context import _url_looks_stale
        # 当前年份视为 fresh (不管月份)
        this_year = "2026"
        assert not _url_looks_stale(
            f"https://weibo.com/{this_year}/story/abc", max_age_days=7,
        )

    def test_keeps_url_without_year_marker(self):
        # 常规 URL (weibo/xhs/bilibili) 通常 path 里没年份 → 不判定, 保留
        from app.services.proactive.trending_context import _url_looks_stale
        assert not _url_looks_stale(
            "https://www.bilibili.com/video/BV1abc", max_age_days=7,
        )
        assert not _url_looks_stale(
            "https://xhslink.com/a/xyz", max_age_days=7,
        )
        assert not _url_looks_stale("", max_age_days=7)

    def test_zero_max_age_disables_filter(self):
        # max_age_days=0 → 关掉 filter, 任何 URL 都返 False
        from app.services.proactive.trending_context import _url_looks_stale
        assert not _url_looks_stale(
            "https://sohu.com/2015/old", max_age_days=0,
        )


class TestHotApiSnippets:
    """DailyHot 6 平台 fanout (proactive_hot_api_url = base URL, 未设→静默返空).

    每平台一个 endpoint 并发拉, 各平台 top-N 汇总, 按 slug 映射打 platform 标签.
    单平台失败不阻塞其它平台 (return_exceptions=False 但内部 try/catch).
    """

    def _setup_env(self, monkeypatch, base_url="https://hot.example"):
        from app.services.proactive import trending_context as tc
        monkeypatch.setattr(tc.settings, "proactive_hot_api_url", base_url)
        monkeypatch.setattr(tc.settings, "proactive_hot_api_key", "")
        monkeypatch.setattr(tc.settings, "proactive_hot_max_age_days", 7)
        monkeypatch.setattr(tc.settings, "chat_link_search_timeout_s", 8.0)

    def _install_router_client(self, monkeypatch, per_platform_responses: dict):
        """安装一个 httpx.AsyncClient mock, 按 URL path 里的 slug 返对应响应.

        per_platform_responses: {slug: dict_or_exception}
        - dict → 作为 .json() 返回
        - Exception 实例 → get() 会 raise 它 (模拟单平台挂)
        - slug 不在 map 里 → 返 {"data": []}
        """
        from app.services.proactive import trending_context as tc

        class _R:
            def __init__(self, payload): self._p = payload
            def raise_for_status(self): pass
            def json(self): return self._p

        class _C:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url):
                # URL 是 "{base}/{slug}"; 取最后一段做 slug
                slug = str(url).rstrip("/").rsplit("/", 1)[-1]
                resp_or_exc = per_platform_responses.get(slug, {"data": []})
                if isinstance(resp_or_exc, Exception):
                    raise resp_or_exc
                return _R(resp_or_exc)

        monkeypatch.setattr(tc.httpx, "AsyncClient", _C)

    @pytest.mark.asyncio
    async def test_returns_empty_when_endpoint_unset(self, monkeypatch):
        from app.services.proactive import trending_context as tc
        monkeypatch.setattr(tc.settings, "proactive_hot_api_url", "")
        text, cands = await tc._hot_api_snippets("任意 topic")
        assert text == "" and cands == ()

    @pytest.mark.asyncio
    async def test_dailyhot_fanout_tags_platform_per_slug(self, monkeypatch):
        """DailyHot 里各平台响应不带 platform 字段, 我们按 slug 映射打标 —
        这样即使 URL 是 s.weibo.com/... (原本 _url_platform 兜底可能识别) 也不依赖."""
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch)
        self._install_router_client(monkeypatch, {
            "weibo": {"data": [
                {"title": "中国足球小将Brava杯", "url": "https://s.weibo.com/weibo?q=x"},
            ]},
            "bilibili": {"data": [
                {"title": "手机拍夜景技巧", "url": "https://www.bilibili.com/video/BV1a"},
            ]},
            "zhihu": {"data": [{"title": "长期记忆边界", "url": "https://www.zhihu.com/q/1"}]},
            # xhs/toutiao/douyin 走默认空
        })
        text, cands = await tc._hot_api_snippets("")
        assert len(cands) == 3
        platforms = sorted(c["platform"] for c in cands)
        assert platforms == ["B站", "微博", "知乎"]  # sort 后中文按 unicode

    @pytest.mark.asyncio
    async def test_single_platform_failure_does_not_block_others(self, monkeypatch):
        """DailyHot 6 平台并发, 单个失败/超时不阻塞其它 → 至少 1 个成功即算成功."""
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch)
        self._install_router_client(monkeypatch, {
            "weibo": RuntimeError("weibo endpoint down"),   # 挂
            "zhihu": {"data": [{"title": "知乎热榜", "url": "https://www.zhihu.com/q/1"}]},
            "bilibili": Exception("timeout"),               # 挂
            # 其它默认空
        })
        text, cands = await tc._hot_api_snippets("")
        assert len(cands) == 1
        assert cands[0]["platform"] == "知乎"

    @pytest.mark.asyncio
    async def test_all_platforms_fail_returns_empty(self, monkeypatch):
        """全部 6 个平台挂 → 返 ("", ()), tavily 顶上."""
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch)
        self._install_router_client(monkeypatch, {
            slug: RuntimeError("all down") for slug in
            ("weibo", "zhihu", "bilibili", "xhs", "toutiao", "douyin")
        })
        text, cands = await tc._hot_api_snippets("")
        assert text == "" and cands == ()

    @pytest.mark.asyncio
    async def test_freshness_filter_by_published_at(self, monkeypatch):
        # published_at 早于 max_age_days → 丢
        from app.services.proactive import trending_context as tc
        from datetime import datetime, timedelta, timezone
        self._setup_env(monkeypatch)
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        fresh = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        self._install_router_client(monkeypatch, {
            "weibo": {"data": [
                {"title": "老热搜", "url": "https://weibo.com/1/old", "published_at": old},
                {"title": "新热搜", "url": "https://weibo.com/1/new", "published_at": fresh},
            ]},
        })
        text, cands = await tc._hot_api_snippets("")
        assert len(cands) == 1
        assert cands[0]["title"] == "新热搜"

    @pytest.mark.asyncio
    async def test_url_stale_filter_still_applies(self, monkeypatch):
        # URL 里带早年份也丢
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch)
        self._install_router_client(monkeypatch, {
            "weibo": {"data": [
                {"title": "老 SEO", "url": "https://weibo.com/1/2020/xxx"},
                {"title": "新", "url": "https://weibo.com/1/2"},
            ]},
        })
        text, cands = await tc._hot_api_snippets("")
        assert len(cands) == 1
        assert cands[0]["title"] == "新"

    @pytest.mark.asyncio
    async def test_uses_mobile_url_when_url_missing(self, monkeypatch):
        # DailyHot 部分平台只带 mobileUrl (weibo 详情页有时是), url 缺就用它
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch)
        self._install_router_client(monkeypatch, {
            "weibo": {"data": [
                {"title": "热搜", "mobileUrl": "https://m.weibo.cn/status/1"},
            ]},
        })
        text, cands = await tc._hot_api_snippets("")
        assert len(cands) == 1
        assert "m.weibo" in cands[0]["url"]

    @pytest.mark.asyncio
    async def test_trailing_slash_in_base_url_stripped(self, monkeypatch):
        """admin 设 base URL 带 / 结尾也不该出 // 双斜杠."""
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch, base_url="https://hot.example/")
        seen_urls: list[str] = []

        class _R:
            def raise_for_status(self): pass
            def json(self): return {"data": []}
        class _C:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url):
                seen_urls.append(str(url))
                return _R()
        monkeypatch.setattr(tc.httpx, "AsyncClient", _C)

        await tc._hot_api_snippets("")
        # 每个 URL 都不该有 "//" (除了 https://)
        for u in seen_urls:
            after_protocol = u.split("://", 1)[-1]
            assert "//" not in after_protocol, f"double slash in {u}"

    def test_extract_hot_items_handles_all_shapes(self):
        from app.services.proactive.trending_context import _extract_hot_items
        # 契约里说的 3 种 + 变体
        assert _extract_hot_items({"items": [1, 2]}) == [1, 2]
        assert _extract_hot_items({"data": [1, 2]}) == [1, 2]
        assert _extract_hot_items({"list": [1, 2]}) == [1, 2]
        assert _extract_hot_items({"results": [1, 2]}) == [1, 2]
        assert _extract_hot_items([1, 2]) == [1, 2]
        assert _extract_hot_items({"other": "x"}) == []
        assert _extract_hot_items(None) == []
        assert _extract_hot_items("string") == []

    def test_parse_iso_variants(self):
        from app.services.proactive.trending_context import _parse_iso
        from datetime import datetime, timezone
        # ISO 字符串
        assert _parse_iso("2026-09-14T13:00:00Z") == datetime(2026, 9, 14, 13, 0, tzinfo=timezone.utc)
        # 秒级 timestamp
        got = _parse_iso(1700000000)
        assert got is not None and got.tzinfo is not None
        # 毫秒级
        got_ms = _parse_iso(1700000000000)
        assert got_ms is not None and got_ms.year >= 2023
        # 已 datetime
        d = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert _parse_iso(d) == d
        # 空 / 垃圾
        assert _parse_iso(None) is None
        assert _parse_iso("") is None
        assert _parse_iso("not a date") is None
