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
    """结构化热榜 API 抽象 (proactive_hot_api_url env). 未配置→静默返空.
    配了则 fetch → 解析多种 JSON 形状 → freshness 过滤 → 转 candidates.
    """

    def _setup_env(self, monkeypatch, url="https://hot.example/today"):
        from app.services.proactive import trending_context as tc
        monkeypatch.setattr(tc.settings, "proactive_hot_api_url", url)
        monkeypatch.setattr(tc.settings, "proactive_hot_api_key", "")
        monkeypatch.setattr(tc.settings, "proactive_hot_max_age_days", 7)
        monkeypatch.setattr(tc.settings, "chat_link_search_timeout_s", 8.0)

    @pytest.mark.asyncio
    async def test_returns_empty_when_endpoint_unset(self, monkeypatch):
        from app.services.proactive import trending_context as tc
        monkeypatch.setattr(tc.settings, "proactive_hot_api_url", "")
        text, cands = await tc._hot_api_snippets("任意 topic")
        assert text == "" and cands == ()

    @pytest.mark.asyncio
    async def test_parses_items_shape(self, monkeypatch):
        # 契约形状 A: {"items": [{...}]}
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch)

        class _FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"items": [
                    {"title": "中国足球小将Brava杯", "url": "https://weibo.com/1/2",
                     "platform": "微博"},
                    {"title": "腰乐队新专辑", "url": "https://music.163.com/album/1"},  # 平台外
                    {"title": "iPhone 17 首销", "url": "https://www.bilibili.com/video/BV1a",
                     "platform": "B站"},
                ]}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url): return _FakeResp()

        monkeypatch.setattr(tc.httpx, "AsyncClient", _FakeClient)
        text, cands = await tc._hot_api_snippets("")
        # 只留支持平台的 2 条 (music.163 被平台白名单挡住)
        assert len(cands) == 2
        platforms = {c["platform"] for c in cands}
        assert platforms == {"微博", "B站"}

    @pytest.mark.asyncio
    async def test_parses_bare_list_shape(self, monkeypatch):
        # 契约形状 B: 直接列表
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch)
        class _R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return [{"title": "热点", "url": "https://weibo.com/1/2"}]
        class _C:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url): return _R()
        monkeypatch.setattr(tc.httpx, "AsyncClient", _C)
        text, cands = await tc._hot_api_snippets("")
        assert len(cands) == 1
        assert "热点" in text

    @pytest.mark.asyncio
    async def test_freshness_filter_by_published_at(self, monkeypatch):
        # published_at 早于 max_age_days → 丢
        from app.services.proactive import trending_context as tc
        from datetime import datetime, timedelta, timezone
        self._setup_env(monkeypatch)
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        fresh = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        class _R:
            def raise_for_status(self): pass
            def json(self):
                return {"items": [
                    {"title": "老热搜", "url": "https://weibo.com/1/old",
                     "published_at": old},
                    {"title": "新热搜", "url": "https://weibo.com/1/new",
                     "published_at": fresh},
                ]}
        class _C:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url): return _R()
        monkeypatch.setattr(tc.httpx, "AsyncClient", _C)
        text, cands = await tc._hot_api_snippets("")
        assert len(cands) == 1
        assert cands[0]["title"] == "新热搜"

    @pytest.mark.asyncio
    async def test_url_stale_filter_still_applies(self, monkeypatch):
        # URL 带早年份也丢 (即使 published_at 缺失 / 或说是新的但 URL 里的年份不对)
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch)
        class _R:
            def raise_for_status(self): pass
            def json(self):
                return {"items": [
                    {"title": "老 SEO", "url": "https://weibo.com/1/2020/xxx"},  # 老年份
                    {"title": "新", "url": "https://weibo.com/1/2"},
                ]}
        class _C:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url): return _R()
        monkeypatch.setattr(tc.httpx, "AsyncClient", _C)
        text, cands = await tc._hot_api_snippets("")
        assert len(cands) == 1
        assert cands[0]["title"] == "新"

    @pytest.mark.asyncio
    async def test_network_failure_returns_empty_not_raises(self, monkeypatch):
        # 契约: 任何异常都视为空 (让 tavily 顶上), 不 raise
        from app.services.proactive import trending_context as tc
        self._setup_env(monkeypatch)
        class _C:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url): raise RuntimeError("network down")
        monkeypatch.setattr(tc.httpx, "AsyncClient", _C)
        text, cands = await tc._hot_api_snippets("")
        assert text == "" and cands == ()

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
