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
