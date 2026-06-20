import pytest

from app.services.chat_links.cards import component_card_for_link
from app.services.chat_links import covers as cover_mod
from app.services.chat_links.covers import cache_link_cover
from app.services.chat_links.extraction import (
    LinkMetadata,
    _absolute_http_url,
    _platform_json_metadata,
    _toutiao_metadata_from_api_value,
    _weibo_metadata_from_api_value,
    _weibo_status_id,
    app_url_for_link,
    extract_first_url,
    extract_urls,
    platform_for_url,
)
from app.services.chat_links.prompt import render_user_message_with_link
from app.services.chat_links import repo as repo_mod
from app.services.chat_links.repo import ChatLinkCard, create_or_update_link_card
from app.services.chat_links import recommendation as rec_mod
from app.services.chat_links.recommendation import (
    _brave_search_urls,
    _site_scoped_query,
    _urls_from_search_response,
    _search_endpoint_urls,
    _tavily_search_urls,
    configured_candidate_urls,
    search_provider_configured,
    maybe_prepare_proactive_link_recommendation,
    should_attempt_proactive_link,
)


def test_extracts_supported_app_links_from_shared_text():
    assert extract_first_url("复制这段内容后打开 xhslink.com/a1B2c3，来自小红书") == (
        "https://xhslink.com/a1B2c3"
    )
    assert extract_urls(
        "看这个 v.douyin.com/iLxyz9/ 还有 weibo.com/123/abc 和 zhihu.com/question/1"
    ) == [
        "https://v.douyin.com/iLxyz9/",
        "https://weibo.com/123/abc",
        "https://zhihu.com/question/1",
    ]


def test_maps_required_platforms():
    cases = [
        ("https://xhslink.com/a/b", "小红书"),
        ("https://weibo.com/123/abc", "微博"),
        ("https://www.toutiao.com/article/1", "今日头条"),
        ("https://v.douyin.com/abc", "抖音"),
        ("https://www.zhihu.com/question/1", "知乎"),
    ]
    for url, expected in cases:
        assert platform_for_url(url) == expected


@pytest.mark.parametrize(
    ("url", "platform", "title", "accent"),
    [
        ("https://xhslink.com/a/b", "小红书", "小红书笔记", "#F43F5E"),
        ("https://weibo.com/123/abc", "微博", "微博动态", "#FF8A00"),
        ("https://www.toutiao.com/article/1", "今日头条", "头条文章", "#D7262E"),
        ("https://v.douyin.com/abc", "抖音", "抖音视频", "#111827"),
        ("https://www.zhihu.com/question/1", "知乎", "知乎问答", "#1772F6"),
    ],
)
def test_required_platforms_render_same_card_and_prompt_flow(url, platform, title, accent):
    link = ChatLinkCard(
        id=f"link-{platform}",
        user_id="u1",
        conversation_id="c1",
        message_id=None,
        role="user",
        source_app="platform-test",
        source_url=url,
        final_url=url,
        platform=platform,
        title=title,
        description=f"{platform} 分享说明",
        author="作者",
        image_url="https://img.example/cover.jpg",
        content_text=f"{title} 的正文内容，agent 应该能读到。",
        original_text=f"{title} 的正文内容，agent 应该能读到。",
        summary=f"{title} 摘要",
        status="ready",
        error=None,
        metadata=None,
    )

    card = component_card_for_link(link)
    metadata = card["payload"]
    rendered = render_user_message_with_link("看看这个", {
        **metadata,
        "title": title,
        "author": "作者",
        "content_text": link.content_text,
        "summary": link.summary,
    })

    assert platform_for_url(url) == platform
    assert card["type"] == "external_link"
    assert card["accent"] == accent
    assert metadata["final_url"] == url
    assert metadata["platform"] == platform
    assert "[链接卡片内容]" in rendered
    assert f"平台：{platform}" in rendered
    assert f"标题：{title}" in rendered
    assert "正文：" in rendered


def test_platform_json_metadata_reads_common_embedded_shapes():
    html = """
    <script id="RENDER_DATA" type="application/json">
      {"note":{"title":"周末咖啡馆","desc":"阳光很好，适合坐在窗边。","nickname":"阿宁","urlDefault":"https://img.example/x.jpg"}}
    </script>
    """
    assert _platform_json_metadata(html, "小红书") == {
        "title": "周末咖啡馆",
        "description": "阳光很好，适合坐在窗边。",
        "author": "阿宁",
        "image_url": "https://img.example/x.jpg",
        "content_text": "周末咖啡馆\n\n阳光很好，适合坐在窗边。",
        "original_text": "阳光很好，适合坐在窗边。",
    }


def test_platform_json_metadata_reads_nested_image_lists():
    html = """
    <script id="RENDER_DATA" type="application/json">
      {"article":{"title":"有图文章","content":"正文","imageList":[{"url_list":["https://img.example/cover.jpg"]}]}}
    </script>
    """

    assert _platform_json_metadata(html, "今日头条")["image_url"] == (
        "https://img.example/cover.jpg"
    )


def test_absolute_http_url_normalizes_cover_candidates():
    assert _absolute_http_url("//cdn.example.com/a.jpg", "https://xhslink.com/abc") == (
        "https://cdn.example.com/a.jpg"
    )
    assert _absolute_http_url("/cover.jpg", "https://www.zhihu.com/question/1") == (
        "https://www.zhihu.com/cover.jpg"
    )
    assert _absolute_http_url("data:image/png;base64,abc", "https://example.com") is None


def test_link_card_prompt_rendering_is_model_visible():
    metadata = {
        "id": "link-1",
        "platform": "知乎",
        "title": "一个关于长期记忆的问题",
        "author": "答主",
        "summary": "讨论伴侣型 AI 如何处理长期记忆边界。",
        "final_url": "https://www.zhihu.com/question/1",
        "status": "ready",
    }
    rendered = render_user_message_with_link("你看看这个", metadata)
    assert "你看看这个" in rendered
    assert "[链接卡片内容]" in rendered
    assert "平台：知乎" in rendered
    assert "标题：一个关于长期记忆的问题" in rendered
    assert "摘要：讨论伴侣型 AI 如何处理长期记忆边界。" in rendered
    assert "围绕链接卡片中已读取到的内容回应" in rendered


@pytest.mark.asyncio
async def test_daily_link_groups_ignore_unbound_preview_cards(monkeypatch):
    captured = {}

    async def fake_query_raw(query, *args):
        captured["query"] = query
        captured["args"] = args
        return []

    monkeypatch.setattr(repo_mod.db, "query_raw", fake_query_raw)

    groups = await repo_mod.list_user_link_groups("user-1")

    assert groups == []
    assert "l.message_id IS NOT NULL" in captured["query"]


def test_component_card_for_link_has_openable_payload():
    link = ChatLinkCard(
        id="link-1",
        user_id="u1",
        conversation_id="c1",
        message_id=None,
        role="user",
        source_app="test",
        source_url="https://v.douyin.com/abc",
        final_url="https://www.douyin.com/video/1",
        platform="抖音",
        title="视频标题",
        description="",
        author=None,
        image_url=None,
        content_text="视频标题",
        original_text="视频标题",
        summary="视频标题",
        status="ready",
        error=None,
        metadata=None,
    )
    card = component_card_for_link(link)
    assert card["type"] == "external_link"
    assert card["payload"]["link_id"] == "link-1"
    assert card["payload"]["final_url"] == "https://www.douyin.com/video/1"
    assert card["payload"]["app_url"] == "snssdk1128://aweme/detail/1"


def test_app_url_for_toutiao_prefers_native_detail_scheme():
    assert (
        app_url_for_link(
            platform="今日头条",
            source_url="https://www.toutiao.com/article/7651359327906710016/",
            final_url="https://www.toutiao.com/article/7651359327906710016/?wid=1",
        )
        == "snssdk141://detail?groupid=7651359327906710016"
    )


def test_weibo_ajax_metadata_beats_visitor_page_title():
    metadata = _weibo_metadata_from_api_value(
        {
            "text_raw": "Codex 做游戏的首周销量出了，39份。",
            "user": {"screen_name": "一起 Vibe"},
            "pic_infos": {
                "pic1": {
                    "large": {
                        "url": "https://wx1.sinaimg.cn/large/example.jpg",
                    }
                }
            },
        }
    )

    assert metadata["title"] == "Codex 做游戏的首周销量出了，39份。"
    assert metadata["description"] == "Codex 做游戏的首周销量出了，39份。"
    assert metadata["author"] == "一起 Vibe"
    assert metadata["image_url"] == "https://wx1.sinaimg.cn/large/example.jpg"


def test_toutiao_api_metadata_extracts_first_article_image():
    metadata = _toutiao_metadata_from_api_value(
        {
            "title": "头条标题",
            "source": "上观新闻",
            "content": '<html><body><img src="https://p3-sign.toutiaoimg.com/cover.jpeg" alt="头条标题">正文内容</body></html>',
        }
    )

    assert metadata["title"] == "头条标题"
    assert metadata["author"] == "上观新闻"
    assert metadata["image_url"] == "https://p3-sign.toutiaoimg.com/cover.jpeg"
    assert "正文内容" in metadata["description"]


def test_weibo_status_id_uses_last_numeric_path_segment():
    assert (
        _weibo_status_id("https://weibo.com/2657550845/5311209856304539")
        == "5311209856304539"
    )


def test_proactive_candidate_urls_only_keep_supported_platforms():
    assert configured_candidate_urls(
        "https://xhslink.com/a, https://example.com/nope; https://weibo.com/1/2"
    ) == [
        "https://xhslink.com/a",
        "https://weibo.com/1/2",
    ]


def test_proactive_link_probability_gate(monkeypatch):
    monkeypatch.setattr(rec_mod.settings, "proactive_link_recommendation_enabled", True)
    monkeypatch.setattr(rec_mod.settings, "proactive_link_recommendation_probability", 0.05)

    assert should_attempt_proactive_link(
        trigger_type="silence_wakeup",
        source="greeting",
        random_value=0.01,
    )
    assert not should_attempt_proactive_link(
        trigger_type="scheduled_scene",
        source="ai_schedule",
        random_value=0.01,
    )
    assert not should_attempt_proactive_link(
        trigger_type="silence_wakeup",
        source="music",
        random_value=0.01,
    )
    assert not should_attempt_proactive_link(
        trigger_type="silence_wakeup",
        source="greeting",
        random_value=0.20,
    )


def test_search_response_urls_only_keep_supported_platforms():
    assert _urls_from_search_response(
        {
            "results": [
                {"url": "https://example.com/not-supported"},
                {"url": "https://www.toutiao.com/article/1"},
                "https://v.douyin.com/abc",
            ]
        }
    ) == [
        "https://www.toutiao.com/article/1",
        "https://v.douyin.com/abc",
    ]


def test_search_response_reads_brave_web_shape():
    assert _urls_from_search_response(
        {
            "web": {
                "results": [
                    {"url": "https://not-supported.example/post"},
                    {"url": "https://weibo.com/123/abc"},
                ]
            }
        }
    ) == ["https://weibo.com/123/abc"]


def test_site_scoped_query_limits_supported_domains():
    query = _site_scoped_query("周末咖啡")

    assert query.startswith("周末咖啡 (")
    assert "site:xiaohongshu.com" in query
    assert "site:weibo.com" in query
    assert "site:toutiao.com" in query
    assert "site:douyin.com" in query
    assert "site:zhihu.com" in query


def test_search_provider_configured_reports_missing_keys(monkeypatch):
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_provider", "tavily")
    monkeypatch.setattr(rec_mod.settings, "tavily_api_key", "")
    assert search_provider_configured() == (False, "TAVILY_API_KEY is not set")

    monkeypatch.setattr(rec_mod.settings, "chat_link_search_provider", "brave")
    monkeypatch.setattr(rec_mod.settings, "brave_search_api_key", "")
    assert search_provider_configured() == (False, "BRAVE_SEARCH_API_KEY is not set")

    monkeypatch.setattr(rec_mod.settings, "chat_link_search_provider", "custom")
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_endpoint", "https://search.example/links")
    assert search_provider_configured() == (True, "custom endpoint configured")


async def test_search_endpoint_posts_query_and_filters_response(monkeypatch):
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_provider", "custom")
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_endpoint", "https://search.example/links")
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_api_key", "secret")
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_timeout_s", 3.0)
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {"url": "https://example.com/skip"},
                    {"url": "https://xhslink.com/post"},
                    {"link": "https://www.zhihu.com/question/2"},
                ]
            }

    class FakeClient:
        def __init__(self, *, timeout, headers, trust_env):
            captured["timeout"] = timeout
            captured["headers"] = headers
            captured["trust_env"] = trust_env

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, endpoint, json):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(rec_mod.httpx, "AsyncClient", FakeClient)

    urls = await _search_endpoint_urls(query="周末咖啡")

    assert urls == ["https://xhslink.com/post", "https://www.zhihu.com/question/2"]
    assert captured["endpoint"] == "https://search.example/links"
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert captured["json"]["query"] == "周末咖啡"
    assert captured["json"]["platforms"] == list(rec_mod.SUPPORTED_PLATFORMS)


async def test_search_endpoint_dispatches_tavily_provider(monkeypatch):
    async def fake_tavily_search_urls(*, query):
        assert query == "咖啡"
        return ["https://xhslink.com/coffee"]

    monkeypatch.setattr(rec_mod.settings, "chat_link_search_provider", "tavily")
    monkeypatch.setattr(rec_mod, "_tavily_search_urls", fake_tavily_search_urls)

    assert await _search_endpoint_urls(query="咖啡") == ["https://xhslink.com/coffee"]


async def test_tavily_provider_posts_domain_scoped_query(monkeypatch):
    monkeypatch.setattr(rec_mod.settings, "tavily_api_key", "tvly-secret")
    monkeypatch.setattr(rec_mod.settings, "tavily_search_endpoint", "https://api.tavily.com/search")
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_timeout_s", 4.0)
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {"url": "https://example.com/skip"},
                    {"url": "https://www.zhihu.com/question/9"},
                ]
            }

    class FakeClient:
        def __init__(self, *, timeout, headers, trust_env):
            captured["timeout"] = timeout
            captured["headers"] = headers
            captured["trust_env"] = trust_env

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, endpoint, json):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(rec_mod.httpx, "AsyncClient", FakeClient)

    urls = await _tavily_search_urls(query="长期记忆")

    assert urls == ["https://www.zhihu.com/question/9"]
    assert captured["endpoint"] == "https://api.tavily.com/search"
    assert captured["headers"]["authorization"] == "Bearer tvly-secret"
    assert captured["json"]["max_results"] == 8
    assert captured["json"]["include_domains"] == list(rec_mod._SEARCH_DOMAINS)
    assert captured["json"]["query"] == "长期记忆"


async def test_brave_provider_gets_domain_scoped_query(monkeypatch):
    monkeypatch.setattr(rec_mod.settings, "brave_search_api_key", "brave-secret")
    monkeypatch.setattr(rec_mod.settings, "brave_search_endpoint", "https://api.search.brave.com/res/v1/web/search")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "web": {
                    "results": [
                        {"url": "https://example.com/skip"},
                        {"url": "https://v.douyin.com/abc"},
                    ]
                }
            }

    class FakeClient:
        def __init__(self, *, timeout, headers, trust_env):
            captured["headers"] = headers
            captured["trust_env"] = trust_env

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, endpoint, params):
            captured["endpoint"] = endpoint
            captured["params"] = params
            return FakeResponse()

    monkeypatch.setattr(rec_mod.httpx, "AsyncClient", FakeClient)

    urls = await _brave_search_urls(query="好看的视频")

    assert urls == ["https://v.douyin.com/abc"]
    assert captured["endpoint"] == "https://api.search.brave.com/res/v1/web/search"
    assert captured["headers"]["x-subscription-token"] == "brave-secret"
    assert captured["params"]["count"] == 8
    assert "site:douyin.com" in captured["params"]["q"]


async def test_provider_without_key_returns_empty_without_network(monkeypatch):
    class FailClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("network should not be called without key")

    monkeypatch.setattr(rec_mod.settings, "tavily_api_key", "")
    monkeypatch.setattr(rec_mod.settings, "brave_search_api_key", "")
    monkeypatch.setattr(rec_mod.httpx, "AsyncClient", FailClient)

    assert await _tavily_search_urls(query="咖啡") == []
    assert await _brave_search_urls(query="咖啡") == []


async def test_prepare_proactive_link_recommendation_builds_assistant_card(monkeypatch):
    monkeypatch.setattr(rec_mod.settings, "proactive_link_recommendation_enabled", True)
    monkeypatch.setattr(rec_mod.settings, "proactive_link_recommendation_probability", 1.0)
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_provider", "custom")
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_endpoint", "")
    monkeypatch.setattr(
        rec_mod.settings,
        "proactive_link_candidate_urls",
        "https://www.zhihu.com/question/1",
    )
    monkeypatch.setattr(rec_mod.random, "random", lambda: 0.0)
    monkeypatch.setattr(rec_mod.random, "choice", lambda urls: urls[0])

    async def fake_extract_link_metadata(*, url, shared_text, timeout=12.0):
        return LinkMetadata(
            source_url=url,
            final_url=url,
            platform="知乎",
            title="关于长期记忆的问题",
            summary="讨论伴侣型 AI 如何处理长期记忆边界。",
            content_text="讨论伴侣型 AI 如何处理长期记忆边界。",
        )

    captured = {}

    async def fake_create_or_update_link_card(**kwargs):
        captured.update(kwargs)
        return ChatLinkCard(
            id="link-ai-1",
            user_id=kwargs["user_id"],
            conversation_id=kwargs["conversation_id"],
            message_id=None,
            role=kwargs["role"],
            source_app=kwargs["source_app"],
            source_url=kwargs["metadata"].source_url,
            final_url=kwargs["metadata"].final_url,
            platform=kwargs["metadata"].platform,
            title=kwargs["metadata"].title,
            description="",
            author=None,
            image_url=None,
            content_text=kwargs["metadata"].content_text,
            original_text=kwargs["metadata"].content_text,
            summary=kwargs["metadata"].summary,
            status="ready",
            error=None,
            metadata=kwargs["extra_metadata"],
        )

    monkeypatch.setattr(rec_mod, "extract_link_metadata", fake_extract_link_metadata)
    monkeypatch.setattr(rec_mod, "create_or_update_link_card", fake_create_or_update_link_card)

    result = await maybe_prepare_proactive_link_recommendation(
        user_id="u1",
        conversation_id="c1",
        trigger_type="silence_wakeup",
        source="greeting",
        topic="长期记忆",
        stage="warming",
        message="看到一个东西想到你。",
    )

    assert result is not None
    assert captured["role"] == "assistant"
    assert captured["source_app"] == "proactive_link_recommendation"
    assert captured["extra_metadata"]["candidate_source"] == "configured_pool"
    assert result.component_card["type"] == "external_link"
    assert result.component_card["payload"]["link_id"] == "link-ai-1"
    assert result.link_card_metadata["role"] == "assistant"


async def test_prepare_proactive_link_recommendation_records_search_source(monkeypatch):
    monkeypatch.setattr(rec_mod.settings, "proactive_link_recommendation_enabled", True)
    monkeypatch.setattr(rec_mod.settings, "proactive_link_recommendation_probability", 1.0)
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_provider", "custom")
    monkeypatch.setattr(rec_mod.settings, "chat_link_search_endpoint", "https://search.example/links")
    monkeypatch.setattr(rec_mod.random, "random", lambda: 0.0)
    monkeypatch.setattr(rec_mod.random, "choice", lambda urls: urls[0])

    async def fake_search_endpoint_urls(*, query):
        return ["https://weibo.com/123/abc"]

    async def fake_extract_link_metadata(*, url, shared_text, timeout=12.0):
        return LinkMetadata(
            source_url=url,
            final_url=url,
            platform="微博",
            title="微博动态",
            summary="一条和用户兴趣相关的微博。",
            content_text="一条和用户兴趣相关的微博。",
        )

    captured = {}

    async def fake_create_or_update_link_card(**kwargs):
        captured.update(kwargs)
        return ChatLinkCard(
            id="link-ai-search",
            user_id=kwargs["user_id"],
            conversation_id=kwargs["conversation_id"],
            message_id=None,
            role=kwargs["role"],
            source_app=kwargs["source_app"],
            source_url=kwargs["metadata"].source_url,
            final_url=kwargs["metadata"].final_url,
            platform=kwargs["metadata"].platform,
            title=kwargs["metadata"].title,
            description="",
            author=None,
            image_url=None,
            content_text=kwargs["metadata"].content_text,
            original_text=kwargs["metadata"].content_text,
            summary=kwargs["metadata"].summary,
            status="ready",
            error=None,
            metadata=kwargs["extra_metadata"],
        )

    monkeypatch.setattr(rec_mod, "_search_endpoint_urls", fake_search_endpoint_urls)
    monkeypatch.setattr(rec_mod, "extract_link_metadata", fake_extract_link_metadata)
    monkeypatch.setattr(rec_mod, "create_or_update_link_card", fake_create_or_update_link_card)

    result = await maybe_prepare_proactive_link_recommendation(
        user_id="u1",
        conversation_id="c1",
        trigger_type="memory_proactive",
        source="user_l2",
        topic="咖啡",
        stage="intimate",
        message="我看到一条微博，感觉你会感兴趣。",
    )

    assert result is not None
    assert captured["extra_metadata"]["candidate_source"] == "search_endpoint"
    assert result.component_card["payload"]["platform"] == "微博"


async def test_cache_link_cover_rewrites_remote_image_to_local_media(monkeypatch, tmp_path):
    monkeypatch.setattr(cover_mod.storage, "_MEDIA_DIR", tmp_path)

    async def fake_download_image(url, referer_url=None):
        assert url == "https://sns-webpic-qc.xhscdn.com/cover.jpg"
        assert referer_url == "https://www.xiaohongshu.com/explore/1"
        return b"image-bytes", "image/jpeg"

    monkeypatch.setattr(cover_mod, "_download_image", fake_download_image)
    metadata = LinkMetadata(
        source_url="https://xhslink.com/a",
        final_url="https://www.xiaohongshu.com/explore/1",
        platform="小红书",
        title="小红书笔记",
        image_url="https://sns-webpic-qc.xhscdn.com/cover.jpg",
    )

    result = await cache_link_cover(user_id="user-id", metadata=metadata)

    assert result.metadata.image_url is not None
    assert result.metadata.image_url.startswith("/chat/media/user-id_")
    assert result.extra_metadata["remote_image_url"] == "https://sns-webpic-qc.xhscdn.com/cover.jpg"
    assert result.extra_metadata["cover_cached_url"] == result.metadata.image_url
    assert (tmp_path / result.extra_metadata["cover_storage_key"]).read_bytes() == b"image-bytes"


async def test_cache_link_cover_skips_non_remote_images(monkeypatch):
    async def fail_download_image(url, referer_url=None):
        raise AssertionError("local images should not be downloaded")

    monkeypatch.setattr(cover_mod, "_download_image", fail_download_image)
    metadata = LinkMetadata(
        source_url="https://weibo.com/1",
        final_url="https://weibo.com/1",
        platform="微博",
        title="微博动态",
        image_url="/chat/media/user-id_cover.jpg",
    )

    result = await cache_link_cover(user_id="user-id", metadata=metadata)

    assert result.metadata is metadata
    assert result.extra_metadata == {}


async def test_create_link_card_removes_cached_cover_when_db_write_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_mod.chat_media_storage, "_MEDIA_DIR", tmp_path)
    storage_key = "user-id_cached.jpg"
    (tmp_path / storage_key).write_bytes(b"image-bytes")
    metadata = LinkMetadata(
        source_url="https://xhslink.com/a",
        final_url="https://www.xiaohongshu.com/explore/1",
        platform="小红书",
        title="小红书笔记",
        image_url="/chat/media/user-id_cached.jpg",
    )

    async def fake_cache_link_cover(*, user_id, metadata):
        return cover_mod.CachedCoverResult(
            metadata=metadata,
            extra_metadata={"cover_storage_key": storage_key},
        )

    async def fail_query_raw(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(repo_mod, "cache_link_cover", fake_cache_link_cover)
    monkeypatch.setattr(repo_mod.db, "query_raw", fail_query_raw)

    with pytest.raises(RuntimeError):
        await create_or_update_link_card(
            user_id="user-id",
            conversation_id="conv-id",
            metadata=metadata,
        )

    assert not (tmp_path / storage_key).exists()
