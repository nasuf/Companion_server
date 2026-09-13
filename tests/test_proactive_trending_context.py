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
