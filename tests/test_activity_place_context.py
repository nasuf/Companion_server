from unittest.mock import AsyncMock

import pytest

from app.services.offline import place_context as context
from app.services.offline.activity_generation import _source_is_usable
from app.services.offline.providers.cleversee import WebSearchResults
from app.services.offline.providers.search import SearchResult

CARD = {"location_name": "春树·花艺(常发广场店)", "city": "镇江市", "address": "黄山西路常发广场5号楼102", "discovery_metadata": {"kind": "place"}}
BODY = "镇江常发广场的一家小店叫春树•花艺。店铺介绍有花艺沙龙、香薰蜡烛与下午茶项目，具体内容可以提前联系门店了解。" * 2


@pytest.mark.parametrize("title,body,matched", [
    ("镇江高颜值花店下午茶体验", BODY, True),
    ("春树花艺(万达店)", BODY, False),
    ("上海花艺下午茶", BODY.replace("镇江", "上海"), False),
    ("镇江咖啡店合集", BODY, False),
    ("查看镇江DIY手工_百度地图搜索结果列表：", BODY, False),
    ("镇江花店", "另一家花店，常发广场里没有春树。", False),
])
def test_brand_city_branch_binding_allows_real_body_evidence_but_not_other_stores(title, body, matched):
    assert context.context_matches(CARD, title, body) is matched


async def test_matching_body_enriches_the_exact_store_and_reuses_one_query(monkeypatch):
    search = AsyncMock(return_value=WebSearchResults([SearchResult(title="镇江高颜值花店下午茶体验", url="https://example.com/place", content=BODY)], source_status="available"))
    monkeypatch.setattr(context.cleversee, "web_search", search)
    monkeypatch.setattr(context.cleversee, "read_page", AsyncMock(return_value=BODY))
    card = {**CARD, "discovery_metadata": {"kind": "place"}}
    rows = await context.load_context(card, "镇江市", _source_is_usable)
    assert len(rows) == 1 and "花艺沙龙" in rows[0].raw_content
    assert card["discovery_metadata"]["context_source_status"] == "available"
    assert search.await_count == 1


@pytest.mark.parametrize("state,expected", [("empty", "empty"), ("failed", "unavailable"), ("not_configured", "not_configured")])
async def test_missing_failed_and_real_empty_sources_are_distinct(monkeypatch, state, expected):
    monkeypatch.setattr(context.cleversee, "web_search", AsyncMock(return_value=WebSearchResults(source_status=state)))
    card = {**CARD, "discovery_metadata": {"kind": "place"}}
    assert await context.load_context(card, "镇江市", _source_is_usable) == []
    assert card["discovery_metadata"]["context_source_status"] == expected
    assert card["discovery_metadata"]["context_search_count"] == 2


async def test_unrelated_results_and_timeouts_do_not_invent_context(monkeypatch):
    search = AsyncMock(return_value=[SearchResult(title="宁波陶艺", url="https://example.com/other", content="其他城市的一家店")])
    monkeypatch.setattr(context.cleversee, "web_search", search)
    card = {**CARD, "discovery_metadata": {"kind": "place"}}
    assert await context.load_context(card, "镇江市", _source_is_usable) == []
    assert card["discovery_metadata"]["context_source_status"] == "no_matching_sources"
    search.side_effect = TimeoutError()
    assert await context.load_context(card, "镇江市", _source_is_usable) == []
    assert card["discovery_metadata"]["context_source_status"] == "timeout"


@pytest.mark.parametrize("payload,state", [({}, "failed"), ({"pageItems": None}, "failed"), ({"pageItems": {}}, "failed"), ({"pageItems": ["invalid"]}, "failed"), ({"pageItems": []}, "empty")])
async def test_provider_malformed_payload_is_not_reported_as_a_real_empty_result(monkeypatch, payload, state):
    monkeypatch.setattr(context.cleversee.settings, "ali_cleversee_api_key", "synthetic-key")
    monkeypatch.setattr(context.cleversee, "_cached", AsyncMock(return_value=payload))
    result = await context.cleversee.web_search("镇江市 花艺")
    assert result == [] and result.source_status == state
