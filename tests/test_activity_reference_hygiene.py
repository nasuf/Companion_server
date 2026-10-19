"""Old public reviews must not become current offers or attendance promises."""
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from app.services.offline import place_context
from app.services.offline.copy_guards import copy_issues
from app.services.offline.reference_hygiene import factual_reference
from app.services.offline.providers.cleversee import WebSearchResults
from app.services.offline.providers.search import SearchResult

NOW = datetime.fromisoformat("2026-10-19T13:47:00+08:00")
CARD = {"location_name": "春树·花艺(常发广场店)", "city": "镇江市", "address": "黄山西路常发广场5号楼102",
        "discovery_metadata": {"kind": "place", "price_info": "138.00", "place_introduction": "花艺小店"}}
STABLE = "镇江常发广场有一家春树·花艺。\n项目：花艺沙龙、香薰蜡烛DIY、下午茶。\n地址：黄山西路常发广场5号楼102。"
OLD = "鲜花盲盒9.9元。店内有鹦鹉大熊和小狗元宝。店里随便拍都好看。环境安静，人不多。限时抢购。"


def test_preserve_verbatim_project_facts_without_old_deals_animals_or_advertising():
    cleaned, removed = factual_reference(STABLE + "\n" + OLD + "\n" + STABLE)
    assert cleaned == STABLE and removed == 5
    assert factual_reference("\n") == ("", 0)
    # Free admission and actual exhibition information are useful stable facts.
    assert factual_reference("免费开放的文物展厅，有陶瓷展品。")[0] == "免费开放的文物展厅，有陶瓷展品。"


@pytest.mark.parametrize("claim,blocked", [
    ("有9.9元鲜花盲盒。", True), ("蜡烛课程138元。", True),
    ("人均138元，课程138元。", True), ("人均参考约138元。", False),
    ("人均￥138.00左右，具体以门店为准。", False),
    ("人均参考9.9元。", True), ("店内有鹦鹉和小狗。", True),
    ("店里随便拍都好看。", True), ("随手出大片。", True),
    ("可以按现场规则拍照。", False), ("可以了解花艺沙龙和蜡烛制作。", False),
])
def test_deterministic_guard_rejects_volatile_claims_even_if_legacy_facts_contain_them(claim, blocked):
    issues = copy_issues(claim, CARD, NOW, facts_text=STABLE + OLD)
    assert bool(issues) is blocked


def test_dated_event_ticket_and_native_animal_facts_follow_their_own_evidence_path():
    event = {**CARD, "discovery_metadata": {"kind": "event"}}
    assert not copy_issues("票价20元。", event, NOW, facts_text="票价20元")
    native_pet = {**CARD, "discovery_metadata": {"kind": "place", "place_introduction": "店内有小狗"}}
    assert not copy_issues("店内有小狗。", native_pet, NOW, facts_text="店内有小狗")


async def test_cleaning_after_body_identity_check_keeps_source_image_provenance(monkeypatch):
    original = SearchResult(title="镇江常发广场花店", url="https://example.com/spring", content=STABLE,
        score=.9, image_url="https://example.com/a.jpg", images=[{"url": "https://example.com/b.jpg"}],
        query_images=[{"url": "https://example.com/query.jpg"}])
    search = AsyncMock(return_value=WebSearchResults([original], source_status="available"))
    monkeypatch.setattr(place_context.cleversee, "web_search", search)
    monkeypatch.setattr(place_context.cleversee, "read_page", AsyncMock(return_value=STABLE + OLD))
    card = {**CARD, "discovery_metadata": {"kind": "place"}}
    rows = await place_context.load_context(card, "镇江市", lambda *_: True)
    assert rows and rows[0].content == rows[0].raw_content == STABLE
    assert rows[0].images == original.images and rows[0].query_images == original.query_images
    assert rows[0].image_url == original.image_url and rows[0].score == original.score
    assert original.raw_content == ""  # Immutable upstream/cache value is untouched.
    assert card["discovery_metadata"]["context_filtered_sentence_count"] >= 5
    assert card["discovery_metadata"]["context_source_status"] == "available"
    assert search.await_count <= 2


async def test_entirely_filtered_article_is_not_reported_as_useful_context(monkeypatch):
    body = "镇江常发广场春树·花艺，9.9元盲盒。" * 8
    monkeypatch.setattr(place_context.cleversee, "web_search", AsyncMock(return_value=WebSearchResults([
        SearchResult(title="镇江常发广场春树花艺", url="https://example.com/spring", content=body, raw_content=body)
    ], source_status="available")))
    card = {**CARD, "discovery_metadata": {"kind": "place"}}
    assert await place_context.load_context(card, "镇江市", lambda *_: True) == []
    meta = card["discovery_metadata"]
    assert meta["context_source_status"] == "no_matching_sources" and meta["context_source_count"] == 0
    assert meta["context_filtered_sentence_count"] == 1
