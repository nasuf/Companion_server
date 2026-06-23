from app.services.offline.activity_generation import (
    _fallback_card,
    _search_query,
    _usable_results,
)
from app.services.offline.providers.search import SearchResult


def test_search_query_localizes_zhenjiang_for_chinese_sources():
    query = _search_query("Zhenjiang", ["音乐爱好者"])

    assert "江苏 镇江" in query
    assert "Zhenjiang" in query
    assert "音乐爱好者" in query


def test_tripadvisor_generic_review_source_is_not_usable_activity_source():
    result = SearchResult(
        title="THE BEST Free Things to Do in Zhenjiang (2026) - Tripadvisor",
        url="https://www.tripadvisor.com/Attractions-g297444-Activities-zft11292-Zhenjiang_Jiangsu.html",
        content=(
            "Highly rated activities with free entry in Zhenjiang. "
            "We had dinner at the Paulaner, which had live music and outdoor"
        ),
        score=0.5,
    )

    assert _usable_results([result], "Zhenjiang") == []


def test_fallback_card_does_not_promote_unverified_source_title_to_place():
    result = SearchResult(
        title="THE BEST Free Things to Do in Zhenjiang (2026) - Tripadvisor",
        url="https://www.tripadvisor.com/Attractions-g297444-Activities-zft11292-Zhenjiang_Jiangsu.html",
        content="We had dinner at the Paulaner, which had live music and outdoor",
    )

    card = _fallback_card("Zhenjiang", ["音乐爱好者"], [result])

    assert "Paulaner" not in card["title"]
    assert card["location_name"] == "镇江"
