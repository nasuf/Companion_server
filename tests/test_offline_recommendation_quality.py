import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.offline import activity_generation as generation
from app.services.offline.providers import cleversee
from app.services.offline.recommendation_content import fallback_detail, useful_detail

MANIFEST = json.loads(
    (
        Path(__file__).parents[1]
        / "scripts/prompt_releases/20261008_activity_detail_quality.json"
    ).read_text()
)
PROMPTS = {p["key"]: p["content"] for p in MANIFEST["prompts"]}
DETAIL = (
    "🌿 换一种放松方式\n这是一处可以了解手作体验的小店。如果最近想把注意力从屏幕上挪开，可以先想想自己更喜欢哪一种手作方向，不用急着把所有项目都安排上。\n\n"
    "💡 可以怎么开始\n到店后问问当天可以参与的项目、用时和价格，再挑一项自己感兴趣的。想慢慢做就留出充足时间，不用把休息变成赶进度。\n\n"
    "🧭 出门小提示\n地点在万达广场A幢2315室。营业信息为周二至周日10:00—21:00；人均消费参考约29元，单个项目价格以店内为准，出发前也可以问问是否需要预约。"
)


def place(identity="one", **overrides):
    return dict(
        candidate_id="poi:" + identity,
        native_poi_id=identity,
        title="海棠陶艺",
        location_name="海棠陶艺",
        category="手作与小店",
        city="镇江市",
        address="万达广场A幢2315室",
        summary="可以去海棠陶艺看看",
        description="海棠陶艺，万达广场A幢2315室。",
        official_url="https://official.example/place",
        native_images=[{"url": "https://photos.example/one.jpg"}],
        discovery_metadata={
            "kind": "place",
            "opening_hours": "周二至周日10:00—21:00",
            "price_info": "29.00",
        },
        **overrides,
    )


@pytest.fixture
def quality(monkeypatch):
    async def prompt(key):
        return PROMPTS.get(key, "核验事实：{facts_json}\n文案：{copy_text}")

    monkeypatch.setattr(generation, "get_prompt_text", prompt)
    monkeypatch.setattr(generation, "get_chat_model", lambda: object())
    monkeypatch.setattr(generation, "get_utility_model", lambda: object())
    monkeypatch.setattr(cleversee, "web_search", AsyncMock(return_value=[]))
    persist = AsyncMock(return_value=["/offline/media/place_good.jpg"])
    monkeypatch.setattr(generation, "persist_activity_images", persist)
    return persist


async def generate(candidates):
    return await generation._native_card(
        candidates,
        user_id="test",
        city="镇江市",
        search_anchor="镇江市",
        tags=["手作"],
        memory="",
        recent=[],
    )


@pytest.mark.parametrize(
    "failure", ["empty", "short", "rejected", "checker_error", "generation_error"]
)
async def test_failed_detail_never_collapses_to_an_address(
    quality, monkeypatch, failure
):
    selection = json.dumps(
        {"candidates": [{"candidate_id": "poi:one", "summary": "全天免费参加手作"}]},
        ensure_ascii=False,
    )
    detail = (
        "" if failure == "empty" else "可以去看看" if failure == "short" else DETAIL
    )
    monkeypatch.setattr(
        generation,
        "invoke_text",
        AsyncMock(
            side_effect=[
                selection,
                TimeoutError()
                if failure == "generation_error"
                else json.dumps({"text": detail}, ensure_ascii=False),
            ]
        ),
    )
    check = AsyncMock(
        side_effect=TimeoutError() if failure == "checker_error" else None,
        return_value={"supported": failure not in {"rejected", "checker_error"}},
    )
    monkeypatch.setattr(generation, "invoke_json", check)
    result = await generate([place()])
    assert useful_detail(result["description"])
    assert "可以怎么安排" in result["description"] and "29.00" in result["description"]
    assert "全天免费" not in result["summary"] + result["description"]
    assert result["image_urls"] == ["/offline/media/place_good.jpg"]
    assert result["discovery_metadata"]["copy_status"].startswith("fallback_")
    if failure in {"empty", "short", "generation_error"}:
        check.assert_not_awaited()


async def test_verified_summary_and_full_detail_use_same_native_facts(
    quality, monkeypatch
):
    summary = "想试试手作的话，可以先挑一项自己感兴趣的体验。"
    monkeypatch.setattr(
        generation,
        "invoke_text",
        AsyncMock(
            side_effect=[
                json.dumps(
                    {"candidates": [{"candidate_id": "poi:one", "summary": summary}]}
                ),
                json.dumps({"text": DETAIL}),
            ]
        ),
    )
    check = AsyncMock(return_value={"supported": True})
    monkeypatch.setattr(generation, "invoke_json", check)
    result = await generate([place()])
    assert result["summary"] == summary and result["description"] == DETAIL
    assert result["title"] == result["location_name"] == "海棠陶艺"
    assert result["discovery_metadata"]["copy_status"] == "verified"
    prompt = generation.invoke_text.call_args_list[1].args[1]
    assert "29.00" in prompt and "周二至周日" in prompt
    assert summary in check.call_args.args[1] and DETAIL in check.call_args.args[1]


async def test_empty_gallery_uses_illustrated_same_category_without_leaking_original_copy(
    quality, monkeypatch
):
    first = place()
    first["native_images"] = []
    second = place("two")
    second.update(title="唐唐陶艺", location_name="唐唐陶艺", address="吾悦广场二楼")
    quality.side_effect = [[], ["/offline/media/place_two.jpg"]]
    monkeypatch.setattr(
        generation,
        "invoke_text",
        AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "candidates": [
                            {"candidate_id": "poi:one", "summary": "海棠陶艺适合你"}
                        ]
                    }
                ),
                json.dumps({"text": DETAIL}),
            ]
        ),
    )
    monkeypatch.setattr(
        generation, "invoke_json", AsyncMock(return_value={"supported": False})
    )
    result = await generate([first, second])
    assert result["native_poi_id"] == "two" and result["image_urls"] == [
        "/offline/media/place_two.jpg"
    ]
    assert (
        "唐唐陶艺" in result["summary"]
        and "海棠" not in result["summary"] + result["description"]
    )
    assert result["search_sources"][0]["poi_id"] == "two"


async def test_partial_gallery_stays_with_selected_place(quality, monkeypatch):
    monkeypatch.setattr(generation, "invoke_text", AsyncMock(return_value="{}"))
    result = await generate([place(), place("two")])
    assert result["native_poi_id"] == "one"
    quality.assert_awaited_once()


async def test_event_with_no_photos_keeps_its_venue_and_session(quality, monkeypatch):
    first = place()
    first.update(title="秋日音乐会", candidate_id="event:session", native_images=[])
    first["discovery_metadata"] = {
        "kind": "event",
        "event": {"description": "秋日音乐会在海棠陶艺举办"},
    }
    quality.return_value = []
    monkeypatch.setattr(generation, "invoke_text", AsyncMock(return_value="{}"))
    result = await generate([first, place("two")])
    assert result["candidate_id"] == "event:session" and result["title"] == "秋日音乐会"
    assert useful_detail(result["description"])
    quality.assert_awaited_once()


def test_fallback_does_not_invent_history_facilities_or_event_ticket_price():
    card = place()
    card.update(
        location_name="镇江江边自来水厂旧址",
        title="镇江江边自来水厂旧址",
        category="水边散步",
    )
    card["discovery_metadata"] = {"kind": "place"}
    text = fallback_detail(card)
    assert (
        useful_detail(text)
        and "1934" not in text
        and "免费" not in text
        and "厂房里" not in text
    )
    card["discovery_metadata"] = {
        "kind": "event",
        "price_info": "29",
        "event": {"description": "秋日音乐会"},
    }
    assert "29元" not in fallback_detail(card)
