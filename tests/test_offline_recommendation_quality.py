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
    async def review(_model, prompt):
        if failure == "checker_error":
            raise TimeoutError()
        # The invented "全天免费" summary is rejected on its own merits now.
        if "全天免费" in prompt:
            return {"supported": False, "unsupported_claims": ["全天免费没有依据"]}
        return {"supported": failure != "rejected"}

    check = AsyncMock(side_effect=review)
    monkeypatch.setattr(generation, "invoke_json", check)
    result = await generate([place()])
    assert useful_detail(result["description"])
    assert "可以怎么安排" in result["description"] and "29.00" in result["description"]
    assert "全天免费" not in result["summary"] + result["description"]
    assert result["image_urls"] == ["/offline/media/place_good.jpg"]
    assert result["discovery_metadata"]["copy_status"].startswith("fallback_")
    assert result["discovery_metadata"]["card_copy_status"].startswith("fallback_")
    # Code decides "全天免费" is unsupported before any model review.
    issues = result["discovery_metadata"]["card_copy_issues"]
    assert any("全天" in item for item in issues) and any("免费" in item for item in issues)
    if failure in {"empty", "short", "generation_error"}:
        check.assert_not_awaited()  # An unusable detail is never reviewed either.


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
    assert result["discovery_metadata"]["card_copy_status"] == "verified"
    prompt = generation.invoke_text.call_args_list[1].args[1]
    assert "29.00" in prompt and "周二至周日" in prompt
    reviewed = [call.args[1] for call in check.await_args_list]
    # Card copy and detail are reviewed separately, against the same facts.
    assert [summary in r for r in reviewed].count(True) == 1
    assert [DETAIL in r for r in reviewed].count(True) == 1
    assert all("周二至周日" in r for r in reviewed)


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


# --- Card copy and detail no longer share one verdict (2026-10-17) ---

GROUNDED = {
    p["key"]: p["content"]
    for p in json.loads(
        (Path(__file__).parents[1] / "scripts/prompt_releases/20261017_activity_copy_grounding.json").read_text()
    )["prompts"]
}


def card_review(prompt):
    return "推荐摘要" in prompt


async def test_card_copy_survives_a_rejected_detail(quality, monkeypatch):
    selection = {"candidate_id": "poi:one", "summary": "想把注意力从屏幕上挪开，可以去海棠陶艺挑一项手作。",
                 "vibe": "安静专注", "suitable": "做陶艺、捏小物件"}
    monkeypatch.setattr(generation, "invoke_text", AsyncMock(side_effect=[
        json.dumps({"candidates": [selection]}, ensure_ascii=False), json.dumps({"text": DETAIL})]))

    async def review(_model, prompt):
        if card_review(prompt):
            return {"supported": True, "unsupported_claims": []}
        return {"supported": False, "unsupported_claims": ["“专治纠结”没有依据"]}

    monkeypatch.setattr(generation, "invoke_json", AsyncMock(side_effect=review))
    result = await generate([place()])
    assert result["summary"] == selection["summary"]
    assert result["vibe"] == "安静专注" and result["suitable"] == "做陶艺、捏小物件"
    assert "可以怎么安排" in result["description"]  # Only the detail fell back.
    meta = result["discovery_metadata"]
    assert (meta["card_copy_status"], meta["copy_status"]) == ("verified", "fallback_rejected")
    assert meta["copy_issues"] == ["“专治纠结”没有依据"] and meta["card_copy_issues"] == []


async def test_card_copy_claims_are_reviewed_against_dated_references(quality, monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    monkeypatch.setattr(generation, "local_now",
                        lambda: datetime(2026, 10, 13, 9, tzinfo=ZoneInfo("Asia/Shanghai")))
    selection = {"candidate_id": "poi:one", "summary": "头疼好些了的话，可以去海棠陶艺慢慢做点手作。",
                 "user_relevance": ["用户头疼好一些了"]}
    monkeypatch.setattr(generation, "invoke_text", AsyncMock(side_effect=[
        json.dumps({"candidates": [selection]}, ensure_ascii=False), "{}"]))
    check = AsyncMock(return_value={"supported": False, "unsupported_claims": ["10-10 的身体状态被写成现在"]})
    monkeypatch.setattr(generation, "invoke_json", check)
    result = await generation._native_card(
        [place()], user_id="test", city="镇江市", search_anchor="镇江市", tags=[], memory="", recent=[],
        evidence={"memory": ["[2026-10-10] 用户头疼好一些了"]})
    reviewed = check.await_args.args[1]
    assert "[2026-10-10] 用户头疼好一些了" in reviewed and "2026-10-13（星期二）" in reviewed
    assert "头疼" not in result["summary"]
    assert result["discovery_metadata"]["card_copy_issues"] == ["10-10 的身体状态被写成现在"]


async def test_detail_is_repaired_once_with_feedback_when_the_prompt_supports_it(quality, monkeypatch):
    async def prompt(key):
        return GROUNDED.get(key) or PROMPTS.get(key, "核验事实：{facts_json}\n文案：{copy_text}")

    monkeypatch.setattr(generation, "get_prompt_text", prompt)
    together = DETAIL.replace("出发前也可以问问是否需要预约。", "出发前也可以问问是否需要预约，我们现场见！")
    texts = AsyncMock(side_effect=["{}", json.dumps({"text": together}, ensure_ascii=False),
                                   json.dumps({"text": DETAIL}, ensure_ascii=False)])
    monkeypatch.setattr(generation, "invoke_text", texts)
    check = AsyncMock(return_value={"supported": True, "unsupported_claims": []})
    monkeypatch.setattr(generation, "invoke_json", check)
    result = await generate([place()])
    assert result["description"] == DETAIL
    assert result["discovery_metadata"]["copy_status"] == "verified"
    assert "线上陪伴" in texts.await_args_list[2].args[1]  # Feedback reached the second draft.
    assert check.await_count == 1  # The "我们现场见" draft never reached the reviewer.


async def test_malformed_review_is_unavailable_and_restating_the_place_is_not_a_summary(quality, monkeypatch):
    selection = {"candidate_id": "poi:one", "summary": "可以去海棠陶艺看看", "vibe": "安静专注"}
    monkeypatch.setattr(generation, "invoke_text", AsyncMock(side_effect=[
        json.dumps({"candidates": [selection]}, ensure_ascii=False), "{}"]))
    check = AsyncMock(return_value="not json")
    monkeypatch.setattr(generation, "invoke_json", check)
    result = await generate([place()])
    reviewed = check.await_args.args[1]
    assert "氛围：安静专注" in reviewed and "推荐摘要" not in reviewed
    assert result["discovery_metadata"]["card_copy_status"] == "fallback_unavailable"
    assert result["summary"] != "可以去海棠陶艺看看" and result.get("vibe") != "安静专注"


async def test_detail_stage_has_its_own_deadline(quality, monkeypatch):
    import asyncio

    async def slow(_model, prompt):
        if "候选原始ID" in prompt:
            return "{}"
        await asyncio.sleep(1)
        return json.dumps({"text": DETAIL})

    monkeypatch.setattr(generation, "_DETAIL_STAGE_BUDGET_S", 0.05)
    monkeypatch.setattr(generation, "invoke_text", slow)
    result = await generate([place()])
    assert result["discovery_metadata"]["copy_status"] == "fallback_unavailable"
    assert useful_detail(result["description"])


async def test_detail_prompt_failure_never_takes_the_card_down(quality, monkeypatch):
    selection = {"candidate_id": "poi:one", "summary": "想把注意力从屏幕上挪开，可以去海棠陶艺挑一项手作。"}
    monkeypatch.setattr(generation, "invoke_text", AsyncMock(return_value=json.dumps(
        {"candidates": [selection]}, ensure_ascii=False)))

    async def prompt(key):
        if key == "offline.activity_recommendation_copy":
            raise RuntimeError("disabled in admin")
        return PROMPTS.get(key, "核验事实：{facts_json}\n文案：{copy_text}")

    monkeypatch.setattr(generation, "get_prompt_text", prompt)
    monkeypatch.setattr(generation, "invoke_json", AsyncMock(return_value={"supported": True, "unsupported_claims": []}))
    result = await generate([place()])
    meta = result["discovery_metadata"]
    assert (meta["card_copy_status"], meta["copy_status"]) == ("verified", "fallback_unavailable")
    assert result["summary"] == selection["summary"] and useful_detail(result["description"])
    # The provider introduction is kept as a fact for the page note.
    assert meta["place_introduction"] == "海棠陶艺，万达广场A幢2315室。"


async def test_failed_selection_is_counted_as_a_fallback_not_by_design(quality, monkeypatch):
    monkeypatch.setattr(generation, "invoke_text", AsyncMock(side_effect=[TimeoutError(), "{}"]))
    monkeypatch.setattr(generation, "invoke_json", AsyncMock())
    result = await generate([place()])
    assert result["discovery_metadata"]["card_copy_status"] == "fallback_no_proposal"


async def test_supported_with_listed_problems_is_not_a_pass(monkeypatch):
    monkeypatch.setattr(generation, "get_prompt_text", AsyncMock(return_value="{facts_json}{copy_text}"))
    monkeypatch.setattr(generation, "get_utility_model", lambda: object())
    monkeypatch.setattr(generation, "invoke_json", AsyncMock(return_value={
        "supported": True, "unsupported_claims": ["“安静”没有依据"]}))
    assert await generation._fact_check({}, "氛围：安静") == (False, ["“安静”没有依据"])
