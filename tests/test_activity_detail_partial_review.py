"""Public venue facts from trace 40cf83f6; no private user quotations."""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.offline import activity_generation as generation
from app.services.offline.detail_review import review_result
from tests.test_offline_opening_hours import card

NOW = datetime.fromisoformat("2026-10-18T22:15:00+08:00")
PARTS = [
    "🏛️ 伯先路上的镇江博物馆\n馆舍落成于1890年，占地70余亩，是地方综合历史博物馆。想留一点时间慢慢看看，可以把它作为了解城市历史的一站，按自己的兴趣挑选参观内容。",
    "✨ 逛点小清单\n6号楼文物展厅有吴文化青铜器、陶瓷器、金银器和工艺品基本陈列，8号楼近代史展厅有《奔流——镇江近代历史陈列》。喜欢拍照的话，可以留意建筑与展陈细节，遵守现场参观规则。",
    "🧭 实用信息\n周四09:00-17:00；10月8日周四全天不开放；国庆节09:00-20:00。出发前确认开放时间。",
]
DRAFT = "\n\n".join(PARTS)
RELEASE = json.loads((Path(__file__).parents[1] / "scripts/prompt_releases/20261019_activity_partial_review.json").read_text())
PROMPTS = {p["key"]: p["content"] for p in RELEASE["prompts"]}


def verdict(ids, bad=()):
    return {"supported": not bad, "unsupported_claims": ["时段表达不准确"] if bad else [],
            "section_results": [{"section_id": i, "supported": i not in bad,
                                 "unsupported_claims": ["时段表达不准确"] if i in bad else []} for i in ids]}


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(generation, "get_prompt_text", AsyncMock(side_effect=lambda key: PROMPTS[key]))
    monkeypatch.setattr(generation, "get_chat_model", lambda: object())
    monkeypatch.setattr(generation, "get_utility_model", lambda: object())
    monkeypatch.setattr(generation, "local_now", lambda: NOW)
    return card()


async def test_repeated_time_rejection_keeps_the_verified_museum_content(setup, monkeypatch):
    write = AsyncMock(return_value=json.dumps({"text": DRAFT}, ensure_ascii=False))
    check = AsyncMock(side_effect=[verdict([0, 1, 2], {2}), verdict([2], {2})])
    monkeypatch.setattr(generation, "invoke_text", write)
    monkeypatch.setattr(generation, "invoke_json", check)
    await generation._apply_reviewed_copy(setup, None, dict(setup), {}, detail_context=dict(setup))
    body, metadata = setup["description"], setup["discovery_metadata"]
    assert PARTS[0] in body and PARTS[1] in body
    assert "国庆节" not in body and "10月8日" not in body
    assert "09:00–17:00" in body and "2026-10-18" in body
    assert metadata["copy_status"] == "verified_partial"
    assert metadata["copy_kept_sections"] == [0, 1] and metadata["copy_dropped_sections"] == [2]
    assert "第3段" in write.await_args.args[1]
    assert metadata["visit_timing"]["status"] == "closed"


async def test_repair_cannot_change_an_already_accepted_paragraph(setup, monkeypatch):
    changed = DRAFT.replace("馆舍落成于1890年", "馆舍落成于2099年").replace(PARTS[2], "🧭 实用信息\n出发前确认当天开放安排。")
    monkeypatch.setattr(generation, "invoke_text", AsyncMock(side_effect=[json.dumps({"text": DRAFT}), json.dumps({"text": changed})]))
    check = AsyncMock(side_effect=[verdict([0, 1, 2], {2}), verdict([2])])
    monkeypatch.setattr(generation, "invoke_json", check)
    text, status, _ = await generation._verified_detail(setup, setup, setup, NOW)
    assert status == "verified" and "1890" in text and "2099" not in text
    second_prompt = check.await_args.args[1]
    assert '"section_id": 0' not in second_prompt.split("推荐详情分段：")[-1]


@pytest.mark.parametrize("repair", [TimeoutError(), "", "单段错误输出"])
async def test_failed_repair_does_not_discard_previously_accepted_facts(setup, monkeypatch, repair):
    writer = AsyncMock(side_effect=[DRAFT, repair])
    monkeypatch.setattr(generation, "_recommendation_copy", writer)
    monkeypatch.setattr(generation, "invoke_json", AsyncMock(return_value=verdict([0, 1, 2], {2})))
    text, status, _ = await generation._verified_detail(setup, setup, setup, NOW)
    assert status == "verified_partial" and PARTS[0] in text and PARTS[1] in text


@pytest.mark.parametrize("broken", [
    {"supported": True, "unsupported_claims": []},
    {"supported": False, "unsupported_claims": ["拒绝"], "section_results": []},
    {"supported": False, "unsupported_claims": ["拒绝"], "section_results": [{"section_id": 0, "supported": True, "unsupported_claims": []}] * 3},
    {"supported": True, "unsupported_claims": [], "section_results": [{"section_id": True, "supported": True, "unsupported_claims": []}]},
    {"supported": False, "unsupported_claims": ["原因"], "section_results": [{"section_id": 0, "supported": True, "unsupported_claims": []}]},
])
def test_missing_duplicate_boolean_or_contradictory_verdict_never_certifies_other_paragraphs(broken):
    assert review_result(broken, {0, 1, 2}).supported is None


@pytest.mark.parametrize("result", [
    {"supported": True, "unsupported_claims": ["仍然有问题"], "section_results": [{"section_id": 0, "supported": True, "unsupported_claims": []}]},
    {"supported": True, "unsupported_claims": [], "section_results": [{"section_id": 0, "supported": True, "unsupported_claims": [42]}]},
    {"supported": False, "unsupported_claims": ["问题"], "section_results": [{"section_id": 0, "supported": False, "unsupported_claims": []}]},
    {"supported": False, "unsupported_claims": ["问题"], "section_results": [{"section_id": 0, "supported": True, "unsupported_claims": []}]},
])
def test_internal_verdict_contradictions_are_unavailable(result):
    assert review_result(result, {0}).supported is None


async def test_unlocalised_rejection_keeps_no_unreviewed_paragraph(setup, monkeypatch):
    monkeypatch.setattr(generation, "_recommendation_copy", AsyncMock(return_value=DRAFT))
    monkeypatch.setattr(generation, "invoke_json", AsyncMock(return_value={"supported": False, "unsupported_claims": ["无法定位的问题"]}))
    text, status, _ = await generation._verified_detail(setup, setup, setup, NOW)
    assert text is None and status == "fallback_rejected"


async def test_cancelled_request_is_not_swallowed_as_a_success(setup, monkeypatch):
    monkeypatch.setattr(generation, "_recommendation_copy", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await generation._verified_detail(setup, setup, setup, NOW)


async def test_fully_rejected_draft_can_be_restructured_with_accurate_section_counts(setup, monkeypatch):
    repaired = "\n\n".join([*PARTS[:2], "🧭 实用信息\n伯先路85号。", "出发前确认当天开放安排。"])
    monkeypatch.setattr(generation, "_recommendation_copy", AsyncMock(side_effect=[DRAFT, repaired]))
    monkeypatch.setattr(generation, "invoke_json", AsyncMock(side_effect=[verdict([0, 1, 2], {0, 1, 2}), verdict([0, 1, 2, 3])]))
    text, status, _ = await generation._verified_detail(setup, setup, setup, NOW)
    metadata = setup["discovery_metadata"]
    assert status == "verified" and "1890" in text
    assert metadata["copy_section_count"] == 4
    assert metadata["copy_kept_sections"] == [0, 1, 2, 3] and metadata["copy_dropped_sections"] == []


async def test_failed_regeneration_does_not_reuse_previous_section_counts(setup, monkeypatch):
    setup["discovery_metadata"].update(copy_section_count=3, copy_kept_sections=[0, 1], copy_dropped_sections=[2])
    monkeypatch.setattr(generation, "_recommendation_copy", AsyncMock(return_value=""))
    text, status, _ = await generation._verified_detail(setup, setup, setup, NOW)
    assert text is None and status == "fallback_unavailable"
    assert "copy_section_count" not in setup["discovery_metadata"]


def test_all_six_web_publications_render_and_keep_expected_snapshots():
    from scripts.publish_offline_quality_prompts import validate_entry
    assert len(RELEASE["prompts"]) == 6
    for entry in RELEASE["prompts"]:
        validate_entry(entry)
        assert len(entry["expected_sha256"]) == 64
