"""Production regressions: floral-tea analogy, seeded fallback and 81-char notes."""
import copy
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.offline import activity_generation as generation, activity_recommendation_message as note
from app.services.offline.copy_guards import copy_issues
from app.services.offline.relevance_review import reference_indices
from scripts.publish_offline_quality_prompts import validate_entry

RELEASE = json.loads((Path(__file__).parents[1] / "scripts/prompt_releases/20261019_activity_copy_fidelity.json").read_text())
PROMPTS = {p["key"]: p["content"] for p in RELEASE["prompts"]}
NOW = datetime.fromisoformat("2026-10-19T13:47:00+08:00")
TEA = "[2026-10-10] 用户喜欢带花香的茶"
EVIDENCE = {"memory": [TEA], "preference": ["爱探店"], "dialogue": []}


def card():
    return dict(candidate_id="poi:spring", title="春树·花艺(常发广场店)", location_name="春树·花艺(常发广场店)",
                city="镇江市", address="黄山西路常发广场5号楼102", category="花艺文创",
                summary="想换条路线走走，可以把这里作为一站。", description="常发广场的一家花艺店。",
                user_relevance=[TEA, "爱探店"], discovery_metadata={"kind": "place", "place_introduction": "常发广场的一家花艺店。"})


def relevance():
    return {"supported": False, "relevant_indices": [1], "unsupported_claims": ["第1项只有感官关联"],
            "reference_results": [{"index": 0, "relation": "attribute_only", "fact_quote": ""},
                                  {"index": 1, "relation": "exploration", "fact_quote": "春树·花艺(常发广场店)"}]}


@pytest.fixture
def setup(monkeypatch):
    for module in (generation, note):
        monkeypatch.setattr(module, "get_prompt_text", AsyncMock(side_effect=lambda k: PROMPTS[k]))
        monkeypatch.setattr(module, "get_chat_model", lambda: object())
        monkeypatch.setattr(module, "get_utility_model", lambda: object())
        monkeypatch.setattr(module, "local_now", lambda: NOW)
    return card()


def test_new_web_versions_render_without_default_edits():
    assert len(RELEASE["prompts"]) == 6
    for entry in RELEASE["prompts"]:
        validate_entry(entry)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "boolean", "foreign_quote", "unknown_relation"])
def test_unusable_relation_verdict_never_certifies_private_quotes(mutation):
    v = relevance()
    if mutation == "missing": v.pop("reference_results")
    elif mutation == "duplicate": v["reference_results"][0]["index"] = 1
    elif mutation == "boolean": v["reference_results"][0]["index"] = False
    elif mutation == "foreign_quote": v["reference_results"][1]["fact_quote"] = "用户爱探店"
    elif mutation == "unknown_relation": v["reference_results"][1]["relation"] = "sounds_similar"
    assert reference_indices(v, 2, note.page_facts(card(), NOW)) is None


def test_exhaustive_grounded_rows_override_redundant_totals_and_whitespace():
    v = relevance()
    v.update(supported=True, relevant_indices=[0, 1])
    v["reference_results"][1]["fact_quote"] = "春树·花艺 (常发广场店)"
    assert reference_indices(v, 2, note.page_facts(card(), NOW)) == [1]


def test_only_genuine_flower_tea_service_evidence_can_support_the_tea_preference():
    facts = note.page_facts(card(), NOW)
    assert reference_indices(relevance(), 2, facts) == [1]
    yes = {"supported": True, "relevant_indices": [0], "unsupported_claims": [],
           "reference_results": [{"index": 0, "relation": "same_activity", "fact_quote": "花茶体验"}]}
    assert reference_indices(yes, 1, facts) is None
    facts["description"] += "店铺介绍提到花茶体验。"
    assert reference_indices(yes, 1, facts) == [0]


async def test_approved_summary_is_used_after_screening_and_repaired_card_copy(setup, monkeypatch):
    checker = AsyncMock(return_value=relevance())
    monkeypatch.setattr(note, "invoke_json", checker)
    candidate = {"candidate_id": setup["candidate_id"], "summary": "爱探店的话，可以了解这家常发广场里的花艺小店。", "vibe": "花艺小店", "suitable": "先了解店内内容"}
    writer = AsyncMock(return_value=json.dumps({"candidates": [candidate]}, ensure_ascii=False))
    monkeypatch.setattr(generation, "invoke_text", writer)
    checks = AsyncMock(return_value=(True, []))
    monkeypatch.setattr(generation, "_fact_check", checks)
    detail = AsyncMock(return_value=("独立核验的地点详情", "verified", []))
    monkeypatch.setattr(generation, "_verified_detail", detail)
    await generation._apply_reviewed_copy(setup, {"summary": "喜欢花香茶，和店里花香适配。"}, copy.deepcopy(setup), EVIDENCE, detail_context=dict(setup))
    context = detail.await_args.args[1]
    assert context["summary"] == candidate["summary"] == setup["summary"]
    assert "想换条路线" not in context["summary"]
    assert setup["user_relevance"] == ["爱探店"]
    assert setup["discovery_metadata"]["relevance_screen"] == {"status": "filtered", "submitted": 2, "kept": 1}
    assert setup["discovery_metadata"]["detail_input_summary_status"] == "verified"
    assert writer.await_count == 1 and checker.await_count == 1
    assert checks.await_count == 1  # Only rewritten copy reaches the reviewer.
    assert TEA not in writer.await_args.args[1]


async def test_failed_card_review_does_not_seed_the_detail_with_fallback(setup, monkeypatch):
    setup["user_relevance"] = []
    monkeypatch.setattr(generation, "_fact_check", AsyncMock(return_value=(None, [])))
    detail = AsyncMock(return_value=("独立核验的地点详情", "verified", []))
    monkeypatch.setattr(generation, "_verified_detail", detail)
    await generation._apply_reviewed_copy(setup, {"summary": "这个还没有通过"}, {}, {}, detail_context=dict(setup))
    assert detail.await_args.args[1]["summary"] == ""
    assert setup["discovery_metadata"]["detail_input_summary_status"] == "unavailable"


async def test_notes_reuse_only_the_bound_relevance_decision_and_meet_length(setup, monkeypatch):
    check = AsyncMock(side_effect=[relevance(), {"supported": True, "unsupported_claims": []}])
    monkeypatch.setattr(note, "invoke_json", check)
    references = note.exact_references(setup["user_relevance"], EVIDENCE)
    await note.prepare_references(setup, references, NOW)
    text = note.generic_message(setup)
    monkeypatch.setattr(note, "invoke_text", AsyncMock(return_value=text))
    await note.attach_recommendation_message(setup, EVIDENCE)
    assert check.await_count == 2  # One relevance review, one prose review.
    assert TEA not in note.invoke_text.await_args.args[1]
    assert setup["discovery_metadata"]["recommendation_message_status"] == "verified"
    assert setup["discovery_metadata"]["recommendation_relevance_status"] == "filtered"
    assert 160 <= len(setup["discovery_metadata"]["recommendation_message"]) <= 260


@pytest.mark.parametrize("length", [81, 91, 159, 261])
async def test_short_or_overlong_runtime_notes_cannot_be_labelled_verified(setup, monkeypatch, length):
    setup["user_relevance"] = []
    monkeypatch.setattr(note, "invoke_text", AsyncMock(return_value="句" * length))
    checker = AsyncMock();monkeypatch.setattr(note, "invoke_json", checker)
    await note.attach_recommendation_message(setup, {})
    meta = setup["discovery_metadata"]
    assert meta["recommendation_message_status"] == "fallback_invalid"
    assert 160 <= meta["recommendation_message_length"] <= 260
    assert meta["recommendation_message_length_bounds"] == [160, 260]
    checker.assert_not_called()


def test_user_fragrance_preference_is_not_a_venue_environment_fact():
    assert copy_issues("和店里的花香很适配。", card(), NOW, facts_text="花艺文创", user_references=1)
    assert not copy_issues("和店里的花香很适配。", card(), NOW, facts_text="店内有花香。", user_references=1)


async def test_length_repair_uses_runtime_small_model_but_still_requires_fact_review(setup, monkeypatch):
    setup["user_relevance"] = []
    primary, repair = object(), object()
    monkeypatch.setattr(note, "get_chat_model", lambda: primary)
    monkeypatch.setattr(note, "get_utility_model", lambda: repair)
    writer = AsyncMock(side_effect=["短" * 81, note.generic_message(setup)])
    check = AsyncMock(return_value={"supported": True, "unsupported_claims": []})
    monkeypatch.setattr(note, "invoke_text", writer)
    monkeypatch.setattr(note, "invoke_json", check)
    await note.attach_recommendation_message(setup, {})
    assert [c.args[0] for c in writer.await_args_list] == [primary, repair]
    assert check.await_count == 1
    meta = setup["discovery_metadata"]
    assert meta["recommendation_message_status"] == "verified_no_evidence"
    assert meta["recommendation_message_repair_route"] == "instruction_length_repair"


async def test_model_summary_cannot_be_its_own_venue_fragrance_evidence(setup, monkeypatch):
    setup["user_relevance"] = []
    setup["summary"] = "店里有花香，可以去看看。"
    claim = "店里有花香。" + note.generic_message(setup)
    monkeypatch.setattr(note, "invoke_text", AsyncMock(return_value=claim))
    checker = AsyncMock();monkeypatch.setattr(note, "invoke_json", checker)
    await note.attach_recommendation_message(setup, {})
    assert setup["discovery_metadata"]["recommendation_message_status"] == "fallback_invalid"
    checker.assert_not_called()


async def test_short_note_expansion_preserves_draft_and_reviews_the_entire_combined_text(setup, monkeypatch):
    setup["user_relevance"] = []
    base = ("常发广场这家花艺小店，可以给下次出门多留一个选择。"
            "如果想换条路线，可以先了解店里的具体内容，再决定要不要去看看。"
            "不用为了推荐打乱原有安排，时间合适时再考虑就好。"
            "出发前确认当天的开放安排，到了现场按自己的节奏决定停留多久。")
    tail = "如果有更感兴趣的事情，也可以先做自己的安排，把这个去处留到之后。要不要先了解一下，再挑个方便的时候去转转？"
    assert len(base) < 160 and 160 <= len(base + tail) <= 260
    monkeypatch.setattr(note, "invoke_text", AsyncMock(side_effect=[base[:80], base, tail]))
    checker = AsyncMock(return_value={"supported": True, "unsupported_claims": []})
    monkeypatch.setattr(note, "invoke_json", checker)
    await note.attach_recommendation_message(setup, {})
    assert setup["discovery_metadata"]["recommendation_message"] == base + tail
    assert "append" in note.invoke_text.await_args.args[1]
    assert base + tail in checker.await_args.args[1]
    assert setup["discovery_metadata"]["recommendation_message_status"] == "verified_no_evidence"


async def test_wrong_phase_schema_is_retried_but_valid_rejection_is_not(setup, monkeypatch):
    check = AsyncMock(side_effect=[{"reference_results": []}, {"supported": False, "unsupported_claims": ["没有项目依据"]}])
    monkeypatch.setattr(note, "invoke_json", check)
    result = await note.review_copy(PROMPTS["offline.recommendation_message_check"], phase="message",
        facts={}, references=[], message="待审文案")
    assert check.await_count == 2 and result["supported"] is False


async def test_failed_relevance_collection_is_not_misreported_as_no_evidence(setup, monkeypatch):
    monkeypatch.setattr(note, "invoke_json", AsyncMock(side_effect=TimeoutError()))
    assert await note.prepare_references(setup, note.exact_references(setup["user_relevance"], EVIDENCE), NOW) == []
    assert setup["user_relevance"] == []
    assert setup["discovery_metadata"]["relevance_screen"] == {"status": "unavailable", "submitted": 2, "kept": 0}


async def test_card_repair_gets_actual_rejection_feedback_and_keeps_accepted_fields(setup, monkeypatch):
    good = {"candidate_id": setup["candidate_id"], "summary": "爱探店的话，可以先了解这家具体花艺小店。", "vibe": "花艺小店", "suitable": "了解花艺内容"}
    first = {**good, "summary": "现在有9.9元体验"}
    writer = AsyncMock(side_effect=[json.dumps({"candidates": [first]}), json.dumps({"candidates": [{**good, "vibe": "改掉已通过字段"}]})])
    monkeypatch.setattr(generation, "invoke_text", writer)
    monkeypatch.setattr(generation, "_fact_check", AsyncMock(side_effect=[
        (False, ["“现在有9.9元体验”：旧探店特价不能证明现在售价"]), (True, [])]))
    status, issues = await generation._verified_card_copy(setup, {"summary": "被排除的用户关联"}, {}, [], NOW,
        missing_status="deterministic", references_filtered=True)
    assert status == "verified" and issues == []
    assert setup["summary"] == good["summary"] and setup["vibe"] == first["vibe"]
    assert "旧探店特价" in writer.await_args.args[1]
    assert "card_copy_repair_writer" in setup["discovery_metadata"]


async def test_card_repair_cannot_switch_destinations(setup, monkeypatch):
    monkeypatch.setattr(generation, "invoke_text", AsyncMock(return_value=json.dumps({"candidates": [{"candidate_id": "poi:someone-else", "summary": "别家门店"}]})))
    status, _ = await generation._verified_card_copy(setup, {"summary": "旧稿"}, {}, [], NOW,
        missing_status="deterministic", references_filtered=True)
    assert status == "fallback_rejected" and setup["candidate_id"] == "poi:spring"
    assert setup["summary"] != "别家门店"
