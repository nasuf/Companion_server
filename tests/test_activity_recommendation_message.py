"""Grounding, failure containment and private evidence boundaries for page notes."""
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.offline import activity_recommendation_message as note
from app.services.offline import copy_guards as guards
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from scripts.publish_offline_quality_prompts import validate_entry

RELEASES = [json.loads((Path(__file__).parents[1] / "scripts/prompt_releases" / name).read_text())
            for name in ("20261017_activity_copy_grounding.json", "20261017_activity_copy_grounding_r2.json")]
RELEASE = {"prompts": [entry for release in RELEASES for entry in release["prompts"]]}
TEXTS = {entry["key"]: entry["content"] for entry in RELEASE["prompts"]}  # Later releases win.
MEMORY = "我喜欢找咖啡馆看书。"
DIALOGUE = "这周想换个地方看看，喝杯咖啡。"
EVIDENCE = {"memory": [MEMORY, "我不喜欢热闹的酒吧。"], "preference": ["咖啡"], "dialogue": [DIALOGUE]}
MESSAGE = (
    "你提过喜欢找咖啡馆看书，这家小岛咖啡可以先放进下次出门的备选里。"
    "地点在伯先路12号，想换一条路线走走的话，可以把它作为一站。"
    "不过不用特意给自己排个满满的行程，先看看介绍，感兴趣再决定。"
    "带不带书、待多久，都按你当天的心情来，具体安排也以现场规则为准。"
    "出发前看看当天的开放信息就好。要不要找个方便的时候去看看？暂时没空也没关系，先留着这个选择。"
)
# The same note without citing the user, for drafts written with no verified quote.
NEUTRAL = MESSAGE.replace("你提过喜欢找咖啡馆看书，", "") + "具体什么时候出门，按你当天的安排来就好。"


def card(**kwargs):
    return dict(title="小岛咖啡(伯先路店)", location_name="小岛咖啡(伯先路店)",
                address="伯先路12号", category="咖啡与茶饮", description="位于伯先路的咖啡馆。",
                summary="可以把这家咖啡馆作为下次散步的一站。", **kwargs)


@pytest.fixture
def writer(monkeypatch):
    monkeypatch.setattr(note, "get_prompt_text", AsyncMock(side_effect=lambda key: TEXTS[key]))
    monkeypatch.setattr(note, "get_chat_model", lambda: object())
    monkeypatch.setattr(note, "get_utility_model", lambda: object())
    text = AsyncMock(return_value=MESSAGE)
    check = AsyncMock(return_value={"supported": True, "unsupported_claims": [], "relevant_indices": [0, 1, 2]})
    monkeypatch.setattr(note, "invoke_text", text)
    monkeypatch.setattr(note, "invoke_json", check)
    return text, check


def test_release_rendering_and_complete_original_quotes():
    for entry in RELEASE["prompts"]:
        validate_entry(entry)
    references = note.exact_references([
        MEMORY, "咖啡", DIALOGUE, MEMORY, "喜欢热闹的酒吧", "我喜欢热闹的酒吧。", None,
        {"text": MEMORY}, "我喜欢找咖啡馆看书", " " + MEMORY,
    ], EVIDENCE)
    assert references == [{"kind": "memory", "text": MEMORY},
                          {"kind": "preference", "text": "咖啡"},
                          {"kind": "dialogue", "text": DIALOGUE}]
    assert note.exact_references("咖啡", EVIDENCE) == []
    assert len(note.exact_references([str(i) for i in range(10)], {"memory": [str(i) for i in range(10)]})) == 6


def test_r2_republishes_on_top_of_the_published_r1():
    import hashlib
    first, follow_up = ({e["key"]: e for e in release["prompts"]} for release in RELEASES)
    for key, entry in follow_up.items():
        assert entry["expected_sha256"] == hashlib.sha256(first[key]["content"].strip().encode()).hexdigest()
        assert entry["allowed_fields"] == first[key]["allowed_fields"]
    check = follow_up["offline.recommendation_message_check"]["content"]
    # Production 10-17: the one phase-agnostic example was copied into relevance verdicts.
    assert "[0,1,2]" in check and check.count('"relevant_indices":[') == 3
    assert "必须放行" in check and "若正好有空" in check and "今晚" in check
    for prompt in (check, follow_up["offline.activity_recommendation_message"]["content"]):
        assert "开放时段没写太细" in prompt and "资料里没细说" in prompt
    # Production 10-17: what the reviewer passed is listed for rejection, and the writers are told too.
    assert "必须拒绝" in check and "刚路过" in check and "挺清静" in check and "看你关注过这场" in check
    for key in ("offline.activity_recommendation_message", "offline.activity_invite_message"):
        assert "路过" in follow_up[key]["content"] and "清静" in follow_up[key]["content"]


async def test_a_no_evidence_draft_citing_the_user_is_repaired(writer):
    text, check = writer
    check.return_value = {"supported": True, "unsupported_claims": [], "relevant_indices": []}
    text.side_effect = [MESSAGE, NEUTRAL]
    result = await note.attach_recommendation_message(card(), {})
    assert "删除“你提过”" in text.call_args.args[1]
    assert check.await_count == 1
    assert result["discovery_metadata"]["recommendation_message"] == NEUTRAL
    assert result["discovery_metadata"]["recommendation_message_status"] == "verified_no_evidence"


@pytest.mark.parametrize("extra,feedback", [
    ("刚路过伯先路那边，看到这家店。", "删除“刚路过”"),
    ("那边环境挺清静的。", "环境“清静”"),
])
async def test_invented_visits_and_environment_are_repaired(writer, extra, feedback):
    text, check = writer
    text.side_effect = [MESSAGE + extra, MESSAGE]
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    assert feedback in text.call_args.args[1]
    assert result["discovery_metadata"]["recommendation_message_status"] == "verified"


async def test_only_selected_exact_evidence_is_written_and_kept_private(writer):
    text, check = writer
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE, "编造的记忆"]), EVIDENCE)
    prompt = str(text.call_args.args[1])
    assert MEMORY in prompt and DIALOGUE in prompt
    assert "我不喜欢热闹的酒吧。" not in prompt and "编造的记忆" not in prompt
    metadata = result["discovery_metadata"]
    assert metadata["recommendation_message"] == MESSAGE
    assert metadata["recommendation_message_status"] == "verified"
    assert metadata["user_relevance_count"] == 3
    assert "user_relevance" not in result
    assert "小岛咖啡(伯先路店)" in check.call_args.args[1]


@pytest.mark.parametrize("raw", ["", "短句", '{"text":"不要显示JSON"}', "[错误]", "```text\n错误", "x" * 321, "介绍" * 70 + "https://bad.test", "介绍" * 70 + "www.bad.test"])
async def test_malformed_copy_never_reaches_page_or_guard(writer, raw):
    text, check = writer
    check.return_value = {"supported": True, "unsupported_claims": [], "relevant_indices": [0]}
    text.return_value = raw
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY]), EVIDENCE)
    metadata = result["discovery_metadata"]
    assert metadata["recommendation_message"] == note.generic_message(result)
    assert metadata["recommendation_message_status"] == "fallback_invalid"
    assert metadata["user_relevance"] == []
    assert check.await_count == 1  # Relevance precheck only; prose never passes to guard.


@pytest.mark.parametrize("checked", [None, {}, {"supported": False, "relevant_indices": [0, 1, 2]},
    {"supported": True}, {"supported": True, "unsupported_claims": ["无依据"], "relevant_indices": [0, 1, 2]},
    {"supported": False, "unsupported_claims": ["无依据"], "relevant_indices": [0, 1, 2]}])
async def test_ungrounded_or_irrelevant_copy_falls_back(writer, checked):
    writer[1].return_value = checked
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    assert result["discovery_metadata"]["recommendation_message_status"] == "fallback_invalid"
    assert result["discovery_metadata"]["user_relevance"] == []
    assert "你提过" not in result["discovery_metadata"]["recommendation_message"]


@pytest.mark.parametrize("indices", [[0, 1], [0], [], [0, 1, 1], [False, 1, 2], "0,1,2", None])
async def test_a_clear_verdict_is_not_overruled_by_a_copied_index_list(writer, indices):
    """Production 10-17: "supported": true next to the example's "[]" or "[0]"."""
    checked = {"supported": True, "unsupported_claims": []}
    if indices is not None:
        checked["relevant_indices"] = indices
    writer[1].return_value = checked
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    meta = result["discovery_metadata"]
    assert meta["recommendation_relevance_status"] == "verified"
    assert meta["recommendation_message_status"] == "verified"
    assert meta["user_relevance_count"] == 3


@pytest.mark.parametrize("relevance", [
    None, {"supported": True}, {"supported": True, "unsupported_claims": ["第 2 条无关"]},
    {"supported": False, "unsupported_claims": [], "relevant_indices": [0, 1, 2]},
    {"supported": False, "unsupported_claims": [], "relevant_indices": [0, 5]},
    {"supported": False, "unsupported_claims": [], "relevant_indices": [0, 0]},
    {"supported": False, "unsupported_claims": []},
])
async def test_unusable_relevance_verdict_writes_from_facts_alone(writer, relevance):
    text, check = writer
    text.return_value = NEUTRAL
    check.side_effect = [relevance, {"supported": True, "unsupported_claims": [], "relevant_indices": []}]
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    meta = result["discovery_metadata"]
    prompt = text.call_args.args[1]
    assert MEMORY not in prompt and DIALOGUE not in prompt  # Unverified quotes never reach the writer.
    assert meta["recommendation_relevance_status"] == "invalid"
    assert meta["recommendation_message_status"] == "verified_no_evidence"
    assert meta["user_relevance"] == [] and meta["user_relevance_count"] == 0


def test_relevance_verdict_reading():
    assert note.relevant_indices({"supported": True, "unsupported_claims": []}, 3) == [0, 1, 2]
    assert note.relevant_indices({"supported": False, "unsupported_claims": ["x"], "relevant_indices": [2, 0]}, 3) == [0, 2]
    assert note.relevant_indices({"supported": False, "unsupported_claims": ["x"], "relevant_indices": []}, 1) == []
    assert note.relevant_indices({"supported": "true", "unsupported_claims": []}, 3) is None
    assert note.relevant_indices([], 3) is None


@pytest.mark.parametrize("which", [0, 1])
async def test_provider_timeouts_keep_factual_recommendation_available(writer, which):
    writer[which].side_effect = TimeoutError()
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    assert result["discovery_metadata"]["recommendation_message_status"] == "fallback_unavailable"
    assert result["title"] == "小岛咖啡(伯先路店)"


async def test_no_evidence_uses_generic_voice_and_event_time(writer):
    text, check = writer
    text.return_value = note.generic_message(card())
    check.return_value = {"supported": True, "unsupported_claims": [], "relevant_indices": []}
    event = card(discovery_metadata={"kind": "event"}, starts_at="2026-10-20T19:00:00+08:00", ends_at="2026-10-20T21:00:00+08:00")
    result = await note.attach_recommendation_message(event, {})
    prompt = text.call_args.args[1]
    assert "说这句话的日期）：[]" in prompt and "发送时间）：[]" in prompt
    assert "2026-10-20T19:00:00+08:00" in prompt
    assert result["discovery_metadata"]["recommendation_message_status"] == "verified_no_evidence"
    assert "活动时间和参与方式" in note.generic_message(event)
    assert "这个去处" in note.generic_message({})


async def test_unpublished_prompt_does_not_invoke_model(writer, monkeypatch):
    monkeypatch.setattr(note, "get_prompt_text", AsyncMock(side_effect=lambda key: PROMPT_DEFINITION_MAP[key].default_text))
    result = await note.attach_recommendation_message(card(), {})
    assert result["discovery_metadata"]["recommendation_message_status"] == "fallback_not_published"
    writer[0].assert_not_called()


async def test_evidence_limits_and_failed_source_is_not_empty(monkeypatch):
    memories = AsyncMock(return_value=[MEMORY, MEMORY, "", "x" * 1001, "m" * 1000, "n" * 1000, "o" * 1000])
    dialogue = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(note.repo, "recommendation_memory_items", memories)
    monkeypatch.setattr(note.repo, "had_recent_crisis_care", AsyncMock(return_value=False))
    monkeypatch.setattr(note, "recommendation_dialogue", dialogue)
    inputs, statuses = await note.evidence_inputs(user_id="u", workspace_id="w", conversation_id="c", tags=["咖啡"])
    assert inputs["memory"] == [MEMORY, "m" * 1000, "n" * 1000]
    assert statuses == {"preference": "collected", "memory": "collected", "dialogue": "failed"}
    memories.return_value = []
    inputs, statuses = await note.evidence_inputs(user_id="u", workspace_id="w", conversation_id=None, tags=[])
    assert inputs == {"preference": [], "memory": [], "dialogue": []}
    assert statuses == {"preference": "empty", "memory": "empty", "dialogue": "not_available"}


async def test_irrelevant_exact_quote_is_removed_before_writing(writer):
    text, check = writer
    check.side_effect = [
        {"supported": False, "relevant_indices": [0], "unsupported_claims": ["厨房习惯与阅读无直接关联"]},
        {"supported": True, "relevant_indices": [0], "unsupported_claims": []},
    ]
    result = await note.attach_recommendation_message(
        card(user_relevance=[MEMORY, "我习惯把厨房收拾整齐。"]),
        {"memory": [MEMORY, "我习惯把厨房收拾整齐。"]},
    )
    assert "厨房" not in text.call_args.args[1]
    assert result["discovery_metadata"]["user_relevance"] == [{"kind": "memory", "text": MEMORY}]
    assert result["discovery_metadata"]["recommendation_relevance_status"] == "filtered"


@pytest.mark.parametrize("repaired", [True, False])
async def test_single_bounded_repair_or_factual_fallback(writer, repaired):
    text, check = writer
    accepted = {"supported": True, "relevant_indices": [0, 1, 2], "unsupported_claims": []}
    rejected = {"supported": False, "relevant_indices": [0, 1, 2], "unsupported_claims": ["资料齐全没有事实依据"]}
    check.side_effect = [accepted, rejected, accepted if repaired else rejected]
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    assert text.await_count == 2 and check.await_count == 3
    assert "资料齐全没有事实依据" in text.call_args.args[1]
    assert result["discovery_metadata"]["recommendation_message_status"] == ("verified" if repaired else "fallback_invalid")


@pytest.mark.parametrize("disclosure", [
    "具体开放时段素材里没有写明。", "原文未提及开放时间。",
    "因为具体的开放时段没有详细列出。", "目前没有特别提到的偏好。",
    "开放时间并未在介绍中写明。", "用户记忆库没有提供相关信息。",
    "具体开放时段未定。", "具体开放时段未在信息中注明。",
    "目前尚不清楚开放时间。", "营业时间不详。",
    "开放时段没写太细，出发前确认一下。", "开放时间没写清楚。",  # Production 10-17, 叠滘.
])
@pytest.mark.parametrize("repaired", [True, False])
async def test_input_disclosure_is_blocked_even_if_model_guard_would_accept(writer, disclosure, repaired):
    text, check = writer
    unsafe = MESSAGE + disclosure
    text.side_effect = [unsafe, MESSAGE if repaired else unsafe]
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    meta = result["discovery_metadata"]
    assert text.await_count == 2
    assert "删除描述输入数据缺失" in text.call_args.args[1]
    assert check.await_count == (2 if repaired else 1)  # Rejected drafts never reach the LLM guard.
    assert meta["recommendation_message_status"] == ("verified" if repaired else "fallback_invalid")
    assert disclosure not in meta["recommendation_message"]
    assert meta["user_relevance_count"] == (3 if repaired else 0)


@pytest.mark.parametrize("advice", [
    "出发前可以确认开放时间，现场安排以场馆公告为准。",
    "如果想看看手作素材或阅读作品原文，可以先确认现场安排。",
])
async def test_normal_opening_advice_does_not_disclose_input_state(writer, advice):
    text, check = writer
    text.return_value = MESSAGE + advice
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    assert result["discovery_metadata"]["recommendation_message_status"] == "verified"
    assert text.await_count == 1 and check.await_count == 2


async def test_dashscope_check_uses_structured_primary_without_mutating_shared_model(writer, monkeypatch):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    model = ChatOpenAI(model="qwen-fixture", api_key="fixture-key", temperature=0.7,
        seed=7, model_kwargs={"response_format": {"type": "text"}})
    object.__setattr__(model, "_companion_provider", "dashscope")
    monkeypatch.setattr(note, "get_utility_model", lambda: model)
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    assert result["discovery_metadata"]["recommendation_message_status"] == "verified"
    for call in writer[1].await_args_list:
        primary = call.args[0]
        assert primary is not model and note.provider_name(primary) == "dashscope"
        assert primary.temperature == 0
        payload = primary._get_request_payload([HumanMessage(content="fixture")])
        assert payload["seed"] == 7 and payload["response_format"] == {"type": "json_object"}
        assert not call.kwargs  # Provider-only options cannot reach the Ollama fallback.
    assert model.temperature == 0.7 and model.model_kwargs == {"response_format": {"type": "text"}}


# --- Dated evidence, computed dates and deterministic objections (2026-10-17) ---

STALE = "[2026-10-11] 用户今天出来走走"
NOW = __import__("datetime").datetime(2026, 10, 13, 23, 47,
                                      tzinfo=__import__("zoneinfo").ZoneInfo("Asia/Shanghai"))


def concert():
    return dict(title="张韶涵「玩家」2026 巡回演唱会镇江站", location_name="镇江市体育会展中心-体育场",
                address="团山路33号", category="演出", description="演唱会在体育场举办。",
                summary="可以留意这场演唱会。", starts_at="2026-10-17T19:30:00+08:00",
                ends_at="2026-10-17T22:00:00+08:00", discovery_metadata={"kind": "event"})


def test_quote_without_date_matches_whole_item_and_keeps_the_dated_original():
    evidence = {"memory": [STALE], "dialogue": ["[2026-10-13 15:18] 想吃烧烤和金针菇"]}
    assert note.exact_references(["用户今天出来走走", "想吃烧烤和金针菇", "用户今天出来"], evidence) == [
        {"kind": "memory", "text": STALE},
        {"kind": "dialogue", "text": "[2026-10-13 15:18] 想吃烧烤和金针菇"},
    ]
    # The date prefix is context: a full-length quote is not dropped for it.
    long_item = "[2026-10-11] " + "m" * 1000
    assert note._whole_items([long_item], 3000) == [long_item]


async def test_writer_and_reviewer_get_today_and_code_computed_session_day(writer, monkeypatch):
    monkeypatch.setattr(note, "local_now", lambda: NOW)
    text, check = writer
    text.return_value = NEUTRAL
    check.return_value = {"supported": True, "unsupported_claims": [], "relevant_indices": []}
    await note.attach_recommendation_message(concert(), {})
    prompt = text.call_args.args[1]
    assert "今天：2026-10-13（星期二）" in prompt
    assert "日期说明：2026-10-17（星期六，今天之后第 4 天，本周六）" in prompt
    facts = check.call_args.args[1]
    assert "2026-10-13（星期二）" in facts and "本周六" in facts


async def test_wrong_year_and_going_along_are_repaired_before_any_model_review(writer, monkeypatch):
    monkeypatch.setattr(note, "local_now", lambda: NOW)
    text, check = writer
    bad = NEUTRAL + "时间是明年十月中旬，到时候要不要一起去凑个热闹？"
    text.side_effect = [bad, NEUTRAL]
    check.return_value = {"supported": True, "unsupported_claims": [], "relevant_indices": []}
    result = await note.attach_recommendation_message(concert(), {})
    feedback = text.call_args.args[1]
    assert "明年" in feedback and "线上陪伴" in feedback
    assert check.await_count == 1  # The rejected draft never reached the reviewer.
    assert result["discovery_metadata"]["recommendation_message_status"] == "verified_no_evidence"
    assert result["discovery_metadata"]["recommendation_message_issues"] == []


async def test_fallback_keeps_bounded_reasons_for_the_admin_view(writer):
    text, check = writer
    check.side_effect = [
        {"supported": True, "relevant_indices": [0], "unsupported_claims": []},
        {"supported": False, "relevant_indices": [0], "unsupported_claims": ["x" * 300, "b"]},
        {"supported": False, "relevant_indices": [0], "unsupported_claims": ["c"]},
    ]
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY]), EVIDENCE)
    meta = result["discovery_metadata"]
    assert meta["recommendation_message_status"] == "fallback_invalid"
    assert meta["recommendation_message_issues"] == ["c"]
    assert meta["user_relevance"] == []


async def test_dated_evidence_from_storage(monkeypatch):
    from datetime import datetime as dt
    from app.services.offline import activity_message_context as context
    from app.services.offline import repository
    rows = AsyncMock(return_value=[{"content": "用户压力大", "said_at": dt(2026, 10, 10, 14, 36)}])
    monkeypatch.setattr(repository.db, "query_raw", rows)
    assert await repository.recommendation_memory_items("u", "w") == ["[2026-10-10] 用户压力大"]
    assert "COALESCE(statement_time, created_at)" in rows.call_args.args[0]
    rows.return_value = [{"content": "第二句", "created_at": dt(2026, 10, 13, 7, 19)},
                         {"content": "第一句", "created_at": dt(2026, 10, 13, 7, 18)}]
    monkeypatch.setattr(context.db, "query_raw", rows)
    assert await context.recommendation_dialogue(user_id="u", workspace_id="w", conversation_id="c") == [
        "[2026-10-13 15:18] 第一句", "[2026-10-13 15:19] 第二句"]


async def test_copied_example_index_without_evidence_is_not_a_rejection(writer):
    text, check = writer
    text.return_value = NEUTRAL
    check.return_value = {"supported": True, "unsupported_claims": [], "relevant_indices": [0]}
    result = await note.attach_recommendation_message(card(), {})
    assert result["discovery_metadata"]["recommendation_message_status"] == "verified_no_evidence"


def test_missing_hours_reach_writers_as_advice_not_as_a_data_state():
    # Production 10-17: "开放时段未提供" came back as "开放时段没写太细" twice in a row.
    facts = note.page_facts(card(discovery_metadata={"kind": "place"}), NOW)
    assert facts["date_time"] == "固定地点；出发前确认开放时间"
    hours = note.page_facts(card(discovery_metadata={"kind": "place", "opening_hours": "08:30-18:00"}), NOW)["date_time"]
    assert "08:30–18:00" in hours and str(NOW.date()) in hours and "出发前确认" in hours
    # Still no basis for all-day access.
    assert guards.copy_issues("随时都能去。", card(), NOW, facts_text=json.dumps(facts, ensure_ascii=False))


def test_page_facts_prefer_the_providers_own_introduction():
    base = {**card(), "description": "📍 这次去哪里\n模型或模板写的详情"}
    base["discovery_metadata"] = {"kind": "place", "place_introduction": "位于伯先路的咖啡馆。"}
    assert note.page_facts(base, NOW)["description"] == "位于伯先路的咖啡馆。"
    base["discovery_metadata"] = {"kind": "place", "copy_status": "verified"}
    assert note.page_facts(base, NOW)["description"].endswith("模型或模板写的详情")
    base["discovery_metadata"] = {"kind": "place", "copy_status": "fallback_rejected"}
    assert note.page_facts(base, NOW)["description"] == ""  # Templates are not facts.


def test_same_words_on_several_days_bind_to_the_earliest():
    evidence = {"memory": ["[2026-10-12] 用户今天出来走走", "[2026-10-10] 用户今天出来走走"]}
    assert note.exact_references(["用户今天出来走走"], evidence) == [
        {"kind": "memory", "text": "[2026-10-10] 用户今天出来走走"}]
    # An exact dated quote still wins.
    assert note.exact_references(["[2026-10-12] 用户今天出来走走"], evidence)[0]["text"].startswith("[2026-10-12]")


async def test_repair_cut_by_the_deadline_still_reports_the_rejection(writer):
    text, check = writer
    check.side_effect = [
        {"supported": True, "relevant_indices": [0], "unsupported_claims": []},
        {"supported": False, "relevant_indices": [0], "unsupported_claims": ["无依据"]},
        TimeoutError(),
    ]
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY]), EVIDENCE)
    meta = result["discovery_metadata"]
    assert meta["recommendation_message_status"] == "fallback_invalid"
    assert meta["recommendation_message_issues"] == ["无依据"]


# --- 2026-10-18: runtime models write shorter; trace shows each stop and the outcome ---

async def test_a_short_draft_gets_one_repair_instead_of_the_template(writer, monkeypatch):
    text, check = writer
    blocks = []
    monkeypatch.setattr(note, "record_guard_block", lambda *args, **_kw: blocks.append(args))
    text.side_effect = ["你提过喜欢找咖啡馆看书，小岛咖啡可以去看看。", MESSAGE]
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    assert result["discovery_metadata"]["recommendation_message_status"] == "verified"
    assert "160—260 字" in text.call_args.args[1]
    assert text.call_args.kwargs["profile"] == "chat_extract"
    assert [(stage, attempt) for stage, attempt, _ in blocks] == [("note", 0)]


async def test_trace_gets_each_code_stop_and_the_copy_outcome(writer, monkeypatch):
    text, check = writer
    blocks, outcomes = [], []
    monkeypatch.setattr(note, "record_guard_block", lambda *args, **_kw: blocks.append(args))
    monkeypatch.setattr(note, "record_copy_outcome", lambda statuses, issues, **_kw: outcomes.append((statuses, issues)))
    text.side_effect = [MESSAGE + "我们现场见！", MESSAGE + "我们现场见！"]
    meta = {"card_copy_status": "fallback_no_proposal", "copy_status": "fallback_rejected", "copy_issues": ["静谧留白"]}
    result = await note.attach_recommendation_message(
        card(user_relevance=[MEMORY, "咖啡", DIALOGUE], discovery_metadata=meta), EVIDENCE)
    assert result["discovery_metadata"]["recommendation_message_status"] == "fallback_invalid"
    assert [attempt for _, attempt, _ in blocks] == [0, 1]
    statuses, issues = outcomes[0]
    assert statuses == {"card_copy": "fallback_no_proposal", "detail": "fallback_rejected", "note": "fallback_invalid"}
    assert issues["detail"] == ["静谧留白"] and any("线上陪伴" in item for item in issues["note"])



def test_reviewer_is_deterministic_on_any_openai_compatible_small_model():
    """The small model follows model management; a provider switch must not bring back random verdicts."""
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="deepseek-v4-flash", api_key="fixture-key", temperature=0.7)
    object.__setattr__(model, "_companion_provider", "deepseek")
    reviewer = note.structured_reviewer(model)
    assert reviewer is not model and reviewer.temperature == 0
    assert "response_format" not in reviewer.model_kwargs  # JSON mode only where known to work.
    assert model.temperature == 0.7
    plain = object()
    assert note.structured_reviewer(plain) is plain



def test_runtime_short_notes_no_longer_pass_the_about_200_character_contract():
    assert 90 < note._NOTE_MIN_CHARS == 160
    assert note._NOTE_MAX_CHARS == 260
    assert note._NOTE_MIN_CHARS > len("短句")



# --- 2026-10-18 (2): after crisis care, low moments are not recommendation reasons ---

@pytest.mark.parametrize("check,reason", [
    (AsyncMock(return_value=True), "crisis_care"),
    (AsyncMock(side_effect=RuntimeError("db")), "crisis_check_unavailable"),  # Unknown: stay careful.
])
async def test_recent_crisis_care_withholds_dialogue_and_emotion_memories(monkeypatch, check, reason):
    memories = AsyncMock(return_value=["[2026-10-10] 用户喜欢带花香的茶"])
    dialogue = AsyncMock(return_value=["[2026-10-11 00:26] 不该被读到"])
    withheld = []
    monkeypatch.setattr(note.repo, "had_recent_crisis_care", check)
    monkeypatch.setattr(note.repo, "recommendation_memory_items", memories)
    monkeypatch.setattr(note, "recommendation_dialogue", dialogue)
    monkeypatch.setattr(note, "record_evidence_withheld", withheld.append)
    inputs, statuses = await note.evidence_inputs(user_id="u", workspace_id="w", conversation_id="c", tags=["爱喝茶"])
    assert inputs["dialogue"] == [] and dialogue.await_count == 0
    assert memories.await_args.kwargs == {"exclude_emotions": True}
    assert inputs["preference"] == ["爱喝茶"] and inputs["memory"] == ["[2026-10-10] 用户喜欢带花香的茶"]
    # Client-visible statuses never say why; the reason stays private.
    assert statuses["dialogue"] == "withheld" and "crisis" not in json.dumps(statuses)
    assert inputs["_withheld"] == reason and withheld == [reason]
    if check.side_effect is None:
        assert check.await_args.kwargs == {"days": note.CRISIS_CARE_LOOKBACK_DAYS}


async def test_without_crisis_care_nothing_is_withheld(monkeypatch):
    memories = AsyncMock(return_value=[])
    monkeypatch.setattr(note.repo, "had_recent_crisis_care", AsyncMock(return_value=False))
    monkeypatch.setattr(note.repo, "recommendation_memory_items", memories)
    monkeypatch.setattr(note, "recommendation_dialogue", AsyncMock(return_value=[]))
    inputs, statuses = await note.evidence_inputs(user_id="u", workspace_id="w", conversation_id="c", tags=[])
    assert "_withheld" not in inputs and memories.await_args.kwargs == {"exclude_emotions": False}
    assert statuses["dialogue"] == "empty"


async def test_the_withheld_reason_stays_in_the_private_activity_record(writer):
    evidence = {**EVIDENCE, "_withheld": "crisis_care"}
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY]), evidence,
                                                      {"preference": "collected", "memory": "collected", "dialogue": "withheld"})
    meta = result["discovery_metadata"]
    assert meta["evidence_withheld"] == "crisis_care"
    assert meta["recommendation_evidence_status"]["dialogue"] == "withheld"


async def test_crisis_care_query_and_emotion_filter(monkeypatch):
    from app.services.offline import repository
    rows = AsyncMock(return_value=[{"one": 1}])
    monkeypatch.setattr(repository.db, "query_raw", rows)
    assert await repository.had_recent_crisis_care("u", "w", days=14) is True
    sql, *params = rows.await_args.args
    assert "crisis_guard_status" in sql and "short_circuit_kind" in sql and "m.role = 'assistant'" in sql
    assert params[:2] == ["u", "w"]
    rows.return_value = []
    assert await repository.had_recent_crisis_care("u", "w", days=14) is False
    await repository.recommendation_memory_items("u", "w", exclude_emotions=True)
    sql, *params = rows.await_args.args
    assert "main_category" in sql and params == ["u", "w", True]
