"""Grounding, failure containment and private evidence boundaries for page notes."""
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.offline import activity_recommendation_message as note
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from scripts.publish_offline_quality_prompts import validate_entry

RELEASE = json.loads((Path(__file__).parents[1] /
    "scripts/prompt_releases/20261010_activity_recommendation_message.json").read_text())
TEXTS = {entry["key"]: entry["content"] for entry in RELEASE["prompts"]}
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
    {"supported": True}, {"supported": True, "unsupported_claims": [], "relevant_indices": [0, 1]},
    {"supported": True, "unsupported_claims": [], "relevant_indices": [0, 1, 1]},
    {"supported": True, "unsupported_claims": [], "relevant_indices": [False, 1, 2]},
    {"supported": True, "unsupported_claims": [], "relevant_indices": "0,1,2"}])
async def test_ungrounded_or_irrelevant_copy_falls_back(writer, checked):
    writer[1].return_value = checked
    result = await note.attach_recommendation_message(card(user_relevance=[MEMORY, "咖啡", DIALOGUE]), EVIDENCE)
    assert result["discovery_metadata"]["recommendation_message_status"] == "fallback_invalid"
    assert result["discovery_metadata"]["user_relevance"] == []
    assert "你提过" not in result["discovery_metadata"]["recommendation_message"]


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
    assert "用户记忆库：[]" in prompt and "对话上下文：[]" in prompt
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
