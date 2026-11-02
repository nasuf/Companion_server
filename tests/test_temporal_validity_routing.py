"""A main-prompt routing reason must not bypass historical fact checking."""

from datetime import datetime, timezone
import json
from unittest.mock import AsyncMock
import asyncio
import pytest
from app.services.chat import orchestrator as chat, temporal_recall
from app.services.chat.intent_dispatcher import IntentType
from app.services.time_domain import validity_context
from app.services.memory.retrieval.context_selector import ClassifiedMemory
from tests.g02_harness_support import configure_pair


@pytest.mark.parametrize(
    "answer", ["那时住在松林旅馆。", None, RuntimeError("dependency unavailable")]
)
@pytest.mark.parametrize("has_assertions", [True, False])
async def test_main_path_uses_checked_history_and_disabled_check_cannot_fall_through(
    monkeypatch, answer, has_assertions
):
    io = configure_pair(monkeypatch)
    io.now = datetime(2026, 11, 1, 2, tzinfo=timezone.utc)
    io.db.message.find_many = AsyncMock(return_value=[])
    fetched = chat.fetch_parallel_context.return_value
    fetched.classified_memories = [
        ClassifiedMemory(
            id="m", text="用户住过松林旅馆", source="user", relevance="strong", score=1
        )
    ]
    fetched.memory_relevance = "strong"
    fetched.timeline = "synthetic date evidence forces the existing main path"
    records = {("user", "m"): [{"assertion_id": "a"}]} if has_assertions else {}
    monkeypatch.setattr(
        validity_context,
        "load_validity_context",
        AsyncMock(
            return_value=(
                "bounded evidence",
                records,
                {"version": 1, "status": "recorded"},
            )
        ),
    )
    monkeypatch.setattr(
        validity_context, "assertions_still_available", AsyncMock(return_value=True)
    )
    checked = (
        AsyncMock(side_effect=answer)
        if isinstance(answer, Exception)
        else AsyncMock(return_value=answer)
    )
    monkeypatch.setattr(temporal_recall, "answer_temporal_recall", checked)
    from app.services import persona_grounding
    from app.services.time_domain import guard

    second_check = AsyncMock(
        side_effect=AssertionError("historical route already used its one check")
    )
    monkeypatch.setattr(guard, "guard_temporal_reply", second_check)

    current_location = AsyncMock(
        side_effect=AssertionError("past facts cannot be overwritten with today's city")
    )
    monkeypatch.setattr(persona_grounding, "guard_reply", current_location)
    events = [
        e
        async for e in chat.stream_chat_response(
            "c-1",
            "2026年8月3日当时我住在哪里？",
            io.agent,
            "u-1",
            forced_intent=IntentType.NONE,
            reply_context={"received_at": io.now.isoformat()},
        )
    ]
    checked.assert_awaited_once()
    assert checked.await_args.kwargs["validity_records"] == records
    assert not io.main_calls and not io.tier_calls
    texts = [json.loads(e["data"])["text"] for e in events if e["event"] == "reply"]
    expected = answer if isinstance(answer, str) else temporal_recall._FALLBACK
    assert texts == [expected]
    assert sum(e["event"] == "done" for e in events) == 1
    current_location.assert_not_called()
    second_check.assert_not_called()


async def test_historical_question_without_recalled_memories_cannot_use_current_profile(
    monkeypatch,
):
    io = configure_pair(monkeypatch)
    io.now = datetime(2026, 11, 1, 2, tzinfo=timezone.utc)
    io.db.message.find_many = AsyncMock(return_value=[])
    fetched = chat.fetch_parallel_context.return_value
    fetched.classified_memories = []
    fetched.memory_relevance = "weak"
    # Use the real empty-candidate loader and payload qualification. Neither
    # today's character profile nor an empty recall set proves a past address.
    from app.services.time_domain import recall

    async def empty_recall(question, memories, diagnostics, **kwargs):
        payload, reason = recall.recall_payload(
            question,
            memories,
            diagnostics["temporal_context"]["query"],
            validity_records=kwargs["validity_records"],
        )
        assert payload is None and reason == "no_candidates"
        assert kwargs["validity_records"] == {}
        return None

    monkeypatch.setattr(temporal_recall, "answer_temporal_recall", empty_recall)
    events = [
        event
        async for event in chat.stream_chat_response(
            "c-1",
            "2026年8月3日当时我住在哪里？",
            io.agent,
            "u-1",
            forced_intent=IntentType.NONE,
            reply_context={"received_at": io.now.isoformat()},
        )
    ]
    assert not io.main_calls and not io.tier_calls
    assert [json.loads(e["data"])["text"] for e in events if e["event"] == "reply"] == [
        temporal_recall._FALLBACK
    ]
    assert sum(e["event"] == "done" for e in events) == 1


@pytest.mark.parametrize("force_main", [False, True])
@pytest.mark.parametrize(
    "message",
    [
        "今天有点累，想喝杯茶",
        "我打算换个手机壳，你喜欢哪种颜色？",
        "翻到了几年前的旧照片，拍得挺有意思",
        "给我写两句关于秋天的小诗吧",
        "我昨天看了部电影，现在想聊聊配乐",
    ],
)
async def test_ordinary_chat_never_repeats_historical_sources(
    monkeypatch, message, force_main
):
    io = configure_pair(monkeypatch)
    io.now = datetime(2026, 11, 1, 2, tzinfo=timezone.utc)
    io.db.message.find_many = AsyncMock(return_value=[])
    fetched = chat.fetch_parallel_context.return_value
    fetched.classified_memories = [
        ClassifiedMemory(
            id="m",
            text="用户曾住过松林旅馆",
            source="user",
            relevance="strong",
            score=1,
        )
    ]
    fetched.memory_relevance = "strong"
    if force_main:
        fetched.timeline = "原有上下文保持不变"
    load = AsyncMock(
        side_effect=AssertionError(
            "ordinary chat must not load extra historical sources"
        )
    )
    recheck = AsyncMock(
        side_effect=AssertionError(
            "ordinary chat has no new historical source snapshot"
        )
    )
    monkeypatch.setattr(validity_context, "load_validity_context", load)
    monkeypatch.setattr(validity_context, "assertions_still_available", recheck)
    events = [
        e
        async for e in chat.stream_chat_response(
            "c-1",
            message,
            io.agent,
            "u-1",
            forced_intent=IntentType.NONE,
            reply_context={"received_at": io.now.isoformat()},
        )
    ]
    assert io.main_calls or io.tier_calls
    assert sum(e["event"] == "done" for e in events) == 1
    assert any(e["event"] == "reply" for e in events)
    load.assert_not_called()
    recheck.assert_not_called()
    assert "来源有效期" not in json.dumps(
        io.main_calls + io.tier_calls, ensure_ascii=False, default=str
    )


async def test_inferred_past_state_question_without_answer_falls_back_to_normal_reply(monkeypatch):
    """TIME07: "九月份我在哪上班" 是推断出的历史问答; 查不到依据时照常回复, 不换成兜底句。"""
    io = configure_pair(monkeypatch)
    io.now = datetime(2026, 11, 2, 2, tzinfo=timezone.utc)
    io.db.message.find_many = AsyncMock(return_value=[])
    fetched = chat.fetch_parallel_context.return_value
    fetched.classified_memories = [
        ClassifiedMemory(id="m", text="用户在B公司上班", source="user", relevance="strong", score=1)
    ]
    fetched.memory_relevance = "strong"
    fetched.timeline = "synthetic date evidence forces the existing main path"
    monkeypatch.setattr(validity_context, "load_validity_context",
                        AsyncMock(return_value=("", {}, {"version": 1, "status": "not_collected"})))
    recall_call = AsyncMock(return_value=None)
    monkeypatch.setattr(temporal_recall, "answer_temporal_recall", recall_call)
    events = [
        e async for e in chat.stream_chat_response(
            "c-1", "九月份我在哪上班？", io.agent, "u-1",
            forced_intent=IntentType.NONE, reply_context={"received_at": io.now.isoformat()},
        )
    ]
    recall_call.assert_awaited_once()
    texts = [json.loads(e["data"])["text"] for e in events if e["event"] == "reply"]
    assert texts and temporal_recall._FALLBACK not in texts
    assert io.main_calls or io.tier_calls
