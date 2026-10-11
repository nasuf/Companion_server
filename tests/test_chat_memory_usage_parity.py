"""Exercise accounting through both real executors, including its identity guard.

Only the final SQL batch is mocked. Replacing record_memory_usage itself with an
AsyncMock would miss the original graph bug: missing event/owner silently skips
all lifecycle updates even though the coroutine was submitted.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.services.chat import orchestrator as chat
from app.services.chat.data_fetch_phase import FetchedContext
from app.services.chat.intent_dispatcher import IntentResult, IntentType
from app.services.memory.lifecycle import lazy_update
from app.services.memory.retrieval import access_log
from app.services.memory.retrieval.context_selector import ClassifiedMemory
from tests.g02_harness_support import configure_pair


def configure_accounting(monkeypatch, executor):
    io = configure_pair(monkeypatch)
    monkeypatch.setattr(settings, "chat_executor", executor)
    io.memory = ClassifiedMemory(
        id="memory-selected", text="用户喜欢在周末做陶艺", relevance="medium",
        score=0.8, source="user",
    )
    chat.fetch_parallel_context.return_value = FetchedContext(
        memory_relevance="medium", classified_memories=[io.memory],
        candidate_ids=[io.memory.id, "memory-candidate", "memory-candidate", ""],
    )
    io.batch = AsyncMock(return_value={"total": 1})
    monkeypatch.setattr(lazy_update, "_apply_batch", io.batch)
    # Shared submission may await the existing access audit as well. Keep its
    # database operations synthetic while preserving the real accounting guard.
    monkeypatch.setattr(access_log, "db", SimpleNamespace(execute_raw=AsyncMock()))
    io.accounting = []
    previous_fire = chat._fire_background

    def capture(coroutine):
        code = getattr(coroutine, "cr_code", None)
        if code and (
            code.co_name == "record_memory_usage"
            or code.co_filename.endswith("/chat/memory_usage.py")
        ):
            io.accounting.append(coroutine)
        else:
            previous_fire(coroutine)

    monkeypatch.setattr(chat, "_fire_background", capture)
    from app.services.chat import multi_intent

    monkeypatch.setattr(multi_intent, "_fire_background", capture)
    return io


async def execute_and_drain(io, text="周末还想继续做陶艺", **kwargs):
    kwargs.setdefault("reply_context", {"received_at": io.now.isoformat()})
    try:
        events = [event async for event in chat.stream_chat_response(
            "c-1", text, io.agent, "u-1", **kwargs
        )]
        while io.accounting:
            await io.accounting.pop(0)
        return events
    finally:
        for coroutine in io.accounting:
            coroutine.close()


def assert_scoped_batches(io, *, event_id, selected=True, candidate=True):
    # The actual public lifecycle function must reach both scoped SQL sides.
    assert io.batch.await_count == 2
    calls = [call.kwargs for call in io.batch.await_args_list]
    assert {call["source"] for call in calls} == {"user", "ai"}
    expected = {}
    if selected:
        expected["memory-selected"] = True
    if candidate:
        expected["memory-candidate"] = False
    for call in calls:
        assert call["event_id"] == event_id
        assert call["user_id"] == "u-1"
        assert call["workspace_id"] == "w-1"
        assert call["signals"] == expected
        assert set(call["ids"]) == set(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
@pytest.mark.parametrize("aggregated", [False, True])
async def test_ordinary_memory_usage_reaches_storage_with_ingress_identity(
    monkeypatch, executor, aggregated,
):
    io = configure_accounting(monkeypatch, executor)
    kwargs = {}
    if aggregated:
        kwargs = {
            "save_user_message": False, "user_message_id": "fragment-final",
            "reply_context": {
                "received_at": io.now.isoformat(),
                "turn_message_ids": ["fragment-first", "fragment-final"],
            },
        }
    events = await execute_and_drain(io, **kwargs)
    assert events[-1]["event"] == "done"
    assert io.db.message.create.await_count == (0 if aggregated else 1)
    assert_scoped_batches(io, event_id="fragment-final" if aggregated else "u-new")
    diagnostics = chat._save_replies.call_args.args[1][0]["response_diagnostics"]
    assert diagnostics["version"] == 1
    assert diagnostics["memory_usage_submission"]["status"] == "submitted"
    assert diagnostics["memory_usage_submission"]["selected_count"] == 1
    assert diagnostics["memory_usage_submission"]["candidate_only_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
@pytest.mark.parametrize("followup", [False, True])
async def test_crisis_memory_usage_keeps_same_identity_and_no_candidate_reward(
    monkeypatch, executor, followup,
):
    from app.services import portrait
    from app.services.memory.retrieval import safety

    io = configure_accounting(monkeypatch, executor)
    io.decision.crisis_force_intent = not followup
    io.decision.crisis_followup_active = followup
    io.decision.crisis_care_turn = True
    io.decision.status = "synthetic_care_decision"
    io.decision.reason = "synthetic_care_reason"
    io.decision.semantic_checked = True
    io.decision.semantic_detected = not followup
    io.decision.recent_crisis_context = "前一轮存在危机风险"
    io.decision.intent_metadata = {"followup": True, "safety_check_mode": "soft"} if followup else {}
    monkeypatch.setattr(chat, "discard_pending_states_for_crisis", AsyncMock())
    monkeypatch.setattr(portrait, "get_latest_portrait", AsyncMock(return_value=None))
    for name in ("retrieve_crisis_memories", "retrieve_crisis_followup_memories"):
        monkeypatch.setattr(safety, name, AsyncMock(return_value=[io.memory]))

    async def handle(message, ctx, **kwargs):
        assert kwargs["classified_memories"] == [io.memory]
        async for event in ctx.finalize(
            "我在，先到安全的地方", kind="crisis_followup" if followup else "crisis",
        ):
            yield event

    monkeypatch.setattr(chat, "handle_crisis_followup" if followup else "handle_crisis", handle)
    events = await execute_and_drain(io, "我现在还是很难受")
    assert events[-1]["event"] == "done"
    chat.fetch_parallel_context.assert_not_called()
    assert_scoped_batches(io, event_id="u-new", candidate=False)
    diagnostics = chat._save_replies.call_args.args[1][0]["response_diagnostics"]
    assert diagnostics["version"] == 1
    assert diagnostics["crisis_guard_status"] == "synthetic_care_decision"
    assert diagnostics["crisis_guard_reason"] == "synthetic_care_reason"
    assert diagnostics["crisis_semantic_checked"] is True
    assert diagnostics["crisis_semantic_detected"] is (not followup)
    assert diagnostics["short_circuit_kind"] == ("crisis_followup" if followup else "crisis")
    assert diagnostics["memory_usage_submission"]["status"] == "submitted"


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
async def test_child_memory_usage_reuses_persisted_parent_event(monkeypatch, executor):
    io = configure_accounting(monkeypatch, executor)
    chat.detect_intent_unified.return_value = IntentResult(
        intent=IntentType.CONVERSATION_END, confidence=0.9,
        metadata={"fragments": {"终结意图": "不聊了", "日常交流": "周末还想继续做陶艺"}},
    )

    async def farewell(text, ctx, *_):
        async for event in ctx.finalize("好，晚点再聊", kind="conversation_end"):
            yield event

    monkeypatch.setattr(chat, "handle_conversation_end", farewell)
    events = await execute_and_drain(io, "周末还想继续做陶艺，不聊了")
    assert sum(event["event"] == "done" for event in events) == 1
    io.db.message.create.assert_awaited_once()
    assert_scoped_batches(io, event_id="u-new")


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
async def test_parent_and_child_share_event_with_distinct_and_overlapping_memories(
    monkeypatch, executor,
):
    io = configure_accounting(monkeypatch, executor)
    child_memory = ClassifiedMemory(
        id="memory-child", text="用户喜欢画水彩", relevance="medium",
        score=0.8, source="user",
    )
    chat.fetch_parallel_context.side_effect = [
        FetchedContext(memory_relevance="medium", classified_memories=[io.memory],
                       candidate_ids=[io.memory.id, "memory-candidate"]),
        FetchedContext(memory_relevance="medium", classified_memories=[io.memory, child_memory],
                       candidate_ids=[io.memory.id, child_memory.id]),
    ]
    chat.detect_intent_unified.return_value = IntentResult(
        intent=IntentType.NONE, confidence=0.9,
        metadata={"fragments": {"日常交流": "周末还想继续做陶艺", "询问当前状态": "你在干嘛"}},
    )
    monkeypatch.setattr(chat, "handle_current_state", AsyncMock(return_value=(False, None)))
    events = await execute_and_drain(io, "周末还想继续做陶艺，你在干嘛")
    assert sum(event["event"] == "done" for event in events) == 1
    io.db.message.create.assert_awaited_once()
    assert io.batch.await_count == 4
    calls = [call.kwargs for call in io.batch.await_args_list]
    for call in calls:
        assert (call["event_id"], call["user_id"], call["workspace_id"]) == ("u-new", "u-1", "w-1")
    assert [call["signals"] for call in calls] == [
        {"memory-selected": True, "memory-candidate": False},
        {"memory-selected": True, "memory-candidate": False},
        {"memory-selected": True, "memory-child": True},
        {"memory-selected": True, "memory-child": True},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
async def test_queued_input_without_persisted_identity_cannot_reward_or_infer_history(
    monkeypatch, executor,
):
    io = configure_accounting(monkeypatch, executor)
    events = await execute_and_drain(
        io, save_user_message=False, user_message_id=None, delivered_from_queue=True,
        reply_context={"received_at": io.now.isoformat(), "turn_message_ids": ["unproven-id"]},
    )
    assert events[-1]["event"] == "done"
    io.db.message.create.assert_not_awaited()
    io.batch.assert_not_awaited()
