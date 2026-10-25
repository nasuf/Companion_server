"""Real persistence and parent completion through both chat executors."""
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.services.chat import orchestrator as chat, post_process, legacy_completion
from app.services.chat.intent_dispatcher import IntentResult, IntentType
from app.services.runtime import turn_followups as t
from tests.g02_harness_support import configure_pair
from tests.test_turn_followups_postgres import turn, followups, redis, origins, receipt


@pytest.mark.parametrize("executor", ["langgraph"])
@pytest.mark.parametrize("failure", [None, "last_save", "registration", "cancel"])
async def test_parent_completion_waits_for_all_fragments(turn, monkeypatch, executor, failure):
    db, user, old_replies, args = turn
    for reply in old_replies:
        await db.message.delete(where={"id": reply.id})
    register = t.register_completed_turn
    io = configure_pair(monkeypatch)
    monkeypatch.setattr(post_process, "db", db)
    monkeypatch.setattr(post_process, "fire_background", lambda coro: coro.close())
    monkeypatch.setattr(t, "register_completed_turn", AsyncMock(
        side_effect=RuntimeError("synthetic registration unavailable") if failure == "registration" else register,
    ))
    writes = []

    async def save(*pos, **kw):
        # No completion receipt may exist while a child is still being saved.
        assert not await db.query_raw("SELECT turn_id FROM turn_followup_events WHERE scope->>'conversation_id'=$1", user.conversationId)
        writes.append(kw)
        if len(writes) == 2 and failure == "last_save":
            return None
        return await post_process.save_replies(*pos, **kw)

    monkeypatch.setattr(chat, "_save_replies", save)
    chat.detect_intent_unified.return_value = IntentResult(
        intent=IntentType.RECORD_REQUEST, confidence=1,
        metadata={"fragments": {"记录请求": "提醒我喝水", "日常交流": "今天很好"}},
    )

    async def special(text, ctx, *a, **kw):
        return True, ctx.finalize("乙" * 25, kind="record_request")

    monkeypatch.setattr(chat, "handle_record_request", special)
    monkeypatch.setattr(chat, "_memory_weak_reply", AsyncMock(return_value="丙" * 25))
    stream = chat.stream_chat_response(user.conversationId, "提醒我喝水，今天很好", io.agent, "u-1",
        save_user_message=False, user_message_id=user.id)
    events = []
    try:
        if failure in {"last_save", "registration"}:
            with pytest.raises(RuntimeError):
                async for event in stream:
                    events.append(event)
        else:
            async for event in stream:
                events.append(event)
                if failure == "cancel":
                    break
    finally:
        await stream.aclose()
    rows = await db.query_raw("SELECT * FROM turn_followup_events WHERE scope->>'conversation_id'=$1", user.conversationId)
    assert legacy_completion.current_completion.get() is None
    if failure:
        assert not rows and not any(e["event"] == "done" for e in events)
        return
    assert len(rows) == 1 and len(writes) == 2
    assert rows[0]["user_message_ids"] == [user.id]
    assert sum(e["event"] == "done" for e in events) == 1
    t.register_completed_turn.assert_awaited_once()
    await t.run_turn_followup({"turn_id": rows[0]["turn_id"], "generation": 0})
    assert (await receipt(db, rows[0]["turn_id"]))["status"] == "completed"
    assert await db.achievementunlock.count(where={"conversationId": user.conversationId, "achievementId": 81}) == 1


async def test_legacy_collector_rejects_cross_turn_and_duplicate_done(monkeypatch):
    register = AsyncMock(return_value=True)
    monkeypatch.setattr(t, "register_completed_turn", register)
    parent = legacy_completion.LegacyCompletion("c")
    parent.record("c", "a1", "turn", ["u"])
    parent.record("c", "a2", "turn", ["u"])
    await parent.finish()
    await parent.finish()
    register.assert_awaited_once_with(conversation_id="c", message_id="a1", turn_id="turn", user_message_ids=["u"])
    with pytest.raises(RuntimeError, match="source_mismatch"):
        parent.record("c", "a3", "other", ["u"])
    with pytest.raises(RuntimeError, match="incomplete"):
        await parent.finish()


async def test_legacy_collector_rejects_foreign_conversation():
    parent = legacy_completion.LegacyCompletion("c")
    with pytest.raises(RuntimeError, match="persistence_failed"):
        parent.record("other", "a", "turn", ["u"])


async def test_completed_graph_registration_failure_precedes_other_finalization(monkeypatch):
    io = configure_pair(monkeypatch)
    monkeypatch.setattr(t, "register_completed_turn", AsyncMock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = [e async for e in chat.stream_chat_response("c-1", "今天很好", io.agent, "u-1")]
    chat.finish_assistant_turn.assert_not_awaited()
    chat._background_post_process.assert_not_called()
