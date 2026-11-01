"""Real PostgreSQL: final send eligibility for optional event follow-ups."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.services.time_domain.followup_decision import (
    checked_proactive_insert,
    guard_reply_followups,
    load_followup_history,
)
from tests.test_memory_evidence_postgres import origins  # noqa: F401  (fixture)


@pytest.fixture
async def followup(origins):  # noqa: F811
    db, uid, wid, _, _, _, _, users, agents, spaces, convs = origins
    said = datetime.now(timezone.utc) - timedelta(hours=2)
    await db.message.create(data={"conversationId": convs[0], "role": "user",
                                  "content": "我明天下午有个面试", "createdAt": said})
    return db, uid, wid, convs


async def _send(db, uid, wid, conversation, message, since=None):
    return await checked_proactive_insert(
        db, conversation_id=conversation, user_id=uid, workspace_id=wid, message=message,
        metadata={"proactive": True}, since=since, now=datetime.now(timezone.utc))


async def test_two_workers_rephrasing_the_same_event_get_one_send(followup):
    db, uid, wid, convs = followup
    results = await asyncio.gather(
        _send(db, uid, wid, convs[0], "面试结果出来了吗？"),
        _send(db, uid, wid, convs[0], "后来收到录用通知了吗？"),
    )
    sent = [r for r in results if r[0]]
    assert len(sent) == 1
    loser = next(r for r in results if not r[0])[2]
    assert loser["suppressed_count"] == 1
    assert loser["decision"]["reason"] == "unanswered_cooldown"
    stored = await db.query_raw(
        "SELECT metadata FROM messages WHERE conversation_id=$1 AND role='assistant'", convs[0])
    assert len(stored) == 1
    stored_decision = stored[0]["metadata"]["response_diagnostics"]["temporal_context"]["followup_decision"]
    assert stored_decision["status"] == "checked" and stored_decision["checked_count"] == 1


async def test_user_reply_during_generation_and_takeover_abort_without_writing(followup):
    db, uid, wid, convs = followup
    started = datetime.now(timezone.utc)
    await db.message.create(data={"conversationId": convs[0], "role": "user", "content": "在吗"})
    _, _, decision = await _send(db, uid, wid, convs[0], "面试顺利吗？", since=started)
    assert decision["aborted"] == "user_replied"
    await db.execute_raw(
        "INSERT INTO human_takeovers(id,user_id,status,updated_at) VALUES($1,$2,'active',NOW())",
        "followup-takeover-" + uid, uid)
    _, _, decision = await _send(db, uid, wid, convs[0], "面试顺利吗？")
    assert decision["aborted"] == "human_takeover"
    assert not await db.query_raw(
        "SELECT 1 FROM messages WHERE conversation_id=$1 AND role='assistant'", convs[0])


async def test_history_read_is_scoped_bounded_and_excludes_other_users(followup):
    db, uid, wid, convs = followup
    history, truncated = await load_followup_history(
        db, user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc))
    contents = [r["content"] for r in history]
    assert "我明天下午有个面试" in contents and not truncated
    # Oldest first, ordered by the database clock.
    assert [r["created_at"] for r in history] == sorted(r["created_at"] for r in history)
    other, _ = await load_followup_history(
        db, user_id="someone-else", workspace_id=wid, now=datetime.now(timezone.utc))
    assert other == []


async def test_ordinary_reply_keeps_text_when_history_is_unavailable(followup, monkeypatch):
    db, uid, wid, convs = followup
    from app.services.time_domain import followup_decision

    async def down(*_args, **_kwargs):
        raise ConnectionError("db down")

    monkeypatch.setattr(followup_decision, "load_followup_history", down)
    replies, decision = await guard_reply_followups(
        ["今天降温了。面试结果出来了吗？"], user_id=uid, workspace_id=wid,
        now=datetime.now(timezone.utc), current_user_message="好冷", database=db)
    assert replies == ["今天降温了。"] and decision["status"] == "unavailable"


from tests.test_proactive_activity_e2e import flow  # noqa: E402,F401  (isolated fixture)


async def _seed_ignored_ask(flow):  # noqa: F811
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    for minutes, role, text in ((26 * 60, "user", "我明天下午有个面试"),
                                (24 * 60, "assistant", "面试结果出来了吗？")):
        await flow.db.message.create(data={"conversationId": flow.conversation, "role": role,
                                           "content": text, "createdAt": now - timedelta(minutes=minutes)})


async def test_proactive_emit_refuses_a_rephrased_repeat_and_reports_why(flow):  # noqa: F811
    from app.services.proactive import emit

    await _seed_ignored_ask(flow)
    reasons: list[str] = []
    sent = await emit.emit_proactive_message(
        conversation_id=flow.conversation, user_id=flow.user, agent_id=flow.agent,
        workspace_id=flow.workspace, message="后来收到录用通知了吗？", trigger_type="silence_wakeup",
        voice_eligible=False, skip_reasons=reasons)
    assert sent == "" and reasons == ["followup_repeat_suppressed"]


async def test_proactive_emit_stores_the_decision_with_an_allowed_first_ask(flow):  # noqa: F811
    from app.services.proactive import emit

    await flow.db.message.create(data={"conversationId": flow.conversation, "role": "user",
                                       "content": "下周要答辩"})
    sent = await emit.emit_proactive_message(
        conversation_id=flow.conversation, user_id=flow.user, agent_id=flow.agent,
        workspace_id=flow.workspace, message="答辩顺利吗？", trigger_type="memory_proactive",
        voice_eligible=False)
    row = await flow.db.message.find_unique(where={"id": sent})
    decision = row.metadata["response_diagnostics"]["temporal_context"]["followup_decision"]
    assert decision["status"] == "checked" and decision["decision"]["reason"] == "user_discussing_event"


async def test_proactive_emit_eligibility_failure_sends_nothing(flow, monkeypatch):  # noqa: F811
    from unittest.mock import AsyncMock
    from app.services.proactive import emit
    from app.services.time_domain import followup_decision

    monkeypatch.setattr(followup_decision, "checked_proactive_insert",
                        AsyncMock(side_effect=TimeoutError()))
    reasons: list[str] = []
    sent = await emit.emit_proactive_message(
        conversation_id=flow.conversation, user_id=flow.user, agent_id=flow.agent,
        workspace_id=flow.workspace, message="面试结果出来了吗？", trigger_type="silence_wakeup",
        voice_eligible=False, skip_reasons=reasons)
    assert sent == "" and reasons == ["followup_decision_unavailable"]
    assert not await flow.db.query_raw(
        "SELECT 1 FROM messages WHERE conversation_id=$1 AND role='assistant'", flow.conversation)


async def test_ordinary_guard_reads_through_the_default_database(flow):  # noqa: F811
    from datetime import datetime, timezone
    from app.services.time_domain.followup_decision import guard_reply_followups

    await _seed_ignored_ask(flow)
    replies, decision = await guard_reply_followups(
        ["今天降温了。录用通知来了没？"], user_id=flow.user, workspace_id=flow.workspace,
        now=datetime.now(timezone.utc), current_user_message="好冷")
    assert replies == ["今天降温了。"] and decision["suppressed_count"] == 1
    replies, decision = await guard_reply_followups(
        ["面试结果出来了吗？"], user_id=flow.user, workspace_id=None,
        now=datetime.now(timezone.utc), current_user_message="")
    assert replies == ["好，我听着。"] and decision["status"] == "scope_unavailable"
