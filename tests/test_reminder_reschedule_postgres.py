"""Real transaction, source and race checks for confirmed reminder rescheduling."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prisma import Json

from app.services.memory.manual_write import MemoryWriteError
from app.services.memory.storage import repo
from app.services.reminder import reschedule as r, scheduling
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_input_write_api_postgres import reject_write
from app.services.reminder.scheduling import notify_reminder_changed as _real_notify


@pytest.fixture
async def rescheduling(origins, monkeypatch):
    db, uid, wid, _, _, _, _, users, agents, spaces, convs = origins
    monkeypatch.setattr(r, "db", db)
    monkeypatch.setattr(scheduling, "notify_reminder_changed", AsyncMock())
    now = datetime.now(timezone.utc).replace(microsecond=0)
    old_time = now + timedelta(days=2)
    memory = await repo.create(
        userId=uid,
        workspaceId=wid,
        content="提醒我喝水",
        level=3,
        importance=0.4,
        type="life",
        mainCategory="生活",
        subCategory="提醒",
        provenance="user_stated",
        occurTime=old_time,
    )
    trigger = await db.timetrigger.create(
        data={
            "userId": uid,
            "aiAgentId": agents[0],
            "triggerTime": old_time,
            "actionType": "reminder",
            "actionData": Json({"memory_id": memory.id, "memory_side": "user"}),
        }
    )
    first = await db.message.create(
        data={
            "conversationId": convs[0],
            "role": "user",
            "content": "把喝水提醒改到明天下午三点",
            "createdAt": now,
        }
    )
    confirmation = await db.message.create(
        data={"conversationId": convs[0], "role": "user", "content": "对"}
    )
    scope = dict(
        conversation_id=convs[0], user_id=uid, workspace_id=wid, agent_id=agents[0]
    )
    candidates = [{"id": memory.id, "source": "user", "content": memory.content}]
    ticket = await r.prepare_ticket(
        candidates=candidates,
        **scope,
        message_ids=[first.id],
        user_message=first.content,
        now=now,
    )

    async def apply(**overrides):
        return await r.apply_ticket(
            ticket,
            **{
                **scope,
                "message_ids": [confirmation.id],
                "user_message": confirmation.content,
                "now": now,
                **overrides,
            },
        )

    yield SimpleNamespace(**locals())
    await db.timetrigger.delete_many(where={"userId": {"in": users}})


async def test_reschedule_is_atomic_and_replay_safe(rescheduling):
    f = rescheduling
    result = await f.apply()
    assert result["status"] == "applied" and result["updated"] == 1
    memory = await f.db.usermemory.find_unique(where={"id": f.memory.id})
    trigger = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert (
        memory.occurTime
        == trigger.triggerTime
        == datetime.fromisoformat(f.ticket["new_time"])
    )
    assert (
        memory.content == f.memory.content
        and trigger.isActive
        and trigger.lastFired is None
    )
    assert (await f.apply())["status"] == "already_applied"
    assert (
        await f.db.memorychangelog.count(
            where={"memoryId": f.memory.id, "operation": "reschedule"}
        )
        == 1
    )
    assert (
        len(
            await f.db.query_raw(
                "SELECT * FROM memory_input_origins WHERE workspace_id=$1", f.wid
            )
        )
        == 1
    )
    assert (
        len(
            await f.db.query_raw(
                "SELECT * FROM memory_evidence_links WHERE memory_id=$1", f.memory.id
            )
        )
        == 3
    )


@pytest.mark.parametrize(
    "table",
    [
        "time_triggers",
        "memory_changelogs",
        "memory_input_origins",
        "memory_evidence_links",
    ],
)
async def test_failure_rolls_back_both_times_and_evidence(rescheduling, table):
    f = rescheduling
    async with reject_write(
        f.db,
        table,
        "TRUE",
        operation="UPDATE" if table == "time_triggers" else "INSERT",
    ):
        with pytest.raises(Exception):
            await f.apply()
    assert (
        await f.db.usermemory.find_unique(where={"id": f.memory.id})
    ).occurTime == f.old_time
    assert (
        await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    ).triggerTime == f.old_time
    assert not await f.db.query_raw(
        "SELECT id FROM memory_input_origins WHERE workspace_id=$1", f.wid
    )
    assert not await f.db.query_raw(
        "SELECT id FROM memory_evidence_links WHERE memory_id=$1", f.memory.id
    )


@pytest.mark.parametrize(
    "change",
    [
        "memory",
        "trigger",
        "source",
        "source_clock",
        "card",
        "delete_source",
        "archive_memory",
        "inactive_trigger",
        "scope",
    ],
)
async def test_changed_target_source_and_scope_are_rejected(rescheduling, change):
    f = rescheduling
    if change == "memory":
        await f.db.usermemory.update(
            where={"id": f.memory.id}, data={"content": "提醒吃饭"}
        )
    elif change == "trigger":
        await f.db.timetrigger.update(
            where={"id": f.trigger.id},
            data={"triggerTime": f.old_time + timedelta(hours=1)},
        )
    elif change == "source":
        await f.db.message.update(
            where={"id": f.first.id}, data={"content": "不用改了"}
        )
    elif change == "source_clock":
        await f.db.message.update(
            where={"id": f.first.id}, data={"createdAt": f.now - timedelta(days=1)}
        )
    elif change == "card":
        await f.db.message.update(
            where={"id": f.first.id}, data={"metadata": Json({"component_card": {}})}
        )
    elif change == "delete_source":
        await f.db.message.delete(where={"id": f.first.id})
    elif change == "archive_memory":
        await f.db.usermemory.update(
            where={"id": f.memory.id}, data={"isArchived": True}
        )
    elif change == "inactive_trigger":
        await f.db.timetrigger.update(
            where={"id": f.trigger.id}, data={"isActive": False}
        )
    elif change == "scope":
        await f.db.chatworkspace.update(
            where={"id": f.wid}, data={"status": "archived"}
        )
    with pytest.raises((MemoryWriteError, ValueError)):
        await f.apply()
    assert not await f.db.query_raw(
        "SELECT id FROM memory_input_origins WHERE workspace_id=$1", f.wid
    )


async def test_concurrent_confirmation_applies_once(rescheduling):
    f = rescheduling
    results = await asyncio.gather(f.apply(), f.apply())
    assert sorted(x["status"] for x in results) == ["already_applied", "applied"]
    assert (
        await f.db.memorychangelog.count(
            where={"memoryId": f.memory.id, "operation": "reschedule"}
        )
        == 1
    )


async def test_expired_foreign_and_same_turn_confirmation_fail(rescheduling):
    f = rescheduling
    for overrides in (
        {"now": f.now + timedelta(minutes=31)},
        {"agent_id": f.agents[-1]},
        {"message_ids": [f.first.id], "user_message": f.first.content},
    ):
        with pytest.raises(MemoryWriteError):
            await f.apply(**overrides)


@pytest.mark.parametrize(
    "text",
    [
        "明天下午三点还是四点",
        "如果改到明天下午三点呢",
        "他说改到明天下午三点",
        "不要改到明天下午三点",
    ],
)
async def test_uncertain_reported_and_denied_target_not_a_proposal(rescheduling, text):
    f = rescheduling
    first = await f.db.message.update(where={"id": f.first.id}, data={"content": text})
    with pytest.raises(MemoryWriteError):
        await r.prepare_ticket(
            candidates=f.candidates,
            **f.scope,
            message_ids=[first.id],
            user_message=text,
            now=f.now,
        )


@pytest.mark.parametrize("action", ["reschedule", "complete", "delete"])
async def test_checkin_mutation_updates_memory_and_trigger_together(
    rescheduling, monkeypatch, action
):
    from app.services.reminder import checkin

    f = rescheduling
    monkeypatch.setattr(checkin, "db", f.db)
    monkeypatch.setattr(checkin, "notify_reminder_changed", AsyncMock())
    user = {"sub": f.uid, "role": "user"}
    if action == "reschedule":
        result = await checkin.update_reminder_for_user(
            f.trigger.id,
            checkin.ReminderUpdate(trigger_time=f.ticket["new_time"]),
            user,
        )
        assert (
            result.trigger_time
            == datetime.fromisoformat(f.ticket["new_time"])
            .astimezone(timezone.utc)
            .isoformat()
        )
    elif action == "complete":
        result = await checkin.complete_reminder_for_user(f.trigger.id, user=user)
        assert result.completed_at is not None and result.last_fired is None
    else:
        await checkin.delete_reminder_for_user(f.trigger.id, user=user)
    memory = await f.db.usermemory.find_unique(where={"id": f.memory.id})
    trigger = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert memory.isArchived == (action != "reschedule")
    assert trigger.isActive == (action == "reschedule")
    if action == "reschedule":
        assert memory.occurTime == trigger.triggerTime
    with pytest.raises(MemoryWriteError):
        await f.apply()  # old chat ticket is stale


@pytest.mark.parametrize("action", ["reschedule", "complete", "delete"])
async def test_checkin_fault_rolls_back_trigger_and_memory(
    rescheduling, monkeypatch, action
):
    from app.services.reminder import checkin

    f = rescheduling
    monkeypatch.setattr(checkin, "db", f.db)
    user = {"sub": f.uid, "role": "user"}
    async with reject_write(f.db, "memory_changelogs", "TRUE"):
        with pytest.raises(Exception):
            if action == "reschedule":
                await checkin.update_reminder_for_user(
                    f.trigger.id,
                    checkin.ReminderUpdate(trigger_time=f.ticket["new_time"]),
                    user,
                )
            elif action == "complete":
                await checkin.complete_reminder_for_user(f.trigger.id, user=user)
            else:
                await checkin.delete_reminder_for_user(f.trigger.id, user=user)
    memory = await f.db.usermemory.find_unique(where={"id": f.memory.id})
    trigger = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert not memory.isArchived and trigger.isActive
    assert memory.occurTime == trigger.triggerTime == f.old_time
    assert (
        trigger.lastFired is None
        and "completed_at" not in trigger.actionData
        and "deleted_at" not in trigger.actionData
    )


async def test_old_scheduler_snapshot_cannot_claim_rescheduled_reminder(rescheduling):
    from app.services.reminder.mutations import mutate_reminder

    f = rescheduling
    await f.apply()
    with pytest.raises(MemoryWriteError, match="reminder_changed"):
        await mutate_reminder(
            f.db,
            trigger=f.trigger,
            trigger_changes={"isActive": False, "lastFired": f.now},
            operation="reminder_claimed",
        )
    assert (await f.db.timetrigger.find_unique(where={"id": f.trigger.id})).isActive


async def test_sql_claim_cas_blocks_stale_scan_after_chat_reschedule(rescheduling):
    from app.services.reminder.mutations import claim_due_reminder

    f = rescheduling
    # Make the original trigger due. The request is still for a future new time.
    due = f.now - timedelta(seconds=20)
    await f.db.timetrigger.update(where={"id": f.trigger.id}, data={"triggerTime": due})
    await f.db.usermemory.update(where={"id": f.memory.id}, data={"occurTime": due})
    ticket = await r.prepare_ticket(
        candidates=f.candidates,
        **f.scope,
        message_ids=[f.first.id],
        user_message=f.first.content,
        now=f.now,
    )
    old = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    await r.apply_ticket(
        ticket,
        **f.scope,
        message_ids=[f.confirmation.id],
        user_message=f.confirmation.content,
        now=f.now,
    )
    assert not await claim_due_reminder(f.db, old, f.now)
    assert (await f.db.timetrigger.find_unique(where={"id": f.trigger.id})).isActive


async def test_sql_claim_two_workers_only_one_and_does_not_complete_event(rescheduling):
    from app.services.reminder.mutations import claim_due_reminder

    f = rescheduling
    trigger = await f.db.timetrigger.update(
        where={"id": f.trigger.id}, data={"triggerTime": f.now - timedelta(seconds=20)}
    )
    claimed = await asyncio.gather(
        claim_due_reminder(f.db, trigger, f.now),
        claim_due_reminder(f.db, trigger, f.now),
    )
    assert sorted(claimed) == [False, True]
    row = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert row.lastFired is not None and not row.actionData.get("completed_at")
    assert not (await f.db.usermemory.find_unique(where={"id": f.memory.id})).isArchived


@pytest.mark.parametrize("state", ["completed", "cancelled", "rescheduled"])
async def test_precheck_atomic_state_uses_user_evidence_not_delivery(
    rescheduling, state
):
    from app.services.reminder.precheck import load_context, apply_state

    f = rescheduling
    text = {
        "completed": "我已经喝完水了",
        "cancelled": "不用提醒我喝水了",
        "rescheduled": "把喝水提醒改到明天下午三点",
    }[state]
    await f.db.message.update(where={"id": f.first.id}, data={"content": text})
    context = await load_context(f.db, agent_id=f.agents[0], user_id=f.uid, now=f.now)
    result = await apply_state(
        f.db, trigger=f.trigger, state=state, context=context, now=f.now
    )
    memory = await f.db.usermemory.find_unique(where={"id": f.memory.id})
    assert result.lastFired is None
    assert memory.isArchived == (state != "rescheduled")
    assert result.isActive == (state == "rescheduled")
    if state == "rescheduled":
        assert memory.occurTime == result.triggerTime
    else:
        assert result.actionData[state + "_at"]
    assert (
        await f.db.memorychangelog.count(
            where={"memoryId": f.memory.id, "operation": "reminder_" + state}
        )
        == 1
    )


@pytest.mark.parametrize(
    "fault", ["edit", "delete", "card", "other_agent", "write_failure"]
)
async def test_precheck_source_or_transaction_failure_never_partially_mutates(
    rescheduling, fault
):
    from app.services.reminder.precheck import load_context, apply_state

    f = rescheduling
    context = await load_context(f.db, agent_id=f.agents[0], user_id=f.uid, now=f.now)
    if fault == "edit":
        await f.db.message.update(
            where={"id": f.first.id}, data={"content": "其实没喝水"}
        )
    elif fault == "delete":
        await f.db.message.delete(where={"id": f.first.id})
    elif fault == "card":
        await f.db.message.update(
            where={"id": f.first.id}, data={"metadata": Json({"component_card": {}})}
        )
    elif fault == "other_agent":
        context = await load_context(
            f.db, agent_id=f.agents[-1], user_id=f.uid, now=f.now
        )
    if fault == "write_failure":
        async with reject_write(f.db, "memory_changelogs", "TRUE"):
            with pytest.raises(Exception):
                await apply_state(
                    f.db,
                    trigger=f.trigger,
                    state="completed",
                    context=context,
                    now=f.now,
                )
    else:
        with pytest.raises((MemoryWriteError, ValueError)):
            await apply_state(
                f.db, trigger=f.trigger, state="completed", context=context, now=f.now
            )
    row = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert (
        row.isActive and row.lastFired is None and "completed_at" not in row.actionData
    )
    assert not (await f.db.usermemory.find_unique(where={"id": f.memory.id})).isArchived


async def test_precheck_does_not_read_ai_guesses_or_unbounded_history(rescheduling):
    from app.services.reminder.precheck import load_context

    f = rescheduling
    await f.db.message.delete_many(where={"conversationId": f.convs[0]})
    await f.db.message.create(
        data={
            "conversationId": f.convs[0],
            "role": "assistant",
            "content": "你喝完水了",
            "createdAt": f.now,
        }
    )
    context = await load_context(f.db, agent_id=f.agents[0], user_id=f.uid, now=f.now)
    assert not context.text and not context.sources
    await f.db.message.create_many(
        data=[
            {
                "conversationId": f.convs[0],
                "role": "user",
                "content": "消息" + str(i),
                "createdAt": f.now,
            }
            for i in range(41)
        ]
    )
    context = await load_context(f.db, agent_id=f.agents[0], user_id=f.uid, now=f.now)
    assert context.truncated and not context.sources


async def test_precheck_periodic_completion_only_completes_current_occurrence(
    rescheduling,
):
    from app.services.reminder.precheck import load_context, apply_state

    f = rescheduling
    data = {**f.trigger.actionData, "recurrence": "daily", "summary": "喝水"}
    trigger = await f.db.timetrigger.update(
        where={"id": f.trigger.id},
        data={"triggerTime": f.now, "actionData": Json(data)},
    )
    context = await load_context(f.db, agent_id=f.agents[0], user_id=f.uid, now=f.now)
    result = await apply_state(
        f.db, trigger=trigger, state="completed", context=context, now=f.now
    )
    assert (
        result.isActive
        and result.lastFired is None
        and result.triggerTime == f.now + timedelta(days=1)
    )
    assert result.actionData["completed_dates"] and not result.actionData.get(
        "completed_at"
    )
    memory = await f.db.usermemory.find_unique(where={"id": f.memory.id})
    assert not memory.isArchived and memory.occurTime == result.triggerTime


@pytest.mark.parametrize("state", ["completed", "cancelled", "rescheduled"])
async def test_real_scheduler_state_path_commits_once_without_sending(
    rescheduling, monkeypatch, state
):
    from app.services.proactive import triggers

    f = rescheduling
    monkeypatch.setattr(triggers, "db", f.db)
    monkeypatch.setattr(triggers, "get_redis", AsyncMock())
    monkeypatch.setattr(
        "app.services.admin_takeover.service.is_user_taken_over",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "app.services.chat.intent_replies.reminder_pre_check",
        AsyncMock(return_value={"state": state}),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.services.proactive.emit.emit_proactive_message", emit)
    data = {**f.trigger.actionData, "summary": "喝水", "sent_to_ai": True}
    row = await f.db.timetrigger.update(
        where={"id": f.trigger.id},
        data={"triggerTime": f.now, "actionData": Json(data)},
    )
    await asyncio.gather(
        triggers._handle_reminder_trigger(row, f.now),
        triggers._handle_reminder_trigger(row, f.now),
    )
    emit.assert_not_awaited()
    assert (
        await f.db.memorychangelog.count(
            where={"memoryId": f.memory.id, "operation": "reminder_" + state}
        )
        == 1
    )
    current = await f.db.timetrigger.find_unique(where={"id": row.id})
    assert current.lastFired is None
    assert current.isActive == (state == "rescheduled")


@pytest.mark.parametrize("state,failure", [
    ("rescheduled", None), ("completed", None), ("cancelled", RuntimeError("db down")),
])
async def test_unprovable_precheck_state_still_delivers_once(rescheduling, monkeypatch, state, failure):
    # "晚上八点再提醒我" has no explicit 改到 marker and "对" proves nothing:
    # the classifier's guess is not applied, and the due reminder is not
    # silently stranded outside the ±1 minute scan window either.
    from app.services.proactive import triggers

    f = rescheduling
    monkeypatch.setattr(triggers, "db", f.db)
    monkeypatch.setattr(triggers, "get_redis", AsyncMock())
    monkeypatch.setattr(
        "app.services.admin_takeover.service.is_user_taken_over",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "app.services.chat.intent_replies.reminder_pre_check",
        AsyncMock(return_value={"state": state}),
    )
    monkeypatch.setattr(
        "app.services.reminder.precheck.load_context",
        AsyncMock(return_value=__import__(
            "app.services.reminder.precheck", fromlist=["PrecheckContext"]
        ).PrecheckContext(truncated=True)),
    )
    from app.services.reminder import mutations

    if failure is not None:
        # Any failure while applying the guessed state, not only a contract
        # error, must leave the reminder deliverable.
        monkeypatch.setattr(
            "app.services.reminder.precheck.apply_state", AsyncMock(side_effect=failure))
    real_claim, claims = mutations.claim_due_reminder, []

    async def counted_claim(*args, **kwargs):
        claims.append(await real_claim(*args, **kwargs))
        return claims[-1]

    monkeypatch.setattr(mutations, "claim_due_reminder", counted_claim)
    # Stop after the delivery claim; the send path itself is covered elsewhere.
    monkeypatch.setattr(triggers, "resolve_workspace_id", AsyncMock(return_value=None))
    monkeypatch.setattr(triggers, "_handle_emit_failure", AsyncMock())
    data = {**f.trigger.actionData, "summary": "喝水", "sent_to_ai": True}
    row = await f.db.timetrigger.update(
        where={"id": f.trigger.id},
        data={"triggerTime": f.now, "actionData": Json(data)},
    )
    await asyncio.gather(
        triggers._handle_reminder_trigger(row, f.now),
        triggers._handle_reminder_trigger(row, f.now),
    )
    # Exactly one worker reaches delivery; neither applies the guessed state.
    assert sorted(claims) == [False, True]
    assert not await f.db.memorychangelog.count(
        where={"memoryId": f.memory.id, "operation": "reminder_" + state}
    )
    current = await f.db.timetrigger.find_unique(where={"id": row.id})
    assert "precheck_state" not in (current.actionData or {})


async def test_completed_before_firing_stays_in_checkin_history_without_false_receipt(
    rescheduling, monkeypatch
):
    from app.services.reminder import checkin

    f = rescheduling
    monkeypatch.setattr(checkin, "db", f.db)
    monkeypatch.setattr(checkin, "notify_reminder_changed", AsyncMock())
    monkeypatch.setattr(checkin, "get_redis", AsyncMock())
    result = await checkin.complete_reminder_for_user(
        f.trigger.id, user={"sub": f.uid, "role": "user"}
    )
    row = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert (
        result.completed_at
        and result.last_fired is None
        and result.event_state == "completed"
    )
    assert checkin._is_open_checkin_item(row)


@pytest.mark.parametrize(
    "mutation",
    ["none", "complete", "delete", "reschedule", "metadata", "foreign_claim"],
)
async def test_failed_dispatch_retry_cannot_revive_user_changes(
    rescheduling, monkeypatch, mutation
):
    from app.services.reminder.mutations import (
        claim_due_reminder,
        retry_claimed_reminder,
    )
    from app.services.reminder import checkin

    f = rescheduling
    monkeypatch.setattr(checkin, "db", f.db)
    monkeypatch.setattr(checkin, "notify_reminder_changed", AsyncMock())
    trigger = await f.db.timetrigger.update(
        where={"id": f.trigger.id}, data={"triggerTime": f.now - timedelta(seconds=1)}
    )
    # Fractional precision must match PostgreSQL timestamp(3), not Python microseconds.
    claimed = f.now + timedelta(microseconds=123456)
    assert await claim_due_reminder(f.db, trigger, claimed)
    user = {"sub": f.uid, "role": "user"}
    if mutation == "complete":
        await checkin.complete_reminder_for_user(trigger.id, user=user)
    if mutation == "delete":
        await checkin.delete_reminder_for_user(trigger.id, user=user)
    if mutation == "reschedule":
        await checkin.update_reminder_for_user(
            trigger.id, checkin.ReminderUpdate(trigger_time=f.ticket["new_time"]), user
        )
    if mutation == "metadata":
        await f.db.timetrigger.update(
            where={"id": trigger.id},
            data={"actionData": Json({**trigger.actionData, "summary": "已修改"})},
        )
    if mutation == "foreign_claim":
        await f.db.timetrigger.update(
            where={"id": trigger.id}, data={"lastFired": claimed + timedelta(seconds=1)}
        )
    result = await retry_claimed_reminder(
        f.db,
        trigger,
        claimed_at=claimed,
        retry_at=f.now + timedelta(seconds=30),
        retry_count=1,
    )
    assert result == (mutation == "none")
    if mutation == "none":
        row = await f.db.timetrigger.find_unique(where={"id": trigger.id})
        assert (
            row.isActive
            and row.lastFired is None
            and row.actionData["retry_count"] == 1
        )
        assert not await retry_claimed_reminder(
            f.db,
            trigger,
            claimed_at=claimed,
            retry_at=f.now + timedelta(seconds=30),
            retry_count=1,
        )


async def test_foreign_conversation_cannot_be_persisted_or_notified(
    rescheduling, monkeypatch
):
    from app.services.reminder import checkin, scheduling
    from fastapi import HTTPException
    from app.services.runtime.ws_manager import manager

    f = rescheduling
    monkeypatch.setattr(checkin, "db", f.db)
    monkeypatch.setattr(scheduling, "db", f.db)
    send = AsyncMock()
    monkeypatch.setattr(manager, "send_event", send)
    with pytest.raises(HTTPException) as failure:
        await checkin.update_reminder_for_user(
            f.trigger.id,
            checkin.ReminderUpdate(conversation_id=f.convs[-1]),
            {"sub": f.uid, "role": "user"},
        )
    assert failure.value.status_code == 409
    # Fixture replaces the scheduling notification entry; retain its real code
    # through a function saved before monkeypatch in the module-level import.
    await _real_notify(f.convs[-1], kind="rescheduled", trigger_id=f.trigger.id)
    send.assert_not_awaited()
    await _real_notify(f.convs[0], kind="rescheduled", trigger_id=f.trigger.id)
    send.assert_awaited_once()


async def test_rescheduling_after_failed_attempt_reactivates_without_marking_complete(
    rescheduling, monkeypatch
):
    from app.services.reminder import checkin

    f = rescheduling
    monkeypatch.setattr(checkin, "db", f.db)
    monkeypatch.setattr(checkin, "notify_reminder_changed", AsyncMock())
    await f.db.timetrigger.update(
        where={"id": f.trigger.id},
        data={
            "isActive": False,
            "lastFired": f.now,
            "actionData": Json({**f.trigger.actionData, "retry_count": 3}),
        },
    )
    result = await checkin.update_reminder_for_user(
        f.trigger.id,
        checkin.ReminderUpdate(trigger_time=f.ticket["new_time"]),
        {"sub": f.uid, "role": "user"},
    )
    assert (
        result.status == "active"
        and result.event_state == "pending"
        and result.last_fired is None
    )
    assert result.retry_count == 0 and result.completed_at is None


async def test_reactivating_a_cancelled_reminder_unarchives_it_so_it_can_fire(
    rescheduling, monkeypatch
):
    from app.services.reminder import checkin
    from app.services.reminder.mutations import claim_due_reminder

    f = rescheduling
    monkeypatch.setattr(checkin, "db", f.db)
    monkeypatch.setattr(checkin, "notify_reminder_changed", AsyncMock())
    # State left by a pre-check cancellation: trigger off, memory archived.
    await f.db.timetrigger.update(
        where={"id": f.trigger.id},
        data={"isActive": False,
              "actionData": Json({**f.trigger.actionData, "cancelled_at": f.now.isoformat()})},
    )
    await f.db.usermemory.update(where={"id": f.memory.id}, data={"isArchived": True})
    result = await checkin.update_reminder_for_user(
        f.trigger.id,
        checkin.ReminderUpdate(trigger_time=f.ticket["new_time"]),
        {"sub": f.uid, "role": "user"},
    )
    assert result.status == "active"
    memory = await f.db.usermemory.find_unique(where={"id": f.memory.id})
    assert not memory.isArchived
    row = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert "cancelled_at" not in row.actionData
    assert await claim_due_reminder(f.db, row, row.triggerTime)


async def test_completed_once_reminder_reads_as_done_and_orphan_can_be_deleted(
    rescheduling, monkeypatch
):
    from app.services.reminder import checkin

    f = rescheduling
    monkeypatch.setattr(checkin, "db", f.db)
    monkeypatch.setattr(checkin, "notify_reminder_changed", AsyncMock())
    owner = {"sub": f.uid, "role": "user"}
    done = await checkin.complete_reminder_for_user(f.trigger.id, user=owner)
    # No invented delivery receipt, but clients reading `status` see it as done.
    assert done.status == "fired" and done.last_fired is None and done.completed_at
    # A reminder whose memory was erased can still be removed by its owner.
    await f.db.usermemory.delete(where={"id": f.memory.id})
    assert await checkin.delete_reminder_for_user(f.trigger.id, user=owner) is None
    row = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert row.actionData.get("deleted_at") and not row.isActive
    with pytest.raises(Exception):
        await checkin.update_reminder_for_user(
            f.trigger.id, checkin.ReminderUpdate(trigger_time=f.ticket["new_time"]), owner)


def _corrupt(ticket, change):
    import copy

    broken = copy.deepcopy(ticket)
    target, pin = broken["targets"][0], broken["sources"][0]
    if change == "target_missing_key":
        del target["trigger_version"]
    elif change == "target_not_object":
        broken["targets"] = ["x"]
    elif change == "target_bad_side":
        target["source"] = "admin"
    elif change == "target_duplicate":
        broken["targets"] = [target, dict(target)]
    elif change == "pin_missing_ref":
        del pin["ref"]
    elif change == "pin_foreign_conversation":
        pin["expected_conversation_id"] = "other"
    elif change == "pin_bad_time":
        pin["expected_statement_time"] = "yesterday"
    elif change == "pin_unknown_field":
        pin["grant"] = True
    elif change == "scope_not_object":
        broken["scope"] = "x"
    return broken


@pytest.mark.parametrize(
    "change",
    [
        "target_missing_key",
        "target_not_object",
        "target_bad_side",
        "target_duplicate",
        "pin_missing_ref",
        "pin_foreign_conversation",
        "pin_bad_time",
        "pin_unknown_field",
        "scope_not_object",
    ],
)
async def test_corrupted_redis_ticket_is_rejected_before_any_write(rescheduling, change):
    from app.services.memory.interaction.deletion import apply_reschedule

    f = rescheduling
    broken = _corrupt(f.ticket, change)
    with pytest.raises(MemoryWriteError) as failure:
        await r.apply_ticket(
            broken,
            **f.scope,
            message_ids=[f.confirmation.id],
            user_message=f.confirmation.content,
            now=f.now,
        )
    assert failure.value.detail == "reschedule_ticket_invalid"
    # The chat entry rejects the same corruption as invalid, not as a crash.
    with pytest.raises(MemoryWriteError):
        await apply_reschedule(
            f.uid,
            f.candidates,
            f.ticket["new_time"],
            ticket=broken,
            message_ids=(f.confirmation.id,),
            user_message=f.confirmation.content,
            **{k: v for k, v in f.scope.items() if k != "user_id"},
        )
    trigger = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert trigger.triggerTime == f.old_time
    assert not await f.db.query_raw(
        "SELECT 1 FROM memory_input_origins WHERE workspace_id=$1", f.wid
    )


async def test_cache_invalidation_outage_does_not_report_committed_reschedule_as_failed(
    rescheduling, monkeypatch
):
    f = rescheduling
    monkeypatch.setattr(
        repo, "bump_cache_version", AsyncMock(side_effect=ConnectionError("redis down"))
    )
    result = await f.apply()
    assert result["status"] == "applied"
    trigger = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    assert trigger.triggerTime == datetime.fromisoformat(f.ticket["new_time"])


async def test_aggregate_notification_without_trigger_id_still_refreshes(
    rescheduling, monkeypatch
):
    from app.services.runtime.ws_manager import manager

    f = rescheduling
    monkeypatch.setattr(scheduling, "db", f.db)
    send = AsyncMock()
    monkeypatch.setattr(manager, "send_event", send)
    await _real_notify(f.convs[0], kind="rescheduled")
    send.assert_awaited_once()


async def _contract_violation(f, change):
    from dataclasses import replace
    from app.services.reminder.mutations import mutate_reminder

    owner = {"sub": f.uid, "role": "user"}
    trigger = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    kwargs = dict(trigger=trigger, trigger_changes={"isActive": False},
                  memory_changes={}, operation="test", actor=owner)
    if change == "unsupported_field":
        kwargs["trigger_changes"] = {"userId": "someone"}
    elif change == "memory_change_without_authority":
        kwargs.update(memory_changes={"isArchived": True}, actor=None)
    elif change == "too_many_sources":
        kwargs["sources"] = tuple(range(41))
    elif change == "bad_memory_side":
        kwargs["trigger"] = replace_trigger(trigger, {"memory_side": "admin"})
    elif change == "scheduler_with_missing_memory":
        await f.db.usermemory.delete(where={"id": f.memory.id})
        kwargs["actor"] = None
    elif change == "memory_reclassified":
        await f.db.usermemory.update(where={"id": f.memory.id}, data={"subCategory": "其他"})
    elif change == "invalid_actor":
        kwargs["actor"] = {"sub": "intruder", "role": "user"}
    elif change == "takeover":
        await f.db.execute_raw(
            "INSERT INTO human_takeovers(id,user_id,status,updated_at) VALUES($1,$2,'active',NOW())",
            "reminder-takeover-" + f.uid, f.uid)
        kwargs.update(actor=None, memory_changes={"isArchived": True}, sources=(replace(
            f.ticket_source, expected_statement_time=f.ticket_source.expected_statement_time),))
    elif change == "orphan_revival":
        await f.db.usermemory.delete(where={"id": f.memory.id})
        kwargs["trigger_changes"] = {"isActive": True}
    elif change == "orphan_stale":
        await f.db.usermemory.delete(where={"id": f.memory.id})
        await f.db.timetrigger.update(where={"id": f.trigger.id}, data={"triggerTime": f.now + timedelta(days=9)})
    return await mutate_reminder(f.db, **kwargs)


def replace_trigger(trigger, data):
    from types import SimpleNamespace as NS
    return NS(**{**trigger.__dict__, "actionData": {**trigger.actionData, **data}})


@pytest.mark.parametrize("change,status,detail", [
    ("unsupported_field", 422, "unsupported_reminder_mutation"),
    ("memory_change_without_authority", 409, "reminder_mutation_source_required"),
    ("too_many_sources", 422, "reminder_mutation_source_limit"),
    ("bad_memory_side", 409, "reminder_memory_unavailable"),
    ("scheduler_with_missing_memory", 409, "reminder_memory_unavailable"),
    ("memory_reclassified", 409, "reminder_memory_scope_changed"),
    ("invalid_actor", 403, "reminder_actor_invalid"),
    ("takeover", 409, "reminder_human_takeover"),
    ("orphan_revival", 409, "reminder_memory_unavailable"),
    ("orphan_stale", 409, "reminder_changed_refresh_and_retry"),
])
async def test_reminder_mutation_contract_rejects_without_writing(rescheduling, change, status, detail):
    from app.services.memory.evidence import EvidenceSource, content_version, message_statement_time

    f = rescheduling
    f.ticket_source = EvidenceSource(
        "message", f.first.id, expected_role="user", expected_conversation_id=f.convs[0],
        expected_version=content_version(f.first.content),
        expected_statement_time=message_statement_time(f.first.createdAt))
    before = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    with pytest.raises(MemoryWriteError) as failure:
        await _contract_violation(f, change)
    assert (failure.value.status_code, failure.value.detail) == (status, detail)
    after = await f.db.timetrigger.find_unique(where={"id": f.trigger.id})
    if change != "orphan_stale":
        assert after.isActive == before.isActive and after.triggerTime == before.triggerTime


async def test_content_edit_rewrites_embedding_in_the_same_transaction(rescheduling, monkeypatch):
    from app.services.reminder import checkin, mutations

    f = rescheduling
    monkeypatch.setattr(checkin, "db", f.db)
    monkeypatch.setattr(checkin, "notify_reminder_changed", AsyncMock())
    monkeypatch.setattr(mutations, "generate_embedding", AsyncMock(return_value=[0.1] * 1024))
    stored = AsyncMock()
    monkeypatch.setattr(mutations, "store_embedding", stored)
    result = await checkin.update_reminder_for_user(
        f.trigger.id, checkin.ReminderUpdate(summary="提醒我喝温水"), {"sub": f.uid, "role": "user"})
    assert result.summary == "提醒我喝温水"
    assert stored.await_args.args[0] == f.memory.id
    memory = await f.db.usermemory.find_unique(where={"id": f.memory.id})
    assert memory.content == "提醒我喝温水"


async def test_claim_and_retry_refuse_an_unknown_memory_side(rescheduling):
    from app.services.reminder.mutations import claim_due_reminder, retry_claimed_reminder

    f = rescheduling
    trigger = replace_trigger(await f.db.timetrigger.find_unique(where={"id": f.trigger.id}),
                              {"memory_side": "admin"})
    assert not await claim_due_reminder(f.db, trigger, f.now)
    assert not await retry_claimed_reminder(f.db, trigger, claimed_at=f.now,
                                            retry_at=f.now + timedelta(seconds=30), retry_count=1)
