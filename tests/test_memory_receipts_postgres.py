"""Real PostgreSQL recovery and erasure. Only model/embedding outputs are synthetic."""

import asyncio
from unittest.mock import AsyncMock
import pytest

from app.services.memory.recording import pipeline, receipts, receipt_followups
from app.services.memory.storage import persistence, embedding, entity_repo
from app.services.memory.storage.reconciliation import ReconciliationDecision
from app.services.runtime import execution_scope
from tests.test_memory_evidence_postgres import origins


@pytest.fixture
async def ledger(origins, monkeypatch):
    db, uid, wid, mid, msg, *_ = origins
    await db.aimemory.delete(where={"id": mid})
    for module in (
        receipts,
        receipt_followups,
        execution_scope,
        persistence,
        embedding,
        entity_repo,
    ):
        monkeypatch.setattr(module, "db", db)
    monkeypatch.setattr(
        persistence, "generate_embedding", AsyncMock(return_value=[1.0] + [0.0] * 1023)
    )
    monkeypatch.setattr(
        persistence,
        "resolve_memory_write",
        AsyncMock(return_value=ReconciliationDecision(action="insert_new")),
    )
    monkeypatch.setattr(persistence, "after_memory_changelog", AsyncMock())
    monkeypatch.setattr(pipeline, "should_memorize", AsyncMock(return_value=True))
    monkeypatch.setattr(pipeline, "should_extract_memory", lambda _: True)
    facts = [
        {
            "content": "用户今天参观了合成展览",
            "importance": 0.6,
            "main_category": "生活",
            "sub_category": "其他",
            "source_indices": [1],
        },
        {
            "content": "用户参加了合成读书活动",
            "importance": 0.6,
            "main_category": "生活",
            "sub_category": "其他",
            "source_indices": [1],
        },
    ]
    extract = AsyncMock(
        return_value={
            "memories": facts,
            "entities": [],
            "preferences": [],
            "topics": [],
        }
    )
    monkeypatch.setattr(pipeline, "extract_memories", extract)
    args = dict(
        user_id=uid,
        workspace_id=wid,
        side="user",
        new_conversation="user: " + msg.content,
        evidence_message_ids=[msg.id],
        evidence_conversation_id=msg.conversationId,
    )
    return db, uid, wid, msg, args, extract


async def test_committed_batch_is_not_extracted_or_written_again(ledger):
    db, uid, wid, msg, args, extract = ledger
    first = await pipeline.process_memory_pipeline(**args)
    assert len(first) == 2
    again = await pipeline.process_memory_pipeline(**args)
    assert set(again) == set(first)
    assert extract.await_count == 1
    assert len(await db.query_raw("SELECT * FROM memory_write_receipts")) == 2
    assert (
        await db.query_raw("SELECT status,extraction FROM memory_extraction_batches")
    ) == [{"status": "completed", "extraction": None}]
    assert (
        await db.memorychangelog.count(where={"userId": uid, "operation": "insert"})
        == 2
    )


@pytest.mark.parametrize("edit", ["content", "archive", "clock"])
async def test_changed_memory_cancels_old_pending_effects(ledger, monkeypatch, edit):
    from datetime import datetime, timezone

    db, uid, wid, msg, args, extract = ledger
    extract.return_value["memories"] = extract.return_value["memories"][:1]
    finish = receipt_followups.finish_followups
    monkeypatch.setattr(
        receipt_followups,
        "finish_followups",
        AsyncMock(side_effect=RuntimeError("interrupt")),
    )
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    mid = (await db.query_raw("SELECT memory_id FROM memory_write_receipts"))[0][
        "memory_id"
    ]
    change = (
        {"content": "管理员已更正"}
        if edit == "content"
        else (
            {"isArchived": True}
            if edit == "archive"
            else {"occurTime": datetime(2020, 1, 1, tzinfo=timezone.utc)}
        )
    )
    await db.usermemory.update(where={"id": mid}, data=change)
    monkeypatch.setattr(receipt_followups, "finish_followups", finish)
    await pipeline.process_memory_pipeline(**args)
    assert (
        await db.query_raw(
            "SELECT followup_done,followup_outcome FROM memory_write_receipts"
        )
    ) == [{"followup_done": True, "followup_outcome": "superseded"}]
    assert (
        await db.memorychangelog.count(
            where={"userId": uid, "operation": "evidence_linked"}
        )
        == 0
    )


async def test_retry_keeps_frozen_time_and_rolls_back_followups_as_a_unit(
    ledger, monkeypatch
):
    from datetime import datetime, timezone
    from app.services.memory.recording import temporal_sources

    db, uid, wid, msg, args, extract = ledger
    body = "一定要记住，昨天去合成博物馆看展"
    stamp = datetime(2026, 10, 20, 1, tzinfo=timezone.utc)
    await db.message.update(
        where={"id": msg.id}, data={"content": body, "createdAt": stamp}
    )
    args = dict(args, new_conversation="user: " + body)
    fact = extract.return_value["memories"][0]
    fact.update(
        content="用户去合成博物馆看展",
        time_quote="昨天",
        time_source_index=1,
        entities=["合成博物馆"],
        topics=["看展"],
        preferences=[{"value": "看展", "category": "爱好"}],
    )
    extract.return_value["memories"] = [fact]
    original = entity_repo.record_topics_for_memory

    async def fail(**kw):
        await original(**kw)
        await kw["database"].execute_raw("SELECT 1/0")

    monkeypatch.setattr(entity_repo, "record_topics_for_memory", fail)
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    assert not await db.query_raw(
        "SELECT id FROM memory_entities WHERE user_id=$1", uid
    )
    assert (
        await db.memorychangelog.count(
            where={"userId": uid, "operation": "user_emphasized"}
        )
        == 0
    )
    monkeypatch.setattr(entity_repo, "record_topics_for_memory", original)
    monkeypatch.setattr(
        temporal_sources,
        "resolve_fact_clock",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("frozen clock must not be reparsed")
        ),
    )
    ids = await pipeline.process_memory_pipeline(**args)
    memory = await db.usermemory.find_unique(where={"id": ids[0]})
    assert memory.occurTime == datetime(2026, 10, 18, 16, tzinfo=timezone.utc)
    assert memory.statementTime == stamp
    assert (
        await db.memorychangelog.count(
            where={"userId": uid, "operation": "user_emphasized"}
        )
        == 1
    )
    assert (
        len(await db.query_raw("SELECT id FROM memory_entities WHERE user_id=$1", uid))
        >= 2
    )
    assert (
        await db.query_raw("SELECT followup_outcome FROM memory_write_receipts")
    ) == [{"followup_outcome": "applied"}]
    assert extract.await_count == 1


async def test_receipt_rejects_changed_identity_late_piece_and_stale_epoch(ledger):
    from dataclasses import replace

    db, uid, wid, msg, args, extract = ledger
    await pipeline.process_memory_pipeline(**args)
    b = (await db.query_raw("SELECT * FROM memory_extraction_batches"))[0]
    scope = await execution_scope.bind_conversation_scope(
        actor_user_id=uid, conversation_id=msg.conversationId
    )
    epoch = (
        await db.query_raw(
            "SELECT generation FROM memory_erasure_epochs WHERE workspace_id=$1", wid
        )
    )[0]["generation"]
    t = receipts.WriteToken(b["id"], 0, epoch, scope, "user")
    with pytest.raises(receipts.MemoryReceiptError, match="input_changed"):
        await receipts.lookup(t, "0" * 64)
    with pytest.raises(receipts.MemoryReceiptError, match="already_completed"):
        await receipts.lookup(t.piece(99), "0" * 64)
    with pytest.raises(receipts.MemoryReceiptError, match="erasure_changed"):
        await receipts.lookup(replace(t, epoch=epoch + 1), "0" * 64)
    with pytest.raises(receipts.MemoryReceiptError, match="plan_invalid"):
        await receipts.lookup(replace(t, side="ai"), "0" * 64)


async def test_skipped_low_score_fact_has_a_receipt_and_no_side_effects(ledger):
    db, uid, wid, msg, args, extract = ledger
    extract.return_value = {
        "memories": [
            {
                "content": "不值得保存的合成事实",
                "importance": 0.01,
                "source_indices": [1],
            }
        ]
    }
    assert await pipeline.process_memory_pipeline(**args) == []
    assert (
        await db.query_raw("SELECT status,memory_id FROM memory_write_receipts")
    ) == [{"status": "skipped", "memory_id": None}]
    assert (
        await pipeline.process_memory_pipeline(**args) == []
        and extract.await_count == 1
    )


async def test_partial_write_resumes_frozen_plan_without_duplicates(
    ledger, monkeypatch
):
    db, uid, wid, msg, args, extract = ledger
    original = persistence._audit_write
    calls = 0

    async def fail_second(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            await a[0].execute_raw("SELECT 1/0")
        await original(*a, **kw)

    monkeypatch.setattr(persistence, "_audit_write", fail_second)
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    assert (
        await db.memorychangelog.count(where={"userId": uid, "operation": "insert"})
        == 1
    )
    extract.side_effect = AssertionError("must resume frozen extraction")
    monkeypatch.setattr(persistence, "_audit_write", original)
    ids = await pipeline.process_memory_pipeline(**args)
    assert len(ids) == 2
    assert (
        await db.memorychangelog.count(where={"userId": uid, "operation": "insert"})
        == 2
    )
    assert extract.await_count == 1


async def test_new_messages_do_not_change_a_partially_written_batch(
    ledger, monkeypatch
):
    db, uid, wid, msg, args, extract = ledger
    finish = receipt_followups.finish_followups
    monkeypatch.setattr(
        receipt_followups,
        "finish_followups",
        AsyncMock(side_effect=RuntimeError("after commit")),
    )
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    monkeypatch.setattr(receipt_followups, "finish_followups", finish)
    next_message = await db.message.create(
        data={
            "conversationId": msg.conversationId,
            "role": "user",
            "content": "新的独立事实",
        }
    )
    extract.return_value = {
        "memories": [
            {
                "content": "用户新报的独立事实",
                "importance": 0.6,
                "main_category": "生活",
                "sub_category": "其他",
                "source_indices": [1],
            }
        ]
    }
    ids = await pipeline.process_memory_pipeline(
        **dict(
            args,
            evidence_message_ids=[msg.id, next_message.id],
            new_conversation=args["new_conversation"]
            + "\nuser: "
            + next_message.content,
        )
    )
    assert len(ids) == 3 and extract.await_count == 2
    assert (
        await db.memorychangelog.count(where={"userId": uid, "operation": "insert"})
        == 3
    )


async def test_deleted_completed_receipt_does_not_resurrect(ledger):
    db, uid, wid, msg, args, extract = ledger
    ids = await pipeline.process_memory_pipeline(**args)
    await db.usermemory.delete(where={"id": ids[0]})
    assert await pipeline.process_memory_pipeline(**args) == [ids[1]]
    assert (
        await db.query_raw(
            "SELECT count(*)::int n FROM memory_write_receipts WHERE status='erased'"
        )
    )[0]["n"] == 1
    assert await db.usermemory.find_unique(where={"id": ids[0]}) is None
    assert extract.await_count == 1


async def test_delete_during_partial_batch_redacts_source_facts(ledger, monkeypatch):
    db, uid, wid, msg, args, extract = ledger
    finish = receipt_followups.finish_followups
    monkeypatch.setattr(
        receipt_followups,
        "finish_followups",
        AsyncMock(side_effect=RuntimeError("after commit")),
    )
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    mid = (await db.query_raw("SELECT memory_id FROM memory_write_receipts"))[0][
        "memory_id"
    ]
    await db.usermemory.delete(where={"id": mid})
    payload = (await db.query_raw("SELECT extraction FROM memory_extraction_batches"))[
        0
    ]["extraction"]
    assert payload["memories"][0] == {"_erased": True}
    assert payload["memories"][1]["content"] == "用户参加了合成读书活动"
    monkeypatch.setattr(receipt_followups, "finish_followups", finish)
    assert len(await pipeline.process_memory_pipeline(**args)) == 1
    assert extract.await_count == 1


async def test_concurrent_recovery_commits_each_fact_once(ledger):
    db, uid, wid, msg, args, extract = ledger
    results = await asyncio.gather(
        *(pipeline.process_memory_pipeline(**args) for _ in range(2)),
        return_exceptions=True,
    )
    assert any(isinstance(r, list) and len(r) == 2 for r in results)
    assert (
        await db.memorychangelog.count(where={"userId": uid, "operation": "insert"})
        == 2
    )
    assert len(await pipeline.process_memory_pipeline(**args)) == 2


async def test_cancellation_after_commit_resumes_without_duplicating_evidence(
    ledger, monkeypatch
):
    db, uid, wid, msg, args, extract = ledger
    real = receipt_followups.finish_followups
    monkeypatch.setattr(
        receipt_followups,
        "finish_followups",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )
    with pytest.raises(asyncio.CancelledError):
        await pipeline.process_memory_pipeline(**args)
    assert (
        await db.memorychangelog.count(where={"userId": uid, "operation": "insert"})
        == 1
    )
    monkeypatch.setattr(receipt_followups, "finish_followups", real)
    await pipeline.process_memory_pipeline(**args)
    await pipeline.process_memory_pipeline(**args)
    assert (
        await db.memorychangelog.count(
            where={"userId": uid, "operation": "evidence_linked"}
        )
        == 2
    )


@pytest.mark.parametrize("mutation", ["content", "time", "delete"])
async def test_changed_or_deleted_source_cannot_resume_stale_payload(
    ledger, monkeypatch, mutation
):
    from datetime import datetime, timezone

    db, uid, wid, msg, args, extract = ledger
    finish = receipt_followups.finish_followups
    monkeypatch.setattr(
        receipt_followups,
        "finish_followups",
        AsyncMock(side_effect=RuntimeError("after commit")),
    )
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    if mutation == "delete":
        await db.message.delete(where={"id": msg.id})
    else:
        await db.message.update(
            where={"id": msg.id},
            data=(
                {"content": "更正内容"}
                if mutation == "content"
                else {"createdAt": datetime(2020, 1, 1, tzinfo=timezone.utc)}
            ),
        )
    monkeypatch.setattr(receipt_followups, "finish_followups", finish)
    new = await db.message.create(
        data={
            "conversationId": msg.conversationId,
            "role": "user",
            "content": "现在的新事实",
        }
    )
    new_args = dict(
        args, evidence_message_ids=[new.id], new_conversation="user: " + new.content
    )
    if mutation != "delete":
        with pytest.raises(pipeline.MemoryExtractionError):
            await pipeline.process_memory_pipeline(**new_args)
    extract.return_value = {"memories": []}
    assert await pipeline.process_memory_pipeline(**new_args) == []
    assert (
        await db.memorychangelog.count(where={"userId": uid, "operation": "insert"})
        == 1
    )
    blocked = await db.query_raw(
        "SELECT extraction FROM memory_extraction_batches WHERE status='blocked'"
    )
    assert blocked == [{"extraction": None}]


async def test_unknown_source_fact_payload_is_removed_on_erasure(ledger, monkeypatch):
    db, uid, wid, msg, args, extract = ledger
    extract.return_value["memories"][0]["source_indices"] = []
    monkeypatch.setattr(
        receipt_followups,
        "finish_followups",
        AsyncMock(side_effect=RuntimeError("after commit")),
    )
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    mid = (await db.query_raw("SELECT memory_id FROM memory_write_receipts"))[0][
        "memory_id"
    ]
    await db.usermemory.delete(where={"id": mid})
    batch = (await db.query_raw("SELECT extraction FROM memory_extraction_batches"))[0]
    assert "用户今天参观了合成展览" not in str(batch)


async def test_reminder_recovery_never_rearms_fired_or_cancelled_trigger(
    ledger, monkeypatch
):
    from datetime import datetime, timezone

    db, uid, wid, msg, args, extract = ledger
    body = "2028年11月2日15点提醒我喝水"
    await db.message.update(where={"id": msg.id}, data={"content": body})
    args = dict(args, new_conversation="user: " + body)
    extract.return_value = {
        "memories": [
            {
                "content": body,
                "importance": 0.6,
                "main_category": "生活",
                "sub_category": "提醒",
                "source_indices": [1],
                "time_quote": "2028年11月2日15点",
                "time_source_index": 1,
                "recurrence": "once",
            }
        ]
    }
    original = receipt_followups.finish_followups

    async def interrupt_after(*a, **kw):
        await original(*a, **kw)
        raise RuntimeError("completion interrupted")

    monkeypatch.setattr(receipt_followups, "finish_followups", interrupt_after)
    with pytest.raises(pipeline.MemoryExtractionError) as failure:
        await pipeline.process_memory_pipeline(**args)
    assert str(failure.value.__cause__) == "completion interrupted", repr(
        failure.value.__cause__
    )
    triggers = await db.timetrigger.find_many(where={"userId": uid})
    assert len(triggers) == 1, await db.query_raw(
        "SELECT last_error,extraction FROM memory_extraction_batches"
    )
    fired = datetime(2026, 10, 23, tzinfo=timezone.utc)
    await db.timetrigger.update(
        where={"id": triggers[0].id}, data={"isActive": False, "lastFired": fired}
    )
    monkeypatch.setattr(receipt_followups, "finish_followups", original)
    assert len(await pipeline.process_memory_pipeline(**args)) == 1
    after = await db.timetrigger.find_many(where={"userId": uid})
    assert (
        len(after) == 1 and after[0].isActive is False and after[0].lastFired == fired
    )
    await db.timetrigger.delete_many(where={"userId": uid})


async def test_split_recovery_has_one_receipt_per_piece(ledger, monkeypatch):
    db, uid, wid, msg, args, extract = ledger
    extract.return_value["memories"] = extract.return_value["memories"][:1]
    monkeypatch.setattr(
        persistence, "_split_for_storage", lambda *_: ["独立短句一", "独立短句二"]
    )
    original = persistence._audit_write
    calls = 0

    async def fail_second(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            await a[0].execute_raw("SELECT 1/0")
        await original(*a, **kw)

    monkeypatch.setattr(persistence, "_audit_write", fail_second)
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    monkeypatch.setattr(persistence, "_audit_write", original)
    await pipeline.process_memory_pipeline(**args)
    rows = await db.query_raw(
        "SELECT piece_index,followup_done FROM memory_write_receipts ORDER BY piece_index"
    )
    assert rows == [
        {"piece_index": 0, "followup_done": True},
        {"piece_index": 1, "followup_done": True},
    ]
    assert (
        await db.memorychangelog.count(where={"userId": uid, "operation": "insert"})
        == 2
    )
    assert extract.await_count == 1


async def test_lifecycle_change_blocks_old_plan_without_poisoning_new_chat(
    ledger, monkeypatch
):
    from uuid import uuid4
    from app.services.memory.recording.receipt_read import recent_receipts

    db, uid, wid, msg, args, extract = ledger
    finish = receipt_followups.finish_followups
    monkeypatch.setattr(
        receipt_followups,
        "finish_followups",
        AsyncMock(side_effect=RuntimeError("interrupt")),
    )
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    await db.execute_raw(
        "UPDATE conversations SET execution_generation=$1 WHERE id=$2",
        str(uuid4()),
        msg.conversationId,
    )
    monkeypatch.setattr(receipt_followups, "finish_followups", finish)
    new = await db.message.create(
        data={
            "conversationId": msg.conversationId,
            "role": "user",
            "content": "新的独立合成事实",
        }
    )
    extract.return_value = {"memories": []}
    assert (
        await pipeline.process_memory_pipeline(
            **dict(
                args,
                evidence_message_ids=[new.id],
                new_conversation="user: " + new.content,
            )
        )
        == []
    )
    batches = await db.query_raw(
        "SELECT status,extraction FROM memory_extraction_batches ORDER BY prepared_at"
    )
    assert batches == [
        {"status": "blocked", "extraction": None},
        {"status": "completed", "extraction": None},
    ]
    report = await recent_receipts()
    assert report["denominator"] == "this_page_only" and len(report["records"]) == 2
    assert "source_changed" in {r["last_error"] for r in report["records"]}
    assert msg.content not in str(report)


async def test_completed_duplicate_does_not_repeat_mention_or_audit(
    ledger, monkeypatch
):
    db, uid, wid, msg, args, extract = ledger
    existing = await db.usermemory.find_first(where={"userId": uid})
    extract.return_value["memories"] = extract.return_value["memories"][:1]
    monkeypatch.setattr(
        persistence,
        "resolve_memory_write",
        AsyncMock(
            return_value=ReconciliationDecision(
                action="drop_duplicate",
                existing_id=existing.id,
                existing_record=existing,
            )
        ),
    )
    assert await pipeline.process_memory_pipeline(**args) == []
    assert await pipeline.process_memory_pipeline(**args) == []
    assert (
        await db.usermemory.find_unique(where={"id": existing.id})
    ).mentionCount == existing.mentionCount + 1
    assert (
        await db.memorychangelog.count(
            where={"memoryId": existing.id, "operation": "dedup_drop"}
        )
        == 1
    )


async def test_concurrent_physical_erasure_and_writer_conflict_remains_recoverable(
    ledger, monkeypatch
):
    """Exercise actual PG row/epoch lock inversion, not mocked lock failures."""
    db, uid, wid, msg, args, extract = ledger
    finish = receipt_followups.finish_followups
    monkeypatch.setattr(
        receipt_followups,
        "finish_followups",
        AsyncMock(side_effect=RuntimeError("interrupt")),
    )
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    row = (await db.query_raw("SELECT * FROM memory_write_receipts"))[0]
    scope = await execution_scope.bind_conversation_scope(
        actor_user_id=uid, conversation_id=msg.conversationId
    )
    epoch = (
        await db.query_raw(
            "SELECT generation FROM memory_erasure_epochs WHERE workspace_id=$1", wid
        )
    )[0]["generation"]
    token = receipts.WriteToken(row["batch_id"], 0, epoch, scope, "user")
    epoch_locked = asyncio.Event()
    memory_locked = asyncio.Event()

    async def writer():
        async with db.tx() as tx:
            await receipts.claim(tx, token, row["input_hash"])
            epoch_locked.set()
            await asyncio.wait_for(memory_locked.wait(), 3)
            await tx.query_raw(
                "SELECT id FROM memories_user WHERE id=$1 FOR UPDATE", row["memory_id"]
            )

    async def eraser():
        await asyncio.wait_for(epoch_locked.wait(), 3)
        async with db.tx() as tx:
            await tx.query_raw(
                "SELECT id FROM memories_user WHERE id=$1 FOR UPDATE", row["memory_id"]
            )
            memory_locked.set()
            await tx.usermemory.delete(where={"id": row["memory_id"]})

    result = await asyncio.wait_for(
        asyncio.gather(writer(), eraser(), return_exceptions=True), 10
    )
    errors = [str(value) for value in result if isinstance(value, Exception)]
    assert len(errors) == 1 and "deadlock" in errors[0].lower()
    # Whichever transaction PG aborted, reapplying erasure is safe and leaves a
    # tombstone. A subsequent chat finishes unrelated facts without resurrecting.
    await db.usermemory.delete_many(where={"id": row["memory_id"]})
    monkeypatch.setattr(receipt_followups, "finish_followups", finish)
    await pipeline.process_memory_pipeline(**args)
    assert await db.usermemory.find_unique(where={"id": row["memory_id"]}) is None
    assert (
        await db.query_raw(
            "SELECT status,memory_id FROM memory_write_receipts WHERE fact_index=0"
        )
    ) == [{"status": "erased", "memory_id": None}]
    assert (
        await db.usermemory.count(
            where={
                "userId": uid,
                "content": extract.return_value["memories"][0]["content"],
            }
        )
        == 0
    )


async def test_day_only_reminder_recovery_prepares_schedule_once(ledger, monkeypatch):
    from datetime import datetime, timezone
    from app.services.proactive import special_dates
    from app.services.schedule_domain import schedule

    db, uid, wid, msg, args, extract = ledger
    body = "2028年11月2日提醒我查看展览安排"
    await db.message.update(where={"id": msg.id}, data={"content": body})
    extract.return_value = {
        "memories": [
            {
                "content": body,
                "importance": 0.6,
                "main_category": "生活",
                "sub_category": "提醒",
                "source_indices": [1],
                "time_quote": "2028年11月2日",
                "time_source_index": 1,
                "recurrence": "once",
            }
        ]
    }
    lookup = AsyncMock(return_value=[{"synthetic": "schedule"}])
    monkeypatch.setattr(schedule, "get_cached_schedule", lookup)
    expected = datetime(2028, 11, 2, 1, 30, tzinfo=timezone.utc)
    seen = []

    def idle(value, day):
        seen.append((value, day.isoformat()))
        return expected

    monkeypatch.setattr(special_dates, "find_first_idle_after_wakeup", idle)
    args = dict(args, new_conversation="user: " + body)
    ids = await pipeline.process_memory_pipeline(**args)
    assert len(ids) == 1
    trigger = await db.timetrigger.find_first(where={"userId": uid})
    assert trigger.triggerTime == expected and trigger.actionData["memory_id"] == ids[0]
    assert seen == [([{"synthetic": "schedule"}], "2028-11-02")]
    await pipeline.process_memory_pipeline(**args)
    lookup.assert_awaited_once()
    assert await db.timetrigger.count(where={"userId": uid}) == 1
    await db.timetrigger.delete_many(where={"userId": uid})
