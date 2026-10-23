"""Atomic row/vector/origin/audit writes on isolated, real PostgreSQL.

Models are synthetic; SQL failures, rollbacks, concurrent writes and cancellation
are real. This is not a claim that the whole extraction batch is replay safe.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.services.memory import evidence
from app.services.memory.manual_write import MemoryWriteError
from app.services.memory.storage import persistence as p, repo, embedding
from app.services.memory.storage.reconciliation import ReconciliationDecision
from tests.test_memory_evidence_postgres import origins


@pytest.fixture
async def storage(origins, monkeypatch):
    db, uid, wid, shared, msg, *_ = origins
    # The provenance fixture deliberately creates an ambiguous legacy ID. These
    # tests target ordinary unambiguous records; ambiguity is tested separately.
    await db.aimemory.delete(where={"id": shared})
    monkeypatch.setattr(p, "db", db)
    monkeypatch.setattr(embedding, "db", db)
    monkeypatch.setattr(p, "generate_embedding", AsyncMock(return_value=[0., 1.] + [0.] * 1022))
    monkeypatch.setattr(p, "after_memory_changelog", AsyncMock())
    monkeypatch.setattr(p, "_generated_self_memory_allowed", AsyncMock(return_value=True))
    monkeypatch.setattr(p, "resolve_memory_write", AsyncMock(return_value=ReconciliationDecision(action="insert_new")))
    origin = evidence.EvidenceSource("message", msg.id, expected_version=evidence.content_version(msg.content))
    args = dict(user_id=uid, workspace_id=wid, content="合成用户参观了展览", level=2,
                main_category="生活", sub_category="生活", evidence_sources=(origin,))
    return db, uid, wid, shared, args


async def durable_snapshot(db, uid):
    return await db.query_raw("""SELECT
        (SELECT count(*)::int FROM memories_user WHERE user_id=$1) AS users,
        (SELECT count(*)::int FROM memories_ai WHERE user_id=$1) AS agents,
        (SELECT count(*)::int FROM memory_evidence_links WHERE user_id=$1) AS origins,
        (SELECT count(*)::int FROM memory_changelogs WHERE user_id=$1) AS audits,
        (SELECT count(*)::int FROM memory_embeddings WHERE memory_id IN
           (SELECT id FROM memories_user WHERE user_id=$1 UNION SELECT id FROM memories_ai WHERE user_id=$1)) AS vectors""", uid)


@pytest.mark.parametrize("side", ["user", "ai"])
@pytest.mark.parametrize("linked", [False, True])
async def test_all_durable_parts_commit_before_hooks(storage, side, linked):
    db, uid, wid, _, args = storage
    args["source"] = side
    if not linked:
        args["evidence_sources"] = None
    mid = await p.store_memory(**args)
    model = db.aimemory if side == "ai" else db.usermemory
    assert (await model.find_unique(where={"id": mid})).content == args["content"]
    assert await db.query_raw("SELECT memory_id FROM memory_embeddings WHERE memory_id=$1", mid)
    rows = await db.memorychangelog.find_many(where={"memoryId": mid})
    assert [row.operation for row in rows] == ["insert"]
    links = await db.query_raw("SELECT id FROM memory_evidence_links WHERE memory_id=$1", mid)
    assert len(links) == int(linked)
    p.after_memory_changelog.assert_awaited_once()


@pytest.mark.parametrize("side", ["user", "ai"])
@pytest.mark.parametrize("stage", ["vector", "audit", "source"])
async def test_new_write_failure_rolls_back_every_part(storage, monkeypatch, side, stage):
    db, uid, _, _, args = storage
    args["source"] = side
    before = await durable_snapshot(db, uid)
    if stage == "source":
        args["evidence_sources"] = (evidence.EvidenceSource("message", "missing-source"),)
    elif stage == "vector":
        async def fail(mid, vec, *, database=None):
            assert database is not None
            await embedding.store_embedding(mid, vec, database=database)
            raise RuntimeError("synthetic vector response failure")
        monkeypatch.setattr(p, "store_embedding", fail)
    else:
        original = p._audit_write
        async def fail(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("synthetic audit failure")
        monkeypatch.setattr(p, "_audit_write", fail)
    delete = AsyncMock(side_effect=AssertionError("no compensating deletion"))
    monkeypatch.setattr(repo, "delete", delete)
    with pytest.raises((RuntimeError, ValueError)):
        await p.store_memory(**args)
    assert await durable_snapshot(db, uid) == before
    delete.assert_not_awaited()
    p.after_memory_changelog.assert_not_awaited()


async def test_cancellation_after_vector_rolls_back_transaction(storage, monkeypatch):
    db, uid, _, _, args = storage
    before = await durable_snapshot(db, uid)
    ready = asyncio.Event()
    async def pause(mid, vec, *, database=None):
        await embedding.store_embedding(mid, vec, database=database)
        ready.set()
        await asyncio.Future()
    monkeypatch.setattr(p, "store_embedding", pause)
    task = asyncio.create_task(p.store_memory(**args))
    await asyncio.wait_for(ready.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await durable_snapshot(db, uid) == before
    p.after_memory_changelog.assert_not_awaited()


@pytest.mark.parametrize("action", ["update_existing", "merge_existing"])
@pytest.mark.parametrize("linked", [False, True])
async def test_update_failure_preserves_old_content_vector_origins_and_audit(storage, monkeypatch, action, linked):
    db, uid, _, mid, args = storage
    record = await repo.find_unique(mid)
    await embedding.store_embedding(mid, [1.] + [0.] * 1023, database=db)
    decision = ReconciliationDecision(action=action, existing_id=mid, existing_record=record,
                                      merged_content="合成完整的新记忆")
    monkeypatch.setattr(p, "resolve_memory_write", AsyncMock(return_value=decision))
    if not linked:
        args["evidence_sources"] = None
    before = await durable_snapshot(db, uid)
    original = p.store_embedding
    async def fail(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("injected update failure")
    monkeypatch.setattr(p, "store_embedding", fail)
    with pytest.raises(RuntimeError, match="injected"):
        await p.store_memory(**args)
    assert (await db.usermemory.find_unique(where={"id": mid})).content == record.content
    assert (await embedding.get_embedding(mid))[0] == 1.
    assert await durable_snapshot(db, uid) == before
    monkeypatch.setattr(p, "store_embedding", original)
    assert await p.store_memory(**args) == mid
    assert (await db.usermemory.find_unique(where={"id": mid})).content == "合成完整的新记忆"
    assert (await db.memorychangelog.find_many(where={"memoryId": mid}))[0].operation == (
        "reconciliation_update" if action == "update_existing" else "reconciliation_merge")


@pytest.mark.parametrize("mutation", ["edit", "archive", "delete"])
async def test_changed_target_is_never_overwritten_or_resurrected(storage, monkeypatch, mutation):
    db, uid, _, mid, args = storage
    record = await repo.find_unique(mid)
    async def prepare(**kwargs):
        if mutation == "delete":
            await db.usermemory.delete(where={"id": mid})
        else:
            await db.usermemory.update(where={"id": mid}, data=(
                {"content": "并发人工修正"} if mutation == "edit" else {"isArchived": True}))
        return ReconciliationDecision(action="update_existing", existing_id=mid, existing_record=record)
    monkeypatch.setattr(p, "resolve_memory_write", prepare)
    with pytest.raises(MemoryWriteError):
        await p.store_memory(**args)
    row = await db.usermemory.find_unique(where={"id": mid})
    if mutation == "delete":
        assert row is None
    elif mutation == "edit":
        assert row.content == "并发人工修正"
    else:
        assert row.isArchived
    assert await db.memorychangelog.count(where={"userId": uid}) == 0


async def test_two_prepared_updates_only_one_commits(storage, monkeypatch):
    db, uid, _, mid, args = storage
    record = await repo.find_unique(mid)
    monkeypatch.setattr(p, "resolve_memory_write", AsyncMock(return_value=ReconciliationDecision(
        action="update_existing", existing_id=mid, existing_record=record)))
    results = await asyncio.gather(p.store_memory(**args), p.store_memory(**dict(args, content="另一份合成内容")),
                                   return_exceptions=True)
    assert results.count(mid) == 1
    assert sum(isinstance(r, MemoryWriteError) for r in results) == 1
    assert await db.memorychangelog.count(where={"userId": uid}) == 1


async def test_name_replacement_audit_failure_keeps_old_name(storage, monkeypatch):
    db, uid, _, mid, args = storage
    await db.usermemory.update(where={"id": mid}, data={"content": "用户姓名为旧名字", "level": 1,
        "mainCategory": "身份", "subCategory": "姓名"})
    args.update(content="用户姓名为新名字", level=1, main_category="身份", sub_category="姓名", _singleton_locked=True)
    before = await durable_snapshot(db, uid)
    monkeypatch.setattr(p, "_audit_write", AsyncMock(side_effect=RuntimeError("audit down")))
    with pytest.raises(RuntimeError, match="audit down"):
        await p.store_memory(**args)
    assert not (await db.usermemory.find_unique(where={"id": mid})).isArchived
    assert await durable_snapshot(db, uid) == before


async def test_ambiguous_legacy_id_rejected_without_changing_either_owner(origins, monkeypatch):
    db, uid, wid, mid, *_ = origins
    record = await repo.find_unique(mid)
    monkeypatch.setattr(p, "db", db)
    monkeypatch.setattr(p, "generate_embedding", AsyncMock(return_value=[1.] + [0.] * 1023))
    monkeypatch.setattr(p, "resolve_memory_write", AsyncMock(return_value=ReconciliationDecision(
        action="update_existing", existing_id=mid, existing_record=record)))
    before = await durable_snapshot(db, uid)
    with pytest.raises(MemoryWriteError, match="memory_id_is_ambiguous"):
        await p.store_memory(uid, "合成修改", workspace_id=wid, level=2,
                             main_category="生活", sub_category="生活")
    assert await durable_snapshot(db, uid) == before
    assert (await db.usermemory.find_unique(where={"id": mid})).content == record.content


async def test_real_sql_audit_error_rolls_back_memory_vector_and_evidence(storage, monkeypatch):
    from prisma.errors import RawQueryError
    db, uid, _, _, args = storage
    before = await durable_snapshot(db, uid)
    async def fail(database, *args, **kwargs):
        await database.execute_raw("SELECT 1 / 0")
    monkeypatch.setattr(p, "_audit_write", fail)
    with pytest.raises(RawQueryError):
        await p.store_memory(**args)
    assert await durable_snapshot(db, uid) == before


@pytest.mark.parametrize("field,value", [("userId", "other-user"), ("workspaceId", "other-workspace"), ("source", "ai")])
async def test_prepared_target_from_another_scope_cannot_be_written(storage, monkeypatch, field, value):
    from dataclasses import replace
    db, uid, _, mid, args = storage
    record = await repo.find_unique(mid)
    invalid = replace(record, **{field: value})
    monkeypatch.setattr(p, "resolve_memory_write", AsyncMock(return_value=ReconciliationDecision(
        action="update_existing", existing_id=mid, existing_record=invalid)))
    before = await durable_snapshot(db, uid)
    with pytest.raises(MemoryWriteError, match="memory_write_scope_changed"):
        await p.store_memory(**args)
    assert await durable_snapshot(db, uid) == before
    assert await repo.find_unique(mid) == record
