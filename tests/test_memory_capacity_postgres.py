"""Cold-vector retention and atomic archive/restore on isolated PostgreSQL."""
from uuid import uuid4
from unittest.mock import AsyncMock
from contextlib import asynccontextmanager
import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.services.memory.lifecycle import capacity, consolidation
from app.services.memory.storage import embedding
from tests.test_memory_lifecycle_postgres import memory_flow, memory, row
from tests.test_runtime_execution_foundation import flow


async def vector(f, mid):
    await f.db.execute_raw("INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector)",
                           mid, "["+",".join(["0.01"]*1024)+"]")


async def cluster(f, monkeypatch, side="ai"):
    monkeypatch.setattr(capacity,"db",f.db)
    monkeypatch.setattr(consolidation,"db",f.db)
    originals=[await memory(f,side,level=3,provenance="daily_summary") for _ in range(5)]
    digest=await memory(f,side,level=3,provenance="consolidated",isArchived=True)
    for m in originals:await vector(f,m.id)
    return originals,digest


async def archive(f, originals, digest, side="ai"):
    await consolidation._archive_originals(source=side,user_id=f.ids["owner"],workspace_id=f.ids["workspace"],
        originals=[{"id":m.id,"content":m.content} for m in originals],digest_id=digest.id)


@pytest.mark.parametrize("side",["user","ai"])
async def test_prune_keeps_text_then_restore_rebuilds_every_vector(memory_flow,monkeypatch,side):
    f=memory_flow;originals,digest=await cluster(f,monkeypatch,side)
    await archive(f,originals,digest,side)
    assert (await capacity.reclaim_consolidated_embeddings(limit=2))[side]==2
    assert (await capacity.reclaim_consolidated_embeddings())[side]==3
    rows=[await row(f,m.id,side) for m in originals]
    assert all(r["is_archived"] and r["content"]=="Synthetic memory" for r in rows)
    monkeypatch.setattr(embedding,"generate_embedding",AsyncMock(return_value=[0.01]*1024))
    restored=await capacity.restore_consolidated_digest(digest.id)
    assert restored["restored"]==5
    vectors=await f.db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",[m.id for m in originals])
    assert vectors[0]["n"]==5
    assert (await row(f,digest.id,side))["is_archived"]
    assert (await capacity.restore_consolidated_digest(digest.id))["restored"]==0


@pytest.mark.parametrize("failure",["provider","dimension","audit"])
async def test_restore_failure_keeps_originals_archived_and_digest_live(memory_flow,monkeypatch,failure):
    f=memory_flow;originals,digest=await cluster(f,monkeypatch)
    await archive(f,originals,digest);await capacity.reclaim_consolidated_embeddings()
    mock=AsyncMock(side_effect=RuntimeError("provider down")) if failure=="provider" else AsyncMock(return_value=[0.1]*(2 if failure=="dimension" else 1024))
    monkeypatch.setattr(embedding,"generate_embedding",mock)
    name="fail_restore_"+uuid4().hex
    if failure=="audit":
        await f.db.execute_raw(f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.new_value='{digest.id}' AND NEW.operation='consolidation_undone' THEN RAISE EXCEPTION 'synthetic audit failure'; END IF; RETURN NEW; END $$")
        await f.db.execute_raw(f"CREATE TRIGGER {name} BEFORE INSERT ON memory_changelogs FOR EACH ROW EXECUTE FUNCTION {name}()")
    try:
        with pytest.raises(Exception):await capacity.restore_consolidated_digest(digest.id)
        rows=[await row(f,m.id,"ai") for m in originals]
        assert all(r["is_archived"] for r in rows)
        assert not (await row(f,digest.id,"ai"))["is_archived"]
        assert (await f.db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",[m.id for m in originals]))[0]["n"]==0
    finally:
        if failure=="audit":
            await f.db.execute_raw(f"DROP TRIGGER {name} ON memory_changelogs")
            await f.db.execute_raw(f"DROP FUNCTION {name}()")


@pytest.mark.parametrize("change",["scope","content","level"])
async def test_archive_conflict_never_partially_archives_or_publishes(memory_flow,monkeypatch,change):
    f=memory_flow;originals,digest=await cluster(f,monkeypatch)
    data={"workspaceId":None} if change=="scope" else {"content":"changed"} if change=="content" else {"level":2}
    await f.db.aimemory.update(where={"id":originals[-1].id},data=data)
    with pytest.raises(RuntimeError):await archive(f,originals,digest)
    rows=[await row(f,m.id,"ai") for m in originals]
    assert not any(r["is_archived"] for r in rows)
    assert (await row(f,digest.id,"ai"))["is_archived"]
    assert await f.db.memorychangelog.count(where={"userId":f.ids["owner"],"operation":"consolidated_into"})==0


@pytest.mark.parametrize("change",["orphan","seed","wrong_scope","archived_digest"])
async def test_cleanup_excludes_unproven_or_protected_archives(memory_flow,monkeypatch,change):
    f=memory_flow;originals,digest=await cluster(f,monkeypatch)
    await archive(f,originals,digest)
    if change=="orphan":await f.db.memorychangelog.delete_many(where={"userId":f.ids["owner"]})
    elif change=="seed":await f.db.aimemory.update_many(where={"id":{"in":[m.id for m in originals]}},data={"provenance":"profile_seed"})
    elif change=="wrong_scope":await f.db.aimemory.update(where={"id":digest.id},data={"workspaceId":None})
    else:await f.db.aimemory.update(where={"id":digest.id},data={"isArchived":True})
    assert (await capacity.reclaim_consolidated_embeddings())["ai"]==0


async def test_cleanup_cannot_delete_vectors_during_restore_commit(memory_flow,monkeypatch):
    f=memory_flow;originals,digest=await cluster(f,monkeypatch)
    await archive(f,originals,digest);await capacity.reclaim_consolidated_embeddings()
    monkeypatch.setattr(embedding,"generate_embedding",AsyncMock(return_value=[0.01]*1024))
    entered,released=asyncio.Event(),asyncio.Event()
    class PausedTransaction:
        def __init__(self,tx):self.tx=tx
        def __getattr__(self,key):return getattr(self.tx,key)
        async def execute_raw(self,sql,*args):
            if sql.startswith("INSERT INTO memory_embeddings") and not entered.is_set():
                entered.set();await released.wait()
            return await self.tx.execute_raw(sql,*args)
    class PausedDatabase:
        def __getattr__(self,key):return getattr(f.db,key)
        @asynccontextmanager
        async def tx(self,**kwargs):
            async with f.db.tx(**kwargs) as tx:yield PausedTransaction(tx)
    monkeypatch.setattr(capacity,"db",PausedDatabase())
    restoring=asyncio.create_task(capacity.restore_consolidated_digest(digest.id))
    try:
        await asyncio.wait_for(entered.wait(),5)
        assert (await asyncio.wait_for(capacity.reclaim_consolidated_embeddings(),5))["ai"]==0
    finally:released.set()
    assert (await restoring)["restored"]==5
    assert (await f.db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",[m.id for m in originals]))[0]["n"]==5


async def test_archive_write_failure_rolls_back_audit_and_every_original(memory_flow,monkeypatch):
    f=memory_flow;originals,digest=await cluster(f,monkeypatch)
    name="fail_archive_"+uuid4().hex
    await f.db.execute_raw(f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.id='{originals[-1].id}' AND NEW.is_archived THEN RAISE EXCEPTION 'synthetic archive failure'; END IF; RETURN NEW; END $$")
    await f.db.execute_raw(f"CREATE TRIGGER {name} BEFORE UPDATE ON memories_ai FOR EACH ROW EXECUTE FUNCTION {name}()")
    try:
        with pytest.raises(Exception):await archive(f,originals,digest)
        rows=[await row(f,m.id,"ai") for m in originals]
        assert not any(r["is_archived"] for r in rows)
        assert (await row(f,digest.id,"ai"))["is_archived"]
        assert await f.db.memorychangelog.count(where={"userId":f.ids["owner"],"operation":"consolidated_into"})==0
    finally:
        await f.db.execute_raw(f"DROP TRIGGER {name} ON memories_ai")
        await f.db.execute_raw(f"DROP FUNCTION {name}()")


async def test_failed_unpublished_digest_is_removed_but_successful_commit_is_preserved(memory_flow,monkeypatch):
    f=memory_flow;originals,digest=await cluster(f,monkeypatch)
    await vector(f,digest.id)
    await consolidation._rollback_digest(source="ai",digest_id=digest.id)
    assert await f.db.aimemory.find_unique(where={"id":digest.id}) is None
    assert not await f.db.query_raw("SELECT memory_id FROM memory_embeddings WHERE memory_id=$1",digest.id)
    originals,published=await cluster(f,monkeypatch)
    await archive(f,originals,published)
    await consolidation._rollback_digest(source="ai",digest_id=published.id)
    assert not (await row(f,published.id,"ai"))["is_archived"]


async def test_abandoned_output_cleanup_preserves_original_text_and_recent_stages(memory_flow,monkeypatch):
    f=memory_flow;monkeypatch.setattr(capacity,"db",f.db)
    abandoned=await memory(f,"ai",level=3,provenance="consolidated",isArchived=True,createdAt=datetime.now(UTC)-timedelta(days=2))
    recent=await memory(f,"ai",level=3,provenance="consolidated",isArchived=True)
    original=await memory(f,"ai",level=3,provenance="daily_summary",isArchived=True,createdAt=datetime.now(UTC)-timedelta(days=60))
    await vector(f,abandoned.id);await vector(f,recent.id);await vector(f,original.id)
    result=await capacity.reclaim_consolidated_embeddings()
    assert result["ai_abandoned_digests"]==1
    assert await f.db.aimemory.find_unique(where={"id":abandoned.id}) is None
    assert await f.db.aimemory.find_unique(where={"id":recent.id}) is not None
    assert (await row(f,original.id,"ai"))["content"]==original.content
