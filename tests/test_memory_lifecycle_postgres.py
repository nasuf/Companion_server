"""Real isolated PostgreSQL: rewards, replay, decay, locks and rollback."""
import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.memory.lifecycle import lazy_update as life
from app.services.memory.lifecycle.value import apply_usage, decayed_value
from tests.test_runtime_execution_foundation import flow


@pytest.fixture
async def memory_flow(flow, monkeypatch):
    monkeypatch.setattr(life, "db", flow.db)
    try:
        yield flow
    finally:
        for table in ("memories_user", "memories_ai"):
            await flow.db.execute_raw(f"DELETE FROM memory_embeddings WHERE memory_id IN (SELECT id FROM {table} WHERE user_id=$1)", flow.ids["owner"])
            await flow.db.execute_raw(f"DELETE FROM {table} WHERE user_id=$1", flow.ids["owner"])
        await flow.db.memorychangelog.delete_many(where={"userId": flow.ids["owner"]})


async def memory(f, side="user", **overrides):
    return await getattr(f.db, "aimemory" if side == "ai" else "usermemory").create(data={
        "id": str(uuid4()), "userId": f.ids["owner"], "workspaceId": f.ids["workspace"],
        "content": "Synthetic memory", "importance": 0.5, "currentScore": 0.5,
        "level": 2, "mainCategory": "生活", "subCategory": "工作",
        "valueUpdatedAt": datetime.now(UTC), **overrides,
    })


async def event(f, **overrides):
    return await f.db.message.create(data={"conversationId": f.ids["conversation"],
        "role": "user", "content": "Synthetic input", **overrides})


async def use(f, msg, *, contributed=(), accessed=(), **overrides):
    return await life.record_memory_usage(event_id=msg.id, user_id=f.ids["owner"],
        workspace_id=f.ids["workspace"], contributed_ids=list(contributed), accessed_ids=list(accessed), **overrides)


async def row(f, mid, side="user"):
    return (await f.db.query_raw(f"SELECT * FROM memories_{side} WHERE id=$1", mid))[0]


@pytest.mark.parametrize("side", ["user", "ai"])
async def test_same_message_concurrent_replay_rewards_once(memory_flow, side):
    f=memory_flow;m=await memory(f,side);msg=await event(f)
    results=await asyncio.gather(*(use(f,msg,contributed=[m.id]) for _ in range(8)))
    assert sum(results)==1
    assert (await row(f,m.id,side))["current_score"]==pytest.approx(0.62,abs=1e-5)
    assert (await f.db.query_raw("SELECT count(*)::int n FROM memory_usage_receipts WHERE event_id=$1",msg.id))[0]["n"]==1


async def test_distinct_concurrent_events_do_not_lose_rewards(memory_flow):
    f=memory_flow;m=await memory(f);msgs=[await event(f) for _ in range(3)]
    assert sum(await asyncio.gather(*(use(f,msg,contributed=[m.id]) for msg in msgs)))==3
    r=await row(f,m.id)
    assert r["current_score"]==pytest.approx(0.86,abs=1e-5) and r["level"]==1


async def test_candidate_upgrades_to_contribution_without_double_credit(memory_flow):
    f=memory_flow;m=await memory(f);msg=await event(f)
    assert await use(f,msg,accessed=[m.id])==1
    assert await use(f,msg,contributed=[m.id])==1
    assert await use(f,msg,accessed=[m.id],contributed=[m.id])==0
    assert (await row(f,m.id))["current_score"]==pytest.approx(0.62,abs=1e-5)


@pytest.mark.parametrize("side", ["user", "ai"])
async def test_maintenance_has_no_reward_or_initial_score_reset(memory_flow, side):
    f=memory_flow
    m=await memory(f,side,importance=0.9,currentScore=0.6,valueUpdatedAt=datetime.now(UTC)-timedelta(days=60))
    result=await life.sweep_stale_values(user_id=f.ids["owner"],sources=(side,))
    r=await row(f,m.id,side)
    assert result["scanned"]==1
    assert r["importance"]==0.9
    assert r["current_score"]==pytest.approx(decayed_value(0.6,60),abs=1e-5)
    assert (await life.sweep_stale_values(user_id=f.ids["owner"],sources=(side,)))["scanned"]==0
    assert (await row(f,m.id,side))["current_score"]==r["current_score"]


@pytest.mark.parametrize("kind", ["other_workspace","wrong_owner","old_event","deleted_conversation","archived_memory","reminder"])
async def test_untrusted_or_expired_inputs_cannot_reward(memory_flow, kind):
    f=memory_flow;changes={}
    if kind=="archived_memory":changes["isArchived"]=True
    if kind=="reminder":changes["subCategory"]="提醒"
    if kind=="other_workspace":changes["workspaceId"]=None
    m=await memory(f,**changes)
    msg=await event(f,**({"createdAt":datetime.now(UTC)-timedelta(days=8)} if kind=="old_event" else {}))
    if kind=="deleted_conversation":await f.db.conversation.update(where={"id":f.ids["conversation"]},data={"isDeleted":True})
    count=await life.record_memory_usage(event_id=msg.id,user_id="wrong" if kind=="wrong_owner" else f.ids["owner"],workspace_id=f.ids["workspace"],contributed_ids=[m.id])
    assert count==0 and (await row(f,m.id))["current_score"]==0.5


async def test_receipt_failure_rolls_back_reward_and_level(memory_flow):
    f=memory_flow;m=await memory(f,currentScore=0.8);msg=await event(f)
    function="fail_usage_"+uuid4().hex
    await f.db.execute_raw(f"CREATE FUNCTION {function}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.event_id='{msg.id}' THEN RAISE EXCEPTION 'synthetic receipt failure'; END IF; RETURN NEW; END $$")
    await f.db.execute_raw(f"CREATE TRIGGER {function} BEFORE INSERT ON memory_usage_receipts FOR EACH ROW EXECUTE FUNCTION {function}()")
    try:
        assert await use(f,msg,contributed=[m.id])==0
        r=await row(f,m.id);assert r["current_score"]==0.8 and r["level"]==2
    finally:
        await f.db.execute_raw(f"DROP TRIGGER {function} ON memory_usage_receipts")
        await f.db.execute_raw(f"DROP FUNCTION {function}()")
    assert await use(f,msg,contributed=[m.id])==1


async def test_singleton_promotion_serializes_across_events(memory_flow):
    f=memory_flow
    memories=[await memory(f,mainCategory="身份",subCategory="姓名",currentScore=0.8) for _ in range(2)]
    msgs=[await event(f) for _ in range(2)]
    await asyncio.gather(*(use(f,msg,contributed=[m.id]) for msg,m in zip(msgs,memories)))
    rows=[await row(f,m.id) for m in memories]
    assert sum(r["level"]==1 for r in rows)==1


async def test_oldest_first_bounded_sweep_eventually_visits_every_row(memory_flow):
    f=memory_flow
    memories=[await memory(f,valueUpdatedAt=datetime.now(UTC)-timedelta(days=60+i)) for i in range(6)]
    results=[await life.sweep_stale_values(user_id=f.ids["owner"],limit=2,sources=("user",)) for _ in range(3)]
    assert all(r["scanned"]==2 for r in results)
    rows=[await row(f,m.id) for m in memories]
    assert all(r["current_score"]<0.5 for r in rows)


async def test_initial_clock_null_is_safe_and_receipt_retention_cannot_reopen_old_event(memory_flow):
    f=memory_flow;m=await memory(f,valueUpdatedAt=None);msg=await event(f)
    assert await use(f,msg,contributed=[m.id])==1
    await f.db.execute_raw("UPDATE memory_usage_receipts SET created_at=NOW()-INTERVAL '31 days' WHERE event_id=$1",msg.id)
    await f.db.message.update(where={"id":msg.id},data={"createdAt":datetime.now(UTC)-timedelta(days=31)})
    assert await life.purge_usage_receipts()==1
    assert await use(f,msg,contributed=[m.id])==0


async def test_new_singleton_and_usage_promotion_share_database_lock(memory_flow,monkeypatch):
    from app.services.memory.storage import persistence,repo
    f=memory_flow
    monkeypatch.setattr(persistence,"db",f.db);monkeypatch.setattr(repo,"db",f.db)
    monkeypatch.setattr(repo,"_invalidate_caches",AsyncMock())
    monkeypatch.setattr(persistence,"generate_embedding",AsyncMock(return_value=[0.01]*1024))
    monkeypatch.setattr(persistence,"store_embedding",AsyncMock())
    monkeypatch.setattr(persistence,"resolve_memory_write",AsyncMock(return_value=SimpleNamespace(action="insert")))
    m=await memory(f,"ai",mainCategory="身份",subCategory="姓名",currentScore=0.8)
    msg=await event(f)
    await asyncio.gather(use(f,msg,contributed=[m.id]),persistence.store_memory(
        f.ids["owner"],"My name is Synthetic",source="ai",main_category="身份",sub_category="姓名",
        level=1,importance=0.9,provenance="profile_seed",workspace_id=f.ids["workspace"],
        _singleton_locked=True,_split_done=True))
    assert await f.db.aimemory.count(where={"userId":f.ids["owner"],"workspaceId":f.ids["workspace"],"mainCategory":"身份","subCategory":"姓名","level":1,"isArchived":False})==1
