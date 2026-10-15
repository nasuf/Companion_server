"""Location sources use real PG transactions; external embeddings are deterministic."""

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prisma import Json

from app.services import location_memory as loc, location_ingress as ingress
from app.services.memory import evidence_read
from tests.test_memory_evidence_postgres import origins
from tests.test_location_enhancements import _location_card


@pytest.fixture
async def location_fixture(origins, monkeypatch):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    monkeypatch.setattr(loc, "db", db)
    monkeypatch.setattr(ingress, "db", db)
    monkeypatch.setattr(loc, "generate_embedding", AsyncMock(return_value=[0.1] * 1024))

    async def create(**changes):
        data = {
            "conversationId": convs[0],
            "role": "user",
            "content": "",
            "metadata": Json({"component_card": _location_card()}),
        }
        data.update(changes)
        return await db.message.create(data=data)

    return db, uid, wid, agents, spaces, convs, create


async def test_atomic_both_sides_receipt_and_deleted_result_never_recreated(
    location_fixture,
):
    db, uid, wid, agents, spaces, convs, create = location_fixture
    m = await create()
    await loc.write_location_share_memories({"message_id": m.id})
    receipt = (
        await db.query_raw(
            "SELECT * FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0]
    assert receipt["status"] == "completed" and len(receipt["memory_ids"]) == 2
    for table in ("memories_user", "memories_ai"):
        row = (
            await db.query_raw(
                f"SELECT * FROM {table} WHERE id=ANY($1::text[])", receipt["memory_ids"]
            )
        )[0]
        assert (
            row["main_category"] == "生活"
            and row["level"] == 3
            and row["provenance"] == "user_stated"
        )
        assert "不代表常住地或Agent所在地" in row["content"]
        result = await evidence_read.memory_evidence_detail(
            user_id=uid,
            workspace_id=wid,
            side="user" if table == "memories_user" else "ai",
            memory_id=row["id"],
        )
        assert result["state"] == "linked" and {
            x["source_kind"] for x in result["items"]
        } == {"input", "message"}
        assert "31.2304" not in str(result) and "中山东一路" not in str(result)
    await db.usermemory.delete(where={"id": receipt["memory_ids"][0]})
    await loc.write_location_share_memories({"message_id": m.id})
    assert (
        await db.usermemory.find_unique(where={"id": receipt["memory_ids"][0]}) is None
    )


async def test_second_side_failure_rolls_back_all_and_retry_is_safe(
    location_fixture, monkeypatch
):
    db, *_, create = location_fixture
    m = await create()
    original = loc.store_embedding
    calls = 0

    async def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic vector failure")
        return await original(*args, **kwargs)

    monkeypatch.setattr(loc, "store_embedding", fail_second)
    with pytest.raises(RuntimeError, match="synthetic"):
        await loc.write_location_share_memories({"message_id": m.id})
    assert (
        await db.query_raw(
            "SELECT status FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0]["status"] == "pending_dispatch"
    assert (
        await db.query_raw(
            "SELECT id FROM memory_input_origins WHERE source_status='location_share'"
        )
        == []
    )
    monkeypatch.setattr(loc, "store_embedding", original)
    await loc.write_location_share_memories({"message_id": m.id})
    assert (
        await db.query_raw(
            "SELECT status FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0]["status"] == "completed"


async def test_concurrent_replay_commits_one_pair(location_fixture):
    db, *_, create = location_fixture
    m = await create()
    await asyncio.wait_for(
        asyncio.gather(
            *(loc.write_location_share_memories({"message_id": m.id}) for _ in range(2))
        ),
        15,
    )
    assert (
        len(
            await db.query_raw(
                "SELECT id FROM memory_input_origins WHERE source_status='location_share'"
            )
        )
        == 1
    )
    ids = (
        await db.query_raw(
            "SELECT memory_ids FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0]["memory_ids"]
    assert len(ids) == 2


@pytest.mark.parametrize(
    "change", ["card", "content", "scope", "delete", "takeover", "after_embed"]
)
async def test_source_change_or_deletion_cancels_without_memory(
    location_fixture, monkeypatch, change
):
    db, uid, wid, agents, spaces, convs, create = location_fixture
    m = await create()

    async def alter():
        if change in {"card", "after_embed"}:
            card = _location_card()
            card["payload"]["latitude"] = 1
            await db.message.update(
                where={"id": m.id}, data={"metadata": Json({"component_card": card})}
            )
        elif change == "content":
            await db.message.update(where={"id": m.id}, data={"content": "changed"})
        elif change == "scope":
            await db.chatworkspace.update(
                where={"id": wid}, data={"agentId": agents[1]}
            )
        elif change == "delete":
            await db.message.delete(where={"id": m.id})
        elif change == "takeover":
            await db.message.update(
                where={"id": m.id},
                data={
                    "metadata": Json(
                        {"human_takeover": True, "component_card": _location_card()}
                    )
                },
            )

    if change == "after_embed":

        async def embedding(text):
            await alter()
            return [0.1] * 1024

        monkeypatch.setattr(loc, "generate_embedding", embedding)
    else:
        await alter()
    await loc.write_location_share_memories({"message_id": m.id})
    assert (
        await db.query_raw(
            "SELECT id FROM memory_input_origins WHERE source_status='location_share'"
        )
        == []
    )
    result = await db.query_raw(
        "SELECT status FROM location_memory_events WHERE message_id=$1", m.id
    )
    assert result == ([] if change == "delete" else [{"status": "cancelled"}])


@pytest.mark.parametrize("role,human", [("assistant", False), ("user", True)])
async def test_trigger_ignores_non_user_and_human_chat(location_fixture, role, human):
    db, *_, create = location_fixture
    m = await create(
        role=role,
        metadata=Json({"human_takeover": human, "component_card": _location_card()}),
    )
    assert (
        await db.query_raw(
            "SELECT message_id FROM location_memory_events WHERE message_id=$1", m.id
        )
        == []
    )


async def test_message_and_outbox_rollback_together(location_fixture):
    db, uid, wid, agents, spaces, convs, create = location_fixture
    with pytest.raises(RuntimeError):
        async with db.tx() as tx:
            m = await tx.message.create(
                data={
                    "conversationId": convs[0],
                    "role": "user",
                    "content": "",
                    "metadata": Json({"component_card": _location_card()}),
                }
            )
            assert await tx.query_raw(
                "SELECT message_id FROM location_memory_events WHERE message_id=$1",
                m.id,
            )
            raise RuntimeError("rollback")
    assert (
        await db.query_raw(
            "SELECT message_id FROM location_memory_events WHERE message_id=$1", m.id
        )
        == []
    )


async def test_dispatch_failure_persists_pending_and_bounded_retry(
    location_fixture, monkeypatch
):
    db, *_, create = location_fixture
    m = await create()
    enqueue = AsyncMock(side_effect=ConnectionError("synthetic"))
    monkeypatch.setattr(loc, "enqueue_runtime_job", enqueue)
    assert await loc.dispatch_location_events(limit=10) == 0
    assert enqueue.await_count == 1
    row = (
        await db.query_raw(
            "SELECT * FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0]
    assert (
        row["status"] == "pending_dispatch" and row["last_error"] == "ConnectionError"
    )
    await db.execute_raw(
        "UPDATE location_memory_events SET next_dispatch_at=NOW() WHERE message_id=$1",
        m.id,
    )
    enqueue.side_effect = None
    enqueue.return_value = "synthetic-job"
    assert await loc.dispatch_location_events() == 1
    assert enqueue.call_args.args == ("location_share_memory", {"message_id": m.id})
    assert (
        await db.query_raw(
            "SELECT status,job_id FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0] == {"status": "queued", "job_id": "synthetic-job"}


async def test_admission_retries_have_one_message_and_one_quota_charge(
    location_fixture, monkeypatch
):
    db, uid, wid, agents, spaces, convs, create = location_fixture
    monkeypatch.setattr(ingress.wallet, "is_vip", AsyncMock(return_value=False))
    quota = AsyncMock(return_value={"allowed": True})
    monkeypatch.setattr(ingress, "consume_one_in_transaction", quota)
    request = ingress.location_request(
        text="", card=_location_card(), client_id="retry-location", attachment_ids=[]
    )

    async def accept():
        return await ingress.accept_location_message(
            conversation_id=convs[0],
            user_id=uid,
            request=request,
            content="",
            metadata={"component_card": _location_card()},
            paid_confirmed=False,
        )

    results = await asyncio.gather(accept(), accept())
    assert results[0][0] == results[1][0] and sorted(r[1] for r in results) == [
        False,
        True,
    ]
    assert quota.await_count == 1
    changed = ingress.location_request(
        text="changed",
        card=_location_card(),
        client_id="retry-location",
        attachment_ids=[],
    )
    with pytest.raises(ingress.ChatRequestConflict):
        await ingress.previous_location_message(
            db, conversation_id=convs[0], user_id=uid, request=changed
        )
    monkeypatch.setattr(
        ingress,
        "consume_one_in_transaction",
        AsyncMock(return_value={"allowed": False}),
    )
    request = ingress.location_request(
        text="", card=_location_card(), client_id="blocked-location", attachment_ids=[]
    )
    with pytest.raises(ingress.ChatIngressQuotaBlocked):
        await accept()
    assert len(await db.query_raw("SELECT message_id FROM location_memory_events")) == 1


from tests.test_runtime_job_queue import redis
from app.services.runtime import job_queue as q


async def test_lost_dispatch_response_recovers_same_job_and_dead_letter_retry(
    location_fixture, redis, monkeypatch
):
    db, *_, create = location_fixture
    m = await create()
    q.register_job_handler(loc.JOB_TYPE, loc.write_location_share_memories)
    original = q.enqueue_runtime_job

    async def lost(*args, **kwargs):
        await original(*args, **kwargs)
        raise ConnectionError("synthetic lost response")

    monkeypatch.setattr(loc, "enqueue_runtime_job", lost)
    assert await loc.dispatch_location_events() == 0
    await db.execute_raw(
        "UPDATE location_memory_events SET next_dispatch_at=NOW() WHERE message_id=$1",
        m.id,
    )
    monkeypatch.setattr(loc, "enqueue_runtime_job", original)
    assert await loc.dispatch_location_events() == 1
    jid = (
        await db.query_raw(
            "SELECT job_id FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0]["job_id"]
    assert await redis.llen(q._READY_KEY) == 1
    await redis.hset(q._job_key(jid), "max_attempts", 1)
    monkeypatch.setattr(
        loc,
        "generate_embedding",
        AsyncMock(side_effect=RuntimeError("synthetic provider unavailable")),
    )
    await q.process_runtime_jobs(1)
    assert (await q.inspect_runtime_job(jid))["status"] == "dead_letter"
    assert (
        await db.query_raw(
            "SELECT status FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0]["status"] == "queued"
    monkeypatch.setattr(loc, "generate_embedding", AsyncMock(return_value=[0.1] * 1024))
    await q.retry_runtime_job(jid)
    await q.process_runtime_jobs(1)
    assert (await q.inspect_runtime_job(jid))["status"] == "succeeded"
    receipt = (
        await db.query_raw(
            "SELECT status,memory_ids FROM location_memory_events WHERE message_id=$1",
            m.id,
        )
    )[0]
    assert receipt["status"] == "completed" and len(receipt["memory_ids"]) == 2
    # A consumer may finish its SQL transaction and lose the Redis lease/ACK.
    await loc.write_location_share_memories({"message_id": m.id})
    assert (
        await db.query_raw(
            "SELECT memory_ids FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0]["memory_ids"] == receipt["memory_ids"]


async def test_durable_location_source_and_receipt_reject_forgery(location_fixture):
    from prisma.errors import RawQueryError

    db, uid, wid, agents, spaces, convs, create = location_fixture
    m = await create()
    source = await loc._load_source(db, m.id)
    with pytest.raises(RawQueryError, match="location_event_identity_immutable"):
        await db.execute_raw(
            "UPDATE location_memory_events SET scope='{}' WHERE message_id=$1", m.id
        )
    payload = {
        "format": loc.EXTRACTOR,
        "message_id": m.id,
        "message_version": source["version"],
        "conversation_id": convs[0],
        "content": m.content,
        "component_card": _location_card(),
        "accepted_at": source["created_at"].isoformat(),
    }
    payload["component_card"]["payload"]["latitude"] = 1
    origin = loc.prepare_input_origin(kind="location_share", payload=payload)
    with pytest.raises(RawQueryError, match="location_input_source_mismatch"):
        await loc.persist_input_origin(
            db, user_id=uid, workspace_id=wid, agent_id=agents[0], origin=origin
        )
    await loc.write_location_share_memories({"message_id": m.id})
    with pytest.raises(RawQueryError, match="location_event_identity_immutable"):
        await db.execute_raw(
            "UPDATE location_memory_events SET status='pending_dispatch' WHERE message_id=$1",
            m.id,
        )
    ids = (
        await db.query_raw(
            "SELECT memory_ids FROM location_memory_events WHERE message_id=$1", m.id
        )
    )[0]["memory_ids"]
    detail = await evidence_read.memory_evidence_detail(
        user_id=uid, workspace_id=wid, side="user", memory_id=ids[0]
    )
    item = next(i for i in detail["items"] if i["source_kind"] == "input")
    assert item["input"] == {
        "kind": "location_share",
        "format_version": "memory-input-v1",
    }


async def test_location_retry_with_real_quota_and_attachment_rollback(
    location_fixture, monkeypatch
):
    db, uid, wid, agents, spaces, convs, create = location_fixture
    monkeypatch.setattr(ingress.wallet, "is_vip", AsyncMock(return_value=False))
    request = ingress.location_request(
        text="", card=_location_card(), client_id="real-quota", attachment_ids=[]
    )

    async def accept(metadata):
        return await ingress.accept_location_message(
            conversation_id=convs[0],
            user_id=uid,
            request=request,
            content="",
            metadata=metadata,
            paid_confirmed=False,
        )

    metadata = {"component_card": _location_card()}
    first, second = await asyncio.gather(accept(metadata), accept(metadata))
    assert first[0] == second[0]
    rows = await db.query_raw(
        "SELECT used FROM user_message_quota WHERE user_id=$1", uid
    )
    assert rows == [{"used": 1}]
    request = ingress.location_request(
        text="",
        card=_location_card(),
        client_id="bad-attachment",
        attachment_ids=["absent"],
    )
    with pytest.raises(ingress.ChatIngressResourceInvalid):
        await accept({**metadata, "attachments": [{"id": "absent"}]})
    assert (
        await db.query_raw("SELECT used FROM user_message_quota WHERE user_id=$1", uid)
    ) == rows
    assert len(await db.query_raw("SELECT message_id FROM location_memory_events")) == 1


async def test_takeover_before_consume_cancels_source_and_blocks_new_location(
    location_fixture,
):
    db, uid, wid, agents, spaces, convs, create = location_fixture
    m = await create()
    await db.execute_raw(
        """INSERT INTO human_takeovers(id,user_id,status,owner_admin_id,updated_at)
        VALUES($1,$2,'active',$2,NOW())""",
        str(uuid4()),
        uid,
    )
    await loc.write_location_share_memories({"message_id": m.id})
    assert (
        await db.query_raw(
            "SELECT status FROM location_memory_events WHERE message_id=$1", m.id
        )
    ) == [{"status": "cancelled"}]
    assert (
        await db.query_raw(
            "SELECT id FROM memory_input_origins WHERE source_status='location_share'"
        )
        == []
    )
    newer = await create()
    assert (
        await db.query_raw(
            "SELECT message_id FROM location_memory_events WHERE message_id=$1",
            newer.id,
        )
        == []
    )
    request = ingress.location_request(
        text="", card=_location_card(), client_id="claim-race", attachment_ids=[]
    )
    with pytest.raises(ingress.ChatIngressResourceInvalid):
        await ingress.accept_location_message(
            conversation_id=convs[0],
            user_id=uid,
            request=request,
            content="",
            metadata={"component_card": _location_card()},
            paid_confirmed=False,
        )
