"""Real migrated loopback SQL and synthetic scopes; no production URL fallback."""

import asyncio
from dataclasses import replace
import json
import os
import signal
import sys
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from prisma import Json, Prisma

from tests.test_runtime_execution_foundation import flow, RuntimeDatabase
from tests.test_chat_ingress_postgres import accept, snapshot
from tests.test_sql_job_postgres import (
    queue,
    expired_lease,
    second_scope,
    registry,
    _PROCESS_WORKER,
    state,
)
from app.services.runtime.execution_scope import (
    ExecutionScopeUnavailable,
    bind_conversation_scope,
)
from app.services.runtime.sql_job_contracts import (
    LeaseLost,
    LeasePolicy,
    WorkerStopRequired,
)
from app.services.runtime.reply_outbox import PreparedReply, prepare_chat_result
from app.services.runtime.sql_outbox import SqlOutbox


async def publish(flow, *, key="one", count=2):
    accepted = await accept(flow, key=key)
    q = queue(flow)
    claim = await q.claim_next("chat-worker")
    prepared = prepare_chat_result(
        claim,
        tuple(
            PreparedReply.capture(
                f"synthetic-{i}", sticker_url="https://synthetic.invalid/sticker"
            )
            for i in range(count)
        ),
        done={"total": count},
    )
    assert await q.finish(claim, prepared.result, commit=prepared.commit)
    return accepted, prepared


async def events(flow, run_id):
    return await flow.db.query_raw(
        "SELECT * FROM runtime_outbox WHERE run_id=$1 ORDER BY sequence", run_id
    )


async def drain(flow):
    store = SqlOutbox(flow.db)
    output = []
    for _ in range(20):
        claim = await store.claim_next("replay", scope=flow.bound, reconnect=True)
        if claim is None:
            break
        output.append(claim.envelope())
        assert await store.acknowledge(flow.bound, claim.id, str(claim.token))
    return output


async def test_reply_events_and_result_commit_atomically(flow):
    accepted, prepared = await publish(flow)
    rows = await events(flow, accepted.run_id)
    assert [r["event_type"] for r in rows] == ["reply", "reply", "done"]
    assert [r["sequence"] for r in rows] == [0, 1, 2]
    assert [r["id"] for r in rows] == prepared.result["event_ids"]
    assert rows[-1]["message_id"] is None
    messages = await flow.db.message.find_many(
        where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
    )
    assert {m.id for m in messages} == set(prepared.result["message_ids"])
    delivered = await drain(flow)
    assert [e["type"] for e in delivered] == ["reply", "reply", "done"]
    assert [e["data"].get("message_id") for e in delivered[:2]] == prepared.result[
        "message_ids"
    ]
    assert await SqlOutbox(flow.db).has_pending(flow.bound) is False


async def test_event_insert_failure_rolls_back_all_messages_and_completion(flow):
    accepted = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    p = prepare_chat_result(
        claim, (PreparedReply.capture("a"), PreparedReply.capture("b"))
    )

    async def fail(tx):
        await p.commit(tx)
        await tx.execute_raw("SELECT 1/0")

    with pytest.raises(Exception):
        await q.finish(claim, p.result, commit=fail)
    assert await events(flow, accepted.run_id) == []
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 0
    )
    assert (
        await flow.db.agentrun.find_unique(where={"id": accepted.run_id})
    ).status == "running"
    assert await q.finish(claim, p.result, commit=p.commit)


async def test_expired_preparation_cannot_publish(flow):
    accepted = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    p = prepare_chat_result(claim, (PreparedReply.capture("a"),))
    await expired_lease(flow, claim)
    with pytest.raises(LeaseLost):
        await q.finish(claim, p.result, commit=p.commit)
    assert await events(flow, accepted.run_id) == []
    fresh = await q.claim_next("fresh")
    p2 = prepare_chat_result(fresh, (PreparedReply.capture("a"),))
    assert p2.result == p.result
    assert await q.finish(fresh, p2.result, commit=p2.commit)


async def test_competing_deliverers_only_one_claim(flow):
    await publish(flow)
    other = RuntimeDatabase(
        Prisma(datasource={"url": flow.url}, http={"trust_env": False})
    )
    await other.connect()
    try:
        claims = await asyncio.gather(
            *(
                SqlOutbox(other if i % 2 else flow.db).claim_next("w" + str(i))
                for i in range(8)
            )
        )
        assert sum(c is not None for c in claims) == 1
    finally:
        await other.disconnect()


async def test_sent_without_ack_is_not_delivered_and_reconnect_bypasses_backoff(flow):
    accepted, _ = await publish(flow)
    s = SqlOutbox(flow.db)
    send = AsyncMock()
    assert await s.deliver_once("sender", send)
    rows = await events(flow, accepted.run_id)
    assert rows[0]["status"] == "pending" and rows[0]["delivered_at"] is None
    assert await s.claim_next("poll") is None
    replay = await s.claim_next("resume", scope=flow.bound, reconnect=True)
    assert replay.id == rows[0]["id"] and replay.token == 2
    assert not await s.acknowledge(flow.bound, replay.id, "1")
    assert await s.acknowledge(flow.bound, replay.id, "2")
    assert await s.acknowledge(
        flow.bound, replay.id, "1"
    )  # Terminal read, no mutation.


async def test_expired_delivery_takeover_rejects_stale_sender(flow):
    accepted, _ = await publish(flow)
    s = SqlOutbox(flow.db)
    old = await s.claim_next("same-worker")
    await flow.db.execute_raw(
        "UPDATE runtime_outbox SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
        old.id,
    )
    fresh = await s.claim_next("same-worker")
    assert fresh.id == old.id and fresh.token == old.token + 1
    with pytest.raises(LeaseLost):
        await s.retry(old)
    assert not await s.acknowledge(flow.bound, old.id, str(old.token))
    assert await s.acknowledge(flow.bound, fresh.id, str(fresh.token))


async def test_client_ack_before_send_returns_does_not_mutate_terminal(flow):
    accepted, _ = await publish(flow)
    s = SqlOutbox(flow.db)

    async def send(conv, envelope):
        assert conv == flow.bound.conversation_id
        assert await s.acknowledge(
            flow.bound, envelope["event_id"], envelope["delivery_token"]
        )

    assert await s.deliver_once("w", send)
    assert (await events(flow, accepted.run_id))[0]["status"] == "delivered"


@pytest.mark.parametrize("mode", ["error", "timeout"])
async def test_send_failure_keeps_event_and_sanitizes_error(flow, mode):
    accepted, _ = await publish(flow)
    s = SqlOutbox(flow.db)

    async def send(*args):
        if mode == "error":
            raise RuntimeError("synthetic secret must not leak")
        await asyncio.sleep(10)

    assert await s.deliver_once("w", send, timeout_seconds=0.02)
    row = (await events(flow, accepted.run_id))[0]
    assert row["status"] == "pending" and row["error"] == {"code": "send_unavailable"}
    assert len(await drain(flow)) == 3


async def test_cancellation_retains_lease_until_expiry(flow):
    accepted, _ = await publish(flow)
    s = SqlOutbox(flow.db)
    entered = asyncio.Event()

    async def send(*args):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(s.deliver_once("w", send))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await events(flow, accepted.run_id))[0]["status"] == "delivering"
    assert await s.claim_next("replacement", scope=flow.bound, reconnect=True) is None


async def test_cross_conversation_ack_and_replay_are_denied(flow):
    accepted, _ = await publish(flow)
    s = SqlOutbox(flow.db)
    claim = await s.claim_next("w")
    other = await second_scope(flow)
    try:
        assert not await s.acknowledge(other, claim.id, str(claim.token))
        assert await s.claim_next("other", scope=other, reconnect=True) is None
        assert (await events(flow, accepted.run_id))[0]["status"] == "delivering"
    finally:
        await flow.db.conversation.update(
            where={"id": other.conversation_id},
            data={
                "userId": flow.bound.owner_user_id,
                "agentId": flow.bound.agent_id,
                "workspaceId": None,
            },
        )
        await flow.db.chatworkspace.delete(where={"id": other.workspace_id})
        await flow.db.aiagent.delete(where={"id": other.agent_id})
        await flow.db.user.delete(where={"id": other.owner_user_id})


async def test_lifecycle_invalidates_ack_and_cancels_old_delivery(flow):
    accepted, _ = await publish(flow)
    s = SqlOutbox(flow.db)
    c = await s.claim_next("w")
    await flow.db.chatworkspace.update(
        where={"id": flow.bound.workspace_id}, data={"status": "archived"}
    )
    with pytest.raises(ExecutionScopeUnavailable):
        await s.acknowledge(flow.bound, c.id, str(c.token))
    assert await s.claim_next("cleanup") is None
    # In-flight send cannot be unsent; expired claims are retired by the scanner.
    await flow.db.execute_raw(
        "UPDATE runtime_outbox SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
        c.id,
    )
    assert await s.claim_next("cleanup") is None
    assert {r["status"] for r in await events(flow, accepted.run_id)} == {"cancelled"}


async def test_new_generation_cannot_replay_old_events(flow):
    accepted, _ = await publish(flow)
    await flow.db.execute_raw(
        "UPDATE conversations SET execution_generation=gen_random_uuid()::text WHERE id=$1",
        flow.bound.conversation_id,
    )
    fresh = await bind_conversation_scope(
        actor_user_id=flow.bound.actor_user_id,
        conversation_id=flow.bound.conversation_id,
        database=flow.db,
    )
    assert (
        await SqlOutbox(flow.db).claim_next("new", scope=fresh, reconnect=True) is None
    )
    assert {r["status"] for r in await events(flow, accepted.run_id)} == {"cancelled"}


async def test_batch_order_and_same_event_after_lost_ack(flow):
    a, _ = await publish(flow, key="a")
    b, _ = await publish(flow, key="b")
    s = SqlOutbox(flow.db)
    c = await s.claim_next("w")
    assert c.run_id == a.run_id
    await s.retry(c)
    assert await s.claim_next("w") is None  # Later batch cannot overtake its head.
    output = await drain(flow)
    assert [x["run_id"] for x in output] == [a.run_id] * 3 + [b.run_id] * 3


@pytest.mark.parametrize(
    "token", [None, True, 1, "", "0", "01", "-1", "1.0", "١", "9223372036854775808"]
)
async def test_bad_ack_tokens_rejected_before_mutation(flow, token):
    with pytest.raises(ValueError):
        await SqlOutbox(flow.db).acknowledge(flow.bound, "synthetic", token)


@pytest.mark.parametrize(
    "case",
    ["empty", "markers", "data_identity", "not_tuple", "oversize", "done_identity"],
)
async def test_prepared_output_rejects_bad_contract(flow, case):
    await accept(flow)
    claim = await queue(flow).claim_next("w")
    with pytest.raises(ValueError):
        if case == "empty":
            PreparedReply.capture("")
        elif case == "markers":
            PreparedReply.capture("[EMO:高兴/50]")
        elif case == "data_identity":
            PreparedReply.capture("hello", message_id="forged")
        elif case == "not_tuple":
            prepare_chat_result(claim, [PreparedReply.capture("hello")])
        elif case == "oversize":
            prepare_chat_result(
                claim, tuple(PreparedReply.capture("x" * 32000) for _ in range(10))
            )
        else:
            prepare_chat_result(
                claim, (PreparedReply.capture("hello"),), done={"event_id": "forged"}
            )


@pytest.mark.skipif(
    sys.platform == "win32", reason="Owned POSIX process group required"
)
@pytest.mark.parametrize(
    "marker",
    [
        "INSERT INTO messages",
        "INSERT INTO runtime_outbox",
        "UPDATE runtime_jobs SET status='succeeded'",
        "RESULT_COMMITTED",
    ],
)
async def test_real_worker_kill_reply_outbox_atomic_recovery(flow, marker):
    graph = "outbox-crash-" + str(uuid4())
    a = await accept(flow, execution=snapshot(graph_version=graph))
    source = _PROCESS_WORKER.replace(
        "from app.services.runtime.sql_job_queue import SqlJobQueue",
        "from app.services.runtime.sql_job_queue import SqlJobQueue\nfrom app.services.runtime.reply_outbox import PreparedReply,prepare_chat_result",
    )
    source = source.replace(
        "assert await q.finish(claim,{'reply':'prepared'},commit=commit)",
        "prepared=prepare_chat_result(claim,(PreparedReply.capture('prepared'),))\n  assert await q.finish(claim,prepared.result,commit=prepared.commit)",
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        source,
        json.dumps([flow.url, a.run_id, flow.bound.conversation_id, graph, marker]),
        env={**os.environ, "APP_ENV": "test", "PYTHON_DOTENV_DISABLED": "1"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:

        async def reached():
            while True:
                line = await process.stdout.readline()
                if line == b"CRASH_WINDOW\n":
                    return
                if not line:
                    raise AssertionError((await process.stderr.read()).decode()[-1600:])

        await asyncio.wait_for(reached(), 60)
        assert os.getpgid(process.pid) == process.pid
        os.killpg(process.pid, signal.SIGKILL)
        assert await asyncio.wait_for(process.wait(), 10) == -signal.SIGKILL
    finally:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
    q = queue(flow, spec=registry(versions=((graph, 1),)))
    if marker != "RESULT_COMMITTED":
        assert await events(flow, a.run_id) == []
        assert (
            await flow.db.message.count(
                where={
                    "conversationId": flow.bound.conversation_id,
                    "role": "assistant",
                }
            )
            == 0
        )
        await flow.db.execute_raw(
            "UPDATE runtime_jobs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
            a.job_id,
        )
        claim = await q.claim_next("recovery")
        prepared = prepare_chat_result(claim, (PreparedReply.capture("prepared"),))
        assert await q.finish(claim, prepared.result, commit=prepared.commit)
    else:
        assert (await state(flow, a.run_id))["run_status"] == "succeeded"
        assert await q.claim_next("recovery") is None
    assert len(await events(flow, a.run_id)) == 2
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 1
    )
    assert [e["type"] for e in await drain(flow)] == ["reply", "done"]


async def test_delivery_refusing_cancellation_requires_worker_stop(flow):
    accepted, _ = await publish(flow)
    s = SqlOutbox(flow.db, policy=LeasePolicy(cancellation_seconds=0.02))
    release = asyncio.Event()

    async def send(*args):
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    try:
        with pytest.raises(WorkerStopRequired):
            await s.deliver_once("w", send, timeout_seconds=0.02)
        assert (await events(flow, accepted.run_id))[0]["status"] == "delivering"
        assert await s.claim_next("replacement") is None
    finally:
        release.set()
        await asyncio.sleep(0.02)


async def test_committed_result_with_lost_ack_is_not_published_twice(flow):
    from tests.test_chat_ingress_postgres import FaultDatabase

    a = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    prepared = prepare_chat_result(claim, (PreparedReply.capture("committed"),))
    with pytest.raises(ConnectionError):
        await queue(flow, database=FaultDatabase(flow.db, commit_unknown=True)).finish(
            claim, prepared.result, commit=prepared.commit
        )
    assert await q.claim_next("recovery") is None
    assert (await state(flow, a.run_id))["run_status"] == "succeeded"
    assert len(await events(flow, a.run_id)) == 2
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 1
    )
    assert len(await drain(flow)) == 2


async def test_message_order_matches_reply_sequence_in_history(flow):
    _, prepared = await publish(flow, count=4)
    messages = await flow.db.message.find_many(
        where={"conversationId": flow.bound.conversation_id, "role": "assistant"},
        order={"createdAt": "asc"},
    )
    assert [m.id for m in messages] == prepared.result["message_ids"]


async def test_replay_frame_and_ack_drain_real_committed_sql(flow, monkeypatch):
    from app.services.runtime import outbox_realtime as api

    monkeypatch.setattr(api, "db", flow.db)
    original = api.bind_conversation_scope

    async def bound(**kwargs):
        return await original(**kwargs, database=flow.db)

    monkeypatch.setattr(api, "bind_conversation_scope", bound)
    a, _ = await publish(flow)
    delivered = []

    class Socket:
        async def send_json(self, envelope):
            delivered.append(envelope)

        async def close(self, **kwargs):
            raise AssertionError(kwargs)

    for _ in range(3):
        await api.handle_delivery_frame(
            Socket(),
            flow.bound.actor_user_id,
            flow.bound.conversation_id,
            {"type": "delivery_resume"},
        )
        reply = next(x for x in reversed(delivered) if "event_id" in x)
        await api.handle_delivery_frame(
            Socket(),
            flow.bound.actor_user_id,
            flow.bound.conversation_id,
            {
                "type": "delivery_ack",
                "data": {
                    "event_id": reply["event_id"],
                    "delivery_token": reply["delivery_token"],
                },
            },
        )
    assert [x["type"] for x in delivered if "event_id" in x] == [
        "reply",
        "reply",
        "done",
    ]
    assert delivered[-1] == {
        "type": "delivery_status",
        "data": {"pending": False, "retry_ms": 2000},
    }
    assert {r["status"] for r in await events(flow, a.run_id)} == {"delivered"}


async def test_conversation_lock_skips_without_claiming(flow):
    await publish(flow)
    async with flow.db.tx() as tx:
        await tx.query_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1,0))::text AS held",
            "chat-ingress:" + flow.bound.conversation_id,
        )
        assert await asyncio.wait_for(SqlOutbox(flow.db).claim_next("other"), 1) is None


async def test_claim_resource_confusion_cannot_modify_another_conversation(flow):
    await publish(flow)
    store = SqlOutbox(flow.db)
    claim = await store.claim_next("w")
    other = await second_scope(flow)
    try:
        with pytest.raises(LeaseLost):
            await store.retry(replace(claim, scope=other))
        assert (await events(flow, claim.run_id))[0]["status"] == "delivering"
    finally:
        await flow.db.conversation.update(
            where={"id": other.conversation_id},
            data={
                "userId": flow.bound.owner_user_id,
                "agentId": flow.bound.agent_id,
                "workspaceId": None,
            },
        )
        await flow.db.chatworkspace.delete(where={"id": other.workspace_id})
        await flow.db.aiagent.delete(where={"id": other.agent_id})
        await flow.db.user.delete(where={"id": other.owner_user_id})


async def test_delivery_claim_commit_ack_loss_leaves_a_reclaimable_lease(flow):
    from tests.test_chat_ingress_postgres import FaultDatabase

    a, _ = await publish(flow)
    fault = FaultDatabase(flow.db, commit_unknown=True)
    fault.query_raw = flow.db.query_raw
    with pytest.raises(ConnectionError):
        await SqlOutbox(fault).claim_next("lost")
    row = (await events(flow, a.run_id))[0]
    assert row["status"] == "delivering" and row["attempts"] == 1
    assert await SqlOutbox(flow.db).claim_next("other") is None
    await flow.db.execute_raw(
        "UPDATE runtime_outbox SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
        row["id"],
    )
    fresh = await SqlOutbox(flow.db).claim_next("reclaim")
    assert fresh.id == row["id"] and fresh.token == 2


async def test_locked_outbox_head_is_skipped_without_wait_or_new_attempt(flow):
    a, _ = await publish(flow)
    head = (await events(flow, a.run_id))[0]
    async with flow.db.tx() as tx:
        await tx.query_raw(
            "SELECT id FROM runtime_outbox WHERE id=$1 FOR UPDATE", head["id"]
        )
        assert await asyncio.wait_for(SqlOutbox(flow.db).claim_next("other"), 1) is None
    assert (await events(flow, a.run_id))[0]["attempts"] == 0


async def test_prepared_callback_cannot_be_committed_under_a_different_claim(flow):
    a = await accept(flow)
    q = queue(flow)
    old = await q.claim_next("old")
    prepared = prepare_chat_result(old, (PreparedReply.capture("old"),))
    await expired_lease(flow, old)
    fresh = await q.claim_next("fresh")
    with pytest.raises(LeaseLost):
        await q.finish(fresh, prepared.result, commit=prepared.commit)
    assert await events(flow, a.run_id) == []
    current = prepare_chat_result(fresh, (PreparedReply.capture("current"),))
    assert await q.finish(fresh, current.result, commit=current.commit)
