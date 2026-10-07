"""Real migrated synthetic PostgreSQL only; no app/prod DATABASE_URL fallback."""

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4

import pytest
from prisma import Json, Prisma

from tests.test_runtime_execution_foundation import (
    RuntimeDatabase,
    action_data,
    flow,
    run_data,
    job_data,
)
from tests.test_chat_ingress_postgres import FaultDatabase, accept, expire, snapshot
from app.services.runtime.chat_ingress import load_chat_turn, lookup_chat_request
from app.services.runtime.chat_ingress_contracts import (
    ChatAggregationPolicy,
    ChatRequestInput,
)
from app.services.runtime.execution_scope import (
    ExecutionScopeUnavailable,
    bind_conversation_scope,
)
from app.services.runtime.sql_job_contracts import (
    HandlerSpec,
    LeaseLost,
    LeasePolicy,
    WorkerStopRequired,
)
from app.services.runtime.sql_job_queue import SqlJobQueue
from app.services.runtime.sql_job_consumer import PreparedJobResult, SqlJobConsumer


def registry(**changes):
    return HandlerSpec(
        **{
            "name": "chat.execute.v1",
            "job_key": "chat:0",
            "versions": (("synthetic-v1", 1),),
            "retry_safe": True,
            **changes,
        }
    )


def queue(flow, *, spec=None, policy=None, database=None):
    return SqlJobQueue(database or flow.db, [spec or registry()], policy=policy)


async def state(flow, run_id):
    return (
        await flow.db.query_raw(
            "SELECT r.status AS run_status,j.* FROM agent_runs r "
            "JOIN runtime_jobs j ON j.run_id=r.id WHERE r.id=$1",
            run_id,
        )
    )[0]


async def expired_lease(flow, job):
    await flow.db.execute_raw(
        "UPDATE runtime_jobs SET lease_expires_at=clock_timestamp()-interval '1 second',"
        "updated_at=clock_timestamp() WHERE id=$1",
        job.id,
    )


async def second_scope(flow):
    owner = await flow.db.user.create(
        data={"username": "runtime-other-" + str(uuid4())}
    )
    agent = await flow.db.aiagent.create(
        data={"userId": owner.id, "name": "Other synthetic"}
    )
    ws = await flow.db.chatworkspace.create(
        data={"userId": owner.id, "agentId": agent.id}
    )
    await flow.db.conversation.update(
        where={"id": flow.ids["other_conversation"]},
        data={"userId": owner.id, "workspaceId": ws.id, "agentId": agent.id},
    )
    return await bind_conversation_scope(
        actor_user_id=owner.id,
        conversation_id=flow.ids["other_conversation"],
        database=flow.db,
    )


async def test_claim_commits_one_attempt_and_fence_without_changing_inputs(flow):
    accepted = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("worker-a")
    assert (
        claim.run_id == accepted.run_id and claim.attempts == claim.fencing_token == 1
    )
    row = await state(flow, claim.run_id)
    assert row["status"] == row["run_status"] == "running"
    assert await q.claim_next("worker-b") is None
    assert json.loads(claim.config_json) == {"model": "synthetic", "revision": 1}
    assert (
        await load_chat_turn(flow.bound, claim.run_id, database=flow.db)
    ).prompt_text == "synthetic"


async def test_eight_competing_workers_claim_only_once(flow):
    await accept(flow)
    other = RuntimeDatabase(
        Prisma(datasource={"url": flow.url}, http={"trust_env": False})
    )
    await other.connect()
    try:
        claims = await asyncio.gather(
            *(
                queue(flow, database=other if i % 2 else flow.db).claim_next(
                    "w-" + str(i)
                )
                for i in range(8)
            )
        )
        assert sum(x is not None for x in claims) == 1
    finally:
        await other.disconnect()


async def test_skips_locked_conversation_and_progresses_another_scope(flow):
    first = await accept(flow)
    scope = await second_scope(flow)
    try:
        other = await accept(flow, key="b", scope=scope)
        async with flow.db.tx() as tx:
            await tx.query_raw(
                "SELECT pg_advisory_xact_lock(hashtextextended($1,0))::text AS held",
                "chat-ingress:" + flow.bound.conversation_id,
            )
            claim = await queue(flow).claim_next("w")
            assert claim.run_id == other.run_id
        assert (await state(flow, first.run_id))["attempts"] == 0
    finally:
        await flow.db.agentrun.delete_many(where={"ownerUserId": scope.owner_user_id})
        await flow.db.message.delete_many(
            where={"conversationId": scope.conversation_id}
        )
        await flow.db.conversation.delete_many(where={"userId": scope.owner_user_id})
        await flow.db.chatworkspace.delete_many(where={"id": scope.workspace_id})
        await flow.db.aiagent.delete_many(where={"id": scope.agent_id})
        await flow.db.user.delete(where={"id": scope.owner_user_id})


async def test_run_row_skip_locked_does_not_wait_or_consume_attempt(flow):
    first = await accept(flow)
    async with flow.db.tx() as tx:
        await tx.query_raw(
            "SELECT id FROM agent_runs WHERE id=$1 FOR UPDATE", first.run_id
        )
        assert await asyncio.wait_for(queue(flow).claim_next("w"), 1) is None
    assert (await state(flow, first.run_id))["attempts"] == 0


async def test_fifo_is_receipt_order_even_when_later_availability_sorts_first(flow):
    first = await accept(flow, key="first")
    second = await accept(flow, key="second")
    await flow.db.execute_raw(
        "UPDATE runtime_jobs SET available_at=clock_timestamp()-interval '1 minute' WHERE id=$1",
        second.job_id,
    )
    q = queue(flow)
    claim = await q.claim_next("w")
    assert claim.run_id == first.run_id
    assert await q.claim_next("other") is None
    await q.finish(claim, {"reply": "one"})
    assert (await q.claim_next("other")).run_id == second.run_id


async def test_backoff_and_not_due_head_prevent_overtaking(flow):
    first = await accept(flow, key="first")
    await accept(flow, key="second")
    await flow.db.execute_raw(
        "UPDATE runtime_jobs SET available_at=clock_timestamp()+interval '1 hour' WHERE id=$1",
        first.job_id,
    )
    assert await queue(flow).claim_next("w") is None


async def test_unknown_graph_head_blocks_later_incompatible_order(flow):
    await accept(flow, key="old", execution=snapshot(graph_version="unsupported-v1"))
    await accept(flow, key="new")
    assert await queue(flow).claim_next("w") is None


async def test_collecting_window_is_sealed_before_claim_and_retries_keep_snapshot(flow):
    a = await accept(
        flow, key="a", text="你", policy=ChatAggregationPolicy("fragment_window", 5)
    )
    await accept(
        flow, key="b", text="好", policy=ChatAggregationPolicy("fragment_window", 5)
    )
    assert await queue(flow).claim_next("w") is None
    await expire(flow, a.run_id)
    claim = await queue(flow).claim_next("w")
    turn = await load_chat_turn(flow.bound, a.run_id, database=flow.db)
    assert turn.phase == "ready" and turn.prompt_text == "你好"
    assert turn.job_id == claim.id and len(turn.message_ids) == 2


async def test_heartbeat_extends_current_lease_but_cannot_revive_expired_one(flow):
    await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    await q.heartbeat(claim)
    assert (await state(flow, claim.run_id))["lease_expires_at"] is not None
    await expired_lease(flow, claim)
    with pytest.raises(LeaseLost):
        await q.heartbeat(claim)


async def test_expired_takeover_increments_fence_and_rejects_every_old_commit(flow):
    await accept(flow)
    q = queue(flow)
    old = await q.claim_next("w")
    await expired_lease(flow, old)
    new = await q.claim_next("w")  # Even the same worker ID cannot reuse a fence.
    assert (new.attempts, new.fencing_token) == (2, 2)
    for operation in [
        q.heartbeat(old),
        q.finish(old, {"late": True}),
        q.fail(old, code="late", retryable=True),
    ]:
        with pytest.raises(LeaseLost):
            await operation
    await q.finish(new, {"reply": "current"})
    result = (await state(flow, new.run_id))["result"]
    assert (json.loads(result) if isinstance(result, str) else result) == {
        "reply": "current"
    }


async def test_owner_or_token_forgery_cannot_commit(flow):
    await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    for forged in [
        replace(claim, worker_id="other"),
        replace(claim, fencing_token=99),
        replace(claim, id=str(uuid4())),
    ]:
        with pytest.raises(LeaseLost):
            await q.finish(forged, {"fake": True})
    assert (await state(flow, claim.run_id))["status"] == "running"


async def test_retry_is_bounded_and_exhaustion_terminates_run(flow):
    a = await accept(flow, execution=snapshot(max_attempts=2))
    q = queue(flow)
    first = await q.claim_next("w")
    await expired_lease(flow, first)
    second = await q.claim_next("w")
    await expired_lease(flow, second)
    assert await q.claim_next("w") is None
    row = await state(flow, a.run_id)
    assert (
        row["status"] == row["run_status"] == "failed"
        and row["attempts"] == 2
        and row["lease_owner"] is None
    )


async def test_expired_unsafe_handler_does_not_execute_again(flow):
    a = await accept(flow)
    q = queue(flow, spec=registry(retry_safe=False))
    await expired_lease(flow, await q.claim_next("w"))
    assert await q.claim_next("other") is None
    assert (await state(flow, a.run_id))["status"] == "failed"


async def test_retryable_failure_releases_lease_but_preserves_attempts_and_fifo(flow):
    a = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    await q.fail(claim, code="transient", retryable=True)
    row = await state(flow, a.run_id)
    assert (
        row["status"] == "retry"
        and row["run_status"] == "queued"
        and row["attempts"] == 1
    )
    assert row["lease_owner"] is None and await q.claim_next("w") is None
    await flow.db.execute_raw(
        "UPDATE runtime_jobs SET available_at=clock_timestamp()-interval '1 second' WHERE id=$1",
        claim.id,
    )
    assert (await q.claim_next("w")).fencing_token == 2


async def test_unknown_provider_effect_is_held_for_reconciliation(flow):
    a = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    action = await flow.db.agentaction.create(
        data=action_data(
            a.run_id, status="started", startedAt=datetime.now(timezone.utc)
        )
    )
    await expired_lease(flow, claim)
    assert await q.claim_next("other") is None
    row = await state(flow, a.run_id)
    assert (
        row["run_status"] == "waiting"
        and row["status"] == "retry"
        and row["attempts"] == 1
    )
    assert (
        await flow.db.agentaction.find_unique(where={"id": action.id})
    ).status == "unknown"
    assert await q.claim_next("other") is None


async def test_scope_reset_rejects_heartbeat_finish_and_retires_stale_run(flow):
    a = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    await flow.db.conversation.update(
        where={"id": flow.ids["conversation"]}, data={"isDeleted": True}
    )
    for operation in [q.heartbeat(claim), q.finish(claim, {"late": True})]:
        with pytest.raises(ExecutionScopeUnavailable):
            await operation
    await expired_lease(flow, claim)
    assert await q.claim_next("other") is None
    assert (await state(flow, a.run_id))["run_status"] == "cancelled"


async def test_inactive_owner_queued_work_is_cancelled_without_execution(flow):
    a = await accept(flow)
    await flow.db.user.update(
        where={"id": flow.ids["owner"]}, data={"status": "disabled"}
    )
    assert await queue(flow).claim_next("w") is None
    assert (await state(flow, a.run_id))["status"] == "cancelled"


async def test_cancel_is_idempotent_and_old_worker_cannot_write(flow):
    a = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    assert await q.cancel_run(flow.bound, a.run_id)
    assert not await q.cancel_run(flow.bound, a.run_id)
    with pytest.raises(LeaseLost):
        await q.finish(claim, {"late": True})


async def test_finish_result_and_business_write_commit_once(flow):
    a = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")

    async def commit(tx):
        await tx.message.create(
            data={
                "conversationId": flow.bound.conversation_id,
                "role": "assistant",
                "content": "prepared",
            }
        )

    await q.finish(claim, {"text": "prepared"}, commit=commit)
    found = await lookup_chat_request(
        flow.bound,
        ChatRequestInput.from_client(client_id="a", text="synthetic"),
        database=flow.db,
    )
    assert found.run_status == found.job_status == "succeeded" and json.loads(
        found.result_json
    ) == {"text": "prepared"}
    with pytest.raises(LeaseLost):
        await q.finish(claim, {"text": "prepared"}, commit=commit)
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 1
    )


async def test_commit_callback_exception_rolls_back_message_job_and_run(flow):
    await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")

    async def commit(tx):
        await tx.message.create(
            data={
                "conversationId": flow.bound.conversation_id,
                "role": "assistant",
                "content": "must rollback",
            }
        )
        raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError):
        await q.finish(claim, {}, commit=commit)
    assert (await state(flow, claim.run_id))["status"] == "running"
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 0
    )


async def test_lease_expiring_inside_business_commit_rolls_back(flow):
    await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    await flow.db.execute_raw(
        "UPDATE runtime_jobs SET lease_expires_at=clock_timestamp()+interval '300 milliseconds' WHERE id=$1",
        claim.id,
    )

    async def commit(tx):
        await tx.message.create(
            data={
                "conversationId": flow.bound.conversation_id,
                "role": "assistant",
                "content": "too late",
            }
        )
        await tx.query_raw("SELECT pg_sleep(0.4)::text AS paused")

    with pytest.raises(LeaseLost):
        await q.finish(claim, {}, commit=commit)
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 0
    )


async def test_local_revoke_blocks_new_and_already_open_fenced_transaction(flow):
    await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    with pytest.raises(LeaseLost):
        async with q.fenced_transaction(claim) as tx:
            await tx.message.create(
                data={
                    "conversationId": flow.bound.conversation_id,
                    "role": "assistant",
                    "content": "revoke",
                }
            )
            claim.revoke()
    with pytest.raises(LeaseLost):
        await q.heartbeat(claim)
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 0
    )


async def test_deadline_expires_without_handler_or_new_attempt(flow):
    a = await accept(
        flow,
        execution=snapshot(
            deadline_at=datetime.now(timezone.utc) + timedelta(seconds=0.4)
        ),
    )
    await asyncio.sleep(0.5)
    assert await queue(flow).claim_next("w") is None
    row = await state(flow, a.run_id)
    assert row["status"] == row["run_status"] == "failed" and row["attempts"] == 0


async def test_consumer_runs_handler_and_commits_result(flow):
    a = await accept(flow)
    q = queue(flow)

    async def handler(claim):
        return PreparedJobResult({"answer": json.loads(claim.payload_json)["run_id"]})

    c = SqlJobConsumer(q, {"chat.execute.v1": handler}, "w")
    assert await c.poll_once() and not await c.poll_once()
    assert (await state(flow, a.run_id))["status"] == "succeeded"


async def test_consumer_failure_retries_only_registered_safe_handler(flow):
    a = await accept(flow)
    q = queue(flow)

    async def handler(claim):
        raise RuntimeError("private provider error must not persist")

    assert await SqlJobConsumer(q, {"chat.execute.v1": handler}, "w").poll_once()
    row = await state(flow, a.run_id)
    assert row["status"] == "retry" and "private" not in str(row["error"])


async def test_heartbeat_loss_cancels_handler_and_blocks_its_late_commit(flow):
    await accept(flow)
    policy = LeasePolicy(lease_seconds=1, heartbeat_seconds=0.1)
    q = queue(flow, policy=policy)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(claim):
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            with pytest.raises(LeaseLost):
                await q.finish(claim, {"late": True})
            cancelled.set()
            raise

    consumer = SqlJobConsumer(q, {"chat.execute.v1": handler}, "w")
    poll = asyncio.create_task(consumer.poll_once())
    await started.wait()
    await q.cancel_run(
        flow.bound,
        (
            await flow.db.query_raw(
                "SELECT id FROM agent_runs WHERE conversation_id=$1",
                flow.bound.conversation_id,
            )
        )[0]["id"],
    )
    assert await asyncio.wait_for(poll, 2)
    assert cancelled.is_set()


async def test_consumer_execution_timeout_revokes_and_leaves_lease_for_recovery(flow):
    a = await accept(flow)
    q = queue(flow, spec=registry(max_execution_seconds=0.1))
    stopped = asyncio.Event()

    async def handler(claim):
        try:
            await asyncio.sleep(30)
        finally:
            stopped.set()

    assert await SqlJobConsumer(q, {"chat.execute.v1": handler}, "w").poll_once()
    assert stopped.is_set() and (await state(flow, a.run_id))["status"] == "running"


async def test_consumer_shutdown_cancel_does_not_requeue_live_work(flow):
    a = await accept(flow)
    q = queue(flow)
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def handler(claim):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            stopped.set()

    c = SqlJobConsumer(q, {"chat.execute.v1": handler}, "w")
    task = asyncio.create_task(c.poll_once())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set() and (await state(flow, a.run_id))["status"] == "running"


async def test_registry_mismatch_and_duplicate_handlers_fail_before_claim(flow):
    async def handler(claim):
        return PreparedJobResult({})

    with pytest.raises(ValueError):
        SqlJobConsumer(queue(flow), {"unknown": handler}, "w")
    with pytest.raises(ValueError):
        SqlJobQueue(flow.db, [registry(), registry()])

    def synchronous(claim):
        return PreparedJobResult({})

    async def streaming(claim):
        yield PreparedJobResult({})

    for unsafe in (synchronous, streaming):
        with pytest.raises(ValueError):
            SqlJobConsumer(queue(flow), {"chat.execute.v1": unsafe}, "w")


class FaultQueueDatabase(FaultDatabase):
    def __getattr__(self, name):
        return getattr(self.database, name)


@pytest.mark.parametrize("marker", ["UPDATE agent_runs", "UPDATE runtime_jobs"])
async def test_claim_write_failure_rolls_back_attempt_lease_run_and_sealing(
    flow, marker
):
    a = await accept(flow, policy=ChatAggregationPolicy("fragment_window", 5))
    await expire(flow, a.run_id)
    q = queue(flow, database=FaultQueueDatabase(flow.db, marker))
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        await q.claim_next("w")
    row = await state(flow, a.run_id)
    assert (
        row["run_status"] == "queued" and row["attempts"] == row["fencing_token"] == 0
    )
    assert (
        await load_chat_turn(flow.bound, a.run_id, database=flow.db)
    ).phase == "collecting"


@pytest.mark.parametrize("marker", ["UPDATE agent_runs", "UPDATE runtime_jobs"])
async def test_finish_write_failure_rolls_back_prepared_business_result(flow, marker):
    await accept(flow)
    claim = await queue(flow).claim_next("w")
    q = queue(flow, database=FaultQueueDatabase(flow.db, marker))

    async def commit(tx):
        await tx.message.create(
            data={
                "conversationId": flow.bound.conversation_id,
                "role": "assistant",
                "content": "rollback",
            }
        )

    with pytest.raises(RuntimeError):
        await q.finish(claim, {}, commit=commit)
    row = await state(flow, claim.run_id)
    assert row["status"] == row["run_status"] == "running" and row["fencing_token"] == 1
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 0
    )


async def test_claim_commit_ack_lost_never_starts_a_second_live_execution(flow):
    a = await accept(flow)
    with pytest.raises(ConnectionError):
        await queue(
            flow, database=FaultQueueDatabase(flow.db, commit_unknown=True)
        ).claim_next("w")
    assert (await state(flow, a.run_id))["attempts"] == 1
    assert await queue(flow).claim_next("other") is None


async def test_finish_commit_ack_lost_returns_stored_result_without_second_reply(flow):
    await accept(flow)
    claim = await queue(flow).claim_next("w")
    q = queue(flow, database=FaultQueueDatabase(flow.db, commit_unknown=True))

    async def commit(tx):
        await tx.message.create(
            data={
                "conversationId": flow.bound.conversation_id,
                "role": "assistant",
                "content": "committed",
            }
        )

    with pytest.raises(ConnectionError):
        await q.finish(claim, {"reply": "committed"}, commit=commit)
    assert await queue(flow).claim_next("other") is None
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 1
    )
    found = await lookup_chat_request(
        flow.bound,
        ChatRequestInput.from_client(client_id="a", text="synthetic"),
        database=flow.db,
    )
    assert found.run_status == "succeeded" and json.loads(found.result_json) == {
        "reply": "committed"
    }


async def test_corrupt_or_deleted_source_cannot_be_claimed(flow):
    a = await accept(flow)
    await flow.db.message.delete(where={"id": a.message_id})
    assert await queue(flow).claim_next("w") is None
    assert (await state(flow, a.run_id))["status"] == "failed"


async def test_boolean_counter_cannot_impersonate_integer_input_shape(flow):
    a = await accept(flow)
    await flow.db.execute_raw(
        "UPDATE agent_runs SET state=jsonb_set(state,'{ingress,message_count}','true') WHERE id=$1",
        a.run_id,
    )
    assert await queue(flow).claim_next("w") is None
    assert (await state(flow, a.run_id))["status"] == "failed"


async def test_blocked_conversation_does_not_fill_bounded_global_scan(flow):
    a = await accept(flow, key="first")
    await accept(flow, key="later")
    await flow.db.execute_raw(
        "UPDATE runtime_jobs SET available_at=clock_timestamp()+interval '1 hour' WHERE id=$1",
        a.job_id,
    )
    scope = await second_scope(flow)
    try:
        other = await accept(flow, key="other", scope=scope)
        assert (
            await queue(flow, policy=LeasePolicy(scan_limit=1)).claim_next("w")
        ).run_id == other.run_id
    finally:
        await flow.db.agentrun.delete_many(where={"ownerUserId": scope.owner_user_id})
        await flow.db.message.delete_many(
            where={"conversationId": scope.conversation_id}
        )
        await flow.db.conversation.delete_many(where={"userId": scope.owner_user_id})
        await flow.db.chatworkspace.delete_many(where={"id": scope.workspace_id})
        await flow.db.aiagent.delete_many(where={"id": scope.agent_id})
        await flow.db.user.delete(where={"id": scope.owner_user_id})


async def test_finish_and_failure_pause_unknown_effect_without_publishing_reply(flow):
    a = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    await flow.db.agentaction.create(
        data=action_data(
            a.run_id, status="started", startedAt=datetime.now(timezone.utc)
        )
    )

    async def commit(tx):
        raise AssertionError("Unresolved provider effect must not publish a reply")

    assert await q.finish(claim, {}, commit=commit) is False
    row = await state(flow, a.run_id)
    assert row["status"] == "retry" and row["run_status"] == "waiting"
    before = row["updated_at"]
    assert await q.claim_next("other") is None
    assert (await state(flow, a.run_id))["updated_at"] == before


async def test_handler_ignoring_cancellation_requires_worker_stop_and_fails_all_late_fences(
    flow,
):
    await accept(flow)
    q = queue(
        flow,
        spec=registry(max_execution_seconds=0.05),
        policy=LeasePolicy(cancellation_seconds=0.05),
    )
    release = asyncio.Event()
    finished = asyncio.Event()

    async def handler(claim):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await release.wait()
            with pytest.raises(LeaseLost):
                await q.finish(claim, {"late": True})
            finished.set()
            return PreparedJobResult({})

    consumer = SqlJobConsumer(q, {"chat.execute.v1": handler}, "w")
    try:
        with pytest.raises(WorkerStopRequired):
            await consumer.poll_once()
        with pytest.raises(WorkerStopRequired):
            await consumer.poll_once()
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 1)


async def test_stop_event_cancels_active_handler_without_early_requeue(flow):
    a = await accept(flow)
    q = queue(flow)
    started = asyncio.Event()
    stopped = asyncio.Event()
    stop = asyncio.Event()

    async def handler(claim):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            stopped.set()

    c = SqlJobConsumer(q, {"chat.execute.v1": handler}, "w")
    task = asyncio.create_task(c.run(stop))
    await started.wait()
    stop.set()
    await asyncio.wait_for(task, 2)
    assert stopped.is_set() and (await state(flow, a.run_id))["status"] == "running"


async def test_real_heartbeat_keeps_long_handler_owned_beyond_original_ttl(flow):
    a = await accept(flow)
    q = queue(flow, policy=LeasePolicy(lease_seconds=0.6, heartbeat_seconds=0.15))

    async def handler(claim):
        await asyncio.sleep(1.0)
        return PreparedJobResult({"long": True})

    assert await SqlJobConsumer(q, {"chat.execute.v1": handler}, "w").poll_once()
    row = await state(flow, a.run_id)
    assert row["status"] == "succeeded" and row["attempts"] == 1


async def test_renewal_db_failure_cancels_preparation_and_propagates_without_fallback(
    flow,
):
    a = await accept(flow)
    q = queue(flow, policy=LeasePolicy(lease_seconds=1, heartbeat_seconds=0.1))
    stopped = asyncio.Event()

    async def failure(claim):
        raise ConnectionError("synthetic unavailable database")

    q.heartbeat = failure

    async def handler(claim):
        try:
            await asyncio.sleep(30)
        finally:
            stopped.set()

    with pytest.raises(ConnectionError):
        await SqlJobConsumer(q, {"chat.execute.v1": handler}, "w").poll_once()
    assert stopped.is_set() and (await state(flow, a.run_id))["status"] == "running"


async def test_legacy_executor_is_never_claimed_by_langgraph_registration(flow):
    a = await accept(flow, execution=snapshot(executor="legacy"))
    assert await queue(flow).claim_next("graph-worker") is None
    assert (await state(flow, a.run_id))["attempts"] == 0
    legacy = queue(flow, spec=registry(executor="legacy"))
    assert (await legacy.claim_next("legacy-worker")).run_id == a.run_id


async def test_background_queue_and_priority_are_explicit_and_independent(flow):
    first = await flow.db.agentrun.create(
        data=run_data(flow, kind="background", graphVersion="synthetic-v1")
    )
    second = await flow.db.agentrun.create(
        data=run_data(flow, kind="background", graphVersion="synthetic-v1")
    )
    await flow.db.runtimejob.create(
        data=job_data(first.id, queue="background", priority=100)
    )
    await flow.db.runtimejob.create(
        data=job_data(second.id, queue="background", priority=1)
    )
    spec = registry(name="synthetic.chat", kind="background")
    q = queue(flow, spec=spec)
    assert await q.claim_next("w", queue="foreground") is None
    a = await q.claim_next("w", queue="background")
    assert a.run_id == second.id
    b = await q.claim_next("other", queue="background")
    assert b.run_id == first.id
    await q.finish(a, {"priority": 1})
    await q.finish(b, {"priority": 100})


async def test_multiple_jobs_need_coordinator_and_cannot_complete_a_shared_run(flow):
    a = await accept(flow)
    await flow.db.runtimejob.create(data=job_data(a.run_id, jobKey="other:0"))
    assert await queue(flow).claim_next("w") is None
    assert (await state(flow, a.run_id))["attempts"] == 0


async def test_child_runs_are_excluded_until_parent_cancellation_coordinator_exists(
    flow,
):
    parent = await flow.db.agentrun.create(
        data=run_data(flow, kind="task", graphVersion="synthetic-v1")
    )
    child = await flow.db.agentrun.create(
        data=run_data(
            flow, kind="task", graphVersion="synthetic-v1", parentRunId=parent.id
        )
    )
    await flow.db.runtimejob.create(data=job_data(child.id))
    q = queue(flow, spec=registry(name="synthetic.chat", kind="task"))
    assert await q.claim_next("w") is None
    assert await q.cancel_run(flow.bound, parent.id)
    assert await q.claim_next("w") is None
    assert (await state(flow, child.id))["attempts"] == 0


async def test_failed_handler_with_unknown_action_waits_instead_of_retrying(flow):
    a = await accept(flow)
    q = queue(flow)
    claim = await q.claim_next("w")
    await flow.db.agentaction.create(
        data=action_data(
            a.run_id, status="started", startedAt=datetime.now(timezone.utc)
        )
    )
    await q.fail(claim, code="handler_failed", retryable=True)
    row = await state(flow, a.run_id)
    assert row["status"] == "retry" and row["run_status"] == "waiting"
    assert await q.claim_next("other") is None


async def test_waiting_unknown_actions_cannot_fill_the_bounded_scan(flow):
    setup = queue(flow)
    for _ in range(3):
        a = await accept(flow, key=f"held-{_}")
        claim = await setup.claim_next("setup")
        assert claim.run_id == a.run_id
        await flow.db.agentaction.create(
            data=action_data(
                a.run_id, status="unknown", startedAt=datetime.now(timezone.utc)
            )
        )
        assert not await setup.finish(claim, {})
    ready = await accept(flow, key="ready")
    q = queue(flow, policy=LeasePolicy(scan_limit=2))
    claim = await q.claim_next("ready")
    assert claim.run_id == ready.run_id
    assert await q.finish(claim, {})


async def test_reconciled_older_chat_waits_for_the_current_running_turn(flow):
    first = await accept(flow)
    q = queue(flow)
    old = await q.claim_next("old")
    action = await flow.db.agentaction.create(
        data=action_data(
            first.run_id, status="unknown", startedAt=datetime.now(timezone.utc)
        )
    )
    assert not await q.finish(old, {})
    second = await accept(flow, key="second")
    current = await q.claim_next("current")
    assert current.run_id == second.run_id
    await flow.db.agentaction.update(
        where={"id": action.id},
        data={"status": "succeeded", "finishedAt": datetime.now(timezone.utc)},
    )
    assert await q.claim_next("resume") is None
    assert await q.finish(current, {})
    resumed = await q.claim_next("resume")
    assert resumed.run_id == first.run_id and resumed.fencing_token == 2
    assert await q.finish(resumed, {})


_PROCESS_WORKER = r"""
import asyncio,json,sys
from contextlib import asynccontextmanager
from uuid import uuid4
from prisma import Prisma
from app.services.runtime.sql_job_contracts import HandlerSpec
from app.services.runtime.sql_job_queue import SqlJobQueue
url,run_id,conversation_id,graph,marker=json.loads(sys.argv[1])
spec=HandlerSpec('chat.execute.v1','chat:0',((graph,1),),retry_safe=True)
class PausingDatabase:
 def __init__(self,database):self.database=database
 def __getattr__(self,name):return getattr(self.database,name)
 @asynccontextmanager
 async def tx(self,**options):
  async with self.database.tx(**options) as real:
   class Transaction:
    def __getattr__(self,name):return getattr(real,name)
    async def query_raw(self,sql,*args):
     result=await real.query_raw(sql,*args)
     if marker in sql:print('CRASH_WINDOW',flush=True);await asyncio.sleep(3600)
     return result
    async def execute_raw(self,sql,*args):
     result=await real.execute_raw(sql,*args)
     if marker in sql:print('CRASH_WINDOW',flush=True);await asyncio.sleep(3600)
     return result
   yield Transaction()
async def commit(tx):
 await tx.execute_raw("INSERT INTO messages (id,conversation_id,role,content) VALUES ($1,$2,'assistant','prepared')",str(uuid4()),conversation_id)
async def main():
 db=Prisma(datasource={'url':url},http={'trust_env':False});await db.connect()
 try:
  q=SqlJobQueue(PausingDatabase(db),[spec]);claim=await q.claim_next('killed-worker')
  assert claim.run_id==run_id
  if marker=='CLAIM_COMMITTED':print('CRASH_WINDOW',flush=True);await asyncio.sleep(3600)
  assert await q.finish(claim,{'reply':'prepared'},commit=commit)
  if marker=='RESULT_COMMITTED':print('CRASH_WINDOW',flush=True);await asyncio.sleep(3600)
 finally:await db.disconnect()
asyncio.run(main())
"""


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX process-group SIGKILL required"
)
@pytest.mark.parametrize(
    "marker",
    [
        "UPDATE agent_runs SET status='running'",
        "UPDATE runtime_jobs SET status='running'",
        "CLAIM_COMMITTED",
        "INSERT INTO messages",
        "UPDATE runtime_jobs SET status='succeeded'",
        "RESULT_COMMITTED",
    ],
)
async def test_real_worker_sigkill_preserves_atomicity_and_prevents_duplicate_reply(
    flow, marker
):
    # flow admits only a named synthetic loopback DB. A unique graph contract
    # isolates discovery from every other suite; no application DATABASE_URL.
    graph = "crash-ci-" + str(uuid4())
    a = await accept(flow, execution=snapshot(graph_version=graph))
    spec = registry(versions=((graph, 1),))
    q = queue(flow, spec=spec)
    env = {
        **os.environ,
        "APP_ENV": "test",
        "PYTHON_DOTENV_DISABLED": "1",
        "ONLINE_MODEL": "false",
        "TRACE_BACKEND": "off",
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        _PROCESS_WORKER,
        json.dumps([flow.url, a.run_id, flow.bound.conversation_id, graph, marker]),
        env=env,
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
        # Kill the owned worker group, including its Prisma engine, so its
        # transaction closes. Never signal pytest or any pre-existing process.
        os.killpg(process.pid, signal.SIGKILL)
        assert await asyncio.wait_for(process.wait(), 10) == -signal.SIGKILL
    finally:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
    row = await state(flow, a.run_id)
    before_commit = marker.startswith("UPDATE") and "status='running'" in marker
    if marker == "RESULT_COMMITTED":
        assert row["status"] == row["run_status"] == "succeeded"
        assert await q.claim_next("recover") is None
    else:
        assert row["attempts"] == (0 if before_commit else 1)
        assert (
            await flow.db.message.count(
                where={
                    "conversationId": flow.bound.conversation_id,
                    "role": "assistant",
                }
            )
            == 0
        )
        if not before_commit:
            await flow.db.execute_raw(
                "UPDATE runtime_jobs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
                a.job_id,
            )
        claim = await q.claim_next("recover")
        assert claim.attempts == (1 if before_commit else 2)
        if not before_commit:
            with pytest.raises(LeaseLost):
                await q.finish(
                    replace(
                        claim, worker_id="killed-worker", fencing_token=1, attempts=1
                    ),
                    {"late": True},
                )

        async def commit(tx):
            await tx.message.create(
                data={
                    "conversationId": flow.bound.conversation_id,
                    "role": "assistant",
                    "content": "prepared",
                }
            )

        assert await q.finish(claim, {"reply": "prepared"}, commit=commit)
    assert (
        await flow.db.message.count(
            where={"conversationId": flow.bound.conversation_id, "role": "assistant"}
        )
        == 1
    )
