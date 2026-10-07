"""SQL ingress on synthetic, migrated loopback PostgreSQL; never application DB."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4

import pytest
from prisma import Json, Prisma
from prisma.errors import RawQueryError

from tests.test_runtime_execution_foundation import RuntimeDatabase, flow  # shared isolated fixture
from app.services.runtime.chat_ingress import (
    ChatIngressCorrupt, ChatRequestConflict, accept_chat_message,
    load_chat_turn, lookup_chat_request, seal_due_chat_turn,
)
from app.services.runtime.chat_ingress_contracts import (
    ChatAggregationPolicy, ChatExecutionSnapshot, ChatRequestInput, PreparedChatMessage,
)
from app.services.runtime.execution_scope import (
    ExecutionScopeExpired, ExecutionScopeUnavailable, bind_conversation_scope,
)


def snapshot(**overrides):
    return ChatExecutionSnapshot.capture(**{
        "executor": "langgraph", "graph_version": "synthetic-v1", "state_version": 1,
        "config": {"model": "synthetic", "revision": 1},
        "prompts": {"reply": {"revision": 1, "content": "synthetic"}},
        "budget": {"model_calls": 8}, **overrides,
    })


def prepared(text, *, context=None, metadata=None):
    now = datetime.now(timezone.utc)
    return PreparedChatMessage.capture(persisted_text=text, prompt_text=text,
        metadata=metadata or {}, reply_context=context or {"received_at": now.isoformat()},
        received_at=now)


async def accept(flow, key="a", text="synthetic", *, policy=None, execution=None,
                 message=None, database=None, scope=None):
    return await accept_chat_message(scope or flow.bound,
        ChatRequestInput.from_client(client_id=key, text=text),
        message or prepared(text), execution or snapshot(),
        policy or ChatAggregationPolicy("immediate"), database=database or flow.db)


async def counts(flow):
    return {"messages": await flow.db.message.count(where={"conversationId": flow.ids["conversation"]}),
            "runs": await flow.db.agentrun.count(where={"conversationId": flow.ids["conversation"]}),
            "jobs": (await flow.db.query_raw("SELECT count(*)::int AS n FROM runtime_jobs j "
                "JOIN agent_runs r ON r.id=j.run_id WHERE r.conversation_id=$1", flow.ids["conversation"]))[0]["n"],
            "receipts": (await flow.db.query_raw("SELECT count(*)::int AS n FROM chat_ingress_receipts "
                "WHERE conversation_id=$1", flow.ids["conversation"]))[0]["n"]}


async def expire(flow, run_id):
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    await flow.db.execute_raw("UPDATE agent_runs SET state=jsonb_set(state,'{ingress,window_due_at}',"
        "to_jsonb($2::text)),updated_at=clock_timestamp() WHERE id=$1", run_id, past.isoformat())
    await flow.db.execute_raw("UPDATE runtime_jobs SET available_at=$2::timestamptz,"
        "updated_at=clock_timestamp() WHERE run_id=$1", run_id, past.isoformat())


async def test_new_message_receipt_run_and_job_commit_together(flow):
    result = await accept(flow, key="a'; DROP TABLE messages; --", text="你好🙂")
    assert result.created and result.phase == "ready"
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}
    row = await flow.db.message.find_unique(where={"id": result.message_id})
    assert row.content == "你好🙂" and row.metadata["client_id"] == "a'; DROP TABLE messages; --"
    turn = await load_chat_turn(flow.bound, result.run_id, database=flow.db)
    assert turn.message_ids == (result.message_id,) and turn.prompt_text == "你好🙂"
    assert turn.snapshot == snapshot()


async def test_retry_ignores_new_rendering_policy_and_configuration(flow):
    first = await accept(flow)
    retry = await accept(flow, message=prepared("changed rendering"),
        execution=snapshot(config={"revision": 99}), policy=ChatAggregationPolicy("fragment_window", 5))
    assert not retry.created
    assert (retry.message_id, retry.run_id, retry.job_id, retry.ordinal, retry.available_at) == (
        first.message_id, first.run_id, first.job_id, first.ordinal, first.available_at)
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}
    assert (await load_chat_turn(flow.bound, first.run_id, database=flow.db)).snapshot == snapshot()


async def test_different_content_with_same_identity_is_a_conflict_without_writes(flow):
    await accept(flow)
    before = await counts(flow)
    with pytest.raises(ChatRequestConflict):
        await accept(flow, text="different")
    with pytest.raises(ChatRequestConflict):
        await lookup_chat_request(flow.bound, ChatRequestInput.from_client(client_id="a", text="different"), database=flow.db)
    assert await counts(flow) == before


async def test_new_lookup_is_read_only_and_existing_lookup_returns_original(flow):
    request = ChatRequestInput.from_client(client_id="a", text="synthetic")
    assert await lookup_chat_request(flow.bound, request, database=flow.db) is None
    assert await counts(flow) == {"messages": 0, "runs": 0, "jobs": 0, "receipts": 0}
    first = await accept(flow)
    found = await lookup_chat_request(flow.bound, request, database=flow.db)
    assert found.message_id == first.message_id and not found.created
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}


async def test_concurrent_same_body_retries_share_one_commit(flow):
    other = RuntimeDatabase(Prisma(datasource={"url": flow.url}, http={"trust_env": False}))
    await other.connect()
    try:
        results = await asyncio.gather(*(accept(flow, database=flow.db if i % 2 else other) for i in range(8)))
        assert sum(item.created for item in results) == 1
        assert len({(item.message_id, item.run_id, item.job_id, item.ordinal) for item in results}) == 1
        assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}
    finally:
        await other.disconnect()


async def test_concurrent_conflicting_requests_cannot_both_succeed(flow):
    results = await asyncio.gather(accept(flow, text="a"), accept(flow, text="b"), return_exceptions=True)
    assert sum(isinstance(item, ChatRequestConflict) for item in results) == 1
    assert sum(getattr(item, "created", False) for item in results) == 1
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}


class FaultDatabase:
    """Wrap real transaction writes, injecting a fault after the selected SQL."""
    def __init__(self, database, marker=None, *, commit_unknown=False):
        self.database, self.marker, self.commit_unknown = database, marker, commit_unknown

    @asynccontextmanager
    async def tx(self, **options):
        async with self.database.tx(**options) as real:
            parent = self
            class FaultTransaction:
                def __getattr__(self, name):
                    return getattr(real, name)

                async def query_raw(self, sql, *args):
                    value = await real.query_raw(sql, *args)
                    if parent.marker and parent.marker in sql:
                        raise RuntimeError("synthetic write failure")
                    return value

                async def execute_raw(self, sql, *args):
                    value = await real.execute_raw(sql, *args)
                    if parent.marker and parent.marker in sql:
                        raise RuntimeError("synthetic write failure")
                    return value
            yield FaultTransaction()
        if self.commit_unknown:
            raise ConnectionError("synthetic commit acknowledgement lost")


@pytest.mark.parametrize("marker", ["INSERT INTO agent_runs", "INSERT INTO runtime_jobs", "INSERT INTO messages",
                                    "INSERT INTO chat_ingress_receipts", "UPDATE agent_runs", "UPDATE runtime_jobs"])
async def test_failure_after_each_real_write_rolls_back_everything(flow, marker):
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        await accept(flow, database=FaultDatabase(flow.db, marker))
    assert await counts(flow) == {"messages": 0, "runs": 0, "jobs": 0, "receipts": 0}
    result = await accept(flow)
    assert result.created
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}


async def test_commit_succeeded_but_acknowledgement_lost_is_recovered_by_same_id(flow):
    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await accept(flow, database=FaultDatabase(flow.db, commit_unknown=True))
    result = await accept(flow)
    assert not result.created
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}


async def test_terminal_retry_returns_stored_result_without_restarting(flow):
    first = await accept(flow)
    now = datetime.now(timezone.utc)
    await flow.db.runtimejob.update(where={"id": first.job_id}, data={"status": "succeeded", "finishedAt": now})
    await flow.db.agentrun.update(where={"id": first.run_id}, data={"status": "succeeded", "finishedAt": now,
        "result": Json({"reply_message_ids": ["synthetic-reply"]})})
    retry = await accept(flow)
    assert not retry.created and retry.run_status == retry.job_status == "succeeded"
    assert json.loads(retry.result_json) == {"reply_message_ids": ["synthetic-reply"]}
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}


async def test_fragments_join_in_order_and_complete_message_closes_window(flow):
    first_time = datetime.now(timezone.utc).isoformat()
    fragment = ChatAggregationPolicy("fragment_window", 5)
    a = await accept(flow, key="a", text="我", policy=fragment,
                     message=prepared("我", context={"received_at": first_time, "received_status": "first"}))
    b = await accept(flow, key="b", text="想", policy=fragment)
    last_time = datetime.now(timezone.utc).isoformat()
    c = await accept(flow, key="c", text="吃火锅", policy=ChatAggregationPolicy("turn_window", 1.2, 4),
                     message=prepared("吃火锅", context={"received_at": last_time, "received_status": "last"}))
    assert a.run_id == b.run_id == c.run_id and c.phase == "ready"
    assert a.job_id == b.job_id == c.job_id
    turn = await load_chat_turn(flow.bound, a.run_id, database=flow.db)
    assert turn.prompt_text == "我想吃火锅"
    assert turn.message_ids == (a.message_id, b.message_id, c.message_id)
    assert json.loads(turn.reply_context_json) == {"received_at": first_time,
        "received_status": "first", "latest_received_at": last_time, "latest_user_emotion": {}}
    assert await counts(flow) == {"messages": 3, "runs": 1, "jobs": 1, "receipts": 3}


async def test_normal_turn_newline_join_and_read_only_rewrites_keep_all_sources(flow):
    policy = ChatAggregationPolicy("turn_window", 30, 60)
    a = await accept(flow, key="a", text="你现在在干嘛", policy=policy)
    b = await accept(flow, key="b", text="你现在在干嘛", policy=policy)
    c = await accept(flow, key="c", text="我今天特别高兴", policy=policy)
    turn = await load_chat_turn(flow.bound, a.run_id, database=flow.db)
    assert a.run_id == b.run_id == c.run_id
    assert turn.prompt_text == "你现在在干嘛\n我今天特别高兴"
    assert turn.message_ids == (a.message_id, b.message_id, c.message_id)
    assert await counts(flow) == {"messages": 3, "runs": 1, "jobs": 1, "receipts": 3}


async def test_new_configuration_cannot_change_an_open_turn_snapshot(flow):
    policy = ChatAggregationPolicy("turn_window", 30, 60)
    first = await accept(flow, key="a", policy=policy)
    second = await accept(flow, key="b", text="next", policy=policy,
        execution=snapshot(config={"revision": 2}, prompts={"reply": "new"}, budget={"model_calls": 99}))
    assert second.run_id == first.run_id
    assert (await load_chat_turn(flow.bound, first.run_id, database=flow.db)).snapshot == snapshot()


async def test_executor_version_change_closes_previous_turn_and_creates_new_run(flow):
    policy = ChatAggregationPolicy("turn_window", 30, 60)
    first = await accept(flow, key="a", policy=policy)
    second = await accept(flow, key="b", text="next", policy=policy,
                          execution=snapshot(graph_version="synthetic-v2"))
    assert second.run_id != first.run_id
    assert (await load_chat_turn(flow.bound, first.run_id, database=flow.db)).phase == "ready"
    assert (await load_chat_turn(flow.bound, second.run_id, database=flow.db)).snapshot.graph_version == "synthetic-v2"


async def test_expired_window_is_never_extended_by_a_new_input(flow):
    policy = ChatAggregationPolicy("turn_window", 30, 60)
    first = await accept(flow, key="a", policy=policy)
    await expire(flow, first.run_id)
    second = await accept(flow, key="b", text="next", policy=policy)
    assert second.run_id != first.run_id
    old_turn = await load_chat_turn(flow.bound, first.run_id, database=flow.db)
    assert old_turn.phase == "ready" and old_turn.message_ids == (first.message_id,)


async def test_turn_quiet_window_cannot_exceed_first_maximum_wait(flow):
    policy = ChatAggregationPolicy("turn_window", 30, 60)
    first = await accept(flow, key="a", policy=policy)
    old = datetime.now(timezone.utc) - timedelta(seconds=59)
    await flow.db.execute_raw("UPDATE agent_runs SET state=jsonb_set(state,'{ingress,first_accepted_at}',"
        "to_jsonb($2::text)),updated_at=clock_timestamp() WHERE id=$1", first.run_id, old.isoformat())
    second = await accept(flow, key="b", text="next", policy=ChatAggregationPolicy("turn_window", 300, 600))
    assert second.run_id == first.run_id
    assert second.available_at == old + timedelta(seconds=60)


async def test_retry_does_not_refresh_an_aggregation_window(flow):
    first = await accept(flow, policy=ChatAggregationPolicy("fragment_window", 5))
    retry = await accept(flow, policy=ChatAggregationPolicy("fragment_window", 5))
    assert retry.available_at == first.available_at
    assert (await load_chat_turn(flow.bound, first.run_id, database=flow.db)).message_ids == (first.message_id,)


async def test_due_window_sealing_is_idempotent_and_does_not_claim_a_job(flow):
    first = await accept(flow, policy=ChatAggregationPolicy("fragment_window", 5, delay_seconds=2))
    assert not await seal_due_chat_turn(flow.bound, first.run_id, database=flow.db)
    await expire(flow, first.run_id)
    assert await seal_due_chat_turn(flow.bound, first.run_id, database=flow.db)
    sealed = await load_chat_turn(flow.bound, first.run_id, database=flow.db)
    assert sealed.phase == "ready"
    assert await seal_due_chat_turn(flow.bound, first.run_id, database=flow.db)
    job = await flow.db.runtimejob.find_unique(where={"id": first.job_id})
    assert job.status == "pending" and job.attempts == 0 and job.fencingToken == 0


async def test_bypass_input_does_not_join_a_normal_turn(flow):
    first = await accept(flow, key="a", policy=ChatAggregationPolicy("turn_window", 30, 60))
    urgent = await accept(flow, key="b", text="urgent", policy=ChatAggregationPolicy("immediate", allow_join=False))
    assert urgent.run_id != first.run_id and urgent.phase == "ready"
    assert (await load_chat_turn(flow.bound, first.run_id, database=flow.db)).phase == "collecting"


async def test_concurrent_new_inputs_have_durable_acceptance_order(flow):
    other = RuntimeDatabase(Prisma(datasource={"url": flow.url}, http={"trust_env": False}))
    await other.connect()
    try:
        policy = ChatAggregationPolicy("turn_window", 30, 60)
        results = await asyncio.gather(*(accept(flow, key=str(i), text=f"ordinary {i}", policy=policy,
            database=flow.db if i % 2 else other) for i in range(6)))
        assert len({item.run_id for item in results}) == 1
        ordered = sorted(results, key=lambda item: item.ordinal)
        turn = await load_chat_turn(flow.bound, results[0].run_id, database=flow.db)
        assert turn.message_ids == tuple(item.message_id for item in ordered)
        by_message = {item.message_id: f"ordinary {i}" for i, item in enumerate(results)}
        assert turn.prompt_text == "\n".join(by_message[item.message_id] for item in ordered)
        assert await counts(flow) == {"messages": 6, "runs": 1, "jobs": 1, "receipts": 6}
    finally:
        await other.disconnect()


async def test_failed_append_preserves_previous_window_and_sources(flow):
    policy = ChatAggregationPolicy("turn_window", 30, 60)
    first = await accept(flow, key="a", policy=policy)
    before = await load_chat_turn(flow.bound, first.run_id, database=flow.db)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        await accept(flow, key="b", text="next", policy=policy,
                     database=FaultDatabase(flow.db, "UPDATE runtime_jobs"))
    assert await load_chat_turn(flow.bound, first.run_id, database=flow.db) == before
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}


async def test_message_limit_closes_a_turn_without_discarding_sources(flow):
    policy = ChatAggregationPolicy("turn_window", 30, 60)
    results = [await accept(flow, key=str(i), text=f"message {i}", policy=policy) for i in range(33)]
    assert len({item.run_id for item in results[:32]}) == 1
    assert results[-1].run_id != results[0].run_id
    first_turn = await load_chat_turn(flow.bound, results[0].run_id, database=flow.db)
    assert first_turn.phase == "ready" and len(first_turn.message_ids) == 32
    assert await counts(flow) == {"messages": 33, "runs": 2, "jobs": 2, "receipts": 33}


async def test_character_limit_starts_a_separate_turn(flow):
    policy = ChatAggregationPolicy("turn_window", 30, 60)
    first = await accept(flow, key="a", text="a" * 32768, policy=policy)
    second = await accept(flow, key="b", text="b", policy=policy)
    assert second.run_id != first.run_id
    assert len((await load_chat_turn(flow.bound, first.run_id, database=flow.db)).prompt_text) == 32768


async def test_expired_input_and_sealing_race_cannot_extend_old_turn(flow):
    policy = ChatAggregationPolicy("fragment_window", 30)
    first = await accept(flow, key="a", text="a", policy=policy)
    await expire(flow, first.run_id)
    _, second = await asyncio.gather(seal_due_chat_turn(flow.bound, first.run_id, database=flow.db),
                                    accept(flow, key="b", text="b", policy=policy))
    assert second.run_id != first.run_id
    old_turn = await load_chat_turn(flow.bound, first.run_id, database=flow.db)
    assert old_turn.phase == "ready" and old_turn.prompt_text == "a"
    assert await counts(flow) == {"messages": 2, "runs": 2, "jobs": 2, "receipts": 2}


async def test_scope_reset_blocks_both_stale_acceptance_and_historical_retry(flow):
    first = await accept(flow)
    await flow.db.execute_raw("UPDATE chat_workspaces SET execution_generation=gen_random_uuid()::text WHERE id=$1", flow.ids["workspace"])
    with pytest.raises(ExecutionScopeExpired):
        await accept(flow, key="b")
    fresh = await bind_conversation_scope(actor_user_id=flow.ids["owner"],
        conversation_id=flow.ids["conversation"], database=flow.db)
    with pytest.raises(ExecutionScopeExpired):
        await accept(flow, scope=fresh)
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}
    assert (await accept(flow, key="b", scope=fresh)).run_id != first.run_id


async def test_archive_restore_does_not_resurrect_an_old_input(flow):
    await accept(flow)
    await flow.db.conversation.update(where={"id": flow.ids["conversation"]},
                                     data={"archivedAt": datetime.now(timezone.utc)})
    with pytest.raises(ExecutionScopeUnavailable):
        await accept(flow, key="b")
    await flow.db.conversation.update(where={"id": flow.ids["conversation"]}, data={"archivedAt": None})
    fresh = await bind_conversation_scope(actor_user_id=flow.ids["owner"],
        conversation_id=flow.ids["conversation"], database=flow.db)
    with pytest.raises(ExecutionScopeExpired):
        await accept(flow, scope=fresh)


async def test_scope_forgery_is_rejected_before_any_write(flow):
    from dataclasses import replace
    with pytest.raises(ExecutionScopeExpired):
        await accept(flow, scope=replace(flow.bound, owner_user_id="untrusted-owner"))
    assert await counts(flow) == {"messages": 0, "runs": 0, "jobs": 0, "receipts": 0}


async def test_same_request_identity_in_another_conversation_is_independent(flow):
    other_workspace, other_agent = str(uuid4()), str(uuid4())
    try:
        await flow.db.aiagent.create(data={"id": other_agent, "userId": flow.ids["owner"], "name": "Other synthetic"})
        await flow.db.chatworkspace.create(data={"id": other_workspace, "userId": flow.ids["owner"],
            "agentId": other_agent, "allowMultipleActive": True})
        await flow.db.conversation.update(where={"id": flow.ids["other_conversation"]},
            data={"workspaceId": other_workspace, "agentId": other_agent})
        other_scope = await bind_conversation_scope(actor_user_id=flow.ids["owner"],
            conversation_id=flow.ids["other_conversation"], database=flow.db)
        first = await accept(flow)
        second = await accept(flow, text="different", scope=other_scope)
        assert first.message_id != second.message_id and first.run_id != second.run_id
    finally:
        await flow.db.agentrun.delete_many(where={"conversationId": flow.ids["other_conversation"]})
        await flow.db.message.delete_many(where={"conversationId": flow.ids["other_conversation"]})
        await flow.db.conversation.update(where={"id": flow.ids["other_conversation"]}, data={"workspaceId": None, "agentId": flow.ids["agent"]})
        await flow.db.chatworkspace.delete_many(where={"id": other_workspace})
        await flow.db.aiagent.delete_many(where={"id": other_agent})


async def test_client_id_metadata_is_owned_by_the_request_identity(flow):
    first = await accept(flow, message=prepared("synthetic", metadata={"client_id": "spoofed"}))
    row = await flow.db.message.find_unique(where={"id": first.message_id})
    assert row.metadata["client_id"] == "a"


async def test_rendered_attachment_text_is_preserved_without_changing_retry_identity(flow):
    request = ChatRequestInput.from_client(client_id="a", text="", attachment_ids=["synthetic-attachment"])
    result = await accept_chat_message(flow.bound, request, prepared("[image: synthetic scene]"),
        snapshot(), ChatAggregationPolicy("immediate"), database=flow.db)
    assert (await load_chat_turn(flow.bound, result.run_id, database=flow.db)).prompt_text == "[image: synthetic scene]"
    same = await lookup_chat_request(flow.bound, request, database=flow.db)
    assert same.message_id == result.message_id
    with pytest.raises(ChatRequestConflict):
        await lookup_chat_request(flow.bound, ChatRequestInput.from_client(client_id="a", text="", attachment_ids=["different"]), database=flow.db)


async def test_provider_retry_and_client_retry_have_separate_identities(flow):
    first = await accept(flow, key="42")
    request = ChatRequestInput.from_wechat(message_id="42", text="synthetic")
    second = await accept_chat_message(flow.bound, request, prepared("synthetic", metadata={"client_id": "untrusted"}),
        snapshot(), ChatAggregationPolicy("immediate"), database=flow.db)
    retry = await lookup_chat_request(flow.bound, request, database=flow.db)
    assert first.run_id != second.run_id and retry.message_id == second.message_id
    assert "client_id" not in (await flow.db.message.find_unique(where={"id": second.message_id})).metadata


async def test_deleted_source_is_not_silently_recreated_from_stored_prompt(flow):
    first = await accept(flow)
    await flow.db.message.delete(where={"id": first.message_id})
    with pytest.raises(ChatIngressCorrupt):
        await load_chat_turn(flow.bound, first.run_id, database=flow.db)


async def test_execution_deadline_prevents_unschedulable_acceptance(flow):
    with pytest.raises(ValueError, match="deadline"):
        await accept(flow, execution=snapshot(deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
    assert await counts(flow) == {"messages": 0, "runs": 0, "jobs": 0, "receipts": 0}


async def test_rejected_append_restores_previous_deadline_window_and_sources(flow, monkeypatch):
    execution = snapshot(deadline_at=datetime.now(timezone.utc) + timedelta(seconds=31))
    first = await accept(flow, policy=ChatAggregationPolicy("turn_window", 30), execution=execution)
    before = await load_chat_turn(flow.bound, first.run_id, database=flow.db)
    async def advancing_clock(tx):
        return first.available_at - timedelta(seconds=28)
    monkeypatch.setattr("app.services.runtime.chat_ingress._now", advancing_clock)
    with pytest.raises(ValueError, match="deadline"):
        await accept(flow, key="b", policy=ChatAggregationPolicy("turn_window", 30), execution=execution)
    assert await load_chat_turn(flow.bound, first.run_id, database=flow.db) == before
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}


async def test_empty_context_uses_server_receipt_time(flow):
    message = PreparedChatMessage.capture(persisted_text="synthetic", prompt_text="synthetic",
        metadata={}, reply_context={}, received_at=datetime.now(timezone.utc))
    first = await accept(flow, message=message)
    turn = await load_chat_turn(flow.bound, first.run_id, database=flow.db)
    assert json.loads(turn.reply_context_json)["received_at"] == message.received_at.isoformat()


async def test_advisory_lock_timeout_has_no_partial_persistence(flow):
    other = RuntimeDatabase(Prisma(datasource={"url": flow.url}, http={"trust_env": False}))
    await other.connect()
    try:
        async with other.tx(timeout=timedelta(seconds=10)) as holding:
            await holding.query_raw("SELECT pg_advisory_xact_lock(hashtextextended($1,0))::text AS held", "chat-ingress:"+flow.ids["conversation"])
            with pytest.raises(RawQueryError) as failure:
                await accept(flow)
            assert failure.value.meta["code"] == "55P03"
        assert await counts(flow) == {"messages": 0, "runs": 0, "jobs": 0, "receipts": 0}
        assert (await accept(flow)).created
    finally:
        await other.disconnect()


@pytest.mark.parametrize("field", ["request_key", "prepared_input", "message_id", "ordinal"])
async def test_receipts_are_immutable_even_to_raw_writers(flow, field):
    first = await accept(flow)
    assignments = {"request_key": "request_key='changed'", "prepared_input": "prepared_input='{}'::jsonb",
                   "message_id": "message_id='changed'", "ordinal": "ordinal=ordinal+1000"}
    with pytest.raises(RawQueryError, match="receipt is immutable"):
        await flow.db.execute_raw("UPDATE chat_ingress_receipts SET " + assignments[field] + " WHERE message_id=$1", first.message_id)


@pytest.mark.parametrize("invalid", ["other_conversation", "assistant", "other_job", "closed_run",
                                    "missing_version", "string_version", "missing_count", "attempted_job"])
async def test_database_guard_rejects_invalid_source_and_execution_associations(flow, invalid):
    first = await accept(flow, key="a", policy=ChatAggregationPolicy("turn_window", 30, 60))
    second = await accept(flow, key="b", policy=ChatAggregationPolicy("immediate", allow_join=False))
    source = await flow.db.message.create(data={"conversationId": flow.ids["other_conversation"] if invalid == "other_conversation" else flow.ids["conversation"],
        "role": "assistant" if invalid == "assistant" else "user", "content": "synthetic guard"})
    run_id, job_id = first.run_id, first.job_id
    if invalid == "other_job": job_id = second.job_id
    if invalid == "closed_run": run_id, job_id = second.run_id, second.job_id
    if invalid in {"missing_version", "string_version", "missing_count"}:
        changed = {"missing_version": "state #- '{ingress,version}'",
                   "string_version": "jsonb_set(state, '{ingress,version}', '\"1\"'::jsonb)",
                   "missing_count": "state #- '{ingress,message_count}'"}[invalid]
        await flow.db.execute_raw("UPDATE agent_runs SET state="+changed+" WHERE id=$1", run_id)
    if invalid == "attempted_job":
        await flow.db.execute_raw("UPDATE runtime_jobs SET attempts=1 WHERE id=$1", job_id)
    with pytest.raises(RawQueryError) as failure:
        await flow.db.execute_raw("INSERT INTO chat_ingress_receipts (id,conversation_id,request_key,source,client_id,"
            "input_fingerprint,request_input,prepared_input,message_id,run_id,job_id,received_at,accepted_at) "
            "SELECT $1,conversation_id,'client:guard','client','guard',input_fingerprint,request_input,"
            "prepared_input,$2,$3,$4,received_at,accepted_at FROM chat_ingress_receipts WHERE message_id=$5",
            str(uuid4()), source.id, run_id, job_id, first.message_id)
    assert failure.value.meta["code"] == "23514"


async def test_ingress_migration_lock_failure_rolls_back_new_objects_and_retries(flow):
    from pathlib import Path
    import os
    import shutil
    import subprocess

    assert shutil.which("psql"), "migration verification requires psql"
    namespace = "ingress_migration_" + uuid4().hex
    other = RuntimeDatabase(Prisma(datasource={"url": flow.url}, http={"trust_env": False}))
    await other.connect()
    sql = Path("prisma/migrations/20261007060000_chat_ingress_receipts/migration.sql").read_text()
    environment = {**os.environ, "PGOPTIONS": f"-c search_path={namespace},public"}
    command = ["psql", "--no-psqlrc", "--dbname", flow.url, "--set", "ON_ERROR_STOP=1"]
    try:
        await flow.db.execute_raw(f'CREATE SCHEMA "{namespace}"')
        async with other.tx(timeout=timedelta(seconds=15)) as holding:
            await holding.execute_raw("LOCK TABLE public.conversations IN SHARE MODE")
            result = await asyncio.to_thread(subprocess.run, command, input=sql, env=environment,
                                            text=True, capture_output=True, timeout=10)
            assert result.returncode != 0 and "lock timeout" in result.stderr
        for object_name in ("chat_ingress_receipts", "chat_ingress_receipts_ordinal_seq"):
            rows = await flow.db.query_raw("SELECT to_regclass($1)::text AS name", f"{namespace}.{object_name}")
            assert rows[0]["name"] is None
        functions = await flow.db.query_raw("SELECT count(*)::int AS n FROM pg_proc p "
            "JOIN pg_namespace n ON p.pronamespace=n.oid WHERE n.nspname=$1", namespace)
        assert functions[0]["n"] == 0
        result = await asyncio.to_thread(subprocess.run, command, input=sql, env=environment,
                                        text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
    finally:
        await flow.db.execute_raw(f'DROP SCHEMA IF EXISTS "{namespace}" CASCADE')
        await other.disconnect()
