"""R01.02 schema contracts on a migrated, synthetic loopback PostgreSQL.

These are transactional storage tests, not a claim that SQL workers/outbox
consumers are connected. Never accept application DATABASE_URL or production.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import os
import json
import re
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from prisma import Json, Prisma
from prisma.errors import RawQueryError

from app.services.runtime.execution_scope import (
    ExecutionScopeExpired, bind_conversation_scope, scoped_transaction,
)


class RuntimeTable:
    """Test-only SQL fixture writer for client-ignored runtime models.

    Production repositories will use parameterized SQL for lease/fencing CAS.
    Keep this helper out of application imports; it has no authorization logic.
    """
    def __init__(self, database, name):
        self.db = database
        self.name = {"agentrun": "agent_runs", "runtimejob": "runtime_jobs",
                     "agentaction": "agent_actions", "runtimeoutbox": "runtime_outbox"}[name]

    @staticmethod
    def column(name):
        assert re.fullmatch(r"[a-zA-Z][a-zA-Z0-9]*", name)
        return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()

    @staticmethod
    def row(record):
        return SimpleNamespace(**{re.sub(r"_([a-z])", lambda m: m[1].upper(), key): value
                                  for key, value in record.items()})

    def conditions(self, where, parameters):
        terms = []
        for key, value in where.items():
            values = value["in"] if isinstance(value, dict) else [value]
            placeholders = []
            for item in values:
                parameters.append(item)
                placeholders.append(f"${len(parameters)}")
            terms.append(f'{self.column(key)} IN ({",".join(placeholders)})' if placeholders else "FALSE")
        return " AND ".join(terms) or "TRUE"

    def assignments(self, data, parameters):
        columns, expressions = [], []
        for key, value in data.items():
            columns.append(self.column(key))
            parameters.append(json.dumps(value.data) if isinstance(value, Json) else value)
            cast = "::jsonb" if isinstance(value, Json) else "::uuid" if key.endswith("Generation") else "::timestamptz" if key.endswith("At") else ""
            expressions.append(f"${len(parameters)}" + cast)
        return columns, expressions

    async def create(self, *, data):
        data = {"id": str(uuid4()), **data, "updatedAt": datetime.now(timezone.utc)}
        parameters = []
        columns, expressions = self.assignments(data, parameters)
        records = await self.db.query_raw(f'INSERT INTO {self.name} ({",".join(columns)}) VALUES ({",".join(expressions)}) RETURNING *', *parameters)
        return self.row(records[0])

    async def update(self, *, where, data):
        parameters = []
        columns, expressions = self.assignments({**data, "updatedAt": datetime.now(timezone.utc)}, parameters)
        assignments = ",".join(f"{col}={value}" for col, value in zip(columns, expressions))
        conditions = self.conditions(where, parameters)
        records = await self.db.query_raw(f'UPDATE {self.name} SET {assignments} WHERE {conditions} RETURNING *', *parameters)
        return self.row(records[0])

    async def find_unique(self, *, where):
        parameters = []
        conditions = self.conditions(where, parameters)
        records = await self.db.query_raw(f'SELECT * FROM {self.name} WHERE {conditions}', *parameters)
        return self.row(records[0]) if records else None

    async def count(self, *, where):
        parameters = []
        conditions = self.conditions(where, parameters)
        return (await self.db.query_raw(f'SELECT count(*)::int AS n FROM {self.name} WHERE {conditions}', *parameters))[0]["n"]

    async def delete_many(self, *, where):
        parameters = []
        conditions = self.conditions(where, parameters)
        return await self.db.execute_raw(f'DELETE FROM {self.name} WHERE {conditions}', *parameters)


class RuntimeDatabase:
    """Preserve the real Prisma connection/transaction while writing raw SQL."""
    def __init__(self, database):
        self.client = database

    def __getattr__(self, name):
        if name in {"agentrun", "runtimejob", "agentaction", "runtimeoutbox"}:
            return RuntimeTable(self.client, name)
        return getattr(self.client, name)

    @asynccontextmanager
    async def tx(self, **options):
        async with self.client.tx(**options) as transaction:
            yield RuntimeDatabase(transaction)


@pytest.fixture
async def flow():
    url = os.getenv("PROACTIVE_E2E_DATABASE_URL")
    if not url:
        pytest.skip("isolated PostgreSQL required")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1"}
    assert parsed.path == "/companion_proactive_e2e"
    database = RuntimeDatabase(Prisma(datasource={"url": url}, http={"trust_env": False}))
    await database.connect()
    ids = {key: str(uuid4()) for key in ("owner", "agent", "workspace", "conversation", "other_conversation")}
    try:
        await database.user.create(data={"id": ids["owner"], "username": "runtime-" + ids["owner"]})
        await database.aiagent.create(data={"id": ids["agent"], "userId": ids["owner"], "name": "Synthetic"})
        await database.chatworkspace.create(data={"id": ids["workspace"], "userId": ids["owner"], "agentId": ids["agent"]})
        for key in ("conversation", "other_conversation"):
            await database.conversation.create(data={"id": ids[key], "userId": ids["owner"],
                "agentId": ids["agent"], "workspaceId": ids["workspace"] if key == "conversation" else None})
        bound = await bind_conversation_scope(actor_user_id=ids["owner"], conversation_id=ids["conversation"], database=database)
        yield SimpleNamespace(db=database, ids=ids, bound=bound, url=url)
    finally:
        await database.agentrun.delete_many(where={"ownerUserId": ids["owner"]})
        await database.message.delete_many(where={"conversationId": {"in": [ids[k] for k in ("conversation", "other_conversation")]}})
        await database.conversation.delete_many(where={"userId": ids["owner"]})
        await database.chatworkspace.delete_many(where={"id": ids["workspace"]})
        await database.aiagent.delete_many(where={"id": ids["agent"]})
        await database.user.delete_many(where={"id": ids["owner"]})
        await database.disconnect()


def run_data(flow, **overrides):
    b = flow.bound
    return {"actorUserId": b.actor_user_id, "ownerUserId": b.owner_user_id,
        "agentId": b.agent_id, "workspaceId": b.workspace_id, "conversationId": b.conversation_id,
        "actorGeneration": b.actor_generation, "ownerGeneration": b.owner_generation,
        "agentGeneration": b.agent_generation, "workspaceGeneration": b.workspace_generation,
        "conversationGeneration": b.conversation_generation,
        "kind": "chat", "requestKey": str(uuid4()), "inputFingerprint": sha256(b"synthetic").hexdigest(),
        "input": Json({"text": "synthetic"}), "graphVersion": "synthetic-v1", "stateVersion": 1,
        "configSnapshot": Json({"model": "synthetic"}), "promptSnapshot": Json({"key": "synthetic-v1"}),
        "budgetSnapshot": Json({"max_steps": 4}), **overrides}


def job_data(run_id, **overrides):
    return {"runId": run_id, "jobKey": "chat:0", "handler": "synthetic.chat",
        "queue": "foreground", "payload": Json({"input": "synthetic"}), **overrides}


def action_data(run_id, **overrides):
    return {"runId": run_id, "actionKey": "synthetic:0", "idempotencyKey": str(uuid4()),
        "kind": "synthetic", "inputFingerprint": sha256(b"synthetic").hexdigest(),
        "input": Json({"amount": 1}), **overrides}


def event_data(run_id, **overrides):
    return {"runId": run_id, "eventKey": "reply:0", "sequence": 0,
        "eventType": "reply", "payload": Json({"content": "synthetic"}), **overrides}


async def test_message_run_job_action_outbox_share_one_transaction(flow):
    async with scoped_transaction(flow.bound, database=flow.db) as tx:
        message = await tx.message.create(data={"conversationId": flow.ids["conversation"], "role": "assistant", "content": "synthetic"})
        run = await tx.agentrun.create(data=run_data(flow))
        await tx.runtimejob.create(data=job_data(run.id))
        await tx.agentaction.create(data=action_data(run.id))
        event = await tx.runtimeoutbox.create(data=event_data(run.id, messageId=message.id))
    assert await flow.db.agentrun.count(where={"id": run.id}) == 1
    assert await flow.db.runtimejob.count(where={"runId": run.id}) == 1
    assert await flow.db.agentaction.count(where={"runId": run.id}) == 1
    assert (await flow.db.runtimeoutbox.find_unique(where={"id": event.id})).messageId == message.id


@pytest.mark.parametrize("step", ["message", "run", "job", "action", "outbox"])
async def test_failure_at_each_write_rolls_back_all_records(flow, step):
    with pytest.raises(RuntimeError, match="synthetic abort"):
        async with scoped_transaction(flow.bound, database=flow.db) as tx:
            await tx.message.create(data={"conversationId": flow.ids["conversation"], "role": "user", "content": "synthetic"})
            if step == "message": raise RuntimeError("synthetic abort")
            run = await tx.agentrun.create(data=run_data(flow))
            if step == "run": raise RuntimeError("synthetic abort")
            await tx.runtimejob.create(data=job_data(run.id))
            if step == "job": raise RuntimeError("synthetic abort")
            await tx.agentaction.create(data=action_data(run.id))
            if step == "action": raise RuntimeError("synthetic abort")
            await tx.runtimeoutbox.create(data=event_data(run.id))
            raise RuntimeError("synthetic abort")
    assert await flow.db.message.count(where={"conversationId": flow.ids["conversation"]}) == 0
    assert await flow.db.agentrun.count(where={"ownerUserId": flow.ids["owner"]}) == 0


async def test_database_constraint_failure_rolls_back_earlier_writes(flow):
    with pytest.raises(RawQueryError) as failure:
        async with scoped_transaction(flow.bound, database=flow.db) as tx:
            run = await tx.agentrun.create(data=run_data(flow))
            await tx.runtimejob.create(data=job_data(run.id))
            await tx.runtimeoutbox.create(data=event_data(run.id, sequence=-1))
    assert failure.value.meta["code"] == "23514"
    assert await flow.db.agentrun.count(where={"ownerUserId": flow.ids["owner"]}) == 0


async def test_concurrent_duplicate_request_has_exactly_one_winner(flow):
    other = RuntimeDatabase(Prisma(datasource={"url": flow.url}, http={"trust_env": False}))
    await other.connect()
    try:
        data = run_data(flow, requestKey="same-client-id")
        results = await asyncio.gather(flow.db.agentrun.create(data=data), other.agentrun.create(data=data), return_exceptions=True)
        assert sum(isinstance(x, RawQueryError) and x.meta.get("code") == "23505" for x in results) == 1, [(type(x).__name__, str(x)) for x in results if isinstance(x, BaseException)]
        assert await flow.db.agentrun.count(where={"conversationId": flow.ids["conversation"], "requestKey": "same-client-id"}) == 1
    finally:
        await other.disconnect()


@pytest.mark.parametrize("delegate,factory", [("runtimejob", job_data), ("agentaction", action_data), ("runtimeoutbox", event_data)])
async def test_duplicate_business_key_is_rejected(flow, delegate, factory):
    run = await flow.db.agentrun.create(data=run_data(flow))
    model = getattr(flow.db, delegate)
    data = factory(run.id)
    await model.create(data=data)
    with pytest.raises(RawQueryError, match="already exists"): await model.create(data=data)
    assert await model.count(where={"runId": run.id}) == 1


async def test_action_global_idempotency_and_event_sequence_are_unique(flow):
    run = await flow.db.agentrun.create(data=run_data(flow))
    second = await flow.db.agentrun.create(data=run_data(flow))
    key = str(uuid4())
    await flow.db.agentaction.create(data=action_data(run.id, idempotencyKey=key))
    with pytest.raises(RawQueryError, match="already exists"):
        await flow.db.agentaction.create(data=action_data(second.id, idempotencyKey=key))
    await flow.db.runtimeoutbox.create(data=event_data(run.id))
    with pytest.raises(RawQueryError, match="already exists"):
        await flow.db.runtimeoutbox.create(data=event_data(run.id, eventKey="another-event"))


@pytest.mark.parametrize("field,value", [("actorGeneration", str(uuid4())), ("requestKey", "changed"),
    ("input", Json({"text": "changed"})), ("configSnapshot", Json({"model": "changed"})),
    ("promptSnapshot", Json({"key": "changed"})), ("budgetSnapshot", Json({"max_steps": 50})),
    ("graphVersion", "changed"), ("stateVersion", 2), ("parentRunId", str(uuid4()))])
async def test_run_snapshot_is_immutable(flow, field, value):
    run = await flow.db.agentrun.create(data=run_data(flow))
    with pytest.raises(Exception, match="runtime snapshot is immutable"):
        await flow.db.agentrun.update(where={"id": run.id}, data={field: value})


@pytest.mark.parametrize("delegate,factory,field,value", [
    ("runtimejob", job_data, "payload", Json({"changed": True})),
    ("runtimejob", job_data, "handler", "changed"),
    ("agentaction", action_data, "input", Json({"changed": True})),
    ("agentaction", action_data, "idempotencyKey", "changed"),
    ("runtimeoutbox", event_data, "payload", Json({"changed": True})),
    ("runtimeoutbox", event_data, "sequence", 1),
])
async def test_retries_cannot_rewrite_job_action_or_event(flow, delegate, factory, field, value):
    run = await flow.db.agentrun.create(data=run_data(flow))
    model = getattr(flow.db, delegate)
    row = await model.create(data=factory(run.id))
    with pytest.raises(Exception, match="runtime snapshot is immutable"):
        await model.update(where={"id": row.id}, data={field: value})


@pytest.mark.parametrize("delegate,factory,invalid", [
    ("agentrun", run_data, {"status": "nonsense"}), ("agentrun", run_data, {"inputFingerprint": "invalid"}),
    ("agentrun", run_data, {"configSnapshot": Json([])}), ("agentrun", run_data, {"scopeVersion": 2}),
    ("agentrun", run_data, {"status": "succeeded"}), ("runtimejob", job_data, {"status": "running"}),
    ("runtimejob", job_data, {"attempts": -1}), ("runtimejob", job_data, {"attempts": 4}),
    ("runtimejob", job_data, {"queue": "other"}), ("runtimejob", job_data, {"payload": Json(None)}),
    ("agentaction", action_data, {"status": "unknown"}), ("agentaction", action_data, {"input": Json([])}),
    ("runtimeoutbox", event_data, {"status": "delivering"}), ("runtimeoutbox", event_data, {"status": "delivered"}),
    ("runtimeoutbox", event_data, {"sequence": -1}), ("runtimeoutbox", event_data, {"payload": Json([])}),
])
async def test_invalid_initial_state_is_rejected(flow, delegate, factory, invalid):
    run = await flow.db.agentrun.create(data=run_data(flow))
    data = factory(flow if delegate == "agentrun" else run.id, **invalid)
    with pytest.raises(RawQueryError) as failure:
        await getattr(flow.db, delegate).create(data=data)
    assert failure.value.meta["code"] == "23514"


async def test_waiting_chat_releases_slot_and_background_can_overlap(flow):
    first = await flow.db.agentrun.create(data=run_data(flow, status="running"))
    with pytest.raises(RawQueryError, match="already exists"): await flow.db.agentrun.create(data=run_data(flow, status="running"))
    await flow.db.agentrun.create(data=run_data(flow, status="running", kind="background"))
    await flow.db.agentrun.update(where={"id": first.id}, data={"status": "waiting"})
    await flow.db.agentrun.create(data=run_data(flow, status="running"))


async def test_unknown_external_action_requires_reconciliation(flow):
    run = await flow.db.agentrun.create(data=run_data(flow))
    action = await flow.db.agentaction.create(data=action_data(run.id, status="unknown", startedAt=datetime.now(timezone.utc)))
    for state in ("planned", "started"):
        with pytest.raises(Exception, match="unknown action requires reconciliation"):
            await flow.db.agentaction.update(where={"id": action.id}, data={"status": state})
    await flow.db.agentaction.update(where={"id": action.id}, data={"status": "succeeded", "finishedAt": datetime.now(timezone.utc), "providerRef": "synthetic-confirmed"})


@pytest.mark.parametrize("delegate,factory,status", [("agentrun", run_data, "succeeded"), ("runtimejob", job_data, "succeeded"),
    ("agentaction", action_data, "succeeded"), ("runtimeoutbox", event_data, "delivered")])
async def test_terminal_records_cannot_be_resurrected(flow, delegate, factory, status):
    run = await flow.db.agentrun.create(data=run_data(flow))
    model = getattr(flow.db, delegate)
    row = run if delegate == "agentrun" else await model.create(data=factory(run.id))
    timestamp = datetime.now(timezone.utc)
    changes = {"status": status, "deliveredAt" if status == "delivered" else "finishedAt": timestamp}
    if delegate == "agentaction": changes["startedAt"] = timestamp
    await model.update(where={"id": row.id}, data=changes)
    with pytest.raises(Exception, match="terminal record is immutable"):
        await model.update(where={"id": row.id}, data={"status": "pending" if delegate != "agentrun" else "queued"})


@pytest.mark.parametrize("delegate,factory,running", [("runtimejob", job_data, "running"), ("runtimeoutbox", event_data, "delivering")])
async def test_lease_shape_and_monotonic_counters(flow, delegate, factory, running):
    run = await flow.db.agentrun.create(data=run_data(flow))
    model = getattr(flow.db, delegate)
    lease = {"status": running, "attempts": 2, "fencingToken": 2, "leaseOwner": "synthetic-worker",
             "leaseExpiresAt": datetime.now(timezone.utc) + timedelta(seconds=60)}
    row = await model.create(data=factory(run.id, **lease))
    await model.update(where={"id": row.id}, data={"leaseExpiresAt": datetime.now(timezone.utc) + timedelta(seconds=90)})
    for changes in ({"attempts": 1}, {"fencingToken": 1}, {"status": "pending"}):
        with pytest.raises(RawQueryError) as failure:
            await model.update(where={"id": row.id}, data=changes)
        assert failure.value.meta["code"] == "23514"
        if "status" not in changes:
            assert "runtime counters cannot decrease" in str(failure.value)
    await model.update(where={"id": row.id}, data={"status": "pending", "leaseOwner": None, "leaseExpiresAt": None})


async def test_child_scope_cannot_change_conversation_or_generation(flow):
    parent = await flow.db.agentrun.create(data=run_data(flow))
    child = await flow.db.agentrun.create(data=run_data(flow, kind="task", parentRunId=parent.id))
    assert child.parentRunId == parent.id
    for changes in ({"conversationId": flow.ids["other_conversation"]}, {"ownerGeneration": str(uuid4())}):
        with pytest.raises(Exception, match="parent scope mismatch"):
            await flow.db.agentrun.create(data=run_data(flow, parentRunId=parent.id, **changes))


async def test_outbox_cannot_reference_another_conversations_message(flow):
    run = await flow.db.agentrun.create(data=run_data(flow))
    message = await flow.db.message.create(data={"conversationId": flow.ids["other_conversation"], "role": "assistant", "content": "synthetic"})
    with pytest.raises(Exception, match="outbox message scope mismatch"):
        await flow.db.runtimeoutbox.create(data=event_data(run.id, messageId=message.id))


async def test_scope_reset_cannot_create_a_stale_run_through_commit_boundary(flow):
    await flow.db.execute_raw("UPDATE chat_workspaces SET execution_generation=gen_random_uuid()::text WHERE id=$1", flow.ids["workspace"])
    with pytest.raises(ExecutionScopeExpired):
        async with scoped_transaction(flow.bound, database=flow.db) as tx:
            await tx.agentrun.create(data=run_data(flow))
    assert await flow.db.agentrun.count(where={"ownerUserId": flow.ids["owner"]}) == 0


async def test_cascade_removes_children_and_all_runtime_records(flow):
    parent = await flow.db.agentrun.create(data=run_data(flow))
    child = await flow.db.agentrun.create(data=run_data(flow, kind="task", parentRunId=parent.id))
    await flow.db.runtimejob.create(data=job_data(child.id))
    await flow.db.agentaction.create(data=action_data(child.id))
    await flow.db.runtimeoutbox.create(data=event_data(child.id))
    await flow.db.conversation.delete(where={"id": flow.ids["conversation"]})
    for name in ("agentrun", "runtimejob", "agentaction", "runtimeoutbox"):
        where = {"id": {"in": [parent.id, child.id]}} if name == "agentrun" else {"runId": child.id}
        assert await getattr(flow.db, name).count(where=where) == 0


@pytest.mark.parametrize("delegate,factory", [("runtimejob", job_data), ("agentaction", action_data), ("runtimeoutbox", event_data)])
async def test_orphan_job_action_event_is_rejected(flow, delegate, factory):
    with pytest.raises(RawQueryError) as failure:
        await getattr(flow.db, delegate).create(data=factory(str(uuid4())))
    assert failure.value.meta["code"] == "23503"


async def test_raw_writer_cannot_bypass_snapshot_preservation(flow):
    run = await flow.db.agentrun.create(data=run_data(flow))
    with pytest.raises(Exception, match="runtime snapshot is immutable"):
        await flow.db.execute_raw("UPDATE agent_runs SET graph_version='tampered' WHERE id=$1", run.id)
    assert (await flow.db.agentrun.find_unique(where={"id": run.id})).graphVersion == "synthetic-v1"


async def test_migration_lock_failure_rolls_back_every_new_object_and_can_retry(flow):
    """Run the real migration in an owned namespace against existing parents.

    A competing table lock makes FK installation time out, after CREATE TABLE.
    Transactional DDL must leave no half-installed tables/functions behind.
    """
    from pathlib import Path
    import subprocess
    import shutil

    assert shutil.which("psql"), "migration failure verification requires psql"
    namespace = "runtime_migration_" + uuid4().hex
    other = RuntimeDatabase(Prisma(datasource={"url": flow.url}, http={"trust_env": False}))
    await other.connect()
    sql = Path("prisma/migrations/20261007013000_runtime_execution_foundation/migration.sql").read_text()
    environment = {**os.environ, "PGOPTIONS": f"-c search_path={namespace},public"}
    command = ["psql", "--no-psqlrc", "--dbname", flow.url, "--set", "ON_ERROR_STOP=1"]
    try:
        await flow.db.execute_raw(f'CREATE SCHEMA "{namespace}"')
        async with other.tx(timeout=timedelta(seconds=15)) as holding:
            await holding.execute_raw("LOCK TABLE public.conversations IN SHARE MODE")
            result = await asyncio.to_thread(subprocess.run, command, input=sql, env=environment,
                                            text=True, capture_output=True, timeout=10)
            assert result.returncode != 0 and "lock timeout" in result.stderr
        for table in ("agent_runs", "runtime_jobs", "agent_actions", "runtime_outbox"):
            rows = await flow.db.query_raw("SELECT to_regclass($1)::text AS name", f"{namespace}.{table}")
            assert rows[0]["name"] is None
        functions = await flow.db.query_raw("SELECT count(*)::int AS n FROM pg_proc p JOIN pg_namespace n ON p.pronamespace=n.oid WHERE n.nspname=$1", namespace)
        assert functions[0]["n"] == 0
        result = await asyncio.to_thread(subprocess.run, command, input=sql, env=environment,
                                        text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
    finally:
        await flow.db.execute_raw(f'DROP SCHEMA IF EXISTS "{namespace}" CASCADE')
        await other.disconnect()
