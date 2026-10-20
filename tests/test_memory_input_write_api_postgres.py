"""Real HTTP/JWT and PostgreSQL contracts for imported and edited memories.

Only embeddings and postcommit cache/notification hooks are replaced. Parsing,
authorization, transactions, source guards and response serialization are real.
Fixtures are synthetic and restricted to the existing loopback-only PG fixture.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from time import perf_counter
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prisma.errors import DataError, RawQueryError
import pytest
from redis.asyncio import Redis

from app.api.admin import agent_templates, memory_repairs
from app.api.public import memories
from app.services.agent_template import knowledge, registry
from app.services.agent_template.knowledge_import import parse_knowledge_document_snapshot
from app.services.auth import create_jwt
from app.services.memory import evidence, evidence_read, manual_edit, repair_queue
from app.services.memory.retrieval import knowledge_hits
from app.services.memory.storage import repo
from app.services.workspace import workspaces
from tests.test_memory_evidence_postgres import origins


DOCUMENT = "合成知识资料\n称呼：合成品牌\n颜色：蓝色\n".encode()
VECTOR = [.012] * 1024


def auth(user_id, role="user"):
    return {"Authorization": "Bearer " + create_jwt(user_id, role=role)}


@asynccontextmanager
async def reject_write(db, table, predicate, *, operation="INSERT"):
    name = "input_test_failure_" + uuid4().hex
    await db.execute_raw(
        f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ "
        f"BEGIN IF {predicate} THEN RAISE EXCEPTION 'synthetic input failure'; "
        "END IF; RETURN NEW; END $$"
    )
    await db.execute_raw(f"CREATE TRIGGER {name} BEFORE {operation} ON {table} "
                         f"FOR EACH ROW EXECUTE FUNCTION {name}()")
    try:
        yield
    finally:
        await db.execute_raw(f"DROP TRIGGER {name} ON {table}")
        await db.execute_raw(f"DROP FUNCTION {name}()")


@pytest.fixture
async def importing(origins, monkeypatch):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    for module in (knowledge, registry, agent_templates, workspaces):
        monkeypatch.setattr(module, "db", db)
    await db.user.update(where={"id": users[-1]}, data={"username": registry.TEMPLATE_SYSTEM_USERNAME})
    await db.aiagent.update(where={"id": agents[-1]}, data={"status": "active"})
    embedding = AsyncMock(return_value=VECTOR)
    monkeypatch.setattr(knowledge, "generate_embedding", embedding)
    monkeypatch.setattr(knowledge, "after_memory_changelog", AsyncMock())
    monkeypatch.setattr(knowledge, "_bust_knowledge_rows_cache", AsyncMock())
    monkeypatch.setattr(repo, "invalidate_scope", AsyncMock())
    application = FastAPI()
    application.include_router(agent_templates.router)
    application.include_router(memory_repairs.router)
    async with AsyncClient(transport=ASGITransport(app=application, raise_app_exceptions=False),
                           base_url="http://isolated.test") as client:
        async def append(raw=DOCUMENT, **overrides):
            items, origin = parse_knowledge_document_snapshot(raw, filename="private-upload.txt")
            args = dict(template_agent_id=agents[-1], template_user_id=users[-1],
                        items=items, input_origin=origin)
            return await knowledge.append_knowledge_to_template(**{**args, **overrides})
        yield origins, append, client, embedding


async def test_admin_upload_binds_exact_private_document_to_each_memory_and_replay(importing):
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    await db.user.update(where={'id':uid},data={'role':'admin'})
    path = f"/admin-api/agent-templates/{agents[-1]}/knowledge/from-document"
    response = await client.post(path, files={"file": ("private-upload.txt", DOCUMENT)}, headers=auth(uid, "admin"))
    assert response.status_code == 200, response.text
    expected, origin = parse_knowledge_document_snapshot(DOCUMENT, filename="private-upload.txt")
    assert response.json()["stored"] == len(expected) > 1
    rows = await db.aimemory.find_many(where={"workspaceId": spaces[-1], "provenance": "knowledge_seed"})
    assert {row.content for row in rows} == {item.summary for item in expected}
    for row in rows:
        assert (row.type, row.level, row.mainCategory, row.subCategory, row.importance, row.provenance) == (
            "life", 1, "生活", "工作", .86, "knowledge_seed",
        )
        assert row.valueUpdatedAt is not None and row.statementTime is not None
        assert row.valueUpdatedAt == row.statementTime
        assert abs((datetime.now(UTC) - row.valueUpdatedAt).total_seconds()) < 30
    snapshots = await db.query_raw("SELECT * FROM memory_input_origins WHERE workspace_id=$1", spaces[-1])
    assert len(snapshots) == 1 and snapshots[0]["source_version"] == origin.version
    payload = json.loads(snapshots[0]["payload_text"])["input"]
    assert payload["upload_sha256"] == hashlib.sha256(DOCUMENT).hexdigest()
    # The upload hash pins the bytes; extraction trims the parser-input text.
    assert payload["text"] == DOCUMENT.decode().strip() and payload["upload_size_bytes"] == len(DOCUMENT)
    assert "private-upload.txt" not in snapshots[0]["payload_text"]
    for row in rows:
        source = await client.get(f"/admin-api/memory-repairs/evidence/ai/{row.id}",
            params={"user_id": users[-1], "workspace_id": spaces[-1]}, headers=auth(uid, "admin"))
        assert source.status_code == 200
        item = source.json()["items"][0]
        assert item["input"] == {"kind": "knowledge_document", "format_version": "memory-input-v1"}
        assert item["relation"] == "recorded_from" and item["current_content"]
        assert item["availability"] == "recorded_snapshot" and item["source_status"] == "knowledge_document"
        assert DOCUMENT.decode() not in source.text and "payload_text" not in source.text
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",
                              [row.id for row in rows]))[0]["n"] == len(rows)
    assert await db.memorychangelog.count(where={"workspaceId": spaces[-1], "operation": "insert"}) == len(rows)
    replay = await append()
    assert replay["stored"] == 0 and replay["skipped_duplicates"] == len(rows)
    assert len(await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", spaces[-1])) == 1


async def test_upload_real_auth_size_and_foreign_template_fail_before_embedding(importing):
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    path = f"/admin-api/agent-templates/{agents[-1]}/knowledge/from-document"
    assert (await client.post(path, files={"file": ("a.txt", DOCUMENT)})).status_code == 401
    assert (await client.post(path, files={"file": ("a.txt", DOCUMENT)}, headers=auth(uid))).status_code == 403
    await db.user.update(where={'id':uid},data={'role':'admin'})
    assert (await client.post(path, files={"file": ("a.txt", b"")}, headers=auth(uid, "admin"))).status_code == 400
    too_large = b"x" * (2 * 1024 * 1024 + 1)
    assert (await client.post(path, files={"file": ("a.txt", too_large)}, headers=auth(uid, "admin"))).status_code == 413
    foreign_path = f"/admin-api/agent-templates/{agents[0]}/knowledge/from-document"
    assert (await client.post(foreign_path, files={"file": ("a.txt", DOCUMENT)}, headers=auth(uid, "admin"))).status_code == 404
    embedding.assert_not_awaited()
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", spaces[-1]) == []


@pytest.mark.parametrize("failure", ["origin", "memory", "vector", "audit", "evidence"])
async def test_document_failure_rolls_back_whole_batch_and_retry_is_complete(importing, failure):
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    table = {"origin": "memory_input_origins", "memory": "memories_ai", "vector": "memory_embeddings",
             "audit": "memory_changelogs", "evidence": "memory_evidence_links"}[failure]
    predicate = (f"EXISTS(SELECT 1 FROM memories_ai WHERE id=NEW.memory_id AND workspace_id='{spaces[-1]}')"
                 if failure == "vector" else f"NEW.workspace_id='{spaces[-1]}'")
    async with reject_write(db, table, predicate):
        with pytest.raises((DataError, RawQueryError), match="synthetic input failure"):
            await append()
    assert await db.aimemory.count(where={"workspaceId": spaces[-1]}) == 0
    assert await db.memorychangelog.count(where={"workspaceId": spaces[-1]}) == 0
    for relation in ("memory_input_origins", "memory_evidence_links"):
        assert await db.query_raw(f"SELECT id FROM {relation} WHERE workspace_id=$1", spaces[-1]) == []
    repo.invalidate_scope.assert_not_awaited()
    knowledge.after_memory_changelog.assert_not_awaited()
    assert (await append())["stored"] > 1


async def test_concurrent_document_import_is_atomic_and_archived_duplicates_are_retained(importing):
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    results = await asyncio.wait_for(asyncio.gather(append(), append()), timeout=15)
    expected, _ = parse_knowledge_document_snapshot(DOCUMENT)
    assert sorted(result["stored"] for result in results) == [0, len(expected)]
    row = await db.aimemory.find_first(where={"workspaceId": spaces[-1]})
    await db.aimemory.update(where={"id": row.id}, data={"isArchived": True})
    assert (await append())["stored"] == 0
    assert (await db.aimemory.find_unique(where={"id": row.id})).isArchived
    assert len(await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", spaces[-1])) == 1


@pytest.mark.parametrize("change", ["disabled_owner", "archived_agent", "archived_workspace", "rebound_workspace"])
async def test_import_scope_changed_during_embedding_writes_nothing(importing, change):
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    async def changed(_):
        if change == "disabled_owner":
            await db.user.update(where={"id": users[-1]}, data={"status": "disabled"})
        elif change == "archived_agent":
            await db.aiagent.update(where={"id": agents[-1]}, data={"status": "archived"})
        elif change == "archived_workspace":
            await db.chatworkspace.update(where={"id": spaces[-1]}, data={"status": "archived"})
        else:
            await db.chatworkspace.update(where={"id": spaces[-1]}, data={"agentId": agents[1]})
        return VECTOR
    embedding.side_effect = changed
    with pytest.raises(ValueError, match="scope_changed"):
        await append()
    assert await db.aimemory.count(where={"workspaceId": spaces[-1]}) == 0
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", spaces[-1]) == []


async def test_wrong_snapshot_and_failed_embedding_never_leave_unattributed_memory(importing):
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    items, origin = parse_knowledge_document_snapshot(DOCUMENT)
    with pytest.raises(ValueError, match="snapshot_mismatch"):
        await append(items=[replace(items[0], summary="mismatched summary"), *items[1:]])
    embedding.assert_not_awaited()
    embedding.side_effect = RuntimeError("synthetic embedding provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await append()
    assert await db.aimemory.count(where={"workspaceId": spaces[-1]}) == 0
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", spaces[-1]) == []


async def test_all_duplicate_new_document_does_not_create_a_misleading_receipt(importing):
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    await append()
    # Different upload bytes, unchanged parsed facts. The document was received
    # but created no new memory, so it must not fabricate a fresh memory origin.
    assert (await append(DOCUMENT + b"\n"))["stored"] == 0
    rows = await db.query_raw("SELECT source_version FROM memory_input_origins WHERE workspace_id=$1", spaces[-1])
    _, initial = parse_knowledge_document_snapshot(DOCUMENT)
    assert rows == [{"source_version": initial.version}]


async def test_historical_duplicate_rows_keep_both_import_queries_bounded(importing, monkeypatch):
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    items, origin = parse_knowledge_document_snapshot(DOCUMENT)
    await db.aimemory.create_many(data=[{
        "userId": users[-1], "workspaceId": spaces[-1], "content": items[0].summary,
        "provenance": "knowledge_seed", "level": 1, "importance": .86,
        "isArchived": index % 2 == 0,
    } for index in range(205)])
    returned_rows = []

    class QueryProbe:
        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, name):
            return getattr(self.actual, name)

        async def query_raw(self, query, *args):
            result = await self.actual.query_raw(query, *args)
            if "FROM memories_ai" in query and "content=ANY" in query:
                returned_rows.append(len(result))
            return result

        @asynccontextmanager
        async def tx(self, **kwargs):
            async with self.actual.tx(**kwargs) as tx:
                yield QueryProbe(tx)

    monkeypatch.setattr(knowledge, "db", QueryProbe(db))
    response = await append()
    assert response["stored"] == len(items) - 1 and response["skipped_duplicates"] == 1
    # Both the preflight and the locked transaction return unique input facts,
    # even when historical duplicates outnumber the entire upload allowance.
    assert returned_rows == [1, 1]
    assert await db.aimemory.count(where={"workspaceId": spaces[-1], "content": items[0].summary}) == 205
    assert len(await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", spaces[-1])) == 1
    assert len(await db.query_raw("SELECT id FROM memory_evidence_links WHERE workspace_id=$1", spaces[-1])) == len(items) - 1


async def test_maximum_200_item_document_commits_complete_batch(importing, record_testsuite_property):
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    await db.user.update(where={'id':uid},data={'role':'admin'})
    raw = ("合成容量资料\n" + "\n".join(f"项目{i:03d}：内容{i:03d}" for i in range(200))).encode()
    expected, _ = parse_knowledge_document_snapshot(raw)
    assert len(expected) == 200
    started = perf_counter()
    response = await client.post(f"/admin-api/agent-templates/{agents[-1]}/knowledge/from-document",
        files={"file": ("synthetic-maximum.txt", raw)}, headers=auth(uid, "admin"))
    elapsed = perf_counter() - started
    record_testsuite_property("knowledge_200_import_seconds_local_sample", round(elapsed, 4))
    record_testsuite_property("knowledge_200_sample_scope", "owned loopback PG; synthetic embedding; not production SLA")
    assert response.status_code == 200, response.text
    assert response.json()["parsed"] == response.json()["stored"] == 200
    assert response.json()["skipped_duplicates"] == 0 and embedding.await_count == 200
    rows = await db.aimemory.find_many(where={"workspaceId": spaces[-1]})
    assert len(rows) == 200 and {row.content for row in rows} == {item.summary for item in expected}
    assert all(row.valueUpdatedAt is not None and row.valueUpdatedAt == row.statementTime for row in rows)
    assert all((row.type, row.level, row.mainCategory, row.subCategory, row.importance, row.provenance) ==
               ("life", 1, "生活", "工作", .86, "knowledge_seed") for row in rows)
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",
                              [row.id for row in rows]))[0]["n"] == 200
    assert await db.memorychangelog.count(where={"workspaceId": spaces[-1], "operation": "insert"}) == 200
    assert len(await db.query_raw("SELECT id FROM memory_evidence_links WHERE workspace_id=$1", spaces[-1])) == 200
    assert len(await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", spaces[-1])) == 1


async def test_append_clone_and_queued_persona_init_have_no_row_lock_cycle(importing, monkeypatch):
    """Reproduce the former three-party cycle with real PostgreSQL locks.

    Clone holds workspace SHARE, append holds persona SHARE, then init queues
    for exclusive persona and clone queues behind it. Append must still finish
    under clone's workspace SHARE so init and clone can both drain.
    """
    o, append, client, embedding = importing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    persona_held, continue_append = asyncio.Event(), asyncio.Event()

    class TransactionProbe:
        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, name):
            return getattr(self.actual, name)

        async def query_raw(self, query, *args):
            result = await self.actual.query_raw(query, *args)
            if "pg_advisory_xact_lock_shared" in query:
                persona_held.set()
                await asyncio.wait_for(continue_append.wait(), timeout=10)
            return result

    class DatabaseProbe:
        def __getattr__(self, name):
            return getattr(db, name)

        @asynccontextmanager
        async def tx(self, **kwargs):
            async with db.tx(**kwargs) as actual:
                yield TransactionProbe(actual)

    monkeypatch.setattr(knowledge, "db", DatabaseProbe())

    async def wait_until_queued(pid):
        for _ in range(200):
            locks = await db.query_raw(
                "SELECT 1 FROM pg_locks WHERE pid=$1 AND locktype='advisory' AND NOT granted", pid,
            )
            if locks:
                return
            await asyncio.sleep(.01)
        pytest.fail("Synthetic advisory waiter did not reach the expected lock queue")

    jobs = []
    try:
        async with db.tx(timeout=timedelta(seconds=20)) as clone:
            await clone.query_raw("SELECT id FROM chat_workspaces WHERE id=$1 FOR SHARE", spaces[-1])
            clone_pid = (await clone.query_raw("SELECT pg_backend_pid() AS pid"))[0]["pid"]
            jobs.append(asyncio.create_task(append()))
            await asyncio.wait_for(persona_held.wait(), timeout=5)
            init_pid_ready = asyncio.Event()
            init_pid = None

            async def initialize():
                nonlocal init_pid
                async with db.tx(timeout=timedelta(seconds=20)) as init:
                    init_pid = (await init.query_raw("SELECT pg_backend_pid() AS pid"))[0]["pid"]
                    init_pid_ready.set()
                    await init.query_raw(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1,0))::text AS held",
                        "persona-init:" + spaces[-1],
                    )

            jobs.append(asyncio.create_task(initialize()))
            await asyncio.wait_for(init_pid_ready.wait(), timeout=5)
            await wait_until_queued(init_pid)
            jobs.append(asyncio.create_task(clone.query_raw(
                "SELECT pg_advisory_xact_lock_shared(hashtextextended($1,0))::text AS held",
                "persona-init:" + spaces[-1],
            )))
            await wait_until_queued(clone_pid)
            continue_append.set()
            results = await asyncio.wait_for(asyncio.gather(*jobs), timeout=8)
            assert results[0]["stored"] > 1
    finally:
        continue_append.set()
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)


@pytest.fixture
async def editing(origins, monkeypatch):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    for module in (manual_edit, repair_queue, memories, workspaces):
        monkeypatch.setattr(module, "db", db)
    await db.aiagent.update(where={"id": agents[0]}, data={"status": "active"})
    admin = await db.user.create(data={"username": "input-admin-" + uuid4().hex, "role": "admin"})
    users.append(admin.id)
    embedding = AsyncMock(return_value=VECTOR)
    monkeypatch.setattr(manual_edit, "generate_embedding", embedding)
    monkeypatch.setattr(manual_edit, "after_memory_changelog", AsyncMock())
    monkeypatch.setattr(repo, "invalidate_scope", AsyncMock())
    repairs = []

    async def target(side="user", *, with_origin=True):
        model = db.aimemory if side == "ai" else db.usermemory
        row = await model.create(data={"userId": uid, "workspaceId": wid,
            "content": "合成的旧记忆", "importance": .63, "level": 2})
        await db.execute_raw("INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector)",
                             row.id, "[" + ",".join(["0.004"] * 1024) + "]")
        if with_origin:
            await bind(evidence.EvidenceSource("message", msg.id), side=side, memory_id=row.id)
        return row

    async def queue(row, side="user"):
        item = await repair_queue.create_memory_repair_item(
            source_type="synthetic_input_edit", source_id=uuid4().hex, user_id=uid,
            workspace_id=wid, agent_id=agents[0], memory_id=row.id, memory_source=side,
            reason="synthetic repair", suggested_action="edit_memory",
        )
        repairs.append(item["id"])
        return item

    application = FastAPI()
    application.include_router(memories.router)
    application.include_router(memory_repairs.router)
    try:
        async with AsyncClient(transport=ASGITransport(app=application, raise_app_exceptions=False),
                               base_url="http://isolated.test") as client:
            yield origins, target, queue, client, embedding, admin.id
    finally:
        if repairs:
            await db.execute_raw("DELETE FROM memory_repair_items WHERE id=ANY($1::text[])", repairs)


@pytest.mark.parametrize("side,actor_role", [("user", "user"), ("ai", "user"), ("user", "admin"), ("ai", "admin")])
async def test_public_edit_is_atomic_and_preserves_previous_origin_on_old_version(editing, side, actor_role):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    row = await target(side)
    actor_id = admin_id if actor_role == "admin" else uid
    response = await client.patch(f"/memories/{row.id}", json={"content": "合成的新记忆"}, headers=auth(actor_id, actor_role))
    assert response.status_code == 200, response.text
    assert response.json()["content"] == "合成的新记忆"
    result = await evidence_read.memory_evidence_detail(user_id=uid, workspace_id=wid, side=side, memory_id=row.id)
    previous = next(item for item in result["items"] if item["source_kind"] == "message")
    current = next(item for item in result["items"] if item["source_kind"] == "input")
    assert previous["current_content"] is False and previous["content_version"] == evidence.content_version(row.content)
    assert current["current_content"] and current["source_status"] == actor_role + "_edit"
    assert current["relation"] == "recorded_from" and current["source_version"]
    assert actor_id not in str(current) and "合成的旧记忆" not in str(result)
    origin = (await db.query_raw("SELECT payload_text FROM memory_input_origins WHERE id=$1", current["source_ref"]))[0]
    private = json.loads(origin["payload_text"])["input"]
    assert private["actor_id"] == actor_id and private["actor_role"] == actor_role
    assert private["previous_version"] == evidence.content_version(row.content)
    actual = await (db.aimemory if side == "ai" else db.usermemory).find_unique(where={"id": row.id})
    assert (actual.importance, actual.level, actual.currentScore, actual.provenance) == (
        row.importance, row.level, row.currentScore, row.provenance,
    )
    assert await db.memorychangelog.count(where={"memoryId": row.id, "operation": "user_edit"}) == 1
    stored_vector = (await db.query_raw("SELECT embedding::text AS value FROM memory_embeddings WHERE memory_id=$1", row.id))[0]
    assert json.loads(stored_vector["value"]) == pytest.approx(VECTOR)


async def test_legacy_edit_gains_only_current_input_origin_and_noop_does_not_invent_history(editing):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    row = await target(with_origin=False)
    for payload in ({}, {"content": row.content}):
        response = await client.patch(f"/memories/{row.id}", json=payload, headers=auth(uid))
        assert response.status_code == 200
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid) == []
    embedding.assert_not_awaited()
    assert (await evidence_read.memory_evidence_detail(user_id=uid, workspace_id=wid, side="user", memory_id=row.id))["state"] == "historical_unknown"
    response = await client.patch(f"/memories/{row.id}", json={"content": "新人工输入"}, headers=auth(uid))
    assert response.status_code == 200
    result = await evidence_read.memory_evidence_detail(user_id=uid, workspace_id=wid, side="user", memory_id=row.id)
    assert len(result["items"]) == 1 and result["items"][0]["source_status"] == "user_edit"


async def test_repair_metadata_only_resolves_atomically_without_fabricating_input_origin(editing):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    row = await target()
    item = await queue(row)
    response = await client.post(f"/admin-api/memory-repairs/{item['id']}/actions",
        json={"action": "edit_memory", "content": row.content, "level": 3, "importance": .4},
        headers=auth(admin_id, "admin"))
    assert response.status_code == 200, response.text
    assert response.json()["repair_item"]["status"] == "resolved"
    actual = await db.usermemory.find_unique(where={"id": row.id})
    assert actual.content == row.content and actual.level == 3 and actual.importance == .4
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid) == []
    current = await evidence_read.memory_evidence_detail(user_id=uid, workspace_id=wid, side="user", memory_id=row.id)
    assert len(current["items"]) == 1 and current["items"][0]["source_kind"] == "message"
    assert current["items"][0]["current_content"]
    embedding.assert_not_awaited()


async def test_repair_content_edit_resolves_with_admin_source_and_retry_cannot_edit_again(editing):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    row = await target("ai")
    item = await queue(row, "ai")
    path = f"/admin-api/memory-repairs/{item['id']}/actions"
    payload = {"action": "edit_memory", "content": "人工更正的合成资料", "reason": "合成核对备注"}
    response = await client.post(path, json=payload, headers=auth(admin_id, "admin"))
    assert response.status_code == 200, response.text
    assert response.json()["repair_item"]["status"] == "resolved"
    actual = await db.aimemory.find_unique(where={"id": row.id})
    assert actual.content == payload["content"]
    assert (actual.level, actual.importance, actual.provenance) == (row.level, row.importance, row.provenance)
    result = await evidence_read.memory_evidence_detail(user_id=uid, workspace_id=wid, side="ai", memory_id=row.id)
    current = next(source for source in result["items"] if source["current_content"])
    assert current["source_status"] == "admin_edit" and current["relation"] == "recorded_from"
    snapshot = (await db.query_raw("SELECT payload_text FROM memory_input_origins WHERE id=$1", current["source_ref"]))[0]
    private = json.loads(snapshot["payload_text"])["input"]
    assert private["repair_item_id"] == item["id"] and private["reason"] == payload["reason"]
    assert private["actor_id"] == admin_id and private["previous_content"] == row.content
    assert await db.memorychangelog.count(where={"memoryId": row.id, "operation": "repair_edit"}) == 1
    assert (await client.post(path, json={**payload, "content": "重复提交不应覆盖"}, headers=auth(admin_id, "admin"))).status_code == 409
    assert (await db.aimemory.find_unique(where={"id": row.id})).content == payload["content"]


async def test_repair_changed_during_embedding_preserves_other_admin_changes(editing):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    row = await target()
    item = await queue(row)
    async def changed(_):
        await db.execute_raw("UPDATE memory_repair_items SET resolution_note=$2 WHERE id=$1",
                             item["id"], "另一管理员的新备注")
        return VECTOR
    embedding.side_effect = changed
    response = await client.post(f"/admin-api/memory-repairs/{item['id']}/actions",
        json={"action": "edit_memory", "content": "基于旧队列状态的修改"}, headers=auth(admin_id, "admin"))
    assert response.status_code == 409, response.text
    latest = await repair_queue.get_memory_repair_item(item["id"])
    assert latest["status"] == "open" and latest["resolution_note"] == "另一管理员的新备注"
    assert (await db.usermemory.find_unique(where={"id": row.id})).content == row.content
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid) == []
    assert await db.memorychangelog.count(where={"memoryId": row.id}) == 0


@pytest.mark.parametrize("fail_transaction", [False, True])
async def test_knowledge_edit_invalidates_exact_live_cache_only_after_commit(editing, monkeypatch, fail_transaction):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    url = os.environ.get("PROACTIVE_E2E_REDIS_URL", "")
    if not url:
        pytest.skip("Knowledge cache requires isolated Redis")
    assert urlsplit(url).hostname in {"localhost", "127.0.0.1"} and urlsplit(url).path == "/14"
    redis = Redis.from_url(url, decode_responses=True)
    monkeypatch.setattr(knowledge_hits, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(knowledge_hits, "db", db)
    row = await target("ai")
    await db.aimemory.update(where={"id": row.id}, data={"provenance": "knowledge_seed"})
    cache_key = knowledge_hits.knowledge_rows_cache_key(wid)
    foreign_key = knowledge_hits.knowledge_rows_cache_key(spaces[-1])
    try:
        before = await knowledge_hits.load_knowledge_rows(wid)
        assert before == [{"id": row.id, "content": row.content}]
        cached_before = await redis.get(cache_key)
        assert json.loads(cached_before) == before
        await redis.set(foreign_key, "foreign-scope-cache", ex=60)
        async def edit():
            return await client.patch(f"/memories/{row.id}", json={"content": "已编辑的知识资料"}, headers=auth(uid))
        if fail_transaction:
            async with reject_write(db, "memory_evidence_links", f"NEW.memory_id='{row.id}'"):
                response = await edit()
            assert response.status_code == 500, response.text
            assert await redis.get(cache_key) == cached_before
            assert await knowledge_hits.load_knowledge_rows(wid) == before
        else:
            response = await edit()
            assert response.status_code == 200, response.text
            assert await redis.get(cache_key) is None
            after = await knowledge_hits.load_knowledge_rows(wid)
            assert after == [{"id": row.id, "content": "已编辑的知识资料"}]
            assert json.loads(await redis.get(cache_key)) == after
        assert await redis.get(foreign_key) == "foreign-scope-cache"
    finally:
        await redis.delete(cache_key, foreign_key)
        await redis.aclose()


@pytest.mark.parametrize("route,failure", [
    (route, failure) for route in ("public", "repair")
    for failure in ("memory", "vector", "origin", "audit", "evidence")
] + [("repair", "repair_resolution")])
async def test_edit_write_failure_keeps_old_memory_vector_sources_and_open_repair(editing, route, failure):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    row = await target()
    item = await queue(row) if route == "repair" else None
    before_sources = await db.query_raw("SELECT id FROM memory_evidence_links WHERE memory_id=$1", row.id)
    before_vector = await db.query_raw("SELECT embedding::text AS value FROM memory_embeddings WHERE memory_id=$1", row.id)
    table = {"memory": "memories_user", "vector": "memory_embeddings", "origin": "memory_input_origins",
             "audit": "memory_changelogs", "evidence": "memory_evidence_links", "repair_resolution": "memory_repair_items"}[failure]
    predicate = (f"NEW.id='{row.id}'" if failure == "memory" else
                 f"NEW.id='{item['id']}'" if failure == "repair_resolution" else
                 f"NEW.workspace_id='{wid}'" if failure == "origin" else f"NEW.memory_id='{row.id}'")
    async with reject_write(db, table, predicate, operation="UPDATE" if failure in {"memory", "repair_resolution"} else "INSERT"):
        if route == "public":
            response = await client.patch(f"/memories/{row.id}", json={"content": "失败不能保留的新正文"}, headers=auth(uid))
        else:
            response = await client.post(f"/admin-api/memory-repairs/{item['id']}/actions",
                json={"action": "edit_memory", "content": "失败不能保留的新正文"}, headers=auth(admin_id, "admin"))
        assert response.status_code == 500, response.text
    assert (await db.usermemory.find_unique(where={"id": row.id})).content == row.content
    assert await db.query_raw("SELECT embedding::text AS value FROM memory_embeddings WHERE memory_id=$1", row.id) == before_vector
    assert await db.query_raw("SELECT id FROM memory_evidence_links WHERE memory_id=$1", row.id) == before_sources
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid) == []
    assert await db.memorychangelog.count(where={"memoryId": row.id}) == 0
    if item:
        assert (await repair_queue.get_memory_repair_item(item["id"]))["status"] == "open"
    repo.invalidate_scope.assert_not_awaited()
    manual_edit.after_memory_changelog.assert_not_awaited()


async def test_two_public_edits_from_same_version_accept_only_one_complete_change(editing):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    row = await target()
    reached = 0
    ready = asyncio.Event()
    async def wait_for_both(_):
        nonlocal reached
        reached += 1
        if reached == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=5)
        return VECTOR
    embedding.side_effect = wait_for_both
    responses = await asyncio.wait_for(asyncio.gather(*(
        client.patch(f"/memories/{row.id}", json={"content": content}, headers=auth(uid))
        for content in ("并发修改甲", "并发修改乙")
    )), timeout=15)
    assert sorted(response.status_code for response in responses) == [200, 409]
    winner = next(response.json()["content"] for response in responses if response.status_code == 200)
    assert (await db.usermemory.find_unique(where={"id": row.id})).content == winner
    assert len(await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid)) == 1
    assert await db.memorychangelog.count(where={"memoryId": row.id}) == 1
    assert len(await db.query_raw("SELECT id FROM memory_evidence_links WHERE memory_id=$1", row.id)) == 2


@pytest.mark.parametrize("change", ["user_generation", "workspace_generation", "agent_generation", "actor_role", "actor_archived", "target_deleted"])
async def test_edit_scope_actor_or_target_changed_during_embedding_is_rejected(editing, change):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    row = await target()
    async def changed(_):
        if change.endswith("_generation"):
            table, identity = {"user_generation": ("users", uid), "workspace_generation": ("chat_workspaces", wid),
                               "agent_generation": ("ai_agents", agents[0])}[change]
            await db.execute_raw(f"UPDATE {table} SET execution_generation=$2 WHERE id=$1", identity, uuid4().hex)
        elif change == "actor_role":
            await db.user.update(where={"id": uid}, data={"role": "admin"})
        elif change == "actor_archived":
            await db.execute_raw("UPDATE users SET archived_at=CURRENT_TIMESTAMP WHERE id=$1", uid)
        else:
            await db.usermemory.delete(where={"id": row.id})
        return VECTOR
    embedding.side_effect = changed
    response = await client.patch(f"/memories/{row.id}", json={"content": "不得落库的新输入"}, headers=auth(uid))
    assert response.status_code in {403, 404, 409}, response.text
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid) == []
    assert await db.memorychangelog.count(where={"memoryId": row.id}) == 0
    if change != "target_deleted":
        assert (await db.usermemory.find_unique(where={"id": row.id})).content == row.content


async def test_public_and_repair_permissions_and_same_id_collision_fail_closed(editing):
    o, target, queue, client, embedding, admin_id = editing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    row = await target()
    item = await queue(row)
    payload = {"content": "无权编辑"}
    assert (await client.patch(f"/memories/{row.id}", json=payload)).status_code == 401
    assert (await client.patch(f"/memories/{row.id}", json=payload, headers=auth(users[1]))).status_code == 403
    repair_path = f"/admin-api/memory-repairs/{item['id']}/actions"
    assert (await client.post(repair_path, json={"action": "edit_memory", **payload}, headers=auth(uid))).status_code == 403
    # A stale elevated JWT cannot bypass the live actor-role check.
    response = await client.post(repair_path, json={"action":"edit_memory", **payload}, headers=auth(uid, "admin"))
    assert response.status_code == 403
    # The legacy fixture deliberately holds user and AI memories with one ID.
    response = await client.patch(f"/memories/{mid}", json=payload, headers=auth(uid))
    assert response.status_code == 409 and response.json()["detail"] == "memory_id_is_ambiguous"
    assert (await db.usermemory.find_unique(where={"id": mid})).content == "合成地点事实"
    assert (await db.aimemory.find_unique(where={"id": mid})).content == "合成地点事实"
    assert await db.query_raw("SELECT id FROM memory_input_origins WHERE workspace_id=$1", wid) == []
