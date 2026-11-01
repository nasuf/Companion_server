"""Real source/version/CAS and physical erasure, not mocked persistence."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from app.services.memory import temporal_history
from app.services.memory.evidence import EvidenceSource, content_version
from app.services.memory.recording.temporal_sources import resolve_fact_clock
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_receipts_postgres import ledger


@pytest.mark.parametrize(
    "changes",
    [
        {"side": "third_party"},
        {"revision_ids": []},
        {"revision_ids": [True]},
        {"revision_ids": [2**63]},
        {"limit": 51},
        {"known_at": datetime(2026, 1, 1)},
        {"at": "2026-01-01"},
    ],
)
async def test_assertion_reader_rejects_unbounded_or_ambiguous_requests(changes):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.memory.temporal_assertions import read_assertions

    db = SimpleNamespace(
        query_raw=AsyncMock(side_effect=AssertionError("must not query"))
    )
    args = dict(
        database=db,
        user_id="u",
        workspace_id="w",
        side="user",
        memory_id="m",
        revision_ids=[1],
    )
    with pytest.raises(ValueError):
        await read_assertions(**{**args, **changes})
    db.query_raw.assert_not_called()


async def test_assertion_reader_marks_corrupt_rows_unknown_without_exposing_them():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.memory.temporal_assertions import read_assertions

    db = SimpleNamespace(
        query_raw=AsyncMock(
            return_value=[{"assertion": {"state": "asserted", "version": 99}}]
        )
    )
    value = await read_assertions(
        database=db,
        user_id="u",
        workspace_id="w",
        side="user",
        memory_id="m",
        revision_ids=[1],
    )
    assert (
        not value["items"] and value["state"] == "not_collected_or_source_unavailable"
    )


@pytest.fixture
async def validity(origins):
    db, uid, wid, mid, msg, bind, *_ = origins
    text = "我从2026年8月1日到2026年8月6日住在旅馆。"
    at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    msg = await db.message.update(
        where={"id": msg.id}, data={"content": text, "createdAt": at}
    )
    raw = EvidenceSource(
        "message",
        msg.id,
        expected_version=content_version(text),
        expected_role="user",
        expected_statement_time=at,
        input_text=text,
        relation="extracted_from",
    )
    fact = {
        "source_indices": [1],
        "temporal_assertion": {
            "kind": "state",
            "status": "completed",
            "source_index": 1,
            "quote": text,
            "valid_from_quote": "2026年8月1日",
            "valid_to_quote": "2026年8月6日",
        },
    }
    clock = resolve_fact_clock(fact, [raw])
    assert clock.basis["validity"]["state"] == "asserted"

    async def read(**kw):
        return await temporal_history.read_history(
            user_id=uid, workspace_id=wid, side="user", memory_id=mid, **kw
        )

    return origins, clock.bind([raw])[0], read


async def test_assertion_is_atomic_scoped_idempotent_and_source_cas(validity):
    f, source, read = validity
    db, uid, wid, mid, _, bind, *_ = f
    assert await bind(source) == 1
    assert await bind(source) == 0
    rows = (await read())["items"][0]["assertions"]
    assert len(rows) == 1 and datetime.fromisoformat(
        rows[0]["validity"]["valid_to"]
    ) == datetime(2026, 8, 6, 16, tzinfo=timezone.utc)
    assert not (await read(known_at=datetime(2026, 1, 1, tzinfo=timezone.utc)))["items"]
    within = await read(at=datetime(2026, 8, 5, tzinfo=timezone.utc))
    assert within["items"][0]["assertions"][0]["membership"] == "inside"
    with pytest.raises(ValueError):
        await bind(source, side="ai")
    await db.message.update(where={"id": source.ref}, data={"content": "原消息已编辑"})
    assert not (await read())["items"][0]["assertions"]
    with pytest.raises(ValueError, match="version_changed"):
        await bind(source)


async def test_concurrent_same_source_and_version_retraction(validity):
    f, source, read = validity
    db, uid, wid, mid, _, bind, *_ = f
    results = await asyncio.gather(bind(source), bind(source))
    assert sum(results) == 1
    before = await read()
    old = before["items"][0]["assertions"][0]
    await db.usermemory.update(where={"id": mid}, data={"content": "更正后的旅馆记录"})
    after = await read()
    assert not after["items"][0]["assertions"]
    assert after["items"][1]["assertions"][0]["retracted_at"]
    assert after["items"][1]["assertions"][0]["validity"] == old["validity"]


async def test_delete_source_and_memory_cascades_no_orphan_assertions(validity):
    f, source, read = validity
    db, uid, wid, mid, _, bind, *_ = f
    await bind(source)
    await db.message.delete(where={"id": source.ref})
    assert not (await read())["items"][0]["assertions"]
    assert not await db.query_raw(
        "SELECT id FROM memory_temporal_assertions WHERE memory_id=$1", mid
    )
    with pytest.raises(ValueError, match="not_found"):
        await bind(source)


async def test_failure_rolls_back_evidence_and_assertion(validity):
    f, source, read = validity
    db, uid, wid, mid, _, bind, *_ = f
    with pytest.raises(Exception):
        async with db.tx() as tx:
            await bind(source, database=tx)
            await tx.execute_raw("SELECT 1/0")
    assert not (await read())["items"][0]["assertions"]
    assert not await db.query_raw(
        "SELECT id FROM memory_evidence_links WHERE memory_id=$1", mid
    )


async def test_asof_api_requires_timezone_and_scope(validity):
    from app.api.admin import memory_repairs
    from app.services.auth import create_jwt

    f, source, read = validity
    db, uid, wid, mid, _, bind, *_ = f
    await bind(source)
    await db.user.update(where={"id": uid}, data={"role": "admin"})
    app = FastAPI()
    app.include_router(memory_repairs.router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = f"/admin-api/memory-repairs/time-history/user/{mid}"
        params = {
            "user_id": uid,
            "workspace_id": wid,
            "at": "2026-08-03T00:00:00+08:00",
        }
        assert (await client.get(url, params=params)).status_code == 401
        headers = {"Authorization": "Bearer " + create_jwt(uid, "admin")}
        response = await client.get(url, params=params, headers=headers)
        assert response.status_code == 200
        assert response.json()["items"][0]["assertions"][0]["membership"] == "inside"
        params["at"] = "2026-08-03T00:00:00"
        assert (
            await client.get(url, params=params, headers=headers)
        ).status_code == 400


async def test_actual_chat_reader_loads_old_validity_and_rechecks_sources(validity):
    from app.services.time_domain.validity_context import load_validity_context
    from app.services.time_domain.validity_context import assertions_still_available
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.time_domain.recall import recall_payload
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    f, source, read = validity
    db, uid, wid, mid, _, bind, *_ = f
    await bind(source)
    await db.usermemory.update(where={"id": mid}, data={"content": "用户目前住在学校"})
    memory = ClassifiedMemory(
        id=mid, text="用户目前住在学校", source="user", relevance="strong", score=1
    )
    question = "2026年8月3日当时我住在哪里？"
    query = resolve_temporal_query(
        question,
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )
    assert query.operation.type == "as_of" and query.status == "resolved"
    text, records, info = await load_validity_context(
        [memory], user_id=uid, workspace_id=wid, query=query, database=db
    )
    assert info["assertion_count"] == 1 and "住在旅馆" in text
    payload, reason = recall_payload(
        question, [memory], query.diagnostics(), validity_records=records
    )
    assert reason == "eligible"
    assert payload["memories"][0]["validity_membership"] == "inside"
    assert payload["memories"][0]["content"] == "合成地点事实"
    assert await assertions_still_available(
        records, [memory], user_id=uid, workspace_id=wid, query=query, database=db
    )
    ordinary = resolve_temporal_query(
        "你好", anchor=query.anchor, anchor_source="persisted_message"
    )
    assert not (
        await load_validity_context(
            [memory], user_id=uid, workspace_id=wid, query=ordinary, database=db
        )
    )[1]
    assert not (
        await load_validity_context(
            [memory], user_id=f[7][-1], workspace_id=wid, query=query, database=db
        )
    )[1]
    await db.message.delete(where={"id": source.ref})
    assert not await assertions_still_available(
        records, [memory], user_id=uid, workspace_id=wid, query=query, database=db
    )
    assert not (
        await load_validity_context(
            [memory], user_id=uid, workspace_id=wid, query=query, database=db
        )
    )[1]


async def test_pipeline_replay_freezes_validity_and_erasure_blocks_resurrection(ledger):
    from app.services.memory.recording import pipeline

    db, uid, wid, msg, args, extract = ledger
    text = "我从2026年8月1日到2026年8月6日住在旅馆。"
    await db.message.update(
        where={"id": msg.id},
        data={"content": text, "createdAt": datetime(2026, 9, 1, tzinfo=timezone.utc)},
    )
    args = {**args, "new_conversation": "user: " + text}
    fact = extract.return_value["memories"][0]
    fact.update(
        content="用户从2026年8月1日到2026年8月6日住在旅馆",
        temporal_assertion={
            "kind": "state",
            "status": "completed",
            "source_index": 1,
            "quote": text,
            "valid_from_quote": "2026年8月1日",
            "valid_to_quote": "2026年8月6日",
        },
    )
    extract.return_value["memories"] = [fact]
    ids = await pipeline.process_memory_pipeline(**args)
    assert len(ids) == 1
    rows = await db.query_raw(
        "SELECT id,assertion FROM memory_temporal_assertions WHERE user_id=$1", uid
    )
    assert len(rows) == 1 and rows[0]["assertion"]["state"] == "asserted"
    assert await pipeline.process_memory_pipeline(**args) == ids
    assert extract.await_count == 1
    await db.usermemory.delete(where={"id": ids[0]})
    assert await pipeline.process_memory_pipeline(**args) == []
    assert not await db.query_raw(
        "SELECT id FROM memory_temporal_assertions WHERE user_id=$1", uid
    )


async def test_replacement_history_is_scoped_bounded_and_cycle_safe(validity):
    from uuid import uuid4
    from app.services.time_domain.validity_context import load_validity_context
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    f, source, _ = validity
    db, uid, wid, mid, _, bind, *_ = f
    await bind(source)
    replacement = await db.usermemory.create(
        data={
            "id": uuid4().hex,
            "userId": uid,
            "workspaceId": wid,
            "content": "现在住在河畔",
        }
    )
    await db.usermemory.update(where={"id": mid}, data={"isArchived": True})

    async def link(old, new, owner=uid, workspace=wid):
        await db.execute_raw(
            """INSERT INTO memory_quality_states
            (memory_id,memory_source,user_id,workspace_id,superseded_by_memory_id)
            VALUES($1,'user',$2,$3,$4) ON CONFLICT(memory_id,memory_source)
            DO UPDATE SET superseded_by_memory_id=$4,user_id=$2,workspace_id=$3""",
            old,
            owner,
            workspace,
            new,
        )

    await link(mid, replacement.id)
    memory = ClassifiedMemory(
        id=replacement.id,
        text=replacement.content,
        source="user",
        relevance="strong",
        score=1,
    )
    query = resolve_temporal_query(
        "2026年8月3日当时我住在哪里？",
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )

    async def load():
        return await load_validity_context(
            [memory], user_id=uid, workspace_id=wid, query=query, database=db
        )

    try:
        _, records, info = await load()
        assert records[("user", replacement.id)][0]["historical_memory_id"] == mid
        assert not info["truncated"]
        with pytest.raises(Exception, match="memory_replacement_scope"):
            await link(mid, replacement.id, owner=f[7][-1])
        with pytest.raises(Exception, match="memory_replacement_scope"):
            await link(mid, replacement.id, workspace=f[9][1])
        with pytest.raises(Exception, match="memory_replacement_cycle"):
            await link(replacement.id, mid)
        # Explicit legacy corruption fixture: migrations do not rewrite old
        # edges. Readers must stay bounded even when a pre-migration cycle exists.
        async with db.tx() as tx:
            await tx.execute_raw("SET LOCAL session_replication_role='replica'")
            await tx.execute_raw(
                """INSERT INTO memory_quality_states
                (memory_id,memory_source,user_id,workspace_id,superseded_by_memory_id)
                VALUES($1,'user',$2,$3,$4)""",
                replacement.id,
                uid,
                wid,
                mid,
            )
        _, records, info = await load()
        assert len(records[("user", replacement.id)]) == 1 and info["truncated"]
        await db.usermemory.delete(where={"id": mid})
        assert not (await load())[1]
    finally:
        await db.execute_raw(
            "DELETE FROM memory_quality_states WHERE memory_id=ANY($1::text[])",
            [mid, replacement.id],
        )


async def test_old_transaction_retraction_cannot_precede_new_assertion(validity):
    f, source, read = validity
    db, uid, wid, mid, _, bind, *_ = f
    async with db.tx() as earlier:
        await earlier.query_raw("SELECT CURRENT_TIMESTAMP AS started")
        await bind(source)
        await earlier.usermemory.update(
            where={"id": mid}, data={"content": "事务启动后才更正"}
        )
    assertions = (await read())["items"][1]["assertions"]
    assert len(assertions) == 1
    assert datetime.fromisoformat(
        assertions[0]["retracted_at"]
    ) >= datetime.fromisoformat(assertions[0]["recorded_at"])


async def test_concurrent_replacement_edges_cannot_create_cycle(validity):
    from uuid import uuid4

    f, _, _ = validity
    db, uid, wid, mid, *_ = f
    other = await db.usermemory.create(
        data={
            "id": uuid4().hex,
            "userId": uid,
            "workspaceId": wid,
            "content": "另一个事实",
        }
    )

    async def edge(a, b):
        return await db.execute_raw(
            """INSERT INTO memory_quality_states
          (memory_id,memory_source,user_id,workspace_id,superseded_by_memory_id)
          VALUES($1,'user',$2,$3,$4)""",
            a,
            uid,
            wid,
            b,
        )

    try:
        results = await asyncio.gather(
            edge(mid, other.id), edge(other.id, mid), return_exceptions=True
        )
        assert sum(isinstance(r, Exception) for r in results) == 1
        assert "memory_replacement_cycle" in str(
            next(r for r in results if isinstance(r, Exception))
        )
    finally:
        await db.execute_raw(
            "DELETE FROM memory_quality_states WHERE memory_id=ANY($1::text[])",
            [mid, other.id],
        )


async def test_late_source_keeps_its_statement_clock_and_replay_cannot_rewrite_basis(
    validity,
):
    f, source, read = validity
    db, uid, wid, mid, msg, bind, *_ = f
    await bind(source)
    older_at = datetime(2026, 8, 20, tzinfo=timezone.utc)
    older = await db.message.create(
        data={
            "conversationId": msg.conversationId,
            "role": "user",
            "content": source.input_text,
            "createdAt": older_at,
        }
    )
    original = replace(
        source, ref=older.id, expected_statement_time=older_at, temporal_basis=None
    )
    fact = {
        "source_indices": [1],
        "temporal_assertion": {
            "kind": "state",
            "status": "completed",
            "source_index": 1,
            "quote": source.input_text,
            "valid_from_quote": "2026年8月1日",
            "valid_to_quote": "2026年8月6日",
        },
    }
    late = resolve_fact_clock(fact, [original]).bind([original])[0]
    await bind(late)
    assertions = (await read())["items"][0]["assertions"]
    assert len(assertions) == 2
    assert datetime.fromisoformat(assertions[0]["validity"]["statement_at"]) > older_at
    assert datetime.fromisoformat(assertions[1]["validity"]["statement_at"]) == older_at
    assert (
        assertions[0]["validity"]["valid_from"]
        == assertions[1]["validity"]["valid_from"]
    )
    before = await db.query_raw(
        "SELECT id,assertion FROM memory_temporal_assertions WHERE memory_id=$1 ORDER BY id",
        mid,
    )
    # The same evidence identity cannot replace its frozen assertion with an
    # invented interval. Either an explicit conflict or a no-op is acceptable.
    import copy

    changed = copy.deepcopy(late.temporal_basis)
    changed["validity"]["valid_from"] = "2026-07-01T00:00:00+08:00"
    try:
        await bind(replace(late, temporal_basis=changed))
    except ValueError:
        pass
    assert (
        await db.query_raw(
            "SELECT id,assertion FROM memory_temporal_assertions WHERE memory_id=$1 ORDER BY id",
            mid,
        )
        == before
    )


async def test_recent_plaintext_is_available_before_extraction_and_revalidated(origins):
    from app.services.time_domain.validity_context import (
        load_validity_context,
        assertions_still_available,
    )
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.time_domain.recall import recall_payload
    from prisma import Json

    db, uid, wid, _, msg, _, _, users, _, spaces, _ = origins
    text = "我从2026年8月1日到2026年8月6日住在松林旅馆。"
    at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    await db.message.update(
        where={"id": msg.id}, data={"content": text, "createdAt": at}
    )
    messages = [{"id": msg.id, "role": "user", "content": text, "createdAt": at}]
    question = "2026年8月3日当时我住在哪里？"
    q = resolve_temporal_query(question, anchor=at, anchor_source="persisted_message")
    args = dict(
        user_id=uid, workspace_id=wid, query=q, database=db, recent_messages=messages
    )
    _, records, info = await load_validity_context([], **args)
    assert info["recent_source_count"] == 1
    payload, reason = recall_payload(
        question, [], q.diagnostics(), validity_records=records
    )
    assert reason == "eligible" and payload["memories"][0]["content"] == text
    assert payload["memories"][0]["validity_membership"] == "not_materialized"
    assert await assertions_still_available(records, [], **args)
    for overrides in (
        {"user_id": users[-1]},
        {"workspace_id": spaces[1]},
        {"recent_messages": [{**messages[0], "content": "伪造内容"}]},
        {"recent_messages": [{**messages[0], "createdAt": at + timedelta(seconds=1)}]},
    ):
        assert not (await load_validity_context([], **{**args, **overrides}))[1]
    for meta in ({"human_takeover": True}, {"component_card": {"title": text}}):
        await db.message.update(where={"id": msg.id}, data={"metadata": Json(meta)})
        assert not await assertions_still_available(records, [], **args)
    await db.message.update(
        where={"id": msg.id}, data={"metadata": Json({}), "content": "已改写"}
    )
    assert not await assertions_still_available(records, [], **args)
    await db.message.delete(where={"id": msg.id})
    assert not await assertions_still_available(records, [], **args)


async def test_old_source_without_assertion_remains_source_backed(validity):
    from app.services.time_domain.validity_context import (
        load_validity_context,
        assertions_still_available,
    )
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.time_domain.recall import recall_payload
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    fixture, source, _ = validity
    database, uid, wid, mid, msg, bind, *_ = fixture
    old = replace(
        source,
        temporal_basis={
            k: v for k, v in source.temporal_basis.items() if k != "validity"
        },
    )
    await bind(old)
    q = resolve_temporal_query(
        "2026年8月3日当时我住在哪里？",
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )
    memories = [
        ClassifiedMemory(
            id=mid, text="合成地点事实", source="user", relevance="strong", score=1
        )
    ]
    _, records, info = await load_validity_context(
        memories, user_id=uid, workspace_id=wid, query=q, database=database
    )
    assert info["linked_source_count"] == 1 and info["assertion_count"] == 0
    payload, reason = recall_payload(
        "2026年8月3日当时我住在哪里？",
        memories,
        q.diagnostics(),
        validity_records=records,
    )
    assert (
        reason == "eligible"
        and payload["memories"][0]["evidence_type"] == "linked_statement"
    )
    assert "旅馆" in payload["memories"][0]["source_excerpt"]
    assert not await database.query_raw(
        "SELECT id FROM memory_temporal_assertions WHERE memory_id=$1", mid
    )
    assert await assertions_still_available(
        records, memories, user_id=uid, workspace_id=wid, query=q, database=database
    )
    await database.message.update(
        where={"id": source.ref}, data={"content": "已编辑原文"}
    )
    assert not await assertions_still_available(
        records, memories, user_id=uid, workspace_id=wid, query=q, database=database
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "delete_source",
        "delete_target",
        "target_edit",
        "source_time",
        "source_role",
        "conversation_delete",
        "takeover",
        "card",
    ],
)
async def test_old_linked_source_live_guards(validity, mutation):
    from prisma import Json
    from app.services.time_domain.validity_context import (
        load_validity_context,
        assertions_still_available,
    )
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    fixture, source, _ = validity
    database, uid, wid, mid, msg, bind, *_ = fixture
    old = replace(
        source,
        temporal_basis={
            k: v for k, v in source.temporal_basis.items() if k != "validity"
        },
    )
    await bind(old)
    q = resolve_temporal_query(
        "2026年8月3日当时我住在哪里？",
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )
    memories = [
        ClassifiedMemory(
            id=mid, text="合成地点事实", source="user", relevance="strong", score=1
        )
    ]
    _, records, _ = await load_validity_context(
        memories, user_id=uid, workspace_id=wid, query=q, database=database
    )
    assert records
    if mutation == "delete_source":
        await database.message.delete(where={"id": source.ref})
    elif mutation == "delete_target":
        await database.usermemory.delete(where={"id": mid})
    elif mutation == "target_edit":
        await database.usermemory.update(
            where={"id": mid}, data={"content": "已修改记忆"}
        )
    elif mutation == "source_time":
        await database.message.update(
            where={"id": source.ref},
            data={"createdAt": datetime(2026, 10, 1, tzinfo=timezone.utc)},
        )
    elif mutation == "source_role":
        await database.message.update(
            where={"id": source.ref}, data={"role": "assistant"}
        )
    elif mutation == "conversation_delete":
        await database.conversation.update(
            where={"id": msg.conversationId}, data={"isDeleted": True}
        )
    elif mutation == "takeover":
        await database.message.update(
            where={"id": source.ref}, data={"metadata": Json({"human_takeover": True})}
        )
    elif mutation == "card":
        await database.message.update(
            where={"id": source.ref},
            data={"metadata": Json({"component_card": {"text": "卡片"}})},
        )
    assert not await assertions_still_available(
        records, memories, user_id=uid, workspace_id=wid, query=q, database=database
    )


async def test_old_linked_source_is_scoped_and_only_read_for_history(validity):
    from app.services.time_domain.validity_context import load_validity_context
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    fixture, source, _ = validity
    database, uid, wid, mid, _, bind, *_ = fixture
    await bind(
        replace(
            source,
            temporal_basis={
                k: v for k, v in source.temporal_basis.items() if k != "validity"
            },
        )
    )
    memories = [
        ClassifiedMemory(
            id=mid, text="合成地点事实", source="user", relevance="strong", score=1
        )
    ]
    query = resolve_temporal_query(
        "2026年8月3日当时我住在哪里？",
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )
    args = dict(user_id=uid, workspace_id=wid, query=query, database=database)
    for overrides in ({"user_id": "other-user"}, {"workspace_id": "other-workspace"}):
        _, records, info = await load_validity_context(
            memories, **{**args, **overrides}
        )
        assert not records and info["linked_source_count"] == 0
    ordinary = resolve_temporal_query(
        "今天怎么样",
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )
    text, records, info = await load_validity_context(
        memories, **{**args, "query": ordinary}
    )
    assert not text and not records and info["linked_source_count"] == 0


async def test_old_context_source_is_not_a_fact(validity):
    from app.services.time_domain.validity_context import load_validity_context
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    fixture, source, _ = validity
    database, uid, wid, mid, _, bind, *_ = fixture
    await bind(replace(source, temporal_basis={"source_role": "context"}))
    memories = [
        ClassifiedMemory(
            id=mid, text="合成地点事实", source="user", relevance="strong", score=1
        )
    ]
    query = resolve_temporal_query(
        "2026年8月3日当时我住在哪里？",
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )
    _, records, info = await load_validity_context(
        memories, user_id=uid, workspace_id=wid, query=query, database=database
    )
    assert not records and info["linked_source_count"] == 0


async def test_assertion_source_cannot_silently_become_card(validity):
    from prisma import Json
    from app.services.time_domain.validity_context import (
        load_validity_context,
        assertions_still_available,
    )
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    fixture, source, read = validity
    database, uid, wid, mid, _, bind, *_ = fixture
    await bind(source)
    memories = [
        ClassifiedMemory(
            id=mid, text="合成地点事实", source="user", relevance="strong", score=1
        )
    ]
    query = resolve_temporal_query(
        "2026年8月3日当时我住在哪里？",
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )
    args = dict(user_id=uid, workspace_id=wid, query=query, database=database)
    _, records, _ = await load_validity_context(memories, **args)
    assert records
    await database.message.update(
        where={"id": source.ref},
        data={"metadata": Json({"component_card": {"text": source.input_text}})},
    )
    assert not await assertions_still_available(records, memories, **args)
    history = await read()
    assert history["items"] and all(not item["assertions"] for item in history["items"])
