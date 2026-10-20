"""Real PG compression transactions; only model/embedding providers are synthetic."""

import asyncio
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest
from prisma.errors import DataError, RawQueryError

from app.api.admin import memory_repairs
from app.services.memory import evidence_read
from app.services.memory.lifecycle import (
    consolidation as cons,
    consolidation_write as writer,
)
from app.services.memory.storage import repo
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_input_write_api_postgres import VECTOR, auth, reject_write


@pytest.fixture
async def compression(origins, monkeypatch):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = origins
    monkeypatch.setattr(writer, "after_memory_changelog", AsyncMock())
    monkeypatch.setattr(cons, "db", db)
    monkeypatch.setattr(
        cons,
        "get_prompt_text",
        AsyncMock(
            return_value="{owner}:{main_category}/{sub_category}\n{memory_items}"
        ),
    )
    monkeypatch.setattr(
        cons, "get_utility_model", lambda: SimpleNamespace(model_name="synthetic-model")
    )
    model = AsyncMock(return_value={"summary": "那段时间经常在午休听轻音乐"})
    monkeypatch.setattr(cons, "invoke_json", model)
    embedding = AsyncMock(return_value=VECTOR)
    monkeypatch.setattr(cons, "generate_embedding", embedding)
    monkeypatch.setattr(
        cons, "_generated_self_memory_allowed", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(repo, "invalidate_scope", AsyncMock())
    f = SimpleNamespace(
        db=db,
        uid=uid,
        wid=wid,
        users=users,
        agents=agents,
        spaces=spaces,
        convs=convs,
        model=model,
        embedding=embedding,
    )

    async def seed(side="ai", n=5, **overrides):
        ids = []
        for i in range(n):
            row = await getattr(
                db, "aimemory" if side == "ai" else "usermemory"
            ).create(
                data={
                    "userId": uid,
                    "workspaceId": wid,
                    "content": f"第{i}天午休听轻音乐",
                    "level": 3,
                    "importance": 0.3,
                    "mainCategory": "生活",
                    "subCategory": "其他",
                    "provenance": "daily_summary",
                    "createdAt": datetime.now(UTC) - timedelta(days=40),
                    **overrides,
                }
            )
            await db.execute_raw(
                "INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector)",
                row.id,
                "[" + ",".join(map(str, VECTOR)) + "]",
            )
            ids.append(row.id)
        return ids

    async def candidates(side="ai"):
        return await cons._load_candidates(source=side, user_id=uid, workspace_id=wid)

    async def prepare(side="ai"):
        return await writer.prepare_cluster(
            db,
            source=side,
            user_id=uid,
            workspace_id=wid,
            cluster=await candidates(side),
        )

    async def commit(inputs, side="ai", database=None, **kwargs):
        return await writer.commit_cluster(
            database or db,
            source=side,
            user_id=uid,
            workspace_id=wid,
            inputs=inputs,
            digest="那段时间经常在午休听轻音乐",
            embedding=VECTOR,
            prompt="PRIVATE_RENDERED_PROMPT",
            model_name="synthetic-model",
            output={"summary": "那段时间经常在午休听轻音乐"},
            **kwargs,
        )

    f.seed, f.candidates, f.prepare, f.commit = seed, candidates, prepare, commit
    yield f


@pytest.mark.parametrize("side", ["user", "ai"])
async def test_real_compression_preserves_originals_and_links_same_side_versions(
    compression, side
):
    f = compression
    ids = await f.seed(side)
    result = await cons.compress_l3_clusters_for_workspace(
        user_id=f.uid, workspace_id=f.wid
    )
    assert (
        result["digests"] == 1
        and result["compressed_rows"] == 5
        and result["failed"] == 0
    )
    digest = result["digest_ids"][0]
    table = "memories_" + side
    rows = await f.db.query_raw(f"SELECT * FROM {table} WHERE id=ANY($1::text[])", ids)
    assert len(rows) == 5 and all(r["is_archived"] for r in rows)
    stored = (await f.db.query_raw(f"SELECT * FROM {table} WHERE id=$1", digest))[0]
    assert (
        stored["level"] == 3
        and stored["importance"] == 0.3
        and stored["current_score"] == 0.3
    )
    assert stored["provenance"] == "consolidated" and not stored["is_archived"]
    evidence = await evidence_read.memory_evidence_detail(
        memory_id=digest, side=side, user_id=f.uid, workspace_id=f.wid
    )
    assert len(evidence["items"]) == 6 and evidence["state"] == "linked"
    parents = [x for x in evidence["items"] if x["source_kind"] == "memory"]
    assert {x["source_ref"] for x in parents} == set(ids)
    assert all(
        x["source_side"] == side
        and x["source_archived"]
        and x["availability"] == "available"
        for x in parents
    )
    receipt = next(x for x in evidence["items"] if x["source_kind"] == "input")
    assert (
        receipt["consolidation"]["parents"] == 5
        and receipt["consolidation"]["side"] == side
    )
    assert receipt["availability"] == "recorded_snapshot"
    private = (
        await f.db.query_raw(
            "SELECT payload_text FROM memory_input_origins WHERE id=$1",
            receipt["source_ref"],
        )
    )[0]["payload_text"]
    payload = json.loads(private)["input"]
    assert payload["model"] == "synthetic-model" and all(
        r["content"] in payload["prompt"] for r in rows
    )
    assert "prompt" not in json.dumps(evidence) and "午休听轻音乐" not in json.dumps(
        evidence, ensure_ascii=False
    )
    assert (
        await cons.compress_l3_clusters_for_workspace(user_id=f.uid, workspace_id=f.wid)
    )["digests"] == 0
    f.model.assert_awaited_once()
    logs = await f.db.query_raw(
        "SELECT * FROM memory_changelogs WHERE new_value=$1 AND operation='consolidated_into'",
        digest,
    )
    assert len(logs) == 5 and {r["memory_id"] for r in logs} == set(ids)


@pytest.mark.parametrize(
    "failure",
    ["origin", "digest", "vector", "link", "insert_audit", "parent_audit", "archive"],
)
async def test_every_write_failure_rolls_back_then_retry_completes(
    compression, failure
):
    f = compression
    ids = await f.seed()
    inputs = await f.prepare()
    table = {
        "origin": "memory_input_origins",
        "digest": "memories_ai",
        "vector": "memory_embeddings",
        "link": "memory_evidence_links",
        "insert_audit": "memory_changelogs",
        "parent_audit": "memory_changelogs",
        "archive": "memories_ai",
    }[failure]
    if failure == "vector":
        pred = f"EXISTS(SELECT 1 FROM memories_ai m WHERE m.id=NEW.memory_id AND m.workspace_id='{f.wid}' AND m.provenance='consolidated')"
    else:
        pred = f"NEW.workspace_id='{f.wid}'"
    if failure == "insert_audit":
        pred += " AND NEW.operation='insert'"
    if failure == "parent_audit":
        pred += " AND NEW.operation='consolidated_into'"
    async with reject_write(
        f.db, table, pred, operation="UPDATE" if failure == "archive" else "INSERT"
    ):
        with pytest.raises((DataError, RawQueryError), match="synthetic input failure"):
            await f.commit(inputs)
    assert (
        await f.db.aimemory.count(
            where={"workspaceId": f.wid, "provenance": "consolidated"}
        )
        == 0
    )
    assert (
        await f.db.aimemory.count(where={"id": {"in": ids}, "isArchived": False}) == 5
    )
    for name in ["memory_evidence_links", "memory_input_origins", "memory_changelogs"]:
        assert (
            await f.db.query_raw(f"SELECT id FROM {name} WHERE workspace_id=$1", f.wid)
            == []
        )
    repo.invalidate_scope.assert_not_awaited()
    assert await f.commit(inputs)


async def test_parallel_same_cluster_has_one_digest_and_receipt(compression):
    f = compression
    await f.seed()
    inputs = await f.prepare()
    results = await asyncio.gather(
        *(f.commit(inputs) for _ in range(6)), return_exceptions=True
    )
    assert sum(isinstance(r, str) for r in results) == 1
    assert (
        await f.db.aimemory.count(
            where={"workspaceId": f.wid, "provenance": "consolidated"}
        )
        == 1
    )
    assert (
        len(
            await f.db.query_raw(
                "SELECT id FROM memory_input_origins WHERE workspace_id=$1", f.wid
            )
        )
        == 1
    )
    assert (
        len(
            await f.db.query_raw(
                "SELECT id FROM memory_evidence_links WHERE workspace_id=$1", f.wid
            )
        )
        == 6
    )


@pytest.mark.parametrize(
    "change",
    [
        "edit",
        "delete",
        "archive",
        "promote",
        "taxonomy",
        "provenance",
        "user_generation",
        "agent_generation",
        "workspace_generation",
        "workspace_archive",
        "agent_city",
        "takeover",
        "ambiguous_id",
    ],
)
async def test_stale_or_foreign_inputs_cannot_archive_any_parent(compression, change):
    f = compression
    ids = await f.seed()
    inputs = await f.prepare()
    updates = {
        "edit": {"content": "改过的原始记忆"},
        "archive": {"isArchived": True},
        "promote": {"level": 2},
        "taxonomy": {"subCategory": "健康"},
        "provenance": {"provenance": "ai_authored"},
    }
    if change in updates:
        await f.db.aimemory.update(where={"id": ids[0]}, data=updates[change])
    elif change == "delete":
        await f.db.aimemory.delete(where={"id": ids[0]})
    elif change == "workspace_archive":
        await f.db.chatworkspace.update(
            where={"id": f.wid}, data={"status": "archived"}
        )
    elif change == "agent_city":
        await f.db.aiagent.update(
            where={"id": f.agents[0]}, data={"city": "changed-city"}
        )
    elif change.endswith("generation"):
        table = {
            "user_generation": "users",
            "agent_generation": "ai_agents",
            "workspace_generation": "chat_workspaces",
        }[change]
        ident = {
            "user_generation": f.uid,
            "agent_generation": f.agents[0],
            "workspace_generation": f.wid,
        }[change]
        await f.db.execute_raw(
            f"UPDATE {table} SET execution_generation=gen_random_uuid()::text WHERE id=$1",
            ident,
        )
    elif change == "takeover":
        await f.db.execute_raw(
            "INSERT INTO human_takeovers(id,user_id,agent_id,workspace_id,conversation_id,status,owner_admin_id,updated_at) VALUES($1,$2,$3,$4,$5,'active',$6,NOW())",
            uuid4().hex,
            f.uid,
            f.agents[0],
            f.wid,
            f.convs[0],
            f.users[-1],
        )
    else:
        await f.db.usermemory.create(
            data={
                "id": ids[0],
                "userId": f.uid,
                "workspaceId": f.wid,
                "content": "歧义的另一侧",
            }
        )
    with pytest.raises((ValueError, writer.MemoryWriteError)):
        await f.commit(inputs)
    assert (
        await f.db.aimemory.count(
            where={"workspaceId": f.wid, "provenance": "consolidated"}
        )
        == 0
    )
    assert (
        await f.db.query_raw(
            "SELECT id FROM memory_input_origins WHERE workspace_id=$1", f.wid
        )
        == []
    )
    assert (
        await f.db.aimemory.count(where={"id": {"in": ids[1:]}, "isArchived": False})
        == 4
    )


async def test_foreign_workspace_mixed_categories_and_candidate_edits_fail_before_model(
    compression,
):
    f = compression
    ids = await f.seed()
    cluster = await f.candidates()
    await f.db.aimemory.update(where={"id": ids[0]}, data={"workspaceId": f.spaces[1]})
    assert (
        await cons._compress_cluster(
            source="ai", user_id=f.uid, workspace_id=f.wid, cluster=cluster
        )
        is None
    )
    f.model.assert_not_awaited()
    await f.db.aimemory.update(
        where={"id": ids[0]}, data={"workspaceId": f.wid, "subCategory": "健康"}
    )
    with pytest.raises(ValueError, match="mixed_categories"):
        await f.prepare()
    await f.db.aimemory.update(
        where={"id": ids[0]}, data={"subCategory": "其他", "content": "changed"}
    )
    with pytest.raises(writer.MemoryWriteError, match="candidate_content_changed"):
        await writer.prepare_cluster(
            f.db, source="ai", user_id=f.uid, workspace_id=f.wid, cluster=cluster
        )


async def test_receipt_survives_digest_deletion_and_cannot_recreate_it(compression):
    f = compression
    ids = await f.seed()
    inputs = await f.prepare()
    digest = await f.commit(inputs)
    await f.db.aimemory.delete(where={"id": digest})
    await f.db.execute_raw(
        "UPDATE memories_ai SET is_archived=false WHERE id=ANY($1::text[])", ids
    )
    # Even if the original archive audit was removed externally, the immutable
    # scoped receipt blocks the same version set from recreating a deleted digest.
    await f.db.memorychangelog.delete_many(
        where={"workspaceId": f.wid, "operation": "consolidated_into"}
    )
    with pytest.raises(writer.MemoryWriteError, match="already_recorded"):
        await f.commit(await f.prepare())
    assert (
        await f.db.aimemory.count(
            where={"workspaceId": f.wid, "provenance": "consolidated"}
        )
        == 0
    )


async def test_parent_deleted_or_changed_and_receipt_deleted_are_reported_without_private_text(
    compression,
):
    f = compression
    ids = await f.seed()
    digest = await f.commit(await f.prepare())
    await f.db.aimemory.delete(where={"id": ids[0]})
    await f.db.aimemory.update(where={"id": ids[1]}, data={"content": "PRIVATE_EDIT"})
    await f.db.execute_raw(
        "DELETE FROM memory_input_origins WHERE workspace_id=$1", f.wid
    )
    result = await evidence_read.memory_evidence_detail(
        memory_id=digest, side="ai", user_id=f.uid, workspace_id=f.wid
    )
    assert sum(x["availability"] == "deleted" for x in result["items"]) == 2
    assert sum(x["availability"] == "changed" for x in result["items"]) == 1
    assert (
        next(x for x in result["items"] if x["source_kind"] == "input")["consolidation"]
        is None
    )
    assert "PRIVATE" not in str(result)


async def test_real_admin_api_pagination_scope_and_privacy(compression):
    f = compression
    await f.seed(n=25)
    digest = await f.commit(await f.prepare())
    app = FastAPI()
    app.include_router(memory_repairs.router)
    path = f"/admin-api/memory-repairs/evidence/ai/{digest}"
    params = {"user_id": f.uid, "workspace_id": f.wid, "limit": 20}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://isolated.test"
    ) as client:
        assert (await client.get(path, params=params)).status_code == 401
        assert (
            await client.get(path, params=params, headers=auth(f.uid))
        ).status_code == 403
        await f.db.user.update(where={'id':f.uid},data={'role':'admin'})
        first = await client.get(path, params=params, headers=auth(f.uid, "admin"))
        assert first.status_code == 200
        second = await client.get(
            path,
            params={**params, "cursor": first.json()["next_cursor"]},
            headers=auth(f.uid, "admin"),
        )
        assert len(first.json()["items"]) + len(second.json()["items"]) == 26
        assert "PRIVATE_RENDERED_PROMPT" not in first.text + second.text
        for overrides, expected in [
            ({"workspace_id": f.spaces[1]}, 404),
            ({"user_id": f.users[-1]}, 404),
            ({"cursor": "bad"}, 400),
            ({"limit": 51}, 422),
        ]:
            assert (
                await client.get(
                    path, params={**params, **overrides}, headers=auth(f.uid, "admin")
                )
            ).status_code == expected


async def test_large_parent_set_is_paginated_and_side_ambiguity_is_rejected(
    compression,
):
    f = compression
    ids = await f.seed(n=101)
    digest = await f.commit(await f.prepare())
    assert (
        len(
            await f.db.query_raw(
                "SELECT id FROM memory_evidence_links WHERE memory_id=$1", digest
            )
        )
        == 102
    )
    page = await evidence_read.memory_evidence_detail(
        memory_id=digest, side="ai", user_id=f.uid, workspace_id=f.wid, limit=50
    )
    assert len(page["items"]) == 50 and page["next_cursor"]


async def test_lost_commit_ack_preserves_published_digest_and_fences_replay(
    compression, monkeypatch
):
    from contextlib import asynccontextmanager

    f = compression
    await f.seed()
    inputs = await f.prepare()

    class LostAckDatabase:
        def __getattr__(self, name):
            return getattr(f.db, name)

        @asynccontextmanager
        async def tx(self, **kwargs):
            async with f.db.tx(**kwargs) as tx:
                yield tx
            raise RuntimeError("synthetic lost commit acknowledgment")

    with pytest.raises(RuntimeError, match="lost commit acknowledgment"):
        await f.commit(inputs, database=LostAckDatabase())
    assert (
        await f.db.aimemory.count(
            where={
                "workspaceId": f.wid,
                "provenance": "consolidated",
                "isArchived": False,
            }
        )
        == 1
    )
    assert (
        len(
            await f.db.query_raw(
                "SELECT id FROM memory_input_origins WHERE workspace_id=$1", f.wid
            )
        )
        == 1
    )
    with pytest.raises(writer.MemoryWriteError):
        await f.commit(inputs)


async def test_postcommit_hook_failure_cannot_undo_digest(compression, monkeypatch):
    f = compression
    await f.seed(occurTime=datetime(2026, 8, 1, tzinfo=UTC), importance=0.8)
    monkeypatch.setattr(
        writer,
        "after_memory_changelog",
        AsyncMock(side_effect=RuntimeError("hook failed")),
    )
    digest = await f.commit(await f.prepare())
    row = await f.db.aimemory.find_unique(where={"id": digest})
    assert (
        row.level == 3
        and row.importance == 0.49
        and row.currentScore == 0.49
        and not row.isArchived
    )
    assert row.occurTime == datetime(2026, 8, 1, tzinfo=UTC)
