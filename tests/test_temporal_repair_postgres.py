"""Bounded repair uses pinned interpretations, never guessed historical dates."""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
from prisma import Json
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.services.memory import temporal_repair as repair
from app.services.memory.manual_write import MemoryWriteError
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_validity_postgres import validity
from tests.test_memory_input_write_api_postgres import reject_write


@pytest.fixture
async def missing_projection(validity, monkeypatch):
    f, source, read = validity
    db, uid, wid, mid, msg, bind, *_ = f
    monkeypatch.setattr(repair, "db", db)
    await db.aimemory.delete(where={"id": mid})
    await db.user.update(where={"id": uid}, data={"role": "admin"})
    await bind(source)
    # Simulate a historical projection gap; retain immutable source evidence.
    await db.execute_raw(
        "DELETE FROM memory_temporal_assertions WHERE memory_id=$1", mid
    )
    args = dict(user_id=uid, workspace_id=wid, side="user", memory_id=mid)
    before = await db.usermemory.find_unique(where={"id": mid})
    page = await repair.preview(**args)
    assert page["writes_performed"] == 0 and len(page["items"]) == 1
    item = page["items"][0]
    assert item["classification"] == "provable"
    request = {
        **args,
        "evidence_id": item["evidence_id"],
        "expected_version": item["expected_version"],
        "actor": {"sub": uid, "role": "admin"},
    }
    return SimpleNamespace(**locals())


async def test_preview_apply_replay_and_two_workers(missing_projection):
    f = missing_projection
    assert not await f.db.query_raw(
        "SELECT id FROM memory_temporal_assertions WHERE memory_id=$1", f.mid
    )
    results = await asyncio.gather(repair.apply(**f.request), repair.apply(**f.request))
    assert sorted(r["status"] for r in results) == ["already_applied", "applied"]
    assert (
        await f.db.memorychangelog.count(
            where={"memoryId": f.mid, "operation": "temporal_projection_backfill"}
        )
        == 1
    )
    assert (await repair.preview(**f.args))["items"][0][
        "classification"
    ] == "already_recorded"
    assert await f.db.usermemory.find_unique(where={"id": f.mid}) == f.before
    rows = await f.db.query_raw(
        "SELECT payload_text FROM memory_input_origins WHERE user_id=$1", f.uid
    )
    assert (
        len(rows) == 1
        and json.loads(rows[0]["payload_text"])["input"]["user_confirmed"] is False
    )
    assert (await f.read())["items"][0]["assertions"]


@pytest.mark.parametrize(
    "mutation",
    ["source", "clock", "card", "takeover", "memory", "delete", "role", "scope"],
)
async def test_stale_source_scope_and_actor_fail_closed(missing_projection, mutation):
    f = missing_projection
    if mutation == "source":
        await f.db.message.update(where={"id": f.source.ref}, data={"content": "改写"})
    if mutation == "clock":
        await f.db.message.update(
            where={"id": f.source.ref},
            data={"createdAt": f.source.expected_statement_time + timedelta(days=1)},
        )
    if mutation == "card":
        await f.db.message.update(
            where={"id": f.source.ref}, data={"metadata": Json({"component_card": {}})}
        )
    if mutation == "takeover":
        await f.db.message.update(
            where={"id": f.source.ref},
            data={"metadata": Json({"human_takeover": True})},
        )
    if mutation == "memory":
        await f.db.usermemory.update(where={"id": f.mid}, data={"content": "已更正"})
    if mutation == "delete":
        await f.db.message.delete(where={"id": f.source.ref})
    if mutation == "role":
        await f.db.user.update(where={"id": f.uid}, data={"role": "user"})
    if mutation == "scope":
        await f.db.chatworkspace.update(
            where={"id": f.wid}, data={"status": "archived"}
        )
    with pytest.raises((MemoryWriteError, ValueError)):
        await repair.apply(**f.request)
    assert not await f.db.query_raw(
        "SELECT id FROM memory_temporal_assertions WHERE memory_id=$1", f.mid
    )
    assert (
        await f.db.memorychangelog.count(
            where={"memoryId": f.mid, "operation": "temporal_projection_backfill"}
        )
        == 0
    )


@pytest.mark.parametrize(
    "table", ["memory_temporal_assertions", "memory_input_origins", "memory_changelogs"]
)
async def test_repair_failure_atomic(missing_projection, table):
    f = missing_projection
    async with reject_write(f.db, table, "TRUE"):
        with pytest.raises(Exception):
            await repair.apply(**f.request)
    assert not await f.db.query_raw(
        "SELECT id FROM memory_temporal_assertions WHERE memory_id=$1", f.mid
    )
    assert not await f.db.query_raw(
        "SELECT id FROM memory_input_origins WHERE user_id=$1", f.uid
    )
    assert (
        await f.db.memorychangelog.count(
            where={"memoryId": f.mid, "operation": "temporal_projection_backfill"}
        )
        == 0
    )


async def test_unknown_basis_is_not_reconstructed_from_similar_text(
    validity, monkeypatch
):
    f, source, _ = validity
    db, uid, wid, mid, _, bind, *_ = f
    monkeypatch.setattr(repair, "db", db)
    await bind(replace(source, temporal_basis={"source_role": "fact"}))
    page = await repair.preview(
        user_id=uid, workspace_id=wid, side="user", memory_id=mid
    )
    assert page["items"][0]["classification"] == "needs_review"
    assert page["items"][0]["proposed"] is None
    assert not await db.query_raw(
        "SELECT id FROM memory_temporal_assertions WHERE memory_id=$1", mid
    )


async def test_real_api_auth_foreign_scope_and_strict_request(missing_projection):
    from app.api.admin.memory_repairs import router
    from app.services.auth import create_jwt

    f = missing_projection
    app = FastAPI()
    app.include_router(router)
    url = f"/admin-api/memory-repairs/time-repair/user/{f.mid}"
    query = {"user_id": f.uid, "workspace_id": f.wid}
    headers = {"Authorization": "Bearer " + create_jwt(f.uid, "admin")}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get(url, params=query)).status_code == 401
        assert (
            await client.get(
                url, params={**query, "workspace_id": f.f[9][-1]}, headers=headers
            )
        ).status_code == 404
        page = (await client.get(url, params=query, headers=headers)).json()
        payload = {
            **query,
            **{k: page["items"][0][k] for k in ("evidence_id", "expected_version")},
        }
        assert (
            await client.post(
                url, json={**payload, "expected_version": "bad"}, headers=headers
            )
        ).status_code == 422
        assert (await client.post(url, json=payload, headers=headers)).json()[
            "status"
        ] == "applied"
        assert (await client.post(url, json=payload, headers=headers)).json()[
            "status"
        ] == "already_applied"
