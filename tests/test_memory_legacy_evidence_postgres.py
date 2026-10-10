"""Legacy IDs remain unverified; real PostgreSQL scope and limit regressions."""
from datetime import datetime, timedelta, timezone
import json

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from app.api.admin import memory_repairs
from app.services.auth import create_jwt
from app.services.memory.evidence import EvidenceSource
from tests.test_memory_evidence_postgres import origins


@pytest.fixture
async def legacy_origins(origins):
    try:
        yield origins
    finally:
        await origins[0].memorychangelog.delete_many(where={"userId": {"in": origins[7]}})


async def log(db, uid, wid, mid, ids=None, raw=None, **kwargs):
    return await db.memorychangelog.create(data={"userId": uid, "workspaceId": wid,
        "memoryId": mid, "operation": "evidence_linked",
        "newValue": raw if raw is not None else json.dumps({"message_ids": ids}), **kwargs})


@pytest.mark.parametrize("side", ["user", "ai"])
@pytest.mark.asyncio
async def test_legacy_valid_reference_and_mixed_snapshots_never_backfill(legacy_origins, side):
    db, uid, wid, mid, msg, bind, detail, *_ = legacy_origins
    await (db.aimemory if side == "user" else db.usermemory).delete(where={"id": mid})
    if side == "ai":
        await db.message.update(where={"id": msg.id}, data={"role": "assistant"})
    assert (await detail(side))["legacy"]["state"] == "none"
    await log(db, uid, wid, mid, [msg.id, msg.id])
    data = await detail(side)
    legacy = data["legacy"]
    assert data["state"] == "historical_unknown" and not data["items"]
    assert legacy["state"] == "pending_verification" and legacy["accessible_references"] == 1
    assert legacy["items"][0]["source_ref"] == msg.id
    assert not legacy["source_version_known"] and not legacy["content_version_known"]
    assert msg.content not in str(data)
    assert not await db.query_raw("SELECT id FROM memory_evidence_links WHERE memory_id=$1", mid)
    await bind(EvidenceSource("message", msg.id), side=side)
    mixed = await detail(side)
    assert mixed["state"] == "linked" and mixed["legacy"]["checked_references"] == 1
    await db.message.update(where={"id": msg.id}, data={"content": "后来改过的合成消息"})
    mixed = await detail(side)
    assert mixed["items"][0]["availability"] == "changed"
    assert mixed["legacy"]["items"][0]["availability"] == "accessible_reference"
    assert not mixed["legacy"]["source_version_known"]
    await (db.usermemory if side == "user" else db.aimemory).update(where={"id": mid}, data={"content": "当前记忆新版本"})
    assert (await detail(side))["state"] == "current_unlinked"


@pytest.mark.asyncio
async def test_legacy_missing_side_collision_is_redacted(legacy_origins):
    db, uid, wid, mid, msg, bind, detail, *_ = legacy_origins
    await log(db, uid, wid, mid, [msg.id])
    for side in ("user", "ai"):
        result = (await detail(side))["legacy"]
        assert result["accessible_references"] == 0
        assert result["items"][0]["availability"] == "ambiguous_side"
        assert msg.id not in str(result) and msg.content not in str(result)


@pytest.mark.asyncio
async def test_legacy_foreign_scope_role_deleted_and_rebound_sources(legacy_origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = legacy_origins
    await db.aimemory.delete(where={"id": mid})
    foreign = []
    for conv in convs[1:]:
        row = await db.message.create(data={"conversationId": conv, "role": "user", "content": "其他作用域"})
        foreign.append(row.id)
    wrong_role = await db.message.create(data={"conversationId": convs[0], "role": "assistant", "content": "AI生成"})
    await log(db, uid, wid, mid, [*foreign, wrong_role.id, "missing-message", msg.id])
    # Unscoped/foreign changelogs cannot supply a reference, even if the ID matches.
    await log(db, users[-1], spaces[-1], mid, ["foreign-secret"])
    await log(db, uid, None, mid, ["unscoped-secret"])
    data = await detail()
    assert data["legacy"]["checked_references"] == 5
    assert data["legacy"]["accessible_references"] == 1
    for secret in foreign + [wrong_role.id, "foreign-secret", "unscoped-secret", "missing-message"]:
        assert secret not in str(data)
    await db.conversation.update(where={"id": convs[0]}, data={"isDeleted": True})
    assert (await detail())["legacy"]["accessible_references"] == 0
    await db.message.delete(where={"id": msg.id})
    data = await detail()
    assert sum(item["availability"] == "missing" for item in data["legacy"]["items"]) == 2
    await db.conversation.update(where={"id": convs[0]}, data={"isDeleted": False})
    fresh = await db.message.create(data={"conversationId": convs[0], "role": "user", "content": "合成新消息"})
    await log(db, uid, wid, mid, [fresh.id])
    await db.chatworkspace.update(where={"id": wid}, data={"agentId": agents[1]})
    assert (await detail())["legacy"]["accessible_references"] == 0
    await db.chatworkspace.update(where={"id": wid}, data={"agentId": None})
    assert (await detail())["legacy"]["accessible_references"] == 0


@pytest.mark.asyncio
async def test_legacy_ambiguous_target_age_and_source_recreated_after_log(legacy_origins):
    db, uid, wid, mid, msg, bind, detail, *_ = legacy_origins
    await db.aimemory.delete(where={"id": mid})
    now = datetime.now(timezone.utc)
    await log(db, uid, wid, mid, [msg.id], createdAt=now-timedelta(days=1))
    assert (await detail())["legacy"]["items"][0]["availability"] == "unavailable"
    await db.usermemory.update(where={"id": mid}, data={"createdAt": now-timedelta(days=2)})
    assert (await detail())["legacy"]["items"][0]["availability"] == "unavailable"


@pytest.mark.asyncio
async def test_legacy_bounded_logs_references_and_malformed_payloads(legacy_origins):
    db, uid, wid, mid, msg, bind, detail, *_ = legacy_origins
    await db.aimemory.delete(where={"id": mid})
    for raw in ["broken", "[]", '{"message_ids":null}', '{"message_ids":[]}',
                '{"message_ids":[2]}', json.dumps({"message_ids": ["x"*201]}),
                "["*3000 + "]"*3000, "x"*16001]:
        await log(db, uid, wid, mid, raw=raw)
    preview = (await detail())["legacy"]
    assert preview["invalid_logs"] == 8 and preview["incomplete"]
    assert preview["state"] == "pending_verification" and not preview["items"]
    for _ in range(13):
        await log(db, uid, wid, mid, [msg.id])
    preview = (await detail())["legacy"]
    assert preview["checked_logs"] == 20 and preview["incomplete"]
    await db.memorychangelog.delete_many(where={"memoryId": mid})
    await log(db, uid, wid, mid, [str(i) for i in range(51)])
    preview = (await detail())["legacy"]
    assert preview["checked_references"] == 50 and preview["incomplete"]
    assert preview["denominator"] == "bounded_legacy_sample"


@pytest.mark.asyncio
async def test_legacy_admin_api_auth_and_no_source_content(legacy_origins):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = legacy_origins
    await db.aimemory.delete(where={"id": mid})
    await log(db, uid, wid, mid, [msg.id])
    app = FastAPI(); app.include_router(memory_repairs.router)
    headers = {"Authorization": "Bearer " + create_jwt("synthetic-admin", role="admin")}
    path = f"/admin-api/memory-repairs/evidence/user/{mid}"
    params = {"user_id": uid, "workspace_id": wid}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://isolated.test") as client:
        assert (await client.get(path, params=params)).status_code == 401
        assert (await client.get(path, params=params, headers={"Authorization": "Bearer "+create_jwt(uid, role="user")})).status_code == 403
        assert (await client.get(path, params={**params,"workspace_id": spaces[1]}, headers=headers)).status_code == 404
        response = await client.get(path, params=params, headers=headers)
        assert response.status_code == 200
        assert response.json()["legacy"]["items"][0]["source_ref"] == msg.id
        assert msg.content not in response.text
    assert not await db.query_raw("SELECT id FROM memory_evidence_links WHERE memory_id=$1", mid)
