"""Semantic history, source-clock CAS, scope and erasure on real PostgreSQL."""

from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from app.api.admin import memory_repairs, runtime_jobs
from app.services.auth import create_jwt
from app.services.memory import temporal_history
from app.services.memory.recording import receipt_read
from tests.test_memory_evidence_postgres import origins
from tests.test_contradiction_write_postgres import confirming


@pytest.fixture
async def history(origins, monkeypatch):
    db, uid, wid, mid, msg, *_ = origins
    monkeypatch.setattr(temporal_history, "db", db)
    monkeypatch.setattr(receipt_read, "db", db)

    async def read(**kw):
        return await temporal_history.read_history(
            user_id=uid, workspace_id=wid, side="user", memory_id=mid, **kw
        )

    return (*origins, read)


async def test_semantic_history_ignores_usage_preserves_clock_and_erases(history):
    db, uid, wid, mid, msg, *_ = history
    read = history[-1]
    first = await read()
    assert len(first["items"]) == 1
    assert first["items"][0]["record_kind"] == "insert"
    await db.usermemory.update(
        where={"id": mid}, data={"importance": 0.7, "mentionCount": 2}
    )
    assert len((await read())["items"]) == 1
    clock = datetime(2025, 1, 2, tzinfo=timezone.utc)
    await db.usermemory.update(
        where={"id": mid},
        data={"occurTime": clock, "statementTime": clock + timedelta(days=1)},
    )
    await db.usermemory.update(
        where={"id": mid}, data={"content": "有明确更正的合成事实"}
    )
    page = await read(limit=2)
    assert len(page["items"]) == 2 and page["next_cursor"]
    assert datetime.fromisoformat(page["items"][0]["event_time"]) == clock
    assert page["items"][0]["validity"] == {
        "status": "unknown",
        "from": None,
        "to": None,
    }
    older = await read(limit=2, before=page["next_cursor"])
    assert len(older["items"]) == 1
    assert not {i["id"] for i in page["items"]} & {i["id"] for i in older["items"]}
    await db.usermemory.delete(where={"id": mid})
    assert (
        await db.query_raw(
            "SELECT id FROM memory_temporal_versions WHERE memory_id=$1 AND side='user'",
            mid,
        )
        == []
    )
    assert await db.query_raw(
        "SELECT id FROM memory_temporal_versions WHERE memory_id=$1 AND side='ai'", mid
    )
    with pytest.raises(LookupError):
        await read()


async def test_legacy_baseline_is_labelled_and_failed_edit_leaves_no_version(history):
    db, uid, wid, mid, *_ = history
    read = history[-1]
    await db.execute_raw(
        "DELETE FROM memory_temporal_versions WHERE memory_id=$1 AND side='user'", mid
    )
    assert (await read())["state"] == "not_collected"
    with pytest.raises(Exception):
        async with db.tx() as tx:
            await tx.usermemory.update(
                where={"id": mid}, data={"content": "rolled back"}
            )
            await tx.execute_raw("SELECT 1/0")
    assert not (await read())["items"]
    await db.usermemory.update(where={"id": mid}, data={"content": "更正后"})
    rows = (await read())["items"]
    assert [r["record_kind"] for r in rows] == ["update", "baseline"]
    assert rows[1]["content"] == "合成地点事实"


async def test_evidence_is_bound_to_semantic_revision(history):
    from app.services.memory.evidence import EvidenceSource

    db, uid, wid, mid, msg, bind, *_ = history
    await bind(EvidenceSource("message", msg.id, relation="recorded_from"))
    first = (
        await db.query_raw(
            "SELECT temporal_version_id FROM memory_evidence_links WHERE memory_id=$1",
            mid,
        )
    )[0]["temporal_version_id"]
    await db.usermemory.update(
        where={"id": mid}, data={"occurTime": datetime(2024, 1, 1, tzinfo=timezone.utc)}
    )
    await bind(EvidenceSource("message", msg.id, relation="recorded_from"))
    rows = await db.query_raw(
        "SELECT temporal_version_id FROM memory_evidence_links WHERE memory_id=$1", mid
    )
    assert len(rows) == 2 and len({r["temporal_version_id"] for r in rows}) == 2
    assert first in {r["temporal_version_id"] for r in rows}


async def test_api_permissions_scope_and_receipt_route_are_enforced(history):
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs, read = history
    app = FastAPI()
    app.include_router(memory_repairs.router)
    app.include_router(runtime_jobs.router)
    endpoint = f"/admin-api/memory-repairs/time-history/user/{mid}"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (
            await client.get(endpoint, params={"user_id": uid, "workspace_id": wid})
        ).status_code == 401
        token = create_jwt(uid, "user")
        assert (
            await client.get(
                endpoint,
                params={"user_id": uid, "workspace_id": wid},
                headers={"Authorization": "Bearer " + token},
            )
        ).status_code == 403
        await db.user.update(where={"id": uid}, data={"role": "admin"})
        token = create_jwt(uid, "admin")
        headers = {"Authorization": "Bearer " + token}
        assert (
            await client.get(
                endpoint, params={"user_id": uid, "workspace_id": wid}, headers=headers
            )
        ).status_code == 200
        for field, value in [("user_id", users[-1]), ("workspace_id", spaces[-1])]:
            params = {"user_id": uid, "workspace_id": wid, field: value}
            assert (
                await client.get(endpoint, params=params, headers=headers)
            ).status_code == 404
        assert (
            await client.get(
                endpoint,
                params={"user_id": uid, "workspace_id": wid, "before": "-1"},
                headers=headers,
            )
        ).status_code == 400
        response = await client.get(
            "/admin-api/runtime-jobs/memory-write-receipts", headers=headers
        )
        assert (
            response.status_code == 200
            and response.json()["denominator"] == "this_page_only"
        )


async def test_confirmed_change_uses_source_clock_not_execution_clock(
    confirming, monkeypatch
):
    f = confirming
    monkeypatch.setattr(temporal_history, "db", f.db)
    result = await f.apply()
    row = await f.db.usermemory.find_unique(where={"id": result["memory_id"]})
    assert row.statementTime == f.second.createdAt
    assert row.occurTime is None
    history = await temporal_history.read_history(
        user_id=f.uid, workspace_id=f.wid, side="user", memory_id=row.id
    )
    assert history["items"][0]["change_type"] == "变化"
    assert history["items"][0]["validity"]["status"] == "unknown"
    assert (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived


async def test_changed_confirmation_timestamp_rejects_correction(confirming):
    f = confirming
    await f.db.message.update(
        where={"id": f.second.id},
        data={"createdAt": f.second.createdAt + timedelta(hours=1)},
    )
    with pytest.raises(ValueError, match="source_time_changed"):
        await f.apply()
    assert not (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived


async def test_revision_binding_cannot_cross_memory_or_be_retargeted(history):
    from uuid import uuid4
    from prisma.errors import RawQueryError
    from app.services.memory.evidence import EvidenceSource

    db, uid, wid, mid, msg, bind, *_ = history
    await bind(EvidenceSource("message", msg.id, relation="recorded_from"))
    original = (
        await db.query_raw(
            "SELECT * FROM memory_evidence_links WHERE memory_id=$1", mid
        )
    )[0]
    other = await db.usermemory.create(
        data={
            "userId": uid,
            "workspaceId": wid,
            "content": "另一个独立合成事实",
            "level": 2,
        }
    )
    revision = (
        await db.query_raw(
            "SELECT id FROM memory_temporal_versions WHERE memory_id=$1", other.id
        )
    )[0]["id"]
    with pytest.raises(RawQueryError, match="immutable"):
        await db.execute_raw(
            "UPDATE memory_evidence_links SET temporal_version_id=$1 WHERE id=$2",
            revision,
            original["id"],
        )
    row = {**original, "id": uuid4().hex * 2, "temporal_version_id": revision}
    columns = [key for key in row if key != "created_at"]
    with pytest.raises(RawQueryError, match="temporal_binding_scope"):
        await db.execute_raw(
            "INSERT INTO memory_evidence_links ("
            + ",".join(columns)
            + ") VALUES ("
            + ",".join(f"${i+1}" for i in range(len(columns)))
            + ")",
            *[row[k] for k in columns],
        )
    await db.execute_raw(
        "DELETE FROM memory_temporal_versions WHERE id=$1",
        original["temporal_version_id"],
    )
    retained = (
        await db.query_raw(
            "SELECT temporal_version_id,source_version FROM memory_evidence_links WHERE id=$1",
            original["id"],
        )
    )[0]
    assert retained == {
        "temporal_version_id": None,
        "source_version": original["source_version"],
    }
