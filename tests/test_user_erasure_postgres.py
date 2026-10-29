"""Owned, disposable PG/Redis fixtures. Never runs against production or shared data."""

import json
import os
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

import pytest
from prisma import Prisma, Json
from prisma.errors import RawQueryError, ForeignKeyViolationError
from redis.asyncio import Redis
from unittest.mock import AsyncMock
from app.services.runtime import data_reset, user_erasure


@pytest.fixture
async def erased_pair(monkeypatch, tmp_path):
    url = os.environ.get("ERASURE_E2E_DATABASE_URL", "")
    redis_url = os.environ.get("ERASURE_E2E_REDIS_URL", "")
    if not url or not redis_url:
        pytest.skip("Requires dedicated account-erasure PostgreSQL and Redis")
    assert (
        urlsplit(url).hostname == "127.0.0.1"
        and urlsplit(url).path == "/companion_erasure_e2e"
    )
    assert urlsplit(redis_url).hostname == "127.0.0.1"
    assert urlsplit(redis_url).path == "/12"
    assert os.environ.get("APP_ENV") != "production"
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    redis = Redis.from_url(redis_url, decode_responses=True)
    monkeypatch.setattr(data_reset, "db", database)
    import app.db

    monkeypatch.setattr(app.db, "db", database)
    monkeypatch.setattr(data_reset, "get_redis", AsyncMock(return_value=redis))
    roots = {
        kind: tmp_path / kind
        for kind in ["chat", "capsule", "avatar", "feedback", "offline"]
    }
    for root in roots.values():
        root.mkdir()
    monkeypatch.setattr(data_reset.chat_media_storage, "_MEDIA_DIR", roots["chat"])
    monkeypatch.setattr(
        data_reset.activity_media_storage, "_MEDIA_DIR", roots["offline"]
    )
    for env, kind in [
        ("CAPSULE_MEDIA_DIR", "capsule"),
        ("USER_AVATAR_DIR", "avatar"),
        ("FEEDBACK_MEDIA_DIR", "feedback"),
    ]:
        monkeypatch.setenv(env, str(roots[kind]))
    records = []
    try:
        await database.execute_raw(
            "INSERT INTO chat_link_storage_state(id) VALUES (1) ON CONFLICT DO NOTHING"
        )
        for _ in range(2):
            user = await database.user.create(
                data={"username": "erase-fixture-" + uuid4().hex}
            )
            agent = await database.aiagent.create(
                data={"userId": user.id, "name": "Synthetic"}
            )
            workspace = await database.chatworkspace.create(
                data={"userId": user.id, "agentId": agent.id}
            )
            conv = await database.conversation.create(
                data={
                    "userId": user.id,
                    "agentId": agent.id,
                    "workspaceId": workspace.id,
                }
            )
            message = await database.message.create(
                data={
                    "conversationId": conv.id,
                    "role": "user",
                    "content": "synthetic private fixture",
                }
            )
            memory = await database.usermemory.create(
                data={
                    "userId": user.id,
                    "workspaceId": workspace.id,
                    "content": "synthetic private memory",
                    "level": 1,
                }
            )
            trace = uuid4().hex
            await database.tracerun.create(
                data={
                    "id": trace,
                    "traceId": trace,
                    "name": "chat_request",
                    "runType": "chain",
                    "startedAt": message.createdAt,
                    "inputsJson": Json(
                        {"conversation_id": conv.id, "message": message.content}
                    ),
                }
            )
            phone = "139" + str(int(uuid4().hex[:10], 16) % 100000000).zfill(8)
            await database.authidentity.create(
                data={
                    "userId": user.id,
                    "provider": "phone",
                    "providerAccountId": phone,
                }
            )
            await redis.set("sms:code:" + phone, "synthetic-code")
            offline_trace = uuid4().hex
            await database.tracerun.create(
                data={
                    "id": offline_trace,
                    "traceId": offline_trace,
                    "name": "chat_request",
                    "runType": "chain",
                    "startedAt": message.createdAt,
                    "inputsJson": Json({"conversation_id": ""}),
                    "extraJson": Json({"metadata": {"owner_user_id": user.id}}),
                }
            )
            await database.chatmessageattachment.create(
                data={
                    "userId": user.id,
                    "conversationId": conv.id,
                    "storageKey": user.id + "_orphan.jpg",
                    "url": "/chat/media/" + user.id + "_orphan.jpg",
                    "mime": "image/jpeg",
                    "size": 11,
                }
            )
            await database.timecapsule.create(
                data={
                    "userId": user.id,
                    "content": "synthetic capsule",
                    "media": Json({"audio": {"storage_key": user.id + "_audio.m4a"}}),
                }
            )
            child = uuid4().hex
            await database.tracerun.create(
                data={
                    "id": child,
                    "traceId": trace,
                    "parentId": trace,
                    "name": "llm",
                    "runType": "llm",
                    "startedAt": message.createdAt,
                }
            )
            await database.execute_raw(
                """INSERT INTO speech_usage(id,user_id,conversation_id,display_mode,duration_seconds)
                VALUES($1,$2,$3,'voice',3)""",
                uuid4().hex,
                user.id,
                conv.id,
            )
            await database.execute_raw(
                """INSERT INTO tts_usage(id,user_id,agent_id,source,provider,model,voice_id,
                raw_characters,billable_characters,duration_milliseconds,audio_bytes,unit_price_cny,cost_cny)
                VALUES($1,$2,$3,'chat','dashscope','synthetic','system',1,1,1,1,0,0)""",
                uuid4().hex,
                user.id,
                agent.id,
            )
            await redis.set("context_window:" + conv.id, "synthetic")
            await redis.set(
                "ws:ticket:" + uuid4().hex,
                json.dumps({"sub": user.id, "conversation_id": conv.id}),
            )
            await redis.set(
                "cache:ret:" + uuid4().hex,
                json.dumps({"memories": [{"id": memory.id}]}),
            )
            for root in roots.values():
                (root / (user.id + "_orphan.jpg")).write_bytes(b"owned image")
                (root / (user.id + "_audio.m4a")).write_bytes(b"owned audio")
                (root / (user.id + "_orphan_t.jpg")).write_bytes(b"thumbnail")
            records.append((user, agent, workspace, conv, message, memory, trace))
            (roots["feedback"] / (user.id + "_preserve_interrupted.tmp")).write_bytes(
                b"synthetic interrupted migration"
            )
            from app.services.chat_links.retention import reserve_cover

            assert await reserve_cover(
                user.id + "_orphan.jpg", len(b"owned image"), database=database
            )
        await redis.set("cache:emb:public-sentinel", "[0.1,0.2]")
        await redis.set("unrelated-sentinel", "keep")
        yield database, redis, roots, records
    finally:
        # Only synthetic owned IDs; the container itself is dedicated as well.
        for user, agent, workspace, conv, message, memory, trace in records:
            await user_erasure.delete_cover_ledger(database, user.id)
            await database.execute_raw(
                "DELETE FROM trace_runs WHERE trace_id=$1 OR extra_json->'metadata'->>'owner_user_id'=$2",
                trace,
                user.id,
            )
            await database.usermemory.delete_many(where={"userId": user.id})
            await database.message.delete_many(where={"conversationId": conv.id})
            await database.conversation.delete_many(where={"id": conv.id})
            await database.chatworkspace.delete_many(where={"id": workspace.id})
            await database.aiagent.delete_many(where={"id": agent.id})
            await database.user.delete_many(where={"id": user.id})
        await redis.aclose()
        await database.disconnect()


@pytest.mark.asyncio
async def test_erasure_removes_private_rows_files_redis_and_preserves_other_owner(
    erased_pair,
):
    database, redis, roots, records = erased_pair
    target, survivor = records
    uid = target[0].id
    other = survivor[0].id
    before = (await database.user.find_unique(where={"id": other})).model_dump()
    other_files = {
        str(p): p.read_bytes()
        for root in roots.values()
        for p in root.glob(other + "_*")
    }
    from app.services.runtime import job_queue

    old = job_queue.get_redis
    job_queue.get_redis = AsyncMock(return_value=redis)
    try:
        own_job = await job_queue.enqueue_runtime_job(
            "synthetic", {"user_id": uid}, idempotency_key="owned-" + uid
        )
        other_job = await job_queue.enqueue_runtime_job(
            "synthetic", {"user_id": other}, idempotency_key="owned-" + other
        )
    finally:
        job_queue.get_redis = old
    phone = (
        await database.authidentity.find_first(where={"userId": uid})
    ).providerAccountId
    peer_phone = (
        await database.authidentity.find_first(where={"userId": other})
    ).providerAccountId
    result = await data_reset.hard_delete_user_data(uid)
    assert await redis.get("sms:code:" + phone) is None
    assert await redis.get("sms:code:" + peer_phone) == "synthetic-code"
    assert result["user"] == 1
    assert await database.user.find_unique(where={"id": uid}) is None
    assert (await database.user.find_unique(where={"id": other})).model_dump() == before
    assert await database.message.find_unique(where={"id": survivor[4].id})
    assert await database.usermemory.find_unique(where={"id": survivor[5].id})
    assert await database.tracerun.find_unique(where={"id": survivor[6]})
    for root in roots.values():
        assert not list(root.glob(uid + "_*"))
    for name, content in other_files.items():
        assert Path(name).read_bytes() == content
    assert await redis.exists("runtime:job:" + own_job) == 0
    assert await redis.exists("runtime:job:" + other_job) == 1
    assert other_job in await redis.lrange("runtime:jobs:ready", 0, -1)
    assert await redis.get("cache:emb:public-sentinel") == "[0.1,0.2]"
    assert await redis.get("unrelated-sentinel") == "keep"
    assert await redis.exists("context_window:" + survivor[3].id) == 1
    assert await redis.exists("context_window:" + target[3].id) == 0
    assert await database.tracerun.count(where={"traceId": target[6]}) == 0
    assert (
        await database.query_raw(
            "SELECT storage_key FROM chat_link_cover_objects WHERE storage_key=$1",
            uid + "_orphan.jpg",
        )
        == []
    )
    assert await database.query_raw(
        "SELECT storage_key FROM chat_link_cover_objects WHERE storage_key=$1",
        other + "_orphan.jpg",
    )
    # A completed deletion cannot be undone by late metering/Trace writes.
    with pytest.raises(RawQueryError):
        await database.execute_raw(
            "INSERT INTO speech_usage(id,user_id,conversation_id,display_mode,duration_seconds) VALUES($1,$2,$3,'voice',1)",
            uuid4().hex,
            uid,
            target[3].id,
        )
    with pytest.raises(Exception):
        await database.tracerun.create(
            data={
                "id": uuid4().hex,
                "traceId": target[6],
                "name": "late",
                "runType": "llm",
                "startedAt": target[4].createdAt,
            }
        )
    with pytest.raises(Exception):
        tid = uuid4().hex
        await database.tracerun.create(
            data={
                "id": tid,
                "traceId": tid,
                "name": "chat_request",
                "runType": "chain",
                "startedAt": target[4].createdAt,
                "inputsJson": Json({"conversation_id": target[3].id}),
            }
        )


@pytest.mark.asyncio
async def test_uninventoried_owner_table_rolls_back_instead_of_false_success(
    erased_pair,
):
    database, redis, roots, records = erased_pair
    uid = records[0][0].id
    await database.execute_raw("CREATE TABLE erasure_fixture_unknown(user_id text)")
    try:
        await database.execute_raw(
            "INSERT INTO erasure_fixture_unknown VALUES($1)", uid
        )
        with pytest.raises(RuntimeError, match="Erasure residual"):
            await data_reset.hard_delete_user_data(uid)
        assert (await database.user.find_unique(where={"id": uid})).status == "deleting"
        assert await database.message.find_unique(where={"id": records[0][4].id})
        assert await database.user.find_unique(where={"id": records[1][0].id})
    finally:
        await database.execute_raw("DROP TABLE erasure_fixture_unknown")


@pytest.mark.asyncio
async def test_media_disk_failure_retains_retry_handles_and_other_files(
    erased_pair, monkeypatch
):
    database, redis, roots, records = erased_pair
    uid = records[0][0].id
    other = records[1][0].id
    blocked = roots["offline"] / (uid + "_audio.m4a")
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("synthetic read-only file")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    with pytest.raises(RuntimeError, match="delete user media file"):
        await data_reset.hard_delete_user_data(uid)
    assert (await database.user.find_unique(where={"id": uid})).status == "deleting"
    assert await database.message.find_unique(where={"id": records[0][4].id})
    assert blocked.exists()
    assert (roots["offline"] / (other + "_audio.m4a")).read_bytes() == b"owned audio"
    monkeypatch.setattr(Path, "unlink", unlink)
    assert (await data_reset.hard_delete_user_data(uid))["user"] == 1
    assert not blocked.exists()


def _offering_id() -> str:
    return uuid4().hex


async def _insert_offering(database, user_id: str, agent_id: str, conversation_id: str) -> str:
    offering_id = _offering_id()
    await database.execute_raw(
        """
        INSERT INTO user_offerings (
            id, user_id, agent_id, conversation_id, kind,
            ticket_amount, agent_value_yuan, status
        )
        VALUES ($1, $2, $3, $4, 'gift', 1, 1, 'sent')
        """,
        offering_id,
        user_id,
        agent_id,
        conversation_id,
    )
    return offering_id


@pytest.mark.asyncio
async def test_conversation_set_null_does_not_stick_deleting_account(erased_pair):
    """ON DELETE SET NULL must not abort erasure once the owner is deleting.

    user_offerings.conversation_id uses SET NULL, and the owner guard fires on
    that column. Without a transaction-local erasure marker the delete rolls
    back and the WeChat identity stays attached to a deleting account.
    """
    database, _, _, records = erased_pair
    target, survivor = records
    uid = target[0].id
    other = survivor[0].id
    owned = await _insert_offering(database, uid, target[1].id, target[3].id)
    peer = await _insert_offering(database, other, survivor[1].id, survivor[3].id)
    try:
        await database.execute_raw("UPDATE users SET status='deleting' WHERE id=$1", uid)
        with pytest.raises(RawQueryError, match="Account unavailable"):
            await database.execute_raw(
                "UPDATE user_offerings SET conversation_id=NULL WHERE id=$1",
                owned,
            )
        # Rebinding the row to someone else stays refused even inside erasure.
        with pytest.raises(RawQueryError, match="Account unavailable"):
            async with database.tx() as tx:
                await tx.execute_raw(
                    "SELECT set_config($1, $2, true)",
                    user_erasure.ERASURE_USER_SETTING,
                    uid,
                )
                await tx.execute_raw(
                    "UPDATE user_offerings SET user_id=$2 WHERE id=$1",
                    owned,
                    other,
                )
        await database.execute_raw("UPDATE users SET status='active' WHERE id=$1", uid)
        assert (await data_reset.hard_delete_user_data(uid))["user"] == 1
        assert await database.user.find_unique(where={"id": uid}) is None
        assert await database.query_raw(
            "SELECT id FROM user_offerings WHERE id=$1", owned
        ) == []
        peer_row = await database.query_raw(
            "SELECT user_id, conversation_id FROM user_offerings WHERE id=$1",
            peer,
        )
        assert peer_row[0]["user_id"] == other
        assert peer_row[0]["conversation_id"] == survivor[3].id
    finally:
        await database.execute_raw(
            "UPDATE users SET status='active' WHERE id=$1 AND status='deleting'",
            uid,
        )


@pytest.mark.asyncio
async def test_media_write_fence_serializes_delete_and_refuses_late_upload(erased_pair):
    import asyncio
    from fastapi import HTTPException

    database, redis, roots, records = erased_pair
    uid = records[0][0].id
    entered = asyncio.Event()
    release = asyncio.Event()
    late = roots["capsule"] / (uid + "_inflight.m4a")

    async def upload():
        async with user_erasure.owner_write_scope(uid, database=database):
            entered.set()
            await release.wait()
            late.write_bytes(b"in-flight bytes")

    task = asyncio.create_task(upload())
    await entered.wait()
    deletion = asyncio.create_task(data_reset.hard_delete_user_data(uid))
    await asyncio.sleep(0.1)
    assert not deletion.done()
    release.set()
    await task
    await deletion
    assert not late.exists()
    with pytest.raises(HTTPException) as error:
        async with user_erasure.owner_write_scope(uid, database=database):
            late.write_bytes(b"must never be written")
    assert error.value.status_code == 401 and not late.exists()


@pytest.mark.asyncio
async def test_wrong_owner_attachment_cannot_delete_another_users_file(erased_pair):
    database, redis, roots, records = erased_pair
    uid = records[0][0].id
    other = records[1][0].id
    foreign = other + "_audio.m4a"
    await database.chatmessageattachment.create(
        data={
            "userId": uid,
            "conversationId": records[0][3].id,
            "storageKey": foreign,
            "url": "/chat/media/" + foreign,
            "mime": "image/jpeg",
            "size": 11,
        }
    )
    with pytest.raises(RuntimeError, match="different owner"):
        await data_reset.hard_delete_user_data(uid)
    assert await database.user.find_unique(where={"id": uid})
    assert (roots["chat"] / foreign).read_bytes() == b"owned audio"


@pytest.mark.asyncio
async def test_deleted_and_deleting_accounts_reject_original_jwt(
    erased_pair, monkeypatch
):
    from fastapi import FastAPI, Depends
    from httpx import AsyncClient, ASGITransport
    from app.api import jwt_auth
    from app.services.auth import create_jwt

    database, redis, roots, records = erased_pair
    uid = records[0][0].id
    other = records[1][0].id
    app = FastAPI()

    @app.get("/probe")
    async def probe(user=Depends(jwt_auth.require_user)):
        return {"id": user["sub"]}

    headers = {"Authorization": "Bearer " + create_jwt(uid, "user")}
    peer = {"Authorization": "Bearer " + create_jwt(other, "user")}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://owned-fixture"
    ) as client:
        assert (await client.get("/probe", headers=headers)).status_code == 200
        await user_erasure.begin_erasure(database, uid)
        assert (await client.get("/probe", headers=headers)).status_code == 401
        await data_reset.hard_delete_user_data(uid)
        assert (await client.get("/probe", headers=headers)).status_code == 401
        assert (await client.get("/probe", headers=peer)).status_code == 200


@pytest.mark.asyncio
async def test_running_private_job_blocks_false_success_and_can_be_retried(erased_pair):
    database, redis, roots, records = erased_pair
    uid = records[0][0].id
    job = uuid4().hex
    await redis.hset(
        "runtime:job:" + job,
        mapping={"status": "running", "payload": json.dumps({"user_id": uid})},
    )
    with pytest.raises(RuntimeError, match="running background job"):
        await data_reset.hard_delete_user_data(uid)
    assert await database.message.find_unique(where={"id": records[0][4].id})
    await redis.hset("runtime:job:" + job, "status", "queued")
    assert (await data_reset.hard_delete_user_data(uid))["user"] == 1


@pytest.mark.asyncio
async def test_last_active_admin_guard_counts_deleting_peer_as_unavailable(erased_pair):
    database, redis, roots, records = erased_pair
    first, second = records[0][0].id, records[1][0].id
    await database.user.update(where={"id": first}, data={"role": "admin"})
    await database.user.update(where={"id": second}, data={"role": "admin"})
    await user_erasure.begin_erasure(database, first)
    with pytest.raises(RuntimeError, match="last active administrator"):
        await user_erasure.begin_erasure(database, second)
    assert (await database.user.find_unique(where={"id": second})).status == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["agent", "memory_collision"])
async def test_malformed_cross_owner_binding_cannot_cascade_into_peer(
    erased_pair, case
):
    database, _, roots, records = erased_pair
    target, peer = records
    foreign_id = uuid4().hex if case == "agent" else target[5].id
    try:
        if case == "agent":
            await database.conversation.create(
                data={"id": foreign_id, "userId": peer[0].id, "agentId": target[1].id}
            )
        else:
            await database.aimemory.create(
                data={
                    "id": foreign_id,
                    "userId": peer[0].id,
                    "content": "synthetic peer memory",
                }
            )
        with pytest.raises(RuntimeError, match="Cross-owner"):
            await data_reset.hard_delete_user_data(target[0].id)
        assert await database.user.find_unique(where={"id": peer[0].id})
        assert await database.message.find_unique(where={"id": target[4].id})
        assert (
            roots["chat"] / (peer[0].id + "_orphan.jpg")
        ).read_bytes() == b"owned image"
        row = (
            await database.conversation.find_unique(where={"id": foreign_id})
            if case == "agent"
            else await database.aimemory.find_unique(where={"id": foreign_id})
        )
        assert row.userId == peer[0].id
    finally:
        model = database.conversation if case == "agent" else database.aimemory
        await model.delete_many(where={"id": foreign_id})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["avatar", "feedback"])
async def test_unavailable_private_media_mount_preserves_sql_retry_source(
    erased_pair, kind
):
    database, _, roots, records = erased_pair
    root = roots[kind]
    detached = root.with_name(kind + "-detached")
    root.rename(detached)
    root.write_bytes(b"synthetic unavailable mount")
    try:
        with pytest.raises(RuntimeError, match="media directory is unavailable"):
            await data_reset.hard_delete_user_data(records[0][0].id)
        assert await database.message.find_unique(where={"id": records[0][4].id})
        assert (
            await database.user.find_unique(where={"id": records[0][0].id})
        ).status == "deleting"
        assert (
            detached / (records[1][0].id + "_orphan.jpg")
        ).read_bytes() == b"owned image"
    finally:
        root.unlink()
        detached.rename(root)


@pytest.mark.asyncio
async def test_healthy_private_writes_are_erased_without_changing_peer(
    erased_pair, monkeypatch
):
    from fastapi import HTTPException
    from app.services.runtime import cache
    from app.services.relationship import consistency
    from app.services.notifications import presence

    database, redis, _, records = erased_pair
    for module in (cache, consistency, presence):
        monkeypatch.setattr(module, "get_redis", AsyncMock(return_value=redis))
    for user, _, workspace, conversation, _, memory, _ in records:
        await redis.delete("context_window:" + conversation.id)
        await consistency.push_to_context_window(
            conversation.id, "user", "synthetic context"
        )
        await cache.bump_cache_version(user.id, workspace.id)
        await cache.cache_set_retrieval(
            "synthetic query",
            user.id,
            {"memories": [{"id": memory.id, "text": "private"}]},
            workspace.id,
        )
        await cache.cache_set_graph_context(
            user.id, {"context": "synthetic graph"}, workspace.id
        )
        await presence.update_presence(
            user_id=user.id,
            device_id="synthetic-device",
            foreground=True,
            workspace_id=workspace.id,
            conversation_id=conversation.id,
        )
        await presence.record_ws_online(user.id, "synthetic-connection")
        await redis.hset(
            "memory:processing:observations:data",
            user.id,
            json.dumps({"user_id": user.id}),
        )
        await redis.zadd("memory:processing:observations:index", {user.id: 1})
        assert await cache.get_cache_version("ret", user.id, workspace.id) == 1
        assert (await cache.cache_retrieval("synthetic query", user.id, workspace.id))[
            "memories"
        ][0]["id"] == memory.id
        assert await cache.cache_graph_context(user.id, workspace.id) == {
            "context": "synthetic graph"
        }
    target, peer = records
    uid, wid, cid = target[0].id, target[2].id, target[3].id
    peer_before = await cache.cache_retrieval("synthetic query", peer[0].id, peer[2].id)
    await data_reset.hard_delete_user_data(uid)
    assert await cache.cache_retrieval("synthetic query", uid, wid) is None
    assert await cache.cache_graph_context(uid, wid) is None
    assert await cache.get_cache_version("ret", uid, wid) == 0
    assert not await redis.exists(
        "context_window:" + cid, presence._key(uid, "synthetic-device")
    )
    assert (
        await redis.zscore(presence._ONLINE_WS_ZKEY, uid + "|synthetic-connection")
        is None
    )
    assert await redis.hget("memory:processing:observations:data", uid) is None
    assert await redis.zscore("memory:processing:observations:index", uid) is None
    assert (
        await cache.cache_retrieval("synthetic query", peer[0].id, peer[2].id)
        == peer_before
    )
    assert await cache.cache_graph_context(peer[0].id, peer[2].id) == {
        "context": "synthetic graph"
    }
    assert await consistency.get_context_window(peer[3].id) == [
        {"role": "user", "content": "synthetic context"}
    ]
    assert await redis.exists(presence._key(peer[0].id, "synthetic-device"))
    assert await redis.hget("memory:processing:observations:data", peer[0].id)
    await cache.cache_set_retrieval("late", uid, {"private": "must not return"}, wid)
    await cache.cache_set_graph_context(uid, {"private": "must not return"}, wid)
    with pytest.raises(HTTPException):
        await consistency.push_to_context_window(cid, "user", "must not return")
    assert not [key async for key in redis.scan_iter(match="*" + uid + "*")]
    await data_reset.hard_delete_user_data(peer[0].id)


@pytest.mark.asyncio
async def test_late_private_producers_cannot_recreate_deleted_data(
    erased_pair, monkeypatch
):
    from fastapi import HTTPException
    from app.services.runtime import job_queue, cache
    from app.services.chat_links import retention
    from app.services.notifications import presence

    database, redis, roots, records = erased_pair
    user, agent, _, conversation, message, _, _ = records[0]
    monkeypatch.setattr(job_queue, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(presence, "get_redis", AsyncMock(return_value=redis))
    payloads = [
        {"user_id": user.id},
        {"agent_id": agent.id},
        {"conversation_id": conversation.id},
        {"message_id": message.id},
    ]
    for deleted in (False, True):
        if deleted:
            await data_reset.hard_delete_user_data(user.id)
        else:
            await user_erasure.begin_erasure(database, user.id)
        before = set(roots["chat"].iterdir())
        with pytest.raises(HTTPException):
            async with user_erasure.conversation_write_scope(
                conversation.id, database=database
            ):
                await redis.set("topics:" + conversation.id, "must not return")
        with pytest.raises(HTTPException):
            await retention.write_cover(
                user_id=user.id,
                conversation_id=conversation.id,
                blob=b"late image",
                mime="image/jpeg",
                database=database,
            )
        assert set(roots["chat"].iterdir()) == before
        await cache.bump_cache_version(user.id, records[0][2].id)
        assert not await redis.exists(
            cache._version_key("ret", user.id, records[0][2].id)
        )
        assert not await redis.exists(
            cache._version_key("graph", user.id, records[0][2].id)
        )
        await presence.record_online(user.id)
        await presence.record_ws_online(user.id, "late-connection")
        await presence.update_presence(
            user_id=user.id, device_id="late-device", foreground=True
        )
        assert not await redis.exists(presence._key(user.id, "late-device"))
        assert await redis.zscore(presence._ONLINE_HB_ZKEY, user.id) is None
        assert (
            await redis.zscore(presence._ONLINE_WS_ZKEY, user.id + "|late-connection")
            is None
        )
        for index, payload in enumerate(payloads):
            key = f"late-producer:{user.id}:{index}"
            with pytest.raises(HTTPException):
                await job_queue.enqueue_runtime_job(
                    "synthetic", payload, idempotency_key=key
                )
            assert not await redis.exists("runtime:job_idem:" + key)


@pytest.mark.asyncio
async def test_cross_owner_trace_binding_refuses_deletion(erased_pair):
    database, _, roots, records = erased_pair
    target, peer = records
    child = uuid4().hex
    await database.tracerun.create(
        data={
            "id": child,
            "traceId": target[6],
            "parentId": target[6],
            "name": "synthetic-misbound",
            "runType": "llm",
            "startedAt": target[4].createdAt,
            "extraJson": Json({"metadata": {"owner_user_id": peer[0].id}}),
        }
    )
    with pytest.raises(RuntimeError, match="Cross-owner Trace"):
        await data_reset.hard_delete_user_data(target[0].id)
    assert await database.tracerun.find_unique(where={"id": child})
    assert await database.user.find_unique(where={"id": peer[0].id})
    assert (roots["chat"] / (peer[0].id + "_orphan.jpg")).read_bytes() == b"owned image"


@pytest.mark.asyncio
async def test_cross_owner_cover_reference_preserves_peer_message_and_media(
    erased_pair,
):
    database, _, roots, records = erased_pair
    target, peer = records
    key = target[0].id + "_orphan.jpg"
    metadata = {
        "link_card": {"image_url": data_reset.chat_media_storage.media_url(key)}
    }
    await database.message.update(
        where={"id": peer[4].id}, data={"metadata": Json(metadata)}
    )
    with pytest.raises(RuntimeError, match="Cross-owner cover"):
        await data_reset.hard_delete_user_data(target[0].id)
    assert (
        await database.message.find_unique(where={"id": peer[4].id})
    ).metadata == metadata
    assert (roots["chat"] / key).read_bytes() == b"owned image"


@pytest.mark.asyncio
async def test_shared_identity_cannot_erase_another_accounts_auth_state(erased_pair):
    database, redis, _, records = erased_pair
    shared = "synthetic-union-" + uuid4().hex
    for owner in records:
        await database.authidentity.create(
            data={
                "userId": owner[0].id,
                "provider": "wechat",
                "providerAccountId": "synthetic-wechat-" + owner[0].id,
                "unionid": shared,
            }
        )
    key = "wechat:state:" + shared
    await redis.set(key, "synthetic shared state")
    try:
        with pytest.raises(RuntimeError, match="Cross-owner identity"):
            await data_reset.hard_delete_user_data(records[0][0].id)
        assert await redis.get(key) == "synthetic shared state"
        assert await database.user.find_unique(where={"id": records[1][0].id})
    finally:
        await redis.delete(key)


@pytest.mark.asyncio
async def test_nullable_private_usage_and_late_vectors_are_erased(erased_pair):
    database, _, _, records = erased_pair
    target, peer = records
    identities = [uuid4().hex for _ in range(3)]
    detached_trace = uuid4().hex
    base = {
        "inputTokens": 0,
        "outputTokens": 0,
        "tokensByModel": Json({}),
        "costCny": 0,
        "callCount": 0,
    }
    vector = "[" + ",".join(["0"] * 1024) + "]"
    try:
        await database.llmusage.create(data={**base, "id": identities[2]})
        await database.tracerun.create(
            data={
                "id": detached_trace,
                "traceId": detached_trace,
                "name": "synthetic-unbound",
                "runType": "chain",
                "startedAt": target[4].createdAt,
                "inputsJson": Json({}),
            }
        )
        for ident, owner in zip(identities[:2], (target, peer)):
            await database.llmusage.create(
                data={
                    **base,
                    "id": ident,
                    "conversationId": owner[3].id,
                    "traceId": detached_trace if owner == target else None,
                }
            )
        await database.execute_raw(
            "INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector)",
            target[5].id,
            vector,
        )
        await user_erasure.begin_erasure(database, target[0].id)
        with pytest.raises(ForeignKeyViolationError):
            await database.llmusage.update(
                where={"id": identities[2]}, data={"conversationId": target[3].id}
            )
        with pytest.raises(ForeignKeyViolationError):
            await database.llmusage.create(
                data={**base, "conversationId": target[3].id}
            )
        with pytest.raises(Exception, match="Memory owner unavailable"):
            await database.execute_raw(
                "UPDATE memory_embeddings SET embedding=$2::extensions.vector WHERE memory_id=$1",
                target[5].id,
                vector,
            )
        await data_reset.hard_delete_user_data(target[0].id)
        assert await database.llmusage.find_unique(where={"id": identities[0]}) is None
        assert await database.llmusage.find_unique(where={"id": identities[1]})
        assert await database.tracerun.find_unique(where={"id": detached_trace}) is None
        with pytest.raises(ForeignKeyViolationError):
            await database.llmusage.create(
                data={**base, "conversationId": target[3].id}
            )
        with pytest.raises(Exception, match="Memory owner unavailable"):
            await database.execute_raw(
                "INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector)",
                target[5].id,
                vector,
            )
        # No-resource system-wide metering remains valid.
        assert await database.llmusage.find_unique(where={"id": identities[2]})
    finally:
        await database.llmusage.delete_many(where={"id": {"in": identities}})
        await database.tracerun.delete_many(where={"id": detached_trace})


@pytest.mark.asyncio
async def test_unknown_indirect_table_cannot_silently_keep_conversation_data(
    erased_pair,
):
    database, _, _, records = erased_pair
    target, peer = records
    table = "erasure_indirect_" + uuid4().hex
    await database.execute_raw(
        f"CREATE TABLE {table}(conversation_id text, private_content text)"
    )
    try:
        await database.execute_raw(
            f"INSERT INTO {table} VALUES($1,$2)",
            target[3].id,
            "synthetic private content",
        )
        with pytest.raises(RuntimeError, match="Erasure resource residual"):
            await data_reset.hard_delete_user_data(target[0].id)
        assert await database.message.find_unique(where={"id": target[4].id})
        assert await database.user.find_unique(where={"id": peer[0].id})
    finally:
        await database.execute_raw(f"DROP TABLE {table}")


@pytest.mark.asyncio
async def test_concurrent_profile_updates_do_not_upgrade_shared_locks(
    erased_pair, monkeypatch
):
    import asyncio
    from app.services import user_profile

    database, _, _, records = erased_pair
    user = records[0][0]
    monkeypatch.setattr(user_profile, "db", database)
    results = await asyncio.gather(
        user_profile.apply_profile_update(user, display_name="first update"),
        user_profile.apply_profile_update(user, display_name="second update"),
    )
    assert all(result.id == user.id for result in results)
    assert (await database.user.find_unique(where={"id": user.id})).displayName in {
        "first update",
        "second update",
    }
