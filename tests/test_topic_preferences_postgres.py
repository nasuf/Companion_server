"""Real SQL source integrity, concurrency, expiry, privacy and admin CAS."""

import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import uuid4

import pytest
from prisma import Json
from app.services.time_domain import topic_store as store
from app.services.time_domain.followup import FollowupPolicy, guard_optional_questions
from tests.test_memory_evidence_postgres import origins

NOW = datetime.now(timezone.utc)


@pytest.fixture
async def topics(origins, monkeypatch):
    database, uid, wid, _, message, *_ = origins
    monkeypatch.setattr(store, "db", database)

    async def send(text, when=None, **extra):
        msg = await database.message.create(
            data={
                "conversationId": message.conversationId,
                "role": "user",
                "content": text,
                "createdAt": when or datetime.now(timezone.utc),
                **extra,
            }
        )
        await store.record_commands(
            user_id=uid, conversation_id=message.conversationId, message_ids={msg.id}
        )
        return msg

    return database, uid, wid, message, send, origins


async def test_stop_survives_new_reader_and_resume_is_scope_specific(topics):
    database, uid, wid, original, send, _ = topics
    first = await send("以后别问面试了")
    await send("以后别问工资了")
    await send("我不喜欢一直被问")
    later = datetime.now(timezone.utc) + timedelta(days=60)
    snapshot = await store.load_snapshot(user_id=uid, workspace_id=wid, now=later)
    assert len(snapshot.items) == 3
    await send("现在可以问面试了")
    await send("你现在可以问我了")
    snapshot = await store.load_snapshot(user_id=uid, workspace_id=wid, now=later)
    assert {(i.scope, i.topic) for i in snapshot.items} == {
        ("topic", "工资"),
        ("global", ""),
    }
    policy = FollowupPolicy(topic_snapshot=snapshot)
    assert guard_optional_questions(["工资多少？", "吃饭了吗？", "我在这呢。"], policy)[
        0
    ] == ["我在这呢。"]
    await database.message.delete(where={"id": first.id})
    assert (
        await database.query_raw(
            "SELECT id FROM conversation_topic_events WHERE source_message_id=$1",
            first.id,
        )
        == []
    )


async def test_replay_concurrency_is_idempotent_and_late_resume_cannot_win(topics):
    database, uid, wid, original, send, _ = topics
    newer = await send("以后别问面试了")
    await asyncio.gather(
        *(
            store.record_commands(
                user_id=uid,
                conversation_id=original.conversationId,
                message_ids={newer.id},
            )
            for _ in range(6)
        )
    )
    await send("可以问面试了", newer.createdAt - timedelta(hours=1))
    rows = await database.query_raw(
        "SELECT revision,action FROM conversation_topics WHERE workspace_id=$1", wid
    )
    assert rows == [{"revision": 1, "action": "stop"}]
    count = (
        await database.query_raw(
            "SELECT count(*) AS count FROM conversation_topic_events WHERE user_id=$1",
            uid,
        )
    )[0]["count"]
    assert count == 2  # one accepted, one audited stale, no replay events


async def test_tied_sources_do_not_grant_permission_and_same_source_last_clause_wins(
    topics,
):
    database, uid, wid, original, send, _ = topics
    anchor = datetime.now(timezone.utc)
    await send("以后别问面试了", anchor)
    await send("现在可以问面试了", anchor)
    snapshot = await store.load_snapshot(
        user_id=uid, workspace_id=wid, now=anchor + timedelta(seconds=1)
    )
    assert len(snapshot.items) == 1
    await send("以后别问面试了。现在可以问面试了", anchor + timedelta(seconds=2))
    assert not (
        await store.load_snapshot(
            user_id=uid, workspace_id=wid, now=anchor + timedelta(seconds=3)
        )
    ).items


async def test_source_edit_deletion_takeover_and_generation_do_not_resurrect(topics):
    database, uid, wid, original, send, origins = topics
    msg = await send("以后别问工资了")
    await database.message.update(where={"id": msg.id}, data={"content": "工资可以聊"})
    snapshot = await store.load_snapshot(
        user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
    )
    assert snapshot.status == "source_unavailable" and not snapshot.items
    await store.record_commands(
        user_id=uid, conversation_id=original.conversationId, message_ids={msg.id}
    )
    assert not (
        await store.load_snapshot(
            user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
        )
    ).items
    await database.message.delete(where={"id": msg.id})
    assert not await database.query_raw(
        "SELECT id FROM conversation_topics WHERE workspace_id=$1", wid
    )
    human = await database.message.create(
        data={
            "conversationId": original.conversationId,
            "role": "user",
            "content": "以后别问工资了",
            "metadata": Json({"human_takeover": True}),
        }
    )
    with pytest.raises(ValueError, match="source_unavailable"):
        await store.record_commands(
            user_id=uid, conversation_id=original.conversationId, message_ids={human.id}
        )
    await send("以后别问面试了")
    await database.execute_raw(
        "UPDATE chat_workspaces SET execution_generation=$2 WHERE id=$1",
        wid,
        str(uuid4()),
    )
    assert not (
        await store.load_snapshot(
            user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
        )
    ).items


async def test_expiry_uses_statement_clock_not_delivery(topics):
    _, uid, wid, _, send, _ = topics
    await send("暂时别问面试了", datetime.now(timezone.utc) - timedelta(hours=4))
    assert not (
        await store.load_snapshot(
            user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
        )
    ).items


async def test_wrong_scope_and_admin_cas_with_original_source_preserved(topics):
    database, uid, wid, original, send, origins = topics
    msg = await send("以后别问面试了")
    page = await store.admin_page(user_id=uid, workspace_id=wid)
    item = page["items"][0]
    assert item["effective"] == "active" and item["source_message_id"] == msg.id
    with pytest.raises(LookupError):
        await store.admin_page(user_id=origins[7][-1], workspace_id=wid)
    with pytest.raises(LookupError):
        await store.admin_correct(
            actor_id=uid,
            user_id=uid,
            workspace_id=wid,
            ident=item["id"],
            expected_revision=1,
            action="resume",
            reason="test",
        )
    admin = await database.user.create(
        data={"username": "topic-admin-" + uuid4().hex, "role": "admin"}
    )
    try:
        result = await store.admin_correct(
            actor_id=admin.id,
            user_id=uid,
            workspace_id=wid,
            ident=item["id"],
            expected_revision=1,
            action="resume",
            reason="合成管理员复核纠正",
        )
        assert result["revision"] == 2
        assert (
            await database.message.find_unique(where={"id": msg.id})
        ).content == msg.content
        with pytest.raises(ValueError, match="revision_conflict"):
            await store.admin_correct(
                actor_id=admin.id,
                user_id=uid,
                workspace_id=wid,
                ident=item["id"],
                expected_revision=1,
                action="stop",
                reason="stale",
            )
        assert not (
            await store.load_snapshot(
                user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
            )
        ).items
    finally:
        await database.user.delete(where={"id": admin.id})


async def test_question_and_response_observations_are_source_bound_idempotent_and_erasable(
    topics,
):
    database, uid, wid, original, send, _ = topics
    await send("以后别问面试了")
    await send("可以问面试了")
    question = await database.message.create(
        data={
            "conversationId": original.conversationId,
            "role": "assistant",
            "content": "面试结果怎么样？",
        }
    )
    answer = await database.message.create(
        data={
            "conversationId": original.conversationId,
            "role": "user",
            "content": "面试结果还没出",
        }
    )
    ids = {question.id, answer.id}
    assert (
        await store.observe_messages(
            user_id=uid, conversation_id=original.conversationId, message_ids=ids
        )
        == 2
    )
    assert (
        await store.observe_messages(
            user_id=uid, conversation_id=original.conversationId, message_ids=ids
        )
        == 0
    )
    row = (await store.admin_page(user_id=uid, workspace_id=wid))["items"][0]
    assert row["asked_count"] == 1 and row["answered_at"] and row["state"] == "open"
    await database.message.update(
        where={"id": question.id}, data={"content": "天气真好"}
    )
    row = (await store.admin_page(user_id=uid, workspace_id=wid))["items"][0]
    assert row["asked_count"] == 0 and row["last_agent_question"] is None
    await database.message.delete(where={"id": answer.id})
    assert (await store.admin_page(user_id=uid, workspace_id=wid))["items"][0][
        "answered_at"
    ] is None


async def test_admin_http_scope_auth_pagination_and_mutation_contract(
    topics, monkeypatch
):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.admin import memory_repairs
    from app.services.auth import create_jwt

    database, uid, wid, original, send, _ = topics
    await send("以后别问工资了")
    admin = await database.user.create(
        data={"username": "topic-http-" + uuid4().hex, "role": "admin"}
    )
    app = FastAPI()
    app.include_router(memory_repairs.router)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://synthetic"
        ) as client:
            path = "/admin-api/memory-repairs/topics/page"
            assert (
                await client.get(path, params={"user_id": uid, "workspace_id": wid})
            ).status_code == 401
            headers = {"Authorization": "Bearer " + create_jwt(admin.id, "admin")}
            page = await client.get(
                path, params={"user_id": uid, "workspace_id": wid}, headers=headers
            )
            assert page.status_code == 200, page.text
            item = page.json()["items"][0]
            response = await client.post(
                "/admin-api/memory-repairs/topics/" + item["id"] + "/correction",
                headers=headers,
                json={
                    "user_id": uid,
                    "workspace_id": wid,
                    "expected_revision": item["revision"],
                    "action": "resume",
                    "reason": "合成核对来源",
                },
            )
            assert response.status_code == 200, response.text
            assert (
                await client.get(
                    path,
                    params={"user_id": uid, "workspace_id": wid, "limit": 51},
                    headers=headers,
                )
            ).status_code == 422
    finally:
        await database.user.delete(where={"id": admin.id})


async def test_large_aggregated_turn_retains_individual_source_clocks(topics):
    database, uid, wid, original, send, _ = topics
    anchor = datetime.now(timezone.utc)
    ids = set()
    for i in range(40):
        text = (
            "以后别问面试了"
            if i == 0
            else "现在可以问面试了" if i == 39 else "普通合成消息" + str(i)
        )
        msg = await database.message.create(
            data={
                "conversationId": original.conversationId,
                "role": "user",
                "content": text,
                "createdAt": anchor + timedelta(seconds=i),
            }
        )
        ids.add(msg.id)
    assert (
        await store.record_commands(
            user_id=uid, conversation_id=original.conversationId, message_ids=ids
        )
        == 2
    )
    snapshot = await store.load_snapshot(
        user_id=uid, workspace_id=wid, now=anchor + timedelta(days=1)
    )
    assert not snapshot.items
    assert (
        await store.record_commands(
            user_id=uid, conversation_id=original.conversationId, message_ids=ids
        )
        == 0
    )


async def test_deleted_latest_preference_does_not_revive_prior_source(topics):
    database, uid, wid, original, send, _ = topics
    older = await send("以后别问面试了")
    newer = await send("现在可以问面试了")
    await database.message.delete(where={"id": newer.id})
    assert not await database.query_raw(
        "SELECT id FROM conversation_topics WHERE workspace_id=$1", wid
    )
    for _ in range(2):
        assert (
            await store.record_commands(
                user_id=uid,
                conversation_id=original.conversationId,
                message_ids={older.id},
            )
            == 0
        )
    assert not (
        await store.load_snapshot(
            user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
        )
    ).items
    # A genuinely new explicit request is still accepted after deletion.
    await send("以后别问面试了")
    assert (
        len(
            (
                await store.load_snapshot(
                    user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
                )
            ).items
        )
        == 1
    )


async def test_aggregate_candidate_gate_does_not_require_joined_text_to_parse(topics):
    database, uid, wid, original, send, _ = topics
    stop = await database.message.create(
        data={
            "conversationId": original.conversationId,
            "role": "user",
            "content": "以后别问工资了",
        }
    )
    ordinary = await database.message.create(
        data={
            "conversationId": original.conversationId,
            "role": "user",
            "content": "我先忙了",
        }
    )
    result = await store.record_turn_preferences(
        user_id=uid,
        conversation_id=original.conversationId,
        message_ids={stop.id, ordinary.id},
        current_message="以后别问工资了我先忙了",
        statement_at=datetime.now(timezone.utc),
    )
    assert result["status"] == "recorded" and result["applied_commands"] == 1
    snapshot = await store.load_snapshot(
        user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
    )
    assert snapshot.items[0].topic == "工资"


async def test_bounded_observations_expiry_and_preference_survival(topics):
    database, uid, wid, original, send, _ = topics
    stop = await send("以后别问面试了")
    anchor = datetime.now(timezone.utc) - timedelta(hours=1)
    ids = []
    for index in range(70):
        msg = await database.message.create(
            data={
                "conversationId": original.conversationId,
                "role": "assistant",
                "content": "面试怎么样？",
                "createdAt": anchor + timedelta(seconds=index),
            }
        )
        ids.append(msg.id)
        await store.observe_messages(
            user_id=uid, conversation_id=original.conversationId, message_ids={msg.id}
        )
    rows = await database.query_raw(
        "SELECT count(*)::int AS n FROM conversation_topic_observations WHERE user_id=$1",
        uid,
    )
    assert rows[0]["n"] == 64
    page = await store.admin_page(user_id=uid, workspace_id=wid)
    assert (
        page["items"][0]["asked_count"] == 64 and page["observation_window_days"] == 30
    )
    snapshot = await store.load_snapshot(
        user_id=uid, workspace_id=wid, now=anchor + timedelta(days=31)
    )
    assert len(snapshot.items) == 1 and snapshot.items[0].asked_count == 0
    # Old observations may be revisited but cannot grow the retained window.
    await store.observe_messages(
        user_id=uid, conversation_id=original.conversationId, message_ids={ids[0]}
    )
    assert (await store.admin_page(user_id=uid, workspace_id=wid))["items"][0][
        "asked_count"
    ] == 64
    assert (
        await database.message.find_unique(where={"id": stop.id})
    ).content == stop.content


async def test_inactive_projection_pruning_keeps_replay_fence_and_live_stop(topics):
    database, uid, wid, original, send, _ = topics
    old = datetime.now(timezone.utc) - timedelta(days=120)
    # An old temporary preference is pruned, but its source cannot recreate it.
    expired = await send("暂时别问工资了", old)
    assert not await database.query_raw(
        "SELECT id FROM conversation_topics WHERE workspace_id=$1", wid
    )
    assert (
        await store.record_commands(
            user_id=uid,
            conversation_id=original.conversationId,
            message_ids={expired.id},
        )
        == 0
    )
    permanent = await send("以后别问面试了", old)
    await send("今天别问加班了")
    snapshot = await store.load_snapshot(
        user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
    )
    assert any(
        i.topic == "面试" and i.source_id == permanent.id for i in snapshot.items
    )


async def test_audit_retention_does_not_let_old_source_override_current_decision(
    topics,
):
    database, uid, wid, original, send, _ = topics
    anchor = datetime.now(timezone.utc) - timedelta(hours=1)
    first = None
    for index in range(135):
        msg = await send(
            "以后别问面试了" if index % 2 == 0 else "可以问面试了",
            anchor + timedelta(seconds=index),
        )
        first = first or msg
    rows = await database.query_raw(
        "SELECT count(*)::int AS n FROM conversation_topic_events WHERE user_id=$1", uid
    )
    assert rows[0]["n"] == 128
    before = (await store.admin_page(user_id=uid, workspace_id=wid))["items"][0]
    assert (
        await store.record_commands(
            user_id=uid, conversation_id=original.conversationId, message_ids={first.id}
        )
        == 0
    )
    after = (await store.admin_page(user_id=uid, workspace_id=wid))["items"][0]
    assert (after["revision"], after["action"]) == (
        before["revision"],
        before["action"],
    )


async def test_background_observation_is_bounded_and_failure_is_best_effort(
    topics, monkeypatch, caplog
):
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    from app.services.time_domain.topic_observations import observe_reply_messages

    database, uid, wid, original, send, _ = topics
    assert (
        await store.observe_messages(
            user_id=uid, conversation_id=original.conversationId, message_ids=set()
        )
        == 0
    )
    assert (
        await store.observe_messages(
            user_id=uid,
            conversation_id=original.conversationId,
            message_ids={str(i) for i in range(9)},
        )
        == 0
    )
    await send("以后别问面试了")
    question = await database.message.create(
        data={
            "conversationId": original.conversationId,
            "role": "assistant",
            "content": "面试怎么样？",
        }
    )
    await observe_reply_messages(
        original.conversationId, {question.id}, database=database
    )
    assert (await store.admin_page(user_id=uid, workspace_id=wid))["items"][0][
        "asked_count"
    ] == 1
    await observe_reply_messages(
        "missing-conversation", {question.id}, database=database
    )
    await observe_reply_messages(
        original.conversationId,
        {question.id},
        database=SimpleNamespace(
            query_raw=AsyncMock(side_effect=RuntimeError("synthetic-error"))
        ),
    )
    assert "reply observation unavailable error=RuntimeError" in caplog.text


async def test_human_card_and_unrelated_messages_cannot_count_as_topic_questions(
    topics,
):
    database, uid, wid, original, send, _ = topics
    await send("以后别问面试了")
    ids = set()
    for text, metadata in [
        ("面试怎么样？", {"human_takeover": True}),
        ("面试怎么样？", {"component_card": {}}),
        ("面试挺辛苦", {}),
        ("吃饭了吗？", {}),
    ]:
        message = await database.message.create(
            data={
                "conversationId": original.conversationId,
                "role": "assistant",
                "content": text,
                "metadata": Json(metadata),
            }
        )
        ids.add(message.id)
    assert (
        await store.observe_messages(
            user_id=uid, conversation_id=original.conversationId, message_ids=ids
        )
        == 0
    )
    assert (await store.admin_page(user_id=uid, workspace_id=wid))["items"][0][
        "asked_count"
    ] == 0


async def test_identity_capacity_never_evicts_opt_out_or_deletion_fences(topics):
    from app.services.time_domain.topic_retention import MAX_TOPIC_IDENTITIES

    database, uid, wid, original, send, _ = topics
    await send("以后别问面试了")
    # Payload-free synthetic tombstones fill the bounded workspace ledger.
    await database.execute_raw(
        """INSERT INTO conversation_topic_fences(topic_id,workspace_id,decision_at,deleted)
        SELECT encode(sha256(convert_to($1||':'||n::text,'UTF8')),'hex'),$1,clock_timestamp(),TRUE
        FROM generate_series(1,$2::int) n""",
        wid,
        MAX_TOPIC_IDENTITIES - 1,
    )
    with pytest.raises(Exception, match="topic_identity_capacity_exhausted"):
        await send("以后别问工资了")
    # Updating an existing identity remains possible, but capacity is not
    # silently presented as permission to start unrecordable new topics.
    await send("现在可以问面试了")
    snapshot = await store.load_snapshot(
        user_id=uid, workspace_id=wid, now=datetime.now(timezone.utc)
    )
    assert snapshot.status == "capacity_exceeded" and snapshot.truncated
    page = await store.admin_page(user_id=uid, workspace_id=wid)
    assert page["identity_capacity"]["used"] == MAX_TOPIC_IDENTITIES
    assert page["identity_capacity"]["exhausted"]
    assert (
        await database.query_raw(
            "SELECT count(*) AS n FROM conversation_topic_fences WHERE workspace_id=$1",
            wid,
        )
    )[0]["n"] == MAX_TOPIC_IDENTITIES


async def test_admin_read_marks_rebound_source_unavailable(topics):
    database, uid, wid, original, send, all_origins = topics
    source = await send("以后别问面试了")
    foreign_conversation = all_origins[-1][-1]
    await database.message.update(
        where={"id": source.id}, data={"conversationId": foreign_conversation}
    )
    page = await store.admin_page(user_id=uid, workspace_id=wid)
    assert page["items"][0]["effective"] == "source_unavailable"
    assert not page["items"][0]["source_available"]
