"""Real migrations, lifecycle versions, ownership and short-commit races.

Only a named disposable loopback database is accepted. No providers or Redis.
"""
import asyncio
import os
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from prisma import Prisma

from app.services.runtime import execution_scope as scope


@pytest.fixture
async def flow():
    url = os.getenv("PROACTIVE_E2E_DATABASE_URL")
    if not url:
        pytest.skip("isolated PostgreSQL required")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1"}
    assert parsed.path == "/companion_proactive_e2e"
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    ids = {key: str(uuid4()) for key in ("owner", "admin", "stranger", "agent", "workspace", "conversation")}
    try:
        for key in ("owner", "admin", "stranger"):
            await database.user.create(data={"id": ids[key], "username": "scope-" + ids[key],
                                            "role": "admin" if key == "admin" else "user"})
        await database.aiagent.create(data={"id": ids["agent"], "userId": ids["owner"], "name": "Synthetic"})
        await database.chatworkspace.create(data={"id": ids["workspace"], "userId": ids["owner"], "agentId": ids["agent"]})
        await database.conversation.create(data={"id": ids["conversation"], "userId": ids["owner"],
                                                 "agentId": ids["agent"], "workspaceId": ids["workspace"]})
        bound = await scope.bind_conversation_scope(actor_user_id=ids["owner"], conversation_id=ids["conversation"], database=database)
        yield SimpleNamespace(db=database, ids=ids, bound=bound, url=url)
    finally:
        await database.message.delete_many(where={"conversationId": ids["conversation"]})
        await database.conversation.delete_many(where={"id": ids["conversation"]})
        await database.chatworkspace.delete_many(where={"id": ids["workspace"]})
        await database.aiagent.delete_many(where={"id": ids["agent"]})
        await database.user.delete_many(where={"id": {"in": [ids[k] for k in ("owner", "admin", "stranger")]}})
        await database.disconnect()


async def insert_reply(tx, flow, text="synthetic reply"):
    return await tx.message.create(data={"conversationId": flow.ids["conversation"], "role": "assistant", "content": text})


async def test_current_owner_and_database_admin_are_distinct(flow):
    bound = await scope.bind_conversation_scope(actor_user_id=flow.ids["admin"], conversation_id=flow.ids["conversation"], database=flow.db)
    assert bound.actor_user_id == flow.ids["admin"] and bound.owner_user_id == flow.ids["owner"]
    with pytest.raises(scope.ExecutionScopeUnavailable):
        await scope.bind_conversation_scope(actor_user_id=flow.ids["stranger"], conversation_id=flow.ids["conversation"], database=flow.db)
    await flow.db.user.update(where={"id": flow.ids["admin"]}, data={"role": "user"})
    with pytest.raises(scope.ExecutionScopeUnavailable):
        async with scope.scoped_transaction(bound, database=flow.db) as tx:
            await insert_reply(tx, flow)
    assert await flow.db.message.count(where={"conversationId": flow.ids["conversation"]}) == 0


@pytest.mark.parametrize("model,resource,change", [
    ("user", "owner", {"status": "archived"}),
    ("aiagent", "agent", {"status": "archived"}),
    ("chatworkspace", "workspace", {"status": "archived"}),
    ("conversation", "conversation", {"isDeleted": True}),
])
async def test_archive_restore_cannot_resurrect_old_execution(flow, model, resource, change):
    delegate = getattr(flow.db, model)
    await delegate.update(where={"id": flow.ids[resource]}, data=change)
    with pytest.raises(scope.ExecutionScopeUnavailable):
        await scope.revalidate_scope(flow.bound, database=flow.db)
    restored = {key: "active" if key == "status" else False for key in change}
    await delegate.update(where={"id": flow.ids[resource]}, data=restored)
    with pytest.raises(scope.ExecutionScopeExpired):
        async with scope.scoped_transaction(flow.bound, database=flow.db) as tx:
            await insert_reply(tx, flow)
    assert await flow.db.message.count(where={"conversationId": flow.ids["conversation"]}) == 0


@pytest.mark.parametrize("model,resource,data", [
    ("aiagent", "agent", {"userId": "stranger"}),
    ("chatworkspace", "workspace", {"userId": "stranger"}),
    ("conversation", "conversation", {"userId": "stranger"}),
])
async def test_independently_valid_foreign_keys_do_not_grant_cross_owner_access(flow, model, resource, data):
    await getattr(flow.db, model).update(where={"id": flow.ids[resource]}, data={key: flow.ids[value] for key, value in data.items()})
    with pytest.raises(scope.ExecutionScopeUnavailable):
        await scope.revalidate_scope(flow.bound, database=flow.db)


async def test_unrelated_profile_and_timestamp_updates_keep_scope_valid(flow):
    await flow.db.user.update(where={"id": flow.ids["owner"]}, data={"displayName": "new nickname"})
    await flow.db.aiagent.update(where={"id": flow.ids["agent"]}, data={"name": "new name"})
    await flow.db.conversation.update(where={"id": flow.ids["conversation"]}, data={"title": "new title"})
    await scope.revalidate_scope(flow.bound, database=flow.db)
    async with scope.scoped_transaction(flow.bound, database=flow.db) as tx:
        message = await insert_reply(tx, flow)
    assert (await flow.db.message.find_unique(where={"id": message.id})).content == "synthetic reply"


async def test_old_writer_omitting_generation_still_rotates_on_restore(flow):
    # Same shape as pre-migration SQL clients; no new field is supplied.
    await flow.db.execute_raw("UPDATE conversations SET is_deleted=true WHERE id=$1", flow.ids["conversation"])
    await flow.db.execute_raw("UPDATE conversations SET is_deleted=false WHERE id=$1", flow.ids["conversation"])
    with pytest.raises(scope.ExecutionScopeExpired):
        await scope.revalidate_scope(flow.bound, database=flow.db)


async def test_delete_recreate_with_same_id_and_explicit_old_generation_is_stale(flow):
    await flow.db.conversation.delete(where={"id": flow.ids["conversation"]})
    with pytest.raises(scope.ExecutionScopeUnavailable):
        await scope.revalidate_scope(flow.bound, database=flow.db)
    await flow.db.conversation.create(data={"id": flow.ids["conversation"], "userId": flow.ids["owner"],
        "agentId": flow.ids["agent"], "workspaceId": flow.ids["workspace"],
        "executionGeneration": flow.bound.conversation_generation})
    with pytest.raises(scope.ExecutionScopeExpired):
        await scope.revalidate_scope(flow.bound, database=flow.db)


async def test_manual_runtime_generation_invalidation(flow):
    await flow.db.execute_raw("UPDATE chat_workspaces SET execution_generation=gen_random_uuid()::text WHERE id=$1", flow.ids["workspace"])
    with pytest.raises(scope.ExecutionScopeExpired):
        await scope.revalidate_scope(flow.bound, database=flow.db)


async def test_callback_failure_rolls_back_all_sql_effects(flow):
    with pytest.raises(RuntimeError, match="synthetic abort"):
        async with scope.scoped_transaction(flow.bound, database=flow.db) as tx:
            await insert_reply(tx, flow)
            raise RuntimeError("synthetic abort")
    assert await flow.db.message.count(where={"conversationId": flow.ids["conversation"]}) == 0


@pytest.mark.parametrize("table,resource", [("users", "owner"), ("ai_agents", "agent"),
                                         ("chat_workspaces", "workspace"), ("conversations", "conversation")])
async def test_lifecycle_writer_waits_until_the_fenced_commit_finishes(flow, table, resource):
    other = Prisma(datasource={"url": flow.url}, http={"trust_env": False})
    await other.connect()
    try:
        async with scope.scoped_transaction(flow.bound, database=flow.db) as tx:
            # PostgreSQL NOWAIT independently proves the row is locked; not a
            # timing assertion that merely hopes the other connection has run.
            with pytest.raises(Exception, match="could not obtain lock"):
                await other.query_raw(f"SELECT id FROM {table} WHERE id=$1 FOR UPDATE NOWAIT", flow.ids[resource])
            await insert_reply(tx, flow)
        await other.chatworkspace.update(where={"id": flow.ids["workspace"]}, data={"status": "archived"})
        with pytest.raises(scope.ExecutionScopeUnavailable):
            await scope.revalidate_scope(flow.bound, database=flow.db)
        assert await flow.db.message.count(where={"conversationId": flow.ids["conversation"]}) == 1
    finally:
        await other.disconnect()


async def test_admin_actor_and_owner_are_both_locked_at_commit(flow):
    admin_scope = await scope.bind_conversation_scope(actor_user_id=flow.ids["admin"], conversation_id=flow.ids["conversation"], database=flow.db)
    other = Prisma(datasource={"url": flow.url}, http={"trust_env": False})
    await other.connect()
    try:
        async with scope.scoped_transaction(admin_scope, database=flow.db):
            for resource in ("admin", "owner"):
                with pytest.raises(Exception, match="could not obtain lock"):
                    await other.query_raw("SELECT id FROM users WHERE id=$1 FOR UPDATE NOWAIT", flow.ids[resource])
    finally:
        await other.disconnect()


async def test_missing_workspace_and_archived_timestamp_fail_closed(flow):
    await flow.db.conversation.update(where={"id": flow.ids["conversation"]}, data={"workspaceId": None})
    with pytest.raises(scope.ExecutionScopeUnavailable):
        await scope.revalidate_scope(flow.bound, database=flow.db)
    await flow.db.conversation.update(where={"id": flow.ids["conversation"]}, data={"workspaceId": flow.ids["workspace"]})
    await flow.db.execute_raw("UPDATE chat_workspaces SET archived_at=CURRENT_TIMESTAMP WHERE id=$1", flow.ids["workspace"])
    with pytest.raises(scope.ExecutionScopeUnavailable):
        await scope.bind_conversation_scope(actor_user_id=flow.ids["owner"], conversation_id=flow.ids["conversation"], database=flow.db)


async def test_concurrent_lifecycle_change_during_lock_wait_is_rechecked(flow):
    other = Prisma(datasource={"url": flow.url}, http={"trust_env": False})
    await other.connect()
    try:
        async with other.tx() as changing:
            await changing.conversation.update(where={"id": flow.ids["conversation"]}, data={"isDeleted": True})
            async def late_commit():
                async with scope.scoped_transaction(flow.bound, database=flow.db) as tx:
                    await insert_reply(tx, flow)
            blocker_pid = (await changing.query_raw("SELECT pg_backend_pid() AS pid"))[0]["pid"]
            pending = asyncio.create_task(late_commit())
            # Query pg_stat_activity rather than sleeping for a guessed duration.
            for _ in range(100):
                waiters = await changing.query_raw("SELECT count(*)::int AS count FROM pg_stat_activity WHERE $1::int = ANY(pg_blocking_pids(pid))", blocker_pid)
                if waiters[0]["count"]:
                    break
                await asyncio.sleep(0.005)
            else:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                pytest.fail("scope commit did not reach the expected lock wait")
        with pytest.raises(scope.ExecutionScopeUnavailable):
            await pending
        assert await flow.db.message.count(where={"conversationId": flow.ids["conversation"]}) == 0
    finally:
        await other.disconnect()


async def test_lock_timeout_fails_closed_without_a_reply(flow):
    other = Prisma(datasource={"url": flow.url}, http={"trust_env": False})
    await other.connect()
    try:
        async with other.tx() as holding:
            await holding.query_raw("SELECT id FROM conversations WHERE id=$1 FOR UPDATE", flow.ids["conversation"])
            with pytest.raises(Exception, match="lock timeout"):
                async with scope.scoped_transaction(flow.bound, database=flow.db) as tx:
                    await insert_reply(tx, flow)
        assert await flow.db.message.count(where={"conversationId": flow.ids["conversation"]}) == 0
    finally:
        await other.disconnect()
