"""Version-pinned preparation and per-message history on disposable PostgreSQL."""

import asyncio
from dataclasses import replace
from unittest.mock import Mock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prisma import Json
from prisma.errors import RawQueryError
import pytest

from app.api.public import daily_share
from app.services.auth import create_jwt
from app.services.chat_links import repo, cards
from app.services.chat_links.covers import CachedCoverResult
from app.services.chat_links.extraction import LinkMetadata
from tests.test_memory_evidence_postgres import origins


@pytest.fixture
async def links(origins, monkeypatch):
    db, uid, _, _, _, _, _, users, _, _, convs = origins
    monkeypatch.setattr(repo, "db", db)

    async def cover(**kwargs):
        return CachedCoverResult(kwargs["metadata"], {})

    monkeypatch.setattr(repo, "cache_link_cover", cover)

    async def prepare(title="first", **kwargs):
        return await repo.create_or_update_link_card(
            user_id=uid,
            conversation_id=convs[0],
            metadata=LinkMetadata(
                source_url="https://example.com/story?token=private",
                final_url="https://example.com/story",
                platform="链接",
                title=title,
                content_text=title,
                original_text=title,
                summary=title,
            ),
            **kwargs,
        )

    async def save(link, **kwargs):
        message = await db.message.create(
            data={
                "conversationId": convs[0],
                "role": link.role,
                "content": "shared",
                "metadata": Json(
                    {
                        "link_card": cards.metadata_for_link_card(link),
                        "component_card": cards.component_card_for_link(link),
                    }
                ),
                **kwargs,
            }
        )
        await repo.bind_link_card_to_message(
            link_id=link.id,
            message_id=message.id,
            user_id=uid,
            conversation_id=convs[0],
        )
        return message

    return db, uid, users, convs, prepare, save


async def test_same_url_versions_and_repeated_shares_preserve_cache_identity(links):
    db, uid, _, convs, prepare, save = links
    first = await prepare()
    m1 = await save(first)
    second = await prepare("changed page")
    assert first.id == second.id and first.snapshot_id != second.snapshot_id
    m2 = await save(second)
    m3 = await save(first)
    groups = await repo.list_user_link_groups(uid)
    history = {x.message_id: x for g in groups for x in g.links}
    assert set(history) == {m1.id, m2.id, m3.id}
    assert [history[m.id].title for m in (m1, m2, m3)] == [
        "first",
        "changed page",
        "first",
    ]
    assert all(
        x.id == first.id and x.occurrence_id == x.message_id for x in history.values()
    )
    assert history[m1.id].component_card["payload"]["snapshot_id"] == first.snapshot_id
    pinned = await repo.find_link_card(
        link_id=first.id,
        snapshot_id=first.snapshot_id,
        user_id=uid,
        conversation_id=convs[0],
    )
    assert pinned.title == "first"
    latest = await repo.find_link_card(
        link_id=first.id, user_id=uid, conversation_id=convs[0]
    )
    assert latest.title == "changed page"
    assert latest.snapshot_id == second.snapshot_id
    with pytest.raises(RawQueryError, match="immutable"):
        await db.execute_raw(
            "UPDATE chat_link_snapshots SET payload='{}'::jsonb WHERE id=$1",
            first.snapshot_id,
        )
    # Deleting the original cache pointer must not invalidate the other shares.
    await db.message.delete(where={"id": m1.id})
    assert (
        await repo.find_link_card(
            link_id=first.id,
            snapshot_id=first.snapshot_id,
            user_id=uid,
            conversation_id=convs[0],
        )
    ).title == "first"
    assert sum(g.count for g in await repo.list_user_link_groups(uid)) == 2
    assert sum(g.count for g in await repo.list_user_link_groups(uid, limit=1)) == 1


async def test_concurrent_previews_keep_each_returned_version_and_scope(links):
    db, uid, users, convs, prepare, save = links
    prepared = await asyncio.gather(*(prepare(str(i)) for i in range(6)))
    assert len({x.id for x in prepared}) == 1
    assert len({x.snapshot_id for x in prepared}) == 6
    for i, link in enumerate(prepared):
        exact = await repo.find_link_card(
            link_id=link.id,
            snapshot_id=link.snapshot_id,
            user_id=uid,
            conversation_id=convs[0],
        )
        assert exact.title == str(i)
    link = prepared[0]
    for owner, conv, version in [
        (users[-1], convs[0], link.snapshot_id),
        (uid, convs[1], link.snapshot_id),
        (uid, convs[0], "missing"),
    ]:
        assert (
            await repo.find_link_card(
                link_id=link.id,
                snapshot_id=version,
                user_id=owner,
                conversation_id=conv,
            )
            is None
        )
    await save(link)
    assert (
        await repo.find_link_card(
            link_id=link.id, user_id=uid, conversation_id=convs[0], require_unbound=True
        )
        is None
    )
    await db.conversation.update(where={"id": convs[0]}, data={"isDeleted": True})
    assert (
        await repo.find_link_card(
            link_id=link.id, user_id=uid, conversation_id=convs[0]
        )
        is None
    )
    assert await repo.list_user_link_groups(uid) == []


async def test_legacy_messages_and_missing_snapshot_remain_honest(links):
    db, uid, _, convs, prepare, save = links
    old = await prepare("historical")
    message = await save(replace(old, snapshot_id=None, snapshot_recorded_at=None))
    await prepare("current")
    history = (await repo.list_user_link_groups(uid))[0].links
    assert history[0].title == "historical" and history[0].message_id == message.id
    assert "snapshot_id" not in history[0].component_card["payload"]
    # An old cache-only row remains discoverable, but isn't given a fake version.
    await db.message.update(where={"id": message.id}, data={"metadata": Json({})})
    history = (await repo.list_user_link_groups(uid))[0].links
    assert (
        history[0].title == "current"
        and "snapshot_id" not in history[0].component_card["payload"]
    )
    await db.message.update(
        where={"id": message.id}, data={"metadata": Json({"link_card": {"id": 13}})}
    )
    assert await repo.list_user_link_groups(uid) == []


async def test_daily_api_is_owner_scoped_and_bounded(links):
    _, uid, users, _, prepare, save = links
    await save(await prepare())
    await save(await prepare("second"))
    app = FastAPI()
    app.include_router(daily_share.router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://isolated"
    ) as client:
        assert (await client.get("/daily-share/links")).status_code == 401
        auth = {"Authorization": "Bearer " + create_jwt(uid, "user")}
        result = await client.get("/daily-share/links?limit=1", headers=auth)
        assert result.status_code == 200 and result.json()["total"] == 1
        assert result.json()["groups"][0]["links"][0]["occurrence_id"]
        assert (
            await client.get("/daily-share/links?limit=1001", headers=auth)
        ).status_code == 422
        other = {"Authorization": "Bearer " + create_jwt(users[-1], "user")}
        assert (await client.get("/daily-share/links", headers=other)).json()[
            "total"
        ] == 0


async def test_snapshot_failure_rolls_back_cache_and_lost_ack_keeps_cover(
    links, monkeypatch
):
    db, uid, _, convs, prepare, _ = links
    first = await prepare()
    # The whole single SQL statement rolls back if its snapshot cannot commit.
    await db.execute_raw(
        "CREATE FUNCTION fail_link_snapshot_test() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'synthetic_snapshot_failure'; END; $$"
    )
    await db.execute_raw(
        "CREATE TRIGGER fail_link_snapshot_test BEFORE INSERT ON chat_link_snapshots FOR EACH ROW EXECUTE FUNCTION fail_link_snapshot_test()"
    )
    try:
        with pytest.raises(RawQueryError, match="synthetic_snapshot_failure"):
            await prepare("must rollback")
    finally:
        await db.execute_raw(
            "DROP TRIGGER fail_link_snapshot_test ON chat_link_snapshots"
        )
        await db.execute_raw("DROP FUNCTION fail_link_snapshot_test()")
    assert (
        await repo.find_link_card(
            link_id=first.id, user_id=uid, conversation_id=convs[0]
        )
    ).title == "first"
    delete = Mock()
    monkeypatch.setattr(repo.chat_media_storage, "delete_media_file", delete)

    class LostAck:
        async def query_raw(self, sql, *args):
            result = await db.query_raw(sql, *args)
            if "WITH cached AS" in sql:
                raise ConnectionError("synthetic lost commit acknowledgement")
            return result

    monkeypatch.setattr(repo, "db", LostAck())
    with pytest.raises(ConnectionError):
        await prepare("committed")
    delete.assert_not_called()
    monkeypatch.setattr(repo, "db", db)
    assert (
        await repo.find_link_card(
            link_id=first.id, user_id=uid, conversation_id=convs[0]
        )
    ).title == "committed"


async def test_invalid_scope_and_old_client_writer_do_not_invent_snapshots(links):
    db, uid, users, convs, prepare, save = links
    link = await prepare()
    foreign = await db.message.create(
        data={"conversationId": convs[-1], "role": "user", "content": "other"}
    )
    await repo.bind_link_card_to_message(
        link_id=link.id, message_id=foreign.id, user_id=uid, conversation_id=convs[0]
    )
    assert (
        await repo.find_link_card(
            link_id=link.id, user_id=uid, conversation_id=convs[0]
        )
    ).message_id is None
    with pytest.raises(ValueError, match="unavailable"):
        await prepare(message_id=foreign.id)
    # During migration overlap the old app can still update the cache. Do not
    # attach an earlier snapshot ID to that new cache content.
    await db.execute_raw(
        "UPDATE chat_link_cards SET title='legacy writer' WHERE id=$1", link.id
    )
    current = await repo.find_link_card(
        link_id=link.id, user_id=uid, conversation_id=convs[0]
    )
    assert current.snapshot_id is None and current.title == "legacy writer"
    pinned = await repo.find_link_card(
        link_id=link.id,
        user_id=uid,
        conversation_id=convs[0],
        snapshot_id=link.snapshot_id,
    )
    assert pinned.title == "first"
    await db.conversation.update(where={"id": convs[0]}, data={"isDeleted": True})
    with pytest.raises(ValueError, match="unavailable"):
        await prepare()
