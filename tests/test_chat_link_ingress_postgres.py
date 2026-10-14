"""Real link preparation/admission on migrated loopback PG; providers are synthetic."""

import asyncio
from dataclasses import replace
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prisma.errors import RawQueryError

from app.services.chat_links import repo, cards
from app.services.chat_links.covers import CachedCoverResult
from app.services.chat_links.extraction import LinkMetadata
from app.services.runtime import chat_entrypoint as entry, chat_link_preparation as preparation
from app.services.runtime.chat_ingress_effects import ChatIngressQuotaBlocked, ChatIngressResourceInvalid
from app.services.runtime.execution_scope import ExecutionScopeExpired
from tests.test_chat_ingress_effects_postgres import balances, seed_quota, submit, attachment
from tests.test_chat_ingress_postgres import counts, snapshot, FaultDatabase
from tests.test_runtime_execution_foundation import flow
from tests.test_chat_entrypoint_postgres import LostCommitDatabase


class PreparationFaultDatabase(FaultDatabase):
    def __getattr__(self, name):
        return getattr(self.database, name)


@pytest.fixture
async def links(flow, monkeypatch):
    from app import db as database
    from app.services.runtime import execution_scope
    from app.services import runtime_config
    from app.services.schedule_domain import schedule
    from app.services.interaction import reply_context, user_turn_aggregation
    from app.services.runtime.chat_ingress_contracts import ChatAggregationPolicy

    monkeypatch.setattr(repo, "db", flow.db)
    monkeypatch.setattr(database, "db", flow.db)
    monkeypatch.setattr(execution_scope, "db", flow.db)
    monkeypatch.setattr(repo, "cache_link_cover", AsyncMock(
        side_effect=lambda **kw: CachedCoverResult(kw["metadata"], {})))
    fetch = AsyncMock(return_value=LinkMetadata(source_url="https://example.com/story",
        final_url="https://example.com/story", platform="链接", title="stored title", content_text="stored body"))
    monkeypatch.setattr(preparation, "extract_link_metadata", fetch)
    monkeypatch.setattr(runtime_config, "bind_agent_context", AsyncMock(return_value=None))
    monkeypatch.setattr(runtime_config, "reset_current_agent", lambda _: None)
    monkeypatch.setattr(entry, "_bound_execution_snapshot", lambda _: snapshot())
    monkeypatch.setattr(schedule, "get_cached_schedule", AsyncMock(return_value={"synthetic": True}))
    monkeypatch.setattr(schedule, "get_current_status", lambda _: {"status": "idle"})
    monkeypatch.setattr(reply_context, "build_reply_timing_context", AsyncMock(return_value={"delay_seconds": 0}))
    monkeypatch.setattr(user_turn_aggregation, "plan_sql_user_message_aggregation",
        AsyncMock(return_value=ChatAggregationPolicy("immediate")))

    async def preview(title="stored title", **kwargs):
        return await repo.create_or_update_link_card(user_id=flow.ids["owner"],
            conversation_id=kwargs.pop("conversation_id", flow.ids["conversation"]),
            metadata=replace(fetch.return_value, title=title, content_text=title), **kwargs)

    async def receive(*, key="one", text="https://example.com/story", card=None, paid=False, **kwargs):
        return await entry.receive_client_chat(actor_user_id=flow.ids["owner"],
            conversation_id=flow.ids["conversation"], database=kwargs.pop("database", flow.db),
            payload={"client_id": key, "message": text, "component_card": card, "paid_confirmed": paid}, **kwargs)
    return flow, fetch, preview, receive


async def binding_rows(flow):
    return await flow.db.query_raw("SELECT b.message_id,b.snapshot_id FROM chat_link_message_bindings b "
        "JOIN messages m ON m.id=b.message_id WHERE m.conversation_id=$1 ORDER BY b.message_id", flow.ids["conversation"])


async def test_real_preparation_text_receipt_and_all_effects_commit_together(links):
    flow, fetch, _, receive = links
    first = await receive()
    msg = await flow.db.message.find_unique(where={"id": first.message_id})
    version = msg.metadata["link_card"]["snapshot_id"]
    assert msg.metadata["link_card"]["content_text"] == "stored body"
    assert msg.metadata["ingress_effects"]["link"] == {"snapshot_id": version, "binding": "atomic"}
    assert await binding_rows(flow) == [{"message_id": first.message_id, "snapshot_id": version}]
    assert await counts(flow) == {"messages": 1, "runs": 1, "jobs": 1, "receipts": 1}
    assert (await balances(flow))[1] == [{"used": 1}]
    receipt = (await flow.db.query_raw("SELECT prepared_input FROM chat_ingress_receipts WHERE message_id=$1", first.message_id))[0]
    assert "stored body" in receipt["prepared_input"]["prompt_text"]
    assert (await receive()).message_id == first.message_id
    fetch.assert_awaited_once()


async def test_pinned_card_ignores_client_forgery_and_refresh_and_allows_repeat_shares(links):
    flow, fetch, preview, receive = links
    first = await preview("first")
    card = cards.component_card_for_link(first)
    card["body"] = "forged body"
    card["payload"]["content_text"] = "forged content"
    await preview("later")
    results = [await receive(key=str(i), card=card, text="look") for i in range(2)]
    assert len(await binding_rows(flow)) == 2
    for result in results:
        msg = await flow.db.message.find_unique(where={"id": result.message_id})
        assert msg.metadata["link_card"]["content_text"] == "first"
        assert msg.metadata["component_card"]["body"] == "first"
    fetch.assert_not_awaited()
    assert (await balances(flow))[1] == [{"used": 2}]


async def test_legacy_cache_without_snapshot_is_captured_at_preparation(links):
    flow, fetch, preview, receive = links
    link = await preview()
    await flow.db.execute_raw("DELETE FROM chat_link_snapshots WHERE id=$1", link.snapshot_id)
    card = cards.component_card_for_link(link)
    del card["payload"]["snapshot_id"]
    result = await receive(card=card)
    bound = (await binding_rows(flow))[0]
    assert bound["message_id"] == result.message_id and bound["snapshot_id"] != link.snapshot_id
    fetch.assert_not_awaited()


async def test_paid_parallel_retries_have_one_binding_and_charge(links):
    flow, fetch, preview, receive = links
    link = await preview()
    card = cards.component_card_for_link(link)
    await seed_quota(flow)
    results = await asyncio.gather(*(receive(card=card, paid=True) for _ in range(6)), return_exceptions=True)
    assert not [r for r in results if isinstance(r, BaseException)]
    assert sum(r.created for r in results) == 1 and len(await binding_rows(flow)) == 1
    assert (await balances(flow))[0] == [{"ticket_balance": 5, "gift_ticket_balance": 0}]
    fetch.assert_not_awaited()


async def test_lost_commit_ack_recovers_link_receipt_without_second_charge(links):
    flow, fetch, preview, receive = links
    card = cards.component_card_for_link(await preview())
    await seed_quota(flow)
    lost = LostCommitDatabase(flow.db)
    async def prepare_and_lose_ack(*args, **kwargs):
        prepared = await entry.prepare_sql_chat_message(*args, **kwargs)
        lost.fail = True
        return prepared
    with pytest.raises(ConnectionError):
        await receive(key="lost", card=card, paid=True, database=lost, prepare=prepare_and_lose_ack)
    recovered = await receive(key="lost", card=card)
    assert not recovered.created and len(await binding_rows(flow)) == 1
    assert (await balances(flow))[1] == [{"used": 21}]
    assert (await balances(flow))[0] == [{"ticket_balance": 5, "gift_ticket_balance": 0}]
    fetch.assert_not_awaited()


@pytest.mark.parametrize("marker", ["INSERT INTO chat_link_message_bindings", "UPDATE user_message_quota",
    "UPDATE user_wallets", "INSERT INTO wallet_ledger", "UPDATE messages SET metadata", "INSERT INTO chat_ingress_receipts"])
async def test_failure_after_each_write_rolls_back_message_binding_and_quota(links, marker):
    flow, _, preview, receive = links
    link = await preview()
    card = cards.component_card_for_link(link)
    await seed_quota(flow)
    before = await balances(flow)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        await receive(card=card, paid=True, database=PreparationFaultDatabase(flow.db, marker))
    assert await counts(flow) == {"messages": 0, "runs": 0, "jobs": 0, "receipts": 0}
    assert await binding_rows(flow) == [] and await balances(flow) == before
    assert (await receive(card=card, paid=True)).created


async def test_rejected_quota_keeps_preview_reusable_without_binding(links):
    flow, _, preview, receive = links
    link = await preview()
    card = cards.component_card_for_link(link)
    await seed_quota(flow)
    with pytest.raises(ChatIngressQuotaBlocked):
        await receive(card=card)
    assert not await binding_rows(flow)
    assert (await receive(card=card, paid=True)).created


@pytest.mark.parametrize("kind", ["foreign_conversation", "assistant", "missing_version", "missing_cache", "bad_payload"])
async def test_invalid_or_foreign_card_cannot_fetch_charge_or_write(links, kind):
    flow, fetch, preview, receive = links
    link = await preview(conversation_id=flow.ids["other_conversation"] if kind == "foreign_conversation" else flow.ids["conversation"],
                         role="assistant" if kind == "assistant" else "user")
    card = cards.component_card_for_link(link)
    if kind == "missing_version": card["payload"]["snapshot_id"] = str(uuid4())
    if kind == "missing_cache": del card["payload"]["link_id"]
    if kind == "bad_payload": card["payload"] = []
    with pytest.raises(ChatIngressResourceInvalid): await receive(card=card)
    assert not await binding_rows(flow) and (await counts(flow))["messages"] == 0
    assert (await balances(flow))[1] == []
    fetch.assert_not_awaited()


async def test_generation_change_during_fetch_cannot_store_preview_or_accept(links):
    flow, fetch, _, receive = links
    async def changed(**kw):
        await flow.db.execute_raw("UPDATE conversations SET execution_generation=$2 WHERE id=$1", flow.ids["conversation"], str(uuid4()))
        return fetch.return_value
    fetch.side_effect = changed
    with pytest.raises(ExecutionScopeExpired): await receive()
    assert await flow.db.query_raw("SELECT id FROM chat_link_cards WHERE conversation_id=$1", flow.ids["conversation"]) == []
    assert (await counts(flow))["messages"] == 0 and (await balances(flow))[1] == []


@pytest.mark.parametrize("tamper", ["projection", "component", "snapshot", "generation"])
async def test_admission_revalidates_prepared_snapshot_and_generation(links, tamper):
    flow, _, preview, _ = links
    link = await preview()
    metadata = {"link_card": cards.metadata_for_link_card(link), "component_card": cards.component_card_for_link(link)}
    card = cards.component_card_for_link(link)
    if tamper == "projection": metadata["link_card"]["content_text"] = "tampered"
    if tamper == "component": metadata["component_card"]["body"] = "tampered"
    if tamper == "snapshot": metadata["link_card"]["snapshot_id"] = str(uuid4())
    scope = replace(flow.bound, conversation_generation=str(uuid4())) if tamper == "generation" else flow.bound
    with pytest.raises((ChatIngressResourceInvalid, ExecutionScopeExpired)):
        await submit(flow, card=card, metadata=metadata, scope=scope)
    assert not await binding_rows(flow) and (await balances(flow))[1] == []


async def test_binding_and_snapshot_are_immutable_and_delete_with_conversation(links):
    flow, _, preview, receive = links
    link = await preview()
    result = await receive(card=cards.component_card_for_link(link))
    with pytest.raises(RawQueryError, match="immutable"):
        await flow.db.execute_raw("UPDATE chat_link_message_bindings SET snapshot_id=$2 WHERE message_id=$1", result.message_id, link.snapshot_id)
    with pytest.raises(RawQueryError):
        await flow.db.execute_raw("DELETE FROM chat_link_snapshots WHERE id=$1", link.snapshot_id)
    # Existing lifecycle deletion removes messages before their restricted parent.
    async with flow.db.tx() as tx:
        await tx.execute_raw("DELETE FROM messages WHERE conversation_id=$1", flow.ids["conversation"])
        await tx.execute_raw("DELETE FROM conversations WHERE id=$1", flow.ids["conversation"])
    assert not await binding_rows(flow)


async def test_bound_snapshot_with_attachments_rejects_without_partial_media_charge(links):
    flow, _, preview, _ = links
    link = await preview()
    ident = await attachment(flow)
    meta = {"link_card": cards.metadata_for_link_card(link), "component_card": cards.component_card_for_link(link), "attachments": [{"id": ident}]}
    await seed_quota(flow)
    with pytest.raises(ChatIngressQuotaBlocked):
        await submit(flow, card=cards.component_card_for_link(link), metadata=meta, ids=[ident])
    assert not await binding_rows(flow)
    assert (await flow.db.query_raw("SELECT message_id FROM chat_message_attachments WHERE id=$1", ident))[0]["message_id"] is None


@pytest.mark.parametrize("payload", [
    {"link_id": ""}, {"snapshot_id": 3}, {"link_id": " padded "},
    {"source_url": 17}, {"source_url": "x" * 2001}, {},
])
async def test_invalid_link_identity_or_url_never_fetches(links, payload):
    flow, fetch, _, receive = links
    with pytest.raises(ChatIngressResourceInvalid):
        await receive(text="share", card={"type": "external_link", "payload": payload})
    fetch.assert_not_awaited()
    assert not await binding_rows(flow) and (await counts(flow))["messages"] == 0


async def test_url_card_uses_server_content_and_fetch_rejection_has_no_side_effects(links):
    flow, fetch, _, receive = links
    card = {"type": "external_link", "body": "forged", "payload": {"source_url": "https://example.com/story"}}
    fetch.side_effect = ValueError("synthetic unsafe URL")
    with pytest.raises(ChatIngressResourceInvalid):
        await receive(text="share", card=card)
    assert (await counts(flow))["messages"] == 0
    fetch.side_effect = None
    result = await receive(text="share", card=card)
    message = await flow.db.message.find_unique(where={"id": result.message_id})
    assert message.metadata["component_card"]["body"] == "stored body"
    assert len(await binding_rows(flow)) == 1


async def test_generation_change_during_cover_removes_only_uncommitted_cover(links, monkeypatch):
    from unittest.mock import Mock
    flow, _, _, receive = links
    delete = Mock()
    monkeypatch.setattr(repo.chat_media_storage, "delete_media_file", delete)

    async def cover(**kw):
        await flow.db.execute_raw("UPDATE conversations SET execution_generation=$2 WHERE id=$1",
                                  flow.ids["conversation"], str(uuid4()))
        return CachedCoverResult(kw["metadata"], {"cover_storage_key": "synthetic-owned-cover"})

    monkeypatch.setattr(repo, "cache_link_cover", cover)
    with pytest.raises(ExecutionScopeExpired):
        await receive()
    delete.assert_called_once_with("synthetic-owned-cover")
    assert not await binding_rows(flow) and (await counts(flow))["messages"] == 0
    assert await flow.db.query_raw("SELECT id FROM chat_link_cards WHERE conversation_id=$1", flow.ids["conversation"]) == []


async def test_scoped_preview_unknown_commit_keeps_committed_snapshot_and_cover(links, monkeypatch):
    from unittest.mock import Mock
    flow, fetch, _, _ = links
    delete = Mock()
    monkeypatch.setattr(repo.chat_media_storage, "delete_media_file", delete)
    with pytest.raises(ConnectionError):
        await repo.create_or_update_link_card(user_id=flow.ids["owner"], conversation_id=flow.ids["conversation"],
            metadata=fetch.return_value, execution_scope=flow.bound, database=PreparationFaultDatabase(flow.db, commit_unknown=True))
    delete.assert_not_called()
    snapshots = await flow.db.query_raw("SELECT s.id FROM chat_link_snapshots s JOIN chat_link_cards c "
        "ON c.id=s.link_card_id WHERE c.conversation_id=$1", flow.ids["conversation"])
    assert len(snapshots) == 1 and not await binding_rows(flow)
    assert (await counts(flow))["messages"] == 0


async def test_preview_scope_arguments_cannot_cross_owner_or_role(links, monkeypatch):
    from app.services.runtime.execution_scope import ExecutionScopeUnavailable
    flow, fetch, _, _ = links
    cover = repo.cache_link_cover
    for overrides in ({"user_id": "foreign"}, {"conversation_id": flow.ids["other_conversation"]}, {"role": "assistant"}):
        args = dict(user_id=flow.ids["owner"], conversation_id=flow.ids["conversation"], role="user")
        with pytest.raises(ExecutionScopeUnavailable):
            await repo.create_or_update_link_card(**(args | overrides), metadata=fetch.return_value, execution_scope=flow.bound)
    cover.assert_not_awaited()


async def test_unexpected_link_metadata_cannot_bypass_ordinary_card_validation(links):
    flow, _, preview, _ = links
    link = await preview()
    with pytest.raises(ChatIngressResourceInvalid):
        await submit(flow, text="ordinary", metadata={"link_card": cards.metadata_for_link_card(link)})
    assert not await binding_rows(flow) and (await counts(flow))["messages"] == 0


async def test_http_and_socket_share_real_link_receipt_and_error_contracts(links, monkeypatch):
    from types import SimpleNamespace
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api import deps
    from app.api.public import chat
    from app.services.admin_takeover import service as takeover
    from app.services.auth import create_jwt

    flow, fetch, preview, _ = links
    monkeypatch.setattr(chat, "db", flow.db)
    monkeypatch.setattr(deps, "is_redis_healthy", lambda: True)
    monkeypatch.setattr(chat, "sql_chat_ingress_enabled", lambda: True)
    monkeypatch.setattr(takeover, "is_conversation_taken_over", AsyncMock(return_value=False))
    app = FastAPI()
    app.include_router(chat.router)
    card = cards.component_card_for_link(await preview())
    payload = {"client_id": "transport", "message": "share", "component_card": card}
    auth = {"Authorization": "Bearer " + create_jwt(flow.ids["owner"], "user")}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://isolated") as client:
        path = "/chat/" + flow.ids["conversation"]
        assert (await client.post(path, json=payload)).status_code == 401
        foreign = {"Authorization": "Bearer " + create_jwt("foreign", "user")}
        assert (await client.post(path, json=payload, headers=foreign)).status_code == 403
        response = await client.post(path, json=payload, headers=auth)
        assert response.status_code == 200 and response.text.count("event: ack") == 1
        assert '"acceptance_only": true' in response.text
        ws = SimpleNamespace(send_json=AsyncMock(), close=AsyncMock())
        await entry.handle_sql_chat_frame(ws, flow.ids["owner"], flow.ids["conversation"], payload)
        ack = ws.send_json.await_args_list[0].args[0]
        assert ack["type"] == "ack" and ack["data"]["duplicate"]
        assert ack["data"]["message_id"] in response.text
        assert len(await binding_rows(flow)) == 1 and (await balances(flow))[1] == [{"used": 1}]
        invalid = {"client_id": "invalid", "message": "share", "component_card": {
            "type": "external_link", "payload": {"link_id": "foreign"}}}
        result = await client.post(path, json=invalid, headers=auth)
        assert result.status_code == 422 and result.json()["detail"]["code"] == "invalid_resource"
        ws.send_json.reset_mock()
        await entry.handle_sql_chat_frame(ws, flow.ids["owner"], flow.ids["conversation"], invalid)
        assert ws.send_json.await_args.args[0]["data"]["code"] == "invalid_resource"
        assert len(await binding_rows(flow)) == 1
    fetch.assert_not_awaited()


async def test_parallel_raw_url_preparations_cannot_duplicate_message_or_payment(links):
    flow, fetch, _, receive = links
    await seed_quota(flow)
    results = await asyncio.gather(*(receive(paid=True) for _ in range(4)), return_exceptions=True)
    assert not [r for r in results if isinstance(r, BaseException)]
    assert len({r.message_id for r in results}) == 1 and sum(r.created for r in results) == 1
    assert len(await binding_rows(flow)) == 1
    assert (await balances(flow))[1] == [{"used": 21}]
    assert 1 <= fetch.await_count <= 4
