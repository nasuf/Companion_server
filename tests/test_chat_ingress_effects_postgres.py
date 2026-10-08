"""Real domain writes on the migrated synthetic loopback DB only."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prisma import Prisma

from app.services import offerings, wallet
from app.services.runtime.chat_ingress import ChatRequestConflict, accept_chat_message, load_chat_turn
from app.services.runtime.chat_ingress_contracts import ChatAggregationPolicy, ChatRequestInput
from app.services.runtime.chat_ingress_effects import (
    ChatIngressEffects, ChatIngressQuotaBlocked, ChatIngressResourceInvalid,
)
from app.services.runtime.execution_scope import ExecutionScopeExpired, bind_conversation_scope
from app.services.vip import chat_quota, config
from tests.test_chat_ingress_postgres import FaultDatabase, counts, prepared, snapshot
from tests.test_runtime_execution_foundation import RuntimeDatabase, flow


async def submit(flow, *, key="a", text="synthetic", ids=None, card=None, metadata=None,
                 paid=False, database=None, scope=None, policy=None):
    return await accept_chat_message(scope or flow.bound,
        ChatRequestInput.from_client(client_id=key, text=text, attachment_ids=ids, component_card=card),
        prepared(text, metadata=metadata), snapshot(), policy or ChatAggregationPolicy("immediate"),
        database=database or flow.db, effects=ChatIngressEffects(paid_confirmed=paid))


async def seed_quota(flow, *, used=20, vip=False, gift=0, permanent=10):
    scope, key, _ = config.message_period(vip)
    await flow.db.execute_raw("INSERT INTO user_message_quota (user_id,period_scope,period_key,used) "
        "VALUES ($1,$2,$3,$4)", flow.ids["owner"], scope, key, used)
    await flow.db.execute_raw("INSERT INTO user_wallets (user_id,ticket_balance,gift_ticket_balance,vip_until) "
        "VALUES ($1,$2,$3,$4::timestamptz)", flow.ids["owner"], permanent, gift,
        (datetime.now(timezone.utc)+timedelta(days=1)).isoformat() if vip else None)


async def balances(flow):
    rows = await flow.db.query_raw("SELECT ticket_balance,gift_ticket_balance FROM user_wallets WHERE user_id=$1", flow.ids["owner"])
    quota = await flow.db.query_raw("SELECT used FROM user_message_quota WHERE user_id=$1", flow.ids["owner"])
    ledger = await flow.db.query_raw("SELECT currency,delta,source_id FROM wallet_ledger WHERE user_id=$1 ORDER BY currency", flow.ids["owner"])
    return rows, quota, ledger


async def attachment(flow, *, conversation=None):
    ident = str(uuid4())
    await flow.db.execute_raw("INSERT INTO chat_message_attachments (id,user_id,conversation_id,mime,size,storage_key,url) "
        "VALUES ($1,$2,$3,'image/png',1,$1,'https://synthetic.invalid/image')", ident, flow.ids["owner"],
        conversation or flow.ids["conversation"])
    return ident


async def offering(flow, *, kind="gift", conversation=None, status="sent", agent=None):
    ident = str(uuid4())
    rows = await flow.db.query_raw("INSERT INTO user_offerings (id,user_id,agent_id,conversation_id,kind,status,ticket_amount,agent_value_yuan) "
        "VALUES ($1,$2,$3,$4,$5,$6,10,10) RETURNING *", ident, flow.ids["owner"], agent or flow.ids["agent"],
        conversation or flow.ids["conversation"], kind, status)
    return offerings.build_offering_card(offerings._offering_from_row(rows[0]))


async def test_free_retry_keeps_one_message_and_one_quota_use(flow):
    first = await submit(flow)
    retry = await submit(flow, paid=True)
    assert retry.message_id == first.message_id and not retry.created
    _, quota, ledger = await balances(flow)
    assert quota == [{"used": 1}] and not ledger
    message = await flow.db.message.find_unique(where={"id": first.message_id})
    assert message.metadata["ingress_effects"]["quota"]["mode"] == "free"


async def test_paid_parallel_retries_charge_once_and_recover_lost_commit_ack(flow):
    await seed_quota(flow, gift=2)
    other = RuntimeDatabase(Prisma(datasource={"url": flow.url}, http={"trust_env": False}))
    await other.connect()
    try:
        results = await asyncio.gather(*(submit(flow, paid=True, database=flow.db if i%2 else other) for i in range(8)))
        assert sum(r.created for r in results) == 1
        first = next(r for r in results if r.created)
        rows, quota, ledger = await balances(flow)
        assert rows == [{"ticket_balance": 7, "gift_ticket_balance": 0}]
        assert quota == [{"used": 21}]
        assert ledger == [{"currency": "gift_ticket", "delta": -2, "source_id": first.message_id},
                          {"currency": "ticket", "delta": -3, "source_id": first.message_id}]
        with pytest.raises(ConnectionError):
            await submit(flow, key="b", paid=True, database=FaultDatabase(flow.db, commit_unknown=True))
        retry = await submit(flow, key="b", paid=True)
        assert not retry.created
        assert (await balances(flow))[1] == [{"used": 22}]
    finally:
        await other.disconnect()


async def test_changed_input_conflict_has_no_second_charge(flow):
    await seed_quota(flow)
    await submit(flow, paid=True)
    before = await balances(flow)
    with pytest.raises(ChatRequestConflict):
        await submit(flow, text="changed", paid=True)
    assert await balances(flow) == before


@pytest.mark.parametrize("vip,cost", [(False, 5), (True, 3)])
async def test_server_vip_and_period_select_real_price(flow, vip, cost):
    await seed_quota(flow, used=5200 if vip else 20, vip=vip)
    await submit(flow, paid=True)
    rows, quota, ledger = await balances(flow)
    assert rows[0]["ticket_balance"] == 10-cost
    assert quota[0]["used"] == (5201 if vip else 21)
    assert sum(row["delta"] for row in ledger) == -cost


@pytest.mark.parametrize("paid,permanent,reason", [(False,10,"paid_confirm"), (True,4,"no_ticket"), (False,0,"no_ticket")])
async def test_blocked_quota_rolls_back_media_and_all_ingress_rows(flow, paid, permanent, reason):
    await seed_quota(flow, permanent=permanent)
    ident = await attachment(flow)
    before = await balances(flow)
    with pytest.raises(ChatIngressQuotaBlocked) as failure:
        await submit(flow, ids=[ident], metadata={"attachments":[{"id":ident}]}, paid=paid)
    assert failure.value.result["reason"] == reason
    assert await balances(flow) == before
    assert await counts(flow) == {"messages":0,"runs":0,"jobs":0,"receipts":0}
    assert (await flow.db.query_raw("SELECT message_id FROM chat_message_attachments WHERE id=$1", ident))[0]["message_id"] is None
    if not paid and permanent == 10:
        assert (await submit(flow, ids=[ident], metadata={"attachments":[{"id":ident}]}, paid=True)).created


@pytest.mark.parametrize("marker", ["UPDATE chat_message_attachments", "UPDATE user_message_quota",
    "UPDATE user_wallets", "INSERT INTO wallet_ledger", "UPDATE messages SET metadata", "INSERT INTO chat_ingress_receipts", "UPDATE runtime_jobs"])
async def test_after_each_effect_write_failure_rolls_back_payment_and_media(flow, marker):
    await seed_quota(flow, gift=2)
    ident = await attachment(flow)
    before = await balances(flow)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        await submit(flow, paid=True, ids=[ident], metadata={"attachments":[{"id":ident}]}, database=FaultDatabase(flow.db, marker))
    assert await balances(flow) == before
    assert await counts(flow) == {"messages":0,"runs":0,"jobs":0,"receipts":0}
    assert (await flow.db.query_raw("SELECT message_id FROM chat_message_attachments WHERE id=$1", ident))[0]["message_id"] is None
    assert (await submit(flow, paid=True, ids=[ident], metadata={"attachments":[{"id":ident}]})).created


async def test_multiple_attachment_partial_validity_rolls_back_all(flow):
    good = await attachment(flow)
    wrong = await attachment(flow, conversation=flow.ids["other_conversation"])
    with pytest.raises(ChatIngressResourceInvalid):
        await submit(flow, ids=[good,wrong], metadata={"attachments":[{"id":good},{"id":wrong}]})
    assert (await balances(flow))[1] == []
    assert await counts(flow) == {"messages":0,"runs":0,"jobs":0,"receipts":0}
    assert (await flow.db.query_raw("SELECT message_id FROM chat_message_attachments WHERE id=$1", good))[0]["message_id"] is None


async def test_two_requests_cannot_bind_one_attachment_or_double_charge(flow):
    ident = await attachment(flow)
    results = await asyncio.gather(*(submit(flow, key=key, ids=[ident], metadata={"attachments":[{"id":ident}]}) for key in ["a","b"]), return_exceptions=True)
    assert sum(isinstance(r,ChatIngressResourceInvalid) for r in results) == 1
    assert (await balances(flow))[1] == [{"used":1}]
    assert await counts(flow) == {"messages":1,"runs":1,"jobs":1,"receipts":1}


@pytest.mark.parametrize("kind", ["gift","red_packet"])
async def test_offering_binding_retry_exempt_without_hooks_or_another_payment(flow, monkeypatch, kind):
    card = await offering(flow, kind=kind)
    hook = AsyncMock(side_effect=AssertionError("legacy background hook must not run"))
    monkeypatch.setattr(offerings, "bind_offering_message", hook)
    first = await submit(flow, card=card, metadata={"component_card":card})
    retry = await submit(flow, card=card, metadata={"component_card":card})
    assert first.message_id == retry.message_id and not retry.created
    rows = await flow.db.query_raw("SELECT message_id FROM user_offerings WHERE id=$1", card["payload"]["offering_id"])
    assert rows == [{"message_id":first.message_id}]
    assert await balances(flow) == ([],[],[])
    hook.assert_not_awaited()


@pytest.mark.parametrize("invalid", ["conversation","received","tampered","missing","kind"])
async def test_offering_revalidated_inside_transaction(flow, invalid):
    card = await offering(flow, conversation=flow.ids["other_conversation"] if invalid=="conversation" else None,
        status="received" if invalid=="received" else "sent")
    if invalid=="tampered": card["payload"]["ticket_amount"]=99999
    if invalid=="missing": card["payload"]["offering_id"]="missing"
    if invalid=="kind": card["type"]="red_packet"
    with pytest.raises(ChatIngressResourceInvalid):
        await submit(flow, card=card, metadata={"component_card":card})
    assert await counts(flow) == {"messages":0,"runs":0,"jobs":0,"receipts":0}


@pytest.mark.parametrize("marker", ["UPDATE user_offerings", "INSERT INTO chat_ingress_receipts"])
async def test_offering_failure_does_not_lose_the_unbound_gift(flow, marker):
    card = await offering(flow)
    with pytest.raises(RuntimeError):
        await submit(flow, card=card, metadata={"component_card":card}, database=FaultDatabase(flow.db, marker))
    assert (await flow.db.query_raw("SELECT message_id FROM user_offerings WHERE id=$1", card["payload"]["offering_id"]))[0]["message_id"] is None
    assert (await submit(flow, card=card, metadata={"component_card":card})).created


async def test_stale_scope_stops_all_business_effects(flow):
    await seed_quota(flow)
    with pytest.raises(ExecutionScopeExpired):
        await submit(flow, paid=True, scope=replace(flow.bound, owner_generation=str(uuid4())))
    assert (await balances(flow))[1] == [{"used":20}]


async def test_aggregation_counts_original_messages_not_merged_turn(flow):
    policy = ChatAggregationPolicy("turn_window",30,60)
    first = await submit(flow, policy=policy)
    second = await submit(flow, key="b", policy=policy)
    assert first.run_id == second.run_id
    assert (await balances(flow))[1] == [{"used":2}]


@pytest.mark.parametrize("card_type", ["location","music_track","external_link"])
async def test_unimplemented_card_effects_cannot_enter_worker(flow, card_type):
    card={"type":card_type,"payload":{}}
    with pytest.raises(ChatIngressResourceInvalid):
        await submit(flow, card=card, metadata={"component_card":card})
    assert await counts(flow) == {"messages":0,"runs":0,"jobs":0,"receipts":0}


async def test_transaction_path_never_uses_global_database(flow, monkeypatch):
    class Forbidden:
        def __getattr__(self, name): raise AssertionError("global DB forbidden")
    await seed_quota(flow)
    monkeypatch.setattr(wallet,"db",Forbidden())
    monkeypatch.setattr(chat_quota,"db",Forbidden())
    assert (await submit(flow,paid=True)).created


async def test_distinct_requests_competing_for_last_free_slot_cannot_both_be_free(flow):
    await seed_quota(flow, used=19)
    results = await asyncio.gather(submit(flow,key="a"), submit(flow,key="b"), return_exceptions=True)
    assert sum(isinstance(r,ChatIngressQuotaBlocked) for r in results) == 1
    assert (await balances(flow))[1] == [{"used":20}]
    assert (await counts(flow))["messages"] == 1


async def test_distinct_offering_requests_cannot_claim_twice(flow):
    card=await offering(flow)
    results=await asyncio.gather(*(submit(flow,key=key,card=card,metadata={"component_card":card}) for key in ["a","b"]),return_exceptions=True)
    assert sum(isinstance(r,ChatIngressResourceInvalid) for r in results) == 1
    assert (await counts(flow))["messages"] == 1
    assert await balances(flow) == ([],[],[])


@pytest.mark.parametrize("legacy", [False,True])
async def test_error_after_actual_wallet_debit_rolls_back_ledger_balance_and_quota(flow,monkeypatch,legacy):
    await seed_quota(flow)
    before=await balances(flow)
    actual=wallet.debit_tickets_prioritized
    async def failing_debit(*args,**kwargs):
        await actual(*args,**kwargs)
        raise ValueError("synthetic ledger error after debit")
    monkeypatch.setattr(wallet,"debit_tickets_prioritized",failing_debit)
    with pytest.raises(ValueError,match="synthetic ledger error"):
        if legacy:
            monkeypatch.setattr(chat_quota,"db",flow.db)
            await chat_quota.consume_one(flow.ids["owner"],is_vip=False,paid_confirmed=True)
        else:
            await submit(flow,paid=True)
    assert await balances(flow) == before
    assert (await counts(flow))["messages"] == 0


async def test_attachment_metadata_order_must_match_original_input(flow):
    a,b=await attachment(flow),await attachment(flow)
    with pytest.raises(ChatIngressResourceInvalid):
        await submit(flow,ids=[a,b],metadata={"attachments":[{"id":b},{"id":a}]})
    assert (await balances(flow))[1] == []


async def test_empty_card_cannot_bypass_real_card_adapter(flow):
    with pytest.raises(ChatIngressResourceInvalid):
        await submit(flow,card={"payload":{}},metadata={"component_card":{"payload":{}}})


async def test_two_conversations_share_one_user_quota_lock(flow):
    await seed_quota(flow,used=19)
    agent_id,workspace_id=str(uuid4()),str(uuid4())
    try:
        await flow.db.aiagent.create(data={"id":agent_id,"userId":flow.ids["owner"],"name":"Other synthetic"})
        await flow.db.chatworkspace.create(data={"id":workspace_id,"userId":flow.ids["owner"],"agentId":agent_id,"allowMultipleActive":True})
        await flow.db.conversation.update(where={"id":flow.ids["other_conversation"]},data={"agentId":agent_id,"workspaceId":workspace_id})
        other_scope=await bind_conversation_scope(actor_user_id=flow.ids["owner"],conversation_id=flow.ids["other_conversation"],database=flow.db)
        results=await asyncio.gather(submit(flow),submit(flow,scope=other_scope),return_exceptions=True)
        assert sum(isinstance(r,ChatIngressQuotaBlocked) for r in results)==1
        assert (await balances(flow))[1]==[{"used":20}]
        messages=await flow.db.message.count(where={"conversationId":{"in":[flow.ids["conversation"],flow.ids["other_conversation"]]}})
        assert messages==1
    finally:
        await flow.db.agentrun.delete_many(where={"conversationId":flow.ids["other_conversation"]})
        await flow.db.message.delete_many(where={"conversationId":flow.ids["other_conversation"]})
        await flow.db.conversation.update(where={"id":flow.ids["other_conversation"]},data={"workspaceId":None,"agentId":flow.ids["agent"]})
        await flow.db.chatworkspace.delete_many(where={"id":workspace_id})
        await flow.db.aiagent.delete_many(where={"id":agent_id})


async def test_failed_paid_append_preserves_previous_window_and_balance(flow):
    await seed_quota(flow,used=19)
    policy=ChatAggregationPolicy("turn_window",30,60)
    first=await submit(flow,policy=policy)
    turn=await load_chat_turn(flow.bound,first.run_id,database=flow.db)
    balance=await balances(flow)
    with pytest.raises(RuntimeError,match="synthetic write failure"):
        await submit(flow,key="b",paid=True,policy=policy,database=FaultDatabase(flow.db,"UPDATE runtime_jobs"))
    assert await load_chat_turn(flow.bound,first.run_id,database=flow.db)==turn
    assert await balances(flow)==balance
    assert await counts(flow)=={"messages":1,"runs":1,"jobs":1,"receipts":1}
