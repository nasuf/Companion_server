"""Shared entrypoint on the migrated synthetic loopback database, including ASGI."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from app.services.auth import create_jwt
from app.services.runtime import chat_entrypoint as entry
from app.services.runtime.chat_ingress import ChatRequestConflict, load_chat_turn
from app.services.runtime.chat_ingress_contracts import ChatAggregationPolicy
from app.services.runtime.chat_ingress_effects import ChatIngressQuotaBlocked, ChatIngressResourceInvalid
from app.services.runtime.execution_scope import ExecutionScopeExpired, ExecutionScopeUnavailable
from tests.test_chat_ingress_effects_postgres import seed_quota, balances, attachment
from tests.test_chat_ingress_effects_postgres import offering
from tests.test_chat_ingress_postgres import counts, prepared, snapshot, FaultDatabase
from tests.test_runtime_execution_foundation import flow


def preparation(*, metadata=None, policy=None):
    async def prepare(scope,request,**kwargs):
        return entry.PreparedChatAcceptance(prepared(json.loads(request.input_json)["text"].strip(),
            metadata=metadata),snapshot(),policy or ChatAggregationPolicy("immediate"))
    return AsyncMock(side_effect=prepare)


async def receive(flow, *, key="id", text="synthetic", database=None, prepare=None, **payload):
    return await entry.receive_client_chat(actor_user_id=flow.ids["owner"],
        conversation_id=flow.ids["conversation"], payload={"client_id":key,"message":text,**payload},
        database=database or flow.db,prepare=prepare or preparation())


async def test_duplicate_skips_preparation_and_payment(flow):
    await seed_quota(flow)
    prep=preparation()
    first=await receive(flow,paid_confirmed=True,prepare=prep)
    before=await balances(flow)
    # Confirmation need not be repeated for an already committed source.
    second=await receive(flow,prepare=prep)
    assert first.message_id==second.message_id and not second.created
    prep.assert_awaited_once()
    assert await balances(flow)==before
    assert await counts(flow)=={"messages":1,"runs":1,"jobs":1,"receipts":1}


async def test_concurrent_same_id_has_one_paid_receipt(flow):
    await seed_quota(flow)
    results=await asyncio.gather(*(receive(flow,paid_confirmed=True) for _ in range(8)))
    assert len({r.message_id for r in results})==1 and sum(r.created for r in results)==1
    rows,quota,ledger=await balances(flow)
    assert rows[0]["ticket_balance"]==5 and quota[0]["used"]==21 and len(ledger)==1


async def test_original_whitespace_is_retry_identity_while_rendering_is_separate(flow):
    first=await receive(flow,text="  synthetic  ")
    assert (await flow.db.message.find_unique(where={"id":first.message_id})).content=="synthetic"
    with pytest.raises(ChatRequestConflict):await receive(flow,text="synthetic")
    assert (await counts(flow))["messages"]==1


async def test_conflict_precedes_provider_preparation(flow):
    await receive(flow)
    prep=preparation()
    with pytest.raises(ChatRequestConflict):await receive(flow,text="changed",prepare=prep)
    prep.assert_not_called()


@pytest.mark.parametrize("value",["false","true",1,0,None])
async def test_invalid_confirmation_cannot_consume_or_bind(flow,value):
    with pytest.raises(ValueError):await receive(flow,paid_confirmed=value)
    assert await counts(flow)=={"messages":0,"runs":0,"jobs":0,"receipts":0}
    assert (await balances(flow))[1]==[]


async def test_paid_draft_can_confirm_same_id_after_rejection(flow):
    await seed_quota(flow)
    with pytest.raises(ChatIngressQuotaBlocked):await receive(flow)
    assert (await counts(flow))["messages"]==0
    assert (await receive(flow,paid_confirmed=True)).created


async def test_resource_preparation_race_recovers_other_commit(flow):
    ident=await attachment(flow)
    winner=[]
    async def raced(scope,request,**kwargs):
        winner.append(await receive(flow,attachments=[ident],prepare=preparation(metadata={"attachments":[{"id":ident}]})))
        raise ChatIngressResourceInvalid()
    result=await receive(flow,attachments=[ident],prepare=raced)
    assert not result.created and result.message_id==winner[0].message_id
    assert (await counts(flow))["messages"]==1


async def test_absent_resource_after_preparation_failure_does_not_fallback(flow):
    with pytest.raises(ChatIngressResourceInvalid):
        await receive(flow,prepare=AsyncMock(side_effect=ChatIngressResourceInvalid()))
    assert (await counts(flow))["messages"]==0


async def test_reset_during_provider_preparation_fences_entire_commit(flow):
    async def reset(scope,request,**kwargs):
        await flow.db.execute_raw("UPDATE conversations SET title='reset' WHERE id=$1",flow.ids["conversation"])
        # Actual lifecycle invalidation, not a client-provided generation.
        await flow.db.execute_raw("UPDATE conversations SET execution_generation=gen_random_uuid() WHERE id=$1",flow.ids["conversation"])
        return await preparation()(scope,request)
    with pytest.raises(ExecutionScopeExpired):await receive(flow,prepare=reset)
    assert (await counts(flow))["messages"]==0 and (await balances(flow))[1]==[]


@pytest.mark.parametrize("marker",["INSERT INTO messages","INSERT INTO chat_ingress_receipts","UPDATE runtime_jobs"])
async def test_receiver_write_failure_leaves_no_receipt_or_quota(flow,marker):
    class ReceiverFaultDatabase(FaultDatabase):
        async def query_raw(self,*args):return await flow.db.query_raw(*args)
    with pytest.raises(RuntimeError):await receive(flow,database=ReceiverFaultDatabase(flow.db,marker))
    assert await counts(flow)=={"messages":0,"runs":0,"jobs":0,"receipts":0}
    assert (await balances(flow))[1]==[]


class LostCommitDatabase:
    def __init__(self,db):self.db=db
    def __getattr__(self,name):return getattr(self.db,name)
    @asynccontextmanager
    async def tx(self,**kwargs):
        async with self.db.tx(**kwargs) as tx:yield tx
        if self.fail:
            self.fail=False
            raise ConnectionError("synthetic commit acknowledgement lost")
    fail=False


async def test_lost_commit_ack_retries_existing_receipt_without_payment(flow):
    await seed_quota(flow)
    db=LostCommitDatabase(flow.db)
    async def prep(scope,request,**kwargs):
        db.fail=True
        return await preparation()(scope,request)
    with pytest.raises(ConnectionError):await receive(flow,database=db,paid_confirmed=True,prepare=prep)
    before=await balances(flow)
    receipt=await receive(flow,database=db)
    assert not receipt.created and await balances(flow)==before


@pytest.mark.parametrize("status",["succeeded","failed","cancelled"])
async def test_terminal_same_id_returns_state_and_original_snapshot(flow,status):
    first=await receive(flow)
    await flow.db.execute_raw("UPDATE agent_runs SET status=$2,finished_at=clock_timestamp(),updated_at=clock_timestamp() WHERE id=$1",first.run_id,status)
    await flow.db.execute_raw("UPDATE runtime_jobs SET status=$2,finished_at=clock_timestamp(),updated_at=clock_timestamp() WHERE id=$1",first.job_id,status)
    prep=preparation()
    retry=await receive(flow,prepare=prep)
    assert not retry.created and retry.run_status==status
    prep.assert_not_called()
    assert len(entry.accepted_chat_events(retry))==1
    assert (await load_chat_turn(flow.bound,retry.run_id,database=flow.db)).snapshot==snapshot()


@pytest.mark.parametrize("with_attachment",[False,True])
async def test_http_asgi_and_ws_receiver_share_same_sql_identity(flow,monkeypatch,with_attachment):
    from app.api import deps
    from app.api.public import chat
    monkeypatch.setattr(chat,"db",flow.db)
    monkeypatch.setattr(chat,"sql_chat_ingress_enabled",lambda:True)
    monkeypatch.setattr(deps,"is_redis_healthy",lambda:True)
    original=entry.receive_client_chat
    attachment_id=await attachment(flow) if with_attachment else None
    prep=preparation(metadata={"attachments":[{"id":attachment_id}]} if attachment_id else None)
    async def receiver(**kwargs):return await original(**kwargs,database=flow.db,prepare=prep)
    monkeypatch.setattr(entry,"receive_client_chat",receiver)
    forbidden=AsyncMock(side_effect=AssertionError("SQL HTTP route fell into Redis"))
    monkeypatch.setattr(chat,"plan_user_message_aggregation",forbidden)
    monkeypatch.setattr(chat,"enqueue_or_append_delayed",forbidden)
    app=FastAPI();app.include_router(chat.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://synthetic") as client:
        headers={"Authorization":"Bearer "+create_jwt(flow.ids["owner"],"user")}
        body={"client_id":"shared-id","message":"synthetic"}
        if attachment_id:body["attachments"]=[{"id":attachment_id}]
        response=await client.post('/chat/'+flow.ids["conversation"],json=body,headers=headers)
        assert response.status_code==200 and response.text.count("event: ack")==1
        events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        ws=SimpleNamespace(send_json=AsyncMock())
        await entry.handle_sql_chat_frame(ws,flow.ids["owner"],flow.ids["conversation"],body)
        ack=ws.send_json.await_args_list[0].args[0]
        assert ack["type"]=="ack" and ack["data"]["duplicate"]
        assert ack["data"]["message_id"]==events[0]["message_id"]
        conflict=await client.post('/chat/'+flow.ids["conversation"],json={**body,"message":"changed"},headers=headers)
        assert conflict.status_code==409 and "event: ack" not in conflict.text
        missing=await client.post('/chat/'+flow.ids["conversation"],json={"message":"synthetic"},headers=headers)
        assert missing.status_code==422 and "event: ack" not in missing.text
    forbidden.assert_not_called();prep.assert_awaited_once()
    assert (await counts(flow))["messages"]==1


async def test_unknown_actor_cannot_prepare_or_commit(flow):
    prep=preparation()
    with pytest.raises(ExecutionScopeUnavailable):
        await entry.receive_client_chat(actor_user_id="not-an-actor",conversation_id=flow.ids["conversation"],
            payload={"client_id":"id","message":"synthetic"},database=flow.db,prepare=prep)
    prep.assert_not_called()
    assert (await counts(flow))["messages"]==0


@pytest.fixture
def real_preparation(flow,monkeypatch):
    import app.db as database_module
    from app.services import offerings,runtime_config
    from app.services.chat_media import repo,vision
    from app.services.chat_media.prompt import attachment_to_metadata
    from app.services.interaction import reply_context,chat_management,user_turn_aggregation
    from app.services.schedule_domain import schedule
    monkeypatch.setattr(database_module,"db",flow.db)
    monkeypatch.setattr(repo,"db",flow.db)
    monkeypatch.setattr(offerings,"db",flow.db)
    monkeypatch.setattr(runtime_config,"bind_agent_context",AsyncMock(return_value=None))
    monkeypatch.setattr(entry,"_bound_execution_snapshot",lambda _:snapshot())
    monkeypatch.setattr(schedule,"get_cached_schedule",AsyncMock(return_value=[{}]))
    monkeypatch.setattr(schedule,"get_current_status",lambda _:{"status":"idle","type":"leisure"})
    monkeypatch.setattr(reply_context,"build_reply_timing_context",AsyncMock(return_value={"delay_seconds":2}))
    monkeypatch.setattr(chat_management,"user_message_aggregation_enabled",lambda:True)
    monkeypatch.setattr(user_turn_aggregation,"load_pending_action",AsyncMock(return_value=None))
    monkeypatch.setattr(user_turn_aggregation,"load_pending_contradiction",AsyncMock(return_value=None))
    async def summaries(attachments,**kwargs):return [attachment_to_metadata(a) for a in attachments]
    monkeypatch.setattr(vision,"ensure_vision_summaries",summaries)
    monkeypatch.setattr(offerings,"build_offering_user_message",AsyncMock(return_value="synthetic offering"))
    return entry.prepare_sql_chat_message


async def test_real_preparer_binds_ordered_media_without_legacy_hooks(flow,real_preparation):
    ids=[await attachment(flow),await attachment(flow)]
    result=await receive(flow,text="  hi  ",attachments=ids,prepare=real_preparation)
    message=await flow.db.message.find_unique(where={"id":result.message_id})
    assert message.content=="hi" and [a["id"] for a in message.metadata["attachments"]]==ids
    assert message.metadata["ingress_effects"]["attachment_ids"]==ids
    assert (await balances(flow))[1][0]["used"]==1
    retry=await receive(flow,text="  hi  ",attachments=ids,prepare=AsyncMock(side_effect=AssertionError("duplicate prepared again")))
    assert retry.message_id==result.message_id


@pytest.mark.parametrize("kind",["gift","red_packet"])
async def test_real_offering_authorization_is_canonical_and_quota_exempt(flow,real_preparation,kind):
    await seed_quota(flow,permanent=0)
    card=await offering(flow,kind=kind)
    ident=card["payload"]["offering_id"]
    card["title"]="untrusted title"
    result=await receive(flow,text="",component_card=card,prepare=real_preparation)
    message=await flow.db.message.find_unique(where={"id":result.message_id})
    assert message.metadata["component_card"]["title"]!="untrusted title"
    assert message.metadata["ingress_effects"]["quota"]["mode"]=="exempt"
    assert (await balances(flow))[1][0]["used"]==20
    turn=await load_chat_turn(flow.bound,result.run_id,database=flow.db)
    context=json.loads(turn.reply_context_json)
    assert context[kind]["offering_id"]==ident and context["delay_seconds"]==0
    assert turn.phase=="ready" and turn.prompt_text=="synthetic offering"


@pytest.mark.parametrize("body",[
    {"message":"https://synthetic.invalid/link"},
    {"component_card":{"type":"location","payload":{"latitude":1,"longitude":1}}},
    {"component_card":{"type":"music_track","payload":{}}},
])
async def test_unsupported_domains_fail_before_charging_or_provider_calls(flow,real_preparation,body):
    with pytest.raises(ChatIngressResourceInvalid):await receive(flow,prepare=real_preparation,**body)
    assert (await counts(flow))["messages"]==0 and (await balances(flow))[1]==[]
