"""Human takeover contracts against migrated, owned loopback SQL and Redis.

No production/provider access. Exercises real transactions, authorization,
concurrent retries, quota rollback, resource binding and admin HTTP contracts.
"""
import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis

from app.services.admin_takeover import service
from app.services.admin_takeover.ingress import persist_user_message
from app.services.auth import create_jwt
from app.services.runtime.chat_ingress import ChatRequestConflict
from app.services.runtime.chat_ingress_effects import ChatIngressQuotaBlocked, ChatIngressResourceInvalid
from app.services.runtime.execution_scope import ExecutionScopeUnavailable
from tests.test_runtime_execution_foundation import flow
from tests.test_chat_ingress_effects_postgres import attachment, balances, offering, seed_quota
from tests.test_chat_ingress_postgres import FaultDatabase


class ScopedFaultDatabase(FaultDatabase):
    def __getattr__(self, name):
        return getattr(self.database, name)


@pytest.fixture
async def human(flow, monkeypatch):
    import app.db
    from app.services.notifications import service as notifications
    from app.services.runtime.ws_manager import manager
    redis_url = os.environ['PROACTIVE_E2E_REDIS_URL']
    assert redis_url.startswith(('redis://127.0.0.1:', 'redis://localhost:'))
    redis = Redis.from_url(redis_url, decode_responses=True)
    ids = [str(uuid4()), str(uuid4())]
    for ident in ids:
        await flow.db.user.create(data={'id': ident, 'username': 'takeover-admin-'+ident, 'role': 'admin'})
    pending = []
    monkeypatch.setattr(service, 'db', flow.db)
    monkeypatch.setattr(app.db, 'db', flow.db)
    monkeypatch.setattr(service, 'get_redis', AsyncMock(return_value=redis))
    monkeypatch.setattr(service, 'fire_background', pending.append)
    monkeypatch.setattr(service, '_kick_notification_dispatch', lambda: None)
    monkeypatch.setattr(notifications, 'notify_agent_message_created', AsyncMock())
    monkeypatch.setattr(manager, 'send_to_admin', AsyncMock(return_value=True))
    monkeypatch.setattr(manager, 'send_event', AsyncMock(return_value=True))
    await service.open_takeover_case(flow.ids['owner'])
    case = (await flow.db.query_raw('SELECT id FROM human_takeovers WHERE user_id=$1', flow.ids['owner']))[0]['id']
    app = FastAPI()
    from app.api.admin.takeovers import router
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        value = SimpleNamespace(flow=flow, admins=ids, redis=redis, case=case, pending=pending,
            client=client, manager=manager, notifications=notifications)
        try:
            yield value
        finally:
            for coroutine in pending:
                coroutine.close()
            await redis.delete(service.active_conversation_key(flow.ids['conversation']),
                service.active_user_key(flow.ids['owner']), service.status_override_key(flow.ids['conversation']))
            for ident in ids:
                await flow.db.user.delete(where={'id': ident})
            await redis.aclose()


async def attach(h):
    f = h.flow
    await service.attach_takeover_agent(user_id=f.ids['owner'], conversation_id=f.ids['conversation'],
        agent_id=f.ids['agent'], workspace_id=f.ids['workspace'])


async def activate(h):
    await attach(h)
    return await service.claim_takeover(h.case, h.admins[0])


def auth(h, n=0, role='admin'):
    return {'Authorization': 'Bearer '+create_jwt(h.admins[n], role)}


async def user_message(h, key='m1', text='你好', metadata=None, paid=False, database=None):
    return await persist_user_message(conversation_id=h.flow.ids['conversation'], user_id=h.flow.ids['owner'],
        text=text, metadata={'client_id': key, **(metadata or {})}, paid_confirmed=paid,
        database=database or h.flow.db)


async def test_pending_attach_dismiss_review_and_notification(human):
    h=human; f=h.flow
    await service.open_takeover_case(f.ids['owner'])
    assert len(h.pending)==1
    await h.pending.pop()
    alerts=await f.db.query_raw("SELECT user_id FROM notification_events WHERE type='admin_new_user' AND dedupe_key=$1",h.case)
    assert {r['user_id'] for r in alerts} >= set(h.admins)
    assert (await service.review_takeover(h.case))['agent'] is None
    assert (await service.list_takeovers(h.admins[0],scope='pending'))[0]['can_claim'] is False
    await attach(h); await attach(h)
    detail=await service.review_takeover(h.case)
    assert detail['can_claim'] and detail['agent']['conversation_id']==f.ids['conversation']
    assert await service.dismiss_takeover(h.case,h.admins[0])
    assert await service.dismiss_takeover(h.case,h.admins[0])
    assert await service.list_takeovers(h.admins[0],scope='pending')==[]
    assert len(await service.list_takeovers(h.admins[1],scope='pending'))==1
    assert not await service.dismiss_takeover('missing',h.admins[0])
    assert await service.review_takeover('missing') is None


async def test_claim_is_exclusive_and_missing_cache_uses_sql(human):
    h=human; await attach(h)
    results=await asyncio.gather(*(service.claim_takeover(h.case,admin) for admin in h.admins), return_exceptions=True)
    assert sum(type(x) is dict for x in results)==1
    assert sum(isinstance(x,service.TakeoverClaimError) and x.code=='already_taken_over' for x in results)==1
    assert await service.is_user_taken_over(h.flow.ids['owner'])
    await h.redis.delete(service.active_conversation_key(h.flow.ids['conversation']),service.active_user_key(h.flow.ids['owner']))
    assert await service.is_conversation_taken_over(h.flow.ids['conversation'])
    assert await service.is_user_taken_over(h.flow.ids['owner'])
    assert not await service.dismiss_takeover(h.case,h.admins[0])
    assert len(await service.list_takeovers(h.admins[0],scope='mine'))+len(await service.list_takeovers(h.admins[1],scope='mine'))==1


@pytest.mark.parametrize('variant,code',[('missing','not_found'),('unattached','agent_not_ready'),('closed','not_pending')])
async def test_claim_rejects_invalid_cases(human,variant,code):
    h=human
    if variant=='closed':
        await attach(h)
        await h.flow.db.execute_raw("UPDATE human_takeovers SET status='closed' WHERE id=$1",h.case)
    with pytest.raises(service.TakeoverClaimError,match=code):
        await service.claim_takeover('missing' if variant=='missing' else h.case,h.admins[0])


async def test_claim_cache_failure_still_routes_and_status_cache_loss_recovers(human,monkeypatch):
    h=human;await attach(h)
    monkeypatch.setattr(service,'_write_active_flags',AsyncMock(side_effect=ConnectionError()))
    assert (await service.claim_takeover(h.case,h.admins[0]))['status']=='active'
    assert await service.is_conversation_taken_over(h.flow.ids['conversation'])
    result=await service.set_status_override(h.case,h.admins[0],'busy')
    assert result['ai_status']=='busy'
    await h.redis.delete(service.status_override_key(h.flow.ids['conversation']))
    assert await service.display_status_override(h.flow.ids['conversation'])=='busy'
    assert not await service.mark_takeover_read(h.case,h.admins[1],'missing')
    with pytest.raises(service.TakeoverClaimError):await service.set_status_override(h.case,h.admins[1],'idle')
    with pytest.raises(ValueError):await service.set_status_override(h.case,h.admins[0],'invalid')
    assert await service.display_status_override(None) is None
    assert not await service.is_conversation_taken_over(None)
    assert not await service.is_user_taken_over(None)


async def test_user_paid_parallel_retry_and_changed_identity(human):
    h=human;await activate(h);await seed_quota(h.flow)
    results=await asyncio.gather(*(user_message(h,paid=True) for _ in range(6)))
    assert len({r['message_id'] for r in results})==1 and sum(r['created'] for r in results)==1
    rows,quota,ledger=await balances(h.flow)
    assert rows[0]['ticket_balance']==5 and quota[0]['used']==21 and len(ledger)==1
    with pytest.raises(ChatRequestConflict):await user_message(h,text='changed',paid=True)
    assert (await balances(h.flow))[1]==quota
    case=(await h.flow.db.query_raw('SELECT unread_count FROM human_takeovers WHERE id=$1',h.case))[0]
    assert case['unread_count']==1
    assert await service.mark_takeover_read(h.case,h.admins[0],results[0]['message_id'])
    assert (await service.list_takeovers(h.admins[0],scope='mine'))[0]['read_state']=='read'


async def test_quota_rejection_and_transaction_failure_rollback_every_effect(human):
    h=human;await activate(h);await seed_quota(h.flow)
    with pytest.raises(ChatIngressQuotaBlocked):await user_message(h)
    assert await h.flow.db.message.count(where={'conversationId':h.flow.ids['conversation']})==0
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        await user_message(h,paid=True,database=ScopedFaultDatabase(h.flow.db,"UPDATE human_takeovers"))
    assert await h.flow.db.message.count(where={'conversationId':h.flow.ids['conversation']})==0
    assert (await balances(h.flow))[1][0]['used']==20
    with pytest.raises(ConnectionError):
        await user_message(h,paid=True,database=ScopedFaultDatabase(h.flow.db,commit_unknown=True))
    assert not (await user_message(h,paid=True))['created']
    assert (await balances(h.flow))[1][0]['used']==21


async def test_resource_binding_is_authoritative_and_retry_safe(human):
    h=human;await activate(h)
    ident=await attachment(h.flow)
    first=await user_message(h,metadata={'attachments':[{'id':ident,'url':'https://forged.invalid'}]})
    retry=await user_message(h,metadata={'attachments':[{'id':ident}]})
    assert first['message_id']==retry['message_id'] and not retry['created']
    message=await h.flow.db.message.find_unique(where={'id':first['message_id']})
    assert message.metadata['attachments'][0]['url']=='https://synthetic.invalid/image'
    assert (await h.flow.db.query_raw('SELECT message_id FROM chat_message_attachments WHERE id=$1',ident))[0]['message_id']==first['message_id']
    wrong=await attachment(h.flow,conversation=h.flow.ids['other_conversation'])
    for bad in [wrong,'missing',ident]:
        with pytest.raises(ChatIngressResourceInvalid):await user_message(h,key=bad,metadata={'attachments':[{'id':bad}]})
    assert (await balances(h.flow))[1][0]['used']==1


@pytest.mark.parametrize('kind',['gift','red_packet'])
async def test_offerings_authorize_bind_and_exempt_quota(human,kind):
    h=human;await activate(h);await seed_quota(h.flow,permanent=0)
    card=await offering(h.flow,kind=kind)
    result=await user_message(h,metadata={'component_card':card})
    message=await h.flow.db.message.find_unique(where={'id':result['message_id']})
    assert message.metadata['component_card']==card and message.metadata['ingress_effects']['quota']['mode']=='exempt'
    assert (await balances(h.flow))[1][0]['used']==20
    with pytest.raises(ChatIngressResourceInvalid):await user_message(h,key='another',metadata={'component_card':card})


@pytest.mark.parametrize('card',[
    {'type':'gift'}, {'type':'gift','payload':{}}, {'type':'music'},
    {'type':'gift','payload':{'offering_id':'missing'}},
    {'type':'external_link'},
    {'type':'external_link','payload':{'source_url':'https://synthetic.invalid/unprepared'}},
    {'type':'external_link','payload':{'link_id':'missing','snapshot_id':'missing'}},
])
async def test_forged_or_unsupported_card_cannot_bypass_quota(human,card):
    await activate(human)
    with pytest.raises(ChatIngressResourceInvalid):await user_message(human,metadata={'component_card':card})
    assert (await balances(human.flow))[1]==[]


async def test_owner_reset_inactive_and_unclaimed_fail_before_charge(human):
    h=human
    with pytest.raises(ExecutionScopeUnavailable):await user_message(h)
    await activate(h)
    await h.flow.db.conversation.update(where={'id':h.flow.ids['conversation']},data={'isDeleted':True})
    with pytest.raises(ExecutionScopeUnavailable):await user_message(h)
    assert (await balances(h.flow))[1]==[]


async def test_admin_message_resource_retry_and_non_owner_are_fenced(human):
    h=human;await activate(h)
    ident=await attachment(h.flow)
    kwargs=dict(takeover_id=h.case,admin_id=h.admins[0],text='人工回复',client_id='admin-1',
        component_card=None,attachments=[])
    first=await service.send_admin_message(**kwargs)
    second=await service.send_admin_message(**kwargs)
    assert first['message_id']==second['message_id'] and second['duplicate']
    assert h.manager.send_event.await_count==2
    assert 'attachments' not in h.manager.send_event.await_args_list[0].args[2]
    h.notifications.notify_agent_message_created.assert_not_awaited()
    await h.pending.pop()
    h.notifications.notify_agent_message_created.assert_awaited_once()
    assert (await balances(h.flow))[1]==[]
    with pytest.raises(ChatRequestConflict):await service.send_admin_message(**{**kwargs,'text':'changed'})
    with pytest.raises(service.TakeoverClaimError):await service.send_admin_message(**{**kwargs,'admin_id':h.admins[1]})
    for extra in [{'attachments':[{'id':ident}]},{'component_card':{'type':'gift'}}]:
        with pytest.raises((ValueError, ChatIngressResourceInvalid)):await service.send_admin_message(**{**kwargs,**extra,'client_id':'another'})
    assert await service.takeover_media_owner(h.admins[0],h.flow.ids['conversation'],is_admin=True)==h.flow.ids['owner']
    assert await service.takeover_media_owner(h.admins[1],h.flow.ids['conversation'],is_admin=True) is None
    assert await service.takeover_media_owner(h.admins[0],h.flow.ids['conversation'],is_admin=False) is None
    await h.flow.db.user.update(where={'id':h.admins[0]},data={'role':'user'})
    with pytest.raises(ExecutionScopeUnavailable):await service.send_admin_message(**{**kwargs,'client_id':'demoted'})


async def test_record_notifies_once_after_commit_and_does_not_duplicate(human):
    h=human;await activate(h)
    args=dict(conversation_id=h.flow.ids['conversation'],user_id=h.flow.ids['owner'],text='hello',metadata={'client_id':'r1'})
    first=await service.record_takeover_user_message(**args)
    retry=await service.record_takeover_user_message(**args)
    assert first['message_id']==retry['message_id'] and retry['duplicate']
    h.manager.send_to_admin.assert_awaited_once()
    alerts=await h.flow.db.query_raw("SELECT dedupe_key FROM notification_events WHERE type='admin_takeover_message' AND message_id=$1",first['message_id'])
    assert len(alerts)==1 and alerts[0]['dedupe_key']==first['message_id']


async def test_read_cursor_keeps_newer_arrivals_unread_and_is_monotonic(human):
    h=human;await activate(h)
    first=await user_message(h,key='first')
    newer=await user_message(h,key='newer')
    assert await service.mark_takeover_read(h.case,h.admins[0],first['message_id'])
    assert (await service.takeover_unread_summary(h.admins[0]))['unread_count']==1
    assert (await service.takeover_unread_summary(h.admins[1]))['unread_count']==0
    assert not await service.mark_takeover_read(h.case,h.admins[1],newer['message_id'])
    assert not await service.mark_takeover_read(h.case,h.admins[0],'missing')
    assert await service.mark_takeover_read(h.case,h.admins[0],newer['message_id'])
    assert await service.mark_takeover_read(h.case,h.admins[0],first['message_id'])
    assert (await service.takeover_unread_summary(h.admins[0]))['unread_count']==0


async def test_message_and_owner_alert_rollback_together(human):
    h=human;await activate(h)
    with pytest.raises(RuntimeError,match='synthetic write failure'):
        await user_message(h,database=ScopedFaultDatabase(h.flow.db,'INSERT INTO notification_events'))
    assert await h.flow.db.message.count(where={'conversationId':h.flow.ids['conversation']})==0
    assert (await service.takeover_unread_summary(h.admins[0]))['unread_count']==0


async def test_complete_inbox_payload_and_named_durable_alert_survive_ws_failure(human):
    h=human;await activate(h)
    await h.flow.db.user.update(where={'id':h.flow.ids['owner']},data={'displayName':'测试昵称'})
    h.manager.send_to_admin.side_effect=ConnectionError('synthetic disconnect')
    ident=await attachment(h.flow)
    text='完整消息'*100
    result=await service.record_takeover_user_message(conversation_id=h.flow.ids['conversation'],
        user_id=h.flow.ids['owner'],text=text,metadata={'client_id':'long','attachments':[{'id':ident}]})
    payload=h.manager.send_to_admin.await_args.args[2]
    assert payload['text']==text and len(payload['preview'])<=80 and payload['user_name']=='测试昵称'
    assert payload['metadata']['attachments'][0]['url']=='https://synthetic.invalid/image'
    alerts=await h.flow.db.query_raw('SELECT title,payload FROM notification_events WHERE message_id=$1',result['message_id'])
    assert len(alerts)==1 and alerts[0]['title']=='测试昵称'
    assert alerts[0]['payload']['title']=='测试昵称'


async def test_admin_http_auth_bounds_claim_status_messages_and_read(human):
    h=human;c=h.client;base='/admin-api/takeovers'
    assert (await c.get(base)).status_code==401
    user_auth={'Authorization':'Bearer '+create_jwt(h.flow.ids['owner'],role='user')}
    assert (await c.get(base,headers=user_auth)).status_code==403
    assert (await c.get(base+'?scope=bad',headers=auth(h))).status_code==400
    assert (await c.get(base+'/missing',headers=auth(h))).status_code==404
    assert (await c.post(base+'/missing/dismiss',headers=auth(h))).status_code==404
    assert (await c.post(base+'/missing/claim',headers=auth(h))).status_code==404
    assert (await c.post(base+'/'+h.case+'/claim',headers=auth(h))).status_code==409
    await attach(h)
    assert (await c.get(base+'/'+h.case,headers=auth(h))).json()['can_claim']
    assert (await c.post(base+'/'+h.case+'/claim',headers=auth(h))).status_code==200
    assert (await c.post(base+'/'+h.case+'/claim',headers=auth(h,1))).status_code==409
    for body,status in [({'text':''},400),({'text':'x'*4001},400),({'text':'  hello  ','client_id':'http1'},200)]:
        assert (await c.post(base+'/'+h.case+'/messages',json=body,headers=auth(h))).status_code==status
    assert (await c.post(base+'/'+h.case+'/messages',json={'text':'changed','client_id':'http1'},headers=auth(h))).status_code==409
    assert (await c.post(base+'/'+h.case+'/messages',json={'text':'hello','attachments':[{'id':'bad'}]},headers=auth(h))).status_code==422
    assert (await c.post(base+'/'+h.case+'/messages',json={'text':'hello'},headers=auth(h,1))).status_code==403
    incoming=await user_message(h)
    assert (await c.get(base+'/unread-summary',headers=user_auth)).status_code==403
    summary=(await c.get(base+'/unread-summary',headers=auth(h))).json()
    assert summary['unread_count']==1 and summary['unread_chats']==1 and summary['sampled_at']
    read={'message_id':incoming['message_id']}
    assert (await c.post(base+'/'+h.case+'/read',json=read,headers=auth(h,1))).status_code==403
    assert (await c.post(base+'/'+h.case+'/read',json=read,headers=auth(h))).status_code==200
    assert (await c.get(base+'/unread-summary',headers=auth(h))).json()['unread_count']==0
    assert (await c.patch(base+'/'+h.case+'/status',json={'status':'bad'},headers=auth(h))).status_code==400
    assert (await c.patch(base+'/'+h.case+'/status',json={'status':'sleep'},headers=auth(h,1))).status_code==403
    assert (await c.patch(base+'/'+h.case+'/status',json={'status':'sleep'},headers=auth(h))).status_code==200
    assert len((await c.get(base+'?scope=mine',headers=auth(h))).json()['items'])==1
    await h.flow.db.user.update(where={'id':h.admins[0]},data={'role':'user'})
    assert (await c.get(base,headers=auth(h))).status_code==403


async def test_nonfinancial_sharing_is_delivered_without_automation_and_pays_quota(human):
    from app.services.chat_links.cards import component_card_for_link, metadata_for_link_card
    from app.services.chat_links.extraction import LinkMetadata
    from app.services.chat_links.repo import create_or_update_link_card

    h=human;await activate(h)
    link=await create_or_update_link_card(user_id=h.flow.ids['owner'],
        conversation_id=h.flow.ids['conversation'], database=h.flow.db,
        metadata=LinkMetadata(source_url='https://synthetic.invalid/article',
            final_url='https://synthetic.invalid/article', platform='链接',
            title='saved share', summary='shared'))
    card=component_card_for_link(link)
    missing_version=json.loads(json.dumps(card))
    missing_version['payload']['snapshot_id']='missing-version'
    with pytest.raises(ChatIngressResourceInvalid):
        await user_message(h,key='missing-version',metadata={'component_card':missing_version})
    assert (await balances(h.flow))[1]==[]
    result=await user_message(h,metadata={'component_card':card})
    saved=await h.flow.db.message.find_unique(where={'id':result['message_id']})
    assert saved.metadata['component_card']==card
    assert saved.metadata['link_card']==metadata_for_link_card(link)
    assert (await balances(h.flow))[1][0]['used']==1
    assert await h.flow.db.agentrun.count(where={'conversationId':h.flow.ids['conversation']})==0


async def test_legacy_link_snapshot_and_binding_roll_back_with_admission(human):
    from app.services.chat_links.cards import component_card_for_link
    from app.services.chat_links.extraction import LinkMetadata
    from app.services.chat_links.repo import create_or_update_link_card

    h=human;await activate(h);await seed_quota(h.flow)
    link=await create_or_update_link_card(user_id=h.flow.ids['owner'],
        conversation_id=h.flow.ids['conversation'], database=h.flow.db,
        metadata=LinkMetadata(source_url='https://synthetic.invalid/legacy',
            final_url='https://synthetic.invalid/legacy', platform='链接', title='legacy'))
    await h.flow.db.execute_raw('DELETE FROM chat_link_snapshots WHERE id=$1',link.snapshot_id)
    card=component_card_for_link(link)
    card['payload'].pop('snapshot_id')
    before=await balances(h.flow)

    async def assert_rolled_back():
        assert await balances(h.flow)==before
        assert await h.flow.db.message.count(where={'conversationId':h.flow.ids['conversation']})==0
        assert await h.flow.db.query_raw('SELECT id FROM chat_link_snapshots WHERE link_card_id=$1',link.id)==[]
        assert await h.flow.db.query_raw('SELECT b.message_id FROM chat_link_message_bindings b JOIN chat_link_snapshots s ON s.id=b.snapshot_id WHERE s.link_card_id=$1',link.id)==[]
        assert (await service.takeover_unread_summary(h.admins[0]))['unread_count']==0

    with pytest.raises(ChatIngressQuotaBlocked):
        await user_message(h,metadata={'component_card':card})
    await assert_rolled_back()
    with pytest.raises(RuntimeError,match='synthetic write failure'):
        await user_message(h,metadata={'component_card':card},paid=True,
            database=ScopedFaultDatabase(h.flow.db,'INSERT INTO notification_events'))
    await assert_rolled_back()
    result=await user_message(h,metadata={'component_card':card},paid=True)
    retry=await user_message(h,metadata={'component_card':card},paid=True)
    assert retry['message_id']==result['message_id'] and not retry['created']
    saved=await h.flow.db.message.find_unique(where={'id':result['message_id']})
    version=saved.metadata['link_card']['snapshot_id']
    assert version and version!=link.snapshot_id
    assert await h.flow.db.query_raw('SELECT id FROM chat_link_snapshots WHERE link_card_id=$1',link.id)==[{'id':version}]
    assert await h.flow.db.query_raw('SELECT message_id FROM chat_link_message_bindings WHERE snapshot_id=$1',version)==[{'message_id':result['message_id']}]
    assert (await balances(h.flow))[1]==[{'used':21}]


async def test_takeover_literal_url_dedupes_as_text_without_link_metadata(human):
    h=human;await activate(h)
    text='人工聊天 https://synthetic.invalid/keep-literal'
    first=await user_message(h,key='literal',text=text)
    retry=await user_message(h,key='literal',text=text)
    assert first['message_id']==retry['message_id'] and not retry['created']
    saved=await h.flow.db.message.find_unique(where={'id':first['message_id']})
    assert saved.content==text
    assert 'link_card' not in saved.metadata and 'component_card' not in saved.metadata
    assert await h.flow.db.query_raw('SELECT id FROM chat_link_cards WHERE conversation_id=$1',h.flow.ids['conversation'])==[]
    assert (await balances(h.flow))[1]==[{'used':1}]
    with pytest.raises(ChatRequestConflict):
        await user_message(h,key='literal',text=text+'/changed')


@pytest.mark.parametrize('resource',[
    {'component_card':{'type':'external_link','payload':{'link_id':'unprepared'}}},
    {'attachments':[{'id':'unprepared'}]},
])
async def test_ordinary_http_resource_rejection_precedes_preparation(human,monkeypatch,resource):
    from app.api.public import chat
    from app.api import deps
    h=human;await attach(h)
    monkeypatch.setattr(chat,'db',h.flow.db)
    monkeypatch.setattr(deps,'is_redis_healthy',lambda:True)
    monkeypatch.setattr(chat,'get_cached_schedule',AsyncMock(side_effect=AssertionError('preparation entered')))
    app=FastAPI();app.include_router(chat.router)
    headers={'Authorization':'Bearer '+create_jwt(h.flow.ids['owner'],'user')}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        response=await client.post('/chat/'+h.flow.ids['conversation'],headers=headers,
            json={'message':'resource','client_id':'unsupported',**resource})
        assert response.status_code==422
        assert response.json()['detail']=={'code':'unsupported_transport_payload',
            'message':'卡片和附件请通过聊天连接发送','client_id':'unsupported'}
    assert await h.flow.db.message.count(where={'conversationId':h.flow.ids['conversation']})==0
    assert (await balances(h.flow))[1]==[]
    chat.get_cached_schedule.assert_not_called()


async def test_sse_takeover_acceptance_retry_and_resource_failure_use_real_transaction(human,monkeypatch):
    from app.api.public import chat
    from app.api import deps
    h=human;await activate(h)
    monkeypatch.setattr(chat,'db',h.flow.db)
    monkeypatch.setattr(deps,'is_redis_healthy',lambda:True)
    monkeypatch.setattr(chat,'get_cached_schedule',AsyncMock(side_effect=AssertionError('AI path entered')))
    app=FastAPI();app.include_router(chat.router)
    headers={'Authorization':'Bearer '+create_jwt(h.flow.ids['owner'],'user')}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        path='/chat/'+h.flow.ids['conversation'];body={'message':'sse hello','client_id':'sse-1'}
        first=await client.post(path,headers=headers,json=body)
        assert first.status_code==200 and 'event: ack' in first.text and '"acceptance_only": true' in first.text
        again=await client.post(path,headers=headers,json=body)
        assert again.status_code==200 and '"duplicate": true' in again.text
        assert (await client.post(path,headers=headers,json={**body,'message':'changed'})).status_code==409
        assert (await client.post(path,headers=headers,json={'message':'bad','client_id':'bad','attachments':[{'id':'missing'}]})).status_code==422
        await h.flow.db.execute_raw('UPDATE user_message_quota SET used=20 WHERE user_id=$1',h.flow.ids['owner'])
        assert (await client.post(path,headers=headers,json={'message':'paid','client_id':'paid'})).status_code==429
    assert await h.flow.db.message.count(where={'conversationId':h.flow.ids['conversation']})==1
    chat.get_cached_schedule.assert_not_called()


async def test_claimed_user_suppresses_all_proactive_entrypoints_before_model_or_memory(human,monkeypatch):
    from datetime import datetime,timezone
    from app.services.proactive import followup,gates,sender,special_dates,triggers
    h=human;await activate(h)
    # Drop both caches to exercise SQL recovery, not just a prefilled flag.
    await h.redis.delete(service.active_user_key(h.flow.ids['owner']),service.active_conversation_key(h.flow.ids['conversation']))
    state=SimpleNamespace(user_id=h.flow.ids['owner'])
    stop=AsyncMock();monkeypatch.setattr(followup,'stop_proactive_state',stop)
    monkeypatch.setattr(followup,'check_workspace',AsyncMock(side_effect=AssertionError('ordinary preparation entered')))
    monkeypatch.setattr(followup,'_judge_and_maybe_follow_up',AsyncMock(side_effect=AssertionError('LLM entered')))
    await followup.process_followup_window(state)
    assert stop.await_args.kwargs['reason']=='human_takeover'
    followup._judge_and_maybe_follow_up.assert_not_called()
    assert (await gates.check_window_gates(state)).reason=='human_takeover'
    assert not await sender.generate_and_send_proactive(state,trigger_type='daily_scene')
    assert not await sender.send_first_greeting(conversation_id=h.flow.ids['conversation'],agent_id=h.flow.ids['agent'],user_id=h.flow.ids['owner'])
    assert not await special_dates.send_special_date_proactive(user_id=h.flow.ids['owner'],agent_id=h.flow.ids['agent'],occasions=[{}])
    trigger=SimpleNamespace(aiAgentId=h.flow.ids['agent'],userId=h.flow.ids['owner'],id='synthetic-trigger')
    assert await triggers._handle_reminder_trigger(trigger,datetime.now(timezone.utc)) is None
    assert await h.flow.db.message.count(where={'conversationId':h.flow.ids['conversation']})==0
