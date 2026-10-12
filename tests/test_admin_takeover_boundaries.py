"""Fault/transport boundaries not modeled by the real SQL happy-path fixture."""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, WebSocketDisconnect, Response
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from app.services.admin_takeover import service
from app.services.runtime import ws_auth, ws_manager
from app.api.admin import takeovers
from app.api.realtime import admin_inbox, ws


def test_nullable_row_aliases_preview_and_list_contract():
    assert service._cell(None,'id') is None
    assert service._cell({'first':None,'second':'value'},'first','second')=='value'
    assert service._cell(SimpleNamespace(first=None,second='value'),'first','second')=='value'
    assert service._cell(SimpleNamespace(),'missing',default='fallback')=='fallback'
    assert service._trim('a'*90,80)=='a'*79+'…'
    assert service._trim(None,80)==''
    row=service._list_item({'id':'t','user_id':'u','unread_count':1,'conversation_id':'c','created_at':'now'})
    assert row['read_state']=='unread' and row['can_claim']
    assert takeovers._claim_http(service.TakeoverClaimError('not_pending')).status_code==409
    with pytest.raises(Exception) as error:takeovers._admin_id({})
    assert error.value.status_code==401


async def test_registration_and_attach_failure_are_isolated(monkeypatch):
    query=AsyncMock(side_effect=RuntimeError('db unavailable'))
    execute=AsyncMock(side_effect=RuntimeError('db unavailable'))
    monkeypatch.setattr(service.db,'query_raw',query)
    monkeypatch.setattr(service.db,'execute_raw',execute)
    await service.open_takeover_case('')
    await service.open_takeover_case('synthetic')
    await service.attach_takeover_agent(user_id='',conversation_id='c',agent_id='a',workspace_id=None)
    await service.attach_takeover_agent(user_id='u',conversation_id='c',agent_id='a',workspace_id=None)
    assert query.await_count==1 and execute.await_count==1


async def test_flags_fallback_database_and_cache_write_failures(monkeypatch):
    redis=SimpleNamespace(get=AsyncMock(return_value=None),set=AsyncMock(side_effect=ConnectionError()))
    monkeypatch.setattr(service,'get_redis',AsyncMock(return_value=redis))
    monkeypatch.setattr(service.db,'query_raw',AsyncMock(return_value=[{'exists':1}]))
    assert await service.is_conversation_taken_over('c')
    assert await service.is_user_taken_over('u')
    monkeypatch.setattr(service,'get_redis',AsyncMock(side_effect=ConnectionError()))
    assert await service.is_conversation_taken_over('c')
    monkeypatch.setattr(service.db,'query_raw',AsyncMock(side_effect=RuntimeError()))
    assert not await service.is_user_taken_over('u')


@pytest.mark.parametrize('phone,expected',[('',None),('123',None),('+86 13800138000','861****8000')])
async def test_phone_is_masked_or_unavailable(monkeypatch,phone,expected):
    monkeypatch.setattr(service.db,'query_raw',AsyncMock(return_value=[{'phone':phone}] if phone else []))
    assert await service._masked_phone('u')==expected


async def test_notification_empty_profile_and_unknown_admin(monkeypatch):
    from app.services.notifications import service as notify
    query=AsyncMock(side_effect=[[],[{'id':None},{'id':'admin'}]])
    monkeypatch.setattr(service.db,'query_raw',query)
    monkeypatch.setattr(notify,'enqueue_notification',AsyncMock())
    from app.services.runtime.ws_manager import manager
    monkeypatch.setattr(service,'_kick_notification_dispatch',lambda:None)
    monkeypatch.setattr(manager,'send_to_admin',AsyncMock())
    await service._notify_admins_new_user('t','u')
    assert manager.send_to_admin.await_args.args[2]['body']=='新用户'
    await service._push_owner_message('a','t','c','m','')
    assert notify.enqueue_notification.await_args.kwargs['body']=='你收到了一条新消息'


async def test_registration_fanout_continues_after_one_inbox_fails(monkeypatch):
    from app.services.runtime.ws_manager import manager
    monkeypatch.setattr(service.db,'query_raw',AsyncMock(side_effect=[
        [{'displayName':'新昵称'}],[{'id':'first'},{'id':'second'}]]))
    send=AsyncMock(side_effect=[ConnectionError('offline'),True])
    monkeypatch.setattr(manager,'send_to_admin',send)
    kick=MagicMock()
    monkeypatch.setattr(service,'_kick_notification_dispatch',kick)
    await service._notify_admins_new_user('case','user')
    assert [call.args[0] for call in send.await_args_list]==['first','second']
    kick.assert_called_once()


def test_dispatch_kick_uses_bounded_existing_dispatcher(monkeypatch):
    from app.services.notifications import dispatcher
    calls=[]
    async def dispatch_due_notifications(*,limit):
        calls.append(limit)
    monkeypatch.setattr(dispatcher,'dispatch_due_notifications',dispatch_due_notifications)
    monkeypatch.setattr(service,'fire_background',lambda coro: asyncio.run(coro))
    service._kick_notification_dispatch()
    assert calls==[50]


@pytest.mark.parametrize('error,status',[(ws_auth.TicketInvalid,401),(ws_auth.TicketRateLimited,429),(ws_auth.TicketUnavailable,503)])
async def test_ticket_service_errors_are_specific(monkeypatch,error,status):
    monkeypatch.setattr(takeovers,'issue_ticket',AsyncMock(side_effect=error()))
    with pytest.raises(Exception) as caught:await takeovers.inbox_ticket(Response(),{'sub':'a','role':'admin'})
    assert caught.value.status_code==status


async def test_ws_commit_errors_are_not_acked_or_fed_to_model(monkeypatch):
    from app.services.runtime.chat_ingress_effects import ChatIngressQuotaBlocked
    socket=SimpleNamespace(send_json=AsyncMock())
    quota={'reason':'paid_confirm','per_msg_cost':.5,'spendable_tickets':1}
    monkeypatch.setattr(service,'record_takeover_user_message',AsyncMock(side_effect=ChatIngressQuotaBlocked(quota)))
    await ws._handle_takeover_user_message(socket,'c','u','hi',client_id='m')
    assert socket.send_json.await_args.args[0]['type']=='quota_blocked'
    monkeypatch.setattr(service,'record_takeover_user_message',AsyncMock(side_effect=ValueError()))
    await ws._handle_takeover_user_message(socket,'c','u','hi',client_id='m')
    assert socket.send_json.await_args.args[0]['type']=='error'
    monkeypatch.setattr(service,'record_takeover_user_message',AsyncMock(return_value={'message_id':'saved'}))
    monkeypatch.setattr(ws,'_send_ack',AsyncMock())
    await ws._handle_takeover_user_message(socket,'c','u','hi',client_id='m')
    ws._send_ack.assert_awaited_once_with(socket,message_id='saved',client_id='m')


async def test_ws_gate_and_handler_fault_keep_socket_alive(monkeypatch):
    socket=SimpleNamespace(send_json=AsyncMock())
    monkeypatch.setattr(service,'is_conversation_taken_over',AsyncMock(side_effect=RuntimeError()))
    monkeypatch.setattr(ws.wallet,'is_vip',AsyncMock(side_effect=RuntimeError()))
    await ws._handle_message(socket,'c','u',SimpleNamespace(id='a'),'hi')
    monkeypatch.setattr(service,'is_conversation_taken_over',AsyncMock(return_value=True))
    monkeypatch.setattr(ws,'_handle_takeover_user_message',AsyncMock(side_effect=RuntimeError()))
    await ws._handle_message(socket,'c','u',SimpleNamespace(id='a'),'hi')
    assert socket.send_json.await_args.args[0]['type']=='error'
    socket.send_json.side_effect=ConnectionError()
    await ws._handle_message(socket,'c','u',SimpleNamespace(id='a'),'hi')


def socket():
    return SimpleNamespace(client_state=WebSocketState.CONNECTED,send_json=AsyncMock(),close=AsyncMock())


async def test_admin_manager_replacement_stale_cleanup_and_publish(monkeypatch):
    mgr=ws_manager.ConnectionManager();a=socket();b=socket()
    monkeypatch.setattr(mgr,'_publish',AsyncMock(return_value=True))
    await mgr.connect_admin('admin',a);await mgr.connect_admin('admin',b)
    a.close.assert_awaited_once()
    await mgr.disconnect_admin('admin',expected=a)
    assert mgr._admin_connections['admin'] is b
    assert not await mgr.send_to_admin('','event')
    assert await mgr.send_to_admin('admin','event',{'value':1})
    mgr._publish.assert_not_called()
    assert b.send_json.await_args.args[0]['data']=={'value':1}
    b.send_json.side_effect=ConnectionError()
    assert await mgr.send_to_admin('admin','event')
    assert 'admin' not in mgr._admin_connections
    mgr._publish.assert_awaited_once_with('ws:admin:{admin}','event',None)
    await mgr.disconnect_admin('missing')
    a.close.side_effect=ConnectionError()
    await mgr.connect_admin('admin',a);await mgr.connect_admin('admin',b)
    assert not mgr._connections
    assert ws_manager._parse_channel('ws:admin:{admin}')==('admin','admin')
    import json
    await mgr._handle_message({'channel':'ws:admin:{admin}','data':json.dumps({'type':'event','data':{}})})


@pytest.fixture
def inbox(monkeypatch):
    app=FastAPI();app.include_router(admin_inbox.router)
    monkeypatch.setattr(admin_inbox,'is_redis_healthy',lambda:True)
    monkeypatch.setattr(admin_inbox,'offered_ticket',lambda _: 'ticket')
    monkeypatch.setattr(admin_inbox,'origin_allowed',lambda _:True)
    monkeypatch.setattr(admin_inbox,'consume_ticket',AsyncMock(return_value=ws_auth.SocketPrincipal('admin','admin',time.time()+60)))
    import app.db as db_module
    monkeypatch.setattr(db_module.db,'query_raw',AsyncMock(return_value=[{'exists':1}]))
    monkeypatch.setattr(admin_inbox.manager,'connect_admin',AsyncMock())
    monkeypatch.setattr(admin_inbox.manager,'disconnect_admin',AsyncMock())
    return TestClient(app)


@pytest.mark.parametrize('case,code',[('redis',1011),('origin',4401),('ticket',4401),('unavailable',1011),('role',4403),('identity',4403),('inactive',4403)])
def test_inbox_auth_precedes_connect(inbox,monkeypatch,case,code):
    if case=='redis':monkeypatch.setattr(admin_inbox,'is_redis_healthy',lambda:False)
    if case=='origin':monkeypatch.setattr(admin_inbox,'origin_allowed',lambda _:False)
    if case in {'ticket','unavailable'}:monkeypatch.setattr(admin_inbox,'consume_ticket',AsyncMock(side_effect=(ws_auth.TicketInvalid if case=='ticket' else ws_auth.TicketUnavailable)()))
    if case in {'role','identity'}:monkeypatch.setattr(admin_inbox,'consume_ticket',AsyncMock(return_value=ws_auth.SocketPrincipal('other' if case=='identity' else 'admin','user' if case=='role' else 'admin',time.time()+60)))
    if case=='inactive':
        import app.db
        app.db.db.query_raw.return_value=[]
    with pytest.raises(WebSocketDisconnect) as caught:
        with inbox.websocket_connect('/ws/admin-inbox/admin'):pass
    assert caught.value.code==code
    admin_inbox.manager.connect_admin.assert_not_called()


def test_inbox_ping_cleanup_timeout_and_expiration(inbox,monkeypatch):
    with inbox.websocket_connect('/ws/admin-inbox/admin') as peer:
        peer.send_json({'type':'ping'})
        assert peer.receive_json()=={'type':'pong'}
    admin_inbox.manager.disconnect_admin.assert_awaited()
    monkeypatch.setattr(admin_inbox,'_IDLE_TIMEOUT',.01)
    with inbox.websocket_connect('/ws/admin-inbox/admin') as peer:
        with pytest.raises(WebSocketDisconnect) as error:peer.receive_json()
        assert error.value.code==4008
    monkeypatch.setattr(admin_inbox,'consume_ticket',AsyncMock(return_value=ws_auth.SocketPrincipal('admin','admin',time.time()-1)))
    with inbox.websocket_connect('/ws/admin-inbox/admin') as peer:
        with pytest.raises(WebSocketDisconnect) as error:peer.receive_json()
        assert error.value.code==4401


@pytest.mark.parametrize('fails',[False,True])
async def test_takeover_audio_stores_validated_audio_without_asr_and_cleans_failed_file(monkeypatch,fails):
    import base64
    from app.api.public import speech
    from app.models.speech import ChatAudioTranscriptionRequest
    from tests.test_speech_to_text import _test_m4a
    audio=_test_m4a(3)
    data=ChatAudioTranscriptionRequest(conversation_id='c',base64=base64.b64encode(audio).decode(),
        mime='audio/mp4',name='voice.m4a',duration_seconds=3,size=len(audio))
    monkeypatch.setattr(speech.chat_media_repo,'conversation_belongs_to_user',AsyncMock(return_value=False))
    monkeypatch.setattr(service,'takeover_media_owner',AsyncMock(return_value='owner'))
    monkeypatch.setattr(service,'is_conversation_taken_over',AsyncMock(return_value=True))
    monkeypatch.setattr(speech,'_enforce_rate_limit',AsyncMock())
    monkeypatch.setattr(speech,'transcribe_audio',AsyncMock(side_effect=AssertionError('ASR entered')))
    monkeypatch.setattr(speech.chat_media_storage,'save_audio_blob',MagicMock(return_value='owned-file'))
    monkeypatch.setattr(speech.chat_media_storage,'media_url',lambda _: '/media/owned-file')
    monkeypatch.setattr(speech.chat_media_storage,'delete_media_file',MagicMock())
    monkeypatch.setattr(speech.chat_media_repo,'create_audio_attachment',AsyncMock(side_effect=RuntimeError('write failure') if fails else None,return_value='attachment'))
    monkeypatch.setattr(speech,'_attachment_response',lambda _:None)
    if fails:
        with pytest.raises(RuntimeError,match='write failure'):await speech.transcribe_chat_audio(data,{'sub':'admin','role':'admin'})
        speech.chat_media_storage.delete_media_file.assert_called_once_with('owned-file')
    else:
        response=await speech.transcribe_chat_audio(data,{'sub':'admin','role':'admin'})
        assert response.model=='human_takeover' and response.text=='［语音］'
        speech.chat_media_storage.delete_media_file.assert_not_called()
    speech._enforce_rate_limit.assert_awaited_once_with('owner')
    speech.transcribe_audio.assert_not_called()
