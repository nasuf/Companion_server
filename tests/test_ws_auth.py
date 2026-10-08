"""Real JWT + routes, synthetic business boundaries and deterministic tickets."""
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from app.api import deps
from app.api.public import chat
from app.api.realtime import ws
from app.config import settings
from app.services.auth import create_jwt
from app.services.runtime import ws_auth as auth


class TicketRedis:
    def __init__(self):
        self.values = {}
        self.rate = 0
        self.ttls = []

    async def set(self, key, value, **kwargs):
        self.ttls.append(kwargs['ex'])
        if key in self.values:
            return False
        self.values[key] = value
        return True

    async def getdel(self, key):
        return self.values.pop(key, None)

    async def eval(self, *args):
        self.rate += 1
        return self.rate


@pytest.fixture
def secured_ws(monkeypatch):
    redis = TicketRedis()
    monkeypatch.setattr(auth, 'get_redis', AsyncMock(return_value=redis))
    monkeypatch.setattr(settings, 'app_env', 'production')
    monkeypatch.setattr(settings, 'cors_allowed_origins', 'https://banshengcomp.com,https://www.banshengcomp.com')
    monkeypatch.setattr(deps, 'is_redis_healthy', lambda: True)
    monkeypatch.setattr(ws, 'is_redis_healthy', lambda: True)
    agent = SimpleNamespace(id='agent', userId='owner', name='Synthetic', status='active')
    conv = SimpleNamespace(id='conv', userId='owner', agent=agent, isDeleted=False, workspaceId='workspace')
    db = MagicMock()
    db.conversation.find_unique = AsyncMock(return_value=conv)
    db.user.find_unique = AsyncMock(return_value=SimpleNamespace(username='Synthetic'))
    monkeypatch.setattr(chat, 'db', db)
    monkeypatch.setattr(ws, 'db', db)
    manager = MagicMock()
    manager.connect = AsyncMock()
    manager.disconnect = AsyncMock()
    monkeypatch.setattr(ws, 'manager', manager)
    monkeypatch.setattr(ws, 'record_ws_online', AsyncMock())
    monkeypatch.setattr(ws, 'remove_ws_online', AsyncMock())
    monkeypatch.setattr(ws, 'send_first_greeting', AsyncMock())
    monkeypatch.setattr(ws, 'fire_background', lambda coroutine: coroutine.close())
    monkeypatch.setattr(ws, '_handle_message', AsyncMock())
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(ws.router)
    return SimpleNamespace(client=TestClient(app), db=db, agent=agent, conv=conv, redis=redis, manager=manager)


def mint(s, user='owner', role='user'):
    r = s.client.post('/chat/conv/ws-ticket', headers={'Authorization': 'Bearer '+create_jwt(user, role)})
    assert r.status_code == 200, r.text
    assert r.headers['cache-control'] == 'no-store'
    return r.json()['ticket']


def protocols(ticket):
    return [auth.CHAT_PROTOCOL, auth.TICKET_PROTOCOL_PREFIX+ticket]


def denied(s, offered=None, code=4401, headers=None, path='/ws/conv'):
    with pytest.raises(WebSocketDisconnect) as error:
        with s.client.websocket_connect(path, subprotocols=offered or [], headers=headers or {}):
            pass
    assert error.value.code == code
    s.manager.connect.assert_not_called()
    ws.record_ws_online.assert_not_called()
    ws.send_first_greeting.assert_not_called()
    ws._handle_message.assert_not_called()


@pytest.mark.parametrize('token', [None, 'bad', 'expired'])
def test_ticket_auth_before_database(secured_ws, token):
    value = create_jwt('owner', 'user', expiry_hours=-1) if token == 'expired' else token
    r = secured_ws.client.post('/chat/conv/ws-ticket', headers={} if value is None else {'Authorization': 'Bearer '+value})
    assert r.status_code == 401
    secured_ws.db.conversation.find_unique.assert_not_called()
    assert not secured_ws.redis.values


@pytest.mark.parametrize('case,status', [('foreign',403),('missing',404),('deleted',410),('agent_missing',404),('pair',403),('initializing',503)])
def test_ticket_resource_guards(secured_ws, case, status):
    s=secured_ws
    user='other' if case=='foreign' else 'owner'
    if case=='missing': s.db.conversation.find_unique.return_value=None
    if case=='deleted': s.conv.isDeleted=True
    if case=='agent_missing': s.conv.agent=None
    if case=='pair': s.agent.userId='other'
    if case=='initializing': s.agent.status='provisioning'
    r=s.client.post('/chat/conv/ws-ticket',headers={'Authorization':'Bearer '+create_jwt(user,'user')})
    assert r.status_code==status
    assert not s.redis.values
    ws._handle_message.assert_not_called()


@pytest.mark.parametrize('role', ['user','admin'])
@pytest.mark.parametrize('origin', [None,'https://banshengcomp.com','https://www.banshengcomp.com','https://servicewechat.com'])
def test_ticket_connect_and_message_preserve_protocol(secured_ws, role, origin):
    s=secured_ws
    ticket=mint(s, 'admin' if role=='admin' else 'owner',role)
    with s.client.websocket_connect('/ws/conv?client=flutter',subprotocols=protocols(ticket),headers={'Origin':origin} if origin else {}) as socket:
        assert socket.accepted_subprotocol==auth.CHAT_PROTOCOL
        socket.send_json({'type':'ping'})
        assert socket.receive_json()=={'type':'pong'}
        socket.send_json({'type':'message','data':{'message':'hello','client_id':'synthetic'}})
        socket.send_json({'type':'ping'})
        assert socket.receive_json()=={'type':'pong'}
    assert not s.redis.values
    ws._handle_message.assert_awaited_once()
    assert ws._handle_message.await_args.kwargs['client_supports_voice'] is True
    s.manager.disconnect.assert_awaited_once()
    assert s.manager.disconnect.await_args.kwargs['expected'] is not None


@pytest.mark.parametrize('value', ['false', 'true', 0, 1, None])
def test_invalid_payment_confirmation_does_not_disconnect_or_charge(secured_ws, value):
    s=secured_ws
    with s.client.websocket_connect('/ws/conv',subprotocols=protocols(mint(s))) as socket:
        socket.send_json({'type':'message','data':{'message':'synthetic','paid_confirmed':value}})
        assert socket.receive_json()['data']['code']=='invalid_request'
        socket.send_json({'type':'ping'})
        assert socket.receive_json()=={'type':'pong'}
    ws._handle_message.assert_not_called()


@pytest.mark.parametrize('role', ['user','admin'])
def test_sql_message_frame_uses_authenticated_actor_and_original_input(secured_ws, monkeypatch, role):
    from app.services.runtime import chat_entrypoint
    s=secured_ws
    monkeypatch.setattr(ws,'sql_chat_ingress_enabled',lambda:True)
    handler=AsyncMock()
    monkeypatch.setattr(chat_entrypoint,'handle_sql_chat_frame',handler)
    actor='admin' if role=='admin' else 'owner'
    body={'message':'  synthetic  ','client_id':'stable','paid_confirmed':False,
          'user_id':'forged','backend':'redis'}
    with s.client.websocket_connect('/ws/conv?client=flutter',subprotocols=protocols(mint(s,actor,role))) as socket:
        socket.send_json({'type':'message','data':body})
        socket.send_json({'type':'ping'})
        assert socket.receive_json()=={'type':'pong'}
    handler.assert_awaited_once()
    assert handler.await_args.args[1:]==(actor,'conv',body)
    assert handler.await_args.kwargs['client_supports_voice'] is True
    ws._handle_message.assert_not_called()


@pytest.mark.parametrize("offered", [[], ["unrelated.protocol"]])
def test_anonymous_connections_rejected_before_business_effects(secured_ws, offered):
    denied(secured_ws, offered)
    secured_ws.db.conversation.find_unique.assert_not_called()



@pytest.mark.parametrize('value', ['bad', 'a'*43])
def test_invalid_supplied_ticket_never_downgrades(secured_ws, value):
    denied(secured_ws,protocols(value))
    secured_ws.db.conversation.find_unique.assert_not_called()


@pytest.mark.parametrize('origin',['https://evil.example','null','https://banshengcomp.com.evil.example'])
def test_browser_origin_rejected_before_ticket_consumption(secured_ws, origin):
    ticket=mint(secured_ws)
    denied(secured_ws,protocols(ticket),headers={'Origin':origin})
    assert auth._key(ticket) in secured_ws.redis.values


def test_ticket_replay_rejected(secured_ws):
    ticket=mint(secured_ws)
    with secured_ws.client.websocket_connect('/ws/conv',subprotocols=protocols(ticket)) as socket:
        socket.send_json({'type':'ping'}); assert socket.receive_json()['type']=='pong'
    secured_ws.manager.connect.reset_mock();ws.record_ws_online.reset_mock();ws.send_first_greeting.reset_mock()
    denied(secured_ws,protocols(ticket))


def test_ticket_bound_to_conversation(secured_ws):
    denied(secured_ws,protocols(mint(secured_ws)),path='/ws/other')


@pytest.mark.parametrize('case,code',[('new_owner',4403),('pair',4403),('deleted',4004),('initializing',1013)])
def test_access_rechecked_after_ticket_issued(secured_ws, case, code):
    ticket=mint(secured_ws)
    if case=='new_owner': secured_ws.conv.userId='other';secured_ws.agent.userId='other'
    if case=='pair': secured_ws.agent.userId='other'
    if case=='deleted': secured_ws.conv.isDeleted=True
    if case=='initializing': secured_ws.agent.status='provisioning'
    denied(secured_ws,protocols(ticket),code)


@pytest.mark.parametrize('reason',['expired_ticket','expired_jwt','malformed'])
def test_expired_or_corrupt_redis_payload(secured_ws, reason):
    ticket=mint(secured_ws);key=auth._key(ticket)
    value=json.loads(secured_ws.redis.values[key])
    if reason=='expired_ticket': value['ticket_exp']=time.time()-1
    if reason=='expired_jwt': value['exp']=time.time()-1
    if reason=='malformed': value['role']=[]
    secured_ws.redis.values[key]=json.dumps(value)
    denied(secured_ws,protocols(ticket))


def test_jwt_expiry_on_existing_socket_prevents_next_message(secured_ws, monkeypatch):
    clock=[time.time()]
    monkeypatch.setattr(auth.time,'time',lambda:clock[0])
    ticket=mint(secured_ws)
    expiry=json.loads(secured_ws.redis.values[auth._key(ticket)])['exp']
    with secured_ws.client.websocket_connect('/ws/conv',subprotocols=protocols(ticket)) as socket:
        socket.send_json({'type':'ping'}); assert socket.receive_json()['type']=='pong'
        clock[0]=expiry+1
        socket.send_json({'type':'message','data':{'message':'expired'}})
        with pytest.raises(WebSocketDisconnect) as error: socket.receive_json()
        assert error.value.code==4401
    ws._handle_message.assert_not_called()


def test_ticket_rate_limit(secured_ws):
    secured_ws.redis.rate=60
    r=secured_ws.client.post('/chat/conv/ws-ticket',headers={'Authorization':'Bearer '+create_jwt('owner','user')})
    assert r.status_code==429 and r.headers['retry-after']=='60'
    assert not secured_ws.redis.values


@pytest.mark.parametrize('operation',['issue','consume'])
def test_redis_failure_closed_without_secrets(secured_ws, monkeypatch, operation):
    ticket=mint(secured_ws) if operation=='consume' else None
    monkeypatch.setattr(auth,'get_redis',AsyncMock(side_effect=RuntimeError('synthetic-secret')))
    if operation=='consume': denied(secured_ws,protocols(ticket),1011)
    else:
        r=secured_ws.client.post('/chat/conv/ws-ticket',headers={'Authorization':'Bearer '+create_jwt('owner','user')})
        assert r.status_code==503 and 'synthetic-secret' not in r.text


@pytest.mark.parametrize('exp',[None,True,float('inf'),0])
async def test_ticket_requires_bounded_expiry(secured_ws,exp):
    with pytest.raises(auth.TicketInvalid): await auth.issue_ticket('conv',{'sub':'owner','role':'user','exp':exp})
    assert not secured_ws.redis.values


def test_development_origin_defaults_match_cors(secured_ws,monkeypatch):
    monkeypatch.setattr(settings,'app_env','development');monkeypatch.setattr(settings,'cors_allowed_origins','')
    assert auth.origin_allowed('http://localhost:5173')
    monkeypatch.setattr(settings,'app_env','production')
    assert not auth.origin_allowed('http://localhost:5173')


@pytest.mark.parametrize('offered',[[auth.CHAT_PROTOCOL],[auth.TICKET_PROTOCOL_PREFIX+'a'*43],protocols('a'*43)+[auth.TICKET_PROTOCOL_PREFIX+'b'*43]])
def test_incomplete_or_multiple_credentials_rejected(secured_ws,offered):
    denied(secured_ws,offered)


def test_jwt_expiry_while_loading_identity_prevents_connection_effects(secured_ws, monkeypatch):
    ticket = mint(secured_ws)
    exp = json.loads(secured_ws.redis.values[auth._key(ticket)])["exp"]
    clock = [time.time()]
    monkeypatch.setattr(auth.time, "time", lambda: clock[0])

    async def identity(**_):
        clock[0] = exp + 1
        return SimpleNamespace(username="Synthetic")

    secured_ws.db.user.find_unique.side_effect = identity
    denied(secured_ws, protocols(ticket))


def test_authentication_mode_logs_contain_no_credentials(secured_ws, caplog):
    ticket = mint(secured_ws)
    with caplog.at_level("INFO", logger=ws.__name__):
        with secured_ws.client.websocket_connect("/ws/conv", subprotocols=protocols(ticket)) as socket:
            socket.send_json({"type": "ping"})
            assert socket.receive_json()["type"] == "pong"
    messages = [record.getMessage() for record in caplog.records]
    assert "ws authentication mode=ticket" in messages
    assert "ws authentication mode=legacy" not in messages
    assert ticket not in caplog.text and "Bearer" not in caplog.text


def test_authentication_cannot_be_disabled_by_environment(monkeypatch):
    from app.config import Settings
    monkeypatch.setenv("WS_AUTH_REQUIRED", "false")
    assert not hasattr(Settings(), "ws_auth_required")


def test_wechat_origin_is_exact_and_does_not_expand_http_cors(secured_ws):
    assert auth.origin_allowed("https://servicewechat.com")
    assert "https://servicewechat.com" not in settings.cors_origins()
    assert not auth.origin_allowed("https://servicewechat.com.evil.example")
    assert not auth.origin_allowed("http://servicewechat.com")


def test_idle_socket_closes_at_jwt_deadline(secured_ws, monkeypatch):
    consume = ws.consume_ticket

    async def short_deadline(ticket, conversation_id):
        principal = await consume(ticket, conversation_id)
        return auth.SocketPrincipal(principal.user_id, principal.role, time.time() + 0.1)

    monkeypatch.setattr(ws, "consume_ticket", short_deadline)
    ticket = mint(secured_ws)
    with secured_ws.client.websocket_connect("/ws/conv", subprotocols=protocols(ticket)) as socket:
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()
        assert error.value.code == 4401
    ws._handle_message.assert_not_called()
    secured_ws.manager.disconnect.assert_awaited_once()


def test_delivery_frames_use_ticket_actor_not_client_scope(secured_ws,monkeypatch):
    from app.services.runtime import outbox_realtime
    callback=AsyncMock()
    monkeypatch.setattr(outbox_realtime,'handle_delivery_frame',callback)
    ticket=mint(secured_ws)
    with secured_ws.client.websocket_connect('/ws/conv',subprotocols=protocols(ticket)) as socket:
        socket.send_json({'type':'delivery_resume','data':{'actor_user_id':'forged','conversation_id':'foreign'}})
        socket.send_json({'type':'delivery_ack','data':{'event_id':'event','delivery_token':'1'}})
        socket.send_json({'type':'ping'})
        assert socket.receive_json()=={'type':'pong'}
    assert [c.args[1:3] for c in callback.await_args_list]==[('owner','conv'),('owner','conv')]
    ws._handle_message.assert_not_called()
