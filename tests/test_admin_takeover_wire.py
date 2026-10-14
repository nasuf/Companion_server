"""Real TCP HTTP/WebSocket across two isolated app processes and real Redis.

Uses an owned migrated loopback SQL fixture; no production account, provider,
cron or app.main startup. Captures child diagnostics and removes owned processes.
"""
import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.parse import urlsplit

import httpx
import pytest
from websockets.asyncio.client import connect

from app.config import settings
from app.services.auth import create_jwt
from app.services.runtime.ws_auth import CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX
from tests.test_admin_takeover_postgres import human, attach
from tests.test_runtime_execution_foundation import flow
from tests.test_chat_ingress_effects_postgres import attachment


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


async def event(peer,kind):
    async with asyncio.timeout(10):
        while True:
            frame=json.loads(await peer.recv())
            if frame.get('type')==kind:return frame.get('data',{})


async def test_claim_user_ack_admin_reply_reconnect_cross_process(human,tmp_path):
    h=human;await attach(h)
    processes=[];logs=[];urls=[]
    env={k:v for k,v in os.environ.items() if not any(x in k for x in ('TOKEN','API_KEY','SECRET','PASSWORD','PROXY'))}
    env.update(DATABASE_URL=h.flow.url,DIRECT_DATABASE_URL=h.flow.url,REDIS_URL=os.environ['PROACTIVE_E2E_REDIS_URL'],
        JWT_SECRET=settings.jwt_secret,APP_ENV='test',PYTHON_DOTENV_DISABLED='1',TRACE_BACKEND='off',
        ONLINE_MODEL='false',CHAT_INGRESS_BACKEND='redis')
    assert urlsplit(env['DATABASE_URL']).hostname in {'127.0.0.1','localhost'}
    assert urlsplit(env['REDIS_URL']).hostname in {'127.0.0.1','localhost'}
    try:
        for n in range(2):
            chosen=port();urls.append(f'http://127.0.0.1:{chosen}')
            log=(tmp_path/f'worker-{n}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-m','uvicorn','tests.fixtures.admin_takeover_wire_app:app',
                '--host','127.0.0.1','--port',str(chosen),'--log-level','warning'],env=env,stdout=log,stderr=log))
        async with httpx.AsyncClient(timeout=10,trust_env=False) as client:
            for target in urls:
                for _ in range(300):
                    try:
                        if (await client.get(target+'/health')).status_code==200:break
                    except httpx.RequestError:pass
                    if any(p.poll() is not None for p in processes):raise AssertionError('wire worker exited')
                    await asyncio.sleep(.1)
                else:raise AssertionError('wire worker startup timeout')
            admin={'Authorization':'Bearer '+create_jwt(h.admins[0],'admin')}
            user={'Authorization':'Bearer '+create_jwt(h.flow.ids['owner'],'user')}
            other={'Authorization':'Bearer '+create_jwt(h.admins[1],'admin')}
            base='/admin-api/takeovers/'+h.case
            claimed=await client.post(urls[1]+base+'/claim',headers=admin)
            assert claimed.status_code==200,claimed.text
            assert (await client.post(urls[0]+base+'/claim',headers=other)).status_code==409
            inbox=(await client.post(urls[0]+'/admin-api/takeovers/inbox-ticket',headers=admin)).json()['ticket']
            conv=h.flow.ids['conversation']
            ticket=(await client.post(urls[0]+f'/chat/{conv}/ws-ticket',headers=user)).json()['ticket']
            async with AsyncExitStack() as stack:
                operator=await stack.enter_async_context(connect(urls[0].replace('http','ws')+'/ws/admin-inbox/'+h.admins[0],
                    subprotocols=[CHAT_PROTOCOL,TICKET_PROTOCOL_PREFIX+inbox],proxy=None))
                peer=await stack.enter_async_context(connect(urls[0].replace('http','ws')+f'/ws/{conv}?client=flutter',
                    subprotocols=[CHAT_PROTOCOL,TICKET_PROTOCOL_PREFIX+ticket],proxy=None))
                await operator.send(json.dumps({'type':'ping'}));assert await event(operator,'pong')=={}
                # Presence crosses workers without messages, quota or model work.
                typing_path=f'/conversations/{conv}/typing'
                last_user_revision=last_admin_revision=0
                for active,revision in ((True,1),(False,2)):
                    response=await client.post(urls[1]+typing_path,headers=user,
                        json={'active':active,'revision':revision,'session_id':'wire-user'})
                    assert response.status_code==200,response.text
                    presence=await event(operator,'takeover_user_typing')
                    assert presence=={'takeover_id':h.case,'conversation_id':conv,
                        'active':active,'revision':presence['revision']}
                    assert presence['revision'] > max(last_user_revision,2_147_483_647)
                    last_user_revision=presence['revision']
                    response=await client.post(urls[1]+base+'/typing',headers=admin,
                        json={'active':active,'revision':revision,'session_id':'wire-admin'})
                    assert response.status_code==200,response.text
                    presence=await event(peer,'takeover_typing')
                    assert presence=={'active':active,'revision':presence['revision']}
                    assert presence['revision'] > max(last_admin_revision,2_147_483_647)
                    last_admin_revision=presence['revision']
                outsider={'Authorization':'Bearer '+create_jwt(h.admins[1],'user')}
                assert (await client.post(urls[1]+typing_path,headers=outsider,
                    json={'active':True,'revision':1,'session_id':'foreign'})).status_code==200
                await operator.send(json.dumps({'type':'ping'}))
                # The next inbox frame must be pong; another user's presence was not published.
                assert json.loads(await operator.recv())['type']=='pong'
                assert (await client.post(urls[1]+base+'/typing',headers=other,
                    json={'active':True,'revision':1,'session_id':'foreign'})).status_code==403
                assert await h.flow.db.message.count(where={'conversationId':conv})==0
                assert await h.flow.db.agentrun.count(where={'conversationId':conv})==0
                long_text='wire user message '*30
                await peer.send(json.dumps({'type':'message','data':{'message':long_text,'client_id':'wire-1'}}))
                ack=await event(peer,'ack');assert ack['message_id']
                received=await event(operator,'takeover_user_message');assert received['message_id']==ack['message_id']
                assert received['text']==long_text.strip() and received['metadata']['takeover_sequence']==1
                summary='/admin-api/takeovers/unread-summary'
                assert (await client.get(urls[1]+summary,headers=admin)).json()['unread_count']==1
                history=(await client.get(urls[1]+f'/conversations/{conv}/messages?include_metadata=true',headers=admin)).json()
                assert history[0]['content']==long_text.strip() and history[0]['metadata']['takeover_sequence']==1
                assert (await client.post(urls[1]+base+'/read',headers=admin,json={'message_id':ack['message_id']})).status_code==200
                assert (await client.get(urls[0]+summary,headers=admin)).json()['unread_count']==0
                # Same frame retry has no duplicate quota or inbox event.
                await peer.send(json.dumps({'type':'message','data':{'message':long_text,'client_id':'wire-1'}}))
                assert (await event(peer,'ack'))['message_id']==ack['message_id']
                reply={'text':'wire human reply','client_id':'reply-1'}
                response=await client.post(urls[1]+base+'/messages',headers=admin,json=reply)
                assert response.status_code==200,response.text
                delivered=await event(peer,'reply');assert delivered['text']=='wire human reply'
                assert delivered['assistant_message_id']==response.json()['message_id']
                assert (await event(peer,'done'))['message_id']==response.json()['message_id']
                assert (await event(operator,'takeover_admin_message'))['message_id']==response.json()['message_id']
                # Delivery is not a read receipt; only the actual user may acknowledge visibility.
                message_id=response.json()['message_id']
                history=(await client.get(urls[1]+f'/conversations/{conv}/messages?include_metadata=true',headers=admin)).json()
                assert next(m for m in history if m['id']==message_id)['read'] is False
                read_path=f'/conversations/{conv}/read'
                assert (await client.post(urls[1]+read_path,headers=outsider,json={'message_ids':[message_id]})).status_code==200
                assert (await h.flow.db.message.find_unique(where={'id':message_id})).metadata.get('takeover_user_read_at') is None
                assert (await client.post(urls[1]+read_path,headers=user,json={'message_ids':[message_id]})).status_code==200
                receipt=await event(operator,'takeover_user_read')
                assert receipt['message_ids']==[message_id] and receipt['client_ids']==['reply-1']
                history=(await client.get(urls[1]+f'/conversations/{conv}/messages?include_metadata=true',headers=admin)).json()
                assert next(m for m in history if m['id']==message_id)['read'] is True

                again=await client.post(urls[0]+base+'/messages',headers=admin,json=reply)
                assert again.json()['message_id']==response.json()['message_id'] and again.json()['duplicate']
                audio_id=await attachment(h.flow)
                await h.flow.db.execute_raw("UPDATE chat_message_attachments SET kind='audio',mime='audio/mp4', "
                    "duration_seconds=3,url='/chat/media/wire.m4a' WHERE id=$1",audio_id)
                voice={'text':'［语音］','client_id':'voice-1','attachments':[{'id':audio_id}]}
                response=await client.post(urls[1]+base+'/messages',headers=admin,json=voice)
                assert response.status_code==200,response.text
                delivered=await event(peer,'reply')
                assert delivered['human_takeover'] and delivered['attachments'][0]['id']==audio_id
                assert delivered['attachments'][0]['duration_seconds']==3
                await event(peer,'done');await event(operator,'takeover_admin_message')
                history=(await client.get(urls[0]+f'/conversations/{conv}/messages?include_metadata=true',headers=user)).json()
                saved=next(m for m in history if m['id']==response.json()['message_id'])
                assert saved['metadata']['attachments']==delivered['attachments'] and saved['read'] is False
            # Reconnect with a freshly minted single-use credential.
            ticket=(await client.post(urls[1]+f'/chat/{conv}/ws-ticket',headers=user)).json()['ticket']
            async with connect(urls[1].replace('http','ws')+f'/ws/{conv}',subprotocols=[CHAT_PROTOCOL,TICKET_PROTOCOL_PREFIX+ticket],proxy=None) as peer:
                await peer.send(json.dumps({'type':'ping'}));assert await event(peer,'pong')=={}
            rows=await h.flow.db.query_raw('SELECT role,content FROM messages WHERE conversation_id=$1 ORDER BY created_at',conv)
            assert rows==[{'role':'user','content':long_text.strip()},{'role':'assistant','content':'wire human reply'},
                {'role':'assistant','content':'［语音］'}]
            assert (await h.flow.db.query_raw('SELECT used FROM user_message_quota WHERE user_id=$1',h.flow.ids['owner']))==[{'used':1}]
            assert await h.flow.db.agentrun.count(where={'conversationId':conv})==0
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        for log in logs:log.close()
        failures=[p for p in processes if p.returncode not in {0,-15}]
        if failures:
            for p in tmp_path.glob('worker-*.log'):print(p.read_text()[-3000:])
