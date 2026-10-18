"""Chromium → real JWT/admin API → dual-side observations in real Redis.

The memory source and progress use real PostgreSQL/Redis; only the external LLM
is controlled. React rendering is covered by the sibling Web browser gate.
"""
import asyncio
from contextlib import suppress
from datetime import timedelta
import os
from pathlib import Path
import socket
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from playwright.async_api import async_playwright
import uvicorn

from app.api.admin.runtime_jobs import router
from app.services.auth import create_jwt
from app.services.chat import post_process as p
from app.services.memory.recording import pipeline, watermark as wm
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_processing_diagnostics import observations, redis


async def test_browser_reads_real_dual_side_failure_and_success(origins, observations, monkeypatch):
    db,uid,wid,_,msg,*_=origins
    monkeypatch.setattr(p,'db',db)
    monkeypatch.setattr(wm,'get_redis',AsyncMock(return_value=observations))
    from app.config import settings
    monkeypatch.setattr(settings,'enable_memory_prefilter',False)
    async def extract(*args,side,**kwargs):
        return {'memories':[]} if side=='user' else {'_extraction_error':True,'_extraction_error_kind':'PRIVATE_PROVIDER'}
    monkeypatch.setattr(pipeline,'extract_memories',extract)
    assistant=await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant','content':'合成助手今天看见一只可爱的小猫咪'})
    baseline=msg.createdAt-timedelta(seconds=1)
    for side in ('user','ai'):await wm.set_watermark(msg.conversationId,side,baseline)
    await p._bg_memory_pipeline(uid,[],msg.conversationId,wid)
    app = FastAPI(); app.include_router(router)

    @app.get('/', response_class=HTMLResponse)
    async def driver():
        return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body><output></output></body></html>'

    sock = socket.socket(); sock.bind(('127.0.0.1',0)); sock.listen(128); sock.setblocking(False)
    origin = f'http://127.0.0.1:{sock.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(app,log_level='warning',lifespan='off'))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    artifacts = Path(os.environ.get('BROWSER_ARTIFACT_DIR','reports/browser')); artifacts.mkdir(parents=True,exist_ok=True)
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if task.done(): await task
                await asyncio.sleep(.02)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                context = await browser.new_context()
                violations = []
                async def fence(route):
                    if route.request.url.startswith(origin+'/'): await route.continue_()
                    else: violations.append('non-fixture request'); await route.abort()
                await context.route('**/*',fence)
                await context.tracing.start(screenshots=True,snapshots=True)
                page = await context.new_page()
                try:
                    await page.goto(origin)
                    async def request(token, path, method='GET'):
                        return await page.evaluate('''async ([token,path,method]) => {
                          const r=await fetch(path,{method,headers:{Authorization:'Bearer '+token}});
                          const value={status:r.status,data:await r.json()};
                          document.querySelector('output').textContent=JSON.stringify(value);return value;
                        }''',[token,path,method])
                    path = '/admin-api/runtime-jobs/memory-processing'
                    assert (await request('',path))['status'] == 401
                    ordinary = create_jwt('synthetic-user',role='user')
                    assert (await request(ordinary,path))['status'] == 403
                    admin = create_jwt('synthetic-admin',role='admin')
                    assert (await request(admin,path+'?limit=51'))['status'] == 422
                    result = await request(admin,path)
                    assert result['status'] == 200
                    row=result['data']['records'][0]
                    assert row['status']=='incomplete' and row['conversation_id']==msg.conversationId
                    user,ai=row['sides']
                    assert user['status']=='completed' and user['memory_count']==0 and user['progress']=='advanced'
                    assert ai['status']=='failed' and ai['memory_count'] is None and ai['progress']=='unchanged'
                    assert ai['reason']=='extraction_failed'
                    assert msg.content not in str(result) and 'PRIVATE_PROVIDER' not in str(result)
                    assert await wm.get_watermark(msg.conversationId,'user') == msg.createdAt
                    assert await wm.get_watermark(msg.conversationId,'ai') == baseline
                    assert result['data']['replay_supported'] is False
                    assert (await request(admin,path+'/missing/retry','POST'))['status']==404
                    assert not violations
                finally:
                    await context.tracing.stop(path=str(artifacts/'memory-processing-real-api-trace.zip'))
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        try: await asyncio.wait_for(task,timeout=10)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError): await task
        sock.close()
        await observations.delete(*(wm._key(msg.conversationId,side) for side in ("user","ai")))
