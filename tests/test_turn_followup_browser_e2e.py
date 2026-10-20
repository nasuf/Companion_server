"""Chromium → real JWT/admin API → SQL retry receipt → Redis consumer.

Uses synthetic users only. React presentation is covered by the Web browser gate.
"""
import asyncio
from contextlib import suppress
import os
from pathlib import Path
import socket

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from playwright.async_api import async_playwright
import uvicorn

from app.api.admin.runtime_jobs import router
from app.services.auth import create_jwt
from app.services.runtime import turn_followups as t, job_queue as q
from tests.test_turn_followups_postgres import turn, followups, origins, redis, receipt


async def test_browser_turn_recovery_uses_real_permissions_and_reconciles_sql(turn, redis, monkeypatch):
    db, user, _, args = turn
    monkeypatch.setattr('app.db.db',db)
    owner=(await db.conversation.find_unique(where={'id':user.conversationId})).userId
    tid = args['turn_id']
    await t.register_completed_turn(**args)
    await t.dispatch_turn_followups(turn_id=tid)
    row = await receipt(db,tid)
    await redis.hset(q._job_key(row['job_id']),'status','dead_letter')
    await db.execute_raw('UPDATE turn_followup_events SET next_dispatch_at=NOW() WHERE turn_id=$1',tid)
    await t.dispatch_turn_followups(turn_id=tid)
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
                    path = '/admin-api/runtime-jobs/turn-followups'
                    action = path+'/'+tid+'/retry'
                    assert (await request('',path))['status'] == 401
                    ordinary = create_jwt(owner,role='user')
                    assert (await request(ordinary,action,'POST'))['status'] == 403
                    await db.user.update(where={'id':owner},data={'role':'admin'})
                    admin = create_jwt(owner,role='admin')
                    result = await request(admin,path+'?status=failed')
                    assert result['status'] == 200 and result['data']['items'][0]['turn_id'] == tid
                    assert user.content not in str(result)
                    assert (await request(admin,action,'POST'))['status'] == 200
                    assert (await request(admin,action,'POST'))['status'] == 409
                    await t.dispatch_turn_followups(turn_id=tid)
                    # Drain the stale ready entry from our injected terminal state,
                    # then the newly registered recovery generation.
                    await q.process_runtime_jobs(max_jobs=2)
                    result = await request(admin,path)
                    assert result['data']['items'][0]['status'] == 'completed'
                    assert await db.achievementunlock.count(where={'conversationId':user.conversationId,'achievementId':81}) == 1
                    assert not violations
                finally:
                    await context.tracing.stop(path=str(artifacts/'turn-followup-real-api-trace.zip'))
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        try: await asyncio.wait_for(task,timeout=10)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError): await task
        sock.close()
