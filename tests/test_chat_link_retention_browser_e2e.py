"""Browser -> authenticated real HTTP -> migrated PG -> historical preservation."""
import asyncio
from contextlib import suppress
import os
from pathlib import Path
import socket

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from playwright.async_api import async_playwright
import uvicorn

from app.api.admin import stats
from app.services.auth import create_jwt
from app.services.chat_links import retention as r
from tests.test_chat_link_retention_postgres import governed, links, origins, age


async def test_browser_admin_cleanup_preserves_real_message_history(governed):
    db,uid,convs,prepare,save,root=governed
    historical=await prepare('historical fixture');await age(db,historical);message=await save(historical)
    unused=await prepare('unused fixture');await age(db,unused)
    # Same cache, separate frozen versions. Preserve the message's first version.
    app=FastAPI();app.include_router(stats.router)
    @app.get('/',response_class=HTMLResponse)
    async def driver():return '<html><body><output></output></body></html>'
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128);sock.setblocking(False)
    origin=f'http://127.0.0.1:{sock.getsockname()[1]}'
    server=uvicorn.Server(uvicorn.Config(app,log_level='warning',lifespan='off'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    artifacts=Path(os.getenv('BROWSER_ARTIFACT_DIR','reports/browser'));artifacts.mkdir(parents=True,exist_ok=True)
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if task.done():await task
                await asyncio.sleep(.02)
        async with async_playwright() as p:
            browser=await p.chromium.launch()
            try:
                context=await browser.new_context();violations=[]
                async def fence(route):
                    if route.request.url.startswith(origin+'/'):await route.continue_()
                    else:violations.append(route.request.url);await route.abort()
                await context.route('**/*',fence);await context.tracing.start(screenshots=True,snapshots=True)
                page=await context.new_page();await page.goto(origin)
                async def request(path,method,role):
                    return await page.evaluate('''async([path,method,token])=>{
                      const r=await fetch(path,{method,headers:{Authorization:'Bearer '+token}});
                      const result={status:r.status,body:await r.json()};
                      document.querySelector('output').textContent=JSON.stringify(result);return result;
                    }''',[path,method,create_jwt(uid,role)])
                endpoint='/admin-api/stats/link-storage'
                assert (await request(endpoint,'GET','user'))['status']==403
                await db.user.update(where={'id':uid},data={'role':'admin'})
                assert (await request(endpoint+'/preview','POST','admin'))['body']['status']=='completed'
                assert await db.query_raw('SELECT id FROM chat_link_snapshots WHERE id=$1',unused.snapshot_id)
                response=await request(endpoint+'/execute?confirmed=true&policy_version='+r.POLICY_VERSION,'POST','admin')
                assert response['status']==200 and response['body']['deleted']['snapshots']==1
                assert response['body']['deleted']['previews']==0
                assert await db.query_raw('SELECT id FROM chat_link_snapshots WHERE id=$1',historical.snapshot_id)
                assert not await db.query_raw('SELECT id FROM chat_link_snapshots WHERE id=$1',unused.snapshot_id)
                assert (await db.message.find_unique(where={'id':message.id})).metadata['link_card']['title']=='historical fixture'
                response=await request(endpoint,'GET','admin')
                assert response['body']['last_run']['status']=='completed'
                assert violations==[]
                await context.tracing.stop(path=artifacts/'link-storage-api-trace.zip')
            finally:await browser.close()
    finally:
        server.should_exit=True
        with suppress(asyncio.TimeoutError):await asyncio.wait_for(task,10)
        if not task.done():task.cancel()
        sock.close()
