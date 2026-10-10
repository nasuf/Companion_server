"""Chromium → real HTTP/JWT/admin route → Prisma/PostgreSQL, synthetic data only.

The production React components are separately exercised in the Web repository.
"""
import asyncio
from contextlib import suppress
import os
from pathlib import Path
import socket

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from playwright.async_api import async_playwright
import pytest
import uvicorn

from app.api.admin import memory_repairs
from app.services.auth import create_jwt
from app.services.memory.evidence import EvidenceSource
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_profile_evidence_postgres import persona
from app.services.memory.profile_evidence import prepare_profile_origin


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", ["message", "profile"])
async def test_browser_readonly_evidence_permissions_and_source_deletion(origins,persona,source_kind):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    app=FastAPI()
    app.include_router(memory_repairs.router)

    @app.get("/", response_class=HTMLResponse)
    async def driver():
        return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body>
        <button>查看来源</button><output></output><script>
        window.read=async(token,path)=>{const response=await fetch(path,{headers:{Authorization:'Bearer '+token}});
          return {status:response.status,data:await response.json()};};
        document.querySelector('button').onclick=async()=>{
          document.querySelector('output').textContent=JSON.stringify(await window.read(window.token,window.path));
        };</script></body></html>"""

    sock=socket.socket();sock.bind(("127.0.0.1",0));sock.listen(128);sock.setblocking(False)
    origin=f"http://127.0.0.1:{sock.getsockname()[1]}"
    server=uvicorn.Server(uvicorn.Config(app,log_level="warning",lifespan="off"))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    artifacts=Path(os.environ.get("BROWSER_ARTIFACT_DIR","reports/browser"))
    artifacts.mkdir(parents=True,exist_ok=True)
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if task.done():await task
                await asyncio.sleep(.02)
        async with async_playwright() as playwright:
            browser=await playwright.chromium.launch()
            try:
                context=await browser.new_context()
                violations=[]
                async def fence(route):
                    if route.request.url.startswith(origin+"/"):await route.continue_()
                    else:
                        violations.append("non-fixture request");await route.abort()
                await context.route("**/*",fence)
                await context.tracing.start(screenshots=True,snapshots=True)
                page=await context.new_page()
                errors=[];page.on("pageerror",lambda error:errors.append(str(error)))
                try:
                    await page.goto(origin)
                    side="ai" if source_kind=="profile" else "user"
                    path=f"/admin-api/memory-repairs/evidence/{side}/{mid}?user_id={uid}&workspace_id={wid}&limit=20"
                    admin=create_jwt("synthetic-evidence-browser-admin",role="admin")
                    async def request(token,selected_path=None):
                        return await page.evaluate("args=>window.read(...args)",[token,selected_path or path])
                    assert (await request(""))["status"]==401
                    assert (await request(create_jwt(uid,role="user")))["status"]==403
                    assert (await request(admin))["data"]["state"]=="historical_unknown"
                    expected_ref=msg.id
                    if source_kind=="profile":
                        _,store,inspect=persona
                        ids=await store(origin=prepare_profile_origin({"private":"Synthetic hidden profile"},None),force=True)
                        mid=ids[0]
                        path=f"/admin-api/memory-repairs/evidence/ai/{mid}?user_id={uid}&workspace_id={wid}&limit=20"
                        expected_ref=(await inspect(mid))["items"][0]["source_ref"]
                    else:
                        await bind(EvidenceSource("message",msg.id))
                    await page.evaluate("args=>{window.token=args[0];window.path=args[1]}",[admin,path])
                    await page.get_by_role("button",name="查看来源").click()
                    await page.wait_for_function("document.querySelector('output').textContent.length>0")
                    linked=await request(admin)
                    assert linked["data"]["state"]=="linked" and linked["data"]["items"][0]["source_ref"]==expected_ref
                    if source_kind=="profile":
                        assert linked["data"]["items"][0]["profile"]["input_status"]=="uncollected"
                        assert "Synthetic hidden profile" not in str(linked)
                    assert msg.content not in str(linked)
                    assert (await request(admin,path.replace(wid,spaces[1])))["status"]==404
                    if source_kind=="profile":
                        await db.execute_raw("DELETE FROM memory_profile_origins WHERE id=$1",expected_ref)
                    else:
                        await db.message.delete(where={"id":msg.id})
                    deleted=(await request(admin))["data"]["items"][0]
                    assert deleted["availability"]=="deleted" and deleted["source_ref"] is None
                    assert not violations and not errors
                    assert await db.usermemory.count(where={"userId":uid})==1
                except BaseException:
                    await page.screenshot(path=str(artifacts/f"memory-evidence-{source_kind}-real-api-failure.png"),full_page=True)
                    raise
                finally:
                    await context.tracing.stop(path=str(artifacts/f"memory-evidence-{source_kind}-real-api-trace.zip"))
            finally:
                await browser.close()
    finally:
        server.should_exit=True
        try:await asyncio.wait_for(task,timeout=10)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):await task
        sock.close()
