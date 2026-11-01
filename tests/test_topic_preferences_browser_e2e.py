"""Browser -> authenticated HTTP -> real PostgreSQL topic preferences.

The separate Web gate covers the production React component and scrolling.
No provider, production database or external browser origin is used here.
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

from app.api.admin.memory_repairs import router
from app.services.auth import create_jwt
from tests.test_memory_evidence_postgres import origins
from tests.test_topic_preferences_postgres import topics


async def test_browser_topic_preferences_permissions_cas_and_live_erasure(topics):
    db, uid, wid, original, send, fixture = topics
    source = await send("以后别问合成面试了")
    app = FastAPI()
    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    async def driver():
        return "<!doctype html><html><body><output></output></body></html>"

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.setblocking(False)
    origin = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    artifacts = Path(os.environ.get("BROWSER_ARTIFACT_DIR", "reports/browser"))
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.02)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                context = await browser.new_context()
                violations = []

                async def fence(route):
                    if route.request.url.startswith(origin + "/"):
                        await route.continue_()
                    else:
                        violations.append("non-fixture request")
                        await route.abort()

                await context.route("**/*", fence)
                await context.tracing.start(screenshots=True, snapshots=True)
                page = await context.new_page()
                try:
                    await page.goto(origin)
                    path = "/admin-api/memory-repairs/topics/page"
                    query = {"user_id": uid, "workspace_id": wid, "limit": "20"}

                    async def request(token, params, *, correction=None):
                        return await page.evaluate(
                            """async ([token,path,params,correction]) => {
                              const url=correction ? '/admin-api/memory-repairs/topics/'+correction.id+'/correction' : path+'?'+new URLSearchParams(params);
                              const r=await fetch(url,{method:correction?'POST':'GET',
                                headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},
                                ...(correction?{body:JSON.stringify(correction.body)}:{})});
                              const result={status:r.status,data:await r.json()};
                              document.querySelector('output').textContent=JSON.stringify(result);
                              return result;
                            }""",
                            [token, path, params, correction],
                        )

                    assert (await request("", query))["status"] == 401
                    assert (await request(create_jwt(uid, "user"), query))[
                        "status"
                    ] == 403
                    from uuid import uuid4

                    admin = await db.user.create(
                        data={
                            "username": "topic-browser-" + uuid4().hex,
                            "role": "admin",
                        }
                    )
                    fixture[7].append(admin.id)
                    token = create_jwt(admin.id, "admin")
                    result = await request(token, query)
                    assert result["status"] == 200 and len(result["data"]["items"]) == 1
                    item = result["data"]["items"][0]
                    assert (
                        item["effective"] == "active"
                        and item["source_message_id"] == source.id
                    )
                    assert (
                        await request(token, {**query, "workspace_id": fixture[9][-1]})
                    )["status"] == 404
                    correction = {
                        "id": item["id"],
                        "body": {
                            "user_id": uid,
                            "workspace_id": wid,
                            "expected_revision": item["revision"],
                            "action": "resume",
                            "reason": "已核对合成来源并恢复此话题",
                        },
                    }
                    assert (await request(token, query, correction=correction))[
                        "status"
                    ] == 200
                    assert (await request(token, query, correction=correction))[
                        "status"
                    ] == 409
                    updated = (await request(token, query))["data"]["items"][0]
                    assert (
                        updated["effective"] == "reopened" and updated["revision"] == 2
                    )
                    assert updated["history"][0]["source_type"] == "admin"
                    assert updated["history"][0]["actor_name"]
                    await db.message.delete(where={"id": source.id})
                    assert not (await request(token, query))["data"]["items"]
                    assert not violations
                finally:
                    await context.tracing.stop(
                        path=str(artifacts / "topic-preferences-real-api-trace.zip")
                    )
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=10)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        sock.close()
