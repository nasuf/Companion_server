"""Browser -> authenticated HTTP -> real PostgreSQL temporal history.

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
from tests.test_memory_validity_postgres import validity


async def test_browser_validity_history_permissions_and_live_erasure(validity):
    fixture, source, _ = validity
    db, uid, wid, mid, _, bind, *_ = fixture
    await bind(source)
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
                    path = f"/admin-api/memory-repairs/time-history/user/{mid}"
                    query = {
                        "user_id": uid,
                        "workspace_id": wid,
                        "limit": "20",
                        "at": "2026-08-03T12:00:00+08:00",
                    }

                    async def request(token, params):
                        return await page.evaluate(
                            """async ([token,path,params]) => {
                          const r=await fetch(path+'?'+new URLSearchParams(params),
                            {headers:{Authorization:'Bearer '+token}});
                          const result={status:r.status,data:await r.json()};
                          document.querySelector('output').textContent=JSON.stringify(result);
                          return result;
                        }""",
                            [token, path, params],
                        )

                    assert (await request("", query))["status"] == 401
                    assert (await request(create_jwt(uid, "user"), query))[
                        "status"
                    ] == 403
                    await db.user.update(where={"id": uid}, data={"role": "admin"})
                    token = create_jwt(uid, "admin")
                    result = await request(token, query)
                    assert result["status"] == 200
                    assertion = result["data"]["items"][0]["assertions"][0]
                    assert assertion["membership"] == "inside"
                    assert assertion["validity"]["truth"] == "speaker_assertion"
                    assert assertion["validity"]["valid_to"] != assertion["recorded_at"]
                    assert not result["data"]["complete_history"]
                    assert (await request(token, {**query, "at": "2026-08-03"}))[
                        "status"
                    ] == 400
                    assert (
                        await request(token, {**query, "workspace_id": fixture[9][-1]})
                    )["status"] == 404
                    old = await request(
                        token, {**query, "known_at": "2026-01-01T00:00:00Z"}
                    )
                    assert old["status"] == 200 and not old["data"]["items"]
                    await db.message.delete(where={"id": source.ref})
                    erased = await request(token, query)
                    assert erased["status"] == 200
                    assert not any(
                        item["assertions"] for item in erased["data"]["items"]
                    )
                    assert not violations
                finally:
                    await context.tracing.stop(
                        path=str(artifacts / "memory-validity-real-api-trace.zip")
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
