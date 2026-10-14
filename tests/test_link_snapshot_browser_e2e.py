"""Actual Chromium HTTP/JWT/message history against owned PostgreSQL rows."""

import asyncio
from contextlib import suppress
import os
from pathlib import Path
import socket

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from playwright.async_api import async_playwright
import uvicorn

from app.api import ownership
from app.api.public import conversations, daily_share
from app.services.auth import create_jwt
from tests.test_chat_link_snapshots_postgres import links, origins


async def test_browser_message_snapshots_and_share_history(links, monkeypatch):
    db, uid, users, convs, prepare, save = links
    monkeypatch.setattr(ownership, "db", db)
    monkeypatch.setattr(conversations, "db", db)
    first = await prepare("first content")
    m1 = await save(first)
    second = await prepare("updated content")
    m2 = await save(second)
    app = FastAPI()
    app.include_router(conversations.router)
    app.include_router(daily_share.router)

    @app.get("/", response_class=HTMLResponse)
    async def driver():
        return '<!doctype html><html><meta charset="utf-8"><body><output></output></body></html>'

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.setblocking(False)
    origin = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    artifacts = Path(os.getenv("BROWSER_ARTIFACT_DIR", "reports/browser"))
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
                        violations.append(route.request.url)
                        await route.abort()

                await context.route("**/*", fence)
                await context.tracing.start(screenshots=True, snapshots=True)
                page = await context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                try:
                    await page.goto(origin)

                    async def get(path, token):
                        return await page.evaluate(
                            """async([path,token])=>{
                            const r=await fetch(path,{headers:{Authorization:'Bearer '+token}});
                            const result={status:r.status,data:await r.json()};
                            document.querySelector('output').textContent=JSON.stringify(result);
                            return result;}""",
                            [path, token],
                        )

                    path = f"/conversations/{convs[0]}/messages?limit=10"
                    assert (await get(path, ""))["status"] == 401
                    assert (await get(path, create_jwt(users[-1], "user")))[
                        "status"
                    ] == 403
                    for role in ["user", "admin"]:
                        result = await get(path, create_jwt(uid, role))
                        assert result["status"] == 200
                        items = {x["id"]: x for x in result["data"]}
                        assert (
                            items[m1.id]["metadata"]["link_card"]["title"]
                            == "first content"
                        )
                        assert (
                            items[m2.id]["metadata"]["link_card"]["snapshot_id"]
                            == second.snapshot_id
                        )
                    shares = await get(
                        "/daily-share/links?limit=10", create_jwt(uid, "user")
                    )
                    assert shares["data"]["total"] == 2
                    assert {
                        x["message_id"]
                        for g in shares["data"]["groups"]
                        for x in g["links"]
                    } == {m1.id, m2.id}
                    await db.message.delete(where={"id": m1.id})
                    assert (await get("/daily-share/links", create_jwt(uid, "user")))[
                        "data"
                    ]["total"] == 1
                    await db.conversation.update(
                        where={"id": convs[0]}, data={"isDeleted": True}
                    )
                    assert (await get(path, create_jwt(uid, "admin")))["status"] == 404
                    assert not errors and not violations
                except BaseException:
                    await page.screenshot(
                        path=str(artifacts / "link-snapshot-api-failure.png"),
                        full_page=True,
                    )
                    raise
                finally:
                    await context.tracing.stop(
                        path=str(artifacts / "link-snapshot-api-trace.zip")
                    )
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 10)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        sock.close()
