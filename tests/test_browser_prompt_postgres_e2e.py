"""Chromium → real HTTP/JWT/admin router → Prisma/PostgreSQL/Redis → hot read.

The HTML is a small test driver, NOT the production React UI (tested separately
in Companion_web). No LLM is invoked, and no production data is copied.
"""
import asyncio
from contextlib import suppress
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from playwright.async_api import async_playwright
from prisma import Prisma
import pytest
from redis.asyncio import Redis
import uvicorn

from app.api.admin import prompts as admin
from app.api.jwt_auth import require_admin_jwt
from app.services.auth import create_jwt
from app.services.prompting import store
from app.services.prompting.registry import PROMPT_DEFINITION_MAP


@pytest.mark.asyncio
async def test_browser_versioned_publication_persists_and_hot_reads(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    redis_url = os.environ.get("PROACTIVE_E2E_REDIS_URL", "")
    if not url or not redis_url:
        pytest.skip("Browser business E2E requires isolated PostgreSQL and Redis")
    assert urlsplit(url).hostname in {"127.0.0.1", "localhost"}
    assert urlsplit(url).path == "/companion_proactive_e2e"
    assert urlsplit(redis_url).hostname in {"127.0.0.1", "localhost"}
    assert urlsplit(redis_url).path == "/14"
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    redis = Redis.from_url(redis_url, decode_responses=True)
    await database.connect()
    definition = replace(PROMPT_DEFINITION_MAP["memory.relevance"], key="test.browser." + uuid4().hex,
                         default_text="合成默认 {message}")
    monkeypatch.setattr(store, "db", database)
    monkeypatch.setattr(store, "PROMPT_DEFINITION_MAP", {definition.key: definition})
    monkeypatch.setattr(store, "PROMPT_DEFINITIONS", [definition])
    monkeypatch.setattr(store, "get_redis", AsyncMock(return_value=redis))
    # Background evaluation is a separate gate. This test proves publication,
    # permissions, concurrency guards and runtime reads, not model quality.
    monkeypatch.setattr(store, "_schedule_eval", lambda _: None)
    token = store._prompt_snapshot.set(None)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0)); sock.listen(128); sock.setblocking(False)
    origin = f"http://127.0.0.1:{sock.getsockname()[1]}"
    app = FastAPI()
    app.include_router(admin.router)

    @app.get("/__hot-read")
    async def hot_read(_=Depends(require_admin_jwt)):
        return {"rendered": (await store.get_prompt_text(definition.key)).format(message="当前请求")}

    @app.get("/", response_class=HTMLResponse)
    async def driver():
        return """<!doctype html><html><meta charset="utf-8"><body>
        <form><textarea aria-label="草稿"></textarea><button>发布</button></form><output></output>
        <script>window.request = async (path, token, method='GET', data=null) => {
          const r=await fetch(path,{method,headers:{'Authorization':'Bearer '+token,
            'Content-Type':'application/json'},body:data===null?null:JSON.stringify(data)});
          return {status:r.status,data:await r.json()};
        }; document.querySelector('form').onsubmit=async event=>{
          event.preventDefault(); const r=await window.request(window.path,window.token,'PUT',
            {content:document.querySelector('textarea').value,expected_revision:window.revision});
          document.querySelector('output').textContent=JSON.stringify(r);
        };</script></body></html>"""

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    task = None
    artifacts = Path(os.environ.get("BROWSER_ARTIFACT_DIR", "reports/browser"))
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        await store.ensure_prompt_templates()
        task = asyncio.create_task(server.serve(sockets=[sock]))
        async with asyncio.timeout(15):
            while not server.started:
                if task.done(): await task
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
                        violations.append("non-fixture browser request")
                        await route.abort()

                await context.route("**/*", fence)
                await context.tracing.start(screenshots=True, snapshots=True)
                page = await context.new_page()
                page_errors = []
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                try:
                    await page.goto(origin)
                    admin_token = create_jwt("synthetic-browser-admin", role="admin")
                    user_token = create_jwt("synthetic-browser-user", role="user")
                    path = "/admin-api/prompts/" + definition.key

                    async def request(path, token, method="GET", data=None):
                        return await page.evaluate("args => window.request(...args)", [path, token, method, data])

                    assert (await request("/admin-api/prompts", ""))["status"] == 401
                    assert (await request("/admin-api/prompts", user_token))["status"] == 403
                    initial = await request("/admin-api/prompts", admin_token)
                    assert initial["status"] == 200 and len(initial["data"]) == 1
                    row = initial["data"][0]
                    await page.evaluate("args => {window.path=args[0];window.token=args[1];window.revision=args[2]}",
                                        [path, admin_token, row["revision"]])
                    await page.get_by_role("textbox").fill("浏览器发布 {message}")
                    await page.get_by_role("button", name="发布").click()
                    await page.wait_for_function("document.querySelector('output').textContent.length > 0")
                    saved = json.loads(await page.locator("output").inner_text())
                    assert saved["status"] == 200
                    assert saved["data"]["web_version"] == 1
                    assert saved["data"]["content_version_type"] == "web"
                    persisted = await database.prompttemplate.find_unique(where={"key": definition.key})
                    assert persisted.content == "浏览器发布 {message}"
                    assert await redis.get(store._redis_key(definition.key)) == persisted.content
                    history = await request(path + "/versions", admin_token)
                    assert history["status"] == 200
                    assert history["data"][0]["change_type"] == "manual_save"
                    assert history["data"][0]["web_version"] == 1
                    assert (await request("/__hot-read", admin_token))["data"]["rendered"] == "浏览器发布 当前请求"
                    stale = await request(path, admin_token, "PUT", {"content": "旧管理员", "expected_revision": row["revision"]})
                    assert stale["status"] == 409
                    denied = await request(path, user_token, "PUT", {"content": "越权写入"})
                    assert denied["status"] == 403
                    assert (await database.prompttemplate.find_unique(where={"key": definition.key})).content == persisted.content
                    assert len(await store.list_prompt_versions(definition.key)) == 2
                    assert not violations and not page_errors
                except BaseException:
                    await page.screenshot(path=str(artifacts / "real-api-failure.png"), full_page=True)
                    raise
                finally:
                    await context.tracing.stop(path=str(artifacts / "real-api-trace.zip"))
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        if task:
            try:
                await asyncio.wait_for(task, timeout=10)
            except TimeoutError:
                task.cancel()
                with suppress(asyncio.CancelledError): await task
        sock.close()
        store._prompt_snapshot.reset(token)
        await database.prompttemplateversion.delete_many(where={"promptKey": definition.key})
        await database.prompttemplate.delete_many(where={"key": definition.key})
        await database.promptpublicationcounter.delete_many(where={"promptKey": definition.key})
        await redis.delete("prompt_snapshot:" + definition.key, store._redis_key(definition.key), store._enabled_redis_key(definition.key))
        await redis.aclose()
        await database.disconnect()
