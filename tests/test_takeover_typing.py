"""Synthetic scoped typing: real Redis ordering, no composer contents or chat writes."""
import os
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from redis import Redis as SyncRedis
from redis.asyncio import Redis

from app.api.admin import takeovers
from app.api.public import conversations
from app.api.jwt_auth import require_user
from app.services.admin_takeover import service
from app.services.runtime.ws_manager import manager


@pytest.fixture(scope="module")
def typing_redis_url(tmp_path_factory):
    external = os.getenv("PROACTIVE_E2E_REDIS_URL")
    if external:
        assert urlsplit(external).hostname in {"127.0.0.1", "localhost"}
        yield external
        return
    binary = shutil.which("redis-server")
    if not binary:
        pytest.skip("isolated redis-server is required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    directory = tmp_path_factory.mktemp("typing-redis")
    process = subprocess.Popen([binary, "--bind", "127.0.0.1", "--port", str(port),
        "--save", "", "--appendonly", "no", "--dir", str(directory)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"redis://127.0.0.1:{port}/0"
    client = SyncRedis.from_url(url)
    try:
        for _ in range(100):
            try:
                if client.ping(): break
            except Exception: time.sleep(.02)
        else: raise AssertionError("isolated Redis failed to start")
        yield url
    finally:
        client.close()
        process.terminate()
        process.wait(timeout=5)


class NamespacedRedis:
    """Own only synthetic keys; CI shares a loopback Redis with other fixtures."""
    def __init__(self, client):
        self.client = client
        self.prefix = "typing-test-" + uuid4().hex + ":"

    async def eval(self, script, count, *args):
        return await self.client.eval(script, count,
            *(self.prefix + str(key) for key in args[:count]), *args[count:])

    async def keys(self, pattern):
        return [key.removeprefix(self.prefix)
            for key in await self.client.keys(self.prefix + pattern)]

    async def ttl(self, key):
        return await self.client.ttl(self.prefix + key)

    async def close(self):
        keys = await self.client.keys(self.prefix + "*")
        if keys:
            await self.client.delete(*keys)
        await self.client.aclose()


@pytest.fixture
async def typing(typing_redis_url, monkeypatch):
    redis = NamespacedRedis(Redis.from_url(typing_redis_url, decode_responses=True))
    monkeypatch.setattr(service, "get_redis", AsyncMock(return_value=redis))
    query = AsyncMock(return_value=[{"id": "case", "ownerAdminId": "admin", "conversationId": "conv"}])
    monkeypatch.setattr(service, "db", SimpleNamespace(query_raw=query, message=SimpleNamespace(
        create=AsyncMock(side_effect=AssertionError("typing must not create messages")))))
    monkeypatch.setattr(manager, "send_to_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(manager, "send_event", AsyncMock(return_value=True))
    yield redis, query
    service.db.message.create.assert_not_awaited()
    await redis.close()


async def test_user_typing_is_scoped_transient_and_orders_late_events(typing):
    redis, query = typing
    assert await service.publish_user_typing("conv", "user", True, revision=1, session_id="session")
    assert await service.publish_user_typing("conv", "user", False, revision=3, session_id="session")
    assert await service.publish_user_typing("conv", "user", True, revision=2, session_id="session")
    assert manager.send_to_admin.await_count == 2
    first, second = manager.send_to_admin.await_args_list
    assert first.args[:2] == ("admin", "takeover_user_typing")
    assert first.args[2] == {"takeover_id": "case", "conversation_id": "conv", "active": True, "revision": first.args[2]["revision"]}
    assert first.args[2]["revision"] > 2_147_483_647
    assert second.args[2]["active"] is False and second.args[2]["revision"] > first.args[2]["revision"]
    assert query.await_args.args[1:] == ("conv", "user")
    keys = await redis.keys("*")
    assert all(key.startswith("takeover:typing:{conv}:user:") for key in keys)
    assert await redis.ttl("takeover:typing:{conv}:user:session:session") > 0


async def test_refresh_is_throttled_but_start_stop_transitions_are_immediate(typing):
    await service.publish_user_typing("conv", "user", True, revision=1, session_id="session")
    await service.publish_user_typing("conv", "user", True, revision=2, session_id="session")
    assert manager.send_to_admin.await_count == 1
    await service.publish_user_typing("conv", "user", False, revision=3, session_id="session")
    await service.publish_user_typing("conv", "user", True, revision=4, session_id="session")
    assert [c.args[2]["active"] for c in manager.send_to_admin.await_args_list] == [True, False, True]


async def test_new_admin_page_gets_monotonic_server_sequence(typing):
    await service.publish_takeover_typing("case", "admin", True, revision=100, session_id="old-page")
    await service.publish_takeover_typing("case", "admin", False, revision=101, session_id="old-page")
    await service.publish_takeover_typing("case", "admin", True, revision=1, session_id="new-page")
    events = [c.args[2] for c in manager.send_event.await_args_list]
    assert events[0]["revision"] > 2_147_483_647
    assert all(a["revision"] < b["revision"] for a, b in zip(events, events[1:]))
    assert [e["active"] for e in events] == [True, False, True]
    # A reordered request from the old page cannot regress its own stop revision.
    await service.publish_takeover_typing("case", "admin", True, revision=99, session_id="old-page")
    assert manager.send_event.await_count == 3


async def test_no_owned_takeover_returns_same_noop_without_notifying(typing):
    _, query = typing
    query.return_value = []
    assert not await service.publish_user_typing("conv", "foreign", True, revision=1, session_id="s")
    assert not await service.publish_takeover_typing("case", "foreign", True, revision=1, session_id="s")
    service.get_redis.assert_not_awaited()
    manager.send_event.assert_not_awaited()
    manager.send_to_admin.assert_not_awaited()


def typing_app():
    app = FastAPI()
    app.include_router(conversations.router)
    app.include_router(takeovers.router)
    app.dependency_overrides[require_user] = lambda: {"sub": "actual-user", "role": "user"}
    app.dependency_overrides[takeovers.require_takeover_admin] = lambda: {"sub": "admin", "role": "admin"}
    return app


async def test_http_uses_token_identity_and_forwards_no_draft_or_claimed_owner(typing):
    _, query = typing
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=typing_app()), base_url="http://test") as client:
        response = await client.post("/conversations/conv/typing", json={"active": True,
            "revision": 1, "session_id": "s", "user_id": "victim", "text": "private draft"})
        assert response.status_code == 200
    assert query.await_args.args[1:] == ("conv", "actual-user")
    assert "private draft" not in str(manager.send_to_admin.await_args)
    assert "victim" not in str(manager.send_to_admin.await_args)


@pytest.mark.parametrize("payload", [
    {"active": "true", "revision": 1, "session_id": "s"},
    {"active": True, "revision": True, "session_id": "s"},
    {"active": True, "revision": -1, "session_id": "s"},
    {"active": True, "revision": 1, "session_id": "x" * 65},
    {"active": True, "revision": 1, "session_id": "a:b"},
])
async def test_http_rejects_invalid_presence_before_lookup(typing, payload):
    _, query = typing
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=typing_app()), base_url="http://test") as client:
        for path in ("/conversations/conv/typing", "/admin-api/takeovers/case/typing"):
            assert (await client.post(path, json=payload)).status_code == 422
    query.assert_not_awaited()


async def test_user_presence_requires_authentication(typing):
    app = FastAPI()
    app.include_router(conversations.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/conversations/conv/typing", json={"active": True, "revision": 1, "session_id": "s"})
    assert response.status_code == 401
    manager.send_to_admin.assert_not_awaited()
