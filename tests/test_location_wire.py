"""Actual HTTP tickets, WS admission, Redis recovery and persisted source reads."""

import asyncio
import json
import os
import subprocess
import sys
from urllib.parse import urlsplit

import httpx
import pytest
from websockets.asyncio.client import connect

from app.config import settings
from app.services.auth import create_jwt
from app.services.runtime.ws_auth import CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX
from tests.test_admin_takeover_wire import port, event
from tests.test_runtime_execution_foundation import flow
from tests.test_location_component_card import _location_card


async def test_location_ws_retry_recovers_dual_memory_and_hydrates_history(
    flow, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "jwt_secret", "synthetic-location-wire-secret-12345")
    chosen = port()
    url = f"http://127.0.0.1:{chosen}"
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(s in k for s in ("TOKEN", "API_KEY", "SECRET", "PASSWORD", "PROXY"))
    }
    env.update(
        DATABASE_URL=flow.url,
        DIRECT_DATABASE_URL=flow.url,
        REDIS_URL=os.environ["PROACTIVE_E2E_REDIS_URL"],
        JWT_SECRET=settings.jwt_secret,
        APP_ENV="test",
        TRACE_BACKEND="off",
        PYTHON_DOTENV_DISABLED="1",
        ONLINE_MODEL="false",
        CHAT_INGRESS_BACKEND="redis",
    )
    for k in ("DATABASE_URL", "REDIS_URL"):
        assert urlsplit(env[k]).hostname in {"localhost", "127.0.0.1"}
    log = (tmp_path / "location-wire.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.fixtures.location_wire_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(chosen),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=log,
        stderr=log,
    )
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            for _ in range(300):
                try:
                    if (await client.get(url + "/health")).status_code == 200:
                        break
                except httpx.RequestError:
                    pass
                assert process.poll() is None
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("wire worker startup timeout")
            conv = flow.ids["conversation"]
            owner = flow.ids["owner"]
            workspace = flow.ids["workspace"]
            user = {"Authorization": "Bearer " + create_jwt(owner, "user")}
            admin = {"Authorization": "Bearer " + create_jwt(owner, "admin")}
            payload = {
                "message": "",
                "client_id": "location-wire-retry",
                "component_card": _location_card(),
            }

            async def socket():
                r = await client.post(url + f"/chat/{conv}/ws-ticket", headers=user)
                assert r.status_code == 200, r.text
                return connect(
                    url.replace("http", "ws") + f"/ws/{conv}?client=flutter",
                    subprotocols=[
                        CHAT_PROTOCOL,
                        TICKET_PROTOCOL_PREFIX + r.json()["ticket"],
                    ],
                    proxy=None,
                )

            async with await socket() as peer:
                await peer.send(json.dumps({"type": "message", "data": payload}))
                ack = await event(peer, "ack")
                mid = ack["message_id"]
                assert ack["client_id"] == payload["client_id"]
                await peer.send(json.dumps({"type": "message", "data": payload}))
                assert (await event(peer, "ack"))["message_id"] == mid
            # Background fire-and-forget was deliberately suppressed. Recovery
            # must discover the durable pending row rather than depend on it.
            r = await client.post(url + "/__test__/recover-location")
            assert r.status_code == 200, r.text
            assert r.json() == {"dispatched": 1, "processed": 1, "reply_calls": 1}
            assert (
                await client.get(
                    url + "/admin-api/runtime-jobs/location-events", headers=user
                )
            ).status_code == 403
            records = (
                await client.get(
                    url + "/admin-api/runtime-jobs/location-events", headers=admin
                )
            ).json()
            receipt = next(i for i in records["items"] if i["message_id"] == mid)
            assert (
                receipt["status"] == "completed"
                and receipt["queue_state"] == "not_needed"
            )
            ids = (
                await flow.db.query_raw(
                    "SELECT memory_ids FROM location_memory_events WHERE message_id=$1",
                    mid,
                )
            )[0]["memory_ids"]
            for side, ident in zip(("user", "ai"), ids):
                r = await client.get(
                    url + f"/admin-api/memory-repairs/evidence/{side}/{ident}",
                    headers=admin,
                    params={"user_id": owner, "workspace_id": workspace},
                )
                assert r.status_code == 200, r.text
                body = r.json()
                assert body["state"] == "linked"
                origin = next(i for i in body["items"] if i["source_kind"] == "input")
                assert origin["input"]["kind"] == "location_share"
                assert "31.2304" not in r.text
            async with await socket() as peer:
                await peer.send(json.dumps({"type": "message", "data": payload}))
                assert (await event(peer, "ack"))["message_id"] == mid
                history = (
                    await client.get(
                        url + f"/conversations/{conv}/messages?include_metadata=true",
                        headers=user,
                    )
                ).json()
                row = next(m for m in history if m["id"] == mid)
                assert (
                    row["metadata"]["component_card"]["payload"]["latitude"] == 31.2304
                )
            assert await flow.db.message.count(where={"conversationId": conv}) == 1
            assert (
                await flow.db.query_raw(
                    "SELECT used FROM user_message_quota WHERE user_id=$1", owner
                )
            ) == [{"used": 1}]
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()
        print((tmp_path / "location-wire.log").read_text()[-5000:])
        # The shared runtime fixture owns the user/conversation; this test also
        # creates memories and their audit rows, which it must clean first.
        for table in ("memories_user", "memories_ai"):
            await flow.db.execute_raw(
                f"DELETE FROM memory_embeddings WHERE memory_id IN (SELECT id FROM {table} WHERE user_id=$1)",
                flow.ids["owner"],
            )
        await flow.db.usermemory.delete_many(where={"userId": flow.ids["owner"]})
        await flow.db.aimemory.delete_many(where={"userId": flow.ids["owner"]})
        await flow.db.memorychangelog.delete_many(where={"userId": flow.ids["owner"]})
