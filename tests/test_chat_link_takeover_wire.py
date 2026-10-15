"""Pinned link cards through real takeover HTTP/WS, PostgreSQL and Redis.

The two API processes use the existing takeover wire application. Only external
cover preparation is replaced when seeding owned snapshots; admission, tickets,
quota, inbox publication, history and read receipts use their real implementations.
This is protocol evidence, not a browser or external-provider test.
"""

import asyncio
from contextlib import AsyncExitStack
import copy
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
from app.services.chat_links import cards, repo
from app.services.chat_links.covers import CachedCoverResult
from app.services.chat_links.extraction import LinkMetadata
from app.services.runtime.ws_auth import CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX
from tests.test_admin_takeover_postgres import attach, human
from tests.test_admin_takeover_wire import event, port
from tests.test_runtime_execution_foundation import flow


@pytest.fixture
async def link_wire(human, tmp_path, monkeypatch):
    h = human
    await attach(h)

    async def cover(**kwargs):
        return CachedCoverResult(kwargs["metadata"], {})

    monkeypatch.setattr(repo, "cache_link_cover", cover)

    async def prepare(title, *, conversation_id=None, role="user"):
        return await repo.create_or_update_link_card(
            user_id=h.flow.ids["owner"],
            conversation_id=conversation_id or h.flow.ids["conversation"],
            role=role,
            source_app="wire_fixture",
            metadata=LinkMetadata(
                source_url="https://example.com/pinned-link",
                final_url="https://example.com/pinned-link",
                platform="链接",
                title=title,
                original_text=title,
                content_text=title,
                summary=title,
            ),
            database=h.flow.db,
        )

    processes, logs, urls = [], [], []
    env = {
        key: value for key, value in os.environ.items()
        if not any(secret in key for secret in ("TOKEN", "API_KEY", "SECRET", "PASSWORD", "PROXY"))
    }
    env.update(
        DATABASE_URL=h.flow.url, DIRECT_DATABASE_URL=h.flow.url,
        REDIS_URL=os.environ["PROACTIVE_E2E_REDIS_URL"], JWT_SECRET=settings.jwt_secret,
        APP_ENV="test", PYTHON_DOTENV_DISABLED="1", TRACE_BACKEND="off",
        ONLINE_MODEL="false", CHAT_INGRESS_BACKEND="redis",
    )
    for key in ("DATABASE_URL", "REDIS_URL"):
        assert urlsplit(env[key]).hostname in {"127.0.0.1", "localhost"}
    try:
        for index in range(2):
            chosen = port()
            urls.append(f"http://127.0.0.1:{chosen}")
            log = (tmp_path / f"link-wire-{index}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "tests.fixtures.admin_takeover_wire_app:app",
                 "--host", "127.0.0.1", "--port", str(chosen), "--log-level", "warning"],
                env=env, stdout=log, stderr=log,
            ))
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            for target in urls:
                for _ in range(300):
                    try:
                        if (await client.get(target + "/health")).status_code == 200:
                            break
                    except httpx.RequestError:
                        pass
                    assert all(process.poll() is None for process in processes), "wire worker exited"
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("wire worker startup timeout")
            yield h, client, urls, prepare
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()
        for path in tmp_path.glob("link-wire-*.log"):
            if path.stat().st_size:
                print(path.read_text()[-4000:])


def sse_events(response):
    result = []
    for block in response.text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            result.append((fields["event"], json.loads(fields["data"])))
    return result


@pytest.mark.parametrize("transport", ["http", "ws"])
async def test_pinned_link_reaches_operator_history_and_read_receipts(link_wire, transport):
    h, client, urls, prepare = link_wire
    conv = h.flow.ids["conversation"]
    admin = {"Authorization": "Bearer " + create_jwt(h.admins[0], "admin")}
    user = {"Authorization": "Bearer " + create_jwt(h.flow.ids["owner"], "user")}
    base = "/admin-api/takeovers/" + h.case
    response = await client.post(urls[1] + base + "/claim", headers=admin)
    assert response.status_code == 200, response.text
    # A recommendation keeps its assistant source; the user's reshare is a new
    # user occurrence and must never turn into the cache's newer version.
    pinned = await prepare("original recommendation", role="assistant")
    newer = await prepare("changed recommendation", role="assistant")
    assert pinned.id == newer.id and pinned.snapshot_id != newer.snapshot_id
    incoming = cards.component_card_for_link(pinned)
    incoming.update(title="forged title", body="forged body")
    incoming["payload"].update(content_text="forged content", summary="forged summary")
    payload = {"message": "看看这个链接", "client_id": "pinned-share", "component_card": incoming}
    inbox = (await client.post(urls[0] + "/admin-api/takeovers/inbox-ticket", headers=admin)).json()["ticket"]
    ticket = (await client.post(urls[0] + f"/chat/{conv}/ws-ticket", headers=user)).json()["ticket"]
    async with AsyncExitStack() as stack:
        operator = await stack.enter_async_context(connect(
            urls[0].replace("http", "ws") + "/ws/admin-inbox/" + h.admins[0],
            subprotocols=[CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + inbox], proxy=None,
        ))
        peer = await stack.enter_async_context(connect(
            urls[0].replace("http", "ws") + f"/ws/{conv}?client=flutter",
            subprotocols=[CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + ticket], proxy=None,
        ))
        await operator.send(json.dumps({"type": "ping"}))
        assert await event(operator, "pong") == {}
        await peer.send(json.dumps({"type": "ping"}))
        assert await event(peer, "pong") == {}

        async def send(value, route=None):
            if (route or transport) == "ws":
                await peer.send(json.dumps({"type": "message", "data": value}))
                return await event(peer, "ack")
            response = await client.post(urls[1] + f"/chat/{conv}", headers=user, json=value)
            assert response.status_code == 200, response.text
            events = sse_events(response)
            assert [kind for kind, _ in events] == ["ack", "done"]
            assert events[1][1]["acceptance_only"] is True
            return events[0][1]

        ack = await send(payload)
        assert ack["client_id"] == "pinned-share" and ack["human_takeover"] is True
        received = await event(operator, "takeover_user_message")
        message_id = ack["message_id"]
        assert received["message_id"] == message_id and received["role"] == "user"
        expected = cards.metadata_for_link_card(pinned)
        assert received["metadata"]["link_card"] == expected
        assert received["metadata"]["component_card"] == cards.component_card_for_link(pinned)
        assert received["metadata"]["takeover_sequence"] == 1
        summary = "/admin-api/takeovers/unread-summary"
        assert (await client.get(urls[1] + summary, headers=admin)).json()["unread_count"] == 1
        assert (await send(payload))["message_id"] == message_id
        assert (await send(payload, "ws" if transport == "http" else "http"))["message_id"] == message_id
        # The retry is acknowledged without another inbox notification.
        await operator.send(json.dumps({"type": "ping"}))
        assert json.loads(await operator.recv())["type"] == "pong"
        for auth in (user, admin):
            history = (await client.get(
                urls[1] + f"/conversations/{conv}/messages?include_metadata=true", headers=auth,
            )).json()
            saved = next(row for row in history if row["id"] == message_id)
            assert saved["role"] == "user" and saved["metadata"]["link_card"] == expected
        shares = (await client.get(urls[1] + "/daily-share/links", headers=user)).json()
        occurrences = [link for group in shares["groups"] for link in group["links"]]
        assert len(occurrences) == 1
        assert occurrences[0]["message_id"] == message_id and occurrences[0]["role"] == "user"
        assert occurrences[0]["component_card"]["payload"]["snapshot_id"] == pinned.snapshot_id
        assert occurrences[0]["title"] == "original recommendation"
        assert (await client.post(urls[1] + base + "/read", headers=admin,
                                 json={"message_id": message_id})).status_code == 200
        assert (await client.get(urls[0] + summary, headers=admin)).json()["unread_count"] == 0
        reply = await client.post(urls[1] + base + "/messages", headers=admin,
                                  json={"text": "已经看到你分享的原始版本", "client_id": "link-reply"})
        assert reply.status_code == 200, reply.text
        assistant_id = reply.json()["message_id"]
        assert (await event(peer, "reply"))["assistant_message_id"] == assistant_id
        assert (await event(peer, "done"))["message_id"] == assistant_id
        assert (await event(operator, "takeover_admin_message"))["message_id"] == assistant_id
        read_path = urls[1] + f"/conversations/{conv}/read"
        assert (await client.post(read_path, headers=user, json={"message_ids": [assistant_id]})).status_code == 200
        receipt = await event(operator, "takeover_user_read")
        assert receipt["message_ids"] == [assistant_id] and receipt["client_ids"] == ["link-reply"]
    # Reconnect to the other process and hydrate the persisted original version.
    ticket = (await client.post(urls[1] + f"/chat/{conv}/ws-ticket", headers=user)).json()["ticket"]
    async with connect(urls[1].replace("http", "ws") + f"/ws/{conv}?client=flutter",
                       subprotocols=[CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + ticket], proxy=None) as reconnected:
        await reconnected.send(json.dumps({"type": "ping"}))
        assert await event(reconnected, "pong") == {}
        history = (await client.get(urls[1] + f"/conversations/{conv}/messages?include_metadata=true", headers=user)).json()
    assert next(row for row in history if row["id"] == message_id)["metadata"]["link_card"] == expected
    assert next(row for row in history if row["id"] == assistant_id)["read"] is True
    assert await h.flow.db.message.count(where={"conversationId": conv}) == 2
    assert await h.flow.db.query_raw("SELECT used FROM user_message_quota WHERE user_id=$1", h.flow.ids["owner"]) == [{"used": 1}]
    assert await h.flow.db.agentrun.count(where={"conversationId": conv}) == 0


async def test_foreign_or_missing_link_version_does_not_admit_charge_or_notify(link_wire):
    h, client, urls, prepare = link_wire
    conv = h.flow.ids["conversation"]
    admin = {"Authorization": "Bearer " + create_jwt(h.admins[0], "admin")}
    user = {"Authorization": "Bearer " + create_jwt(h.flow.ids["owner"], "user")}
    response = await client.post(urls[1] + "/admin-api/takeovers/" + h.case + "/claim", headers=admin)
    assert response.status_code == 200, response.text
    valid = await prepare("owned version")
    foreign = await prepare("different conversation", conversation_id=h.flow.ids["other_conversation"])
    invalid_cards = [cards.component_card_for_link(foreign),
                     {"type": "external_link", "payload": {"source_url": "https://example.com/unprepared"}}]
    missing = copy.deepcopy(cards.component_card_for_link(valid))
    missing["payload"]["snapshot_id"] = "missing-version"
    invalid_cards.append(missing)
    mismatched = copy.deepcopy(cards.component_card_for_link(valid))
    mismatched["payload"]["snapshot_id"] = foreign.snapshot_id
    invalid_cards.append(mismatched)
    for field in ("link_id", "snapshot_id"):
        for invalid in (None, "", "  ", [], 42, "x" * 81):
            card = copy.deepcopy(cards.component_card_for_link(valid))
            card["payload"][field] = invalid
            invalid_cards.append(card)
    ticket = (await client.post(urls[0] + f"/chat/{conv}/ws-ticket", headers=user)).json()["ticket"]
    async with connect(urls[0].replace("http", "ws") + f"/ws/{conv}?client=flutter",
                       subprotocols=[CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + ticket], proxy=None) as peer:
        for index, card in enumerate(invalid_cards):
            payload = {"message": "bad link", "client_id": f"invalid-{index}", "component_card": card}
            response = await client.post(urls[1] + f"/chat/{conv}", headers=user, json=payload)
            assert response.status_code == 422, response.text
            assert response.json()["detail"]["code"] == "invalid_resource"
            await peer.send(json.dumps({"type": "message", "data": payload}))
            assert (await event(peer, "error"))["client_id"] == payload["client_id"]
        await peer.send(json.dumps({"type": "ping"}))
        assert await event(peer, "pong") == {}
    assert await h.flow.db.message.count(where={"conversationId": conv}) == 0
    assert await h.flow.db.query_raw("SELECT used FROM user_message_quota WHERE user_id=$1", h.flow.ids["owner"]) == []
    assert (await client.get(urls[0] + "/admin-api/takeovers/unread-summary", headers=admin)).json()["unread_count"] == 0
    assert await h.flow.db.query_raw("SELECT id FROM notification_events WHERE type='admin_takeover_message' AND payload->>'conversation_id'=$1", conv) == []


async def test_takeover_literal_url_and_legacy_cache_have_distinct_metadata(link_wire):
    h, client, urls, prepare = link_wire
    conv = h.flow.ids["conversation"]
    admin = {"Authorization": "Bearer " + create_jwt(h.admins[0], "admin")}
    user = {"Authorization": "Bearer " + create_jwt(h.flow.ids["owner"], "user")}
    response = await client.post(urls[1] + "/admin-api/takeovers/" + h.case + "/claim", headers=admin)
    assert response.status_code == 200, response.text
    literal = "人工消息里的网址 https://example.com/plain-text-only"
    request = {"message": literal, "client_id": "literal-url"}
    response = await client.post(urls[1] + f"/chat/{conv}", headers=user, json=request)
    assert response.status_code == 200, response.text
    first_id = sse_events(response)[0][1]["message_id"]
    retry = await client.post(urls[0] + f"/chat/{conv}", headers=user, json=request)
    assert retry.status_code == 200, retry.text
    assert sse_events(retry)[0][1]["message_id"] == first_id
    saved = await h.flow.db.message.find_unique(where={"id": first_id})
    assert saved.content == literal and "link_card" not in saved.metadata
    assert "component_card" not in saved.metadata
    assert await h.flow.db.query_raw("SELECT id FROM chat_link_cards WHERE conversation_id=$1", conv) == []
    assert await h.flow.db.query_raw("SELECT used FROM user_message_quota WHERE user_id=$1", h.flow.ids["owner"]) == [{"used": 1}]

    legacy = await prepare("legacy stored content")
    await h.flow.db.execute_raw("DELETE FROM chat_link_snapshots WHERE id=$1", legacy.snapshot_id)
    card = cards.component_card_for_link(legacy)
    card["payload"].pop("snapshot_id")
    card["body"] = "untrusted replacement"
    request = {"message": "旧卡片", "client_id": "legacy-cache", "component_card": card}
    response = await client.post(urls[1] + f"/chat/{conv}", headers=user, json=request)
    assert response.status_code == 200, response.text
    legacy_id = sse_events(response)[0][1]["message_id"]
    saved = await h.flow.db.message.find_unique(where={"id": legacy_id})
    projection = saved.metadata["link_card"]
    assert projection["id"] == legacy.id and projection["title"] == "legacy stored content"
    assert projection["snapshot_id"] and projection["snapshot_id"] != legacy.snapshot_id
    assert projection["snapshot_recorded_at"]
    retry = await client.post(urls[0] + f"/chat/{conv}", headers=user, json=request)
    assert retry.status_code == 200, retry.text
    assert sse_events(retry)[0][1]["message_id"] == legacy_id
    assert await h.flow.db.query_raw("SELECT id FROM chat_link_snapshots WHERE link_card_id=$1", legacy.id) == [{"id": projection["snapshot_id"]}]
    assert await h.flow.db.query_raw("SELECT used FROM user_message_quota WHERE user_id=$1", h.flow.ids["owner"]) == [{"used": 2}]
    assert await h.flow.db.message.count(where={"conversationId": conv}) == 2
    assert await h.flow.db.agentrun.count(where={"conversationId": conv}) == 0


async def test_ordinary_http_rejects_resources_before_silent_drop(link_wire):
    h, client, urls, prepare = link_wire
    conv = h.flow.ids["conversation"]
    user = {"Authorization": "Bearer " + create_jwt(h.flow.ids["owner"], "user")}
    card = cards.component_card_for_link(await prepare("resource must not disappear"))
    for resource in ({"component_card": card}, {"attachments": [{"id": "unbound-media"}]}):
        response = await client.post(urls[1] + f"/chat/{conv}", headers=user,
            json={"message": "不要丢弃资源", "client_id": "unsupported-resource", **resource})
        assert response.status_code == 422, response.text
        assert response.json()["detail"] == {
            "code": "unsupported_transport_payload",
            "message": "卡片和附件请通过聊天连接发送",
            "client_id": "unsupported-resource",
        }
    assert await h.flow.db.message.count(where={"conversationId": conv}) == 0
    assert await h.flow.db.query_raw("SELECT used FROM user_message_quota WHERE user_id=$1", h.flow.ids["owner"]) == []
