"""Network probe run inside the same isolated network as the test workers."""

import asyncio
import json
import time

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from app.services.runtime.ws_auth import CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX

from app.services.auth import create_jwt


async def wait_state(client, worker, predicate, *, timeout=10):
    deadline = time.monotonic() + timeout
    last_observation = "No response"
    while time.monotonic() < deadline:
        try:
            result = await client.get(f"http://{worker}:8000/test/state")
            if result.status_code == 200 and predicate(result.json()):
                return result.json()
            last_observation = f"HTTP {result.status_code}: {result.text[:300]}"
        except httpx.HTTPError as exc:
            last_observation = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(0.05)
    raise AssertionError(f"State deadline exceeded for {worker} ({timeout}s); {last_observation}")


async def main():
    completed = []
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        for worker in ("worker-a", "worker-b"):
            # Cold production-module imports on shared CI CPUs take longer
            # than reconnect cleanup. Only startup gets this larger budget.
            await wait_state(client, worker, lambda _: True, timeout=60)
        # Give both real psubscribe loops a chance to register; verify via delivery below.
        await asyncio.sleep(0.2)
        base = "http://worker-a:8000"
        paths = [("POST", "/chat/conv-1"), ("POST", "/chat/proactive/agent-1?user_id=owner"),
                 ("GET", "/chat/proactive/agent-1/history?user_id=owner")]
        for method, path in paths:
            for headers, expected in [({}, 401), ({"Authorization": "Bearer broken"}, 401),
                                      ({"Authorization": f"Bearer {create_jwt('intruder', 'user')}"}, 403)]:
                response = await client.request(method, base + path, headers=headers,
                                                json={"message": "hello"} if method == "POST" else None)
                assert response.status_code == expected, response.text
        state = (await client.get(base + "/test/state")).json()
        assert state["effects"] == {"messages": 0, "history": 0, "trigger": 0}
        completed.append("HTTP JWT/ownership rejects have no business side effects (9 requests)")
        headers = {"Authorization": f"Bearer {create_jwt('owner', 'user')}"}
        for method, path in paths:
            response = await client.request(method, base + path, headers=headers,
                                            json={"message": "hello"} if method == "POST" else None)
            assert response.status_code == 200, response.text
        assert (await client.get(base + "/test/state")).json()["effects"] == {
            "messages": 1, "history": 1, "trigger": 1}
        completed.append("Authenticated HTTP routes preserve success contracts")
        async def mint():
            response = await client.post(base + "/chat/conv-1/ws-ticket", headers=headers)
            assert response.status_code == 200, response.text
            assert response.headers["cache-control"] == "no-store"
            return [CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + response.json()["ticket"]]

        await client.post(base + "/test/hold-cleanup")
        async with connect("ws://worker-a:8000/ws/conv-1", subprotocols=await mint()) as old:
            await old.send(json.dumps({"type": "ping"}))
            assert json.loads(await asyncio.wait_for(old.recv(), 3)) == {"type": "pong"}
            async with connect("ws://worker-a:8000/ws/conv-1", subprotocols=await mint()) as new:
                await new.send(json.dumps({"type": "ping"}))
                assert json.loads(await asyncio.wait_for(new.recv(), 3)) == {"type": "pong"}
                await old.wait_closed()
                assert old.close_code == 4001
                await wait_state(client, "worker-a", lambda s: s["held_cleanup"] == 1)
                await client.post(base + "/test/release-cleanup")
                state = await wait_state(client, "worker-a", lambda s: s["completed_cleanup"] == 1)
                assert state["connections"] == ["conv-1"]
                for worker, scope, event in [("worker-a", "conv", "local"),
                                              ("worker-b", "conv", "cross-worker"),
                                              ("worker-b", "workspace", "workspace")]:
                    payload = {"type": "stream", "data": {"chunk": event}}
                    response = await client.post(f"http://{worker}:8000/test/send/{scope}", json=payload)
                    assert response.status_code == 200
                    assert json.loads(await asyncio.wait_for(new.recv(), 3)) == payload
                try:
                    await asyncio.wait_for(new.recv(), 0.2)
                    raise AssertionError("Duplicate WebSocket delivery")
                except asyncio.TimeoutError:
                    pass
                completed.append("Late old endpoint finally preserves reconnected socket, local/cross-worker/workspace delivery")
        await wait_state(client, "worker-a", lambda s: not s["connections"] and not s["workspace_convs"])
        completed.append("Current socket disconnect clears routing indexes")
        async def rejected(worker, protocols=None, path="conv-1", origin=None):
            try:
                async with connect(f"ws://{worker}:8000/ws/{path}", subprotocols=protocols, origin=origin):
                    raise AssertionError("Unauthenticated socket accepted")
            except InvalidStatus as exc:
                assert exc.response.status_code == 403

        # Ticket issued by worker A is consumed once by B using shared Redis.
        offered = await mint()
        async with connect("ws://worker-b:8000/ws/conv-1", subprotocols=offered) as authenticated:
            assert authenticated.subprotocol == CHAT_PROTOCOL
            await authenticated.send(json.dumps({"type": "ping"}))
            assert json.loads(await asyncio.wait_for(authenticated.recv(), 3)) == {"type": "pong"}
            await rejected("worker-a", offered)
            payload = {"type": "stream", "data": {"chunk": "authenticated-cross-worker"}}
            await client.post(base + "/test/send/conv", json=payload)
            assert json.loads(await asyncio.wait_for(authenticated.recv(), 3)) == payload
        completed.append("Tickets cross workers and cannot be replayed; authenticated Pub/Sub delivery preserved")
        await wait_state(client, "worker-b", lambda s: not s["connections"])

        # Two simultaneous handshakes compete for one Redis GETDEL.
        offered = await mint()
        async def race(worker):
            try:
                socket = await connect(f"ws://{worker}:8000/ws/conv-1", subprotocols=offered)
                return socket
            except InvalidStatus as exc:
                assert exc.response.status_code == 403
                return None
        racers = await asyncio.gather(race("worker-a"), race("worker-b"))
        assert sum(s is not None for s in racers) == 1
        for socket in racers:
            if socket is not None:
                await socket.send(json.dumps({"type": "ping"}))
                assert json.loads(await asyncio.wait_for(socket.recv(), 3)) == {"type": "pong"}
                await socket.close()
        completed.append("Simultaneous cross-worker handshakes admit exactly one connection")
        for worker in ("worker-a", "worker-b"):
            await wait_state(client, worker, lambda s: not s["connections"])
        await rejected("worker-a", await mint(), path="other")
        await rejected("worker-a", await mint(), origin="https://evil.example")
        completed.append("Wrong conversation and browser Origin rejected before connection effects")
        for worker in ("worker-a", "worker-b"):
            await rejected(worker)
            async with connect(f"ws://{worker}:8000/ws/conv-1", subprotocols=await mint()) as socket:
                await socket.send(json.dumps({"type": "ping"}))
                assert json.loads(await asyncio.wait_for(socket.recv(), 3)) == {"type": "pong"}
        completed.append("Mandatory authentication rejects old clients while accepting ticket clients")
        await wait_state(client, "test-entry", lambda _: True, timeout=60)
        proxy_ticket = await client.post("http://test-entry:8000/api/chat/conv-1/ws-ticket", headers=headers)
        assert proxy_ticket.status_code == 200, proxy_ticket.text
        async with connect("ws://test-entry:8000/api/ws/conv-1?client=flutter", subprotocols=[
                CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + proxy_ticket.json()["ticket"]]) as socket:
            assert socket.subprotocol == CHAT_PROTOCOL
            await socket.send(json.dumps({"type": "ping"}))
            assert json.loads(await asyncio.wait_for(socket.recv(), 3)) == {"type": "pong"}
        completed.append("Nginx API prefix rewrite preserves ticket subprotocol and Flutter client query")
        # Check all supported origins at the actual proxy boundary.
        for origin in ("https://banshengcomp.com", "https://www.banshengcomp.com", "https://servicewechat.com"):
            response = await client.post("http://test-entry:8000/api/chat/conv-1/ws-ticket", headers=headers)
            async with connect("ws://test-entry:8000/api/ws/conv-1", origin=origin, subprotocols=[
                    CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + response.json()["ticket"]]) as socket:
                await socket.send(json.dumps({"type": "ping"}))
                assert json.loads(await asyncio.wait_for(socket.recv(), 3)) == {"type": "pong"}
        completed.append("Web/H5/WeChat exact origins accepted through Nginx with authenticated tickets")

    print(json.dumps({"passed": len(completed), "checks": completed,
                      "scope": "Real HTTP/WS endpoints and Redis; synthetic DB/AI; two processes"}))


if __name__ == "__main__":
    asyncio.run(main())
