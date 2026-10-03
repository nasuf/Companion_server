"""Network probe run inside the same isolated network as the test workers."""

import asyncio
import json
import time

import httpx
from websockets.asyncio.client import connect

from app.services.auth import create_jwt


async def wait_state(client, worker, predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            result = await client.get(f"http://{worker}:8000/test/state")
            if result.status_code == 200 and predicate(result.json()):
                return result.json()
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.05)
    raise AssertionError(f"State deadline exceeded for {worker}")


async def main():
    completed = []
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        for worker in ("worker-a", "worker-b"):
            await wait_state(client, worker, lambda _: True)
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
        await client.post(base + "/test/hold-cleanup")
        async with connect("ws://worker-a:8000/ws/conv-1") as old:
            await old.send(json.dumps({"type": "ping"}))
            assert json.loads(await asyncio.wait_for(old.recv(), 3)) == {"type": "pong"}
            async with connect("ws://worker-a:8000/ws/conv-1") as new:
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
    print(json.dumps({"passed": len(completed), "checks": completed,
                      "scope": "Real HTTP/WS endpoints and Redis; synthetic DB/AI; two processes"}))


if __name__ == "__main__":
    asyncio.run(main())
