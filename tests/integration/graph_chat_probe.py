"""Authenticated HTTP/ticket/WS/Redis → production stream adapter → real graph."""

import asyncio
import json
import time

import httpx
from websockets.asyncio.client import connect

from app.services.auth import create_jwt
from app.services.runtime.ws_auth import CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX


async def main():
    completed = []
    headers = {"Authorization": "Bearer " + create_jwt("owner", "user")}
    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(60):
            try:
                response = await client.get("http://test-entry:8000/test/state")
                if response.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            await asyncio.sleep(1)
        else:
            raise AssertionError("Test service did not start")
        response = await client.post(
            "http://test-entry:8000/api/chat/conv-1/ws-ticket", headers=headers
        )
        response.raise_for_status()
        ticket = response.json()["ticket"]
        async with connect(
            "ws://test-entry:8000/api/ws/conv-1?client=flutter",
            subprotocols=[CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + ticket],
        ) as socket:
            await socket.send(json.dumps({"type": "ping"}))
            assert json.loads(await socket.recv()) == {"type": "pong"}
            completed.append(
                "Authenticated Flutter connection through Nginx and Redis tickets"
            )
            for host in ("worker-a", "worker-b"):
                request = asyncio.create_task(
                    client.post(f"http://{host}:8000/test/generate", json={})
                )
                frames = [
                    json.loads(await asyncio.wait_for(socket.recv(), 10))
                    for _ in range(2)
                ]
                result = await request
                result.raise_for_status()
                assert [frame["type"] for frame in frames] == ["reply", "done"], frames
                assert frames[0]["data"]["text"] == "ordinary reply"
                assert frames[0]["data"]["index"] == 0
                assert result.json() == {
                    "saves": 1,
                    "finish": 1,
                    "background": 1,
                    "turn_achievement": 1,
                }
                completed.append(
                    f"{host}: graph persists once and delivers through Redis to the active WebSocket"
                )
            response = await client.post(
                "http://worker-a:8000/test/generate", json={"fail_save": True}
            )
            assert response.status_code == 500
            # Reply delivery precedes persistence; failure must never emit done.
            frame = json.loads(await asyncio.wait_for(socket.recv(), 5))
            assert frame["type"] == "reply"
            try:
                await asyncio.wait_for(socket.recv(), 0.5)
                raise AssertionError(
                    "Failed persistence emitted a terminal success frame"
                )
            except asyncio.TimeoutError:
                pass
            completed.append(
                "Persistence failure emits no done and does not replay the legacy executor"
            )
        base = "http://worker-a:8000"
        await client.post(base + "/test/graph-control", json={"reset": True})
        async def mint():
            result = await client.post("http://test-entry:8000/api/chat/conv-1/ws-ticket", headers=headers)
            result.raise_for_status()
            return [CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + result.json()["ticket"]]

        async def until_done(socket):
            frames = []
            for _ in range(8):
                frame = json.loads(await asyncio.wait_for(socket.recv(), 10))
                frames.append(frame)
                if frame["type"] == "done":
                    return frames
            raise AssertionError("No terminal frame in bounded message stream")

        async with connect("ws://test-entry:8000/api/ws/conv-1?client=flutter", subprotocols=await mint()) as socket:
            await socket.send(json.dumps({"type": "message", "data": {"message": "actual websocket ingress", "client_id": "ingress-1"}}))
            frames = await until_done(socket)
            assert sum(f["type"] == "ack" for f in frames) == 1, frames
            assert [f["data"]["text"] for f in frames if f["type"] == "reply"] == ["ordinary reply"], frames
            state = (await client.post(base + "/test/graph-control", json={})).json()
            assert {k: state[k] for k in ("saves", "finish", "background", "turn_achievement", "user_messages")} == {
                "saves": 1, "finish": 1, "background": 1, "turn_achievement": 1, "user_messages": 1}, state
            completed.append("Actual authenticated WS message ingress persists user once and runs the graph to done")

        await client.post(base + "/test/hold-cleanup")
        await client.post(base + "/test/graph-control", json={"reset": True, "hold": True})
        async with connect("ws://test-entry:8000/api/ws/conv-1", subprotocols=await mint()) as old:
            await old.send(json.dumps({"type": "message", "data": {"message": "reconnect during generation", "client_id": "ingress-2"}}))
            deadline = time.monotonic() + 10
            while True:
                state = (await client.post(base + "/test/graph-control", json={})).json()
                if state["generating"]:
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError("Graph generation did not reach hold point")
                await asyncio.sleep(0.05)
            async with connect("ws://test-entry:8000/api/ws/conv-1", subprotocols=await mint()) as new:
                await new.send(json.dumps({"type": "ping"}))
                assert json.loads(await asyncio.wait_for(new.recv(), 3)) == {"type": "pong"}
                await client.post(base + "/test/graph-control", json={"release": True})
                frames = await until_done(new)
                assert [f["data"]["text"] for f in frames if f["type"] == "reply"] == ["ordinary reply"]
                await client.post(base + "/test/release-cleanup")
                await asyncio.sleep(0.1)
                await new.send(json.dumps({"type": "ping"}))
                assert json.loads(await asyncio.wait_for(new.recv(), 3)) == {"type": "pong"}
                state = (await client.post(base + "/test/graph-control", json={})).json()
                assert state["saves"] == state["finish"] == state["background"] == state["turn_achievement"] == 1
                assert state["user_messages"] == 2
                completed.append("In-flight graph reconnect delivers reply/done to the new socket and stale cleanup preserves it")
    print(
        json.dumps(
            {
                "passed": len(completed),
                "checks": completed,
                "scope": "Real LangGraph, production WS stream adapter, HTTP/Nginx/Redis; synthetic domain IO",
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
