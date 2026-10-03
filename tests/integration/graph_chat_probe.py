"""Authenticated HTTP/ticket/WS/Redis → production stream adapter → real graph."""

import asyncio
import json

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
            # Legacy text delivery precedes persistence; failure must never emit done.
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
