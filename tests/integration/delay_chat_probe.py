"""Actual ticket handshake + two processes + Redis queue + reconnect delivery."""
import asyncio
import json
import time

import httpx
from websockets.asyncio.client import connect

from app.services.auth import create_jwt
from app.services.runtime.ws_auth import CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX
from chat_safety_probe import wait_state


async def main():
    a, b = "http://worker-a:8000", "http://worker-b:8000"
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        for worker in ("worker-a", "worker-b"):
            await wait_state(client, worker, lambda _: True, timeout=60)
        async def socket(worker):
            response = await client.post(a + "/chat/conv-1/ws-ticket", headers={"Authorization": "Bearer " + create_jwt("owner", "user")})
            assert response.status_code == 200, response.text
            return await connect(f"ws://{worker}:8000/ws/conv-1", subprotocols=[CHAT_PROTOCOL, TICKET_PROTOCOL_PREFIX + response.json()["ticket"]])
        first = await socket("worker-a")
        started = time.monotonic()
        for text, ident in [("甲", "first"), ("乙", "second")]:
            response = await client.post(a + "/test/delay-enqueue", json={"message": text, "id": ident, "delay": 2})
            assert response.status_code == 200, response.text
            event = json.loads(await asyncio.wait_for(first.recv(), 3))
            assert event["type"] == "pending" and event["data"]["status"] == "queued", event
        await first.close()
        second = await socket("worker-b")
        event = json.loads(await asyncio.wait_for(second.recv(), 3))
        assert event["type"] == "pending" and event["data"]["reconnected"] is True, event
        # An existing queue must drain even after the switch is turned off.
        await client.post(a + "/test/delay-disable")
        await client.post(b + "/test/delay-disable")
        await asyncio.sleep(max(0, 2.1 - (time.monotonic() - started)))
        results = await asyncio.gather(client.post(a + "/test/delay-scan"), client.post(b + "/test/delay-scan"))
        for response in results:
            assert response.status_code == 200, response.text
        events = []
        while not events or events[-1]["type"] != "done":
            events.append(json.loads(await asyncio.wait_for(second.recv(), 4)))
        replies = [e for e in events if e["type"] == "reply"]
        assert len(replies) == 1 and replies[0]["data"]["text"] == "甲 乙", events
        state = (await client.post(a + "/test/delay-scan")).json()
        assert len(state["generated"]) == 1, state
        try:
            extra = await asyncio.wait_for(second.recv(), .3)
            raise AssertionError("duplicate delivery: " + extra)
        except asyncio.TimeoutError:
            pass
        # Enqueue rejection still delivers once using the already persisted ID.
        response = await client.post(a + "/test/delay-enqueue", json={"message": "fallback", "id": "fallback-1", "delay": 3, "reject_enqueue": True})
        assert response.status_code == 200
        fallback_events = []
        while not fallback_events or fallback_events[-1]["type"] != "done":
            fallback_events.append(json.loads(await asyncio.wait_for(second.recv(), 4)))
        assert [e["type"] for e in fallback_events] == ["reply", "done"], fallback_events
        assert fallback_events[0]["data"]["message_id"] == "fallback-1"
        assert fallback_events[0]["data"]["save_user_message"] is False
        await second.close()
        # Models stay blocked until all independent conversations have started.
        # This would deadlock/time out with serial generation under the scan lock.
        for i in range(3):
            await client.post(a + "/test/delay-enqueue", json={"conversation_id": f"due-{i}", "message": f"held-{i}", "id": f"due-message-{i}", "delay": .01})
        await asyncio.sleep(.03)
        scan_start = time.monotonic()
        scans = await asyncio.gather(client.post(a + "/test/delay-scan"), client.post(b + "/test/delay-scan"))
        assert all(r.status_code == 200 for r in scans)
        assert time.monotonic() - scan_start < 2
        async with asyncio.timeout(4):
            while True:
                state = (await client.post(a + "/test/delay-scan")).json()
                concurrent = [json.loads(x) for x in state["started"] if json.loads(x)["conversation_id"].startswith("due-")]
                if len(concurrent) == 3:
                    break
                await asyncio.sleep(.03)
        assert len({x["conversation_id"] for x in concurrent}) == 3
        assert max(x["started_at"] for x in concurrent) - min(x["started_at"] for x in concurrent) < 1
        assert all(x["timings"][0]["due_to_start_seconds"] < 2 for x in concurrent)
        await client.post(a + "/test/delay-release")
        async with asyncio.timeout(4):
            while True:
                state = (await client.post(b + "/test/delay-scan")).json()
                delivered = [json.loads(x) for x in state["generated"] if json.loads(x)["conversation_id"].startswith("due-")]
                if len(delivered) == 3:
                    break
                await asyncio.sleep(.03)
        assert len({x["conversation_id"] for x in delivered}) == 3
    print(json.dumps({"passed": True, "concurrent_due_samples": concurrent, "checks": ["three due conversations start before any completes", "global scan lock released before model completion", "enqueue failure delivers zero-delay once without resaving", "delay merge preserves arrival order", "queued event", "authenticated reconnect restores queued state", "two workers claim once", "cross-worker reply delivered once", "disabled switch drains existing queue"], "synthetic": ["model", "message persistence", "admission fixture"], "real": ["Redis Lua", "scheduler scan", "WS tickets", "Redis pubsub", "two processes"]}))


if __name__ == "__main__":
    asyncio.run(main())
