"""Test-only ASGI harness: production routes + real Redis; synthetic DB/AI.

Only mounted by scripts/test_chat_safety_e2e.py into an internal Docker network.
Never register these control endpoints in app.main.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

# Imports of the real chat route are substantial on a cold two-core CI runner.
# Keep a startup breadcrumb even before Uvicorn can log its lifespan startup.
print("chat safety harness imports started", flush=True)

from fastapi import FastAPI
from redis.asyncio import Redis

from app.api import deps, ownership
from app.api.public import chat
from app.api.realtime import ws
from app.config import settings
from app.services.runtime import ws_manager, ws_auth


redis = Redis.from_url(settings.redis_url, decode_responses=True)
manager = ws_manager.ConnectionManager()
cleanup_release = asyncio.Event()
cleanup_release.set()
held_cleanup = 0
completed_cleanup = 0
effects = {"messages": 0, "history": 0, "trigger": 0}
agent = SimpleNamespace(id="agent-1", userId="owner", name="Synthetic", status="active")
conv = SimpleNamespace(id="conv-1", userId="owner", agent=agent,
                       isDeleted=False, workspaceId="workspace-1")


async def save_message(**_):
    effects["messages"] += 1
    return SimpleNamespace(id=f"synthetic-{effects['messages']}")


async def history(*_, **__):
    effects["history"] += 1
    return [{"content": "synthetic history"}]


async def trigger(**_):
    effects["trigger"] += 1
    return {"ok": True, "message": "synthetic proactive"}


db = SimpleNamespace(
    conversation=SimpleNamespace(find_unique=AsyncMock(return_value=conv)),
    aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent)),
    user=SimpleNamespace(find_unique=AsyncMock(return_value=SimpleNamespace(username="Synthetic"))),
    message=SimpleNamespace(create=save_message),
)
chat.db = ws.db = ownership.db = db
deps.is_redis_healthy = ws.is_redis_healthy = lambda: True
chat.get_proactive_history = history
chat.send_manual_or_triggered_proactive = trigger
chat.resolve_workspace_id = AsyncMock(return_value="workspace-1")
chat.get_cached_schedule = AsyncMock(return_value=[{"activity": "free"}])
chat.get_current_status = lambda _: {"status": "idle"}
chat.build_reply_timing_context = AsyncMock(return_value={"delay_seconds": 2})
chat.plan_user_message_aggregation = AsyncMock(return_value=SimpleNamespace(
    should_wait=False, metadata={}, final_message="hello", final_context={"delay_seconds": 2}))
chat.enqueue_or_append_delayed = AsyncMock()
chat.mark_user_replied_for_conversation = AsyncMock()
chat.fire_background = ws.fire_background = lambda coroutine: coroutine.close()
ws.send_first_greeting = AsyncMock()
ws.record_ws_online = ws.remove_ws_online = AsyncMock()
ws.manager = manager


async def get_redis():
    return redis


ws_manager.get_redis = ws_auth.get_redis = get_redis
disconnect = manager.disconnect


async def hold_stale_cleanup(conv_id, *, expected=None):
    global held_cleanup, completed_cleanup
    if expected is not None and manager.get(conv_id) is not expected:
        held_cleanup += 1
        await cleanup_release.wait()
    await disconnect(conv_id, expected=expected)
    completed_cleanup += 1


manager.disconnect = hold_stale_cleanup


@asynccontextmanager
async def lifespan(_):
    await redis.ping()
    await manager.start_subscriber()
    try:
        yield
    finally:
        cleanup_release.set()
        await manager.stop_subscriber()
        await redis.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(chat.router)
app.include_router(ws.router)


@app.get("/test/state")
async def state():
    return {"connections": sorted(manager._connections), "held_cleanup": held_cleanup,
            "completed_cleanup": completed_cleanup,
            "effects": effects, "workspace_convs": {
                key: sorted(value) for key, value in manager._workspace_convs.items()}}


@app.post("/test/hold-cleanup")
async def hold_cleanup():
    cleanup_release.clear()
    return {"ok": True}


@app.post("/test/release-cleanup")
async def release_cleanup():
    cleanup_release.set()
    return {"ok": True}


@app.post("/test/send/{scope}")
async def send(scope: str, data: dict):
    if scope == "workspace":
        return {"count": await manager.send_to_workspace("workspace-1", data["type"], data["data"])}
    return {"ok": await manager.send_event("conv-1", data["type"], data["data"])}


# G01 extension: exercise the real production queue/stream adapter and graph.
# This control endpoint exists exclusively on the isolated test ASGI app.
import os
if os.environ.get("CHAT_GRAPH_E2E") == "1":
    from graph_harness_support import configure_chat
    class TestPatches:
        def setattr(self, target, name, value):
            setattr(target, name, value)
    graph_io = configure_chat(TestPatches())
    settings.chat_graph_all_conversations = os.environ.get("CHAT_GRAPH_ALL_CONVERSATIONS") == "true"
    settings.chat_graph_conversation_allowlist = "" if settings.chat_graph_all_conversations else "conv-1"
    from app.services.chat import orchestrator as graph_chat
    ws_manager.manager = manager
    graph_release = asyncio.Event()
    graph_release.set()
    graph_generating = False
    # Exercise actual WS ingress too; only domain/quota/media IO is synthetic.
    graph_io.db.conversation.find_unique.return_value = conv
    ws.wallet.is_vip = AsyncMock(return_value=True)
    ws.chat_quota.consume_one = AsyncMock(return_value={"allowed": True})
    ws.chat_media_repo.get_message_attachments = AsyncMock(return_value=[])
    ws.chat_media_repo.bind_attachments_to_message = AsyncMock()
    ws.ensure_vision_summaries = AsyncMock(return_value=[])
    ws.get_cached_schedule = AsyncMock(return_value=[{"activity": "free"}])
    ws.get_current_status = lambda _: {"activity": "free", "status": "idle"}
    ws.build_reply_timing_context = AsyncMock(return_value={"delay_seconds": 0})
    ws.mark_user_replied_for_conversation = AsyncMock()
    from app.services.offline import activity_companion
    activity_companion.note_user_interaction = AsyncMock()

    async def immediate_plan(**kwargs):
        return ws.UserMessageAggregationPlan(
            route="immediate", agent_id=kwargs["agent_id"], user_id=kwargs["user_id"],
            conversation_id=kwargs["conversation_id"], text=kwargs["text"], metadata={},
            final_message=kwargs["text"], final_context=kwargs["reply_context"],
            fallback_message=kwargs["text"], fallback_context=kwargs["reply_context"],
        )
    ws.plan_user_message_aggregation = immediate_plan

    async def controlled_reply(**_):
        global graph_generating
        graph_generating = True
        try:
            await graph_release.wait()
            return ["ordinary reply"], "ordinary reply", False, {"emotion": "中性"}
        finally:
            graph_generating = False
    graph_chat._generate_reply.side_effect = controlled_reply

    @app.post("/test/graph-control")
    async def control_graph(data: dict):
        if data.get("hold"):
            graph_release.clear()
        if data.get("release"):
            graph_release.set()
        if data.get("reset"):
            graph_chat._save_replies.reset_mock()
            graph_chat.finish_assistant_turn.reset_mock()
            graph_chat._background_post_process.reset_mock()
            graph_io.achievement.reset_mock()
            graph_chat._save_replies.return_value = "assistant-1"
        return {"generating": graph_generating, "saves": graph_chat._save_replies.await_count,
                "all_conversations": settings.chat_graph_all_conversations,
                "allowlist": settings.chat_graph_conversation_allowlist,
                "finish": graph_chat.finish_assistant_turn.await_count,
                "background": graph_chat._background_post_process.call_count,
                "turn_achievement": graph_io.achievement.call_count,
                "user_messages": effects["messages"]}

    @app.post("/test/generate")
    async def generate_graph(data: dict):
        graph_chat._save_replies.reset_mock()
        graph_chat.finish_assistant_turn.reset_mock()
        graph_chat._background_post_process.reset_mock()
        graph_io.achievement.reset_mock()
        graph_chat._save_replies.return_value = None if data.get("fail_save") else "assistant-1"
        await ws._queue_reply(
            None, conversation_id="conv-1", agent=graph_io.agent, user_id="owner",
            user_message=data.get("message", "synthetic ordinary message"),
            user_message_id="user-1", reply_context={"delay_seconds":0, "turn_message_ids":["user-1"]},
        )
        return {"saves":graph_chat._save_replies.await_count,
                "finish":graph_chat.finish_assistant_turn.await_count,
                "background":graph_chat._background_post_process.call_count,
                "turn_achievement":graph_io.achievement.call_count}
