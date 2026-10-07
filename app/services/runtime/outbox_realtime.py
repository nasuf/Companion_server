"""Authenticated socket adapter. No implicit consumer or fallback submission."""

import logging

from app.services.runtime.sql_job_contracts import WorkerStopRequired
from app.db import db
from app.services.runtime.execution_scope import (
    bind_conversation_scope,
    ExecutionScopeUnavailable,
)
from app.services.runtime.sql_outbox import SqlOutbox


async def handle_delivery_frame(
    websocket, actor_user_id: str, conversation_id: str, frame: dict
) -> None:
    # Each frame binds current SQL authority. Session expiry is enforced by the
    # enclosing ticket-authenticated WS loop; request payloads cannot supply scope.
    try:
        scope = await bind_conversation_scope(
            actor_user_id=actor_user_id, conversation_id=conversation_id
        )
        store = SqlOutbox(db)
        if frame["type"] == "delivery_ack":
            data = frame.get("data")
            if type(data) is not dict:
                raise ValueError("Invalid delivery acknowledgement")
            await store.acknowledge(
                scope, data.get("event_id"), data.get("delivery_token")
            )
        else:

            async def send(_conversation_id, envelope):
                # Replays return to this authenticated socket, never a different
                # user's connection found through a mutable manager lookup.
                await websocket.send_json(envelope)

            await store.deliver_once("ws-replay", send, scope=scope, reconnect=True)
        await websocket.send_json(
            {
                "type": "delivery_status",
                "data": {"pending": await store.has_pending(scope), "retry_ms": 2000},
            }
        )
    except ValueError:
        await websocket.send_json(
            {"type": "delivery_error", "data": {"code": "invalid_ack"}}
        )
    except ExecutionScopeUnavailable:
        await websocket.close(code=4403, reason="conversation_access_denied")
    except WorkerStopRequired:
        raise
    except Exception:
        logging.getLogger(__name__).warning("Outbox socket operation unavailable")
        await websocket.send_json(
            {"type": "delivery_error", "data": {"code": "storage_unavailable"}}
        )
