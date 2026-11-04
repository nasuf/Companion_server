"""StepFun realtime websocket setup.

The API key stays on this connection. Clients never see it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

from app.services.realtime_voice.protocol import history_item_event, session_update_event

logger = logging.getLogger(__name__)


class StepfunSetupError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class StepfunSocket:
    """Thin wrapper around the raw websocket with JSON send/recv and send lock."""

    def __init__(self, websocket) -> None:
        self._ws = websocket
        self._send_lock = asyncio.Lock()

    async def send_event(self, event: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._ws.send(json.dumps(event, ensure_ascii=False))

    async def recv_event(self) -> dict[str, Any]:
        raw = await self._ws.recv()
        if isinstance(raw, bytes):
            raise StepfunSetupError("provider sent binary frame outside an event")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StepfunSetupError("provider sent invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
            raise StepfunSetupError("provider sent an event without a type")
        return payload

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict[str, Any]:
        from websockets.exceptions import ConnectionClosed

        try:
            return await self.recv_event()
        except ConnectionClosed as exc:
            raise StopAsyncIteration from exc


def connect_stepfun(url: str, api_key: str):
    """Return a websockets async context manager for the StepFun endpoint."""
    from websockets.asyncio.client import connect

    headers = {"Authorization": f"Bearer {api_key}"}
    kwargs = {
        "open_timeout": 10,
        "close_timeout": 3,
        "max_size": 8_000_000,
    }
    try:
        return connect(url, additional_headers=headers, **kwargs)
    except TypeError:
        return connect(url, extra_headers=headers, **kwargs)


def realtime_url(base_url: str, model: str) -> str:
    base = base_url.strip().rstrip("/")
    return base + "?" + urlencode({"model": model})


async def configure_session(
    sock: StepfunSocket,
    *,
    instructions: str,
    voice: str,
    history: list[dict[str, str]],
    silence_duration_ms: int = 600,
    prefix_padding_ms: int = 500,
    enable_memory_tool: bool = True,
) -> None:
    """Wait for session.created, send config, seed conversation history."""
    await _wait_until(sock, "session.created", 8)
    await sock.send_event(session_update_event(
        instructions=instructions,
        voice=voice,
        silence_duration_ms=silence_duration_ms,
        prefix_padding_ms=prefix_padding_ms,
        enable_memory_tool=enable_memory_tool,
    ))
    await _wait_until(sock, "session.updated", 8)
    if not history:
        return
    pending: list[str] = []
    for index, turn in enumerate(history):
        item_id = f"hist_{index}"
        pending.append(item_id)
        await sock.send_event(history_item_event(
            item_id=item_id,
            role=turn["role"],
            text=turn["text"],
        ))
    deadline = time.monotonic() + 3
    while pending and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            event = await asyncio.wait_for(sock.recv_event(), timeout=max(0.1, remaining))
        except TimeoutError:
            break
        if event.get("type") == "error":
            logger.warning("[realtime-voice] history seed returned an error: %s", _error_message(event))
            break
        if event.get("type") != "conversation.item.created":
            continue
        item = event.get("item")
        item_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(item_id, str) and item_id in pending:
            pending.remove(item_id)
    if pending:
        logger.warning("[realtime-voice] history seed incomplete, remaining=%s", len(pending))


async def _wait_until(sock: StepfunSocket, event_type: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            event = await asyncio.wait_for(sock.recv_event(), timeout=max(0.1, remaining))
        except TimeoutError as exc:
            raise StepfunSetupError(f"timed out waiting for {event_type}") from exc
        if event.get("type") == event_type:
            return
        if event.get("type") == "error":
            raise StepfunSetupError(_error_message(event))
    raise StepfunSetupError(f"timed out waiting for {event_type}")


def _error_message(event: dict) -> str:
    error = event.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"][:180]
    return "provider rejected the session"
