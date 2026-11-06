"""DashScope Qwen-Omni realtime websocket setup.

The API key stays on this connection. Clients never see it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

from app.config import settings
from app.services.realtime_voice.protocol import history_item_event, session_update_event

logger = logging.getLogger(__name__)


class RealtimeSetupError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RealtimeSocket:
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
            raise RealtimeSetupError("provider sent binary frame outside an event")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RealtimeSetupError("provider sent invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
            raise RealtimeSetupError("provider sent an event without a type")
        return payload

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict[str, Any]:
        from websockets.exceptions import ConnectionClosed

        try:
            return await self.recv_event()
        except ConnectionClosed as exc:
            raise StopAsyncIteration from exc


def _host() -> str:
    # DashScope recommends the workspace endpoint; the shared one also works.
    workspace = settings.dashscope_workspace_id.strip()
    return f"{workspace}.cn-beijing.maas.aliyuncs.com" if workspace else "dashscope.aliyuncs.com"


def realtime_url(model: str) -> str:
    return f"wss://{_host()}/api-ws/v1/realtime?" + urlencode({"model": model})


def http_base_url() -> str:
    return f"https://{_host()}"


def connect_realtime(url: str, api_key: str):
    """Return a websockets async context manager for the DashScope endpoint."""
    from websockets.asyncio.client import connect

    headers = {"Authorization": f"Bearer {api_key}"}
    # proxy=None: the provider is reached directly, never through a proxy
    # picked up from the environment.
    kwargs = {"open_timeout": 10, "close_timeout": 3, "max_size": 16_000_000, "proxy": None}
    try:
        return connect(url, additional_headers=headers, **kwargs)
    except TypeError:
        kwargs.pop("proxy")
        return connect(url, extra_headers=headers, **kwargs)


async def configure_session(
    sock: RealtimeSocket,
    *,
    instructions: str,
    voice: str,
    history: list[dict[str, str]],
    silence_duration_ms: int = 600,
    vad_threshold: float = 0.5,
    temperature: float | None = None,
    enable_memory_tool: bool = True,
) -> None:
    """Wait for session.created, send config, seed conversation history.

    DashScope accepts any voice id here and rejects an unknown one only when
    the first reply is generated, closing the socket; callers resolve the
    voice beforehand (voices.realtime_voice_available).
    """
    await _wait_until(sock, "session.created", 8)
    await sock.send_event(session_update_event(
        instructions=instructions,
        voice=voice,
        silence_duration_ms=silence_duration_ms,
        vad_threshold=vad_threshold,
        temperature=temperature,
        enable_memory_tool=enable_memory_tool,
    ))
    await _wait_until(sock, "session.updated", 8)
    if not history:
        return
    for index, turn in enumerate(history):
        await sock.send_event(history_item_event(
            item_id=f"hist_{index}",
            role=turn["role"],
            text=turn["text"],
        ))
    # The provider assigns its own item ids, so count acknowledgements.
    remaining = len(history)
    deadline = time.monotonic() + 3
    while remaining and time.monotonic() < deadline:
        try:
            event = await asyncio.wait_for(
                sock.recv_event(), timeout=max(0.1, deadline - time.monotonic()),
            )
        except TimeoutError:
            break
        if event.get("type") == "error":
            logger.warning("[realtime-voice] history seed returned an error: %s", error_message(event))
            break
        item = event.get("item")
        if (
            event.get("type") == "conversation.item.created"
            and isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
        ):
            remaining -= 1
    if remaining:
        logger.warning("[realtime-voice] history seed incomplete, remaining=%s", remaining)


async def _wait_until(sock: RealtimeSocket, event_type: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            event = await asyncio.wait_for(sock.recv_event(), timeout=max(0.1, remaining))
        except TimeoutError as exc:
            raise RealtimeSetupError(f"timed out waiting for {event_type}") from exc
        if event.get("type") == event_type:
            return
        if event.get("type") == "error":
            raise RealtimeSetupError(error_message(event))
    raise RealtimeSetupError(f"timed out waiting for {event_type}")


def error_message(event: dict) -> str:
    error = event.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"][:180]
    return "provider rejected the session"
