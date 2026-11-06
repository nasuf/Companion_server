"""DashScope Qwen-Omni realtime event builders.

Audio: mono PCM16 little-endian, 24 kHz, both directions.
Server VAD detects the end of a turn but does not reply on its own
(create_response false): the bridge fetches memories first, then asks for
the reply. It does not cancel replies on its own either (interrupt_response
false): the bridge decides whether speech over the agent is a barge-in or
the agent's own voice leaking back into the microphone.
"""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from typing import Any

from app.config import settings

SAMPLE_RATE = 24000
MAX_CLIENT_PCM_BYTES = 256 * 1024

# ---------------------------------------------------------------------------
# recall_memory tool — the model calls this when it needs concrete facts.
# Most casual turns skip it entirely; only "需要具体事实" triggers a call.
# ---------------------------------------------------------------------------
RECALL_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": (
            "回忆与用户当前话题相关的记忆。"
            "当你需要回忆和用户之间的某件事、用户的某个偏好或事实、"
            "或者你自己的经历时调用。"
            "不需要每句话都调用，只在确实需要具体事实时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "你想回忆的内容，用一句话描述",
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def event_id() -> str:
    return "evt_" + uuid.uuid4().hex


def session_update_event(
    *,
    instructions: str,
    voice: str,
    silence_duration_ms: int = 600,
    vad_threshold: float = 0.5,
    temperature: float | None = None,
    enable_memory_tool: bool = True,
) -> dict[str, Any]:
    """Full session config: server VAD without auto reply, input transcription."""
    session: dict[str, Any] = {
        "modalities": ["text", "audio"],
        "instructions": instructions,
        "voice": voice,
        "audio": {
            "input": {"format": {"type": "pcm", "sample_rate": SAMPLE_RATE}},
            "output": {"format": {"type": "pcm", "sample_rate": SAMPLE_RATE}},
        },
        "turn_detection": {
            "type": "server_vad",
            "threshold": max(-1.0, min(1.0, float(vad_threshold))),
            "silence_duration_ms": max(200, min(6000, int(silence_duration_ms))),
            "create_response": False,
            "interrupt_response": False,
        },
        "input_audio_transcription": {"model": settings.dashscope_realtime_transcription_model},
    }
    if temperature is not None:
        session["temperature"] = max(0.0, min(1.99, float(temperature)))
    if enable_memory_tool:
        session["tools"] = [RECALL_MEMORY_TOOL]
        session["tool_choice"] = "auto"
    return {"event_id": event_id(), "type": "session.update", "session": session}


def history_item_event(*, item_id: str, role: str, text: str) -> dict[str, Any]:
    return {
        "event_id": event_id(),
        "type": "conversation.item.create",
        "item": {
            "id": item_id,
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    }


def system_note_event(text: str) -> dict[str, Any]:
    """A system message placed just before a reply.

    DashScope reads instructions changed mid-call only for later sessions,
    not the next reply (0 of 5 replies used a memory added that way, 4 of 4
    used it from a system item; 2026-11-06), so per-turn context travels as
    an item that the bridge deletes once the reply is done.
    """
    return {
        "event_id": event_id(),
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def item_delete_event(item_id: str) -> dict[str, Any]:
    return {"event_id": event_id(), "type": "conversation.item.delete", "item_id": item_id}


def append_audio_event(pcm: bytes) -> dict[str, Any]:
    return {
        "event_id": event_id(),
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode("ascii"),
    }


def commit_audio_event() -> dict[str, Any]:
    """Manual commit — used by push-to-talk or forced turn end."""
    return {"event_id": event_id(), "type": "input_audio_buffer.commit"}


def response_create_event() -> dict[str, Any]:
    return {"event_id": event_id(), "type": "response.create"}


def response_cancel_event() -> dict[str, Any]:
    return {"event_id": event_id(), "type": "response.cancel"}


def function_call_output_event(*, call_id: str, output: str) -> dict[str, Any]:
    """Return the result of a tool execution to the model."""
    return {
        "event_id": event_id(),
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        },
    }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def decode_client_pcm(raw: bytes | str | None) -> bytes | None:
    """Accept raw bytes or base64-encoded PCM16."""
    if isinstance(raw, bytes):
        pcm = raw
    elif isinstance(raw, str) and raw:
        try:
            pcm = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            return None
    else:
        return None
    if len(pcm) < 2:
        return None
    even = len(pcm) - (len(pcm) % 2)
    pcm = pcm[:even]
    if not pcm or len(pcm) > MAX_CLIENT_PCM_BYTES:
        return None
    return pcm


def parse_client_message(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    kind = payload.get("type")
    if not isinstance(kind, str) or not kind:
        return None
    return payload


def provider_item_id(event: dict[str, Any]) -> str | None:
    item = event.get("item")
    if isinstance(item, dict) and isinstance(item.get("id"), str):
        return item["id"]
    item_id = event.get("item_id")
    if isinstance(item_id, str):
        return item_id
    return None


def assistant_transcript(event: dict[str, Any]) -> str:
    """Extract text from transcript.done or response.done. Skips function_call items."""
    direct = event.get("transcript")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    text = event.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    response = event.get("response")
    parts: list[str] = []
    if isinstance(response, dict):
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                for key in ("transcript", "text"):
                    value = content.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break
    return "\n".join(parts).strip()
