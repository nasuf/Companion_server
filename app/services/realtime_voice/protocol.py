"""StepFun realtime event builders.

Audio: mono PCM16 little-endian, 24 kHz.
Server VAD enabled — StepFun handles turn detection natively.
"""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from typing import Any

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
    prefix_padding_ms: int = 500,
    energy_threshold: int | None = None,
    enable_memory_tool: bool = True,
) -> dict[str, Any]:
    """Server VAD + optional recall_memory tool."""
    tools: list[dict[str, Any]] = []
    if enable_memory_tool:
        tools.append(RECALL_MEMORY_TOOL)
    session: dict[str, Any] = {
        "modalities": ["text", "audio"],
        "instructions": instructions,
        "voice": voice,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": {
            "type": "server_vad",
            "silence_duration_ms": silence_duration_ms,
            "prefix_padding_ms": prefix_padding_ms,
        },
    }
    if energy_threshold is not None:
        session["turn_detection"]["energy_awakeness_threshold"] = max(0, min(5000, int(energy_threshold)))
    if tools:
        session["tools"] = tools
        session["tool_choice"] = "auto"
    return {"event_id": event_id(), "type": "session.update", "session": session}


def session_instructions_event(instructions: str) -> dict[str, Any]:
    """Partial session.update: StepFun merges it, keeping tools, VAD and voice."""
    return {
        "event_id": event_id(),
        "type": "session.update",
        "session": {"instructions": instructions},
    }


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
