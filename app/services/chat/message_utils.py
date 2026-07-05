"""聊天消息上下文的无副作用工具函数（orchestrator 拆分 R1）。

从 orchestrator.py 提取：这些函数只操作消息 dict/记录对象，无 DB/Redis/LLM
副作用，是主编排流中最独立的一层。orchestrator 通过 re-export 保持既有
导入路径兼容（tests 仍可 from orchestrator import）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _previous_assistant_message(recent_messages: list[Any], current_user_message_id: str | None) -> Any | None:
    """Find the assistant turn immediately before the current user turn."""
    current_index = len(recent_messages)
    if current_user_message_id:
        for idx, message in enumerate(recent_messages):
            if getattr(message, "id", None) == current_user_message_id:
                current_index = idx
                break
    for message in reversed(recent_messages[:current_index]):
        if getattr(message, "role", None) == "assistant":
            return message
    return None


def _current_turn_message_ids(
    reply_context: dict | None,
    current_user_message_id: str | None,
) -> set[str]:
    ids: set[str] = set()
    raw_ids = (reply_context or {}).get("turn_message_ids")
    if isinstance(raw_ids, list):
        ids.update(item for item in raw_ids if isinstance(item, str) and item)
    if current_user_message_id:
        ids.add(current_user_message_id)
    return ids


def _achievement_turn_id(turn_message_ids: set[str] | list[str]) -> str | None:
    if not turn_message_ids:
        return None
    return "user-turn:" + ",".join(sorted(turn_message_ids))


def _ensure_current_user_message(
    messages: list[dict],
    *,
    user_message: str,
    user_message_id: str | None,
    reply_context: dict | None,
) -> list[dict]:
    """Ensure the prompt context contains the current user turn.

    Normal chat saves the user message before fetching history, but delayed /
    aggregated delivery can hit visibility or payload races. If the current
    turn is absent, the main LLM sees the previous user turn as latest and can
    regenerate the prior reply. Add a synthetic current turn as a final guard.
    """
    if user_message_id:
        if any(m.get("id") == user_message_id for m in messages):
            return messages
    elif messages:
        last = messages[-1]
        if last.get("role") == "user" and last.get("content") == user_message:
            return messages

    received_at = None
    if user_message_id and reply_context:
        received_at = reply_context.get("latest_received_at") or reply_context.get("received_at")
    if user_message_id and (not isinstance(received_at, str) or not received_at):
        received_at = datetime.now(UTC).isoformat()

    return [
        *messages,
        {
            "id": user_message_id,
            "role": "user",
            "content": user_message,
            "createdAt": received_at,
            "synthetic_current": True,
        },
    ]


def collapse_turn_fragments(
    messages: list[dict],
    *,
    turn_message_ids: set[str],
    combined_text: str,
    combined_id: str | None,
) -> list[dict]:
    """Fold the current turn's fragment rows into one coherent user message.

    Fragment aggregation persists each piece ("我" / "喜欢" / "你") as a separate
    DB row, then processes them as one combined message ("我喜欢你"). Left as-is,
    the reply prompt shows the fragments as N separate user turns and the model
    may answer only the last one. This rebuilds the tail as a single user turn
    with the combined text so the LLM sees the message the user actually meant.

    Only affects the reply prompt — the memory pipeline still reads the original
    per-row messages (its watermark tracks message ids individually).

    No-op when there is 0/1 fragment or nothing combined to inject.
    """
    if len(turn_message_ids) <= 1 or not combined_text.strip():
        return messages

    kept = [m for m in messages if m.get("id") not in turn_message_ids]
    if len(kept) == len(messages):
        return messages  # none of the turn ids are in this window

    fragment_times = [
        _parse_message_created_at(m.get("createdAt"))
        for m in messages
        if m.get("id") in turn_message_ids
    ]
    latest = max((t for t in fragment_times if t is not None), default=None)
    kept.append({
        "id": combined_id,
        "role": "user",
        "content": combined_text,
        "createdAt": latest.isoformat() if latest else None,
        "coalesced_turn": True,
    })
    return kept


def _parse_message_created_at(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _max_user_created_at(messages: list[dict]) -> datetime | None:
    times = [
        ts for ts in (
            _parse_message_created_at(m.get("createdAt"))
            for m in messages
            if m.get("role") == "user"
        )
        if ts is not None
    ]
    return max(times, default=None)
