"""Bounded, timestamped recent turns shared by proactive message generators."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.schedule_domain.time_service import _TZ

RECENT_MESSAGE_LIMIT = 60  # Allows multi-bubble replies within ten user turns.


def format_recent_turns(messages: list[dict[str, Any]], *, max_turns: int = 10) -> str:
    """Chronological input; retain newest turns and never cut a role/time prefix."""
    turns: list[list[str]] = []
    for message in messages:
        role = message.get("role")
        text = " ".join(str(message.get("content") or "").split())
        if role not in {"user", "assistant"} or not text:
            continue
        stamp = message.get("created_at")
        if isinstance(stamp, str):
            try:
                stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                stamp = None
        if isinstance(stamp, datetime):
            stamp = (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).astimezone(_TZ)
            prefix = f"[{stamp:%Y-%m-%d %H:%M}] "
        else:
            prefix = "[时间未提供] "
        if role == "user" or not turns:
            turns.append([])
        turns[-1].append(f"{prefix}{'用户' if role == 'user' else 'AI'}：{text[:240]}")
    lines: list[str] = []
    size = 0
    for turn in reversed(turns[-max_turns:]):
        chunk = "\n".join(turn)
        if size + len(chunk) > 6000:
            # Very long multi-bubble turn: keep newest complete rows within budget.
            if not lines:
                for line in reversed(turn):
                    if size + len(line) + 1 > 6000:
                        break
                    lines.insert(0, line)
                    size += len(line) + 1
            break
        lines.insert(0, chunk)
        size += len(chunk) + 1
    return "\n".join(lines)
