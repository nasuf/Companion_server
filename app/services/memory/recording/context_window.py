"""Bounded disambiguation history, separate from exact extraction identities."""

from app.services.chat.message_utils import (
    group_dialogue_turns, last_user_turns, TURN_BUBBLE_MAX_GAP,
)

CONTEXT_TURNS = 3
CONTEXT_MAX_ROWS = 100
CONTEXT_MAX_CHARS = 8000


def history_rows(messages: list[dict]) -> list[dict]:
    # Keep original rows/IDs. Merging is for rendering only, never source binding.
    rows = messages[-CONTEXT_MAX_ROWS:]
    # Extraction disambiguation must not merge user statements hours apart.
    # Template dialogue deliberately uses the helper's unlimited user gap.
    groups = group_dialogue_turns(rows, user_max_gap=TURN_BUBBLE_MAX_GAP)
    selected = last_user_turns(groups, CONTEXT_TURNS, lambda group: group[0].get('role'))
    return [row for group in selected for row in group]


def format_history(messages: list[dict]) -> str:
    """Use existing bubble boundaries (proactive/10min); cap context, not facts."""
    from app.services.memory.recording.chat_sources import extraction_text

    groups: list[str] = []
    for group in reversed(group_dialogue_turns(messages, user_max_gap=TURN_BUBBLE_MAX_GAP)):
        role = group[0].get("role")
        if role == "assistant":
            groups.append("assistant: " + " / ".join(extraction_text(row) for row in group))
        elif role == "user":
            # Keep the extraction template's existing per-source user lines.
            groups.extend("user: " + extraction_text(row) for row in reversed(group))
    # Prefer complete, newest groups. A single oversized group is marked partial.
    kept, size = [], 0
    for group in groups:
        if size + len(group) + 1 > CONTEXT_MAX_CHARS:
            if not kept:
                kept.append(group[:CONTEXT_MAX_CHARS - 12] + " [上下文已截断]")
            break
        kept.append(group)
        size += len(group) + 1
    return "\n".join(reversed(kept))
