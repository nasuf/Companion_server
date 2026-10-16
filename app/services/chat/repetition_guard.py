"""Phrases the AI has already repeated across its recent replies.

The main reply sees its own last few replies in the history and tends to copy
them: on 2026-10-15 it ended four replies in a row with "你今天还忙吗", until
the user said it sounded like an AI. Phrases that already appear in two or more
of the recent replies are listed back to the model so it can avoid a third.
"""

from __future__ import annotations

import re

_RECENT_REPLIES = 6
_MIN_REPLIES = 2
_MAX_PHRASES = 3
_MIN_PHRASE_CHARS = 4
_CLAUSE_SPLIT = re.compile(r"\|\||[。！？!?，,；;~～…\s]+")
_NON_WORD = re.compile(r"[\W_]+")


def _recent_ai_replies(messages: list[dict]) -> list[str]:
    replies: list[list[str]] = []
    previous_role = None
    for message in messages:
        role = message.get("role")
        text = str(message.get("content") or "").strip()
        if role == "assistant" and text:
            if previous_role == "assistant" and replies:
                replies[-1].append(text)
            else:
                replies.append([text])
        if text:
            previous_role = role
    return [" ".join(parts) for parts in replies[-_RECENT_REPLIES:]]


def find_repeated_ai_phrases(messages: list[dict]) -> list[str]:
    """Clauses found in at least two of the AI's last few replies, most repeated first."""
    seen_in: dict[str, set[int]] = {}
    display: dict[str, str] = {}
    for index, reply in enumerate(_recent_ai_replies(messages)):
        for clause in _CLAUSE_SPLIT.split(reply):
            key = _NON_WORD.sub("", clause)
            # Laughter and filler ("哈哈哈哈", "嗯嗯嗯嗯") repeat naturally.
            if len(key) < _MIN_PHRASE_CHARS or len(set(key)) <= 2:
                continue
            seen_in.setdefault(key, set()).add(index)
            display[key] = clause.strip()
    repeated = [key for key, replies in seen_in.items() if len(replies) >= _MIN_REPLIES]
    repeated.sort(key=lambda key: (-len(seen_in[key]), -max(seen_in[key])))
    return [display[key] for key in repeated[:_MAX_PHRASES]]
