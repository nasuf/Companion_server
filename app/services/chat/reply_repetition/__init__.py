"""Keeping the main reply from repeating what the AI already said.

One place for all repetition handling on the main reply, in the order it runs:

- prompt_inputs: phrases repeated across recent replies (「避免重复」) and the
  previous reply quoted for this turn (ack / overlap / other sections);
- review.rewrite_if_restated: rewrite once when the whole draft restates the
  previous turn (rate-limited by breaker);
- review.drop_restated: drop near-verbatim single bubbles before sending.

Switches and thresholds are admin-tunable (policy); the prompt sections are
switched in prompt management.
"""
from __future__ import annotations

from app.services.chat.reply_repetition.rules import (
    find_repeated_ai_phrases,
    is_acknowledgement,
    previous_reply_for_prompt,
)


def prompt_inputs(messages: list[dict], current_turn_ids: set[str], user_message: str,
                  diagnostics: dict | None = None) -> tuple[list[str], tuple[str, str] | None]:
    """(repeated phrases, (situation, previous reply quote) | None) for build_system_prompt."""
    phrases = find_repeated_ai_phrases(messages)
    previous = previous_reply_for_prompt(messages, current_turn_ids, user_message)
    if diagnostics is not None:
        record = diagnostics.setdefault("reply_repetition", {})
        record["repeated_phrases"] = len(phrases)
        record["situation"] = previous[0] if previous else None
    return phrases, previous


__all__ = ["prompt_inputs", "is_acknowledgement"]
