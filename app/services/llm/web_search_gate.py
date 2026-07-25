"""Decide whether a user message needs live web information.

Why an explicit gate instead of letting the model choose:
measured against the real production system prompt (~4.7k chars of persona +
format constraints), `doubao-seed-character` with `tool_choice: "auto"`
triggered a search in 0 of 16 trials — it answers "我帮你查下" and never calls
the tool. The same prompt with `tool_choice: "required"` searches reliably and
answers correctly. So the decision has to happen on our side, and the search
call must force the tool.

Why no keyword prefilter (removed 2026-07-25 after production traces):
the thing most worth searching is a proper noun the AI cannot know — "八仙看
过没" matches no topic keyword, so the prefilter skipped the classifier
entirely and the reply degraded to "没看过诶，讲什么的呀？". A keyword list
can never cover proper nouns, so every message above the chit-chat length
floor now goes to the classifier. It costs ~¥0.00035 per message and runs
inside the existing parallel fetch, so it adds no latency.

The classifier fails closed (no search) so this can never break a reply.
"""

from __future__ import annotations

import logging

from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompting.utils import render_prompt

logger = logging.getLogger(__name__)

# "嗯" / "在吗" / "好的" — no external entity can hide in this few characters,
# and these dominate message volume, so skipping them is free accuracy.
_MIN_LENGTH = 4


def is_worth_classifying(message: str) -> bool:
    """Length floor — the only prefilter left (see module docstring)."""
    return len((message or "").strip()) >= _MIN_LENGTH


async def needs_web_search(message: str, context: str = "") -> bool:
    """Small-model verdict: does answering this need external world info?

    The criterion is deliberately not "can the AI answer without it" — for
    "你看过 X 吗" the answer is trivially yes ("没看过"), which is a
    conversational dead end. It is "is there an external entity the AI cannot
    know, where looking it up turns a deflection into a real reply".

    Returns False on any failure — a missed search degrades to today's
    behaviour, while a wrong search costs a plugin call and ~2s of latency.
    """
    try:
        raw = await render_prompt(
            "chat.web_search_decision",
            {"message": message, "context": context or "(无)"},
            lambda p: invoke_text(get_utility_model(), p),
        )
    except Exception as e:  # noqa: BLE001 — gate must never break the reply
        logger.warning(f"[WEB-SEARCH-GATE] classify failed: {e}")
        return False
    answer = (raw or "").strip()
    # "不需要联网" contains "需要联网", so the negative must be checked first.
    if "不需要" in answer or "无需" in answer:
        return False
    return "需要联网" in answer or answer.startswith("需要")
