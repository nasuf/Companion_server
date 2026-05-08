"""Small-model memory pre-filter (spec §2.1.2 / §2.2.2).

Runs BEFORE the expensive big-model extraction step, saving cost on messages
that are clearly not memorable.

Uses registered prompts (managed via admin UI):
- `memory.judgement_user` — Spec §2.1.2 「记忆判断」
- `memory.judgement_ai`   — Spec §2.2.2 「AI信息记忆判断」

Output per spec: plain text "记" or "不记" (not JSON).
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompting.store import get_prompt_text
from app.services.prompting.utils import SafeDict

logger = logging.getLogger(__name__)

Side = Literal["user", "ai"]

_PROMPT_KEY_BY_SIDE: dict[Side, str] = {
    "user": "memory.judgement_user",
    "ai": "memory.judgement_ai",
}

_STABLE_AI_SELF_MEMORY_PATTERNS = [
    re.compile(r"我.{0,8}(喜欢|爱|讨厌|不喜欢|不太喜欢|不太感冒|偏爱|欣赏|迷恋)"),
    re.compile(r"(承包|陪伴).{0,12}我.{0,12}(青春|童年|学生时代|小时候|成长)"),
    re.compile(r"我(以前|小时候|从前|过去|当年|青春|学生时代).{0,40}(喜欢|爱|经常|常常|总是|听|看|玩|去|吃|喝)"),
    re.compile(r"我(一直|始终|总觉得|总认为|觉得|认为).{1,40}(重要|好|不好|有意思|有感觉|值得|适合|不适合)"),
    re.compile(r"(现在|如今).{0,16}(还是|仍然|依然).{0,24}(喜欢|爱|有感觉|怀念|会想起)"),
]


def _has_stable_ai_self_memory(message: str) -> bool:
    """Detect AI-side stable preferences, long-running experiences, or views."""
    return any(pattern.search(message) for pattern in _STABLE_AI_SELF_MEMORY_PATTERNS)


async def should_memorize(message: str, side: Side = "user") -> bool:
    """Return True if the message is worth extracting memories from.

    Args:
        message: The text to judge.
        side: "user" → spec §2.1.2；"ai" → spec §2.2.2.

    Uses the smallest available model for speed. Expected latency: <500ms.
    On LLM failure we fail open (return True) so the big model decides.
    """
    if side == "ai" and _has_stable_ai_self_memory(message):
        return True

    try:
        template = await get_prompt_text(_PROMPT_KEY_BY_SIDE[side])
        prompt = template.format_map(SafeDict({"message": message}))
        raw = await invoke_text(get_utility_model(), prompt)
        decision = (raw or "").strip()
        # Spec output: plain "记" or "不记". "不记" must come first in check
        # because it contains "记".
        if "不记" in decision:
            return False
        if "记" in decision:
            return True
        # Unrecognized output → fail open
        logger.debug(f"Memory pre-filter unrecognized output ({side}): {decision[:40]!r}")
        return True
    except Exception as e:
        logger.warning(f"Memory pre-filter LLM failed ({side}): {e}")
        return True
