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


async def should_memorize(message: str, side: Side = "user") -> bool:
    """Return True if the message is worth extracting memories from.

    Args:
        message: The text to judge.
        side: "user" → spec §2.1.2；"ai" → spec §2.2.2.

    Uses the smallest available model for speed. Expected latency: <500ms.
    On LLM failure we fail open (return True) so the big model decides.
    """
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
