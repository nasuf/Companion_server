from __future__ import annotations

import random
from enum import StrEnum
from typing import Callable

from app.redis_client import get_redis
from app.services.runtime_config import get_effective_tts_probability


class VoiceContext(StrEnum):
    NORMAL_CHAT = "normal_chat"
    PROACTIVE_CHAT = "proactive_chat"
    BOUNDARY = "boundary"
    CRISIS = "crisis"
    REMINDER = "reminder"
    DELETION = "deletion"
    CURRENT_STATE = "current_state"
    COMPONENT_CARD = "component_card"
    SYSTEM = "system"


_ELIGIBLE_CONTEXTS = {
    VoiceContext.NORMAL_CHAT,
    VoiceContext.PROACTIVE_CHAT,
}


_PROBABILITY_KEY = "runtime:tts_output_probability"


async def effective_tts_probability() -> int:
    """Read the cross-worker probability, falling back to process config."""
    try:
        redis = await get_redis()
        value = await redis.get(_PROBABILITY_KEY)
        if value is not None:
            return max(0, min(100, int(value)))
    except Exception:
        pass
    return get_effective_tts_probability()


async def should_generate_voice(
    *,
    context: VoiceContext,
    client_supports_voice: bool,
    probability: int | None = None,
    random_value: Callable[[], float] = random.random,
) -> bool:
    """Decide once per assistant message whether speech should be generated."""
    if context not in _ELIGIBLE_CONTEXTS or not client_supports_voice:
        return False
    chance = (
        await effective_tts_probability()
        if probability is None
        else max(0, min(100, int(probability)))
    )
    if chance <= 0:
        return False
    if chance >= 100:
        return True
    return random_value() < chance / 100.0
