from __future__ import annotations

import hashlib
from typing import Any

from app.db import db


_VOICE_BY_PROFILE = {
    "female": {
        "EF": "Momo",
        "ET": "Vivian",
        "IF": "Serena",
        "IT": "Maia",
    },
    "male": {
        "EF": "Ethan",
        "ET": "Moon",
        "IF": "Kai",
        "IT": "Mochi",
    },
}
SUPPORTED_VOICE_IDS = frozenset(
    voice
    for profiles in _VOICE_BY_PROFILE.values()
    for voice in profiles.values()
)


def _mbti_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or "").upper()
    return str(value or "").upper()


def select_voice_id(
    *,
    gender: str | None,
    mbti: Any,
    stable_key: str,
) -> str:
    """Choose one of eight curated Qwen voices deterministically."""
    gender_key = "male" if str(gender or "").lower() == "male" else "female"
    personality = _mbti_type(mbti)
    if len(personality) >= 3:
        profile = f"{personality[0]}{personality[2]}"
        if profile in _VOICE_BY_PROFILE[gender_key]:
            return _VOICE_BY_PROFILE[gender_key][profile]
    choices = tuple(_VOICE_BY_PROFILE[gender_key].values())
    digest = hashlib.sha256(stable_key.encode("utf-8")).digest()
    return choices[int.from_bytes(digest[:2], "big") % len(choices)]


async def ensure_agent_voice(agent: Any) -> str:
    """Return an immutable voice id, atomically backfilling legacy agents."""
    current = str(getattr(agent, "ttsVoiceId", "") or "")
    if current in SUPPORTED_VOICE_IDS:
        return current
    agent_id = str(getattr(agent, "id"))
    voice_id = select_voice_id(
        gender=getattr(agent, "gender", None),
        mbti=getattr(agent, "currentMbti", None) or getattr(agent, "mbti", None),
        stable_key=agent_id,
    )
    rows = await db.query_raw(
        """
        UPDATE ai_agents
        SET tts_voice_id = $1, updated_at = NOW()
        WHERE id = $2 AND tts_voice_id IS NULL
        RETURNING tts_voice_id
        """,
        voice_id,
        agent_id,
    )
    if rows:
        return str(rows[0].get("tts_voice_id") or voice_id)
    refreshed = await db.aiagent.find_unique(where={"id": agent_id})
    stored = str(getattr(refreshed, "ttsVoiceId", "") or "")
    return stored if stored in SUPPORTED_VOICE_IDS else voice_id
