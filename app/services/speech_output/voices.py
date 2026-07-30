from __future__ import annotations

import hashlib
from typing import Any

from app.db import db


_VOICE_BY_PROFILE = {
    "female": {
        "EF": "Cherry",
        "ET": "Maia",
        "IF": "Serena",
        "IT": "Maia",
    },
    "male": {
        "EF": "Ethan",
        "ET": "Ethan",
        "IF": "Kai",
        "IT": "Kai",
    },
}
PREFERRED_VOICE_IDS = frozenset(
    voice
    for profiles in _VOICE_BY_PROFILE.values()
    for voice in profiles.values()
)
_LEGACY_VOICE_REPLACEMENTS = {
    "Momo": "Cherry",
    "Vivian": "Maia",
    "Moon": "Ethan",
    "Mochi": "Kai",
}


def _mbti_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or "").upper()
    return str(value or "").upper()


def _cache_voice_on_agent(agent: Any, voice_id: str) -> None:
    try:
        agent.ttsVoiceId = voice_id
    except (AttributeError, TypeError, ValueError):
        pass


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
    choices = tuple(dict.fromkeys(_VOICE_BY_PROFILE[gender_key].values()))
    digest = hashlib.sha256(stable_key.encode("utf-8")).digest()
    return choices[int.from_bytes(digest[:2], "big") % len(choices)]


async def ensure_agent_voice(agent: Any) -> str:
    """Return a stable natural voice id and migrate stylized legacy voices."""
    current = str(getattr(agent, "ttsVoiceId", "") or "")
    if current in PREFERRED_VOICE_IDS:
        return current
    agent_id = str(getattr(agent, "id"))
    voice_id = _LEGACY_VOICE_REPLACEMENTS.get(
        current,
        select_voice_id(
            gender=getattr(agent, "gender", None),
            mbti=getattr(agent, "currentMbti", None)
            or getattr(agent, "mbti", None),
            stable_key=agent_id,
        ),
    )
    if current and current not in _LEGACY_VOICE_REPLACEMENTS:
        return current
    rows = await db.query_raw(
        """
        UPDATE ai_agents
        SET tts_voice_id = $1, updated_at = NOW()
        WHERE id = $2
          AND tts_voice_id IS NOT DISTINCT FROM $3
        RETURNING tts_voice_id
        """,
        voice_id,
        agent_id,
        current or None,
    )
    if rows:
        stored = str(rows[0].get("tts_voice_id") or voice_id)
        _cache_voice_on_agent(agent, stored)
        return stored
    refreshed = await db.aiagent.find_unique(where={"id": agent_id})
    stored = str(getattr(refreshed, "ttsVoiceId", "") or "")
    if stored:
        resolved = _LEGACY_VOICE_REPLACEMENTS.get(stored, stored)
        _cache_voice_on_agent(agent, resolved)
        return resolved
    return voice_id
