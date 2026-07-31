from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db import db


QWEN_AUDIO_TTS_MODEL = "qwen-audio-3.0-tts-plus"
SYSTEM_VOICE_BY_GENDER = {
    "female": "longanlingxin",
    "male": "longanlufeng",
}


@dataclass(frozen=True)
class AgentTtsSettings:
    voice_id: str
    rate: float
    pitch: float
    volume: int
    seed: int
    instruction: str | None
    auto_emotion: bool
    emotion_scale: float


def _cache_voice_on_agent(agent: Any, voice_id: str) -> None:
    try:
        agent.ttsVoiceId = voice_id
    except (AttributeError, TypeError, ValueError):
        pass


async def _voice_exists(voice_id: str) -> bool:
    if not voice_id:
        return False
    rows = await db.query_raw(
        """
        SELECT 1
        FROM tts_voice_profiles
        WHERE provider = 'dashscope'
          AND model = $1
          AND voice_id = $2
        LIMIT 1
        """,
        QWEN_AUDIO_TTS_MODEL,
        voice_id,
    )
    return bool(rows)


async def assign_random_voice(
    *,
    agent_id: str,
    gender: str | None,
    agent: Any | None = None,
) -> str:
    """Pick uniformly from the enabled gender pool and persist atomically."""
    gender_key = "male" if str(gender or "").lower() == "male" else "female"
    rows = await db.query_raw(
        """
        SELECT voice_id
        FROM tts_voice_profiles
        WHERE provider = 'dashscope'
          AND model = $1
          AND gender = $2
          AND enabled = true
        ORDER BY random()
        LIMIT 1
        """,
        QWEN_AUDIO_TTS_MODEL,
        gender_key,
    )
    voice_id = (
        str(rows[0].get("voice_id") or "")
        if rows
        else SYSTEM_VOICE_BY_GENDER[gender_key]
    )
    await db.execute_raw(
        """
        UPDATE ai_agents
        SET tts_voice_id = $1, updated_at = NOW()
        WHERE id = $2
        """,
        voice_id,
        agent_id,
    )
    if agent is not None:
        _cache_voice_on_agent(agent, voice_id)
    return voice_id


async def ensure_agent_voice(agent: Any) -> str:
    """Return a Plus-compatible voice, lazily migrating old Qwen assignments."""
    agent_id = str(getattr(agent, "id"))
    rows = await db.query_raw(
        """
        SELECT tts_voice_id, gender
        FROM ai_agents
        WHERE id = $1
        LIMIT 1
        """,
        agent_id,
    )
    current = (
        str(rows[0].get("tts_voice_id") or "")
        if rows
        else str(getattr(agent, "ttsVoiceId", "") or "")
    )
    gender = (
        str(rows[0].get("gender") or "")
        if rows
        else getattr(agent, "gender", None)
    )
    if await _voice_exists(current):
        _cache_voice_on_agent(agent, current)
        return current
    return await assign_random_voice(
        agent_id=agent_id,
        gender=gender,
        agent=agent,
    )


async def get_agent_tts_settings(agent_id: str) -> AgentTtsSettings:
    """Load the latest DB values so admin edits cross workers immediately."""
    rows = await db.query_raw(
        """
        SELECT
            tts_voice_id,
            gender,
            tts_rate,
            tts_pitch,
            tts_volume,
            tts_seed,
            tts_instruction,
            tts_auto_emotion,
            tts_emotion_scale
        FROM ai_agents
        WHERE id = $1
        LIMIT 1
        """,
        agent_id,
    )
    if not rows:
        raise LookupError("Agent not found")
    row = rows[0]
    voice_id = str(row.get("tts_voice_id") or "")
    if not await _voice_exists(voice_id):
        voice_id = await assign_random_voice(
            agent_id=agent_id,
            gender=str(row.get("gender") or ""),
        )
    return AgentTtsSettings(
        voice_id=voice_id,
        rate=float(row.get("tts_rate") or 1.0),
        pitch=float(row.get("tts_pitch") or 1.0),
        volume=int(row.get("tts_volume") if row.get("tts_volume") is not None else 50),
        seed=int(row.get("tts_seed") or 0),
        instruction=(
            str(row.get("tts_instruction")).strip()
            if row.get("tts_instruction")
            else None
        ),
        auto_emotion=bool(
            row.get("tts_auto_emotion")
            if row.get("tts_auto_emotion") is not None
            else True
        ),
        emotion_scale=float(
            row.get("tts_emotion_scale")
            if row.get("tts_emotion_scale") is not None
            else 1.0
        ),
    )
