from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db import db
from app.services.speech_output.client import TTS_PROVIDER


# Voice messages and calls share one voice per agent: both run on the
# realtime model the library is bound to, and clones target that model.
TTS_MODEL = "qwen3.8-omni-flash-realtime"
DEFAULT_SPEECH_RATE = 0.9


async def resolve_agent_tts_model() -> str:
    """The realtime model, which must be the one the voice library targets."""
    from app.config import settings
    from app.services.runtime_config import get_effective_realtime_model
    from app.services.speech_output.client import SpeechSynthesisError

    model = ((await get_effective_realtime_model()) or settings.dashscope_realtime_model).strip()
    if model != TTS_MODEL:
        raise SpeechSynthesisError(f"当前实时模型与音色库不兼容，请配置 {TTS_MODEL}")
    return model


@dataclass(frozen=True)
class AgentTtsSettings:
    voice_id: str
    rate: float
    volume: int
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
        WHERE provider = $1
          AND model = $2
          AND voice_id = $3
        LIMIT 1
        """,
        TTS_PROVIDER,
        TTS_MODEL,
        voice_id,
    )
    return bool(rows)


def _gender_key(gender: str | None) -> str:
    return "male" if str(gender or "").lower() == "male" else "female"


async def default_voice_for_gender(gender: str | None) -> str | None:
    """The admin-chosen default voice (tts_voice_profiles.is_default)."""
    rows = await db.query_raw(
        """
        SELECT voice_id
        FROM tts_voice_profiles
        WHERE provider = $1
          AND model = $2
          AND gender = $3
          AND enabled = true
          AND is_default = true
        LIMIT 1
        """,
        TTS_PROVIDER,
        TTS_MODEL,
        _gender_key(gender),
    )
    if not rows:
        return None
    return str(rows[0].get("voice_id") or "") or None


async def assign_random_voice(
    *,
    agent_id: str,
    gender: str | None,
    agent: Any | None = None,
) -> str:
    """Bind the gender default voice, or a random enabled one if none is set."""
    gender_key = _gender_key(gender)
    voice_id = await default_voice_for_gender(gender_key)
    if not voice_id:
        rows = await db.query_raw(
            """
            SELECT voice_id
            FROM tts_voice_profiles
            WHERE provider = $1
              AND model = $2
              AND gender = $3
              AND enabled = true
            ORDER BY random()
            LIMIT 1
            """,
            TTS_PROVIDER,
            TTS_MODEL,
            gender_key,
        )
        if not rows or not rows[0].get("voice_id"):
            raise LookupError(f"No enabled {gender_key} voice in the library")
        voice_id = str(rows[0]["voice_id"])
    changed = await db.execute_raw(
        """
        UPDATE ai_agents
        SET tts_voice_id = $1, updated_at = NOW()
        WHERE id = $2
          AND NOT EXISTS (
              SELECT 1 FROM tts_voice_profiles profile
              WHERE profile.provider = $3
                AND profile.model = $4
                AND profile.voice_id = ai_agents.tts_voice_id
          )
        """,
        voice_id,
        agent_id,
        TTS_PROVIDER,
        TTS_MODEL,
    )
    if not changed:
        current = await db.query_raw(
            "SELECT tts_voice_id FROM ai_agents WHERE id = $1 LIMIT 1",
            agent_id,
        )
        if not current or not current[0].get("tts_voice_id"):
            raise LookupError("Agent voice assignment failed")
        voice_id = str(current[0]["tts_voice_id"])
    if agent is not None:
        _cache_voice_on_agent(agent, voice_id)
    return voice_id


async def ensure_agent_voice(agent: Any) -> str:
    """Return a library voice, lazily replacing ids from a retired provider."""
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
            tts_volume,
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
        rate=float(row.get("tts_rate") or DEFAULT_SPEECH_RATE),
        volume=int(row.get("tts_volume") if row.get("tts_volume") is not None else 50),
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
