from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db import db
from app.services.speech_output.client import TTS_PROVIDER


# Voice messages and realtime calls share one StepFun voice per agent; clones
# made with this model are accepted by both endpoints.
TTS_MODEL = "stepaudio-2.5-tts"
DEFAULT_SPEECH_RATE = 0.9
# Official voices verified against both stepaudio-2.5-tts and
# stepaudio-2.5-realtime. They also back realtime calls when an agent voice is
# rejected, so keep them equal to STEPFUN_REALTIME_VOICE_*.
SYSTEM_VOICE_BY_GENDER = {
    "female": "elegantgentle-female",
    "male": "wenrounansheng",
}
# Named clones are the default for every agent of that gender. The system
# voices above remain only when these profiles are missing or disabled.
DEFAULT_VOICE_DISPLAY_NAME = {
    "female": "女声-2",
    "male": "男声-2",
}


async def resolve_agent_tts_model() -> str:
    """The voice library and enrollment target one model, including clones."""
    from app.services.runtime_config import get_effective_tts_model
    from app.services.speech_output.client import SpeechSynthesisError

    model = (await get_effective_tts_model()).strip()
    if model != TTS_MODEL:
        raise SpeechSynthesisError(f"当前模型与 StepFun 音色库不兼容，请配置 {TTS_MODEL}")
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


async def assign_random_voice(
    *,
    agent_id: str,
    gender: str | None,
    agent: Any | None = None,
) -> str:
    """Bind the gender default clone, or a random enabled voice if it is absent."""
    gender_key = "male" if str(gender or "").lower() == "male" else "female"
    rows = await db.query_raw(
        """
        SELECT voice_id
        FROM tts_voice_profiles
        WHERE provider = $1
          AND model = $2
          AND gender = $3
          AND enabled = true
          AND display_name = $4
        LIMIT 1
        """,
        TTS_PROVIDER,
        TTS_MODEL,
        gender_key,
        DEFAULT_VOICE_DISPLAY_NAME[gender_key],
    )
    if not rows:
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
    voice_id = (
        str(rows[0].get("voice_id") or "")
        if rows
        else SYSTEM_VOICE_BY_GENDER[gender_key]
    )
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
