"""Emotion system.

VAD (Valence-Arousal-Dominance) emotion model.
Extracts emotion from user messages and manages AI emotion state.
"""

import logging

from app.db import db
from app.redis_client import get_redis, DEFAULT_TTL
from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompts.extraction_prompts import EMOTION_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

_VAD_DIMS = ("valence", "arousal", "dominance")

# VAD to tone descriptor mapping
TONE_MAP = {
    (1, 1, 1): "enthusiastic and confident",
    (1, 1, -1): "excited but uncertain",
    (1, -1, 1): "calm and content",
    (1, -1, -1): "peaceful and accepting",
    (-1, 1, 1): "frustrated and assertive",
    (-1, 1, -1): "anxious and stressed",
    (-1, -1, 1): "melancholic but composed",
    (-1, -1, -1): "sad and withdrawn",
}


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _lerp_vad(current: dict, target: dict, rate: float) -> dict:
    """Linear interpolation between two VAD dicts."""
    return {
        dim: current.get(dim, 0.0) + (target.get(dim, 0.0) - current.get(dim, 0.0)) * rate
        for dim in _VAD_DIMS
    }


def compute_baseline_emotion(personality: dict) -> dict:
    """Compute baseline VAD from Big Five personality traits."""
    e = personality.get("extraversion", 0.5)
    a = personality.get("agreeableness", 0.5)
    n = personality.get("neuroticism", 0.5)
    o = personality.get("openness", 0.5)
    c = personality.get("conscientiousness", 0.5)

    return {
        "valence": _clamp((e - 0.5) * 0.4 + (a - 0.5) * 0.2 + (o - 0.5) * 0.1 - (n - 0.5) * 0.3),
        "arousal": _clamp((e - 0.5) * 0.3 + (n - 0.5) * 0.4),
        "dominance": _clamp((e - 0.5) * 0.2 + (c - 0.5) * 0.3 - (n - 0.5) * 0.2),
    }


async def extract_emotion(message: str) -> dict:
    """Extract VAD emotion from a user message."""
    model = get_utility_model()
    prompt = EMOTION_EXTRACTION_PROMPT.format(message=message)

    try:
        result = await invoke_json(model, prompt)
        return {
            "valence": _clamp(float(result.get("valence", 0.0))),
            "arousal": _clamp(float(result.get("arousal", 0.0))),
            "dominance": _clamp(float(result.get("dominance", 0.0))),
            "primary_emotion": result.get("primary_emotion", "neutral"),
            "confidence": _clamp(float(result.get("confidence", 0.5)), 0.0, 1.0),
        }
    except Exception as e:
        logger.warning(f"Emotion extraction failed: {e}")
        return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0, "primary_emotion": "neutral", "confidence": 0.0}


def update_emotion_state(
    current: dict,
    input_emotion: dict,
    topic_intimacy: float = 50.0,
) -> dict:
    """Fuse AI emotion with user emotion using intimacy-weighted blend.

    E_target = α * E_ai + β * E_user
    Weights are normalized to sum to 1.0. Higher intimacy → more user influence.
    """
    intimacy_norm = _clamp(topic_intimacy / 100.0, 0.0, 1.0)

    # AI retains more at low intimacy, user gains influence at high intimacy
    alpha_raw = 0.7 - intimacy_norm * 0.2   # 0.7 → 0.5
    beta_raw = 0.3 + intimacy_norm * 0.2    # 0.3 → 0.5
    total = alpha_raw + beta_raw
    alpha = alpha_raw / total
    beta = beta_raw / total

    return {
        dim: alpha * current.get(dim, 0.0) + beta * input_emotion.get(dim, 0.0)
        for dim in _VAD_DIMS
    }


def emotion_to_tone(emotion: dict) -> str:
    """Map VAD emotion to a tone descriptor string."""
    v_sign = 1 if emotion.get("valence", 0) >= 0 else -1
    a_sign = 1 if emotion.get("arousal", 0) >= 0 else -1
    d_sign = 1 if emotion.get("dominance", 0) >= 0 else -1
    return TONE_MAP.get((v_sign, a_sign, d_sign), "neutral and balanced")


async def get_ai_emotion(agent_id: str) -> dict:
    """Get current AI emotion state from DB, with Redis cache."""
    redis = await get_redis()
    cache_key = f"emotion:{agent_id}"

    cached = await redis.hgetall(cache_key)
    if cached:
        return {dim: float(cached.get(dim, 0)) for dim in _VAD_DIMS}

    state = await db.aiemotionstate.find_unique(where={"agentId": agent_id})
    if state:
        emotion = {dim: getattr(state, dim) for dim in _VAD_DIMS}
    else:
        emotion = {dim: 0.0 for dim in _VAD_DIMS}

    await redis.hset(cache_key, mapping={k: str(v) for k, v in emotion.items()})
    await redis.expire(cache_key, DEFAULT_TTL)
    return emotion


def apply_memory_emotion_influence(
    current_emotion: dict,
    memory_emotions: list[dict],
    influence_weight: float = 0.05,
) -> dict:
    """Apply emotion influence from recalled memories."""
    if not memory_emotions:
        return current_emotion

    avg = {dim: 0.0 for dim in _VAD_DIMS}
    for mem_emo in memory_emotions:
        for dim in _VAD_DIMS:
            avg[dim] += mem_emo.get(dim, 0.0)
    for dim in _VAD_DIMS:
        avg[dim] /= len(memory_emotions)

    return _lerp_vad(current_emotion, avg, influence_weight)


async def decay_emotion_toward_baseline(agent_id: str, personality: dict | None = None) -> None:
    """Decay current emotion toward personality baseline.

    decay_rate = 0.05 + (1 - stability) * 0.1
    where stability = 1 - neuroticism
    """
    current = await get_ai_emotion(agent_id)
    personality = personality or {}
    n = personality.get("neuroticism", 0.5)
    decay_rate = 0.05 + n * 0.1  # simplified: stability = 1-n, so (1-stability)*0.1 = n*0.1

    baseline = compute_baseline_emotion(personality)
    decayed = _lerp_vad(current, baseline, decay_rate)

    await save_ai_emotion(agent_id, decayed)


async def save_ai_emotion(agent_id: str, emotion: dict) -> None:
    """Save AI emotion state (VAD only) to DB and cache."""
    vad = {dim: emotion.get(dim, 0.0) for dim in _VAD_DIMS}

    await db.aiemotionstate.upsert(
        where={"agentId": agent_id},
        data={
            "create": {
                "agent": {"connect": {"id": agent_id}},
                **vad,
            },
            "update": vad,
        },
    )

    redis = await get_redis()
    cache_key = f"emotion:{agent_id}"
    await redis.hset(cache_key, mapping={k: str(v) for k, v in vad.items()})
    await redis.expire(cache_key, DEFAULT_TTL)
