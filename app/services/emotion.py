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


def compute_baseline_emotion(personality: dict) -> dict:
    """Compute baseline VAD from Big Five personality traits.

    Maps personality dimensions to resting emotional state:
    - High extraversion → positive valence, higher arousal
    - High neuroticism → negative valence, higher arousal
    - High agreeableness → positive valence, lower dominance
    - High openness → slightly positive valence
    - High conscientiousness → higher dominance
    """
    e = personality.get("extraversion", 0.5)
    a = personality.get("agreeableness", 0.5)
    n = personality.get("neuroticism", 0.5)
    o = personality.get("openness", 0.5)
    c = personality.get("conscientiousness", 0.5)

    valence = (e - 0.5) * 0.4 + (a - 0.5) * 0.2 + (o - 0.5) * 0.1 - (n - 0.5) * 0.3
    arousal = (e - 0.5) * 0.3 + (n - 0.5) * 0.4
    dominance = (e - 0.5) * 0.2 + (c - 0.5) * 0.3 - (n - 0.5) * 0.2

    return {
        "valence": max(-1.0, min(1.0, valence)),
        "arousal": max(-1.0, min(1.0, arousal)),
        "dominance": max(-1.0, min(1.0, dominance)),
    }


async def extract_emotion(message: str) -> dict:
    """Extract VAD emotion from a user message."""
    model = get_utility_model()
    prompt = EMOTION_EXTRACTION_PROMPT.format(message=message)

    try:
        result = await invoke_json(model, prompt)
        return {
            "valence": max(-1.0, min(1.0, float(result.get("valence", 0.0)))),
            "arousal": max(-1.0, min(1.0, float(result.get("arousal", 0.0)))),
            "dominance": max(-1.0, min(1.0, float(result.get("dominance", 0.0)))),
            "primary_emotion": result.get("primary_emotion", "neutral"),
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
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

    E_target = α * E_ai + β * E_user + γ * empathy_vector
    - α, β, γ are weighted by topic_intimacy (0-100)
    - Higher intimacy → stronger empathy response
    """
    # Intimacy-based weights (normalized 0-1)
    intimacy_norm = max(0.0, min(1.0, topic_intimacy / 100.0))

    # Base weights: AI retains more control at low intimacy
    alpha = 0.7 - intimacy_norm * 0.2   # AI self: 0.7 → 0.5
    beta = 0.2 + intimacy_norm * 0.15   # User influence: 0.2 → 0.35
    gamma = 0.1 + intimacy_norm * 0.05  # Empathy: 0.1 → 0.15

    result = {}
    for dim in ("valence", "arousal", "dominance"):
        ai_val = current.get(dim, 0.0)
        user_val = input_emotion.get(dim, 0.0)

        # Empathy vector: move toward user's emotion direction
        empathy = user_val * 0.5  # damped empathy

        result[dim] = alpha * ai_val + beta * user_val + gamma * empathy

    return result


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

    # Try cache
    cached = await redis.hgetall(cache_key)
    if cached:
        return {
            "valence": float(cached.get("valence", 0)),
            "arousal": float(cached.get("arousal", 0)),
            "dominance": float(cached.get("dominance", 0)),
        }

    # From DB
    state = await db.aiemotionstate.find_unique(where={"agentId": agent_id})
    if state:
        emotion = {
            "valence": state.valence,
            "arousal": state.arousal,
            "dominance": state.dominance,
        }
    else:
        emotion = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

    # Cache
    await redis.hset(cache_key, mapping={k: str(v) for k, v in emotion.items()})
    await redis.expire(cache_key, DEFAULT_TTL)

    return emotion


def apply_memory_emotion_influence(
    current_emotion: dict,
    memory_emotions: list[dict],
    influence_weight: float = 0.05,
) -> dict:
    """Apply emotion influence from recalled memories.

    When memories with emotional content are retrieved, they subtly
    shift the AI's current emotion. Each memory contributes a small
    nudge toward its emotional valence.
    """
    if not memory_emotions:
        return current_emotion

    # Average the emotional content of recalled memories
    avg = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
    for mem_emo in memory_emotions:
        for dim in avg:
            avg[dim] += mem_emo.get(dim, 0.0)
    for dim in avg:
        avg[dim] /= len(memory_emotions)

    # Apply subtle influence
    result = {}
    for dim in ("valence", "arousal", "dominance"):
        cur = current_emotion.get(dim, 0.0)
        result[dim] = cur + (avg[dim] - cur) * influence_weight

    return result


async def decay_emotion_toward_baseline(agent_id: str, personality: dict | None = None) -> None:
    """Decay current emotion toward personality baseline.

    decay_rate = 0.05 + (1 - stability) * 0.1
    where stability = 1 - neuroticism
    """
    current = await get_ai_emotion(agent_id)
    personality = personality or {}
    n = personality.get("neuroticism", 0.5)
    stability = 1.0 - n
    decay_rate = 0.05 + (1 - stability) * 0.1

    baseline = compute_baseline_emotion(personality)

    decayed = {}
    for dim in ("valence", "arousal", "dominance"):
        cur = current.get(dim, 0.0)
        base = baseline.get(dim, 0.0)
        decayed[dim] = cur + (base - cur) * decay_rate

    await save_ai_emotion(agent_id, decayed)


async def save_ai_emotion(agent_id: str, emotion: dict) -> None:
    """Save AI emotion state to DB and cache."""
    # Update DB
    await db.aiemotionstate.upsert(
        where={"agentId": agent_id},
        data={
            "create": {
                "agent": {"connect": {"id": agent_id}},
                "valence": emotion["valence"],
                "arousal": emotion["arousal"],
                "dominance": emotion["dominance"],
            },
            "update": {
                "valence": emotion["valence"],
                "arousal": emotion["arousal"],
                "dominance": emotion["dominance"],
            },
        },
    )

    # Update cache
    redis = await get_redis()
    cache_key = f"emotion:{agent_id}"
    await redis.hset(cache_key, mapping={k: str(v) for k, v in emotion.items()})
    await redis.expire(cache_key, DEFAULT_TTL)
