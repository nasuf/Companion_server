"""Emotion system.

VAD (Valence-Arousal-Dominance) emotion model with seven-dim personality support.
Extracts emotion from user messages and manages AI emotion state.
"""

import logging

from app.db import db
from app.redis_client import get_redis, DEFAULT_TTL
from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompts.extraction_prompts import EMOTION_EXTRACTION_PROMPT
from app.services.trait_model import get_seven_dim, get_dim

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

# --- 3B.3 12标签 PAD 映射表 ---

PAD_LABEL_TABLE: dict[str, dict[str, float]] = {
    "快乐":  {"valence": 0.7, "arousal": 0.4, "dominance": 0.3},
    "悲伤":  {"valence": -0.7, "arousal": -0.3, "dominance": -0.5},
    "愤怒":  {"valence": -0.6, "arousal": 0.7, "dominance": 0.5},
    "恐惧":  {"valence": -0.6, "arousal": 0.6, "dominance": -0.6},
    "惊讶":  {"valence": 0.1, "arousal": 0.7, "dominance": -0.1},
    "厌恶":  {"valence": -0.5, "arousal": 0.3, "dominance": 0.3},
    "信任":  {"valence": 0.5, "arousal": -0.2, "dominance": 0.3},
    "期待":  {"valence": 0.4, "arousal": 0.3, "dominance": 0.2},
    "好奇":  {"valence": 0.3, "arousal": 0.4, "dominance": 0.1},
    "无聊":  {"valence": -0.2, "arousal": -0.5, "dominance": -0.2},
    "困惑":  {"valence": -0.2, "arousal": 0.2, "dominance": -0.4},
    "感动":  {"valence": 0.6, "arousal": 0.3, "dominance": -0.2},
}

# English-to-Chinese label map for backward compatibility
_EMOTION_LABEL_MAP = {
    "joy": "快乐", "happiness": "快乐",
    "sadness": "悲伤", "sad": "悲伤",
    "anger": "愤怒", "angry": "愤怒",
    "fear": "恐惧", "afraid": "恐惧",
    "surprise": "惊讶", "surprised": "惊讶",
    "disgust": "厌恶",
    "trust": "信任",
    "anticipation": "期待",
    "curiosity": "好奇", "curious": "好奇",
    "boredom": "无聊", "bored": "无聊",
    "confusion": "困惑", "confused": "困惑",
    "moved": "感动", "touched": "感动",
}


def label_to_vad(label: str) -> dict | None:
    """Convert emotion label to VAD values."""
    cn_label = _EMOTION_LABEL_MAP.get(label, label)
    return PAD_LABEL_TABLE.get(cn_label)


def vad_to_label(vad: dict) -> str:
    """Find closest emotion label for given VAD values."""
    v = vad.get("valence", 0)
    a = vad.get("arousal", 0)
    d = vad.get("dominance", 0)

    best_label = "快乐"
    best_dist = float("inf")
    for label, ref in PAD_LABEL_TABLE.items():
        dist = (v - ref["valence"]) ** 2 + (a - ref["arousal"]) ** 2 + (d - ref["dominance"]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _lerp_vad(current: dict, target: dict, rate: float) -> dict:
    """Linear interpolation between two VAD dicts."""
    return {
        dim: current.get(dim, 0.0) + (target.get(dim, 0.0) - current.get(dim, 0.0)) * rate
        for dim in _VAD_DIMS
    }


# --- 3B.1 基线改用七维公式 ---

def compute_baseline_emotion(personality: dict, seven_dim: dict | None = None) -> dict:
    """Compute baseline VAD from seven-dim personality traits.

    公式:
    p = 0.2 + (活泼度-0.5)*0.4 + (幽默度-0.5)*0.4 + (随性度-0.5)*0.2
    a = 0.5 + (活泼度-0.5)*0.3 + (脑洞度-0.5)*0.3 + (幽默度-0.5)*0.2 + (感性度-0.5)*0.2
    d = 0.5 + (计划度-0.5)*0.3 + (理性度-0.5)*0.3 + (活泼度-0.5)*0.2 + (幽默度-0.5)*0.2
        - (随性度-0.5)*0.2 - (感性度-0.5)*0.2
    """
    if seven_dim:
        lively = get_dim(seven_dim, "活泼度")
        rational = get_dim(seven_dim, "理性度")
        emotional = get_dim(seven_dim, "感性度")
        planned = get_dim(seven_dim, "计划度")
        spontaneous = get_dim(seven_dim, "随性度")
        creative = get_dim(seven_dim, "脑洞度")
        humor = get_dim(seven_dim, "幽默度")

        p = 0.2 + (lively - 0.5) * 0.4 + (humor - 0.5) * 0.4 + (spontaneous - 0.5) * 0.2
        a = 0.5 + (lively - 0.5) * 0.3 + (creative - 0.5) * 0.3 + (humor - 0.5) * 0.2 + (emotional - 0.5) * 0.2
        d = (0.5 + (planned - 0.5) * 0.3 + (rational - 0.5) * 0.3 + (lively - 0.5) * 0.2
             + (humor - 0.5) * 0.2 - (spontaneous - 0.5) * 0.2 - (emotional - 0.5) * 0.2)

        return {
            "valence": _clamp(p),
            "arousal": _clamp(a),
            "dominance": _clamp(d),
        }

    # Fallback: Big Five computation
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


def compute_emotional_stability(seven_dim: dict) -> float:
    """计算情绪稳定性系数。

    stability = 0.5 + (理性度-0.5)*0.4 + (计划度-0.5)*0.3 - (感性度-0.5)*0.3 - (随性度-0.5)*0.2
    """
    rational = get_dim(seven_dim, "理性度")
    planned = get_dim(seven_dim, "计划度")
    emotional = get_dim(seven_dim, "感性度")
    spontaneous = get_dim(seven_dim, "随性度")

    stability = (0.5 + (rational - 0.5) * 0.4 + (planned - 0.5) * 0.3
                 - (emotional - 0.5) * 0.3 - (spontaneous - 0.5) * 0.2)
    return _clamp(stability, 0.0, 1.0)


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


# --- 3B.2 融合公式 + 共情向量 ---

# 亲密度→权重表: growth_intimacy ranges → (α, β, γ)
_FUSION_WEIGHT_TABLE = [
    (0, 30, 0.5, 0.2, 0.3),
    (31, 60, 0.4, 0.3, 0.3),
    (61, 85, 0.3, 0.4, 0.3),
    (86, 100, 0.2, 0.5, 0.3),
]


def _get_fusion_weights(topic_intimacy: float) -> tuple[float, float, float]:
    """Get α, β, γ fusion weights based on topic intimacy."""
    for lo, hi, alpha, beta, gamma in _FUSION_WEIGHT_TABLE:
        if lo <= topic_intimacy <= hi:
            return alpha, beta, gamma
    return 0.2, 0.5, 0.3  # default high intimacy


def update_emotion_state(
    current: dict,
    input_emotion: dict,
    topic_intimacy: float = 50.0,
    seven_dim: dict | None = None,
) -> dict:
    """Fuse AI emotion with user emotion using empathy vector.

    E_target = α * E_ai + β * E_user + γ * empathy_vector
    empathy_vector = (p_user * 感性度, a_user * 感性度, d_user * 感性度)
    """
    alpha, beta, gamma = _get_fusion_weights(topic_intimacy)

    # Compute empathy vector
    emotional_sensitivity = get_dim(seven_dim, "感性度") if seven_dim else 0.5
    empathy = {
        dim: input_emotion.get(dim, 0.0) * emotional_sensitivity
        for dim in _VAD_DIMS
    }

    result = {}
    for dim in _VAD_DIMS:
        e_ai = current.get(dim, 0.0)
        e_user = input_emotion.get(dim, 0.0)
        e_empathy = empathy[dim]
        result[dim] = _clamp(alpha * e_ai + beta * e_user + gamma * e_empathy)

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


# --- 3B.5 记忆情绪权重 0.05→0.2 ---

def apply_memory_emotion_influence(
    current_emotion: dict,
    memory_emotions: list[dict],
    influence_weight: float = 0.2,
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


# --- 3B.4 情绪衰减用 stability ---

async def decay_emotion_toward_baseline(
    agent_id: str,
    personality: dict | None = None,
    seven_dim: dict | None = None,
) -> None:
    """Decay current emotion toward personality baseline.

    decay_rate = 0.05 + (1 - stability) * 0.1
    """
    current = await get_ai_emotion(agent_id)
    personality = personality or {}

    if seven_dim:
        stability = compute_emotional_stability(seven_dim)
    else:
        n = personality.get("neuroticism", 0.5)
        stability = 1.0 - n

    decay_rate = 0.05 + (1 - stability) * 0.1

    baseline = compute_baseline_emotion(personality, seven_dim)
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
