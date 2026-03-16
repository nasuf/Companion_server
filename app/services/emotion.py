"""Emotion system.

PAD (Pleasure-Arousal-Dominance) emotion model with seven-dim personality support.
Extracts emotion from user messages and manages AI emotion state.
"""

import logging

from app.db import db
from app.redis_client import get_redis, DEFAULT_TTL
from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompts.extraction_prompts import EMOTION_EXTRACTION_PROMPT
from app.services.trait_model import get_seven_dim, get_dim

logger = logging.getLogger(__name__)

_PAD_DIMS = ("pleasure", "arousal", "dominance")

# PAD to tone descriptor mapping
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
    "高兴":  {"pleasure": 0.8,  "arousal": 0.7, "dominance": 0.6},
    "悲伤":  {"pleasure": -0.6, "arousal": 0.3, "dominance": 0.2},
    "愤怒":  {"pleasure": -0.7, "arousal": 0.8, "dominance": 0.7},
    "恐惧":  {"pleasure": -0.5, "arousal": 0.8, "dominance": 0.1},
    "惊讶":  {"pleasure": 0.2,  "arousal": 0.9, "dominance": 0.3},
    "厌恶":  {"pleasure": -0.4, "arousal": 0.5, "dominance": 0.4},
    "中性":  {"pleasure": 0.0,  "arousal": 0.3, "dominance": 0.5},
    "焦虑":  {"pleasure": -0.3, "arousal": 0.7, "dominance": 0.2},
    "失望":  {"pleasure": -0.5, "arousal": 0.2, "dominance": 0.1},
    "欣慰":  {"pleasure": 0.5,  "arousal": 0.2, "dominance": 0.5},
    "感激":  {"pleasure": 0.7,  "arousal": 0.3, "dominance": 0.4},
    "戏谑":  {"pleasure": 0.6,  "arousal": 0.6, "dominance": 0.7},
}

# English-to-Chinese label map for backward compatibility
_EMOTION_LABEL_MAP = {
    "joy": "高兴", "happiness": "高兴", "happy": "高兴",
    "sadness": "悲伤", "sad": "悲伤",
    "anger": "愤怒", "angry": "愤怒",
    "fear": "恐惧", "afraid": "恐惧",
    "surprise": "惊讶", "surprised": "惊讶",
    "disgust": "厌恶",
    "neutral": "中性",
    "anxious": "焦虑", "anxiety": "焦虑",
    "disappointed": "失望", "disappointment": "失望",
    "relieved": "欣慰", "relief": "欣慰",
    "grateful": "感激", "gratitude": "感激",
    "playful": "戏谑", "teasing": "戏谑",
}


def label_to_pad(label: str) -> dict | None:
    """Convert emotion label to PAD values."""
    cn_label = _EMOTION_LABEL_MAP.get(label, label)
    return PAD_LABEL_TABLE.get(cn_label)


# --- Quick keyword emotion estimate (no LLM) ---

_QUICK_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "高兴": ["哈哈", "开心", "太好了", "好棒", "耶", "太开心", "好高兴"],
    "悲伤": ["难过", "伤心", "哭", "呜呜", "好难受", "心碎"],
    "愤怒": ["生气", "气死", "烦死", "讨厌", "受不了"],
    "焦虑": ["焦虑", "紧张", "担心", "害怕", "不安"],
    "感激": ["谢谢", "感谢", "多谢", "感恩"],
}


def quick_emotion_estimate(message: str) -> dict | None:
    """快速关键词情绪推断（无LLM），用于热路径填补当前消息情绪空缺。"""
    for label, keywords in _QUICK_EMOTION_KEYWORDS.items():
        if any(kw in message for kw in keywords):
            return PAD_LABEL_TABLE.get(label)
    return None


def pad_to_label(pad: dict) -> str:
    """Find closest emotion label for given PAD values."""
    v = pad.get("pleasure", 0)
    a = pad.get("arousal", 0)
    d = pad.get("dominance", 0)

    best_label = "中性"
    best_dist = float("inf")
    for label, ref in PAD_LABEL_TABLE.items():
        dist = (v - ref["pleasure"]) ** 2 + (a - ref["arousal"]) ** 2 + (d - ref["dominance"]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


_PAD_RANGES = {"pleasure": (-1.0, 1.0), "arousal": (0.0, 1.0), "dominance": (0.0, 1.0)}


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _clamp_pad(dim: str, value: float) -> float:
    """Clamp a PAD dimension value to its valid range."""
    lo, hi = _PAD_RANGES[dim]
    return max(lo, min(hi, value))


def _lerp_pad(current: dict, target: dict, rate: float) -> dict:
    """Linear interpolation between two PAD dicts."""
    return {
        dim: current.get(dim, 0.0) + (target.get(dim, 0.0) - current.get(dim, 0.0)) * rate
        for dim in _PAD_DIMS
    }


# --- 3B.1 基线改用七维公式 ---

def compute_baseline_emotion(personality: dict, seven_dim: dict | None = None) -> dict:
    """Compute baseline PAD from seven-dim personality traits.

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
            "pleasure": _clamp_pad("pleasure", p),
            "arousal": _clamp_pad("arousal", a),
            "dominance": _clamp_pad("dominance", d),
        }

    # Fallback: Big Five computation
    e = personality.get("extraversion", 0.5)
    a = personality.get("agreeableness", 0.5)
    n = personality.get("neuroticism", 0.5)
    o = personality.get("openness", 0.5)
    c = personality.get("conscientiousness", 0.5)

    return {
        "pleasure": _clamp_pad("pleasure", (e - 0.5) * 0.4 + (a - 0.5) * 0.2 + (o - 0.5) * 0.1 - (n - 0.5) * 0.3),
        "arousal": _clamp_pad("arousal", 0.5 + (e - 0.5) * 0.3 + (n - 0.5) * 0.4),
        "dominance": _clamp_pad("dominance", 0.5 + (e - 0.5) * 0.2 + (c - 0.5) * 0.3 - (n - 0.5) * 0.2),
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
    """Extract PAD emotion from a user message."""
    model = get_utility_model()
    prompt = EMOTION_EXTRACTION_PROMPT.format(message=message)

    try:
        result = await invoke_json(model, prompt)
        return {
            "pleasure": _clamp_pad("pleasure", float(result.get("pleasure", 0.0))),
            "arousal": _clamp_pad("arousal", float(result.get("arousal", 0.5))),
            "dominance": _clamp_pad("dominance", float(result.get("dominance", 0.5))),
            "primary_emotion": result.get("primary_emotion", "neutral"),
            "confidence": _clamp(float(result.get("confidence", 0.5)), 0.0, 1.0),
        }
    except Exception as e:
        logger.warning(f"Emotion extraction failed: {e}")
        return {"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5, "primary_emotion": "neutral", "confidence": 0.0}


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
        for dim in _PAD_DIMS
    }

    result = {}
    for dim in _PAD_DIMS:
        e_ai = current.get(dim, 0.0)
        e_user = input_emotion.get(dim, 0.0)
        e_empathy = empathy[dim]
        result[dim] = _clamp_pad(dim, alpha * e_ai + beta * e_user + gamma * e_empathy)

    return result


def emotion_to_tone(emotion: dict) -> str:
    """Map PAD emotion to a tone descriptor string."""
    v_sign = 1 if emotion.get("pleasure", 0) >= 0 else -1
    a_sign = 1 if emotion.get("arousal", 0.5) >= 0.5 else -1
    d_sign = 1 if emotion.get("dominance", 0.5) >= 0.5 else -1
    return TONE_MAP.get((v_sign, a_sign, d_sign), "neutral and balanced")


async def get_ai_emotion(agent_id: str) -> dict:
    """Get current AI emotion state from DB, with Redis cache."""
    redis = await get_redis()
    cache_key = f"emotion:{agent_id}"

    cached = await redis.hgetall(cache_key)
    if cached:
        return {dim: float(cached.get(dim, 0)) for dim in _PAD_DIMS}

    state = await db.aiemotionstate.find_unique(where={"agentId": agent_id})
    if state:
        emotion = {dim: getattr(state, dim) for dim in _PAD_DIMS}
    else:
        emotion = {"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}

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

    avg = {dim: 0.0 for dim in _PAD_DIMS}
    for mem_emo in memory_emotions:
        for dim in _PAD_DIMS:
            avg[dim] += mem_emo.get(dim, 0.0)
    for dim in _PAD_DIMS:
        avg[dim] /= len(memory_emotions)

    return _lerp_pad(current_emotion, avg, influence_weight)


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
    decayed = _lerp_pad(current, baseline, decay_rate)

    await save_ai_emotion(agent_id, decayed)


async def save_ai_emotion(agent_id: str, emotion: dict) -> None:
    """Save AI emotion state (PAD only) to DB and cache."""
    pad = {dim: _clamp_pad(dim, emotion.get(dim, 0.0)) for dim in _PAD_DIMS}

    await db.aiemotionstate.upsert(
        where={"agentId": agent_id},
        data={
            "create": {
                "agent": {"connect": {"id": agent_id}},
                **pad,
            },
            "update": pad,
        },
    )

    redis = await get_redis()
    cache_key = f"emotion:{agent_id}"
    await redis.hset(cache_key, mapping={k: str(v) for k, v in pad.items()})
    await redis.expire(cache_key, DEFAULT_TTL)
