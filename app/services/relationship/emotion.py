"""Emotion system.

PAD (Pleasure-Arousal-Dominance) emotion model driven by MBTI personality.
Extracts emotion from user messages and manages AI emotion state.
"""

import logging

from app.db import db
from app.redis_client import get_redis, DEFAULT_TTL
from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompting.store import get_prompt_text
from app.services.mbti import get_mbti, signal as mbti_signal

logger = logging.getLogger(__name__)

_PAD_DIMS = ("pleasure", "arousal", "dominance")

# PAD to tone descriptor mapping
TONE_MAP = {
    (1, 1, 1): "热情而笃定",
    (1, 1, -1): "兴奋但不太踏实",
    (1, -1, 1): "平静而满足",
    (1, -1, -1): "安宁而接纳",
    (-1, 1, 1): "烦躁但强撑着",
    (-1, 1, -1): "焦虑而紧绷",
    (-1, -1, 1): "低落但克制",
    (-1, -1, -1): "难过而退缩",
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
    """Convert emotion label to PAD values (returns a copy)."""
    cn_label = _EMOTION_LABEL_MAP.get(label, label)
    entry = PAD_LABEL_TABLE.get(cn_label)
    return dict(entry) if entry else None


# --- Quick keyword emotion estimate (no LLM) ---

# Only covers high-confidence keyword-detectable emotions (5/12).
# 恐惧/惊讶/厌恶/中性/失望/欣慰/戏谑 are omitted intentionally —
# they require sentence-level context that keyword matching can't provide.
_QUICK_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "高兴": ["哈哈", "开心", "太好了", "好棒", "耶", "太开心", "好高兴"],
    "悲伤": ["难过", "伤心", "哭", "呜呜", "好难受", "心碎", "委屈", "不好", "不开心", "想哭"],
    "愤怒": ["生气", "气死", "烦死", "讨厌", "受不了", "烦", "火大", "气炸"],
    "焦虑": ["焦虑", "紧张", "担心", "害怕", "不安", "崩溃", "撑不住", "糟糕", "很累"],
    "感激": ["谢谢", "感谢", "多谢", "感恩"],
}


def quick_emotion_estimate(message: str) -> dict | None:
    """快速关键词情绪推断（无LLM），用于热路径填补当前消息情绪空缺。"""
    for label, keywords in _QUICK_EMOTION_KEYWORDS.items():
        if any(kw in message for kw in keywords):
            entry = PAD_LABEL_TABLE.get(label)
            return dict(entry) if entry else None
    return None


def pad_to_label(pad: dict) -> str:
    """Find closest emotion label for given PAD values."""
    v = pad.get("pleasure", _PAD_DEFAULTS["pleasure"])
    a = pad.get("arousal", _PAD_DEFAULTS["arousal"])
    d = pad.get("dominance", _PAD_DEFAULTS["dominance"])

    best_label = "中性"
    best_dist = float("inf")
    for label, ref in PAD_LABEL_TABLE.items():
        dist = (v - ref["pleasure"]) ** 2 + (a - ref["arousal"]) ** 2 + (d - ref["dominance"]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


_PAD_RANGES = {"pleasure": (-1.0, 1.0), "arousal": (0.0, 1.0), "dominance": (0.0, 1.0)}
_PAD_DEFAULTS = {dim: (lo + hi) / 2 for dim, (lo, hi) in _PAD_RANGES.items()}
# → {"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _clamp_pad(dim: str, value: float) -> float:
    """Clamp a PAD dimension value to its valid range."""
    lo, hi = _PAD_RANGES[dim]
    return _clamp(value, lo, hi)


def _lerp_pad(current: dict, target: dict, rate: float) -> dict:
    """Linear interpolation between two PAD dicts."""
    result = {}
    for dim in _PAD_DIMS:
        c = current.get(dim, _PAD_DEFAULTS[dim])
        t = target.get(dim, _PAD_DEFAULTS[dim])
        result[dim] = c + (t - c) * rate
    return result


# --- 3B.1 基线计算 ---

def compute_baseline_emotion(mbti: dict | None) -> dict:
    """Compute baseline PAD from MBTI using a deterministic formula.

    Used by emotion decay (every 5 min) — must be fast and deterministic.
    For initial creation, use compute_baseline_emotion_llm() instead.
    """
    lively = mbti_signal(mbti, "lively")
    rational = mbti_signal(mbti, "rational")
    emotional = mbti_signal(mbti, "emotional")
    planned = mbti_signal(mbti, "planned")
    spontaneous = mbti_signal(mbti, "spontaneous")
    creative = mbti_signal(mbti, "creative")
    humor = mbti_signal(mbti, "humor")

    p = 0.2 + (lively - 0.5) * 0.4 + (humor - 0.5) * 0.4 + (spontaneous - 0.5) * 0.2
    a = 0.5 + (lively - 0.5) * 0.3 + (creative - 0.5) * 0.3 + (humor - 0.5) * 0.2 + (emotional - 0.5) * 0.2
    d = (0.5 + (planned - 0.5) * 0.3 + (rational - 0.5) * 0.3 + (lively - 0.5) * 0.2
         + (humor - 0.5) * 0.2 - (spontaneous - 0.5) * 0.2 - (emotional - 0.5) * 0.2)

    return {
        "pleasure": _clamp_pad("pleasure", p),
        "arousal": _clamp_pad("arousal", a),
        "dominance": _clamp_pad("dominance", d),
    }


async def compute_baseline_emotion_llm(
    mbti: dict | None,
    *,
    name: str = "",
    background: str = "",
    gender: str = "",
) -> dict:
    """Compute initial baseline PAD via small model (LLM).

    Used only during agent creation. Falls back to formula on LLM failure.
    """
    prompt = (await get_prompt_text("emotion.baseline")).format(
        name=name or "AI",
        gender=gender or "未设定",
        personality=str(mbti or {}),
        background=background or "暂无",
    )

    try:
        result = await invoke_json(get_utility_model(), prompt)
        return {
            "pleasure": _clamp_pad("pleasure", float(result.get("pleasure", 0.0))),
            "arousal": _clamp_pad("arousal", float(result.get("arousal", 0.5))),
            "dominance": _clamp_pad("dominance", float(result.get("dominance", 0.5))),
        }
    except Exception as e:
        logger.warning(f"LLM baseline emotion failed, falling back to formula: {e}")
        return compute_baseline_emotion(mbti)


def compute_emotional_stability(mbti: dict | None) -> float:
    """计算情绪稳定性系数。

    spec §1.2 后用 MBTI 推导：T(理性) + J(计划) 高 → 稳定；F(感性) + P(知觉) 高 → 不稳定
        stability = 0.5 + (T-0.5)*0.4 + (J-0.5)*0.3 - (F-0.5)*0.3 - (P-0.5)*0.2
    """
    rational = mbti_signal(mbti, "rational")
    planned = mbti_signal(mbti, "planned")
    emotional = mbti_signal(mbti, "emotional")
    spontaneous = mbti_signal(mbti, "spontaneous")

    stability = (0.5 + (rational - 0.5) * 0.4 + (planned - 0.5) * 0.3
                 - (emotional - 0.5) * 0.3 - (spontaneous - 0.5) * 0.2)
    return _clamp(stability, 0.0, 1.0)


async def extract_emotion(message: str) -> dict:
    """Extract PAD emotion from a user message."""
    model = get_utility_model()
    prompt = (await get_prompt_text("emotion.extraction")).format(message=message)

    try:
        result = await invoke_json(model, prompt)
        return {
            "pleasure": _clamp_pad("pleasure", float(result.get("pleasure", _PAD_DEFAULTS["pleasure"]))),
            "arousal": _clamp_pad("arousal", float(result.get("arousal", _PAD_DEFAULTS["arousal"]))),
            "dominance": _clamp_pad("dominance", float(result.get("dominance", _PAD_DEFAULTS["dominance"]))),
            "primary_emotion": result.get("primary_emotion", "neutral"),
            "confidence": _clamp(float(result.get("confidence", 0.5)), 0.0, 1.0),
        }
    except Exception as e:
        logger.warning(f"Emotion extraction failed: {e}")
        return {**_PAD_DEFAULTS, "primary_emotion": "neutral", "confidence": 0.0}


# --- 3B.2 融合公式 + 共情向量 ---

# 亲密度阶段→融合权重 (α=AI权重, β=用户权重, γ=共情权重)
# 阶段划分复用 intimacy.RELATIONSHIP_STAGES
from app.services.relationship.intimacy import get_relationship_stage

_STAGE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "普通朋友": (0.5, 0.2, 0.3),
    "好朋友":   (0.4, 0.3, 0.3),
    "挚友":     (0.3, 0.4, 0.3),
    "灵魂伴侣": (0.2, 0.5, 0.3),
}


def _get_fusion_weights(topic_intimacy: float) -> tuple[float, float, float]:
    """Get α, β, γ fusion weights based on topic intimacy."""
    stage = get_relationship_stage(topic_intimacy)
    return _STAGE_WEIGHTS.get(stage, (0.2, 0.5, 0.3))


def update_emotion_state(
    current: dict,
    input_emotion: dict,
    topic_intimacy: float = 50.0,
    mbti: dict | None = None,
) -> dict:
    """Fuse AI emotion with user emotion using empathy vector.

    E_target = α * E_ai + β * E_user + γ * empathy_vector
    empathy_vector = (p_user * F程度, a_user * F程度, d_user * F程度)
    spec §1.2: F 程度即 MBTI 的 (100-TF)/100 信号。
    """
    alpha, beta, gamma = _get_fusion_weights(topic_intimacy)

    emotional_sensitivity = mbti_signal(mbti, "emotional")
    empathy = {
        dim: input_emotion.get(dim, _PAD_DEFAULTS[dim]) * emotional_sensitivity
        for dim in _PAD_DIMS
    }

    result = {}
    for dim in _PAD_DIMS:
        e_ai = current.get(dim, _PAD_DEFAULTS[dim])
        e_user = input_emotion.get(dim, _PAD_DEFAULTS[dim])
        e_empathy = empathy[dim]
        result[dim] = _clamp_pad(dim, alpha * e_ai + beta * e_user + gamma * e_empathy)

    return result


def emotion_to_tone(emotion: dict) -> str:
    """Map PAD emotion to a tone descriptor string."""
    v_sign = 1 if emotion.get("pleasure", 0) >= 0 else -1
    a_sign = 1 if emotion.get("arousal", 0.5) >= 0.5 else -1
    d_sign = 1 if emotion.get("dominance", 0.5) >= 0.5 else -1
    return TONE_MAP.get((v_sign, a_sign, d_sign), "平稳而克制")


async def get_ai_emotion(agent_id: str) -> dict:
    """Get current AI emotion state from DB, with Redis cache."""
    redis = await get_redis()
    cache_key = f"emotion:{agent_id}"

    cached = await redis.hgetall(cache_key)
    if cached:
        return {dim: float(cached.get(dim, _PAD_DEFAULTS[dim])) for dim in _PAD_DIMS}

    state = await db.aiemotionstate.find_unique(where={"agentId": agent_id})
    if state:
        emotion = {dim: getattr(state, dim) for dim in _PAD_DIMS}
    else:
        emotion = dict(_PAD_DEFAULTS)

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
            avg[dim] += mem_emo.get(dim, _PAD_DEFAULTS[dim])
    for dim in _PAD_DIMS:
        avg[dim] /= len(memory_emotions)

    return _lerp_pad(current_emotion, avg, influence_weight)


# --- 3B.4 情绪衰减用 stability ---

async def decay_emotion_toward_baseline(
    agent_id: str,
    mbti: dict | None = None,
) -> None:
    """Decay current emotion toward MBTI-derived baseline.

    decay_rate = 0.05 + (1 - stability) * 0.1
    """
    current = await get_ai_emotion(agent_id)
    stability = compute_emotional_stability(mbti)
    decay_rate = 0.05 + (1 - stability) * 0.1
    baseline = compute_baseline_emotion(mbti)
    decayed = _lerp_pad(current, baseline, decay_rate)
    await save_ai_emotion(agent_id, decayed)


async def save_ai_emotion(agent_id: str, emotion: dict) -> None:
    """Save AI emotion state (PAD only) to DB and cache."""
    pad = {dim: _clamp_pad(dim, emotion.get(dim, _PAD_DEFAULTS[dim])) for dim in _PAD_DIMS}

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
