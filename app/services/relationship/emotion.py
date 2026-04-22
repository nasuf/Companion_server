"""Emotion system — PAD (Pleasure-Arousal-Dominance) model.

Spec §3.2: per-message LLM 计算 AI/用户 PAD。`ai_emotion_states` 表是**只读缓存**——
hot path 每次重算并回写，proactive / GET / 延迟计算等只读路径直接读缓存。
"""

import logging

from app.db import db
from app.redis_client import get_redis, DEFAULT_TTL
from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompting.store import get_prompt_text

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

# --- Quick keyword emotion estimate (no LLM) ---

# Only covers high-confidence keyword-detectable emotions (5/12).
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


_PAD_RANGES = {"pleasure": (-1.0, 1.0), "arousal": (0.0, 1.0), "dominance": (0.0, 1.0)}
_PAD_DEFAULTS = {dim: (lo + hi) / 2 for dim, (lo, hi) in _PAD_RANGES.items()}
# → {"pleasure": 0.0, "arousal": 0.5, "dominance": 0.5}


def _clamp_pad(dim: str, value: float) -> float:
    """Clamp a PAD dimension value to its valid range."""
    lo, hi = _PAD_RANGES[dim]
    return max(lo, min(hi, value))


async def _invoke_pad(prompt_key: str, **format_args) -> dict:
    """Format a PAD prompt and invoke utility LLM; clamp 3 dims, fallback to neutral on error."""
    prompt = (await get_prompt_text(prompt_key)).format(**format_args)
    try:
        result = await invoke_json(get_utility_model(), prompt)
        return {
            dim: _clamp_pad(dim, float(result.get(dim, _PAD_DEFAULTS[dim])))
            for dim in _PAD_DIMS
        }
    except Exception as e:
        logger.warning(f"{prompt_key} failed, falling back to neutral defaults: {e}")
        return dict(_PAD_DEFAULTS)


async def compute_ai_pad(
    *,
    current_time: str,
    schedule_status: str,
    current_activity: str,
    recent_context: str,
) -> dict:
    """Spec §3.2 AIPAD值判断：4 项参考信息 → AI PAD。失败回退中性默认。"""
    return await _invoke_pad(
        "emotion.ai_pad",
        current_time=current_time or "（未知）",
        current_status=schedule_status or "空闲",
        current_activity=current_activity or "自由活动",
        recent_context=recent_context or "（无）",
    )


async def extract_emotion(message: str) -> dict:
    """Spec §3.3 + 指令模版 P26「用户PAD值判断」：只输出 PAD 三维值。"""
    return await _invoke_pad("emotion.extraction", message=message)


def emotion_to_tone(emotion: dict) -> str:
    """Map PAD emotion to a tone descriptor string."""
    v_sign = 1 if emotion.get("pleasure", 0) >= 0 else -1
    a_sign = 1 if emotion.get("arousal", 0.5) >= 0.5 else -1
    d_sign = 1 if emotion.get("dominance", 0.5) >= 0.5 else -1
    return TONE_MAP.get((v_sign, a_sign, d_sign), "平稳而克制")


# --- Cache (ai_emotion_states) ---

async def get_ai_emotion(agent_id: str) -> dict:
    """读上一轮 compute_ai_pad 回写的缓存，供 proactive / GET / 延迟计算等只读路径使用。"""
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


async def save_ai_emotion(agent_id: str, emotion: dict) -> None:
    """Write computed PAD to cache (DB + Redis) for downstream readers."""
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
