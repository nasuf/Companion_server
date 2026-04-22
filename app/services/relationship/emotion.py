"""Emotion system — PAD (Pleasure-Arousal-Dominance) model.

Spec §3.2 前置步骤：每条用户消息触发一次 `emotion.ai_pad` 小模型调用，
输入 (作息状态, 当前活动, 近期对话上下文) 输出 AI 当前 PAD。
没有持久化的 AI 情绪状态，没有衰减，没有基线，没有用户情绪融合。

`ai_emotion_states` 表/Redis `emotion:{agent_id}` 被降级为 **只读缓存**：
- 热路径每次聊天计算新 PAD 后写入缓存
- Proactive 主动消息 / GET /emotions 读接口 / 消息接收时的延迟计算从缓存读取
  （read-only 路径下不再调 LLM，避免 per-tick 成本）
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


# --- Spec §3.2 AIPAD值判断 ---

async def compute_ai_pad(
    *,
    current_time: str,
    schedule_status: str,
    current_activity: str,
    recent_context: str,
) -> dict:
    """Spec §3.2 AIPAD值判断：每条用户消息触发一次小模型调用，输出 AI 当前 PAD。

    Inputs 严格对齐 spec §3.2 四项参考信息：
      - 当前时间（UTC+8，ISO 格式）
      - 当前作息状态（空闲/忙碌/很忙碌/睡眠等）
      - 当前正在做（活动名）
      - 对话上下文（近期消息片段）

    无 prev_pad / baseline / 衰减。LLM 失败时回退到中性默认 {0.0, 0.5, 0.5}。
    """
    prompt = (await get_prompt_text("emotion.ai_pad")).format(
        current_time=current_time or "（未知）",
        current_status=schedule_status or "空闲",
        current_activity=current_activity or "自由活动",
        recent_context=recent_context or "（无）",
    )
    try:
        result = await invoke_json(get_utility_model(), prompt)
        return {
            "pleasure": _clamp_pad("pleasure", float(result.get("pleasure", _PAD_DEFAULTS["pleasure"]))),
            "arousal": _clamp_pad("arousal", float(result.get("arousal", _PAD_DEFAULTS["arousal"]))),
            "dominance": _clamp_pad("dominance", float(result.get("dominance", _PAD_DEFAULTS["dominance"]))),
        }
    except Exception as e:
        logger.warning(f"compute_ai_pad failed, falling back to neutral defaults: {e}")
        return dict(_PAD_DEFAULTS)


async def extract_emotion(message: str) -> dict:
    """Spec §3.3 + 指令模版 P26「用户PAD值判断」：只输出 PAD 三维值。"""
    model = get_utility_model()
    prompt = (await get_prompt_text("emotion.extraction")).format(message=message)

    try:
        result = await invoke_json(model, prompt)
        return {
            "pleasure": _clamp_pad("pleasure", float(result.get("pleasure", _PAD_DEFAULTS["pleasure"]))),
            "arousal": _clamp_pad("arousal", float(result.get("arousal", _PAD_DEFAULTS["arousal"]))),
            "dominance": _clamp_pad("dominance", float(result.get("dominance", _PAD_DEFAULTS["dominance"]))),
        }
    except Exception as e:
        logger.warning(f"Emotion extraction failed: {e}")
        return dict(_PAD_DEFAULTS)


def emotion_to_tone(emotion: dict) -> str:
    """Map PAD emotion to a tone descriptor string."""
    v_sign = 1 if emotion.get("pleasure", 0) >= 0 else -1
    a_sign = 1 if emotion.get("arousal", 0.5) >= 0.5 else -1
    d_sign = 1 if emotion.get("dominance", 0.5) >= 0.5 else -1
    return TONE_MAP.get((v_sign, a_sign, d_sign), "平稳而克制")


# --- Cache (ai_emotion_states) ---

async def get_ai_emotion(agent_id: str) -> dict:
    """Read last-computed AI PAD from cache.

    降级为缓存后，此函数仅用于：
      - 消息接收时的延迟计算（spec §6.2，需要 arousal 判定高情绪状态）
      - Proactive 主动消息的 mood label
      - GET /emotions/{agent_id}/current 读接口
    Chat hot path 不读此缓存，总是通过 compute_ai_pad 重新计算。
    """
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
