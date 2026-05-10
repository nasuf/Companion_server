"""Emotion label helpers.

The runtime no longer computes three-axis emotion vectors.  Emotion handling is kept as a
coarse label plus intensity signal that is easy to reason about in prompts,
reply timing, emoji/sticker decoration, and diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

EMOTION_LABELS = (
    "高兴", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶",
    "中性", "焦虑", "失望", "欣慰", "感激", "戏谑",
)

POSITIVE_EMOTIONS = {"高兴", "欣慰", "感激", "戏谑"}
NEGATIVE_EMOTIONS = {"悲伤", "愤怒", "恐惧", "厌恶", "焦虑", "失望"}
HIGH_ENERGY_EMOTIONS = {"愤怒", "恐惧", "惊讶", "焦虑", "高兴", "戏谑"}

_TONE_BY_LABEL = {
    "高兴": "轻快而亲近",
    "悲伤": "低落但克制",
    "愤怒": "烦躁但强撑着",
    "恐惧": "不安而紧绷",
    "惊讶": "惊讶但清醒",
    "厌恶": "抗拒而克制",
    "中性": "平稳而克制",
    "焦虑": "焦虑而紧绷",
    "失望": "失落但克制",
    "欣慰": "平静而满足",
    "感激": "温和而感激",
    "戏谑": "轻松而俏皮",
}

_QUICK_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "高兴": ["哈哈", "开心", "太好了", "好棒", "耶", "太开心", "好高兴"],
    "悲伤": ["难过", "伤心", "想哭", "哭", "呜呜", "好难受", "心碎", "委屈", "不好", "不开心"],
    "愤怒": ["生气", "气死", "烦死", "讨厌", "受不了", "火大", "气炸"],
    "恐惧": ["害怕", "恐惧", "吓死", "怕死", "很怕"],
    "焦虑": ["焦虑", "紧张", "担心", "不安", "崩溃", "撑不住", "糟糕", "很累"],
    "失望": ["失望", "没意思", "算了", "白期待", "心凉"],
    "感激": ["谢谢", "感谢", "多谢", "感恩", "辛苦了"],
    "惊讶": ["啊？", "啊?", "真的假的", "不会吧", "震惊"],
    "厌恶": ["恶心", "反感", "膈应", "厌恶"],
}


def _clamp_intensity(value: Any, default: int = 50) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(0, min(100, parsed))


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def normalize_emotion_label(value: Any) -> str:
    label = str(value or "").strip()
    return label if label in EMOTION_LABELS else "中性"


def neutral_emotion(*, source: str = "fallback") -> dict[str, Any]:
    return {
        "emotion": "中性",
        "intensity": 0,
        "confidence": 0.0,
        "source": source,
    }


def quick_emotion_estimate(message: str) -> dict[str, Any] | None:
    """Fast keyword emotion estimate used before the async LLM result exists."""
    text = message or ""
    for label, keywords in _QUICK_EMOTION_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            intensity = 72 if label in HIGH_ENERGY_EMOTIONS else 58
            return {
                "emotion": label,
                "intensity": intensity,
                "confidence": 0.65,
                "source": "quick",
            }
    return None


def normalize_emotion_result(result: Any, *, source: str = "llm") -> dict[str, Any]:
    if not isinstance(result, dict):
        return neutral_emotion(source="fallback")
    return {
        "emotion": normalize_emotion_label(result.get("emotion")),
        "intensity": _clamp_intensity(result.get("intensity"), default=50),
        "confidence": _clamp_confidence(result.get("confidence"), default=0.6),
        "source": source,
    }


async def analyze_user_emotion(message: str) -> dict[str, Any]:
    """Analyze the user's current message as label + intensity.

    This replaces the former user vector extraction.  The output is deliberately
    compact and prompt-readable: downstream code should reason on the label and
    coarse intensity rather than abstract vector dimensions.
    """
    prompt = (await get_prompt_text("emotion.user_label")).format(message=message)
    try:
        result = await invoke_json(get_utility_model(), prompt)
        return normalize_emotion_result(result, source="llm")
    except Exception as e:
        logger.warning(f"emotion.user_label failed, falling back to keyword estimate: {e}")
        return quick_emotion_estimate(message) or neutral_emotion(source="fallback")


def is_negative_emotion(emotion: dict[str, Any] | None) -> bool:
    if not emotion:
        return False
    label = normalize_emotion_label(emotion.get("emotion"))
    return label in NEGATIVE_EMOTIONS and _clamp_intensity(emotion.get("intensity"), default=0) >= 35


def is_high_emotion(emotion: dict[str, Any] | None) -> bool:
    if not emotion:
        return False
    label = normalize_emotion_label(emotion.get("emotion"))
    intensity = _clamp_intensity(emotion.get("intensity"), default=0)
    return intensity >= 70 or (label in HIGH_ENERGY_EMOTIONS and intensity >= 55)


def emotion_to_tone(emotion: dict[str, Any] | None) -> str:
    label = normalize_emotion_label((emotion or {}).get("emotion"))
    return _TONE_BY_LABEL.get(label, "平稳而克制")
