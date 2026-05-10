"""表情推荐服务。

基于情绪标签匹配表情，纯计算无LLM。
"""

from __future__ import annotations

import random

# 情绪→表情映射
EMOJI_MAP: dict[str, list[str]] = {
    "高兴": ["😄", "😊", "🥰", "😁", "🎉", "✨"],
    "悲伤": ["😢", "😭", "🥺", "💔", "😞"],
    "愤怒": ["😤", "😠", "💢", "🔥"],
    "惊讶": ["😮", "😲", "🤯", "❗"],
    "恐惧": ["😨", "😰", "🫣"],
    "厌恶": ["😒", "🙄", "😑"],
    "中性": ["😊", "🙂", "👌"],
    "焦虑": ["😰", "😟", "🫤", "💭"],
    "失望": ["😞", "😔", "💔"],
    "欣慰": ["😌", "🥰", "💕"],
    "感激": ["🙏", "🥹", "💕", "❤️"],
    "戏谑": ["😏", "😜", "🤪", "😈"],
}

# 正面/负面/中性分类
POSITIVE_EMOTIONS = {"高兴", "欣慰", "感激", "戏谑"}
NEGATIVE_EMOTIONS = {"悲伤", "愤怒", "恐惧", "厌恶", "焦虑", "失望"}
NEUTRAL_EMOTIONS = {"惊讶", "中性"}


def recommend_emoji(
    primary_emotion: str | None = None,
    count: int = 3,
) -> list[str]:
    """推荐表情。优先使用显式情绪标签，未知时回退中性。"""
    pool = EMOJI_MAP.get(primary_emotion or "中性", EMOJI_MAP["中性"])
    return random.sample(pool, min(count, len(pool)))


def should_add_emoji(intensity: int = 0) -> bool:
    """Emotion intensity based emoji probability."""
    signal = max(0.0, min(1.0, intensity / 100))
    p_base = random.uniform(0, 0.4)
    p_final = min(0.8, max(0, p_base + signal * 0.5))
    return random.random() < p_final


def pick_one_emoji(
    primary_emotion: str | None = None,
) -> str:
    """从推荐列表中随机选一个emoji。"""
    candidates = recommend_emoji(primary_emotion, count=3)
    return random.choice(candidates) if candidates else ""


def should_add_sticker(intensity: int = 0) -> bool:
    """Emotion intensity based sticker probability."""
    signal = max(0.0, min(1.0, intensity / 100))
    p_base = random.uniform(0, 0.4)
    p_final = min(0.7, max(0, p_base + signal * 0.4))
    return random.random() < p_final

