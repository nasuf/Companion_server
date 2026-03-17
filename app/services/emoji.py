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
    pleasure: float = 0.0,
    arousal: float = 0.0,
    primary_emotion: str | None = None,
    count: int = 3,
) -> list[str]:
    """推荐表情。

    优先使用 primary_emotion 匹配，回退到 PAD 推断。
    """
    # 如果有明确情绪标签，优先使用
    if primary_emotion and primary_emotion in EMOJI_MAP:
        pool = EMOJI_MAP[primary_emotion]
        return random.sample(pool, min(count, len(pool)))

    # 基于 PAD 推断情绪类别
    if pleasure > 0.3:
        if arousal > 0.5:
            category = "高兴"
        else:
            category = "欣慰"
    elif pleasure < -0.3:
        if arousal > 0.5:
            category = "愤怒"
        else:
            category = "悲伤"
    else:
        if arousal > 0.5:
            category = "惊讶"
        else:
            category = "中性"

    pool = EMOJI_MAP.get(category, ["😊"])
    return random.sample(pool, min(count, len(pool)))


def should_add_emoji(arousal: float = 0.0) -> bool:
    """PRD §3.3.2: P = min(0.8, max(0, P_base(0~0.4) + A×0.5))"""
    p_base = random.uniform(0, 0.4)
    p_final = min(0.8, max(0, p_base + arousal * 0.5))
    return random.random() < p_final


def pick_one_emoji(
    pleasure: float = 0.0,
    arousal: float = 0.0,
    primary_emotion: str | None = None,
) -> str:
    """从推荐列表中随机选一个emoji。"""
    candidates = recommend_emoji(pleasure, arousal, primary_emotion, count=3)
    return random.choice(candidates) if candidates else ""


def should_add_sticker(arousal: float = 0.0) -> bool:
    """PRD §3.3.3: P = min(0.7, max(0, P_base(0~0.4) + A×0.4))"""
    p_base = random.uniform(0, 0.4)
    p_final = min(0.7, max(0, p_base + arousal * 0.4))
    return random.random() < p_final


