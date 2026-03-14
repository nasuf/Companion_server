"""表情推荐服务。

基于情绪标签匹配表情，纯计算无LLM。
"""

from __future__ import annotations

import random

# 情绪→表情映射
EMOJI_MAP: dict[str, list[str]] = {
    "快乐": ["😄", "😊", "🥰", "😁", "🎉", "✨"],
    "悲伤": ["😢", "😭", "🥺", "💔", "😞"],
    "愤怒": ["😤", "😠", "💢", "🔥"],
    "惊讶": ["😮", "😲", "🤯", "❗"],
    "恐惧": ["😨", "😰", "🫣"],
    "厌恶": ["😒", "🙄", "😑"],
    "信任": ["🤝", "💪", "👍", "❤️"],
    "期待": ["🤩", "😍", "🙏", "🎯"],
    "好奇": ["🤔", "👀", "❓", "🧐"],
    "无聊": ["😴", "🥱", "😪"],
    "困惑": ["😅", "🤷", "❓", "💭"],
    "感动": ["🥹", "😭", "💕", "🫶"],
}

# 正面/负面/中性分类
POSITIVE_EMOTIONS = {"快乐", "信任", "期待", "感动"}
NEGATIVE_EMOTIONS = {"悲伤", "愤怒", "恐惧", "厌恶"}
NEUTRAL_EMOTIONS = {"惊讶", "好奇", "无聊", "困惑"}


def recommend_emoji(
    valence: float = 0.0,
    arousal: float = 0.0,
    primary_emotion: str | None = None,
    count: int = 3,
) -> list[str]:
    """推荐表情。

    优先使用 primary_emotion 匹配，回退到 VAD 推断。
    """
    # 如果有明确情绪标签，优先使用
    if primary_emotion and primary_emotion in EMOJI_MAP:
        pool = EMOJI_MAP[primary_emotion]
        return random.sample(pool, min(count, len(pool)))

    # 基于 VAD 推断情绪类别
    if valence > 0.3:
        if arousal > 0.3:
            category = "快乐"
        else:
            category = "信任"
    elif valence < -0.3:
        if arousal > 0.3:
            category = "愤怒"
        else:
            category = "悲伤"
    else:
        if arousal > 0.3:
            category = "惊讶"
        elif arousal < -0.3:
            category = "无聊"
        else:
            category = "好奇"

    pool = EMOJI_MAP.get(category, ["😊"])
    return random.sample(pool, min(count, len(pool)))


def should_include_emoji(personality: dict, arousal: float = 0.0) -> bool:
    """判断是否应该在回复中包含表情。"""
    e = personality.get("extraversion", 0.5)
    n = personality.get("neuroticism", 0.5)
    # 外向+高情绪表达 → 更常用表情
    probability = 0.2 + e * 0.3 + n * 0.2 + abs(arousal) * 0.1
    return random.random() < probability
