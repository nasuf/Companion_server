"""表情包推荐服务。

PRD §5.7: 根据目标情感向量从表情包库中推荐合适的表情包。
算法: emotion匹配 + intensity距离 → match_score → 过滤(≥0.3) → 随机选一个。
"""

from __future__ import annotations

import json
import random

from app.db import db

# 12类情绪标签典型PAD值 (PRD §5.10.1)
_EMOTION_PAD: dict[str, tuple[float, float, float]] = {
    "高兴": (0.8, 0.7, 0.6),
    "悲伤": (-0.6, 0.3, 0.2),
    "愤怒": (-0.7, 0.8, 0.7),
    "恐惧": (-0.5, 0.8, 0.1),
    "惊讶": (0.2, 0.9, 0.3),
    "厌恶": (-0.4, 0.5, 0.4),
    "中性": (0.0, 0.3, 0.5),
    "焦虑": (-0.3, 0.7, 0.2),
    "失望": (-0.5, 0.2, 0.1),
    "欣慰": (0.5, 0.2, 0.5),
    "感激": (0.7, 0.3, 0.4),
    "戏谑": (0.6, 0.6, 0.7),
}


def _pad_to_emotion(p: float, a: float, d: float) -> str:
    """将PAD向量映射到最近的情绪标签（欧氏距离）。"""
    best, best_dist = "中性", float("inf")
    for label, (ep, ea, ed) in _EMOTION_PAD.items():
        dist = (p - ep) ** 2 + (a - ea) ** 2 + (d - ed) ** 2
        if dist < best_dist:
            best, best_dist = label, dist
    return best


def _arousal_to_intensity(arousal: float) -> int:
    """arousal(0~1) → intensity(1~5)。PRD §5.7.2.2。"""
    clamped = max(0.0, min(1.0, arousal))
    return min(5, int(clamped * 4) + 1)


async def recommend_sticker(
    pleasure: float = 0.0,
    arousal: float = 0.0,
    dominance: float = 0.5,
    primary_emotion: str | None = None,
) -> dict | None:
    """推荐一个表情包。

    Returns:
        {"id": int, "url": str, "match_score": float} 或 None
    """
    if primary_emotion and primary_emotion in _EMOTION_PAD:
        target_emotion = primary_emotion
    else:
        target_emotion = _pad_to_emotion(pleasure, arousal, dominance)
    target_intensity = _arousal_to_intensity(arousal)

    # 查询包含 target_emotion 的表情包（PostgreSQL jsonb 查询）
    rows = await db.query_raw(
        """
        SELECT id, url, emotion_tags, intensity
        FROM stickers
        WHERE emotion_tags::jsonb @> $1::jsonb
        """,
        json.dumps([{"emotion": target_emotion}]),
    )

    if not rows:
        return None

    # 计算 match_score 并过滤
    candidates: list[tuple[dict, float]] = []
    for row in rows:
        tags = row["emotion_tags"] if isinstance(row["emotion_tags"], list) else []
        weight = sum(
            t.get("weight", 0.5) for t in tags
            if isinstance(t, dict) and t.get("emotion") == target_emotion
        )
        score = weight * (1 - abs(row["intensity"] - target_intensity) / 5)
        if score >= 0.3:
            candidates.append((row, score))

    if not candidates:
        return None

    chosen, score = random.choice(candidates)
    return {"id": chosen["id"], "url": chosen["url"], "match_score": round(score, 2)}
