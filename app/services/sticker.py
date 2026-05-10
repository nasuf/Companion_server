"""表情包推荐服务。

根据目标情绪标签和强度从表情包库中推荐合适的表情包。
算法: emotion匹配 + intensity距离 → match_score → 过滤(≥0.3) → 随机选一个。
"""

from __future__ import annotations

import json
import random

from app.db import db

_KNOWN_EMOTIONS = {
    "高兴", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶",
    "中性", "焦虑", "失望", "欣慰", "感激", "戏谑",
}


def _label_to_intensity_bucket(intensity: int) -> int:
    """0-100 intensity -> 1-5 sticker intensity bucket."""
    clamped = max(0, min(100, int(intensity or 0)))
    return min(5, clamped // 25 + 1)


async def recommend_sticker(
    primary_emotion: str | None = None,
    intensity: int = 50,
) -> dict | None:
    """推荐一个表情包。

    Returns:
        {"id": int, "url": str, "match_score": float} 或 None
    """
    target_emotion = primary_emotion if primary_emotion in _KNOWN_EMOTIONS else "中性"
    target_intensity = _label_to_intensity_bucket(intensity)

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
