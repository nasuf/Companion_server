"""表情包推荐服务。

根据目标情绪标签和强度从表情包库中推荐合适的表情包。
算法: emotion匹配 + intensity距离 → match_score → 过滤(≥0.3) →
排除最近用过的(跨轮去重, W6 与 emoji 对齐) → 随机选一个。
"""

from __future__ import annotations

import json
import logging
import random

from app.db import db
from app.services.runtime.recent_items import load_recent_items, remember_item

logger = logging.getLogger(__name__)

_KNOWN_EMOTIONS = {
    "高兴", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶",
    "中性", "焦虑", "失望", "欣慰", "感激", "戏谑",
}

# W6 跨轮去重 (与 reply_post_process 的 emoji:recent:* 同模式):
# 多轮同情绪对话下 random.choice 可能连续抽中同一张表情包, 真人不会这样.
_RECENT_STICKER_KEY = "sticker:recent:{conversation_id}"
_RECENT_STICKER_KEEP = 2


async def _load_recent_sticker_ids(conversation_id: str | None) -> set[str]:
    """读最近用过的 sticker id。公共实现见 runtime/recent_items.py。"""
    if not conversation_id:
        return set()
    return await load_recent_items(
        _RECENT_STICKER_KEY.format(conversation_id=conversation_id),
        _RECENT_STICKER_KEEP,
    )


async def _remember_sticker(conversation_id: str | None, sticker_id: object) -> None:
    if not conversation_id or sticker_id is None:
        return
    await remember_item(
        _RECENT_STICKER_KEY.format(conversation_id=conversation_id),
        str(sticker_id), _RECENT_STICKER_KEEP,
    )


def _label_to_intensity_bucket(intensity: int) -> int:
    """0-100 intensity -> 1-5 sticker intensity bucket."""
    clamped = max(0, min(100, int(intensity or 0)))
    return min(5, clamped // 25 + 1)


async def recommend_sticker(
    primary_emotion: str | None = None,
    intensity: int = 50,
    conversation_id: str | None = None,
) -> dict | None:
    """推荐一个表情包。

    conversation_id: 供跨轮去重 (W6)；不传则退回无去重的旧行为。

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

    # W6 跨轮去重: 排除最近用过的; 全部被排除时回退全池 (小库存不至于没得选)
    recent_ids = await _load_recent_sticker_ids(conversation_id)
    fresh = [
        (row, score) for row, score in candidates
        if str(row["id"]) not in recent_ids
    ] or candidates

    chosen, score = random.choice(fresh)
    await _remember_sticker(conversation_id, chosen["id"])
    return {"id": chosen["id"], "url": chosen["url"], "match_score": round(score, 2)}
