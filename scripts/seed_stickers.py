"""填充表情包库 — 至少100个表情包，覆盖12类情绪。

Usage:
    cd Server
    python -m scripts.seed_stickers
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 表情包种子数据：每个情绪类别若干条
# url 使用占位符格式，实际部署时替换为 CDN 地址
_BASE = "https://cdn.example.com/stickers"

_SEED: list[dict] = []
_ID = 0


def _add(emotion: str, weight: float, intensity: int, style: str, tags: list[str], count: int = 1, extra_emotions: list[dict] | None = None):
    global _ID
    for i in range(count):
        _ID += 1
        emotion_tags = [{"emotion": emotion, "weight": weight}]
        if extra_emotions:
            emotion_tags.extend(extra_emotions)
        _SEED.append({
            "url": f"{_BASE}/{emotion.lower()}_{_ID:03d}.webp",
            "emotion_tags": json.dumps(emotion_tags),
            "intensity": intensity,
            "style": style,
            "tags": json.dumps(tags),
        })


# ── 高兴 (15) ──
_add("高兴", 0.9, 5, "可爱", ["庆祝"], count=3)
_add("高兴", 0.8, 4, "幽默", ["日常"], count=3)
_add("高兴", 0.7, 3, "可爱", ["问候"], count=3)
_add("高兴", 0.6, 2, "文艺", ["日常"], count=3)
_add("高兴", 0.9, 5, "搞怪", ["庆祝", "惊喜"], count=3)

# ── 悲伤 (10) ──
_add("悲伤", 0.9, 2, "可爱", ["安慰"], count=3)
_add("悲伤", 0.7, 3, "文艺", ["共情"], count=3)
_add("悲伤", 0.8, 1, "可爱", ["安慰"], count=2)
_add("悲伤", 0.6, 2, "严肃", ["共情"], count=2)

# ── 愤怒 (8) ──
_add("愤怒", 0.8, 4, "搞怪", ["吐槽"], count=3)
_add("愤怒", 0.7, 5, "幽默", ["吐槽"], count=3)
_add("愤怒", 0.6, 3, "可爱", ["日常"], count=2)

# ── 恐惧 (6) ──
_add("恐惧", 0.8, 4, "可爱", ["惊吓"], count=3)
_add("恐惧", 0.7, 3, "搞怪", ["惊吓"], count=3)

# ── 惊讶 (8) ──
_add("惊讶", 0.9, 5, "搞怪", ["惊喜"], count=3)
_add("惊讶", 0.7, 4, "可爱", ["惊喜"], count=3)
_add("惊讶", 0.6, 3, "幽默", ["日常"], count=2)

# ── 厌恶 (6) ──
_add("厌恶", 0.7, 3, "搞怪", ["吐槽"], count=3)
_add("厌恶", 0.6, 2, "幽默", ["吐槽"], count=3)

# ── 中性 (8) ──
_add("中性", 0.5, 2, "可爱", ["日常", "问候"], count=3)
_add("中性", 0.5, 1, "文艺", ["日常"], count=3)
_add("中性", 0.4, 2, "幽默", ["问候"], count=2)

# ── 焦虑 (8) ──
_add("焦虑", 0.8, 4, "可爱", ["安慰"], count=3)
_add("焦虑", 0.7, 3, "文艺", ["共情"], count=3)
_add("焦虑", 0.6, 3, "搞怪", ["安慰"], count=2)

# ── 失望 (6) ──
_add("失望", 0.8, 2, "可爱", ["安慰"], count=3)
_add("失望", 0.7, 1, "文艺", ["共情"], count=3)

# ── 欣慰 (8) ──
_add("欣慰", 0.8, 2, "可爱", ["鼓励", "暖心"], count=3)
_add("欣慰", 0.7, 1, "文艺", ["暖心"], count=3)
_add("欣慰", 0.6, 2, "幽默", ["鼓励"], count=2)

# ── 感激 (8) ──
_add("感激", 0.9, 3, "可爱", ["感谢"], count=3)
_add("感激", 0.7, 2, "文艺", ["感谢"], count=3)
_add("感激", 0.8, 3, "幽默", ["感谢", "暖心"], count=2)

# ── 戏谑 (9) ──
_add("戏谑", 0.9, 4, "搞怪", ["调侃"], count=3)
_add("戏谑", 0.8, 5, "幽默", ["调侃", "日常"], count=3)
_add("戏谑", 0.7, 3, "搞怪", ["日常"], count=3)

# 混合情绪（同时标注多个情绪）
_add("高兴", 0.6, 3, "可爱", ["问候", "暖心"], count=2, extra_emotions=[{"emotion": "欣慰", "weight": 0.5}])
_add("悲伤", 0.5, 2, "可爱", ["安慰"], count=2, extra_emotions=[{"emotion": "失望", "weight": 0.6}])
_add("戏谑", 0.7, 4, "搞怪", ["调侃"], count=2, extra_emotions=[{"emotion": "高兴", "weight": 0.4}])

assert len(_SEED) >= 100, f"需要至少100个表情包，当前 {len(_SEED)} 个"


async def main():
    from app.db import db

    await db.connect()

    try:
        existing = await db.query_raw("SELECT COUNT(*)::int AS cnt FROM stickers")
        cnt = existing[0]["cnt"] if existing else 0
        if cnt > 0:
            print(f"stickers 表已有 {cnt} 条数据，跳过填充。如需重新填充请先清空。")
            return

        for s in _SEED:
            await db.execute_raw(
                """
                INSERT INTO stickers (url, emotion_tags, intensity, style, tags)
                VALUES ($1, $2::jsonb, $3, $4, $5::jsonb)
                """,
                s["url"], s["emotion_tags"], s["intensity"], s["style"], s["tags"],
            )

        print(f"成功填充 {len(_SEED)} 个表情包。")

        # 验证
        result = await db.query_raw("SELECT COUNT(*)::int AS cnt FROM stickers")
        print(f"验证: stickers 表当前 {result[0]['cnt']} 条。")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
