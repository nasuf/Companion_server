"""W4 AI 情绪连续性（PAD 驱动行为的 MVP 形态）。

现状缺口：AI 的"情绪"只存在于单轮——每轮识别回复情绪后仅用于 emoji/贴纸
装饰，下一轮完全清零。真人的心情有惯性：上一轮聊到伤心事，下一轮不会
瞬间满血复活。

MVP 设计：
- 存储：每轮回复情绪（W1b 标记已免费产出）写入 Redis
  `ai_mood:{conversation_id}`，零额外 LLM 成本
- 衰减：读取时按时间指数衰减（半衰期 30min），衰减后强度 <25 或中性
  → 视为无心情，不注入
- 注入：「你的心情」段——告诉 LLM 当下情绪基调 + 行为提示（开心话多、
  低落话少），让情绪影响**行为**而不只是贴图

归属说明：虽在 relationship/ 目录（与 user 侧 emotion.py 相邻便于对照），
本质是 AI 的内部状态而非关系认知；未来若出现 ai_status_history 等更多
AI 状态模块，建议一起重组到 services/chat/ai_state/。

与完整 PAD 三维模型的差距（后续方向）：这里用 12 类离散标签 + 强度，
够驱动语气/长度；连续 PAD 空间与情绪转移矩阵留待产品定义情绪模型后做。
"""

from __future__ import annotations

import json
import logging
import time

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_MOOD_KEY = "ai_mood:{conversation_id}"
_MOOD_TTL_S = 4 * 3600
_HALF_LIFE_S = 30 * 60      # 半衰期 30min: 一小时后残留 1/4
_MIN_EFFECTIVE = 25         # 衰减后低于此强度视为已平复
_SKIP_EMOTIONS = {"中性", ""}

# 行为提示: 情绪 → 语气/长度引导 (正向描述, 不是硬约束)
_MOOD_BEHAVIOR_HINTS = {
    "高兴": "可以活泼话多一点",
    "欣慰": "语气可以柔软满足一点",
    "感激": "语气可以温热一点",
    "戏谑": "可以带点玩笑劲",
    "悲伤": "话可以少一点、软一点",
    "失望": "语气可以淡一点，不用强撑热情",
    "焦虑": "可以流露一点心不在焉",
    "愤怒": "语气可以冲一点点，但别对用户撒气",
    "恐惧": "可以带点不安",
    "厌恶": "可以冷淡一点点",
    "惊讶": "可以带点没回过神的感觉",
}


async def save_ai_mood(
    conversation_id: str | None, emotion: str | None, intensity: int,
) -> None:
    """每轮回复后记录 AI 情绪（Redis 写，~1ms，失败静默）。"""
    if not conversation_id or not emotion or emotion in _SKIP_EMOTIONS:
        return
    try:
        redis = await get_redis()
        await redis.set(
            _MOOD_KEY.format(conversation_id=conversation_id),
            json.dumps({
                "emotion": emotion,
                "intensity": max(0, min(100, int(intensity))),
                "ts": time.time(),
            }),
            ex=_MOOD_TTL_S,
        )
    except Exception:
        pass


async def load_ai_mood(conversation_id: str | None) -> dict | None:
    """读取并按时间衰减 AI 情绪；已平复/中性/失败返回 None。"""
    if not conversation_id:
        return None
    try:
        redis = await get_redis()
        raw = await redis.get(_MOOD_KEY.format(conversation_id=conversation_id))
        if not raw:
            return None
        data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        emotion = str(data.get("emotion", ""))
        if emotion in _SKIP_EMOTIONS:
            return None
        elapsed = max(0.0, time.time() - float(data.get("ts", 0)))
        effective = int(data.get("intensity", 0)) * (0.5 ** (elapsed / _HALF_LIFE_S))
        if effective < _MIN_EFFECTIVE:
            return None
        return {"emotion": emotion, "intensity": int(effective)}
    except Exception:
        return None


def format_ai_mood_text(mood: dict | None) -> str:
    """渲染成「你的心情」段素材；无心情返回空串。"""
    if not mood:
        return ""
    emotion = mood["emotion"]
    strength = "还挺明显" if mood["intensity"] >= 55 else "淡淡的"
    hint = _MOOD_BEHAVIOR_HINTS.get(emotion, "")
    hint_part = f"，{hint}" if hint else ""
    return f"{emotion}（{strength}）{hint_part}"
