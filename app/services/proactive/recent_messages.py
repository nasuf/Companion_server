"""主动消息 anti-repetition 守卫 (2026-09-14 task#12).

用户测两次沉默唤醒, 拿到**一字不差**的两条消息 (
"最近好像大家都在聊赵雷当爸爸了"). 根因: trending 缓存 + V3 分类器确定性 + LLM
temp=0.7 下相同 prompt 有相当概率完全重复.

## 策略

Redis 追踪每个 workspace 最近 N (=5) 条主动消息文本, TTL 24h. 生成完消息后:

- 完全一致 (exact match) → is_repeat = True, sender 决定重生成 or 不发
- 高度相似 (字符 shingling ≥ 阈值) → is_repeat = True

后续接入 embedding 相似度更精准, 但当前 shingling (2-char 集合 Jaccard) 已经
足够抓"复读机"型重复.
"""

from __future__ import annotations

import json
import logging

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_RECENT_KEY = "proactive:recent_messages:{workspace_id}"
_RECENT_MAX = 5           # 最多追多少条 (超出 LPUSH 后 LTRIM)
_RECENT_TTL_S = 24 * 3600 # 24h 后自动过期 (主动消息节奏本来就低, 一天足够查重)

# 相似度阈值: 2-char shingle Jaccard ≥ 此值判为重复.
# 0.6 手工调过: "最近好像大家都在聊赵雷当爸爸了" 跟 "最近好像大家都在聊赵雷当爸爸了"
# → 1.0 命中; 跟 "最近好像大家都在聊 iPhone 17 秒空" → ~0.35 不命中;
# 跟 "最近好像大家都在聊赵雷当奶爸" → ~0.7 命中 (换个字仍算复读).
_SIMILARITY_THRESHOLD = 0.6


def _shingles(text: str) -> set[str]:
    """2-char sliding window. 中文 char 计 (不做分词)."""
    s = (text or "").strip()
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _similarity(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


async def is_repeat_of_recent(
    workspace_id: str | None, text: str,
    *, similarity_threshold: float = _SIMILARITY_THRESHOLD,
) -> bool:
    """workspace 最近 N 条主动消息里若有跟 text 相似度 ≥ 阈值的 → True."""
    if not workspace_id or not text:
        return False
    try:
        redis = await get_redis()
        raw = await redis.lrange(_RECENT_KEY.format(workspace_id=workspace_id), 0, _RECENT_MAX - 1)
    except Exception as exc:  # noqa: BLE001 — Redis 挂了, 宁可放行也不阻塞发送
        logger.warning("[proactive-recent] redis lrange failed ws=%s: %s",
                       (workspace_id or "")[:8], exc)
        return False
    for item in raw or []:
        try:
            prev = json.loads(item.decode() if isinstance(item, (bytes, bytearray)) else item)
            prev_text = str(prev.get("text") or "")
        except Exception:  # noqa: BLE001
            continue
        sim = _similarity(text, prev_text)
        if sim >= similarity_threshold:
            logger.info(
                "[proactive-recent] repeat detected ws=%s sim=%.2f "
                "new_len=%d prev_len=%d",
                (workspace_id or "")[:8], sim, len(text), len(prev_text),
            )
            return True
    return False


async def remember_recent(workspace_id: str | None, text: str) -> None:
    """把这条主动消息塞进最近记录 (LPUSH + LTRIM 到 N 条, TTL 刷 24h)."""
    if not workspace_id or not text:
        return
    key = _RECENT_KEY.format(workspace_id=workspace_id)
    payload = json.dumps({"text": text}, ensure_ascii=False)
    try:
        redis = await get_redis()
        pipe = redis.pipeline()
        pipe.lpush(key, payload)
        pipe.ltrim(key, 0, _RECENT_MAX - 1)
        pipe.expire(key, _RECENT_TTL_S)
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[proactive-recent] redis remember failed ws=%s: %s",
                       (workspace_id or "")[:8], exc)
