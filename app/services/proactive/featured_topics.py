"""主动交流·最近 featured 话题追踪 (2026-09-14).

## 背景

V3 分类器 (topic_source.classify_topic_source) 100% 确定性 —— 相同 trending
candidates 永远选 hot_ok[0] (第一条). DailyHot 上游本身也缓存 top-N, 短时间内
热榜前几名不动. 结果: 用户短时间连续测多次主动消息 → 每次都是"赵雷当爸爸".

用户配的 proactive_trending_cache_ttl_s=60 秒不解决这个 — admin_test 直接
bypass_cache, 且 DailyHot 自己就返一样的 top-N.

## 策略

Redis 追踪 workspace 最近 N (=10) 条已 featured 的 title, TTL 6h (让"社交谈资"
在半天内不重复; 6h 后自然过期, 因为热榜话题本来也在更换).

分类器把这个集合作为"排除 title"过滤 candidates. 第 1 次挑"赵雷当爸爸"; 第 2 次
自动跳过, 选下一条; 依此类推. 排除完全部 candidates → kind="none" → silence_plain
兜底.

## 与 recent_messages 的差异

- recent_messages 追踪的是**生成的消息文本** (LLM 输出), 用于 anti-repetition
  retry 判定. 追**结果**.
- featured_topics 追踪的是**分类器挑中的候选 title** (DailyHot 输入源), 用于
  下次选源时避开. 追**输入**.

两者都存在意义: recent_messages 抓 "LLM 复读同一句" (即使 topic 换了); 而
featured_topics 抓 "topic 就是同一个" (话题源头没变). 一个防语言复读, 一个防
话题复读, 正交.
"""

from __future__ import annotations

import logging

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_FEATURED_KEY = "proactive:featured_topics:{workspace_id}"
_FEATURED_MAX = 10           # 追多少条 featured (超出 LPUSH+LTRIM)
_FEATURED_TTL_S = 6 * 3600   # 6h; 热榜话题本来也在半天内换, 到期自然清理


async def get_recent_featured(workspace_id: str | None) -> set[str]:
    """workspace 最近 N 条已 featured 的 title 集合. Redis 挂 → 空集合 (放行)."""
    if not workspace_id:
        return set()
    try:
        redis = await get_redis()
        raw = await redis.lrange(
            _FEATURED_KEY.format(workspace_id=workspace_id), 0, _FEATURED_MAX - 1,
        )
    except Exception as exc:  # noqa: BLE001 — Redis 挂了不该阻塞主动消息发送
        logger.warning("[proactive-featured] redis lrange failed ws=%s: %s",
                       (workspace_id or "")[:8], exc)
        return set()
    out: set[str] = set()
    for item in raw or []:
        if isinstance(item, (bytes, bytearray)):
            item = item.decode("utf-8", errors="ignore")
        t = str(item or "").strip()
        if t:
            out.add(t)
    return out


async def remember_featured(workspace_id: str | None, title: str) -> None:
    """把 featured 的 title 加进最近集合 (LPUSH + LTRIM + EXPIRE)."""
    if not workspace_id or not title:
        return
    key = _FEATURED_KEY.format(workspace_id=workspace_id)
    try:
        redis = await get_redis()
        pipe = redis.pipeline()
        pipe.lpush(key, title.strip())
        pipe.ltrim(key, 0, _FEATURED_MAX - 1)
        pipe.expire(key, _FEATURED_TTL_S)
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[proactive-featured] redis remember failed ws=%s: %s",
                       (workspace_id or "")[:8], exc)
