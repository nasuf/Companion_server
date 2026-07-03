"""W3 关系时长感知 MVP：给 LLM"你们认识多久、聊过多少"的素材。

现状缺口：第 1 天和第 100 天，AI 对用户的语气和关系感完全一样——
intimacy_stage 只有 P1-P5 档位，没有"时间厚度"。真人朋友会说"都认识
仨月了你还跟我客气"，这需要 LLM 知道具体的相识时长和互动规模。

MVP 边界：只提供事实素材（认识天数 + 大约轮数），由 LLM 自然化用；
不做好感度/关系类型/衰减——那些需要产品定义关系模型后再立项
（与 intimacy_stage 的整合决策见 CLAUDE.md）。

成本：Redis 缓存 6h；miss 时 2 个 DB 查询（conversation.createdAt +
message.count）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.db import db
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_META_KEY = "relmeta:{conversation_id}"
_META_TTL_S = 6 * 3600


async def get_relation_meta(conversation_id: str | None) -> dict | None:
    """返回 {"days_known": int, "approx_turns": int}；失败返回 None（段内不展示）。"""
    if not conversation_id:
        return None
    key = _META_KEY.format(conversation_id=conversation_id)
    redis = None
    try:
        redis = await get_redis()
        cached = await redis.get(key)
        if cached:
            raw = cached.decode() if isinstance(cached, bytes) else str(cached)
            data = json.loads(raw)
            return _finalize(data)
    except Exception:
        pass

    try:
        conv = await db.conversation.find_unique(where={"id": conversation_id})
        if conv is None or getattr(conv, "createdAt", None) is None:
            return None
        n_messages = await db.message.count(
            where={"conversationId": conversation_id},
        )
    except Exception as e:
        logger.debug(f"relation meta lookup failed: {e}")
        return None

    data = {
        "created_at": conv.createdAt.isoformat(),
        # 1 轮 ≈ 用户 1 条 + AI 1-3 条, 粗略除 3 (素材只需要量级感)
        "approx_turns": max(1, n_messages // 3),
    }
    if redis is not None:
        try:
            await redis.set(key, json.dumps(data), ex=_META_TTL_S)
        except Exception:
            pass
    return _finalize(data)


def _finalize(data: dict) -> dict | None:
    """缓存的 created_at 转成当下的天数（天数必须现算，不能缓存）。"""
    try:
        created = datetime.fromisoformat(str(data["created_at"]))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days = max(0, (datetime.now(timezone.utc) - created).days)
        return {"days_known": days, "approx_turns": int(data["approx_turns"])}
    except Exception:
        return None


def format_relation_meta_line(meta: dict | None) -> str:
    """渲染成关系段的补充句；无数据返回空串（模板占位符原地消失）。

    量级感即可：轮数向下取整到 10（"大约 320 轮"），首日特殊措辞。
    """
    if not meta:
        return ""
    days = meta.get("days_known", 0)
    turns = meta.get("approx_turns", 0)
    turns_text = f"聊过大约 {max(1, turns // 10 * 10) if turns >= 10 else turns} 轮"
    if days <= 0:
        return f"你们今天刚认识，{turns_text}。"
    return f"你们认识 {days} 天了，{turns_text}。"
