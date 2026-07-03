"""W2 中期记忆 MVP：重逢时的「上次聊到」摘要（借鉴 MaiBot mid-term memory）。

现状缺口：聊天历史按 token 预算截断即彻底遗忘；记忆系统只存抽取出的原子
事实，不存对话流本身。用户隔了几小时回来说"还是有点紧张"，如果关键信息
没被抽取管线落成记忆，AI 接不住这个"还"字。

MVP 设计（与 B2 重逢感知闭环）：
- 触发：重逢 gap ≥ RECAP_GAP_SECONDS（与话题栈重置阈值对齐，3h）
- 生成：对间隔前的最近若干条消息做一次小模型摘要（1-2 句，≤60 字）
- 缓存：Redis 按「会话+间隔前最后一条消息 id」为 key —— 同一次重逢内的
  多轮消息复用同一份摘要，LLM 只调一次
- 注入：独立「上次聊到」section（chat.session_recap_section），紧跟
  重逢感知段，让 AI 能主动接"上次你说到 X，后来怎么样了"
- 失败：任何环节失败返回 None，段不注入（增强项，不阻塞主路径）

与 MaiBot 完整版 mid-term memory 的差距（V2 方向）：召回线索 embedding +
相似度唤起（对话中途也能"想起上午聊过的"），多级摘要（周/月归档）。
"""

from __future__ import annotations

import logging

from app.observability.events import EVT_SESSION_RECAP
from app.redis_client import get_redis
from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompting.store import get_prompt_text
from app.services.prompting.utils import safe_format

logger = logging.getLogger(__name__)

# 与 topic.TOPIC_RESET_GAP_SECONDS / prompt_builder._REENGAGE_LONG_S 对齐:
# 话题栈都清了, 说明"上一段对话"已成为过去 — 正是需要摘要接续的时刻.
RECAP_GAP_SECONDS = 3 * 3600

_RECAP_KEY = "recap:{conversation_id}:{anchor_message_id}"
_RECAP_TTL_S = 7 * 86400
_RECAP_SOURCE_MESSAGES = 15   # 摘要输入: 间隔前最近 N 条
_RECAP_MAX_LEN = 80           # 摘要长度硬上限 (LLM 超出则截断)


def _pre_gap_messages(
    messages: list[dict], exclude_ids: set[str] | None,
) -> list[dict]:
    """取间隔前的消息（排除当前轮）。gap ≥3h 时历史尾部即间隔前对话。"""
    exclude = exclude_ids or set()
    return [
        m for m in messages
        if not (m.get("id") and m["id"] in exclude)
        and not m.get("synthetic_current")
        and str(m.get("content", "")).strip()
    ]


async def get_or_build_session_recap(
    conversation_id: str,
    messages: list[dict],
    *,
    gap_seconds: float | None,
    exclude_ids: set[str] | None = None,
) -> str | None:
    """重逢时返回「上次聊到」摘要；不满足条件 / 失败返回 None。"""
    if not conversation_id or gap_seconds is None or gap_seconds < RECAP_GAP_SECONDS:
        return None
    pre_gap = _pre_gap_messages(messages, exclude_ids)
    if len(pre_gap) < 2:
        return None  # 上一段对话太短, 没什么可"聊到"的

    anchor_id = str(pre_gap[-1].get("id") or "") or "na"
    key = _RECAP_KEY.format(
        conversation_id=conversation_id, anchor_message_id=anchor_id,
    )
    try:
        redis = await get_redis()
        cached = await redis.get(key)
        if cached:
            text = cached.decode() if isinstance(cached, bytes) else str(cached)
            return text or None
    except Exception:
        redis = None  # Redis 不可用: 仍可生成, 只是不缓存

    recap = await _summarize(pre_gap[-_RECAP_SOURCE_MESSAGES:])
    if not recap:
        return None
    logger.info(
        f"[RECAP] built len={len(recap)}",
        extra={"event": EVT_SESSION_RECAP, "recap_len": len(recap),
               "n_source_messages": min(len(pre_gap), _RECAP_SOURCE_MESSAGES)},
    )
    if redis is not None:
        try:
            await redis.set(key, recap, ex=_RECAP_TTL_S)
        except Exception:
            pass
    return recap


async def _summarize(messages: list[dict]) -> str | None:
    """小模型生成 1-2 句对话摘要。失败返回 None。"""
    lines = [
        f"{'用户' if m.get('role') == 'user' else '你'}: {str(m.get('content', ''))[:120]}"
        for m in messages
    ]
    try:
        tpl = await get_prompt_text("chat.session_recap")
        prompt = safe_format(tpl, {"conversation": "\n".join(lines)})
        raw = await invoke_text(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"session recap generation failed: {e}")
        return None
    recap = (raw or "").strip().strip("「」\"'")
    if not recap or len(recap) < 4:
        return None
    return recap[:_RECAP_MAX_LEN]
