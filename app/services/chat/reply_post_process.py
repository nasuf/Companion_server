"""Spec §5 回复加工 + §6.5 延迟解释的组合工具。

从 orchestrator 尾部抽出：情绪抽取 → 延迟解释（≥1min）→ 逐条 emoji/表情包 → yield。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Awaitable, Callable

from app.services.interaction.reply_context import actual_delay_seconds
from app.services.emoji import pick_one_emoji, should_add_emoji, should_add_sticker
from app.services.sticker import recommend_sticker

logger = logging.getLogger(__name__)


def extract_pad_for_decoration(emotion: Any) -> tuple[float, float, float, str | None]:
    """从 AI 情绪字段抽 (pleasure, arousal, dominance, primary_emotion)。"""
    emo = emotion if isinstance(emotion, dict) else {}
    return (
        float(emo.get("pleasure", 0.0)),
        float(emo.get("arousal", 0.0)),
        float(emo.get("dominance", 0.5)),
        emo.get("primary_emotion"),
    )


def _resolve_primary_emotion(reply_emotion: dict | None, pad_primary: str | None) -> str | None:
    """spec §5：优先用 ai_reply_emotion 输出的 emotion 标签，回退到 AI PAD 缓存。"""
    if reply_emotion and reply_emotion.get("emotion"):
        return reply_emotion["emotion"]
    return pad_primary


async def _build_delay_explanation_text(
    reply_context: dict | None,
    elapsed: float,
    *,
    delay_reply_fn: Callable[..., Awaitable[str | None]],
    fallback_fn: Callable[..., Awaitable[str]],
    agent,
    user_message: str,
) -> str | None:
    """spec §6.5：延迟 ≥1min 时生成单独的解释回复文本。"""
    received_status = (reply_context or {}).get("received_status") or {}
    activity = str(received_status.get("activity", "")).strip() or "处理自己的事"
    status_label = str(received_status.get("status", "idle"))
    minutes = max(1, round(elapsed / 60))
    received_at = str((reply_context or {}).get("received_at", ""))
    try:
        text = await delay_reply_fn(
            received_time=received_at,
            current_time=datetime.now().strftime("%H:%M"),
            activity=activity,
            status=status_label,
            delay_minutes=minutes,
        )
        if not text:
            text = await fallback_fn(
                agent, user_message,
                f"你{minutes}分钟前收到用户消息但在忙，现在才回复。用1句简短自然的解释。",
            )
        return (text or "").strip() or None
    except Exception as e:
        logger.warning(f"Delay explanation generation failed: {e}")
        return None


async def emit_replies(
    replies: list[str],
    *,
    reply_context: dict | None,
    reply_index_offset: int,
    sub_intent_mode: bool,
    emotion: Any,
    agent,
    user_message: str,
    delay_reply_fn: Callable[..., Awaitable[str | None]],
    fallback_fn: Callable[..., Awaitable[str]],
    emitted_replies: list[dict],
    reply_emotion: dict | None = None,
    reply_is_fallback: bool = False,
) -> AsyncGenerator[dict, None]:
    """spec §5/§6.4-§6.5：延迟解释 + emoji/sticker + reply SSE 事件流。

    emitted_replies 传入的空列表会被原地填充（用于后续 `_save_replies`）。
    sub_intent_mode=True 时跳过延迟解释（父调用已推送）。
    reply_emotion: spec §5 step 1 的 ai_reply_emotion 输出 `{emotion, intensity}`，
    若提供则优先用其 emotion 标签匹配 EMOJI_MAP，否则回退到 AI PAD 缓存的 primary_emotion。
    """
    ai_pleasure, ai_arousal, ai_dominance, pad_primary = extract_pad_for_decoration(emotion)
    ai_primary_emotion = _resolve_primary_emotion(reply_emotion, pad_primary)
    sticker_used = False  # 一个回合最多一个表情包

    # §6.4/§6.5 延迟解释
    elapsed = None if sub_intent_mode else actual_delay_seconds(reply_context)
    delay_explain_offset = 0
    if elapsed is not None and elapsed >= 60:
        explain_text = await _build_delay_explanation_text(
            reply_context, elapsed,
            delay_reply_fn=delay_reply_fn, fallback_fn=fallback_fn,
            agent=agent, user_message=user_message,
        )
        if explain_text:
            data: dict = {
                "text": explain_text,
                "index": reply_index_offset,
                "delay_explanation": True,
            }
            emitted_replies.append(data)
            yield {"event": "reply", "data": json.dumps(data)}
            delay_explain_offset = 1

    # §5 逐条 emoji / sticker / 推送
    for i, reply_text in enumerate(replies):
        added_emoji = False
        if should_add_emoji(ai_arousal):
            emoji = pick_one_emoji(ai_pleasure, ai_arousal, ai_primary_emotion)
            if emoji:
                reply_text += emoji
                added_emoji = True

        sticker_url: str | None = None
        if not added_emoji and not sticker_used and should_add_sticker(ai_arousal):
            try:
                result = await recommend_sticker(
                    ai_pleasure, ai_arousal, ai_dominance, ai_primary_emotion,
                )
                if result:
                    sticker_url = result["url"]
                    sticker_used = True
            except Exception:
                pass

        if i > 0 or delay_explain_offset or reply_index_offset > 0:
            await asyncio.sleep(random.uniform(0.3, 0.8))

        data: dict = {
            "text": reply_text,
            "index": reply_index_offset + i + delay_explain_offset,
        }
        if sticker_url:
            data["sticker_url"] = sticker_url
        if reply_is_fallback:
            # spec-audit: 主 LLM + Ollama 全挂, 走了静态兜底文本;
            # 前端可据此显示"重新回答"按钮或隐藏 emoji 等非必要装饰.
            data["reply_failed"] = True
        emitted_replies.append(data)
        yield {"event": "reply", "data": json.dumps(data)}
