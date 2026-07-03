"""Spec §5 回复加工 + §6.5 延迟解释的组合工具。

从 orchestrator 尾部抽出：延迟解释（≥1min）→ 逐条 emoji/表情包 → yield。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections.abc import AsyncGenerator
from datetime import datetime as _dt
from typing import Any, Awaitable, Callable

from app.config import settings
from app.observability.events import EVT_LLM_FAIL, EVT_REPLY_DECORATION
from app.services.schedule_domain.time_service import _now_corrected, _TZ
from app.services.interaction.reply_context import actual_delay_seconds
from app.services.chat.typo import maybe_typo
from app.services.emoji import pick_one_emoji, should_add_emoji, should_add_sticker
from app.services.prompting.store import get_prompt_text_or_default
from app.services.prompting.utils import safe_format
from app.services.sticker import recommend_sticker

logger = logging.getLogger(__name__)


_DELAY_PREFACE_RE = re.compile(
    r"^\s*(不好意思|抱歉|刚才|刚刚|我刚|才看到|才回过神|睡着了|眯过去了|"
    r"刚醒|没看到消息|回复慢了)[，,\s]*(?:[^。！？!?]{0,28}"
    r"(?:才看到|没看到|睡着|刚醒|回复慢|回得慢|耽误了|现在才回)[。！？!?，,\s]*)?"
)


def strip_duplicate_delay_preface(text: str) -> str:
    """已有独立 delay explanation 时，移除主回复开头重复的迟复道歉/解释。"""
    stripped = (text or "").strip()
    if not stripped:
        return ""
    cleaned, n = _DELAY_PREFACE_RE.subn("", stripped, count=1)
    if n:
        return cleaned.strip()
    return stripped


def _format_received_at(iso_str: str) -> str:
    """把 reply_context 里的 UTC ISO ('2026-05-03T00:51+00:00') 转成 Shanghai
    HH:MM 给 LLM 用. 跟 current_time 同格式, 防 LLM 混淆 tz 编出"早上 8 点收到"
    之类离谱话. 解析失败 → 返"刚刚" 兜底 (LLM 用 delay_minutes 是真值源)."""
    if not iso_str:
        return "刚刚"
    try:
        dt = _dt.fromisoformat(iso_str)
    except (ValueError, TypeError):
        return "刚刚"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_TZ)
    return dt.astimezone(_TZ).strftime("%H:%M")


def _reply_decoration_signal(reply_emotion: dict | None) -> tuple[str | None, int]:
    """Return (emotion label, intensity) from ai_reply_emotion output."""
    if not isinstance(reply_emotion, dict):
        return None, 0
    label = str(reply_emotion.get("emotion") or "").strip() or None
    try:
        intensity = int(float(reply_emotion.get("intensity", 0)))
    except (TypeError, ValueError):
        intensity = 0
    return label, max(0, min(100, intensity))


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
    # received_at 从 reply_context 拿到的是 UTC ISO ("2026-05-03T00:51+00:00").
    # 直接喂 LLM 会让它跟 current_time (HH:MM Shanghai) 混淆: "你在 00:51 收到...
    # 现在 08:51" — LLM 可能编出"早上 8 点"之类离谱话. 统一格式化成 Shanghai HH:MM.
    raw_received = (
        (reply_context or {}).get("latest_received_at")
        or (reply_context or {}).get("received_at", "")
    )
    received_time_str = _format_received_at(str(raw_received))
    try:
        text = await delay_reply_fn(
            received_time=received_time_str,
            # 必须用项目 _TZ (Asia/Shanghai), 不能裸 datetime.now() — 后者跟
            # 服务器系统时区走, UTC 容器下 LLM 看到"现在 00:51" 会回"夜深了"
            # (跟 sender.py:243 同根 bug). 用 _now_corrected 保留 NTP drift 修正.
            current_time=_now_corrected().strftime("%H:%M"),
            activity=activity,
            status=status_label,
            delay_minutes=minutes,
        )
        if not text:
            # 结构性兜底指令: 停用时退回代码默认 (与 conversation_end_fallback /
            # schedule_missing_context 两个 sibling 语义一致, 延迟解释链路不断).
            fallback_tpl = await get_prompt_text_or_default(
                "reply.delay_explanation_fallback_instruction"
            )
            text = await fallback_fn(
                agent, user_message,
                safe_format(fallback_tpl, {"delay_minutes": minutes}),
            )
        return (text or "").strip() or None
    except Exception as e:
        logger.warning(f"Delay explanation generation failed: {e}")
        return None


_RECENT_EMOJI_KEY = "emoji:recent:{conversation_id}"
_RECENT_EMOJI_KEEP = 3
_RECENT_EMOJI_TTL_S = 6 * 3600


async def _load_recent_emojis(conversation_id: str | None) -> set[str]:
    """读最近用过的 emoji (跨轮重复回避). Redis 不可用时静默返回空集."""
    if not conversation_id:
        return set()
    try:
        from app.redis_client import get_redis

        redis = await get_redis()
        items = await redis.lrange(
            _RECENT_EMOJI_KEY.format(conversation_id=conversation_id),
            0, _RECENT_EMOJI_KEEP - 1,
        )
        return {i.decode() if isinstance(i, bytes) else str(i) for i in items}
    except Exception:
        return set()


async def _remember_emoji(conversation_id: str | None, emoji: str) -> None:
    if not conversation_id or not emoji:
        return
    try:
        from app.redis_client import get_redis

        redis = await get_redis()
        key = _RECENT_EMOJI_KEY.format(conversation_id=conversation_id)
        pipe = redis.pipeline()
        pipe.lpush(key, emoji)
        pipe.ltrim(key, 0, _RECENT_EMOJI_KEEP - 1)
        pipe.expire(key, _RECENT_EMOJI_TTL_S)
        await pipe.execute()
    except Exception:
        pass


def _should_explain_delay(elapsed_s: float) -> bool:
    """C5 拟人度: 延迟解释按延迟时长概率化, 不再 ≥1min 必解释.

    真人不会每次迟回都交代原因 — 短延迟多数不说, 隔得越久越可能带一句
    "刚才在忙". spec §6.5 字面是 ≥1min 必发延迟解释; 概率化是刻意的拟人度
    偏离 (每次都解释反而暴露"系统在管理 AI 的时间").
    """
    if elapsed_s < 60:
        return False
    if elapsed_s < 300:
        p = 0.35
    elif elapsed_s < 1800:
        p = 0.6
    else:
        p = 0.85
    return random.random() < p


async def emit_replies(
    replies: list[str],
    *,
    reply_context: dict | None,
    reply_index_offset: int,
    sub_intent_mode: bool,
    agent,
    user_message: str,
    delay_reply_fn: Callable[..., Awaitable[str | None]],
    fallback_fn: Callable[..., Awaitable[str]],
    emitted_replies: list[dict],
    reply_emotion: dict | None = None,
    reply_is_fallback: bool = False,
    conversation_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """spec §5/§6.4-§6.5：延迟解释 + emoji/sticker + reply SSE 事件流。

    emitted_replies 传入的空列表会被原地填充（用于后续 `_save_replies`）。
    sub_intent_mode=True 时跳过延迟解释（父调用已推送）。
    reply_emotion: spec §5 step 1 的 ai_reply_emotion 输出 `{emotion, intensity}`。
    conversation_id: 供 emoji 跨轮重复回避 (C4); 不传则只做轮内去重。
    """
    ai_primary_emotion, emotion_intensity = _reply_decoration_signal(reply_emotion)
    sticker_used = False  # 一个回合最多一个表情包
    # C4 拟人度: 一个回合最多一个 emoji (原来每条 reply 独立掷骰, 2-3 条回复
    # 可能条条带表情, 是"装饰感"的主要来源); 且排除最近几轮用过的, 防复读.
    emoji_used_this_turn = False
    recent_emojis = await _load_recent_emojis(conversation_id)

    # §6.4/§6.5 延迟解释
    elapsed = None if sub_intent_mode else actual_delay_seconds(reply_context)
    delay_explain_offset = 0
    if elapsed is not None and _should_explain_delay(elapsed):
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
    normal_reply_count = 0
    for i, reply_text in enumerate(replies):
        if delay_explain_offset and not reply_is_fallback:
            reply_text = strip_duplicate_delay_preface(reply_text)
            if not reply_text:
                continue

        # E1 拟人度: 错别字注入 (默认关). 在 emoji 装饰前做, 只动正文汉字.
        typo_correction: str | None = None
        if settings.typo_enabled and not reply_is_fallback:
            reply_text, typo_correction = maybe_typo(
                reply_text, rate=settings.typo_rate,
            )

        added_emoji = False
        emoji_used: str | None = None
        if not emoji_used_this_turn and should_add_emoji(emotion_intensity):
            emoji = pick_one_emoji(ai_primary_emotion, exclude=recent_emojis)
            if emoji:
                reply_text += emoji
                added_emoji = True
                emoji_used = emoji
                emoji_used_this_turn = True
                await _remember_emoji(conversation_id, emoji)

        sticker_url: str | None = None
        if not added_emoji and not sticker_used and should_add_sticker(emotion_intensity):
            try:
                result = await recommend_sticker(
                    primary_emotion=ai_primary_emotion,
                    intensity=emotion_intensity,
                )
                if result:
                    sticker_url = result["url"]
                    sticker_used = True
            except Exception as e:
                logger.debug(
                    f"sticker recommend failed: {e}",
                    extra={"event": EVT_LLM_FAIL, "stage": "sticker_recommend"},
                )

        # 单条装饰决策 — DEBUG 因为 1 条 user msg 可能 emit 1-3 reply, 频率不低
        decoration_kind = "emoji" if added_emoji else ("sticker" if sticker_url else "none")
        logger.debug(
            f"[REPLY-DECO] reply[{i}] kind={decoration_kind}",
            extra={
                "event": EVT_REPLY_DECORATION,
                "reply_index": reply_index_offset + i,
                "decoration_kind": decoration_kind,
                "emoji": emoji_used,
                "sticker_url": sticker_url,
                "ai_emotion": ai_primary_emotion,
                "emotion_intensity": emotion_intensity,
            },
        )

        if normal_reply_count > 0 or delay_explain_offset or reply_index_offset > 0:
            await asyncio.sleep(random.uniform(0.3, 0.8))

        data: dict = {
            "text": reply_text,
            "index": reply_index_offset + normal_reply_count + delay_explain_offset,
        }
        if ai_primary_emotion:
            data["ai_emotion"] = ai_primary_emotion
            data["emotion_intensity"] = emotion_intensity
        if sticker_url:
            data["sticker_url"] = sticker_url
        if reply_is_fallback:
            # spec-audit: 主 LLM + Ollama 全挂, 走了静态兜底文本;
            # 前端可据此显示"重新回答"按钮或隐藏 emoji 等非必要装饰.
            data["reply_failed"] = True
        emitted_replies.append(data)
        normal_reply_count += 1
        yield {"event": "reply", "data": json.dumps(data)}

        # E1: 自我纠正气泡 ("*正确字", 微信惯例). 短暂停顿后追加, 像真人
        # 发出去才发现打错. 只对 maybe_typo 决定需要纠正的错字触发.
        if typo_correction:
            await asyncio.sleep(random.uniform(0.5, 1.2))
            correction_data: dict = {
                "text": f"*{typo_correction}",
                "index": reply_index_offset + normal_reply_count + delay_explain_offset,
                "typo_correction": True,
            }
            emitted_replies.append(correction_data)
            normal_reply_count += 1
            yield {"event": "reply", "data": json.dumps(correction_data)}
