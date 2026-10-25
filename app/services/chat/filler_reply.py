"""Optional emoji-only replies, disabled by default and guarded by whole turns.

The global runtime probability is bound once per chat turn. No model or database
calls here; diagnostics contain decisions, never message text.
"""

from __future__ import annotations

import math
import random
import unicodedata
from datetime import UTC, datetime
from typing import Any

from app.services.chat.message_utils import _parse_message_created_at, message_field
from app.services.runtime_config import resolve_config_sync

from app.services.emoji import EMOJI_MAP, is_emoji_only_message


# 词表维护提示: interaction/aggregation.py (常用应答词→不算碎片) 与
# memory/recording/filter.py (_FILLER_WORDS→不进记忆) 各有独立词表, 语义不同
# (碎片判定/记忆过滤/轻回应) 但词汇有重叠 — 增删高频语气词时三处一起看.
_POSITIVE_FILLERS = frozenset({
    "哈哈", "哈哈哈", "哈哈哈哈", "嘿嘿", "嘻嘻", "666", "赞", "棒", "nice",
    "笑死", "太好了",
})
_NEUTRAL_FILLERS = frozenset({
    "嗯", "嗯嗯", "哦", "噢", "喔", "哦哦", "好", "好的", "好吧", "好嘞",
    "行", "可以", "ok", "okk", "okay", "对", "是", "是的", "在", "在的",
    "收到", "了解",
})

_QUESTION_ENDINGS = ("吗", "呢", "么", "不", "嘛", "对吧", "是吧", "好吧", "如何", "怎样", "怎么样")


def _strip_question_tail(text: str) -> str:
    value = text.rstrip()
    while value and unicodedata.category(value[-1])[0] not in "LN":
        value = value[:-1]
    return value


def is_question_like(text: str | None) -> bool:
    """Any question bubble blocks the shortcut, including '对吧？🙂' / '好吗 😊'."""
    value = text or ""
    if "?" in value or "？" in value:
        return True
    return any(_strip_question_tail(bubble).endswith(_QUESTION_ENDINGS)
               for bubble in value.splitlines() if bubble.strip())


def build_filler_emoji_reply(
    user_message: str,
    *,
    previous_assistant_text: str | None = None,
    previous_assistant: Any = None,
    received_at: datetime | str | None = None,
    diagnostics: dict | None = None,
    probability: float | None = None,
    rng: random.Random | None = None,
) -> str | None:
    """Return emoji only when every guard passes; otherwise use the full pipeline."""
    r = rng or random
    text = (message_field(previous_assistant, "content") if previous_assistant is not None
            else previous_assistant_text)
    configured = resolve_config_sync().filler_emoji_probability if probability is None else probability
    enabled_probability = float(configured)
    if not math.isfinite(enabled_probability) or not 0 <= enabled_probability <= 1:
        enabled_probability = 0.0
    normalized = (user_message or "").strip().lower().rstrip("~～!！。.")
    is_filler = normalized in _POSITIVE_FILLERS or normalized in _NEUTRAL_FILLERS
    previous_emoji = is_emoji_only_message(text or "")
    received = _parse_message_created_at(received_at)
    previous_at = _parse_message_created_at(message_field(previous_assistant, "createdAt"))
    gap = (received - previous_at).total_seconds() if received and previous_at else None
    previous_user = message_field(previous_assistant, "previous_user_text")
    repeat: bool | None = False
    if previous_emoji:
        if previous_user is None:
            repeat = None
        elif str(previous_user).strip() == user_message.strip():
            repeat = None if gap is None or gap < 0 else gap <= 60
    from app.services.chat.intent_dispatcher import previous_turn_offered_schedule_change

    guards = {
        "is_filler": is_filler,
        "previous_turn_is_question": is_question_like(text),
        "previous_turn_is_emoji_only": previous_emoji,
        "previous_turn_offered_schedule_change": previous_turn_offered_schedule_change(text),
        "repeated_after_emoji_within_60s": repeat,
    }
    reason = (
        "disabled" if enabled_probability == 0 else
        "not_filler" if not is_filler else
        "previous_turn_question" if guards["previous_turn_is_question"] else
        "previous_turn_schedule_offer" if guards["previous_turn_offered_schedule_change"] else
        "repeat_after_emoji" if repeat else
        "previous_turn_emoji_only" if previous_emoji else
        "probability_miss" if r.random() >= enabled_probability else "selected"
    )
    if diagnostics is not None:
        diagnostics["filler_emoji"] = {
            "version": 1, "reason": reason, "probability": enabled_probability,
            "guards": guards,
            "previous_turn_bubble_count": len(message_field(previous_assistant, "source_message_ids", ()) or ()),
            "seconds_since_previous_reply": gap,
            "repeat_window_seconds": 60,
            "sampled_at": datetime.now(UTC).isoformat(),
        }
        if reason == "selected":
            diagnostics.update(version=1, reply_path="filler_emoji", main_prompt_built=False)
    if reason != "selected":
        return None
    return r.choice(EMOJI_MAP["高兴" if normalized in _POSITIVE_FILLERS else "中性"])
