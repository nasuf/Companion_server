"""Phase E2（拟人度）：纯语气词的概率性"仅表情"轻回应。

真人收到"嗯 / 哈哈 / 好的"这类纯应答词时，不会每次都认真组织一句话回复——
经常只回一个表情，甚至就让对话自然停在那里。原管线对每条消息都跑完整
意图识别 + 主 LLM，对纯语气词输出一句"正经回复"，是"应答机感"的来源之一。

此模块提供概率性短路：命中纯语气词表 + 概率命中 → 返回一个匹配情绪的
emoji 文本，orchestrator 直接以短路回复发出（跳过意图/检索/主 LLM）；
未命中概率（70%）仍走完整管线，保证行为多样性。

安全护栏：AI 上一句是提问时**绝不**走此路径——"好"/"嗯"此时是对提问的
答复（如作息调整确认），必须交给意图管线结合上下文理解。
"""

from __future__ import annotations

import random

from app.services.emoji import EMOJI_MAP

# 概率: 命中语气词后走"仅表情"的比例; 其余走完整管线.
FILLER_EMOJI_PROBABILITY = 0.3

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

_QUESTION_ENDINGS = ("?", "？", "吗", "呢", "么", "不", "嘛")


def is_question_like(text: str | None) -> bool:
    """AI 上一句是否像提问（"好/嗯"很可能是在回答它）。"""
    t = (text or "").rstrip()
    return bool(t) and t.endswith(_QUESTION_ENDINGS)


def build_filler_emoji_reply(
    user_message: str,
    *,
    previous_assistant_text: str | None = None,
    rng: random.Random | None = None,
) -> str | None:
    """纯语气词 → 概率性返回一个匹配情绪的 emoji；不适用时返回 None。

    返回 None 的情况：非语气词 / AI 上一句是提问 / 概率未命中。
    """
    r = rng or random
    if is_question_like(previous_assistant_text):
        return None
    normalized = (user_message or "").strip().lower().rstrip("~～!！。.")
    if normalized in _POSITIVE_FILLERS:
        pool = EMOJI_MAP["高兴"]
    elif normalized in _NEUTRAL_FILLERS:
        pool = EMOJI_MAP["中性"]
    else:
        return None
    if r.random() >= FILLER_EMOJI_PROBABILITY:
        return None
    return r.choice(pool)
