"""记忆消息过滤器。

多信号准入，决定是否对消息进行记忆提取。
核心目标是过滤纯寒暄/回声词，同时避免错过简短但高价值的自我披露。
"""

from __future__ import annotations

import re

from app.services.rules.memory_keywords import (
    CORE_PROFILE_PATTERNS,
    FIRST_PERSON_TERMS,
    FIRST_PERSON_TERMS_EN,
    MEMORY_EMOTION_WORDS,
    MEMORY_EMOTION_WORDS_EN,
    MEMORY_FACT_WORDS,
    MEMORY_FACT_WORDS_EN,
    MEMORY_FILLER_WORDS,
    MEMORY_TIME_WORDS,
    MEMORY_TIME_WORDS_EN,
    SELF_DISCLOSURE_PATTERNS,
    SELF_DISCLOSURE_PATTERNS_EN,
)


def _word_set(message: str) -> set[str]:
    """Extract lowercase latin words for lightweight English heuristics."""
    return {w.lower() for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", message)}


def should_extract_memory(message: str) -> bool:
    """判断消息是否值得进行记忆提取。

    Spec §2.1.1: 纯语气词或长度≤2的常用应答词直接丢弃。
    """
    if not message or not message.strip():
        return False

    msg = message.strip()

    # Spec §2.1.1: 硬拒短消息 + 语气词
    if len(msg) <= 2 and msg.lower() in MEMORY_FILLER_WORDS:
        return False
    if msg.lower() in MEMORY_FILLER_WORDS:
        return False

    msg_lower = msg.lower()
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', message))
    latin_words = _word_set(msg)
    non_space_chars = len(re.sub(r"\s+", "", msg))
    total_weight = 0

    # 强信号：明确的个人资料/自我披露，直接放行。
    if any(p.search(msg_lower if "i" in p.pattern or "my" in p.pattern else msg) for p in CORE_PROFILE_PATTERNS):
        return True

    # Rule 1: 基础长度 (w=1)
    if chinese_chars >= 5 or non_space_chars >= 12:
        total_weight += 1

    # Rule 2: 第一人称 (w=1)
    if any(p in msg for p in FIRST_PERSON_TERMS) or any(p in latin_words for p in FIRST_PERSON_TERMS_EN):
        total_weight += 1

    # Rule 3: 情感词 (w=2)
    if any(w in msg for w in MEMORY_EMOTION_WORDS) or any(w in latin_words for w in MEMORY_EMOTION_WORDS_EN):
        total_weight += 2

    # Rule 4: 时间词 (w=1)
    if any(w in msg for w in MEMORY_TIME_WORDS) or any(w in msg_lower for w in MEMORY_TIME_WORDS_EN):
        total_weight += 1

    # Rule 5: 事实词 (w=1)
    if any(w in msg for w in MEMORY_FACT_WORDS) or any(w in latin_words for w in MEMORY_FACT_WORDS_EN):
        total_weight += 1

    # Rule 6: 自我暴露句式 (w=2)
    if any(p.search(msg) for p in SELF_DISCLOSURE_PATTERNS) or any(
        p.search(msg_lower) for p in SELF_DISCLOSURE_PATTERNS_EN
    ):
        total_weight += 2

    # Rule 7: 长消息/含数字的简历式事实 (w=1)
    if chinese_chars >= 30 or non_space_chars >= 30:
        total_weight += 1
    if re.search(r"\d", msg) and (
        any(p in msg for p in FIRST_PERSON_TERMS)
        or any(p in latin_words for p in FIRST_PERSON_TERMS_EN)
        or any(w in msg for w in {"岁", "年", "月", "天"})
        or any(w in latin_words for w in {"years", "year", "old"})
    ):
        total_weight += 1

    return total_weight >= 2
