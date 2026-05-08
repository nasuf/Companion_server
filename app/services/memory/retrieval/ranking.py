"""Lightweight memory reranking helpers.

The vector model is the broad recall layer. These helpers add small,
deterministic signals that matter in companion chat: literal keyword overlap,
topic/category alignment, and safety/emotional context.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.memory.polarity import query_semantic_conflict_reasons
from app.services.memory.retrieval.relevance import compute_display_score


SAFETY_QUERY_KEYWORDS: tuple[str, ...] = (
    "想死", "不想活", "活不下去", "活着没意思", "活着没意义",
    "轻生", "自杀", "自残", "自伤", "跳楼", "跳河", "跳桥", "跳轨",
    "结束生命", "结束自己", "了结自己", "消失算了", "撑不住",
)

DISTRESS_KEYWORDS: tuple[str, ...] = (
    "难过", "委屈", "崩溃", "压力", "焦虑", "抑郁", "孤独",
    "想哭", "哭", "绝望", "痛苦", "撑不住", "心情不好", "很累",
    "低落", "沮丧", "受不了",
)

EMOTIONAL_SAFETY_SUBCATEGORIES: tuple[str, ...] = (
    "悲伤", "恐惧", "焦虑", "失望", "孤独", "遗憾",
)

RECALL_HINT_KEYWORDS: tuple[str, ...] = (
    "还记得", "记不记得", "记得", "以前", "之前", "去年", "前年",
    "上次", "那次", "那件事", "当时", "那时候", "很久", "小时候",
    "过去", "曾经", "后来呢", "然后呢",
)

_CATEGORY_QUERY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "情绪": SAFETY_QUERY_KEYWORDS + DISTRESS_KEYWORDS + (
        "开心", "高兴", "生气", "害怕", "恐惧", "失望", "遗憾",
    ),
    "生活": (
        "工作", "上班", "老板", "同事", "学校", "考试", "旅行", "搬家",
        "住院", "出院", "手术", "生病", "健康", "宠物", "生活",
    ),
    "身份": (
        "名字", "几岁", "年龄", "生日", "家人", "妈妈", "爸爸", "父母",
        "妻子", "丈夫", "女朋友", "男朋友", "职业", "住哪", "哪里人",
    ),
    "偏好": (
        "喜欢", "不喜欢", "讨厌", "爱吃", "不吃", "偏好", "雷区",
        "习惯", "口味", "审美",
    ),
    "思维": (
        "想法", "观点", "价值观", "人生", "目标", "理想", "信仰",
        "怎么看", "觉得",
    ),
}

_LEXICAL_KEYWORDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        kw
        for kws in _CATEGORY_QUERY_KEYWORDS.values()
        for kw in kws
        if len(kw) >= 2
    )
)


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


def is_safety_query(query: str) -> bool:
    return contains_any(query, SAFETY_QUERY_KEYWORDS)


def is_distress_or_safety_query(query: str) -> bool:
    return contains_any(query, SAFETY_QUERY_KEYWORDS + DISTRESS_KEYWORDS)


def is_recall_query(query: str) -> bool:
    return contains_any(query, RECALL_HINT_KEYWORDS)


def infer_query_main_categories(query: str) -> set[str]:
    return {
        category
        for category, keywords in _CATEGORY_QUERY_KEYWORDS.items()
        if contains_any(query, keywords)
    }


def is_safety_memory(memory: dict[str, Any]) -> bool:
    text = _memory_text(memory)
    return (
        memory.get("source") == "user"
        and (
            memory.get("sub_category") in EMOTIONAL_SAFETY_SUBCATEGORIES
            or contains_any(text, SAFETY_QUERY_KEYWORDS + DISTRESS_KEYWORDS)
        )
    )


def _memory_text(memory: dict[str, Any]) -> str:
    return f"{memory.get('summary') or ''} {memory.get('content') or ''}".strip()


def _keyword_overlap_count(query: str, memory_text: str) -> int:
    count = sum(1 for kw in _LEXICAL_KEYWORDS if kw in query and kw in memory_text)
    ascii_query_words = set(re.findall(r"[A-Za-z0-9_]{3,}", query.lower()))
    if ascii_query_words:
        ascii_memory_words = set(re.findall(r"[A-Za-z0-9_]{3,}", memory_text.lower()))
        count += len(ascii_query_words & ascii_memory_words)
    return count


def rank_memory_candidate(
    memory: dict[str, Any],
    query: str,
    *,
    default_similarity: float = 1.0,
) -> tuple[float, list[str]]:
    """Return a display rank score plus human-readable rank reasons.

    Boosts are intentionally modest except for safety context. They should
    rescue clearly relevant memories, not drown out vector similarity.
    """
    text = _memory_text(memory)
    raw_similarity = memory.get("similarity")
    similarity = (
        default_similarity
        if raw_similarity is None
        else float(raw_similarity)
    )
    raw_importance = memory.get("importance")
    importance = 0.0 if raw_importance is None else float(raw_importance)
    score = compute_display_score(
        importance=importance,
        last_accessed_at=(
            memory.get("last_accessed_at")
            or memory.get("updated_at")
            or memory.get("created_at")
        ),
        similarity=similarity,
    )

    reasons: list[str] = []
    boost = 1.0

    overlap = _keyword_overlap_count(query, text)
    if overlap:
        boost += min(0.30, 0.08 * overlap)
        reasons.append("关键词命中")

    inferred_categories = infer_query_main_categories(query)
    main_category = memory.get("main_category")
    if main_category and main_category in inferred_categories:
        boost += 0.25
        reasons.append("话题类别匹配")

    retrieval_source = str(memory.get("_retrieval_source") or "")
    if memory.get("_entity_match") or "entity" in retrieval_source:
        boost += 0.20
        reasons.append("实体命中")

    if is_distress_or_safety_query(query) and is_safety_memory(memory):
        boost += 0.80 if is_safety_query(query) else 0.45
        reasons.append("安全/情绪相关")

    mention_count = int(memory.get("mention_count") or 0)
    if mention_count >= 3:
        boost += 0.05
        reasons.append("多次提及")

    score *= boost

    # Crisis/distress phrases often contain "不" ("活不下去") but are not asking
    # for negated preferences/facts. Do not downweight safety memories there.
    if not is_distress_or_safety_query(query):
        conflict_reasons = query_semantic_conflict_reasons(query, text)
        if conflict_reasons:
            severe = any(
                reason in {"否定极性", "偏好立场", "伴侣状态", "伴侣身份", "就医状态"}
                for reason in conflict_reasons
            )
            score *= 0.3 if severe else 0.55
            reasons.append(f"语义对立降权:{'、'.join(conflict_reasons)}")

    return score, reasons
