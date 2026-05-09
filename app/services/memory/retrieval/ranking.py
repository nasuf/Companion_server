"""Lightweight memory reranking helpers.

The vector model is the broad recall layer. These helpers add small,
deterministic signals that matter in companion chat: literal keyword overlap,
topic/category alignment, and safety/emotional context.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.memory.polarity import query_semantic_conflict_reasons
from app.services.memory.retrieval.query_patterns import asks_ai_stable_relation
from app.services.memory.retrieval.relevance import compute_display_score


SAFETY_QUERY_KEYWORDS: tuple[str, ...] = (
    "想死", "不想活", "活不下去", "活着没意思", "活着没意义",
    "轻生", "自杀", "自残", "自伤", "跳楼", "跳河", "跳桥", "跳轨",
    "结束生命", "结束自己", "了结自己", "消失算了", "撑不住",
)

DISTRESS_KEYWORDS: tuple[str, ...] = (
    "难过", "委屈", "崩溃", "压力", "焦虑", "抑郁", "孤独",
    "想哭", "哭", "绝望", "痛苦", "撑不住", "心情不好", "很累",
    "低落", "沮丧", "受不了", "空落落", "空唠唠", "心里空", "有点空",
)

EMOTIONAL_SAFETY_SUBCATEGORIES: tuple[str, ...] = (
    "悲伤", "恐惧", "焦虑", "失望", "孤独",
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

_NAME_QUERY_TERMS: tuple[str, ...] = ("叫什么", "名字", "姓名", "叫啥", "叫作")
_NAMED_MEMORY_TERMS: tuple[str, ...] = ("叫", "名字", "姓名")
_THIRD_PERSON_TERMS: tuple[str, ...] = ("她", "他", "ta", "TA", "对方", "那个人")
_RELATION_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("老板", "上司", "领导", "直属领导", "主管", "经理", "leader"),
    ("妈妈", "母亲", "妈"),
    ("爸爸", "父亲", "爸"),
    ("妻子", "老婆", "太太"),
    ("丈夫", "老公", "先生"),
    ("男朋友", "男友"),
    ("女朋友", "女友"),
)
_RELATION_SUBCATEGORIES: tuple[str, ...] = ("社会关系", "亲属关系")
_USER_SELF_TERMS: tuple[str, ...] = ("我", "我的", "我最近", "我之前")
_PREFERENCE_QUERY_TERMS: tuple[str, ...] = (
    "喜欢", "不喜欢", "讨厌", "爱吃", "不吃", "偏好", "雷区", "习惯", "口味",
)
_REMINDER_QUERY_TERMS: tuple[str, ...] = (
    "提醒", "待办", "事项", "闹钟", "review", "复盘",
)
_REMINDER_SUBCATEGORIES: tuple[str, ...] = ("提醒",)
_IDENTITY_QUERY_TERMS: tuple[str, ...] = (
    "基本信息", "个人信息", "信息", "资料", "名字", "姓名", "叫什么",
    "年龄", "几岁", "生日", "住哪", "哪里人", "职业", "性别", "身高",
    "体型",
)
_USER_IDENTITY_SUBCATEGORIES: tuple[str, ...] = (
    "姓名", "年龄", "生日", "现居地", "职业/与经济", "性别", "身高",
    "体型", "居住", "其他",
)

_EXACT_TEXT_MATCH_FLOOR = 1.20
_HIGH_SIMILARITY_THRESHOLD = 0.86
_HIGH_SIMILARITY_FLOOR = 0.94


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


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _is_exact_or_contained_match(query: str, memory_text: str) -> bool:
    q = _compact(query)
    m = _compact(memory_text)
    return bool(q and m and (q == m or q in m or m in q))


def _is_name_query(query: str) -> bool:
    return contains_any(query, _NAME_QUERY_TERMS)


def _is_named_relation_query(query: str) -> bool:
    return _is_name_query(query) and (
        contains_any(query, _THIRD_PERSON_TERMS)
        or any(contains_any(query, aliases) for aliases in _RELATION_ALIAS_GROUPS)
    )


def _is_named_relation_memory(memory: dict[str, Any]) -> bool:
    text = _memory_text(memory)
    return (
        memory.get("source") == "user"
        and (
            memory.get("sub_category") in _RELATION_SUBCATEGORIES
            or any(contains_any(text, aliases) for aliases in _RELATION_ALIAS_GROUPS)
        )
        and contains_any(text, _NAMED_MEMORY_TERMS)
    )


def _is_user_self_name_memory(memory: dict[str, Any]) -> bool:
    return (
        memory.get("source") == "user"
        and memory.get("main_category") == "身份"
        and memory.get("sub_category") == "姓名"
    )


def _is_user_preference_query(query: str) -> bool:
    return contains_any(query, _USER_SELF_TERMS) and contains_any(
        query, _PREFERENCE_QUERY_TERMS,
    )


def _is_user_reminder_query(query: str) -> bool:
    return contains_any(query, _USER_SELF_TERMS) and contains_any(
        query, _REMINDER_QUERY_TERMS,
    )


def _is_user_identity_query(query: str) -> bool:
    return (
        not _is_named_relation_query(query)
        and contains_any(query, _USER_SELF_TERMS)
        and contains_any(query, _IDENTITY_QUERY_TERMS)
    )


def _is_user_preference_memory(memory: dict[str, Any]) -> bool:
    text = _memory_text(memory)
    return (
        memory.get("source") == "user"
        and (
            memory.get("main_category") == "偏好"
            or contains_any(text, _PREFERENCE_QUERY_TERMS)
            or "过敏" in text
        )
    )


def _is_user_reminder_memory(memory: dict[str, Any]) -> bool:
    text = _memory_text(memory)
    return (
        memory.get("source") == "user"
        and (
            memory.get("sub_category") in _REMINDER_SUBCATEGORIES
            or contains_any(text, _REMINDER_QUERY_TERMS)
        )
    )


def _is_user_identity_memory(memory: dict[str, Any]) -> bool:
    return (
        memory.get("source") == "user"
        and (
            memory.get("main_category") == "身份"
            or memory.get("sub_category") in _USER_IDENTITY_SUBCATEGORIES
        )
    )


def _is_ai_identity_memory(memory: dict[str, Any]) -> bool:
    return (
        memory.get("source") == "ai"
        and (
            memory.get("main_category") == "身份"
            or memory.get("sub_category") in _USER_IDENTITY_SUBCATEGORIES
        )
    )


def _is_ai_self_memory(memory: dict[str, Any]) -> bool:
    return memory.get("source") == "ai"


def _keyword_overlap_count(query: str, memory_text: str) -> int:
    count = sum(1 for kw in _LEXICAL_KEYWORDS if kw in query and kw in memory_text)
    if contains_any(query, _NAME_QUERY_TERMS) and contains_any(
        memory_text, _NAMED_MEMORY_TERMS
    ):
        count += 1
    for aliases in _RELATION_ALIAS_GROUPS:
        if contains_any(query, aliases) and contains_any(memory_text, aliases):
            count += 1
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

    exact_text_match = _is_exact_or_contained_match(query, text)
    if exact_text_match:
        score = max(score, _EXACT_TEXT_MATCH_FLOOR)
        reasons.append("精确文本命中")
    elif similarity >= _HIGH_SIMILARITY_THRESHOLD:
        score = max(score, _HIGH_SIMILARITY_FLOOR)
        reasons.append("高相似向量命中")

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

    if _is_named_relation_query(query) and not exact_text_match:
        if _is_named_relation_memory(memory):
            boost += 0.60
            reasons.append("关系命名相关")
        elif memory.get("source") == "ai":
            boost *= 0.55
            reasons.append("关系名查询降权:AI记忆")
        elif _is_user_self_name_memory(memory):
            boost *= 0.65
            reasons.append("关系名查询降权:用户本人姓名")

    if _is_user_preference_query(query) and not exact_text_match:
        if _is_user_preference_memory(memory):
            boost += 0.45
            reasons.append("用户偏好相关")
        elif memory.get("source") == "ai":
            boost *= 0.65
            reasons.append("用户偏好查询降权:AI记忆")

    if _is_user_reminder_query(query) and not exact_text_match:
        if _is_user_reminder_memory(memory):
            boost += 0.95
            reasons.append("用户提醒相关")
        elif memory.get("source") == "ai":
            boost *= 0.55
            reasons.append("用户提醒查询降权:AI记忆")

    if _is_user_identity_query(query) and not exact_text_match:
        if _is_user_identity_memory(memory):
            boost += 0.55
            reasons.append("用户身份相关")
        elif _is_ai_identity_memory(memory):
            boost *= 0.50
            reasons.append("用户身份查询降权:AI身份记忆")

    if asks_ai_stable_relation(query) and not exact_text_match:
        if _is_ai_self_memory(memory):
            boost += 0.55
            reasons.append("AI自我记忆相关")
        elif memory.get("source") == "user":
            boost *= 0.60
            reasons.append("AI自我查询降权:用户记忆")

    if (
        is_distress_or_safety_query(query)
        and not exact_text_match
        and memory.get("source") == "ai"
        and (
            memory.get("main_category") == "情绪"
            or contains_any(text, SAFETY_QUERY_KEYWORDS + DISTRESS_KEYWORDS)
        )
    ):
        boost *= 0.65
        reasons.append("安全查询降权:AI情绪记忆")

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
