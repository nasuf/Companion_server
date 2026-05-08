"""Context selector.

Selects memories to fit within the 800-token prompt budget.
Classifies each memory by relevance: strong (score ≥ 0.7) / medium (0.4-0.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

MemorySource = Literal["user", "ai"]


@dataclass
class ClassifiedMemory:
    """记忆项，附带相关度分级。

    source 区分"用户告诉过你的事" (memories_user) vs "你自己的人设/经历"
    (memories_ai). 混在一起喂给 LLM 会让 AI 把自己的记忆当成用户在描述自己,
    触发人设串戏 → 回退到 AI 助手 persona. 下游 prompt_builder 和分级 tier
    prompt (强/中) 都依赖这个字段做双槽分流.
    """
    text: str
    relevance: str  # "strong" | "medium"
    score: float
    id: str = ""  # memory row ID for access logging
    importance: float = 0.5
    similarity: float = 0.8
    mention_count: int = 0
    main_category: str | None = None
    sub_category: str | None = None
    created_at: datetime | str | None = None
    last_accessed_at: datetime | str | None = None
    display_score: float = 0.0  # set by reranking in orchestrator
    rank_reasons: list[str] | None = None
    source: MemorySource = "user"  # 上游 vector_search 必填


def split_by_source(
    mems: list[ClassifiedMemory] | None,
) -> tuple[list[str], list[str]]:
    """按 source 拆 (user_texts, ai_texts). 用于 prompt 的双槽分流."""
    user_t: list[str] = []
    ai_t: list[str] = []
    for m in mems or []:
        (ai_t if m.source == "ai" else user_t).append(m.text)
    return user_t, ai_t


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Chinese: ~1.5 token per char; ASCII: ~0.25 token per char."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_chars = len(text) - cjk
    return int(cjk * 1.5 + ascii_chars * 0.25)


MAX_MEMORIES_INJECTED = 10  # spec §3.2 step 4: 前 10 条硬上限
_SAFETY_MEMORY_QUOTA = 3
_HIGH_SIMILARITY_MEMORY_QUOTA = 2
_KEYWORD_USER_MEMORY_QUOTA = 2
_NAMED_RELATION_MEMORY_QUOTA = 1
_MIN_USER_MEMORY_QUOTA = 3
_MIN_PROTECTED_SCORE = 0.35
_HIGH_SIMILARITY_THRESHOLD = 0.86
_SAFETY_REASON = "安全/情绪相关"
_HIGH_SIMILARITY_REASON = "高相似向量命中"
_KEYWORD_REASON = "关键词命中"
_PROTECTED_SAFETY_REASON = "保护槽:安全情绪"
_PROTECTED_HIGH_SIMILARITY_REASON = "保护槽:高相似向量"
_PROTECTED_KEYWORD_REASON = "保护槽:字面命中"
_PROTECTED_NAMED_RELATION_REASON = "保护槽:关系命名"
_PROTECTED_USER_REASON = "保护槽:用户记忆"
_EMOTIONAL_SUBCATEGORIES = {"悲伤", "恐惧", "焦虑", "失望", "孤独"}
_NAME_QUERY_TERMS = ("叫什么", "名字", "姓名", "叫啥", "叫作")
_THIRD_PERSON_TERMS = ("她", "他", "ta", "TA", "对方", "那个人")
_RELATION_TERMS = (
    "老板", "上司", "领导", "直属领导", "主管", "经理",
    "妈妈", "母亲", "爸爸", "父亲",
    "妻子", "老婆", "丈夫", "老公",
    "男朋友", "男友", "女朋友", "女友",
)
_RELATION_SUBCATEGORIES = {"社会关系", "亲属关系"}


def _memory_text(mem: dict) -> str:
    return mem.get("summary") or mem.get("content") or ""


def _memory_key(mem: dict) -> str:
    return str(mem.get("id") or _memory_text(mem))


def _memory_score(mem: dict) -> float:
    return float(mem.get("rank_score", mem.get("score", 0.5)) or 0.0)


def _memory_similarity(mem: dict) -> float:
    return float(mem.get("similarity", 0.0) or 0.0)


def _memory_source(mem: dict) -> MemorySource:
    return "ai" if mem.get("source") == "ai" else "user"


def _rank_reasons(mem: dict) -> set[str]:
    return {str(reason) for reason in (mem.get("rank_reasons") or [])}


def _append_rank_reason(mem: dict, reason: str) -> None:
    reasons = list(mem.get("rank_reasons") or [])
    if reason not in reasons:
        reasons.append(reason)
        mem["rank_reasons"] = reasons


def _is_safety_memory(mem: dict) -> bool:
    if _memory_source(mem) != "user":
        return False
    reasons = _rank_reasons(mem)
    importance = float(mem.get("importance", 0.0) or 0.0)
    return (
        _SAFETY_REASON in reasons
        or (
            mem.get("sub_category") in _EMOTIONAL_SUBCATEGORIES
            and importance >= 0.65
        )
    )


def _is_keyword_user_memory(mem: dict) -> bool:
    return (
        _memory_source(mem) == "user"
        and _KEYWORD_REASON in _rank_reasons(mem)
        and _memory_score(mem) >= _MIN_PROTECTED_SCORE
    )


def _is_eligible_user_memory(mem: dict) -> bool:
    return _memory_source(mem) == "user" and _memory_score(mem) >= _MIN_PROTECTED_SCORE


def _is_high_similarity_memory(mem: dict) -> bool:
    return (
        _memory_similarity(mem) >= _HIGH_SIMILARITY_THRESHOLD
        or _HIGH_SIMILARITY_REASON in _rank_reasons(mem)
        or "精确文本命中" in _rank_reasons(mem)
    )


def _is_named_relation_query(query: str | None) -> bool:
    if not query:
        return False
    return any(term in query for term in _NAME_QUERY_TERMS) and (
        any(term in query for term in _THIRD_PERSON_TERMS)
        or any(term in query for term in _RELATION_TERMS)
    )


def _is_named_relation_memory(mem: dict) -> bool:
    if _memory_source(mem) != "user" or _memory_score(mem) < _MIN_PROTECTED_SCORE:
        return False
    text = _memory_text(mem)
    return (
        mem.get("sub_category") in _RELATION_SUBCATEGORIES
        or any(term in text for term in _RELATION_TERMS)
    ) and ("叫" in text or "名字" in text or "姓名" in text)


def _to_classified_memory(mem: dict) -> ClassifiedMemory:
    text = _memory_text(mem)
    score = _memory_score(mem)
    relevance = "strong" if score >= 0.7 else "medium"
    return ClassifiedMemory(
        text=text,
        relevance=relevance,
        score=score,
        id=mem.get("id", ""),
        importance=float(mem.get("importance", 0.5)),
        similarity=float(mem.get("similarity", 0.8)),
        mention_count=int(mem.get("mention_count") or 0),
        main_category=mem.get("main_category"),
        sub_category=mem.get("sub_category"),
        created_at=mem.get("created_at"),
        last_accessed_at=(
            mem.get("last_accessed_at")
            or mem.get("updated_at")
            or mem.get("created_at")
        ),
        display_score=score,
        rank_reasons=list(mem.get("rank_reasons") or []),
        source=_memory_source(mem),
    )


def select_context(
    ranked_memories: list[dict],
    token_budget: int = 800,
    max_items: int = MAX_MEMORIES_INJECTED,
    query: str | None = None,
) -> list[ClassifiedMemory]:
    """Select memories to fit within token budget, with relevance classification.

    spec §3.2 step 4: 前 `max_items` 条 + 不超过 `token_budget` tokens。
    两条限制取较严的一个。

    Classification:
    - strong: rank_score ≥ 0.7 → "你清楚记得的事"
    - medium: 0.4 ≤ rank_score < 0.7 → "你有印象的事"

    Returns list of ClassifiedMemory.
    """
    if max_items <= 0 or token_budget <= 0:
        return []

    selected_rows: list[dict] = []
    used_tokens = 0
    seen_ids: set[str] = set()

    def try_add(mem: dict, protected_reason: str | None = None) -> bool:
        nonlocal used_tokens
        if len(selected_rows) >= max_items:
            return False
        key = _memory_key(mem)
        if not key or key in seen_ids:
            return False
        text = _memory_text(mem)
        if not text:
            return False
        tokens = estimate_tokens(text)
        if used_tokens + tokens > token_budget:
            return False
        if protected_reason:
            _append_rank_reason(mem, protected_reason)
        seen_ids.add(key)
        selected_rows.append(mem)
        used_tokens += tokens
        return True

    safety_added = 0
    for mem in ranked_memories:
        if safety_added >= min(_SAFETY_MEMORY_QUOTA, max_items):
            break
        if _is_safety_memory(mem) and try_add(mem, _PROTECTED_SAFETY_REASON):
            safety_added += 1

    named_relation_added = 0
    if _is_named_relation_query(query):
        for mem in ranked_memories:
            if named_relation_added >= min(
                _NAMED_RELATION_MEMORY_QUOTA,
                max_items - len(selected_rows),
            ):
                break
            if _is_named_relation_memory(mem) and try_add(
                mem, _PROTECTED_NAMED_RELATION_REASON,
            ):
                named_relation_added += 1

    high_similarity_added = 0
    high_similarity_candidates = sorted(
        ranked_memories,
        key=lambda mem: _memory_similarity(mem),
        reverse=True,
    )
    for mem in high_similarity_candidates:
        if high_similarity_added >= min(
            _HIGH_SIMILARITY_MEMORY_QUOTA,
            max_items - len(selected_rows),
        ):
            break
        if _is_high_similarity_memory(mem) and try_add(
            mem,
            _PROTECTED_HIGH_SIMILARITY_REASON,
        ):
            high_similarity_added += 1

    keyword_added = 0
    for mem in ranked_memories:
        if keyword_added >= min(_KEYWORD_USER_MEMORY_QUOTA, max_items - len(selected_rows)):
            break
        if _is_keyword_user_memory(mem) and try_add(mem, _PROTECTED_KEYWORD_REASON):
            keyword_added += 1

    min_user_quota = min(_MIN_USER_MEMORY_QUOTA, max_items)
    selected_user_count = sum(1 for mem in selected_rows if _memory_source(mem) == "user")
    if selected_user_count < min_user_quota:
        for mem in ranked_memories:
            if selected_user_count >= min_user_quota:
                break
            if _is_eligible_user_memory(mem) and try_add(mem, _PROTECTED_USER_REASON):
                selected_user_count += 1

    for mem in ranked_memories:
        if len(selected_rows) >= max_items:
            break
        try_add(mem)

    return [_to_classified_memory(mem) for mem in selected_rows]
