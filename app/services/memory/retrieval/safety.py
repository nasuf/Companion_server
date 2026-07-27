"""Safety-sensitive memory retrieval.

Crisis replies should not depend on generic mixed-source memory selection. This
module retrieves user-side emotional/safety memories directly and ranks them
with the same lightweight signals as the normal retriever.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.db import db
from app.services.memory.retrieval.context_selector import ClassifiedMemory, select_context
from app.services.memory.retrieval.hybrid import hybrid_retrieve
from app.services.memory.retrieval.ranking import (
    DISTRESS_KEYWORDS,
    EMOTIONAL_SAFETY_SUBCATEGORIES,
    SAFETY_QUERY_KEYWORDS,
    rank_memory_candidate,
)
from app.services.memory.retrieval.trace import record_retrieval_session
from app.services.memory.retrieval.vector_search import search_similar
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

_CRISIS_LEVELS = [1, 2]
_CRISIS_VECTOR_TOP_K = 80
_CRISIS_KEYWORD_LIMIT = 40
_CRISIS_TOKEN_BUDGET = 500
_CRISIS_FOLLOWUP_SAFETY_LIMIT = 3
_CRISIS_FOLLOWUP_TOPICAL_LIMIT = 5
_CRISIS_FOLLOWUP_TOPIC_BUDGET = 700
_CRISIS_SAFETY_REASON = "保护槽:危机安全背景"
_CRISIS_TOPICAL_REASON = "保护槽:当前话题"
_TOPIC_CONTEXT_MAX_LINES = 8
_TOPIC_FALLBACK_MIN_SCORE = 0.30
_TOPIC_FALLBACK_CATEGORIES = {"生活", "偏好", "身份", "情绪", "思维"}
_TOPIC_FALLBACK_SAFETY_SUBCATEGORIES = set(EMOTIONAL_SAFETY_SUBCATEGORIES)


async def _search_crisis_keyword_memories(
    user_id: str,
    workspace_id: str | None,
    *,
    limit: int = _CRISIS_KEYWORD_LIMIT,
) -> list[dict]:
    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    patterns = [f"%{kw}%" for kw in SAFETY_QUERY_KEYWORDS + DISTRESS_KEYWORDS]
    emotion_subcategories = list(EMOTIONAL_SAFETY_SUBCATEGORIES)
    return await db.query_raw(
        """
        SELECT id, content, level, importance, mention_count,
               type, main_category, sub_category,
               created_at, updated_at,
               COALESCE(updated_at, created_at) AS last_accessed_at,
               'user' AS source,
               1.0 AS similarity
        FROM memories_user
        WHERE user_id = $1
          AND workspace_id = $2
          AND is_archived = false
          AND level = ANY($3::int[])
          AND (
              sub_category = ANY($4::text[])
              OR content ILIKE ANY($5::text[])
          )
        ORDER BY importance DESC, updated_at DESC NULLS LAST, created_at DESC
        LIMIT $6
        """,
        user_id,
        workspace_id,
        _CRISIS_LEVELS,
        emotion_subcategories,
        patterns,
        limit,
    )


async def _collect_crisis_memory_candidates(
    message: str,
    user_id: str,
    *,
    workspace_id: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    vector_task = search_similar(
        message,
        user_id,
        top_k=_CRISIS_VECTOR_TOP_K,
        workspace_id=workspace_id,
        levels=_CRISIS_LEVELS,
    )
    keyword_task = _search_crisis_keyword_memories(user_id, workspace_id)
    vector_results, keyword_results = await asyncio.gather(
        vector_task, keyword_task, return_exceptions=True,
    )

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_results, source_label in (
        (vector_results, "vector"),
        (keyword_results, "safety_keyword"),
    ):
        if isinstance(source_results, Exception):
            logger.warning(f"crisis memory {source_label} search failed: {source_results}")
            continue
        for row in source_results or []:
            if row.get("source") != "user":
                continue
            mid = row.get("id", "")
            if not mid or mid in seen:
                continue
            row = dict(row)
            row["_retrieval_source"] = source_label
            score, reasons = rank_memory_candidate(row, message)
            row["rank_score"] = score
            row["rank_reasons"] = reasons
            seen.add(mid)
            candidates.append(row)

    candidates.sort(key=lambda r: float(r.get("rank_score", 0)), reverse=True)
    raw_count = (
        (len(vector_results) if isinstance(vector_results, list) else 0)
        + (len(keyword_results) if isinstance(keyword_results, list) else 0)
    )
    return candidates, raw_count


async def retrieve_crisis_memories(
    message: str,
    user_id: str,
    *,
    workspace_id: str | None = None,
    limit: int = 5,
) -> list[ClassifiedMemory]:
    """Return safety-relevant user memories for a crisis reply.

    The vector side keeps semantic recall; the keyword/category side guarantees
    that known emotional or self-harm memories are candidates even if generic
    L1 facts would otherwise crowd them out.
    """
    candidates, raw_count = await _collect_crisis_memory_candidates(
        message,
        user_id,
        workspace_id=workspace_id,
    )
    selected = select_context(
        candidates,
        token_budget=_CRISIS_TOKEN_BUDGET,
        max_items=limit,
        query=message,
    )
    record_retrieval_session(
        strategy="crisis_safety",
        query=message,
        workspace_id=workspace_id,
        raw_count=raw_count,
        candidate_count=len(candidates),
        selected_count=len(selected),
        candidates=candidates,
        selected=selected,
        notes={
            "vector_top_k": _CRISIS_VECTOR_TOP_K,
            "keyword_limit": _CRISIS_KEYWORD_LIMIT,
        },
    )
    return selected


def _append_memory_reason(memory: ClassifiedMemory, reason: str) -> ClassifiedMemory:
    reasons = list(memory.rank_reasons or [])
    if reason not in reasons:
        reasons.append(reason)
        memory.rank_reasons = reasons
    return memory


def _memory_text(row: dict[str, Any]) -> str:
    return str(row.get("content") or "")


def _is_safety_memory_row(row: dict[str, Any]) -> bool:
    text = _memory_text(row)
    reasons = {str(reason) for reason in (row.get("rank_reasons") or [])}
    return (
        "安全/情绪相关" in reasons
        or (
            row.get("main_category") == "情绪"
            and row.get("sub_category") in _TOPIC_FALLBACK_SAFETY_SUBCATEGORIES
        )
        or any(term in text for term in SAFETY_QUERY_KEYWORDS + DISTRESS_KEYWORDS)
    )


def _build_followup_topic_query(message: str, recent_context: str) -> str:
    """Build a topic-focused query for crisis aftercare retrieval.

    The safety channel already receives the full crisis context. The topical
    channel should keep the user's attempted subject change, not let older
    self-harm lines dominate ranking.
    """
    safety_terms = tuple(SAFETY_QUERY_KEYWORDS + DISTRESS_KEYWORDS)
    punctuation = "用户: AI: ，。！？!?、,.；;：:（）()[]【】\"'“”‘’… \t"

    def strip_safety_terms(line: str) -> str:
        cleaned = line.strip()
        for term in safety_terms:
            cleaned = cleaned.replace(term, " ")
        return " ".join(cleaned.split()).strip(punctuation)

    user_topic_lines: list[str] = []
    other_topic_lines: list[str] = []
    for line in (recent_context or "").splitlines():
        cleaned = strip_safety_terms(line)
        if not cleaned:
            continue
        if line.lstrip().startswith("用户:"):
            user_topic_lines.append(cleaned)
        elif not line.lstrip().startswith("AI:"):
            other_topic_lines.append(cleaned)
    topic_lines = user_topic_lines or other_topic_lines
    topic_lines = topic_lines[-_TOPIC_CONTEXT_MAX_LINES:]
    parts = [*topic_lines, message.strip()]
    return "\n".join(part for part in parts if part).strip() or message


def _select_topical_from_candidates(
    candidates: list[dict[str, Any]],
    topic_query: str,
    *,
    limit: int,
    seen_keys: set[str],
) -> list[ClassifiedMemory]:
    reranked: list[dict[str, Any]] = []
    for row in candidates:
        if row.get("source") == "ai":
            continue
        if _is_safety_memory_row(row):
            continue
        main_category = row.get("main_category")
        if main_category and main_category not in _TOPIC_FALLBACK_CATEGORIES:
            continue
        item = dict(row)
        score, reasons = rank_memory_candidate(item, topic_query)
        if score < _TOPIC_FALLBACK_MIN_SCORE and float(item.get("similarity", 0) or 0) < 0.5:
            continue
        item["rank_score"] = score
        item["rank_reasons"] = reasons
        reranked.append(item)

    reranked.sort(key=lambda item: float(item.get("rank_score", 0)), reverse=True)
    selected = select_context(
        reranked,
        token_budget=_CRISIS_FOLLOWUP_TOPIC_BUDGET,
        max_items=limit,
        query=topic_query,
    )
    result: list[ClassifiedMemory] = []
    for memory in selected:
        key = memory.id or memory.text
        if not key or key in seen_keys:
            continue
        result.append(memory)
        if len(result) >= limit:
            break
    return result


async def retrieve_crisis_followup_memories(
    message: str,
    user_id: str,
    *,
    recent_context: str = "",
    workspace_id: str | None = None,
    safety_limit: int = _CRISIS_FOLLOWUP_SAFETY_LIMIT,
    topical_limit: int = _CRISIS_FOLLOWUP_TOPICAL_LIMIT,
) -> list[ClassifiedMemory]:
    """Return memory for crisis aftercare as two channels.

    Crisis follow-up is still safety-sensitive, but the user may intentionally
    move to another topic to regulate emotion. Keep a small safety background
    channel and retrieve current-topic memories with the normal hybrid path so
    the reply can stay connected to what the user is actually talking about.
    """
    safety_query = f"{recent_context}\n{message}".strip()
    topic_query = _build_followup_topic_query(message, recent_context)
    crisis_candidates_task = _collect_crisis_memory_candidates(
        safety_query or message,
        user_id,
        workspace_id=workspace_id,
    )
    topical_task = hybrid_retrieve(
        message,
        user_id,
        workspace_id=workspace_id,
        token_budget=_CRISIS_FOLLOWUP_TOPIC_BUDGET,
        enhanced_query=topic_query if topic_query != message else None,
    )
    crisis_candidate_result, topical_result = await asyncio.gather(
        crisis_candidates_task,
        topical_task,
        return_exceptions=True,
    )

    merged: list[ClassifiedMemory] = []
    seen: set[str] = set()
    crisis_candidates: list[dict[str, Any]] = []
    crisis_raw_count = 0

    def add(memory: ClassifiedMemory, reason: str) -> None:
        key = memory.id or memory.text
        if not key or key in seen:
            return
        seen.add(key)
        merged.append(_append_memory_reason(memory, reason))

    if isinstance(crisis_candidate_result, Exception):
        logger.warning(
            f"crisis followup candidate memory search failed: {crisis_candidate_result}"
        )
    else:
        crisis_candidates, crisis_raw_count = crisis_candidate_result
        safety_candidates = [
            row for row in crisis_candidates
            if _is_safety_memory_row(row)
        ]
        safety_result = select_context(
            safety_candidates,
            token_budget=_CRISIS_TOKEN_BUDGET,
            max_items=safety_limit,
            query=safety_query or message,
        )
        record_retrieval_session(
            strategy="crisis_safety",
            query=safety_query or message,
            workspace_id=workspace_id,
            raw_count=crisis_raw_count,
            candidate_count=len(crisis_candidates),
            selected_count=len(safety_result),
            candidates=crisis_candidates,
            selected=safety_result,
            notes={
                "vector_top_k": _CRISIS_VECTOR_TOP_K,
                "keyword_limit": _CRISIS_KEYWORD_LIMIT,
                "followup_topic_query": topic_query[:120],
            },
        )
        for memory in safety_result or []:
            add(memory, _CRISIS_SAFETY_REASON)

    topical_added = 0
    if isinstance(topical_result, Exception):
        logger.warning(f"crisis followup topical memory search failed: {topical_result}")
    else:
        topical_memories = (
            topical_result.get("memories") if isinstance(topical_result, dict) else []
        ) or []
        for memory in topical_memories:
            if topical_added >= topical_limit:
                break
            if isinstance(memory, ClassifiedMemory):
                if memory.source == "ai":
                    continue
                before = len(merged)
                add(memory, _CRISIS_TOPICAL_REASON)
                if len(merged) > before:
                    topical_added += 1

    if topical_added < topical_limit and crisis_candidates:
        fallback_memories = _select_topical_from_candidates(
            crisis_candidates,
            topic_query,
            limit=topical_limit - topical_added,
            seen_keys=seen,
        )
        for memory in fallback_memories:
            add(memory, _CRISIS_TOPICAL_REASON)

    return merged
