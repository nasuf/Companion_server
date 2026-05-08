"""Safety-sensitive memory retrieval.

Crisis replies should not depend on the generic top-10 memory budget. This
module retrieves user-side emotional/safety memories directly and ranks them
with the same lightweight signals as the normal retriever.
"""

from __future__ import annotations

import asyncio
import logging

from app.db import db
from app.services.memory.retrieval.context_selector import ClassifiedMemory, select_context
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
        SELECT id, content, summary, level, importance, mention_count,
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
              OR summary ILIKE ANY($5::text[])
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

    candidates: list[dict] = []
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
            row["_retrieval_source"] = source_label
            score, reasons = rank_memory_candidate(row, message)
            row["rank_score"] = score
            row["rank_reasons"] = reasons
            seen.add(mid)
            candidates.append(row)

    candidates.sort(key=lambda r: float(r.get("rank_score", 0)), reverse=True)
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
        raw_count=(
            (len(vector_results) if isinstance(vector_results, list) else 0)
            + (len(keyword_results) if isinstance(keyword_results, list) else 0)
        ),
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
