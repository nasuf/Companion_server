"""Hybrid retrieval orchestrator.

Combines vector search + graph queries for comprehensive memory retrieval.

Pipeline (no LLM calls — pure data operations):

  parallel(vector_search + graph_context) -> fusion -> ranker -> context_selector

Includes Redis caching for retrieval results and graph context.
"""

import asyncio
import logging
from datetime import datetime

from app.services.memory.query_analyzer import analyze_query
from app.services.memory.vector_search import search_similar, search_by_time_range
from app.services.memory.ranker import rank_memories
from app.services.memory.context_selector import select_context
from app.services.graph.queries import get_relationship_context
from app.services.runtime.cache import (
    cache_retrieval,
    cache_set_retrieval,
    cache_graph_context,
    cache_set_graph_context,
)

logger = logging.getLogger(__name__)


async def hybrid_retrieve(
    message: str,
    user_id: str,
    workspace_id: str | None = None,
    token_budget: int = 800,
) -> dict:
    """Perform hybrid retrieval and return context for prompt.

    No LLM calls — only vector search + graph queries + ranking.

    Returns dict with:
      - memories: list[str] (formatted for prompt)
      - graph_context: dict (topics, entities)
    """
    # Check cache
    cached = await cache_retrieval(message, user_id, workspace_id=workspace_id)
    if cached:
        logger.debug("Hybrid retrieval cache hit")
        return cached

    analysis = await analyze_query(message)
    retrieve_memory = bool(analysis.get("retrieve_memory", True))
    retrieve_graph = bool(analysis.get("retrieve_graph", False))
    context_entities = analysis.get("entities", []) or []
    context_categories = analysis.get("main_categories", []) or []
    context_sub_categories = analysis.get("sub_categories", []) or []
    levels = analysis.get("levels", [2, 3]) or [2, 3]
    time_range = analysis.get("time_range")

    # Parallel: vector search + graph context + optional time search
    vector_task = (
        search_similar(
            message,
            user_id,
            top_k=50,
            workspace_id=workspace_id,
            main_categories=context_categories,
            sub_categories=context_sub_categories,
            levels=levels,
        )
        if retrieve_memory else asyncio.sleep(0, result=[])
    )
    graph_task = (
        get_relationship_context(
            user_id,
            workspace_id=workspace_id,
            main_categories=context_categories,
            sub_categories=context_sub_categories,
        )
        if retrieve_graph else asyncio.sleep(0, result=None)
    )
    time_task = (
        search_by_time_range(
            user_id,
            datetime.fromisoformat(time_range["start"]),
            datetime.fromisoformat(time_range["end"]),
            limit=20,
            workspace_id=workspace_id,
        )
        if time_range else asyncio.sleep(0, result=[])
    )

    vector_results, graph_result, time_results = await asyncio.gather(
        vector_task, graph_task, time_task, return_exceptions=True
    )

    # Merge vector + time results (union by id)
    all_candidates: list[dict] = []
    seen_ids: set[str] = set()
    for source_results, label in [(vector_results, "vector"), (time_results, "time")]:
        if isinstance(source_results, Exception):
            logger.warning(f"{label} search failed: {source_results}")
            continue
        for mem in (source_results or []):
            mid = mem.get("id", "")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                all_candidates.append(mem)

    # Rank (no entity context needed without query analyzer)
    ranked = rank_memories(
        all_candidates,
        context_entities=context_entities,
        context_categories=context_categories,
        context_sub_categories=context_sub_categories,
        top_k=20,
    )

    # Select within token budget
    memory_strings = select_context(ranked, token_budget)

    # Graph context (with caching)
    graph_context = None
    if isinstance(graph_result, Exception):
        logger.warning(f"Graph context failed: {graph_result}")
        graph_context = await cache_graph_context(user_id, workspace_id=workspace_id)
    else:
        graph_context = graph_result
        if graph_context:
            try:
                await cache_set_graph_context(user_id, graph_context, workspace_id=workspace_id)
            except Exception:
                pass

    result = {
        "memories": memory_strings if memory_strings else None,
        "graph_context": graph_context,
        "analysis": analysis,
    }

    # Cache the result
    try:
        await cache_set_retrieval(message, user_id, result, workspace_id=workspace_id)
    except Exception:
        pass

    return result
