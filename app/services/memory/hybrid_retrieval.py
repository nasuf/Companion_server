"""Hybrid retrieval orchestrator.

Combines vector search + graph queries for comprehensive memory retrieval.

Pipeline (no LLM calls — pure data operations):

  parallel(vector_search + graph_context) -> fusion -> ranker -> context_selector

Includes Redis caching for retrieval results and graph context.
"""

import asyncio
import logging

from app.services.memory.vector_search import search_similar
from app.services.memory.ranker import rank_memories
from app.services.memory.context_selector import select_context
from app.services.graph.queries import get_relationship_context
from app.services.cache import (
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

    # Parallel: vector search + graph context (no LLM needed)
    vector_task = search_similar(message, user_id, top_k=50, workspace_id=workspace_id)
    graph_task = get_relationship_context(user_id, workspace_id=workspace_id)

    vector_results, graph_result = await asyncio.gather(
        vector_task, graph_task, return_exceptions=True
    )

    # Process vector results
    all_candidates: list[dict] = []
    if isinstance(vector_results, Exception):
        logger.warning(f"Vector search failed: {vector_results}")
    else:
        for mem in vector_results:
            all_candidates.append(mem)

    # Rank (no entity context needed without query analyzer)
    ranked = rank_memories(all_candidates, context_entities=[], top_k=20)

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
    }

    # Cache the result
    try:
        await cache_set_retrieval(message, user_id, result, workspace_id=workspace_id)
    except Exception:
        pass

    return result
