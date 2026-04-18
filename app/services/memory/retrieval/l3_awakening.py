"""L3 memory awakening flow.

Product spec §3.2 step 2-3 + §3.5: When memory relevance is "strong" and
L1/L2 results are insufficient, or when intent is "recall distant memory",
search L3 memories and inject the top matches into the prompt.

L3 memories never upgrade — they stay L3 forever. But they CAN participate
in recall when explicitly requested.
"""

from __future__ import annotations

import logging

from app.services.memory.retrieval.vector_search import search_by_embedding
from app.services.llm.models import get_embedding_model

logger = logging.getLogger(__name__)

# Spec §3.2 step 3: L3 similarity threshold is lower (0.6 vs 0.7 for L1/L2)
_L3_SIMILARITY_THRESHOLD = 0.6
_L3_MAX_CANDIDATES = 30
_L3_INJECT_LIMIT = 5


async def search_l3_memories(
    query: str,
    user_id: str,
    workspace_id: str | None = None,
) -> list[dict]:
    """Search L3 memories by vector similarity.

    Returns up to _L3_INJECT_LIMIT results (the spec says: search → top 30
    candidates → inject top 5 into context).
    """
    try:
        model = get_embedding_model()
        query_vec = await model.aembed_query(query)
    except Exception as e:
        logger.warning(f"L3 embedding failed: {e}")
        return []

    candidates = await search_by_embedding(
        query_vec,
        user_id,
        top_k=_L3_MAX_CANDIDATES,
        workspace_id=workspace_id,
        levels=[3],
    )

    # Filter by similarity threshold
    filtered = [
        r for r in candidates
        if r.get("similarity", 0) >= _L3_SIMILARITY_THRESHOLD
    ]

    # Return top N for prompt injection
    return filtered[:_L3_INJECT_LIMIT]


def should_awaken_l3(l1_l2_count: int) -> bool:
    """Should we search L3 memories?

    The relevance classifier already gates this to "strong" relevance only.
    Within that gate, we awaken L3 when L1+L2 results are sparse —
    suggesting the user is asking about something not in recent memory.
    """
    return l1_l2_count < 3
