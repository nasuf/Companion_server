"""Memory retrieval service.

Retrieves memories using a combination strategy:
- 5 semantic (vector similarity)
- 3 recent
- 2 high importance
"""

import asyncio
import logging

from app.db import db
from app.services.memory.vector_search import search_similar

logger = logging.getLogger(__name__)


def _memory_to_dict(m, similarity: float = 0.0) -> dict:
    """Convert a Prisma memory object to a dict."""
    return {
        "id": m.id,
        "content": m.content,
        "summary": m.summary,
        "level": m.level,
        "importance": m.importance,
        "type": m.type,
        "created_at": str(m.createdAt),
        "similarity": similarity,
    }


async def retrieve_memories(
    query: str,
    user_id: str,
    semantic_k: int = 5,
    recent_k: int = 3,
    important_k: int = 2,
) -> list[dict]:
    """Retrieve relevant memories using combined strategy.

    Runs all three queries concurrently, then deduplicates.
    """
    # Run all queries in parallel
    semantic_results, recent, important = await asyncio.gather(
        search_similar(query, user_id, top_k=semantic_k * 2),
        db.memory.find_many(
            where={"userId": user_id, "isArchived": False},
            order={"createdAt": "desc"},
            take=recent_k * 2,
        ),
        db.memory.find_many(
            where={"userId": user_id, "isArchived": False},
            order={"importance": "desc"},
            take=important_k * 2,
        ),
    )

    seen_ids: set[str] = set()
    results: list[dict] = []

    # 1. Semantic search
    for r in semantic_results[:semantic_k]:
        mid = r["id"]
        if mid not in seen_ids:
            seen_ids.add(mid)
            results.append(r)

    # 2. Recent memories
    for m in recent[:recent_k]:
        if m.id not in seen_ids:
            seen_ids.add(m.id)
            results.append(_memory_to_dict(m))

    # 3. High importance memories
    for m in important[:important_k]:
        if m.id not in seen_ids:
            seen_ids.add(m.id)
            results.append(_memory_to_dict(m))

    return results


def format_memories_for_prompt(memories: list[dict]) -> list[str]:
    """Format memory dicts into strings suitable for prompt injection."""
    return [
        m.get("summary") or m.get("content", "")
        for m in memories
        if m.get("summary") or m.get("content")
    ]
