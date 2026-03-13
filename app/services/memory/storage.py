"""Memory storage service.

Stores memories to PostgreSQL with deduplication (cosine > 0.9).
Classifies into L1/L2/L3 levels.
"""

import logging

from app.db import db
from app.services.memory.embedding import generate_embedding, store_embedding
from app.services.memory.vector_search import search_by_embedding

logger = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.9


async def is_duplicate(
    user_id: str,
    content: str,
    embedding: list[float],
) -> bool:
    """Check if a similar memory already exists (cosine > 0.9)."""
    results = await search_by_embedding(embedding, user_id, top_k=5)
    for r in results:
        sim = r.get("similarity", 0)
        if isinstance(sim, str):
            sim = float(sim)
        if sim > DEDUP_THRESHOLD:
            logger.info(f"Duplicate memory detected (similarity={sim:.3f}): {content[:50]}")
            return True
    return False


async def store_memory(
    user_id: str,
    content: str,
    summary: str | None = None,
    level: int = 3,
    importance: float = 0.5,
    memory_type: str | None = None,
) -> str | None:
    """Store a memory with deduplication.

    Returns memory_id if stored, None if duplicate.
    """
    # Generate embedding
    embedding = await generate_embedding(content)

    # Deduplication check
    if await is_duplicate(user_id, content, embedding):
        return None

    # Store in PostgreSQL
    memory = await db.memory.create(
        data={
            "user": {"connect": {"id": user_id}},
            "content": content,
            "summary": summary or content[:200],
            "level": level,
            "importance": importance,
            "type": memory_type,
        }
    )

    # Store embedding
    await store_embedding(memory.id, embedding)

    logger.info(f"Stored memory L{level} (importance={importance:.2f}): {content[:50]}")
    return memory.id
