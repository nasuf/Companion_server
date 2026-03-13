"""Vector search service using pgvector.

Performs cosine similarity search on memory_embeddings table.
"""

import logging

from app.db import db
from app.services.memory.embedding import generate_embedding

logger = logging.getLogger(__name__)


def _format_vector(embedding: list[float]) -> str:
    """Format embedding list as a pgvector-compatible string."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def search_by_embedding(
    embedding: list[float],
    user_id: str,
    top_k: int = 50,
) -> list[dict]:
    """Search using a pre-computed embedding vector."""
    vec_str = _format_vector(embedding)
    return await db.query_raw(
        """
        SELECT
            m.id,
            m.content,
            m.summary,
            m.level,
            m.importance,
            m.type,
            m.created_at,
            1 - (me.embedding <=> $1::vector) AS similarity
        FROM memory_embeddings me
        JOIN memories m ON m.id = me.memory_id
        WHERE m.user_id = $2
          AND m.is_archived = false
        ORDER BY me.embedding <=> $1::vector
        LIMIT $3
        """,
        vec_str,
        user_id,
        top_k,
    )


async def search_similar(
    query: str,
    user_id: str,
    top_k: int = 50,
) -> list[dict]:
    """Search for similar memories by text query (generates embedding first)."""
    embedding = await generate_embedding(query)
    return await search_by_embedding(embedding, user_id, top_k)
