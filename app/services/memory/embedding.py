"""Embedding service for memory vectors.

Uses OllamaEmbeddings via LangChain to generate embeddings,
and raw SQL for pgvector storage/retrieval.
Includes Redis caching for embeddings.
"""

import logging

from app.db import db
from app.services.llm.models import get_embedding_model
from app.services.cache import cache_embedding, cache_set_embedding

logger = logging.getLogger(__name__)


async def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text (with cache)."""
    # Check cache
    cached = await cache_embedding(text)
    if cached:
        return cached

    model = get_embedding_model()
    embedding = await model.aembed_query(text)

    # Cache the result
    try:
        await cache_set_embedding(text, embedding)
    except Exception:
        pass

    return embedding


async def store_embedding(memory_id: str, embedding: list[float]) -> None:
    """Store an embedding in the memory_embeddings table."""
    from app.services.memory.vector_search import _format_vector
    vec_str = _format_vector(embedding)
    await db.execute_raw(
        """
        INSERT INTO memory_embeddings (memory_id, embedding)
        VALUES ($1, $2::extensions.vector)
        ON CONFLICT (memory_id) DO UPDATE SET embedding = $2::extensions.vector
        """,
        memory_id,
        vec_str,
    )


async def get_embedding(memory_id: str) -> list[float] | None:
    """Retrieve the embedding for a memory."""
    rows = await db.query_raw(
        """
        SELECT embedding::text FROM memory_embeddings WHERE memory_id = $1
        """,
        memory_id,
    )
    if not rows:
        return None
    # Parse the vector string [x,y,z,...] back to list
    vec_str = rows[0]["embedding"]
    if vec_str:
        return [float(v) for v in vec_str.strip("[]").split(",")]
    return None
