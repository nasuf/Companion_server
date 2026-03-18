"""Vector search service using pgvector.

Performs cosine similarity search on memory_embeddings table.
"""

import logging
from datetime import datetime

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
    """Search using a pre-computed embedding vector.

    Searches across both memories_user and memories_ai via UNION.
    """
    vec_str = _format_vector(embedding)
    return await db.query_raw(
        """
        SELECT * FROM (
            (SELECT
                m.id, m.content, m.summary, m.level, m.importance, m.type,
                m.created_at, 'user' AS source,
                1 - (me.embedding <=> $1::extensions.vector) AS similarity
            FROM memory_embeddings me
            JOIN memories_user m ON m.id = me.memory_id
            WHERE m.user_id = $2 AND m.is_archived = false
            ORDER BY me.embedding <=> $1::extensions.vector
            LIMIT $3)
            UNION ALL
            (SELECT
                m.id, m.content, m.summary, m.level, m.importance, m.type,
                m.created_at, 'ai' AS source,
                1 - (me.embedding <=> $1::extensions.vector) AS similarity
            FROM memory_embeddings me
            JOIN memories_ai m ON m.id = me.memory_id
            WHERE m.user_id = $2 AND m.is_archived = false
            ORDER BY me.embedding <=> $1::extensions.vector
            LIMIT $3)
        ) combined
        ORDER BY similarity DESC
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


async def search_by_time_range(
    user_id: str,
    start_time: datetime,
    end_time: datetime,
    source: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """按时间范围检索记忆（基于 occur_time 字段）。

    PRD §9.3.4: 用户提及过去时间时，召回对应时间段的记忆。
    """
    if source == "user":
        return await db.query_raw(
            """
            SELECT id, content, summary, level, importance, type,
                   occur_time, created_at, 'user' AS source
            FROM memories_user
            WHERE user_id = $1 AND is_archived = false
              AND occur_time >= $2 AND occur_time < $3
            ORDER BY importance DESC
            LIMIT $4
            """,
            user_id, start_time, end_time, limit,
        )
    elif source == "ai":
        return await db.query_raw(
            """
            SELECT id, content, summary, level, importance, type,
                   occur_time, created_at, 'ai' AS source
            FROM memories_ai
            WHERE user_id = $1 AND is_archived = false
              AND occur_time >= $2 AND occur_time < $3
            ORDER BY importance DESC
            LIMIT $4
            """,
            user_id, start_time, end_time, limit,
        )
    # 查两表
    return await db.query_raw(
        """
        SELECT * FROM (
            (SELECT id, content, summary, level, importance, type,
                    occur_time, created_at, 'user' AS source
             FROM memories_user
             WHERE user_id = $1 AND is_archived = false
               AND occur_time >= $2 AND occur_time < $3
             ORDER BY importance DESC
             LIMIT $4)
            UNION ALL
            (SELECT id, content, summary, level, importance, type,
                    occur_time, created_at, 'ai' AS source
             FROM memories_ai
             WHERE user_id = $1 AND is_archived = false
               AND occur_time >= $2 AND occur_time < $3
             ORDER BY importance DESC
             LIMIT $4)
        ) combined
        ORDER BY importance DESC
        LIMIT $4
        """,
        user_id, start_time, end_time, limit,
    )
