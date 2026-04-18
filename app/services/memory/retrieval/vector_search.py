"""Vector search service using pgvector.

Performs cosine similarity search on memory_embeddings table.
"""

import logging
from datetime import datetime

from app.db import db
from app.services.memory.storage.embedding import generate_embedding
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)


def format_vector(embedding: list[float]) -> str:
    """Format embedding list as a pgvector-compatible string."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def search_by_embedding(
    embedding: list[float],
    user_id: str,
    top_k: int = 50,
    workspace_id: str | None = None,
    main_categories: list[str] | None = None,
    sub_categories: list[str] | None = None,
    levels: list[int] | None = None,
) -> list[dict]:
    """Search using a pre-computed embedding vector.

    Searches across both memories_user and memories_ai via UNION.
    """
    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    vec_str = format_vector(embedding)
    return await db.query_raw(
        """
        SELECT * FROM (
            (SELECT
                m.id, m.content, m.summary, m.level, m.importance, m.type, m.main_category, m.sub_category,
                m.created_at, 'user' AS source,
                1 - (me.embedding <=> $1::extensions.vector) AS similarity
            FROM memory_embeddings me
            JOIN memories_user m ON m.id = me.memory_id
            WHERE m.user_id = $2
              AND m.workspace_id = $3
              AND m.is_archived = false
              AND ($4::text[] IS NULL OR m.main_category = ANY($4::text[]))
              AND ($5::text[] IS NULL OR m.sub_category = ANY($5::text[]))
              AND ($6::int[] IS NULL OR m.level = ANY($6::int[]))
            ORDER BY me.embedding <=> $1::extensions.vector
            LIMIT $7)
            UNION ALL
            (SELECT
                m.id, m.content, m.summary, m.level, m.importance, m.type, m.main_category, m.sub_category,
                m.created_at, 'ai' AS source,
                1 - (me.embedding <=> $1::extensions.vector) AS similarity
            FROM memory_embeddings me
            JOIN memories_ai m ON m.id = me.memory_id
            WHERE m.user_id = $2
              AND m.workspace_id = $3
              AND m.is_archived = false
              AND ($4::text[] IS NULL OR m.main_category = ANY($4::text[]))
              AND ($5::text[] IS NULL OR m.sub_category = ANY($5::text[]))
              AND ($6::int[] IS NULL OR m.level = ANY($6::int[]))
            ORDER BY me.embedding <=> $1::extensions.vector
            LIMIT $7)
        ) combined
        ORDER BY similarity DESC
        LIMIT $7
        """,
        vec_str,
        user_id,
        workspace_id,
        main_categories or None,
        sub_categories or None,
        levels or None,
        top_k,
    )


async def search_similar(
    query: str,
    user_id: str,
    top_k: int = 50,
    workspace_id: str | None = None,
    main_categories: list[str] | None = None,
    sub_categories: list[str] | None = None,
    levels: list[int] | None = None,
) -> list[dict]:
    """Search for similar memories by text query (generates embedding first)."""
    embedding = await generate_embedding(query)
    return await search_by_embedding(
        embedding,
        user_id,
        top_k,
        workspace_id=workspace_id,
        main_categories=main_categories,
        sub_categories=sub_categories,
        levels=levels,
    )


async def search_by_time_range(
    user_id: str,
    start_time: datetime,
    end_time: datetime,
    source: str | None = None,
    limit: int = 10,
    workspace_id: str | None = None,
) -> list[dict]:
    """按时间范围检索记忆（基于 occur_time 字段）。

    PRD §9.3.4: 用户提及过去时间时，召回对应时间段的记忆。
    """
    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    if source == "user":
        return await db.query_raw(
            """
            SELECT id, content, summary, level, importance, type, main_category, sub_category,
                   occur_time, created_at, 'user' AS source
            FROM memories_user
            WHERE user_id = $1 AND workspace_id = $2 AND is_archived = false
              AND occur_time >= $3 AND occur_time < $4
            ORDER BY importance DESC
            LIMIT $5
            """,
            user_id, workspace_id, start_time, end_time, limit,
        )
    elif source == "ai":
        return await db.query_raw(
            """
            SELECT id, content, summary, level, importance, type, main_category, sub_category,
                   occur_time, created_at, 'ai' AS source
            FROM memories_ai
            WHERE user_id = $1 AND workspace_id = $2 AND is_archived = false
              AND occur_time >= $3 AND occur_time < $4
            ORDER BY importance DESC
            LIMIT $5
            """,
            user_id, workspace_id, start_time, end_time, limit,
        )
    # 查两表
    return await db.query_raw(
        """
        SELECT * FROM (
            (SELECT id, content, summary, level, importance, type, main_category, sub_category,
                    occur_time, created_at, 'user' AS source
             FROM memories_user
             WHERE user_id = $1 AND workspace_id = $2 AND is_archived = false
               AND occur_time >= $3 AND occur_time < $4
             ORDER BY importance DESC
             LIMIT $5)
            UNION ALL
            (SELECT id, content, summary, level, importance, type, main_category, sub_category,
                    occur_time, created_at, 'ai' AS source
             FROM memories_ai
             WHERE user_id = $1 AND workspace_id = $2 AND is_archived = false
               AND occur_time >= $3 AND occur_time < $4
             ORDER BY importance DESC
             LIMIT $5)
        ) combined
        ORDER BY importance DESC
        LIMIT $5
        """,
        user_id, workspace_id, start_time, end_time, limit,
    )
