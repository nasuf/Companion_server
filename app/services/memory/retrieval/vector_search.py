"""Vector search service using pgvector.

Performs cosine similarity search on memory_embeddings table.
"""

import asyncio
import logging
from datetime import datetime

from app.db import db
from app.services.memory.storage.embedding import generate_embedding
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)


def format_vector(embedding: list[float]) -> str:
    """Format embedding list as a pgvector-compatible string."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


def _embedding_search_arm(table: str, source: str) -> str:
    """One UNION arm of the embedding search, parameterized only by table/source.

    Keeping a single template guarantees the user/ai arms stay byte-identical
    except for the table name and the `source` literal — avoids the two arms
    silently drifting apart on future column changes.
    """
    return f"""
        (SELECT
            m.id, m.content, m.level, m.importance, m.current_score,
            m.provenance, m.mention_count,
            m.type, m.main_category, m.sub_category,
            m.created_at, m.updated_at,
            COALESCE(m.updated_at, m.created_at) AS last_accessed_at,
            '{source}' AS source,
            1 - (me.embedding OPERATOR(extensions.<=>) $1::extensions.vector) AS similarity
        FROM memory_embeddings me
        JOIN {table} m ON m.id = me.memory_id
        WHERE m.user_id = $2
          AND m.workspace_id = $3
          AND m.is_archived = false
          AND ($4::text[] IS NULL OR m.main_category = ANY($4::text[]))
          AND ($5::text[] IS NULL OR m.sub_category = ANY($5::text[]))
          AND ($6::int[] IS NULL OR m.level = ANY($6::int[]))
        ORDER BY me.embedding OPERATOR(extensions.<=>) $1::extensions.vector
        LIMIT $7)
    """


async def search_by_embedding(
    embedding: list[float],
    user_id: str,
    top_k: int = 50,
    workspace_id: str | None = None,
    main_categories: list[str] | None = None,
    sub_categories: list[str] | None = None,
    levels: list[int] | None = None,
    sources: list[str] | None = None,
) -> list[dict]:
    """Search using a pre-computed embedding vector.

    By default searches both memories_user and memories_ai (UNION). Pass
    `sources=["user"]` / `["ai"]` to scope to a single owner — critical for
    dedup, where matching a user memory against an AI self-memory (or vice
    versa) would drop or mis-route data across the owner boundary.
    """
    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    vec_str = format_vector(embedding)

    wanted = {s for s in (sources or ["user", "ai"]) if s in ("user", "ai")}
    if not wanted:
        wanted = {"user", "ai"}
    arms: list[str] = []
    if "user" in wanted:
        arms.append(_embedding_search_arm("memories_user", "user"))
    if "ai" in wanted:
        arms.append(_embedding_search_arm("memories_ai", "ai"))

    query = (
        "SELECT * FROM (\n"
        + "\n            UNION ALL\n".join(arms)
        + "\n        ) combined\n        ORDER BY similarity DESC\n        LIMIT $7"
    )
    return await db.query_raw(
        query,
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


async def search_similar_tiers(
    query: str,
    user_id: str,
    tiers: list[tuple[list[int], int]],
    workspace_id: str | None = None,
) -> list[dict]:
    """一次嵌入, 分别搜多个层级组, 合并成一个扁平列表返回。

    存在的理由是 `search_similar` 每次调用都自己嵌一遍: 检索热层 (L1/L2) 和冷层
    (L3) 用的是同一个 query, 分别调两次就会对同一段文本嵌两次。两次是并行发起的,
    Redis 缓存挡不住同时 miss —— 结果是每轮对话给 Ollama 多压一次嵌入调用, 而
    嵌入是 CPU 密集的已知瓶颈。

    tiers 形如 [([1, 2], 50), ([3], 3)]: 前者热层取 50 条, 后者冷层取 3 条。
    每层的数量上限由各自的 top_k 保证。

    返回扁平列表而不是按层分组: 每行本来就带 level 字段, 调用方要区分冷热直接看
    它即可。分组返回会让所有调用方 (和它们的测试) 多套一层嵌套, 没有换来信息。
    """
    embedding = await generate_embedding(query)
    per_tier = await asyncio.gather(*[
        search_by_embedding(
            embedding, user_id, top_k, workspace_id=workspace_id, levels=levels,
        )
        for levels, top_k in tiers
    ])
    return [row for rows in per_tier for row in rows]


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
            SELECT id, content, level, importance, mention_count,
                   type, main_category, sub_category,
                   occur_time, created_at, updated_at,
                   COALESCE(updated_at, created_at) AS last_accessed_at,
                   'user' AS source
            FROM memories_user
            WHERE user_id = $1 AND workspace_id = $2 AND is_archived = false
              AND occur_time >= $3::timestamp AND occur_time < $4::timestamp
            ORDER BY importance DESC
            LIMIT $5
            """,
            user_id, workspace_id, start_time, end_time, limit,
        )
    elif source == "ai":
        return await db.query_raw(
            """
            SELECT id, content, level, importance, mention_count,
                   type, main_category, sub_category,
                   occur_time, created_at, updated_at,
                   COALESCE(updated_at, created_at) AS last_accessed_at,
                   'ai' AS source
            FROM memories_ai
            WHERE user_id = $1 AND workspace_id = $2 AND is_archived = false
              AND occur_time >= $3::timestamp AND occur_time < $4::timestamp
            ORDER BY importance DESC
            LIMIT $5
            """,
            user_id, workspace_id, start_time, end_time, limit,
        )
    # 查两表
    return await db.query_raw(
        """
        SELECT * FROM (
            (SELECT id, content, level, importance, mention_count,
                    type, main_category, sub_category,
                    occur_time, created_at, updated_at,
                    COALESCE(updated_at, created_at) AS last_accessed_at,
                    'user' AS source
             FROM memories_user
             WHERE user_id = $1 AND workspace_id = $2 AND is_archived = false
               AND occur_time >= $3::timestamp AND occur_time < $4::timestamp
             ORDER BY importance DESC
             LIMIT $5)
            UNION ALL
            (SELECT id, content, level, importance, mention_count,
                    type, main_category, sub_category,
                    occur_time, created_at, updated_at,
                    COALESCE(updated_at, created_at) AS last_accessed_at,
                    'ai' AS source
             FROM memories_ai
             WHERE user_id = $1 AND workspace_id = $2 AND is_archived = false
               AND occur_time >= $3::timestamp AND occur_time < $4::timestamp
             ORDER BY importance DESC
             LIMIT $5)
        ) combined
        ORDER BY importance DESC
        LIMIT $5
        """,
        user_id, workspace_id, start_time, end_time, limit,
    )
