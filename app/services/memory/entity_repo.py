"""Entity knowledge layer — Postgres-native replacement for the Neo4j projection.

Stores canonical entities (people / places / orgs / topics / preferences) that
appear across a user's memories, plus a many-to-many `memory_mentions` edge
pointing back to memories_user / memories_ai.

Design choices:
- Unique index on (user_id, workspace_id, entity_type, canonical_name) so
  upsert is idempotent and races collapse to one row.
- `aliases` TEXT[] allows "妈妈"/"我妈"/"妈" to resolve to the same entity.
- `mention_count` + `last_mentioned_at` are pre-aggregated on write to avoid
  GROUP BY on the hot retrieval path.
- Callers pass source='user'|'ai' so mentions keep a back-pointer to which
  memories table owns the row (we can't FK into a union).
"""

from __future__ import annotations

import logging
import uuid

from app.db import db

logger = logging.getLogger(__name__)

ENTITY_KINDS = {"person", "place", "org", "pet", "topic", "preference", "other"}


async def upsert_entity(
    *,
    user_id: str,
    workspace_id: str | None,
    canonical_name: str,
    entity_type: str,
    role: str | None = None,
    aliases: list[str] | None = None,
    metadata: dict | None = None,
) -> str:
    """Idempotent upsert: returns the entity id.

    mention_count / last_mentioned_at are NOT bumped here — that's the job
    of `link_memory_to_entity`, which ensures the bump only happens when
    an actual new memory→entity link is created.
    """
    if entity_type not in ENTITY_KINDS:
        entity_type = "other"
    name = canonical_name.strip()
    if not name:
        raise ValueError("canonical_name must be non-empty")

    new_id = str(uuid.uuid4())
    # Postgres' ON CONFLICT DO UPDATE with a RETURNING id gives us either
    # the existing id (DO UPDATE always fires) or the freshly inserted one.
    # Merging aliases: array_cat + distinct to accumulate without duplicates.
    rows = await db.query_raw(
        """
        INSERT INTO memory_entities
            (id, user_id, workspace_id, canonical_name, entity_type, role,
             aliases, metadata, created_at, updated_at)
        VALUES
            ($1, $2, $3, $4, $5, $6, $7::text[], $8::jsonb, NOW(), NOW())
        ON CONFLICT (user_id, workspace_id, entity_type, canonical_name)
        DO UPDATE SET
            aliases = (
                SELECT ARRAY(
                    SELECT DISTINCT unnest(memory_entities.aliases || EXCLUDED.aliases)
                )
            ),
            role = COALESCE(memory_entities.role, EXCLUDED.role),
            metadata = COALESCE(memory_entities.metadata, '{}'::jsonb) || COALESCE(EXCLUDED.metadata, '{}'::jsonb),
            updated_at = NOW()
        RETURNING id
        """,
        new_id,
        user_id,
        workspace_id,
        name,
        entity_type,
        role,
        aliases or [],
        _jsonb(metadata),
    )
    return rows[0]["id"]


async def link_memory_to_entity(
    *,
    memory_id: str,
    memory_source: str,
    entity_id: str,
    user_id: str,
    workspace_id: str | None,
) -> bool:
    """Create a memory↔entity edge and bump stats on first creation.

    Returns True if a new edge was inserted; False if the edge already
    existed (idempotent on re-runs / pipeline retries).
    """
    rows = await db.query_raw(
        """
        INSERT INTO memory_mentions
            (memory_id, memory_source, entity_id, user_id, workspace_id, created_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        ON CONFLICT (memory_id, entity_id) DO NOTHING
        RETURNING memory_id
        """,
        memory_id,
        memory_source,
        entity_id,
        user_id,
        workspace_id,
    )
    if not rows:
        return False  # edge existed; stats already counted

    await db.execute_raw(
        """
        UPDATE memory_entities
        SET mention_count = mention_count + 1,
            last_mentioned_at = NOW(),
            updated_at = NOW()
        WHERE id = $1
        """,
        entity_id,
    )
    return True


async def record_entities_for_memory(
    *,
    memory_id: str,
    memory_source: str,
    user_id: str,
    workspace_id: str | None,
    entities: list[dict],
) -> int:
    """High-level helper used by the memory pipeline.

    `entities` is the shape produced by `extract_memories`:
        [{"name": "妈妈", "type": "person", "role": "mother"?}, ...]
    Rows missing a name are silently dropped.
    Returns the count of NEW edges created (for logging/metrics).
    """
    new_edges = 0
    for ent in entities or []:
        name = (ent.get("name") or "").strip()
        if not name:
            continue
        try:
            entity_id = await upsert_entity(
                user_id=user_id,
                workspace_id=workspace_id,
                canonical_name=name,
                entity_type=ent.get("type", "other"),
                role=ent.get("role"),
                aliases=ent.get("aliases") or None,
            )
            created = await link_memory_to_entity(
                memory_id=memory_id,
                memory_source=memory_source,
                entity_id=entity_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )
            if created:
                new_edges += 1
        except Exception as e:
            logger.warning(f"entity upsert failed for '{name}': {e}")
    return new_edges


async def record_topics_for_memory(
    *,
    memory_id: str,
    memory_source: str,
    user_id: str,
    workspace_id: str | None,
    topics: list[str],
) -> int:
    """Topics are modeled as entities of type='topic'."""
    as_entities = [{"name": t, "type": "topic"} for t in (topics or []) if t]
    return await record_entities_for_memory(
        memory_id=memory_id,
        memory_source=memory_source,
        user_id=user_id,
        workspace_id=workspace_id,
        entities=as_entities,
    )


async def record_preferences_for_memory(
    *,
    memory_id: str,
    memory_source: str,
    user_id: str,
    workspace_id: str | None,
    preferences: list[dict],
) -> int:
    """Preferences as entities of type='preference'. canonical_name uses the
    pref value; role carries the category (food/color/...)."""
    as_entities: list[dict] = []
    for pref in preferences or []:
        cat = (pref.get("category") or "").strip()
        val = (pref.get("value") or "").strip()
        if not val:
            continue
        as_entities.append({
            "name": val,
            "type": "preference",
            "role": cat or None,
        })
    return await record_entities_for_memory(
        memory_id=memory_id,
        memory_source=memory_source,
        user_id=user_id,
        workspace_id=workspace_id,
        entities=as_entities,
    )


# ── read side ──

async def top_entities(
    *,
    user_id: str,
    workspace_id: str | None,
    entity_type: str | None = None,
    main_categories: list[str] | None = None,
    sub_categories: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Top-N entities by mention count, optionally scoped to a category.

    When main_categories / sub_categories are given, we join through
    memory_mentions → memories and group by entity to only count mentions
    in matching memories. Otherwise we read the pre-aggregated counter.
    """
    if main_categories or sub_categories:
        rows = await db.query_raw(
            """
            SELECT e.canonical_name AS name, e.entity_type AS type, e.role AS role,
                   COUNT(*) AS mention_count
            FROM memory_entities e
            JOIN memory_mentions mm ON mm.entity_id = e.id
            LEFT JOIN memories_user mu ON mu.id = mm.memory_id AND mm.memory_source = 'user'
            LEFT JOIN memories_ai   ma ON ma.id = mm.memory_id AND mm.memory_source = 'ai'
            WHERE e.user_id = $1
              AND ($2::text  IS NULL OR e.workspace_id = $2)
              AND ($3::text  IS NULL OR e.entity_type = $3)
              AND e.is_archived = false
              AND ($4::text[] IS NULL
                   OR COALESCE(mu.main_category, ma.main_category) = ANY($4))
              AND ($5::text[] IS NULL
                   OR COALESCE(mu.sub_category, ma.sub_category) = ANY($5))
              AND COALESCE(mu.is_archived, ma.is_archived, false) = false
            GROUP BY e.id, e.canonical_name, e.entity_type, e.role
            ORDER BY mention_count DESC
            LIMIT $6
            """,
            user_id,
            workspace_id,
            entity_type,
            main_categories,
            sub_categories,
            limit,
        )
    else:
        rows = await db.query_raw(
            """
            SELECT canonical_name AS name, entity_type AS type, role, mention_count
            FROM memory_entities
            WHERE user_id = $1
              AND ($2::text IS NULL OR workspace_id = $2)
              AND ($3::text IS NULL OR entity_type = $3)
              AND is_archived = false
              AND mention_count > 0
            ORDER BY mention_count DESC
            LIMIT $4
            """,
            user_id,
            workspace_id,
            entity_type,
            limit,
        )
    return [dict(r) for r in rows]


async def get_relationship_context(
    *,
    user_id: str,
    workspace_id: str | None,
    main_categories: list[str] | None = None,
    sub_categories: list[str] | None = None,
) -> dict:
    """Shape-compatible with the old Neo4j `get_relationship_context`.

    Returns {"topics": [...], "entities": [...], "categories": [...]} where
    entities are "name (type)" strings and categories come from a direct
    SQL aggregation over memories_* (no graph projection needed).
    """
    top_topics = await top_entities(
        user_id=user_id, workspace_id=workspace_id,
        entity_type="topic",
        main_categories=main_categories, sub_categories=sub_categories,
        limit=10,
    )
    top_people_and_places = await top_entities(
        user_id=user_id, workspace_id=workspace_id,
        entity_type=None,  # union of person/place/org/pet
        main_categories=main_categories, sub_categories=sub_categories,
        limit=20,
    )
    # Filter to person/place/org/pet only
    named_entities = [
        e for e in top_people_and_places
        if e["type"] in {"person", "place", "org", "pet"}
    ][:10]

    category_rows = await db.query_raw(
        """
        SELECT main_category, sub_category, COUNT(*) AS count FROM (
            SELECT main_category, sub_category FROM memories_user
            WHERE user_id = $1
              AND ($2::text IS NULL OR workspace_id = $2)
              AND is_archived = false
              AND main_category IS NOT NULL AND sub_category IS NOT NULL
              AND ($3::text[] IS NULL OR main_category = ANY($3))
              AND ($4::text[] IS NULL OR sub_category = ANY($4))
            UNION ALL
            SELECT main_category, sub_category FROM memories_ai
            WHERE user_id = $1
              AND ($2::text IS NULL OR workspace_id = $2)
              AND is_archived = false
              AND main_category IS NOT NULL AND sub_category IS NOT NULL
              AND ($3::text[] IS NULL OR main_category = ANY($3))
              AND ($4::text[] IS NULL OR sub_category = ANY($4))
        ) x
        GROUP BY main_category, sub_category
        ORDER BY count DESC
        LIMIT 12
        """,
        user_id,
        workspace_id,
        main_categories,
        sub_categories,
    )

    return {
        "topics": [t["name"] for t in top_topics],
        "entities": [
            f"{e['name']} ({e['type']})" for e in named_entities
        ],
        "categories": [
            f"{r['main_category']} / {r['sub_category']}" for r in category_rows
        ],
    }


async def get_user_preferences(
    *,
    user_id: str,
    workspace_id: str | None,
    limit: int = 50,
) -> list[dict]:
    """Return all preferences the user has expressed.

    Shape matches the old Neo4j `get_user_preferences`:
        [{"category": "food", "value": "ramen", "count": 3}, ...]
    """
    rows = await db.query_raw(
        """
        SELECT role AS category, canonical_name AS value, mention_count AS count
        FROM memory_entities
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
          AND entity_type = 'preference'
          AND is_archived = false
        ORDER BY mention_count DESC
        LIMIT $3
        """,
        user_id,
        workspace_id,
        limit,
    )
    return [dict(r) for r in rows]


async def get_related_memories(
    *,
    user_id: str,
    workspace_id: str | None,
    entity_name: str,
    limit: int = 10,
) -> list[dict]:
    """All memories mentioning an entity by its canonical name or any alias."""
    rows = await db.query_raw(
        """
        WITH matched_entities AS (
            SELECT id FROM memory_entities
            WHERE user_id = $1
              AND ($2::text IS NULL OR workspace_id = $2)
              AND ($3 = canonical_name OR $3 = ANY(aliases))
        )
        SELECT * FROM (
            SELECT mm.memory_id AS id, mu.summary, mu.content, mu.importance,
                   mu.created_at, 'user' AS source
            FROM memory_mentions mm
            JOIN matched_entities me ON me.id = mm.entity_id
            JOIN memories_user mu ON mu.id = mm.memory_id AND mm.memory_source = 'user'
            WHERE mu.is_archived = false

            UNION ALL

            SELECT mm.memory_id AS id, ma.summary, ma.content, ma.importance,
                   ma.created_at, 'ai' AS source
            FROM memory_mentions mm
            JOIN matched_entities me ON me.id = mm.entity_id
            JOIN memories_ai ma ON ma.id = mm.memory_id AND mm.memory_source = 'ai'
            WHERE ma.is_archived = false
        ) r
        ORDER BY importance DESC, created_at DESC
        LIMIT $4
        """,
        user_id,
        workspace_id,
        entity_name.strip(),
        limit,
    )
    return [dict(r) for r in rows]


# ── maintenance ──

# Archive an entity if we haven't seen it in this many days AND it was
# never mentioned more than a handful of times. Stored here (not in
# memory/config.py) because it's entity-specific and will likely be tuned
# separately from memory decay.
ARCHIVE_STALE_DAYS = 180          # 半年未提及
ARCHIVE_MAX_MENTION = 3           # 且全历史提及 ≤ 3 次

# Duplicate merge (entity consolidation) thresholds
MERGE_COSINE_THRESHOLD = 0.92     # embedding 相似度下限
MERGE_CANDIDATE_LIMIT = 500       # 每次扫最近活跃的前 N 个 entity，避免 O(N²)


async def archive_stale_entities(
    *,
    user_id: str | None = None,
    workspace_id: str | None = None,
    stale_days: int = ARCHIVE_STALE_DAYS,
    max_mention: int = ARCHIVE_MAX_MENTION,
) -> int:
    """把长期没被提到、总频次也很低的实体标记为归档。

    归档不是物理删除：mentions 仍保留，memory 检索主路径不受影响。只是
    不再出现在 top_entities / relationship_context 里，避免长期对话后
    老弱实体污染 prompt 上下文。

    若 user_id/workspace_id 都为 None，则全库扫（供 scheduler 使用）。
    """
    result = await db.execute_raw(
        """
        UPDATE memory_entities
        SET is_archived = true, updated_at = NOW()
        WHERE is_archived = false
          AND ($1::text IS NULL OR user_id = $1)
          AND ($2::text IS NULL OR workspace_id = $2)
          AND mention_count <= $3
          AND (
              last_mentioned_at IS NULL
              OR last_mentioned_at < NOW() - ($4::int || ' days')::interval
          )
        """,
        user_id,
        workspace_id,
        max_mention,
        stale_days,
    )
    return result or 0


async def _embed_entity(name: str, aliases: list[str]) -> list[float]:
    """Entities are identified primarily by canonical name; we mix in
    aliases so '妈妈' and '我妈' embed close together naturally."""
    from app.services.llm.models import get_embedding_model
    model = get_embedding_model()
    text = name if not aliases else f"{name}（也叫：{','.join(aliases[:5])}）"
    return await model.aembed_query(text)


async def _load_active_entities(
    user_id: str, workspace_id: str | None, entity_type: str, limit: int,
) -> list[dict]:
    rows = await db.query_raw(
        """
        SELECT id, canonical_name, aliases, mention_count, last_mentioned_at,
               role, metadata
        FROM memory_entities
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
          AND entity_type = $3
          AND is_archived = false
        ORDER BY last_mentioned_at DESC NULLS LAST, mention_count DESC
        LIMIT $4
        """,
        user_id, workspace_id, entity_type, limit,
    )
    return [dict(r) for r in rows]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


async def _merge_entity_pair(winner_id: str, loser_id: str) -> None:
    """Merge loser into winner: move mentions, accumulate aliases, then
    delete loser. Done in explicit steps (pgvector isn't involved here
    so the operations are easy to sequence)."""
    # 1. Union aliases + add loser's canonical_name to winner.aliases
    await db.execute_raw(
        """
        UPDATE memory_entities w
        SET aliases = (
            SELECT ARRAY(
                SELECT DISTINCT x FROM unnest(
                    w.aliases || l.aliases || ARRAY[l.canonical_name]
                ) AS x WHERE x IS NOT NULL AND x <> w.canonical_name
            )
        ),
        mention_count = w.mention_count + l.mention_count,
        last_mentioned_at = GREATEST(
            w.last_mentioned_at, l.last_mentioned_at
        ),
        updated_at = NOW()
        FROM memory_entities l
        WHERE w.id = $1 AND l.id = $2
        """,
        winner_id, loser_id,
    )
    # 2. Re-point mentions. ON CONFLICT: the same memory already links to
    # both entities → drop the duplicate loser edge.
    await db.execute_raw(
        """
        UPDATE memory_mentions
        SET entity_id = $1
        WHERE entity_id = $2
          AND NOT EXISTS (
              SELECT 1 FROM memory_mentions mm2
              WHERE mm2.memory_id = memory_mentions.memory_id
                AND mm2.entity_id = $1
          )
        """,
        winner_id, loser_id,
    )
    # 3. Delete the loser. Cascade clears any still-duplicate edges.
    await db.execute_raw(
        "DELETE FROM memory_entities WHERE id = $1", loser_id,
    )


async def merge_duplicate_entities(
    *,
    user_id: str,
    workspace_id: str | None,
    entity_type: str = "person",
    similarity_threshold: float = MERGE_COSINE_THRESHOLD,
    candidate_limit: int = MERGE_CANDIDATE_LIMIT,
) -> int:
    """LLM-agnostic consolidation: cluster entities of the same type whose
    names embed near each other, merge the smaller into the larger.

    Runs per (user, workspace, entity_type). Embedding cost = O(N) per run,
    clustering is O(N²) but bounded by candidate_limit (default 500) so in
    practice it's a few hundred ms.
    Returns number of entities merged away.
    """
    entities = await _load_active_entities(
        user_id, workspace_id, entity_type, candidate_limit,
    )
    if len(entities) < 2:
        return 0

    # Embed each entity's display name once
    embeddings: list[list[float]] = []
    for ent in entities:
        try:
            vec = await _embed_entity(
                ent["canonical_name"], ent.get("aliases") or [],
            )
        except Exception as e:
            logger.warning(f"entity embed failed for {ent['id']}: {e}")
            vec = []
        embeddings.append(vec)

    # Greedy single-link clustering: iterate pairs, merge if cosine high.
    # Winner = higher mention_count (tie broken by earlier created_at via
    # list order from the DB query).
    merged_ids: set[str] = set()
    merges = 0
    for i in range(len(entities)):
        if entities[i]["id"] in merged_ids or not embeddings[i]:
            continue
        for j in range(i + 1, len(entities)):
            if entities[j]["id"] in merged_ids or not embeddings[j]:
                continue
            sim = _cosine(embeddings[i], embeddings[j])
            if sim < similarity_threshold:
                continue
            # pick winner by higher mention_count
            if entities[i]["mention_count"] >= entities[j]["mention_count"]:
                winner, loser = entities[i], entities[j]
            else:
                winner, loser = entities[j], entities[i]
            try:
                await _merge_entity_pair(winner["id"], loser["id"])
                merged_ids.add(loser["id"])
                merges += 1
                logger.info(
                    f"Merged entity '{loser['canonical_name']}' → "
                    f"'{winner['canonical_name']}' (cos={sim:.3f})"
                )
            except Exception as e:
                logger.warning(
                    f"merge failed {loser['id']}->{winner['id']}: {e}"
                )
            # If entity i itself was the loser, stop scanning its pairs
            if winner["id"] != entities[i]["id"]:
                break
    return merges


async def consolidate_entities_globally() -> dict[str, int]:
    """Scheduled batch: for every active (user, workspace) with entities,
    archive stale ones and merge duplicates (person/place/topic only —
    preferences are value-keyed so merging them silently changes meaning).

    Returns per-stage counts for logging/metrics.
    """
    archived = await archive_stale_entities()
    # Iterate distinct (user_id, workspace_id) scopes that still have
    # active entities. Cheap single query because of the index.
    scope_rows = await db.query_raw(
        """
        SELECT DISTINCT user_id, workspace_id
        FROM memory_entities
        WHERE is_archived = false
        """
    )
    total_merges = 0
    for row in scope_rows:
        for etype in ("person", "place", "topic", "org"):
            try:
                total_merges += await merge_duplicate_entities(
                    user_id=row["user_id"],
                    workspace_id=row.get("workspace_id"),
                    entity_type=etype,
                )
            except Exception as e:
                logger.warning(
                    f"merge pass failed for {row['user_id']}/{etype}: {e}"
                )
    return {"archived": archived, "merged": total_merges}


# ── misc ──

def _jsonb(value: dict | None) -> str | None:
    """Prisma raw params expect JSON as string when casting to jsonb."""
    if value is None:
        return None
    import json
    return json.dumps(value, ensure_ascii=False)
