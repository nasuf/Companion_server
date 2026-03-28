"""Memory storage service.

Stores memories to PostgreSQL with deduplication (cosine > 0.9).
Classifies into L1/L2/L3 levels.
"""

import logging
from datetime import datetime

from app.db import db
from app.services.memory import memory_repo
from app.services.memory.embedding import generate_embedding, store_embedding
from app.services.memory.taxonomy import resolve_taxonomy
from app.services.memory.vector_search import search_by_embedding
from app.services.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

# Map legacy/mixed type values to the standard English enum
_TYPE_NORMALIZE_MAP: dict[str, str] = {
    # Standard values (passthrough)
    "identity": "identity",
    "emotion": "emotion",
    "preference": "preference",
    "life": "life",
    "thought": "thought",
    "consolidated": "consolidated",
    # Old English values
    "event": "life",
    "relationship": "life",
    "fact": "identity",
    # Old Chinese values (from self_memory)
    "感受": "emotion",
    "体验": "life",
    "思考": "thought",
    "生活": "life",
    "关系": "life",
    # System types
    "compressed": "consolidated",
}


def normalize_memory_type(memory_type: str | None) -> str | None:
    """Normalize a memory type value to the standard English enum.

    Standard types: identity, emotion, preference, life, thought, consolidated.
    Returns None if input is None, or the mapped value (original value if no mapping found).
    """
    if memory_type is None:
        return None
    return _TYPE_NORMALIZE_MAP.get(memory_type, memory_type)


async def log_memory_changelog(
    user_id: str,
    memory_id: str,
    operation: str,
    old_value: str | None = None,
    new_value: str | None = None,
    workspace_id: str | None = None,
) -> None:
    """Write a memory changelog entry for portrait generation."""
    try:
        workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)
        await db.memorychangelog.create(
            data={
                "user": {"connect": {"id": user_id}},
                "memoryId": memory_id,
                "operation": operation,
                "oldValue": old_value,
                "newValue": new_value,
                "workspaceId": workspace_id,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to write changelog: {e}")

DEDUP_THRESHOLD = 0.9


async def is_duplicate(
    user_id: str,
    content: str,
    embedding: list[float],
    workspace_id: str | None = None,
) -> bool:
    """Check if a similar memory already exists (cosine > 0.9)."""
    results = await search_by_embedding(embedding, user_id, top_k=5, workspace_id=workspace_id)
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
    main_category: str | None = None,
    sub_category: str | None = None,
    source: str = "user",
    occur_time: datetime | None = None,
    workspace_id: str | None = None,
) -> str | None:
    """Store a memory with deduplication.

    Returns memory_id if stored, None if duplicate.
    Args:
        source: "user" for memories about the user, "ai" for AI self-memories.
    """
    taxonomy = resolve_taxonomy(
        main_category=main_category,
        sub_category=sub_category,
        legacy_type=normalize_memory_type(memory_type),
    )
    memory_type = normalize_memory_type(taxonomy.legacy_type)

    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)

    # Generate embedding
    embedding = await generate_embedding(content)

    # Deduplication check
    if await is_duplicate(user_id, content, embedding, workspace_id=workspace_id):
        return None

    # Store in PostgreSQL (routed to memories_user or memories_ai)
    create_data = dict(
        userId=user_id,
        content=content,
        summary=summary or content[:200],
        level=level,
        importance=importance,
        type=memory_type,
        mainCategory=taxonomy.main_category,
        subCategory=taxonomy.sub_category,
        workspaceId=workspace_id,
    )
    if occur_time is not None:
        create_data["occurTime"] = occur_time
    memory = await memory_repo.create(source=source, **create_data)

    # Store embedding
    await store_embedding(memory.id, embedding)

    # Log changelog for portrait generation
    await log_memory_changelog(
        user_id,
        memory.id,
        "insert",
        new_value=content,
        workspace_id=workspace_id,
    )

    logger.info(f"Stored memory L{level} (importance={importance:.2f}): {content[:50]}")
    return memory.id
