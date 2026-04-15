"""Async memory pipeline.

Orchestrates: extract -> conflict check -> score -> dedup -> store -> embed -> entity link.
Runs as FastAPI BackgroundTasks (non-blocking).
"""

import logging
from datetime import datetime

from app.services.memory.entity_repo import (
    record_entities_for_memory,
    record_preferences_for_memory,
    record_topics_for_memory,
)
from app.services.memory.extraction import extract_memories
from app.services.memory.filter import should_extract_memory
from app.services.memory.storage import store_memory
from app.services.memory.conflict import detect_conflicts, resolve_conflict
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)


async def process_memory_pipeline(
    user_id: str,
    conversation_text: str,
) -> list[str]:
    """Run the full memory extraction and storage pipeline.

    Returns list of stored memory IDs.
    """
    workspace_id = await resolve_workspace_id(user_id=user_id)

    # Step 0: Filter — skip extraction if the entire segment is low-value
    if not should_extract_memory(conversation_text):
        logger.debug("Conversation segment filtered out by memory filter, skipping extraction")
        return []

    # Step 1: Extract structured memories
    extraction = await extract_memories(conversation_text)
    memories = extraction.get("memories", [])

    if not memories:
        logger.info("No memories extracted from conversation")
        return []

    stored_ids: list[str] = []

    # Step 2: Store each memory with dedup and conflict check
    for mem in memories:
        summary = mem.get("summary", "")
        importance = mem.get("importance", 0.5)
        memory_type = mem.get("type")
        main_category = mem.get("main_category")
        sub_category = mem.get("sub_category")

        # Per spec《产品手册·背景信息》§2.3 — level is derived from importance
        # score (0-100), not whatever level the LLM may have guessed:
        #   ≥ 0.85 → L1   |   0.50-0.84 → L2   |   0.10-0.49 → L3   |   < 0.10 → drop
        if importance < 0.10:
            logger.debug(f"Memory dropped (importance={importance:.2f} < 0.10): {summary[:40]}")
            continue
        elif importance >= 0.85:
            level = 1
        elif importance >= 0.50:
            level = 2
        else:
            level = 3

        # Parse occur_time from extraction result
        occur_time: datetime | None = None
        raw_time = mem.get("occur_time")
        if raw_time and isinstance(raw_time, str):
            try:
                occur_time = datetime.fromisoformat(raw_time)
            except ValueError:
                pass

        # Adjust importance based on emotion
        emotion = mem.get("emotion")
        if emotion:
            pleasure_abs = abs(emotion.get("pleasure", 0.0))
            importance = min(1.0, importance + pleasure_abs * 0.2)

        # Step 2a: Conflict check for L1 memories
        if level == 1:
            try:
                conflict = await detect_conflicts(user_id, mem, workspace_id=workspace_id)
                if conflict:
                    action = await resolve_conflict(user_id, conflict, mem)
                    if action in ("updated", "demoted"):
                        logger.info(f"Conflict resolved ({action}), skipping duplicate store")
                        continue
            except Exception as e:
                logger.warning(f"Conflict check failed: {e}")

        memory_id = await store_memory(
            user_id=user_id,
            content=summary,
            summary=summary,
            level=level,
            importance=importance,
            memory_type=memory_type,
            main_category=main_category,
            sub_category=sub_category,
            occur_time=occur_time,
            workspace_id=workspace_id,
            source=mem.get("owner", "user"),
        )

        if memory_id:
            stored_ids.append(memory_id)

            # Step 3: Link entities / topics / preferences to this memory.
            # Best-effort: failure here is advisory (retrieval still works
            # from the memory row + pgvector) so we log-and-continue.
            memory_source = mem.get("owner", "user")
            try:
                await record_entities_for_memory(
                    memory_id=memory_id,
                    memory_source=memory_source,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    entities=extraction.get("entities", []),
                )
                await record_topics_for_memory(
                    memory_id=memory_id,
                    memory_source=memory_source,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    topics=extraction.get("topics", []),
                )
                await record_preferences_for_memory(
                    memory_id=memory_id,
                    memory_source=memory_source,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    preferences=extraction.get("preferences", []),
                )
            except Exception as e:
                logger.warning(f"Entity linking failed for memory {memory_id}: {e}")

    logger.info(f"Pipeline complete: {len(stored_ids)}/{len(memories)} memories stored")
    return stored_ids
