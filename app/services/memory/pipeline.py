"""Async memory pipeline.

Orchestrates: extract -> conflict check -> score -> dedup -> store -> embed -> graph update.
Runs as FastAPI BackgroundTasks (non-blocking).
"""

import logging

from app.services.memory.extraction import extract_memories
from app.services.memory.filter import should_extract_memory
from app.services.memory.storage import store_memory
from app.services.memory.conflict import detect_conflicts, resolve_conflict
from app.services.graph_service import update_graph_from_extraction

logger = logging.getLogger(__name__)


async def process_memory_pipeline(
    user_id: str,
    conversation_text: str,
) -> list[str]:
    """Run the full memory extraction and storage pipeline.

    Returns list of stored memory IDs.
    """
    # Step 0: Filter — skip extraction for low-value messages
    # Extract last user message from conversation text for filtering
    lines = conversation_text.strip().split("\n")
    last_user_msg = ""
    for line in reversed(lines):
        if line.startswith("user:"):
            last_user_msg = line[5:].strip()
            break
    if last_user_msg and not should_extract_memory(last_user_msg):
        logger.debug("Message filtered out by memory filter, skipping extraction")
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
        level = mem.get("level", 3)
        importance = mem.get("importance", 0.5)
        memory_type = mem.get("type")

        # Adjust importance based on emotion
        emotion = mem.get("emotion")
        if emotion:
            pleasure_abs = abs(emotion.get("pleasure", 0.0))
            importance = min(1.0, importance + pleasure_abs * 0.2)

        # Step 2a: Conflict check for L1 memories
        if level == 1:
            try:
                conflict = await detect_conflicts(user_id, mem)
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
        )

        if memory_id:
            stored_ids.append(memory_id)

            # Step 3: Update graph
            try:
                await update_graph_from_extraction(user_id, memory_id, extraction)
            except Exception as e:
                logger.warning(f"Graph update failed for memory {memory_id}: {e}")

    logger.info(f"Pipeline complete: {len(stored_ids)}/{len(memories)} memories stored")
    return stored_ids
