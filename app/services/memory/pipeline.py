"""Async memory pipeline.

Orchestrates: extract -> score -> dedup -> store -> embed -> graph update.
Runs as FastAPI BackgroundTasks (non-blocking).
"""

import logging

from app.services.memory.extraction import extract_memories
from app.services.memory.storage import store_memory
from app.services.graph_service import update_graph_from_extraction

logger = logging.getLogger(__name__)


async def process_memory_pipeline(
    user_id: str,
    conversation_text: str,
) -> list[str]:
    """Run the full memory extraction and storage pipeline.

    Returns list of stored memory IDs.
    """
    # Step 1: Extract structured memories
    extraction = await extract_memories(conversation_text)
    memories = extraction.get("memories", [])

    if not memories:
        logger.info("No memories extracted from conversation")
        return []

    stored_ids: list[str] = []

    # Step 2: Store each memory with dedup
    for mem in memories:
        summary = mem.get("summary", "")
        level = mem.get("level", 3)
        importance = mem.get("importance", 0.5)
        memory_type = mem.get("type")

        # Adjust importance based on emotion
        emotion = mem.get("emotion")
        if emotion:
            valence = abs(emotion.get("valence", 0.0))
            importance = min(1.0, importance + valence * 0.2)

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
