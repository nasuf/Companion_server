"""Memory compression service.

Weekly: 10 daily memories -> 1 summary
Monthly: 4 weekly summaries -> 1 monthly summary
"""

import logging
from datetime import datetime, timedelta, timezone

from app.db import db
from app.services.memory.embedding import generate_embedding, store_embedding
from app.services.llm.models import get_utility_model, invoke_text

logger = logging.getLogger(__name__)

COMPRESS_PROMPT = """Compress the following {count} memories into a single concise summary (2-3 sentences).
Preserve the most important facts, preferences, and events.

Memories:
{memories}

Compressed summary:"""


async def _compress_batch(
    user_id: str,
    batch: list,
    target_level: int,
    importance_bump: float,
    model,
) -> None:
    """Compress a batch of memories into a single summary."""
    mem_texts = "\n".join(f"- {m.content}" for m in batch)
    prompt = COMPRESS_PROMPT.format(count=len(batch), memories=mem_texts)

    summary = await invoke_text(model, prompt)
    summary = summary.strip()

    max_importance = max(m.importance for m in batch)
    compressed = await db.memory.create(
        data={
            "user": {"connect": {"id": user_id}},
            "content": summary,
            "summary": summary,
            "level": target_level,
            "importance": min(1.0, max_importance + importance_bump),
            "type": "compressed",
        }
    )

    embedding = await generate_embedding(summary)
    await store_embedding(compressed.id, embedding)

    # Batch archive old memories
    batch_ids = [m.id for m in batch]
    await db.memory.update_many(
        where={"id": {"in": batch_ids}},
        data={"isArchived": True},
    )

    logger.info(
        f"Compressed {len(batch)} L{batch[0].level} -> 1 L{target_level} for user {user_id}"
    )


async def compress_weekly() -> None:
    """Compress daily memories older than 7 days into weekly summaries."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    memories = await db.memory.find_many(
        where={
            "level": 3,
            "isArchived": False,
            "type": {"not": "compressed"},
            "createdAt": {"lt": cutoff},
        },
        order={"createdAt": "asc"},
        take=100,
    )

    if len(memories) < 3:
        logger.info("Not enough L3 memories for weekly compression")
        return

    user_groups: dict[str, list] = {}
    for m in memories:
        user_groups.setdefault(m.userId, []).append(m)

    model = get_utility_model()

    for user_id, user_mems in user_groups.items():
        for i in range(0, len(user_mems), 10):
            batch = user_mems[i : i + 10]
            if len(batch) < 2:
                continue
            try:
                await _compress_batch(user_id, batch, target_level=2, importance_bump=0.05, model=model)
            except Exception as e:
                logger.error(f"Weekly compression failed: {e}")


async def compress_monthly() -> None:
    """Compress weekly summaries older than 30 days into monthly summaries."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    memories = await db.memory.find_many(
        where={
            "level": 2,
            "isArchived": False,
            "type": {"in": ["compressed", "consolidated"]},
            "createdAt": {"lt": cutoff},
        },
        order={"createdAt": "asc"},
        take=50,
    )

    if len(memories) < 2:
        logger.info("Not enough L2 summaries for monthly compression")
        return

    user_groups: dict[str, list] = {}
    for m in memories:
        user_groups.setdefault(m.userId, []).append(m)

    model = get_utility_model()

    for user_id, user_mems in user_groups.items():
        for i in range(0, len(user_mems), 4):
            batch = user_mems[i : i + 4]
            if len(batch) < 2:
                continue
            try:
                await _compress_batch(user_id, batch, target_level=1, importance_bump=0.1, model=model)
            except Exception as e:
                logger.error(f"Monthly compression failed: {e}")
