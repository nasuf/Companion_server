"""Memory compression service.

Weekly: 10 daily memories -> 1 summary
Monthly: 4 weekly summaries -> 1 monthly summary
"""

import logging
from datetime import datetime, timedelta, timezone

from app.services.memory import memory_repo
from app.services.memory.embedding import generate_embedding, store_embedding
from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompt_store import get_prompt_text

logger = logging.getLogger(__name__)

COMPRESS_PROMPT = """请将下面 {count} 条记忆压缩成一段 2-3 句的简洁中文总结。
保留最重要的事实、偏好和事件。

记忆：
{memories}

要求：
- 输出必须是自然中文
- 不要输出英文总结，除非专有名词本身是英文

压缩总结："""


async def _compress_batch(
    user_id: str,
    batch: list,
    target_level: int,
    importance_bump: float,
    model,
) -> None:
    """Compress a batch of memories into a single summary."""
    mem_texts = "\n".join(f"- {m.content}" for m in batch)
    prompt = (await get_prompt_text("memory.compression")).format(count=len(batch), memories=mem_texts)

    summary = await invoke_text(model, prompt)
    summary = summary.strip()

    max_importance = max(m.importance for m in batch)
    # Inherit source from batch
    batch_source = "ai" if any(m.source == "ai" for m in batch) else "user"
    compressed = await memory_repo.create(
        source=batch_source,
        userId=user_id,
        content=summary,
        summary=summary,
        level=target_level,
        importance=min(1.0, max_importance + importance_bump),
        type="compressed",
    )

    embedding = await generate_embedding(summary)
    await store_embedding(compressed.id, embedding)

    # Batch archive old memories (source=None to cover mixed-source batches)
    batch_ids = [m.id for m in batch]
    await memory_repo.update_many(
        where={"id": {"in": batch_ids}},
        data={"isArchived": True},
    )

    logger.info(
        f"Compressed {len(batch)} L{batch[0].level} -> 1 L{target_level} for user {user_id}"
    )


async def compress_weekly() -> None:
    """Compress daily memories older than 7 days into weekly summaries."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    memories = await memory_repo.find_many(
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

    memories = await memory_repo.find_many(
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
