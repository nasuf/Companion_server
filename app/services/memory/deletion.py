"""Memory deletion by user request.

Detects user intent to delete memories and executes deletion pipeline.
"""

import logging

from app.db import db
from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory.embedding import generate_embedding
from app.services.memory.vector_search import search_by_embedding
from app.services.memory.storage import log_memory_changelog

logger = logging.getLogger(__name__)

# Keywords that may indicate deletion intent
DELETION_KEYWORDS = [
    "忘了", "忘掉", "别记了", "不记得", "删除", "删掉",
    "不要记", "别提了", "忘记", "去掉", "移除",
    "forget", "delete", "remove", "don't remember",
]

DELETION_INTENT_PROMPT = """判断用户是否在要求AI忘记/删除某条记忆。

用户消息：{message}

返回JSON：
{{
  "is_deletion_request": true/false,
  "target_description": "用户想删除的记忆描述（如无则为null）",
  "confidence": 0.0-1.0
}}

规则：
- 用户说"忘了吧"、"别记了"、"删掉那个"等属于删除请求
- 用户说"我忘了"（表达自己忘记）不是删除请求
- 只有明确要求AI删除/忘记时才返回true"""

DELETION_RESPONSE_TEMPLATES = [
    "好的，那件事我不会再提了。",
    "嗯，已经忘掉了~",
    "了解，以后不会再提起这个了。",
    "好吧，就当没发生过。",
]


async def detect_deletion_intent(message: str) -> dict | None:
    """Detect if user wants to delete a memory.

    Returns deletion intent info or None.
    """
    # Quick keyword check
    has_keyword = any(kw in message for kw in DELETION_KEYWORDS)
    if not has_keyword:
        return None

    # Confirm with LLM
    prompt = DELETION_INTENT_PROMPT.format(message=message)
    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"Deletion intent detection failed: {e}")
        return None

    if not result.get("is_deletion_request", False):
        return None
    if result.get("confidence", 0) < 0.8:
        return None

    return result


async def delete_memories_by_description(
    user_id: str,
    description: str,
) -> int:
    """Find and delete memories matching the description.

    Returns number of deleted memories.
    """
    # Generate embedding for the target description
    embedding = await generate_embedding(description)

    # Find similar memories
    results = await search_by_embedding(embedding, user_id, top_k=5)

    deleted = 0
    for r in results:
        sim = r.get("similarity", 0)
        if isinstance(sim, str):
            sim = float(sim)

        # Only delete if similarity is high enough
        if sim < 0.7:
            continue

        memory_id = r.get("memory_id")
        if not memory_id:
            continue

        # Log before delete
        memory = await db.memory.find_unique(where={"id": memory_id})
        if memory:
            await log_memory_changelog(
                user_id, memory_id, "delete",
                old_value=memory.content,
            )

        # Delete embedding first (raw SQL)
        try:
            await db.execute_raw(
                "DELETE FROM memory_embeddings WHERE memory_id = $1",
                memory_id,
            )
        except Exception as e:
            logger.warning(f"Failed to delete embedding for {memory_id}: {e}")

        # Delete memory
        try:
            await db.memory.delete(where={"id": memory_id})
            deleted += 1
            logger.info(f"Deleted memory {memory_id}: {r.get('content', '')[:50]}")
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id}: {e}")

    return deleted


def get_deletion_response() -> str:
    """Get a natural language response for memory deletion."""
    import random
    return random.choice(DELETION_RESPONSE_TEMPLATES)
