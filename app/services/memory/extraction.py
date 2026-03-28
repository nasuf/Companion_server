"""Memory extraction service.

Uses a small LLM to extract structured memory from conversation.
Output: {memories, entities, preferences, topics}
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory.taxonomy import resolve_taxonomy
from app.services.prompt_store import get_prompt_text

logger = logging.getLogger(__name__)


async def extract_memories(conversation: str) -> dict:
    """Extract structured memory from a conversation string.

    Returns dict with keys: memories, entities, preferences, topics.
    """
    model = get_utility_model()
    now = datetime.now(ZoneInfo(settings.schedule_timezone))
    prompt = (await get_prompt_text("memory.extraction")).format(
        conversation=conversation,
        current_time=now.strftime("%Y-%m-%d %H:%M %A"),
    )

    try:
        result = await invoke_json(model, prompt)
        # Validate structure
        result.setdefault("memories", [])
        result.setdefault("entities", [])
        result.setdefault("preferences", [])
        result.setdefault("topics", [])
        sanitized_memories = []
        for mem in result["memories"]:
            if not isinstance(mem, dict):
                continue
            from app.services.memory.normalization import normalize_memory_category
            taxonomy = await normalize_memory_category(
                main_category=mem.get("main_category"),
                sub_category=mem.get("sub_category"),
                legacy_type=mem.get("type"),
            )
            mem["main_category"] = taxonomy.main_category
            mem["sub_category"] = taxonomy.sub_category
            mem["type"] = taxonomy.legacy_type
            sanitized_memories.append(mem)
        result["memories"] = sanitized_memories
        return result
    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")
        return {"memories": [], "entities": [], "preferences": [], "topics": []}
