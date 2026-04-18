"""Memory extraction service.

Uses a small LLM to extract structured memory from conversation.
Output: {memories, entities, preferences, topics}
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.llm.models import get_chat_model, invoke_json
from app.services.memory.taxonomy import TAXONOMY_MATRIX
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


def _taxonomy_list_text() -> str:
    """Render the L1 (user) taxonomy as bullet text for the extraction prompt.

    Always uses the L1 superset — extraction assigns a tentative level and
    the downstream pipeline will demote/reject if the (source, level, main)
    slice turns out to forbid the sub-category.
    """
    return "\n".join(
        f"- {main}：{'、'.join(subs)}"
        for main, subs in TAXONOMY_MATRIX["user"][1].items()
    )


async def extract_memories(conversation: str) -> dict:
    """Extract structured memory from a conversation string.

    Returns dict with keys: memories, entities, preferences, topics.
    """
    model = get_chat_model()  # spec §2.1.3: 大模型精细处理(拆分+分类+打分)
    now = datetime.now(ZoneInfo(settings.schedule_timezone))
    prompt = (await get_prompt_text("memory.extraction")).format(
        conversation=conversation,
        current_time=now.strftime("%Y-%m-%d %H:%M %A"),
        taxonomy_list=_taxonomy_list_text(),
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
                summary=mem.get("summary", ""),
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
