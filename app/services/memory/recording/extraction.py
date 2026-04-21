"""Memory extraction service.

Spec 第二部分 §2.1 (用户侧) / §2.2 (AI 侧)：两条独立管线，**输入侧决定归属**，
不再让 LLM 从混合对话里推断 owner。由 `side="user"` / `side="ai"` 选择对应的
指令模版（memory.extraction_user / memory.extraction_ai）。

Output: {memories, entities, preferences, topics}
"""

import logging
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.llm.models import get_chat_model, invoke_json
from app.services.memory.taxonomy import TAXONOMY_MATRIX
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

Side = Literal["user", "ai"]

_PROMPT_KEY_BY_SIDE: dict[Side, str] = {
    "user": "memory.extraction_user",
    "ai": "memory.extraction_ai",
}


def _taxonomy_list_text(side: Side) -> str:
    """Render the L1 taxonomy bullets for the given side's extraction prompt.

    Always uses the L1 superset — extraction assigns a tentative level and
    the downstream pipeline will demote/reject if the (source, level, main)
    slice turns out to forbid the sub-category.
    """
    return "\n".join(
        f"- {main}：{'、'.join(subs)}"
        for main, subs in TAXONOMY_MATRIX[side][1].items()
    )


async def extract_memories(conversation: str, side: Side = "user") -> dict:
    """Extract structured memory from a conversation string for one side.

    Args:
        conversation: Recent dialogue lines (`user:` / `assistant:` prefixed).
        side: Which speaker's memories to extract ("user" | "ai"). The owner
            is determined by this path, not inferred by the LLM.

    Returns dict with keys: memories, entities, preferences, topics.
    """
    model = get_chat_model()  # spec §2.1.3 / §2.2.3: 大模型精细处理
    now = datetime.now(ZoneInfo(settings.schedule_timezone))
    prompt = (await get_prompt_text(_PROMPT_KEY_BY_SIDE[side])).format(
        conversation=conversation,
        current_time=now.strftime("%Y-%m-%d %H:%M %A"),
        taxonomy_list=_taxonomy_list_text(side),
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
