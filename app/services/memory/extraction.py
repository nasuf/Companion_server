"""Memory extraction service.

Uses a small LLM to extract structured memory from conversation.
Output: {memories, entities, preferences, topics}
"""

import logging

from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompts.extraction_prompts import MEMORY_EXTRACTION_PROMPT as EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


async def extract_memories(conversation: str) -> dict:
    """Extract structured memory from a conversation string.

    Returns dict with keys: memories, entities, preferences, topics.
    """
    model = get_utility_model()
    prompt = EXTRACTION_PROMPT.format(conversation=conversation)

    try:
        result = await invoke_json(model, prompt)
        # Validate structure
        result.setdefault("memories", [])
        result.setdefault("entities", [])
        result.setdefault("preferences", [])
        result.setdefault("topics", [])
        return result
    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")
        return {"memories": [], "entities": [], "preferences": [], "topics": []}
