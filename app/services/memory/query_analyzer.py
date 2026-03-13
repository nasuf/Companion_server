"""Query analyzer for hybrid retrieval.

Uses small model to determine retrieval strategy for a user message.
"""

import logging

from app.services.llm.models import get_utility_model, invoke_json

logger = logging.getLogger(__name__)

ANALYZER_PROMPT = """Analyze the user message and determine what retrieval is needed.

User message: {message}

Return JSON:
{{
  "intent": "ask_preference|ask_event|ask_person|casual|greeting|question|other",
  "entities": ["list of named entities mentioned"],
  "retrieve_memory": true or false,
  "retrieve_graph": true or false,
  "retrieve_structured": true or false
}}

Rules:
- retrieve_memory: true if the message references past conversations or needs context
- retrieve_graph: true if the message involves relationships, preferences, or entities
- retrieve_structured: true if the message asks for specific stored data (name, birthday, etc.)
- For casual greetings, all can be false
"""


async def analyze_query(message: str) -> dict:
    """Analyze user message to determine retrieval strategy."""
    model = get_utility_model()
    prompt = ANALYZER_PROMPT.format(message=message)

    try:
        result = await invoke_json(model, prompt)
        result.setdefault("intent", "other")
        result.setdefault("entities", [])
        result.setdefault("retrieve_memory", True)
        result.setdefault("retrieve_graph", False)
        result.setdefault("retrieve_structured", False)
        return result
    except Exception as e:
        logger.warning(f"Query analysis failed: {e}")
        return {
            "intent": "other",
            "entities": [],
            "retrieve_memory": True,
            "retrieve_graph": False,
            "retrieve_structured": False,
        }
