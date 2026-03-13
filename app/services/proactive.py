"""Proactive sharing service.

AI-initiated messages based on memory + personality + relationship.
"""

import logging

from app.services.llm.models import get_chat_model, invoke_text
from app.services.memory.retrieval import retrieve_memories, format_memories_for_prompt
from app.services.emotion import get_ai_emotion

logger = logging.getLogger(__name__)

PROACTIVE_PROMPT = """You are an AI companion considering whether to proactively share something with the user.

Your current emotion state:
Valence: {valence}, Arousal: {arousal}, Dominance: {dominance}

Recent memories about the user:
{memories}

Based on your personality and these memories, generate a short, natural message
to share with the user. It could be:
- A follow-up on something they mentioned
- A relevant thought or observation
- A caring check-in

If there's nothing meaningful to share, return "SKIP".

Message:"""


async def generate_proactive_message(
    user_id: str,
    agent_id: str,
) -> str | None:
    """Generate a proactive message, or None if nothing to share."""
    try:
        # Get memories
        memories = await retrieve_memories("", user_id, semantic_k=3, recent_k=5, important_k=2)
        memory_strings = format_memories_for_prompt(memories)

        # Get emotion
        emotion = await get_ai_emotion(agent_id)

        prompt = PROACTIVE_PROMPT.format(
            valence=emotion.get("valence", 0),
            arousal=emotion.get("arousal", 0),
            dominance=emotion.get("dominance", 0),
            memories="\n".join(f"- {m}" for m in memory_strings) or "No memories yet.",
        )

        model = get_chat_model()
        response = await invoke_text(model, prompt)
        response = response.strip()

        if response == "SKIP" or len(response) < 5:
            return None

        return response

    except Exception as e:
        logger.error(f"Proactive message generation failed: {e}")
        return None
