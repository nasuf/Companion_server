"""Personality system.

Big Five personality model with vector anchoring to prevent drift.
"""

import logging

from app.services.llm.models import get_embedding_model

logger = logging.getLogger(__name__)

ANCHOR_THRESHOLD = 0.85


def big_five_to_description(personality: dict) -> str:
    """Convert Big Five scores to natural language description."""
    descriptions = []

    openness = personality.get("openness", 0.5)
    if openness > 0.7:
        descriptions.append("curious and imaginative")
    elif openness < 0.3:
        descriptions.append("practical and conventional")

    conscientiousness = personality.get("conscientiousness", 0.5)
    if conscientiousness > 0.7:
        descriptions.append("organized and disciplined")
    elif conscientiousness < 0.3:
        descriptions.append("spontaneous and flexible")

    extraversion = personality.get("extraversion", 0.5)
    if extraversion > 0.7:
        descriptions.append("outgoing and energetic")
    elif extraversion < 0.3:
        descriptions.append("reserved and thoughtful")

    agreeableness = personality.get("agreeableness", 0.5)
    if agreeableness > 0.7:
        descriptions.append("warm and empathetic")
    elif agreeableness < 0.3:
        descriptions.append("direct and analytical")

    neuroticism = personality.get("neuroticism", 0.5)
    if neuroticism > 0.7:
        descriptions.append("emotionally expressive and sensitive")
    elif neuroticism < 0.3:
        descriptions.append("calm and emotionally stable")

    if not descriptions:
        return "balanced and adaptable"

    return ", ".join(descriptions)


async def compute_personality_vector(personality: dict) -> list[float]:
    """Generate an embedding vector from personality description."""
    description = big_five_to_description(personality)
    model = get_embedding_model()
    return await model.aembed_query(description)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_personality_anchor(
    new_vector: list[float],
    anchor_vector: list[float],
    threshold: float = ANCHOR_THRESHOLD,
) -> list[float]:
    """Check personality drift and reset to anchor if exceeded.

    Returns the vector to use (either new or anchor).
    """
    sim = cosine_similarity(new_vector, anchor_vector)
    if sim < threshold:
        logger.warning(f"Personality drift detected (similarity={sim:.3f}), resetting to anchor")
        return anchor_vector
    return new_vector


def update_personality_traits(
    current: dict,
    observed: dict,
    learning_rate: float = 0.05,
) -> dict:
    """Update personality traits with a slow learning rate.

    trait_new = trait_old * (1 - lr) + observed * lr
    """
    updated = {}
    for key in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        current_val = current.get(key, 0.5)
        observed_val = observed.get(key, current_val)
        updated[key] = current_val * (1 - learning_rate) + observed_val * learning_rate
    return updated
