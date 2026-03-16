"""Memory scoring service.

score = rule_score * 0.4 + llm_score * 0.6

rule_score = 0.35*recency + 0.35*entity_match + 0.2*emotional_weight + 0.1*novelty
llm_score = 0.6*relevance + 0.4*importance
"""

import logging
import math
from datetime import datetime, timezone

from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompts.extraction_prompts import MEMORY_SCORING_PROMPT as LLM_SCORING_PROMPT

logger = logging.getLogger(__name__)

# Time decay constant (7 days in seconds)
TAU = 7 * 24 * 3600


def compute_recency(created_at: datetime) -> float:
    """Exponential time decay: e^(-dt/tau)."""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    dt = (now - created_at).total_seconds()
    return math.exp(-dt / TAU)


def compute_entity_match(memory_entities: list[str], context_entities: list[str]) -> float:
    """Score based on entity overlap."""
    if not memory_entities or not context_entities:
        return 0.0
    mem_set = {e.lower() for e in memory_entities}
    ctx_set = {e.lower() for e in context_entities}
    overlap = mem_set & ctx_set
    return min(1.0, len(overlap) / max(1, len(ctx_set)))


def compute_emotional_weight(emotion: dict | None) -> float:
    """Score based on emotional intensity."""
    if not emotion:
        return 0.0
    pleasure_intensity = abs(emotion.get("pleasure", 0.0))  # [-1,1], neutral=0
    arousal_intensity = abs(emotion.get("arousal", 0.5) - 0.5) * 2  # [0,1], neutral=0.5 → rescale to [0,1]
    return min(1.0, (pleasure_intensity + arousal_intensity) / 2)


def compute_rule_score(
    created_at: datetime,
    memory_entities: list[str],
    context_entities: list[str],
    emotion: dict | None,
    novelty: float = 1.0,
) -> float:
    """Compute rule-based score."""
    recency = compute_recency(created_at)
    entity = compute_entity_match(memory_entities, context_entities)
    emotional = compute_emotional_weight(emotion)

    return (
        0.35 * recency
        + 0.35 * entity
        + 0.20 * emotional
        + 0.10 * novelty
    )


async def compute_llm_score(memory_summary: str, context: str) -> float:
    """Compute LLM-based relevance + importance score."""
    model = get_utility_model()
    prompt = LLM_SCORING_PROMPT.format(
        memory_summary=memory_summary, context=context
    )
    try:
        result = await invoke_json(model, prompt)
        relevance = float(result.get("relevance", 0.5))
        importance = float(result.get("importance", 0.5))
        return 0.6 * relevance + 0.4 * importance
    except Exception as e:
        logger.error(f"LLM scoring failed: {e}")
        return 0.5


async def compute_memory_score(
    memory_summary: str,
    created_at: datetime,
    memory_entities: list[str],
    context_entities: list[str],
    emotion: dict | None,
    context: str,
    novelty: float = 1.0,
) -> float:
    """Compute final memory score: rule*0.4 + llm*0.6."""
    rule = compute_rule_score(
        created_at, memory_entities, context_entities, emotion, novelty
    )
    llm = await compute_llm_score(memory_summary, context)
    return rule * 0.4 + llm * 0.6
