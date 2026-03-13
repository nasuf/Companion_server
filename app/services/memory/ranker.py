"""Memory ranker.

Ranks candidate memories using:
  0.35 * semantic_similarity
+ 0.25 * recency
+ 0.25 * importance
+ 0.15 * relationship_weight

Input: top 50 candidates -> Output: top 20 ranked results.
"""

from datetime import datetime, timezone

from app.services.memory.scoring import compute_recency


def _parse_datetime(created_at: datetime | str) -> datetime:
    """Parse a datetime from string or pass through datetime objects."""
    if isinstance(created_at, str):
        try:
            return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)
    return created_at


def compute_ranker_score(
    similarity: float,
    created_at: datetime | str,
    importance: float,
    has_entity_match: bool,
) -> float:
    """Compute the hybrid ranking score for a memory."""
    recency = compute_recency(_parse_datetime(created_at))

    relationship_weight = 1.0 if has_entity_match else 0.3

    return (
        0.35 * similarity
        + 0.25 * recency
        + 0.25 * importance
        + 0.15 * relationship_weight
    )


def rank_memories(
    candidates: list[dict],
    context_entities: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """Rank candidate memories and return top_k.

    Each candidate dict should have: similarity, created_at, importance.
    """
    entity_set = {e.lower() for e in (context_entities or [])}

    scored = []
    for mem in candidates:
        # Check entity match
        mem_content = (mem.get("summary") or mem.get("content") or "").lower()
        has_match = any(e in mem_content for e in entity_set) if entity_set else False

        score = compute_ranker_score(
            similarity=float(mem.get("similarity", 0)),
            created_at=mem.get("created_at", ""),
            importance=float(mem.get("importance", 0.5)),
            has_entity_match=has_match,
        )
        mem["rank_score"] = score
        scored.append(mem)

    scored.sort(key=lambda x: x["rank_score"], reverse=True)
    return scored[:top_k]
