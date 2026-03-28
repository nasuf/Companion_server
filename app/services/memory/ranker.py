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
    category_match: bool = False,
    level: int = 3,
) -> float:
    """Compute the hybrid ranking score for a memory."""
    recency = compute_recency(_parse_datetime(created_at))

    relationship_weight = 1.0 if has_entity_match else 0.3
    category_weight = 1.0 if category_match else 0.4
    level_weight = {1: 0.95, 2: 1.0, 3: 0.85}.get(level, 0.85)

    return (
        0.30 * similarity
        + 0.25 * recency
        + 0.25 * importance
        + 0.10 * relationship_weight
        + 0.10 * category_weight
    ) * level_weight


def rank_memories(
    candidates: list[dict],
    context_entities: list[str] | None = None,
    context_categories: list[str] | None = None,
    context_sub_categories: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """Rank candidate memories and return top_k.

    Each candidate dict should have: similarity, created_at, importance.
    """
    entity_set = {e.lower() for e in (context_entities or [])}
    category_set = {c.lower() for c in (context_categories or [])}
    sub_category_set = {c.lower() for c in (context_sub_categories or [])}

    scored = []
    for mem in candidates:
        # Check entity match
        mem_content = (mem.get("summary") or mem.get("content") or "").lower()
        has_match = any(e in mem_content for e in entity_set) if entity_set else False
        mem_main_category = str(mem.get("main_category", "")).lower()
        mem_sub_category = str(mem.get("sub_category", "")).lower()
        category_match = (
            (mem_main_category in category_set if category_set else False)
            or (mem_sub_category in sub_category_set if sub_category_set else False)
        )

        score = compute_ranker_score(
            similarity=float(mem.get("similarity", 0)),
            created_at=mem.get("created_at", ""),
            importance=float(mem.get("importance", 0.5)),
            has_entity_match=has_match,
            category_match=category_match,
            level=int(mem.get("level", 3) or 3),
        )
        mem["rank_score"] = score
        scored.append(mem)

    scored.sort(key=lambda x: x["rank_score"], reverse=True)
    return scored[:top_k]
