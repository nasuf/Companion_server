"""Context selector.

Selects memories to fit within the 800-token prompt budget.
Classifies each memory by relevance: strong (score ≥ 0.7) / medium (0.4-0.7).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassifiedMemory:
    """记忆项，附带相关度分级。"""
    text: str
    relevance: str  # "strong" | "medium"
    score: float


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Chinese: ~1.5 token per char; ASCII: ~0.25 token per char."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_chars = len(text) - cjk
    return int(cjk * 1.5 + ascii_chars * 0.25)


def select_context(
    ranked_memories: list[dict],
    token_budget: int = 800,
) -> list[ClassifiedMemory]:
    """Select memories to fit within token budget, with relevance classification.

    Classification:
    - strong: rank_score ≥ 0.7 → "你清楚记得的事"
    - medium: 0.4 ≤ rank_score < 0.7 → "你有印象的事"

    Returns list of ClassifiedMemory.
    """
    selected: list[ClassifiedMemory] = []
    used_tokens = 0
    seen_ids: set[str] = set()

    for mem in ranked_memories:
        mid = mem.get("id", "")
        if mid in seen_ids:
            continue

        text = mem.get("summary") or mem.get("content", "")
        if not text:
            continue
        tokens = estimate_tokens(text)

        if used_tokens + tokens > token_budget:
            continue

        score = float(mem.get("rank_score", mem.get("score", 0.5)))
        if score >= 0.7:
            relevance = "strong"
        else:
            relevance = "medium"

        seen_ids.add(mid)
        selected.append(ClassifiedMemory(text=text, relevance=relevance, score=score))
        used_tokens += tokens

    return selected
