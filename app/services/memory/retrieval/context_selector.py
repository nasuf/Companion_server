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
    id: str = ""  # memory row ID for access logging
    importance: float = 0.5
    similarity: float = 0.8
    created_at: str | None = None
    display_score: float = 0.0  # set by reranking in orchestrator


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Chinese: ~1.5 token per char; ASCII: ~0.25 token per char."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_chars = len(text) - cjk
    return int(cjk * 1.5 + ascii_chars * 0.25)


MAX_MEMORIES_INJECTED = 10  # spec §3.2 step 4: 前 10 条硬上限


def select_context(
    ranked_memories: list[dict],
    token_budget: int = 800,
    max_items: int = MAX_MEMORIES_INJECTED,
) -> list[ClassifiedMemory]:
    """Select memories to fit within token budget, with relevance classification.

    spec §3.2 step 4: 前 `max_items` 条 + 不超过 `token_budget` tokens。
    两条限制取较严的一个。

    Classification:
    - strong: rank_score ≥ 0.7 → "你清楚记得的事"
    - medium: 0.4 ≤ rank_score < 0.7 → "你有印象的事"

    Returns list of ClassifiedMemory.
    """
    selected: list[ClassifiedMemory] = []
    used_tokens = 0
    seen_ids: set[str] = set()

    for mem in ranked_memories:
        if len(selected) >= max_items:
            break

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
        relevance = "strong" if score >= 0.7 else "medium"

        seen_ids.add(mid)
        selected.append(ClassifiedMemory(
            text=text,
            relevance=relevance,
            score=score,
            id=mid,
            importance=float(mem.get("importance", 0.5)),
            similarity=float(mem.get("similarity", 0.8)),
            created_at=mem.get("created_at"),
        ))
        used_tokens += tokens

    return selected
