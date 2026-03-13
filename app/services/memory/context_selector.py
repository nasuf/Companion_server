"""Context selector.

Selects memories to fit within the 800-token prompt budget.
Strategy: 5 semantic + 3 recent + 2 high importance (deduplicated).
"""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars for English)."""
    return len(text) // 4


def select_context(
    ranked_memories: list[dict],
    token_budget: int = 800,
) -> list[str]:
    """Select memories to fit within token budget.

    Returns list of memory summary strings.
    """
    selected: list[str] = []
    used_tokens = 0
    seen_ids: set[str] = set()

    for mem in ranked_memories:
        mid = mem.get("id", "")
        if mid in seen_ids:
            continue

        text = mem.get("summary") or mem.get("content", "")
        tokens = estimate_tokens(text)

        if used_tokens + tokens > token_budget:
            continue

        seen_ids.add(mid)
        selected.append(text)
        used_tokens += tokens

    return selected
