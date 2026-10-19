"""A relevance verdict needs a supported relation, not a shared adjective."""
from __future__ import annotations

import hashlib
import json
from typing import Any

_ALLOWED = {"same_activity", "same_place", "exploration"}
_RELATIONS = _ALLOWED | {"attribute_only", "unrelated", "negative"}


def reference_indices(value: Any, count: int, facts: dict) -> list[int] | None:
    if not isinstance(value, dict):
        return None
    results = value.get("reference_results")
    if not isinstance(results, list) or len(results) != count:
        return None
    corpus = "\n".join(str(facts.get(k) or "") for k in ("activity_name", "location", "category", "description"))
    seen, kept = set(), []
    for row in results:
        if (not isinstance(row, dict) or type(row.get("index")) is not int
                or row["index"] not in range(count) or row["index"] in seen
                or not isinstance(row.get("relation"), str) or row["relation"] not in _RELATIONS):
            return None
        seen.add(row["index"])
        if row["relation"] in _ALLOWED:
            quote = row.get("fact_quote")
            # Formatting whitespace changes no fact; words and punctuation stay
            # exact. The user's original reference is never normalized.
            if (not isinstance(quote, str) or len(quote.strip()) < 2
                    or "".join(quote.split()) not in "".join(corpus.split())):
                return None
            kept.append(row["index"])
    # Exhaustive per-reference verdicts are authoritative. Redundant totals in
    # earlier Web versions were copied from examples and contradicted the rows.
    return sorted(kept)


def binding(card: dict, references: list[dict], template: str) -> str:
    # Ephemeral reuse within this one generation; never cached across users.
    identity = {k: card.get(k) for k in ("candidate_id", "title", "location_name", "address", "category", "starts_at", "ends_at")}
    value = json.dumps([identity, references, template], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(value.encode()).hexdigest()
