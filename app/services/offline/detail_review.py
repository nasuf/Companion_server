"""Partial acceptance requires an explicit verdict for every submitted paragraph."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services.offline.copy_guards import bounded_issues


def paragraphs(text: str) -> list[str]:
    values = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    return values if 1 <= len(values) <= 8 else []


@dataclass(frozen=True)
class ParagraphReview:
    supported: bool | None
    rejected: frozenset[int] = frozenset()
    issues: tuple[str, ...] = ()


def review_result(value: Any, submitted: set[int]) -> ParagraphReview:
    if (not isinstance(value, dict) or type(value.get("supported")) is not bool
            or not isinstance(value.get("unsupported_claims"), list)):
        return ParagraphReview(None)
    issues = bounded_issues(value.get("unsupported_claims"), limit=8, length=200)
    results = value.get("section_results")
    # A whole-text rejection can still reject everything. An alleged pass with
    # no paragraph verdicts cannot certify anything under this new contract.
    if results is None:
        return ParagraphReview(None if value["supported"] else False, issues=tuple(issues))
    if not isinstance(results, list) or len(results) != len(submitted):
        return ParagraphReview(None)
    seen, rejected = set(), set()
    for item in results:
        if (not isinstance(item, dict) or type(item.get("section_id")) is not int
                or item["section_id"] not in submitted or item["section_id"] in seen
                or type(item.get("supported")) is not bool
                or not isinstance(item.get("unsupported_claims"), list)):
            return ParagraphReview(None)
        ident = item["section_id"]
        seen.add(ident)
        claims = bounded_issues(item["unsupported_claims"], limit=3, length=200)
        if item["supported"] and item["unsupported_claims"] or not item["supported"] and not claims:
            return ParagraphReview(None)
        if not item["supported"]:
            rejected.add(ident)
            issues.extend(f"第{ident + 1}段：{claim}" for claim in claims)
    if seen != submitted or value["supported"] != (not rejected):
        return ParagraphReview(None)
    if not rejected and value["unsupported_claims"]:
        return ParagraphReview(None)
    return ParagraphReview(not rejected, frozenset(rejected), tuple(dict.fromkeys(issues)))


def review_payload(parts: dict[int, str]) -> str:
    import json
    return "推荐详情分段：" + json.dumps(
        {"sections": [{"section_id": i, "text": text} for i, text in parts.items()]}, ensure_ascii=False)


def retained_text(parts: dict[int, str]) -> str | None:
    text = "\n\n".join(parts[i] for i in sorted(parts))
    # A lone heading or a surviving address is not a useful partial description.
    return text if len(text) >= 80 and any(len(part) >= 35 for part in parts.values()) else None
