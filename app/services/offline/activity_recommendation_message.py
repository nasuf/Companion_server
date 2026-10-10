"""Private, verbatim relevance evidence → grounded page recommendation message.

Personal inputs are collected after public discovery/cache reads. Only the final
destination's evidence is used, and none is exposed through the public API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from app.services.llm.models import invoke_json, invoke_text
from app.services.offline import repository as repo
from app.services.offline.activity_message_context import recommendation_dialogue
from app.services.offline.content import plain_text
from app.services.offline.llm import (
    get_offline_chat_model as get_chat_model,
    get_offline_small_model as get_utility_model,
)
from app.services.prompting.store import get_prompt_text
from app.services.prompting.registry import PROMPT_DEFINITION_MAP

logger = logging.getLogger(__name__)
_KINDS = ("memory", "preference", "dialogue")


def _whole_items(items: list[str], budget: int) -> list[str]:
    kept: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip() or item in kept:
            continue
        if len(item) <= min(1000, budget):
            kept.append(item)
            budget -= len(item)
    return kept


async def evidence_inputs(
    *, user_id: str, workspace_id: str | None, conversation_id: str | None, tags: list[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    evidence = {"preference": _whole_items(tags[:9], 500), "memory": [], "dialogue": []}
    statuses = {"preference": "collected" if evidence["preference"] else "empty"}
    # Failure to read one private source must not turn assistant memories into a
    # fallback source or prevent a factual recommendation.
    for kind, fetch, budget in (
        ("memory", lambda: repo.recommendation_memory_items(user_id, workspace_id), 3000),
        ("dialogue", lambda: recommendation_dialogue(
            user_id=user_id, workspace_id=workspace_id, conversation_id=conversation_id,
        ), 3000),
    ):
        if kind == "dialogue" and not conversation_id:
            statuses[kind] = "not_available"
            continue
        try:
            async with asyncio.timeout(3):
                evidence[kind] = _whole_items(await fetch(), budget)
            statuses[kind] = "collected" if evidence[kind] else "empty"
        except Exception as exc:
            statuses[kind] = "failed"
            logger.info("[offline] recommendation evidence %s unavailable (%s)", kind, type(exc).__name__)
    return evidence, statuses


def exact_references(raw: Any, evidence: dict[str, list[str]]) -> list[dict[str, str]]:
    """Complete-item equality, not substring matching: preserves negation/context."""
    if not isinstance(raw, list):
        return []
    references: list[dict[str, str]] = []
    seen: set[str] = set()
    for quote in raw[:12]:
        if not isinstance(quote, str) or quote in seen:
            continue
        kind = next((kind for kind in _KINDS if quote in evidence.get(kind, [])), None)
        if kind:
            references.append({"kind": kind, "text": quote})
            seen.add(quote)
        if len(references) == 6:
            break
    return references


def generic_message(card: dict[str, Any]) -> str:
    name = plain_text(card.get("title") or card.get("location_name")) or "这个去处"
    event = (card.get("discovery_metadata") or {}).get("kind") == "event"
    timing = "先看看活动时间和参与方式，再按自己的安排决定" if event else "出发前看看开放时间，再按自己的安排决定"
    return (
        f"想把{name}推荐给你，给下次出门多留一个选择。"
        f"{timing}，不用为了这次推荐特意挤出时间。"
        "可以先读读这里的介绍，看看是不是你想尝试的内容。"
        "感兴趣的话，要不要找个方便的时候去看看？暂时不想去也没关系，先放着就好。"
    )


async def attach_recommendation_message(
    card: dict[str, Any], evidence: dict[str, list[str]],
    source_statuses: dict[str, str] | None = None,
) -> dict[str, Any]:
    from app.services.offline.activity_generation import _date_time_text

    references = exact_references(card.pop("user_relevance", []), evidence)
    # No raw search pages, unrelated memories, URLs or internal instructions go
    # into the writing prompt. Exact venue/session fields are never rewritten.
    facts = {
        "activity_name": card.get("title") or card.get("location_name") or "",
        "date_time": _date_time_text(card),
        "location": " ".join(str(card.get(k) or "") for k in ("location_name", "address")).strip(),
        "category": card.get("category") or "",
        "description": card.get("description") or "",
        "activity_summary": card.get("summary") or "",
    }
    message, status = generic_message(card), "fallback_unavailable"
    relevance_status = "not_run"
    try:
        async with asyncio.timeout(30):
            template = await get_prompt_text("offline.activity_recommendation_message")
            check_template = await get_prompt_text("offline.recommendation_message_check")
            if any(text == PROMPT_DEFINITION_MAP[key].default_text for key, text in (
                ("offline.activity_recommendation_message", template),
                ("offline.recommendation_message_check", check_template),
            )):
                status = "fallback_not_published"
                raise ValueError("Recommendation message templates await publication")

            async def check_message(text: str) -> Any:
                prompt = check_template.format(
                    review_phase="relevance" if not text else "message",
                    facts_json=json.dumps(facts, ensure_ascii=False),
                    references_json=json.dumps(references, ensure_ascii=False),
                    message=json.dumps(text, ensure_ascii=False),
                )
                async with asyncio.timeout(8):
                    return await invoke_json(get_utility_model(), prompt)

            # Verify relevance before writing, so an unrelated but exact quote
            # cannot prime the writer to invent a connection to the venue.
            if references:
                relevance = await check_message("")
                indices = relevance.get("relevant_indices") if isinstance(relevance, dict) else None
                if (not isinstance(indices, list) or any(type(i) is not int or
                        i < 0 or i >= len(references) for i in indices)
                        or len(set(indices)) != len(indices)
                        or not isinstance(relevance.get("supported"), bool)
                        or not isinstance(relevance.get("unsupported_claims"), list)
                        or (relevance["supported"] is True and
                            (len(indices) != len(references) or relevance["unsupported_claims"] != []))
                        or (relevance["supported"] is False and len(indices) == len(references))):
                    status = "fallback_invalid"
                    raise ValueError("Invalid relevance check")
                relevance_status = "verified" if len(indices) == len(references) else "filtered"
                references = [references[i] for i in sorted(indices)]
            else:
                relevance_status = "no_evidence"
            grouped = {kind: [r["text"] for r in references if r["kind"] == kind] for kind in _KINDS}
            feedback: list[str] = []
            # At most one repair, with the entire relevance/write/check sequence
            # sharing a deadline. A failed repair always leaves a factual fallback.
            for attempt in range(2):
                prompt = template.format(
                    **facts,
                    user_memory=json.dumps(grouped["memory"], ensure_ascii=False),
                    user_preference=json.dumps(grouped["preference"], ensure_ascii=False),
                    dialogue_context=json.dumps(grouped["dialogue"], ensure_ascii=False),
                    revision_feedback=json.dumps(feedback, ensure_ascii=False),
                )
                async with asyncio.timeout(15):
                    raw = (await invoke_text(get_chat_model(), prompt)).strip()
                candidate = re.sub(r"\s+", " ", raw)
                if (not 100 <= len(candidate) <= 320 or re.search(r"https?://|www\.", candidate)
                        or candidate.startswith(("{", "[", "```"))):
                    status = "fallback_invalid"
                    break
                checked = await check_message(candidate)
                indices = checked.get("relevant_indices") if isinstance(checked, dict) else None
                if (isinstance(checked, dict) and checked.get("supported") is True
                        and checked.get("unsupported_claims") == []
                        and isinstance(indices, list)
                        and all(type(i) is int and 0 <= i < len(references) for i in indices)
                        and len(set(indices)) == len(indices)):
                    message = candidate
                    status = "verified" if references else "verified_no_evidence"
                    break
                status = "fallback_invalid"
                issues = checked.get("unsupported_claims") if isinstance(checked, dict) else None
                if attempt or not isinstance(issues, list) or not issues:
                    break
                feedback = [item[:200] for item in issues[:5] if isinstance(item, str)]
                if not feedback:
                    break
    except TimeoutError:
        status = "fallback_unavailable"
        logger.info("[offline] recommendation message deadline reached")
    except Exception as exc:
        logger.info("[offline] recommendation message fallback (%s)", type(exc).__name__)
    if status.startswith("fallback"):
        references = []  # Rejected assertions must not survive in storage.
    metadata = card.setdefault("discovery_metadata", {})
    metadata.update(
        recommendation_message=message,
        recommendation_message_status=status,
        recommendation_message_generated_at=datetime.now(UTC).isoformat(),
        user_relevance=references,
        user_relevance_count=len(references),
        recommendation_relevance_status=relevance_status,
        recommendation_evidence_status=source_statuses or {kind: "not_collected" for kind in _KINDS},
    )
    logger.info("[offline] recommendation message=%s references=%d", status, len(references))
    return card
