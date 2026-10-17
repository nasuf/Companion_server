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

from langchain_openai import ChatOpenAI

from app.services.llm.models import invoke_json, invoke_text
from app.services.llm.resilience import provider_name
from app.services.offline import repository as repo
from app.services.offline.activity_message_context import recommendation_dialogue
from app.services.offline.content import plain_text
from app.services.offline.copy_guards import (
    bounded_issues, copy_issues, date_relation, local_now, today_text,
)
from app.services.offline.evidence_text import mask_quotes, undated
from app.services.offline.llm import (
    get_offline_chat_model as get_chat_model,
    get_offline_small_model as get_utility_model,
)
from app.services.prompting.store import get_prompt_text
from app.services.prompting.registry import PROMPT_DEFINITION_MAP

logger = logging.getLogger(__name__)
EVIDENCE_KINDS = ("memory", "preference", "dialogue")
_MAX_ITEM_CHARS = 1000


def _whole_items(items: list[str], budget: int) -> list[str]:
    kept: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip() or item in kept:
            continue
        # The date prefix is context, not quote length: a 1000-char quote stays.
        if len(undated(item)) <= _MAX_ITEM_CHARS and len(item) <= budget:
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
    """Complete-item equality, not substring matching: preserves negation/context.

    A quote may omit the date prefix; it still has to equal a whole item, and the
    stored reference keeps the dated original so later writers see when it was said.
    """
    if not isinstance(raw, list):
        return []
    items = [(kind, item) for kind in EVIDENCE_KINDS for item in evidence.get(kind, [])]
    references: list[dict[str, str]] = []
    seen: set[str] = set()
    for quote in raw[:12]:
        if not isinstance(quote, str):
            continue
        # The same words said on several days: the earliest date is the most
        # conservative reading of a transient state ("今天出来走走").
        same_body = sorted((item, kind) for kind, item in items if undated(item) == undated(quote))
        match = next(((kind, item) for kind, item in items if item == quote), None) or (
            (same_body[0][1], same_body[0][0]) if same_body else None)
        if match and match[1] not in seen:
            references.append({"kind": match[0], "text": match[1]})
            seen.add(match[1])
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


def page_facts(card: dict[str, Any], now: datetime) -> dict[str, str]:
    """The only activity facts user-visible copy may rely on.

    No raw search pages, unrelated memories, URLs or internal instructions go
    into the writing prompt. Exact venue/session fields are never rewritten.
    ``today`` and ``date_relation`` exist because models cannot compute either:
    without them a concert four days away was written as "明年".
    """
    from app.services.offline.activity_generation import _date_time_text

    metadata = card.get("discovery_metadata") or {}
    # The provider's own introduction is a fact; the page detail is model copy
    # (whatever its review missed would become a "fact") or a template (which
    # writers then echo). Reviewed detail is used only when nothing else exists.
    description = metadata.get("place_introduction") or (metadata.get("event") or {}).get("description")
    if not description and metadata.get("copy_status") == "verified":
        description = card.get("description")
    return {
        "activity_name": card.get("title") or card.get("location_name") or "",
        "date_time": _date_time_text(card),
        "date_relation": date_relation(card, now),
        "today": today_text(now),
        "location": " ".join(str(card.get(k) or "") for k in ("location_name", "address")).strip(),
        "category": card.get("category") or "",
        "description": description or "",
        "activity_summary": card.get("summary") or "",
    }


def is_published(key: str, text: str) -> bool:
    return text != PROMPT_DEFINITION_MAP[key].default_text


async def review_copy(
    check_template: str, *, phase: str, facts: dict[str, str],
    references: list[dict[str, str]], message: str,
) -> Any:
    """One structured review call; callers own the deadline and the verdict."""
    prompt = check_template.format(
        review_phase=phase,
        facts_json=json.dumps(facts, ensure_ascii=False),
        references_json=json.dumps(references, ensure_ascii=False),
        message=json.dumps(message, ensure_ascii=False),
    )
    model = get_utility_model()
    if provider_name(model) == "dashscope" and isinstance(model, ChatOpenAI):
        # Configure this call's primary model without mutating the shared cached
        # instance or leaking provider-only options into invoke_json's
        # independent Ollama fallback.
        model = model.model_copy(update={
            "temperature": 0,
            "model_kwargs": {**model.model_kwargs, "response_format": {"type": "json_object"}},
        })
    async with asyncio.timeout(8):
        return await invoke_json(model, prompt)


def accepted_review(checked: Any, reference_count: int) -> bool:
    if not (isinstance(checked, dict) and checked.get("supported") is True
            and checked.get("unsupported_claims") == []):
        return False
    if not reference_count:
        # Nothing to index: a copied example "[0]" is noise, not a verdict.
        return True
    indices = checked.get("relevant_indices")
    return (isinstance(indices, list)
            and all(type(i) is int and 0 <= i < reference_count for i in indices)
            and len(set(indices)) == len(indices))


async def attach_recommendation_message(
    card: dict[str, Any], evidence: dict[str, list[str]],
    source_statuses: dict[str, str] | None = None,
) -> dict[str, Any]:
    references = exact_references(card.pop("user_relevance", []), evidence)
    private_quotes = [r["text"] for r in references]
    now = local_now()
    facts = page_facts(card, now)
    message, status = generic_message(card), "fallback_unavailable"
    relevance_status = "not_run"
    issues: list[str] = []
    rejected: list[str] = []  # Reasons of the last rejected draft, kept across a repair.
    try:
        async with asyncio.timeout(30):
            template = await get_prompt_text("offline.activity_recommendation_message")
            check_template = await get_prompt_text("offline.recommendation_message_check")
            if not (is_published("offline.activity_recommendation_message", template)
                    and is_published("offline.recommendation_message_check", check_template)):
                status = "fallback_not_published"
                raise ValueError("Recommendation message templates await publication")

            # Verify relevance before writing, so an unrelated but exact quote
            # cannot prime the writer to invent a connection to the venue.
            if references:
                relevance = await review_copy(check_template, phase="relevance", facts=facts,
                                              references=references, message="")
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
                    issues = ["关联依据核验结果格式无效"]
                    raise ValueError("Invalid relevance check")
                relevance_status = "verified" if len(indices) == len(references) else "filtered"
                references = [references[i] for i in sorted(indices)]
            else:
                relevance_status = "no_evidence"
            grouped = {kind: [r["text"] for r in references if r["kind"] == kind] for kind in EVIDENCE_KINDS}
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
                    status, issues = "fallback_invalid", ["寄语长度或格式不符合要求"]
                    break
                # Objections code can decide exactly never reach the model guard.
                issues = copy_issues(candidate, card, now,
                                     facts_text=json.dumps(facts, ensure_ascii=False))
                if issues:
                    status, feedback, rejected = "fallback_invalid", issues, issues
                    if attempt:
                        break
                    continue
                checked = await review_copy(check_template, phase="message", facts=facts,
                                            references=references, message=candidate)
                if accepted_review(checked, len(references)):
                    message = candidate
                    status = "verified" if references else "verified_no_evidence"
                    issues = []
                    break
                status = "fallback_invalid"
                found = bounded_issues(checked.get("unsupported_claims") if isinstance(checked, dict) else None,
                                       limit=5, length=200)
                issues = rejected = found or ["核验未通过且未给出原因"]
                if attempt or not found:
                    break
                feedback = found
    except TimeoutError:
        # A repair cut off by the deadline: the first draft was still rejected.
        if rejected:
            status, issues = "fallback_invalid", rejected
        else:
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
        recommendation_message_issues=(mask_quotes(bounded_issues(issues), private_quotes)
                                       if status.startswith("fallback") else []),
        recommendation_message_generated_at=datetime.now(UTC).isoformat(),
        user_relevance=references,
        user_relevance_count=len(references),
        recommendation_relevance_status=relevance_status,
        recommendation_evidence_status=source_statuses or {kind: "not_collected" for kind in EVIDENCE_KINDS},
    )
    logger.info("[offline] recommendation message=%s references=%d", status, len(references))
    return card
