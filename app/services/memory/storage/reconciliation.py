"""Memory write reconciliation.

This layer decides whether a newly extracted memory should be inserted,
dropped, or used to update an existing memory.  It deliberately treats
taxonomy as a soft signal: source/workspace are hard boundaries, the top-level
category is the primary candidate pool, and sub-category only helps ranking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from app.services.memory.config import DEDUP_THRESHOLD
from app.services.memory.polarity import semantic_conflict_reasons
from app.services.memory.retrieval.vector_search import search_by_embedding
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.repo import MemoryRecord, Source
from app.services.memory.taxonomy import is_singleton

logger = logging.getLogger(__name__)


def _is_write_protected(record: MemoryRecord) -> bool:
    """Rows that write-time reconciliation must never update/merge-mutate.

    - Singleton L1 facts (姓名/年龄/生日 …): a singleton's core value changing
      is by definition a contradiction — that goes through the spec §4
      user-confirmation flow, not a silent overwrite.
    - profile_seed rows (any category): persona ground truth seeded at agent
      provisioning. Chat-time enrichments are stored separately instead of
      rewriting the seed, so the persona can never drift via reconciliation.
    - knowledge_seed rows: admin-published template knowledge (公司/产品/活动
      facts). Only the admin append/sync pipeline may manage them; chat-time
      writes must never merge into or rewrite them.

    Containment-gated enrichment stays allowed for non-seed, non-singleton L1
    (richer restatements of learned facts).
    """
    if getattr(record, "provenance", None) in ("profile_seed", "knowledge_seed"):
        return True
    return record.level == 1 and is_singleton(record.mainCategory, record.subCategory)


# Backward-compatible alias (tests/callers may reference the older name).
_is_protected_singleton_l1 = _is_write_protected

ReconciliationAction = Literal[
    "insert_new",
    "drop_duplicate",
    "update_existing",
    "merge_existing",
    "keep_separate",
    "needs_confirmation",
]


@dataclass
class ReconciliationDecision:
    action: ReconciliationAction
    existing_id: str | None = None
    existing_record: MemoryRecord | None = None
    reason: str = ""
    merged_content: str | None = None


_PUNCT_RE = re.compile(r"[\s，。！？?~～,.、:：；;“”\"'‘’（）()【】\[\]《》<>·\-—_]+")
_GENERIC_RE = re.compile(
    r"(用户|自己|本人|我的|我|AI|ai|喜欢|不喜欢|讨厌|觉得|认为|已经|现在|目前|曾经|"
    r"一只|一个|一种|这个|那个|这件事|那件事|比较|特别|非常|很|会|是|的|了|在|有)"
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _text_of(record: MemoryRecord) -> str:
    return record.content or ""


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    text = _PUNCT_RE.sub("", text)
    # Owner words are phrasing artifacts. Removing them lets "我养了芝麻" match
    # older memories like "养了一只叫芝麻的黑猫".
    text = re.sub(r"^(用户|我|我的|AI|ai|自己|本人)", "", text)
    return text


def _core(text: str | None) -> str:
    return _GENERIC_RE.sub("", _normalize(text))


def _ngrams(text: str, n: int = 2) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _signals(text: str | None, entities: list[str] | None, topics: list[str] | None) -> set[str]:
    signals: set[str] = set()
    for value in (entities or []) + (topics or []):
        value = _core(str(value))
        if len(value) >= 2:
            signals.add(value)
            signals.update(_ngrams(value))

    core = _core(text)
    for span in _CJK_RE.findall(core):
        if len(span) >= 2:
            signals.update(_ngrams(span))
        if len(span) >= 3:
            signals.add(span)
    return {s for s in signals if len(s) >= 2}


def _record_from_vector(row: dict, source: Source) -> MemoryRecord:
    return MemoryRecord(
        id=str(row.get("id")),
        userId=str(row.get("user_id") or row.get("userId") or ""),
        type=row.get("type"),
        source=source,
        level=int(row.get("level") or 3),
        content=str(row.get("content") or ""),
        importance=float(row.get("importance") or 0),
        mentionCount=int(row.get("mention_count") or row.get("mentionCount") or 0),
        isArchived=False,
        occurTime=row.get("occur_time") or row.get("occurTime"),
        createdAt=row.get("created_at") or row.get("createdAt"),
        updatedAt=row.get("updated_at") or row.get("updatedAt"),
        mainCategory=row.get("main_category") or row.get("mainCategory"),
        subCategory=row.get("sub_category") or row.get("subCategory"),
        workspaceId=row.get("workspace_id") or row.get("workspaceId"),
        provenance=row.get("provenance"),
    )


def _relation(
    *,
    new_text: str,
    old_text: str,
    new_entities: list[str] | None,
    new_topics: list[str] | None,
    old_main: str | None,
    new_main: str | None,
    vector_similarity: float | None,
) -> ReconciliationAction:
    if semantic_conflict_reasons(new_text, old_text):
        return "needs_confirmation"

    new_norm = _normalize(new_text)
    old_norm = _normalize(old_text)
    if not new_norm or not old_norm:
        return "keep_separate"

    same_main = bool(new_main and old_main and new_main == old_main)
    if not same_main:
        return "keep_separate"

    if new_norm == old_norm:
        return "drop_duplicate"

    # High-confidence textual coverage. This catches recall echoes where a
    # reply rephrases a subset of an injected memory.
    if len(new_norm) >= 6 and new_norm in old_norm:
        return "drop_duplicate"
    if len(old_norm) >= 6 and old_norm in new_norm:
        return "update_existing"

    new_signals = _signals(new_text, new_entities, new_topics)
    old_signals = _signals(old_text, None, None)
    if not new_signals or not old_signals:
        return "keep_separate"

    shared = new_signals & old_signals
    old_covered = len(shared) / max(1, len(old_signals))
    new_covered = len(shared) / max(1, len(new_signals))

    # Conservative structured coverage. We only update when the old memory's
    # discriminative terms are mostly present in a materially richer new text.
    if old_covered >= 0.70 and len(new_norm) > len(old_norm) * 1.20:
        return "update_existing"
    if new_covered >= 0.80 and len(old_norm) >= len(new_norm):
        return "drop_duplicate"

    if vector_similarity is not None and vector_similarity > DEDUP_THRESHOLD:
        return "drop_duplicate"
    return "keep_separate"


def _related_enough_for_llm(
    *,
    new_text: str,
    old_text: str,
    new_entities: list[str] | None,
    new_topics: list[str] | None,
    old_main: str | None,
    new_main: str | None,
    vector_similarity: float | None,
) -> bool:
    """Whether a candidate is worth an expensive adjudication call."""
    if not (new_main and old_main and new_main == old_main):
        return False
    if vector_similarity is not None and vector_similarity >= 0.78:
        return True
    new_signals = _signals(new_text, new_entities, new_topics)
    old_signals = _signals(old_text, None, None)
    if not new_signals or not old_signals:
        return False
    shared = new_signals & old_signals
    if not shared:
        return False
    new_covered = len(shared) / max(1, len(new_signals))
    old_covered = len(shared) / max(1, len(old_signals))
    return max(new_covered, old_covered) >= 0.45


def _safe_action(value: str | None) -> ReconciliationAction:
    if value in {
        "insert_new",
        "drop_duplicate",
        "update_existing",
        "merge_existing",
        "keep_separate",
        "needs_confirmation",
    }:
        return value  # type: ignore[return-value]
    return "keep_separate"


async def _llm_adjudicate(
    *,
    source: Source,
    new_text: str,
    old_text: str,
    new_main: str | None,
    new_sub: str | None,
    old_main: str | None,
    old_sub: str | None,
) -> ReconciliationDecision | None:
    """Ask the chat model only for related-but-ambiguous memory pairs.

    A small model is good for coarse filtering; this is a semantic write
    decision that can delete or mutate memory, so we use the chat model and
    keep the invocation narrow.
    """
    try:
        from app.services.llm.models import get_chat_model, invoke_json
        from app.services.prompting.store import get_prompt_text

        prompt = (await get_prompt_text("memory.reconciliation")).format(
            source=source,
            old_main=old_main or "",
            old_sub=old_sub or "",
            old_text=old_text,
            new_main=new_main or "",
            new_sub=new_sub or "",
            new_text=new_text,
        )
        result = await invoke_json(get_chat_model(), prompt, profile="chat_extract")
    except Exception as e:
        logger.warning(f"Memory reconciliation LLM adjudication failed: {e}")
        return None

    if not isinstance(result, dict):
        return None
    action = _safe_action(str(result.get("action") or ""))
    if action in {"insert_new", "keep_separate", "needs_confirmation"}:
        return ReconciliationDecision(action=action, reason=str(result.get("reason") or "llm"))
    # The `memory.reconciliation` prompt names its merged-text field
    # `merged_summary`; the stored column is `content`.
    merged_text = result.get("merged_summary")
    if not isinstance(merged_text, str) or not merged_text.strip():
        merged_text = new_text if action == "update_existing" else None
    return ReconciliationDecision(
        action=action,
        reason=str(result.get("reason") or "llm"),
        merged_content=merged_text,
    )


async def _category_candidates(
    *,
    user_id: str,
    source: Source,
    workspace_id: str | None,
    main_category: str | None,
) -> list[MemoryRecord]:
    where = {
        "userId": user_id,
        "workspaceId": workspace_id,
        "isArchived": False,
    }
    if main_category:
        where["mainCategory"] = main_category
    return await memory_repo.find_many(
        source=source,
        where=where,
        order={"updatedAt": "desc"},
        take=50,
    )


async def resolve_memory_write(
    *,
    user_id: str,
    source: Source,
    workspace_id: str | None,
    content: str,
    embedding: list[float],
    main_category: str | None,
    sub_category: str | None,
    entities: list[str] | None = None,
    topics: list[str] | None = None,
    exclude_id: str | None = None,
    allow_llm: bool = True,
) -> ReconciliationDecision:
    """Resolve the write action for a newly extracted memory.

    The function is intentionally high-precision. Ambiguous related facts are
    kept separate so the system loses less information; later offline hygiene
    can merge low-confidence cases.
    """
    text = content
    candidates: dict[str, tuple[MemoryRecord, float | None]] = {}

    try:
        for record in await _category_candidates(
            user_id=user_id,
            source=source,
            workspace_id=workspace_id,
            main_category=main_category,
        ):
            if record.id != exclude_id:
                candidates[record.id] = (record, None)
    except Exception as e:
        logger.warning(f"Memory reconciliation category candidates failed: {e}")

    try:
        vector_rows = await search_by_embedding(
            embedding,
            user_id,
            top_k=10,
            workspace_id=workspace_id,
            main_categories=[main_category] if main_category else None,
        )
        for row in vector_rows:
            row_source = row.get("source")
            if row_source != source:
                continue
            mid = str(row.get("id") or "")
            if not mid or mid == exclude_id:
                continue
            sim = row.get("similarity")
            if isinstance(sim, str):
                sim = float(sim)
            if mid in candidates:
                candidates[mid] = (candidates[mid][0], float(sim))
            else:
                candidates[mid] = (_record_from_vector(row, source), float(sim))
    except Exception as e:
        logger.warning(f"Memory reconciliation vector candidates failed: {e}")

    best_update: ReconciliationDecision | None = None
    ambiguous_for_llm: list[tuple[MemoryRecord, float | None]] = []
    for record, sim in candidates.values():
        old_text = _text_of(record)
        action = _relation(
            new_text=text,
            old_text=old_text,
            new_entities=entities,
            new_topics=topics,
            old_main=record.mainCategory,
            new_main=main_category,
            vector_similarity=sim,
        )
        if action == "needs_confirmation":
            # Current contradiction flow is specialized for explicit L1
            # user-message conflicts. Do not silently overwrite; keep both.
            logger.info(
                f"Memory reconciliation conflict candidate kept separate: "
                f"new='{text[:40]}' existing='{old_text[:40]}'"
            )
            continue
        if action == "drop_duplicate":
            return ReconciliationDecision(
                action="drop_duplicate",
                existing_id=record.id,
                existing_record=record,
                reason="existing_covers_new",
            )
        if action == "update_existing":
            if _is_write_protected(record):
                logger.info(
                    f"Memory reconciliation refused to update write-protected row "
                    f"(prov={getattr(record, 'provenance', None)}, L{record.level}, "
                    f"{record.mainCategory}/{record.subCategory}); keeping separate: "
                    f"new='{text[:40]}'"
                )
                continue
            best_update = ReconciliationDecision(
                action="update_existing",
                existing_id=record.id,
                existing_record=record,
                reason="new_covers_existing",
                merged_content=content,
            )
            continue
        if allow_llm and _related_enough_for_llm(
            new_text=text,
            old_text=old_text,
            new_entities=entities,
            new_topics=topics,
            old_main=record.mainCategory,
            new_main=main_category,
            vector_similarity=sim,
        ):
            ambiguous_for_llm.append((record, sim))

    if best_update is not None:
        return best_update

    ambiguous_for_llm.sort(key=lambda item: item[1] or 0, reverse=True)
    for record, _sim in ambiguous_for_llm[:3]:
        old_text = _text_of(record)
        decision = await _llm_adjudicate(
            source=source,
            new_text=text,
            old_text=old_text,
            new_main=main_category,
            new_sub=sub_category,
            old_main=record.mainCategory,
            old_sub=record.subCategory,
        )
        if decision is None:
            continue
        if decision.action in {"drop_duplicate", "update_existing", "merge_existing"}:
            if (
                decision.action in {"update_existing", "merge_existing"}
                and _is_write_protected(record)
            ):
                logger.info(
                    f"Memory reconciliation LLM wanted to mutate write-protected row "
                    f"(prov={getattr(record, 'provenance', None)}, L{record.level}, "
                    f"{record.mainCategory}/{record.subCategory}); keeping separate: "
                    f"new='{text[:40]}'"
                )
                continue
            decision.existing_id = record.id
            decision.existing_record = record
            if decision.action == "drop_duplicate":
                return decision
            if not decision.merged_content:
                decision.merged_content = content
            return decision
        if decision.action == "needs_confirmation":
            logger.info(
                f"Memory reconciliation LLM found conflict; keeping separate: "
                f"new='{text[:40]}' existing='{old_text[:40]}'"
            )

    return best_update or ReconciliationDecision(action="insert_new", reason="no_related_existing")
