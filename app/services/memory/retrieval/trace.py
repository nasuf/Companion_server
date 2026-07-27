"""Per-chat memory retrieval trace collection.

The LangSmith trace tree only sees LLM calls. Memory retrieval is local Python
and SQL work, so we collect a compact structured summary in a ContextVar and
persist it on the first assistant message metadata. The web Trace page can then
render it next to the LLM trace without adding a new hot-path table.
"""

from __future__ import annotations

import copy
import re
import uuid
from contextvars import ContextVar, Token
from datetime import datetime
from typing import Any


_current_sessions: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "memory_retrieval_trace_sessions",
    default=None,
)

_MAX_CANDIDATES_PER_SESSION = 20
_MAX_TEXT_PREVIEW = 160
_MAX_ANALYSIS_ITEMS = 12
_MAX_MATCH_TERMS = 8
_MAX_FEEDBACK_MEMORY_IDS = 10
_CJK_STOP_TERMS = {
    "用户", "自己", "这个", "那个", "这些", "那些", "事情", "相关",
    "之前", "最近", "现在", "已经", "没有", "不是", "一个", "时候",
    "表达", "告诉", "觉得", "知道", "可以", "需要", "应该", "可能",
}
_MISTAKE_VERBS = "记错|记岔|搞错|弄错|理解错|误会|说错|说反|弄反|搞反"
_CORRECTION_REGEXES: tuple[tuple[str, str, float], ...] = (
    (
        "记错",
        rf"(?:你|ai|它)[^，。！？!?]{{0,8}}(?:{_MISTAKE_VERBS})"
        rf"|(?<!我)(?:{_MISTAKE_VERBS})(?:了|啦|啊|呀|，|,|。|！|!|$)",
        0.88,
    ),
    (
        "没说过",
        r"(?:我)?(?:没|没有)说过|(?:我)?(?:没|没有)说这个|没这回事|别乱说|不要乱说|别瞎说",
        0.86,
    ),
    (
        "直接否定",
        r"^(?:不是|不对|错了|不是这样|不是这个|不是那样)(?:啊|呀|啦|哦|噢|，|,|。|\.|！|!|\s|$)",
        0.72,
    ),
    (
        "泛否定",
        r"^(?:哪有)(?:啊|呀|啦|嘛|，|,|。|\.|！|!|\s|$)",
        0.68,
    ),
    ("澄清", r"不是这个意思|我说的是|我是说|我的意思是", 0.78),
)
_AMBIGUOUS_CORRECTION_LABELS = {"泛否定"}
_FEEDBACK_CONTEXT_TERMS = ("你", "记", "记忆", "记得", "说", "刚刚", "刚才", "回复", "上条")
_AMBIGUOUS_FEEDBACK_CONTEXT_TERMS = ("记", "记忆", "记得", "说", "回复", "上条")


def start_retrieval_trace() -> Token:
    """Start collecting retrieval sessions for the current chat request."""
    return _current_sessions.set([])


def reset_retrieval_trace(token: Token) -> None:
    _current_sessions.reset(token)


def snapshot_retrieval_traces() -> list[dict[str, Any]]:
    sessions = _current_sessions.get() or []
    return copy.deepcopy(sessions)


def build_retrieval_quality_analysis(
    retrievals: list[dict[str, Any]] | None,
    *,
    assistant_reply: str = "",
    user_message: str = "",
) -> dict[str, Any] | None:
    """Build a compact, deterministic quality report for the Trace modal.

    This is intentionally heuristic. It does not claim that the LLM truly
    "used" a memory; it only marks `likely_used` when the final reply contains
    visible lexical anchors from an injected memory. The goal is fast debugging
    and trend analysis without another hot-path LLM call.
    """
    if not retrievals:
        return None

    sessions = [s for s in retrievals if isinstance(s, dict)]
    if not sessions:
        return None

    selected_items: list[dict[str, Any]] = []
    candidate_count = 0
    raw_count = 0
    superseded_count = 0
    final_gate_dropped_candidates = 0
    signal_counts = {
        "keyword": 0,
        "entity": 0,
        "safety": 0,
        "time": 0,
        "l3": 0,
        "enhanced_query": 0,
        "cache_hit": 0,
        "semantic_conflict": 0,
    }
    semantic_conflict_keys: set[str] = set()

    for session in sessions:
        notes = session.get("notes") if isinstance(session.get("notes"), dict) else {}
        selected = session.get("selected") if isinstance(session.get("selected"), list) else []
        candidates = session.get("candidates") if isinstance(session.get("candidates"), list) else []
        candidate_count += int(session.get("candidate_count") or len(candidates) or 0)
        raw_count += int(session.get("raw_count") or 0)
        if notes.get("superseded_by_later_retrieval"):
            superseded_count += 1
        if session.get("cache_hit"):
            signal_counts["cache_hit"] += 1
        if session.get("enhanced_query"):
            signal_counts["enhanced_query"] += 1
        for candidate in candidates:
            if isinstance(candidate, dict) and _has_semantic_conflict_reason(candidate):
                key = _memory_signal_key(candidate)
                if key not in semantic_conflict_keys:
                    semantic_conflict_keys.add(key)
                    signal_counts["semantic_conflict"] += 1
        strategy = str(session.get("strategy") or "")
        if (
            notes.get("final_injected") is False
            and not notes.get("superseded_by_later_retrieval")
            and candidates
        ):
            final_gate_dropped_candidates += len(candidates)
        for item in selected:
            if isinstance(item, dict):
                row = dict(item)
                row["_session_id"] = session.get("session_id")
                row["_strategy"] = strategy
                selected_items.append(row)

    user_selected = sum(1 for item in selected_items if item.get("source") == "user")
    ai_selected = sum(1 for item in selected_items if item.get("source") == "ai")
    likely_used_count = 0
    analyzed_items: list[dict[str, Any]] = []

    for item in selected_items:
        reasons = [str(r) for r in (item.get("rank_reasons") or [])]
        text = str(item.get("text") or "")
        retrieval_source = str(item.get("retrieval_source") or "")
        strategy = str(item.get("_strategy") or "")
        matched = _matched_terms(text, assistant_reply)
        likely_used = bool(matched)
        if likely_used:
            likely_used_count += 1
        if any("关键词" in reason for reason in reasons):
            signal_counts["keyword"] += 1
        if any("实体" in reason for reason in reasons) or "entity" in retrieval_source:
            signal_counts["entity"] += 1
        if any("安全" in reason or "情绪" in reason for reason in reasons):
            signal_counts["safety"] += 1
        if _has_semantic_conflict_reason(item):
            key = _memory_signal_key(item)
            if key not in semantic_conflict_keys:
                semantic_conflict_keys.add(key)
                signal_counts["semantic_conflict"] += 1
        if strategy == "explicit_time":
            signal_counts["time"] += 1
        if strategy == "l3_awaken":
            signal_counts["l3"] += 1

        if len(analyzed_items) < _MAX_ANALYSIS_ITEMS:
            analyzed_items.append({
                "id": item.get("id") or "",
                "session_id": item.get("_session_id"),
                "strategy": strategy,
                "source": item.get("source") or "",
                "score": item.get("score"),
                "text": _text_preview(text),
                "rank_reasons": reasons,
                "likely_used": likely_used,
                "matched_terms": matched,
            })

    selected_count = len(selected_items)
    likely_unused_count = max(0, selected_count - likely_used_count)
    visible_use_rate = likely_used_count / selected_count if selected_count else 0.0
    user_memory_share = user_selected / selected_count if selected_count else 0.0
    selection_rate = selected_count / candidate_count if candidate_count else 0.0
    observations: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if signal_counts["enhanced_query"]:
        observations.append(_notice(
            "enhanced_query",
            f"{signal_counts['enhanced_query']} 次召回使用了 enhanced_query。",
        ))
    if signal_counts["entity"]:
        observations.append(_notice("entity_recall", "本轮有实体命中信号参与召回。"))
    if signal_counts["keyword"]:
        observations.append(_notice("keyword_recall", "本轮有关键词命中信号参与重排。"))
    if signal_counts["semantic_conflict"]:
        observations.append(_notice(
            "semantic_conflict",
            f"{signal_counts['semantic_conflict']} 条候选被语义对立规则降权。",
        ))
    if signal_counts["safety"]:
        observations.append(_notice("safety_memory", "本轮注入了安全/情绪相关记忆。"))
    if signal_counts["time"]:
        observations.append(_notice("time_recall", "本轮包含显式时间记忆召回。"))
    if signal_counts["l3"]:
        observations.append(_notice("l3_recall", "本轮包含 L3 久远记忆召回。"))
    if superseded_count:
        observations.append(_notice(
            "superseded_retrieval",
            f"{superseded_count} 次早期召回被后续召回结果替换。",
        ))

    if selected_count == 0 and candidate_count > 0:
        warnings.append(_notice(
            "candidates_not_injected",
            "有候选记忆但最终没有注入 prompt，需检查 relevance gate 或 token/quota。",
            "warning",
        ))
    has_final_gate_drop = bool(final_gate_dropped_candidates)
    if has_final_gate_drop:
        warnings.append(_notice(
            "final_gate_dropped_candidates",
            f"{final_gate_dropped_candidates} 条候选没有进入最终 prompt。",
            "warning",
        ))
    has_prompt_dilution = False
    if selected_count > 0 and likely_used_count == 0:
        has_prompt_dilution = True
        warnings.append(_notice(
            "no_visible_memory_use",
            "回复文本里没有看到明显引用已注入记忆的词面线索。",
            "warning",
        ))
    elif selected_count >= 5 and likely_unused_count / selected_count >= 0.7:
        has_prompt_dilution = True
        warnings.append(_notice(
            "many_injected_not_visible",
            "注入记忆较多，但大部分没有在回复文本中出现明显引用线索，可能存在 prompt 稀释。",
            "warning",
        ))
    if selected_count > 0 and user_selected == 0 and ai_selected > 0:
        warnings.append(_notice(
            "no_user_memory_selected",
            "本轮只注入了 AI 侧记忆，没有用户侧记忆。",
            "warning",
        ))

    return {
        "version": 1,
        "method": "lexical_overlap_v1",
        "user_message_preview": _text_preview(user_message),
        "reply_preview": _text_preview(assistant_reply),
        "session_count": len(sessions),
        "raw_count": raw_count,
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "selected_user_count": user_selected,
        "selected_ai_count": ai_selected,
        "likely_used_count": likely_used_count,
        "likely_unused_count": likely_unused_count,
        "signal_counts": signal_counts,
        "quality_metrics": {
            "visible_use_rate": round(visible_use_rate, 4),
            "user_memory_share": round(user_memory_share, 4),
            "selection_rate": round(selection_rate, 4),
            "warning_count": len(warnings),
            "has_final_gate_drop": has_final_gate_drop,
            "has_prompt_dilution": has_prompt_dilution,
        },
        "observations": observations,
        "warnings": warnings,
        "items": analyzed_items,
    }


def build_memory_retrieval_feedback(
    *,
    user_message: str,
    previous_assistant_reply: str,
    previous_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Detect whether the next user turn may be correcting memory usage.

    This is an observability signal, not a memory mutation. It only fires when
    the previous assistant message actually had retrieval metadata and the next
    user turn contains a correction-like phrase. The frontend can then surface
    it on the previous reply's Trace modal for retrieval quality analysis.
    """
    metadata = previous_metadata if isinstance(previous_metadata, dict) else {}
    if not metadata:
        return None
    analysis = metadata.get("memory_retrieval_analysis")
    retrievals = metadata.get("memory_retrievals")
    if not isinstance(analysis, dict) and not isinstance(retrievals, list):
        return None

    text = str(user_message or "").strip()
    if not text:
        return None

    matched_phrases: list[str] = []
    confidence = 0.0
    for label, pattern, score in _CORRECTION_REGEXES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched_phrases.append(label)
            confidence = max(confidence, score)

    if not matched_phrases:
        return None

    has_feedback_context = any(term in text for term in _FEEDBACK_CONTEXT_TERMS)
    if (
        set(matched_phrases).issubset(_AMBIGUOUS_CORRECTION_LABELS)
        and not any(term in text for term in _AMBIGUOUS_FEEDBACK_CONTEXT_TERMS)
    ):
        return None

    if has_feedback_context:
        confidence += 0.08
    if isinstance(analysis, dict) and (_safe_int(analysis.get("likely_used_count")) or 0) > 0:
        confidence += 0.06
    confidence = round(min(confidence, 0.98), 2)

    memory_ids = _feedback_memory_ids(metadata)
    notes = [
        "下一轮用户可能在纠正本次回复中的记忆使用。",
        "这是关键词启发式信号，不代表系统已确认哪条记忆错误。",
    ]
    if memory_ids:
        notes.append("memory_ids 优先来自本次回复可能使用或最终注入的记忆。")

    return {
        "version": 1,
        "method": "correction_keyword_v1",
        "signal": "potential_memory_correction",
        "confidence": confidence,
        "user_message_preview": _text_preview(text),
        "assistant_reply_preview": _text_preview(previous_assistant_reply),
        "matched_phrases": matched_phrases,
        "memory_ids": memory_ids,
        "notes": notes,
    }


def make_retrieval_session_id(prefix: str = "ret") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value)
    return text if text else None


def _text_preview(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= _MAX_TEXT_PREVIEW:
        return text
    return text[:_MAX_TEXT_PREVIEW] + "..."


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _cjk_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(segment) <= 4 and segment not in _CJK_STOP_TERMS:
            terms.add(segment)
        for size in (2, 3, 4):
            if len(segment) < size:
                continue
            for idx in range(0, len(segment) - size + 1):
                term = segment[idx:idx + size]
                if term not in _CJK_STOP_TERMS and "用户" not in term:
                    terms.add(term)
    return terms


def _lexical_terms(text: str) -> set[str]:
    normalized = _normalize_text(text)
    terms = _cjk_terms(normalized)
    terms.update(re.findall(r"[a-z0-9_]{3,}", normalized))
    return terms


def _matched_terms(memory_text: str, assistant_reply: str) -> list[str]:
    reply_text = _normalize_text(assistant_reply)
    if not reply_text:
        return []
    matches = [
        term for term in _lexical_terms(memory_text)
        if term and term in reply_text
    ]
    # Prefer longer terms first; they are more informative than generic bigrams.
    matches.sort(key=lambda term: (-len(term), term))
    return matches[:_MAX_MATCH_TERMS]


def _has_semantic_conflict_reason(item: dict[str, Any]) -> bool:
    reasons = item.get("rank_reasons")
    if not isinstance(reasons, list):
        return False
    return any("语义对立" in str(reason) for reason in reasons)


def _memory_signal_key(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("text") or item.get("content") or "")


def _notice(code: str, message: str, severity: str = "info") -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def _feedback_memory_ids(metadata: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    analysis = metadata.get("memory_retrieval_analysis")
    if isinstance(analysis, dict):
        items = analysis.get("items")
        if isinstance(items, list):
            likely_used = [
                item for item in items
                if isinstance(item, dict) and item.get("likely_used")
            ]
            for item in likely_used or items:
                if isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))

    retrievals = metadata.get("memory_retrievals")
    if isinstance(retrievals, list):
        for session in retrievals:
            if not isinstance(session, dict):
                continue
            selected = session.get("selected")
            if not isinstance(selected, list):
                continue
            for item in selected:
                if isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))

    unique_ids: list[str] = []
    for mid in ids:
        if mid and mid not in unique_ids:
            unique_ids.append(mid)
        if len(unique_ids) >= _MAX_FEEDBACK_MEMORY_IDS:
            break
    return unique_ids


def memory_trace_item(memory: Any, *, selected: bool = False) -> dict[str, Any]:
    """Serialize a dict or ClassifiedMemory-like object for trace display."""
    if isinstance(memory, str):
        return {
            "id": "",
            "source": "",
            "level": None,
            "text": _text_preview(memory),
            "importance": None,
            "similarity": None,
            "score": None,
            "rank_reasons": [],
            "retrieval_source": None,
            "main_category": None,
            "sub_category": None,
            "last_accessed_at": None,
            "selected": selected,
        }

    if isinstance(memory, dict):
        text = memory.get("content") or memory.get("text") or ""
        return {
            "id": memory.get("id") or "",
            "source": memory.get("source") or "",
            "level": _safe_int(memory.get("level")),
            "text": _text_preview(text),
            "importance": _safe_float(memory.get("importance")),
            "similarity": _safe_float(memory.get("similarity")),
            "score": _safe_float(
                memory.get("rank_score")
                if memory.get("rank_score") is not None
                else memory.get("score")
            ),
            "rank_reasons": list(memory.get("rank_reasons") or []),
            "retrieval_source": memory.get("_retrieval_source") or memory.get("retrieval_source"),
            "main_category": memory.get("main_category"),
            "sub_category": memory.get("sub_category"),
            "last_accessed_at": _safe_iso(
                memory.get("last_accessed_at")
                or memory.get("updated_at")
                or memory.get("created_at")
            ),
            "selected": selected,
        }

    text = getattr(memory, "text", "") or getattr(memory, "content", "") or ""
    return {
        "id": getattr(memory, "id", "") or "",
        "source": getattr(memory, "source", "") or "",
        "level": _safe_int(getattr(memory, "level", None)),
        "text": _text_preview(text),
        "importance": _safe_float(getattr(memory, "importance", None)),
        "similarity": _safe_float(getattr(memory, "similarity", None)),
        "score": _safe_float(
            getattr(memory, "display_score", None)
            or getattr(memory, "score", None)
        ),
        "rank_reasons": list(getattr(memory, "rank_reasons", None) or []),
        "retrieval_source": getattr(memory, "retrieval_source", None),
        "main_category": getattr(memory, "main_category", None),
        "sub_category": getattr(memory, "sub_category", None),
        "last_accessed_at": _safe_iso(
            getattr(memory, "last_accessed_at", None)
            or getattr(memory, "created_at", None)
        ),
        "selected": selected,
    }


def _selected_id_set(selected: list[Any]) -> set[str]:
    ids = {
        str(item.get("id"))
        for item in selected
        if isinstance(item, dict) and item.get("id")
    }
    ids.update(
        str(getattr(item, "id", ""))
        for item in selected
        if getattr(item, "id", "")
    )
    return {mid for mid in ids if mid}


def _selected_trace_items(selected: list[Any]) -> list[dict[str, Any]]:
    return [memory_trace_item(memory, selected=True) for memory in selected]


def _mark_candidate_selection(
    candidates: list[dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        mid = str(item.get("id") or "")
        item["selected"] = bool(mid and mid in selected_ids)
        marked.append(item)
    return marked


def replace_latest_retrieval_selection(
    *,
    strategy: str,
    selected: list[Any] | None,
    final_injected: bool,
) -> None:
    """Align trace selection with memories that actually reached the prompt.

    Retrieval can run before the relevance gate finishes. If relevance later
    returns weak, or an enhanced-query retry supersedes the first retrieval,
    the earlier selected list is not truly prompt-injected. This function marks
    older sessions as superseded and rewrites the latest session's selected
    list to match the final prompt input.
    """
    sessions = _current_sessions.get()
    if sessions is None:
        return
    matching = [s for s in sessions if s.get("strategy") == strategy]
    if not matching:
        return

    selected_list = selected or []
    selected_items = _selected_trace_items(selected_list)
    selected_ids = _selected_id_set(selected_list)
    latest = matching[-1]
    for session in matching:
        notes = dict(session.get("notes") or {})
        if session is latest:
            notes["final_injected"] = bool(final_injected and selected_items)
            session["selected"] = selected_items
            session["selected_count"] = len(selected_items)
            candidates = session.get("candidates")
            if isinstance(candidates, list):
                session["candidates"] = _mark_candidate_selection(candidates, selected_ids)
        else:
            notes["final_injected"] = False
            notes["superseded_by_later_retrieval"] = True
            session["selected"] = []
            session["selected_count"] = 0
            candidates = session.get("candidates")
            if isinstance(candidates, list):
                session["candidates"] = _mark_candidate_selection(candidates, set())
        session["notes"] = notes


def record_retrieval_session(
    *,
    strategy: str,
    query: str,
    workspace_id: str | None = None,
    enhanced_query: str | None = None,
    memory_relevance: str | None = None,
    trigger_label: str | None = None,
    cache_hit: bool = False,
    raw_count: int | None = None,
    candidate_count: int | None = None,
    selected_count: int | None = None,
    candidates: list[Any] | None = None,
    selected: list[Any] | None = None,
    notes: dict[str, Any] | None = None,
) -> str | None:
    """Append one retrieval session if collection is active."""
    sessions = _current_sessions.get()
    if sessions is None:
        return None

    selected_ids = _selected_id_set(selected or [])

    candidate_items = []
    for memory in (candidates or [])[:_MAX_CANDIDATES_PER_SESSION]:
        mid = memory.get("id") if isinstance(memory, dict) else getattr(memory, "id", "")
        candidate_items.append(memory_trace_item(memory, selected=bool(mid and mid in selected_ids)))

    selected_items = _selected_trace_items(selected or [])
    session_id = make_retrieval_session_id(strategy)
    sessions.append({
        "session_id": session_id,
        "strategy": strategy,
        "query": _text_preview(query),
        "enhanced_query": _text_preview(enhanced_query) if enhanced_query else None,
        "workspace_id": workspace_id,
        "memory_relevance": memory_relevance,
        "trigger_label": trigger_label,
        "cache_hit": cache_hit,
        "raw_count": raw_count,
        "candidate_count": candidate_count,
        "selected_count": selected_count if selected_count is not None else len(selected_items),
        "candidates": candidate_items,
        "selected": selected_items,
        "notes": notes or {},
    })
    return session_id
