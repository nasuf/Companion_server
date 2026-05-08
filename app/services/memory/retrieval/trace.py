"""Per-chat memory retrieval trace collection.

The LangSmith trace tree only sees LLM calls. Memory retrieval is local Python
and SQL work, so we collect a compact structured summary in a ContextVar and
persist it on the first assistant message metadata. The web Trace page can then
render it next to the LLM trace without adding a new hot-path table.
"""

from __future__ import annotations

import copy
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


def start_retrieval_trace() -> Token:
    """Start collecting retrieval sessions for the current chat request."""
    return _current_sessions.set([])


def reset_retrieval_trace(token: Token) -> None:
    _current_sessions.reset(token)


def snapshot_retrieval_traces() -> list[dict[str, Any]]:
    sessions = _current_sessions.get() or []
    return copy.deepcopy(sessions)


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
        text = memory.get("summary") or memory.get("content") or memory.get("text") or ""
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

    text = getattr(memory, "text", "") or getattr(memory, "summary", "") or ""
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
