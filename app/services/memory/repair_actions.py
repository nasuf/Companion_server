"""Admin-reviewed repair actions for memory quality issues."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from app.services.memory.manual_write import MemoryWriteError
from app.services.memory.repair_write import OPERATIONS, apply_atomic_repair
from app.services.memory.storage import repo as memory_repo
from app.services.memory.manual_edit import MemoryEditError, edit_memory_with_evidence
from app.services.memory.repair_queue import (
    get_memory_repair_item,
)

RepairAction = Literal[
    "archive_memory",
    "downgrade_memory",
    "edit_memory",
    "insert_replacement_memory",
    "mark_verified",
    "merge_memories",
]


class MemoryRepairActionError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class MemoryRepairActionPayload:
    action: RepairAction
    memory_id: str | None = None
    memory_ids: list[str] = field(default_factory=list)
    source: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    content: str | None = None
    level: int | None = None
    importance: float | None = None
    memory_type: str | None = None
    main_category: str | None = None
    sub_category: str | None = None
    reason: str | None = None
    expected_updated_at: str | None = None


async def apply_memory_repair_action(item_id: str, *, payload: MemoryRepairActionPayload,
                                     admin_id: str | None) -> dict[str, Any]:
    repair_item = await get_memory_repair_item(item_id)
    if repair_item is None:
        raise MemoryRepairActionError(404, "memory_repair_item_not_found")
    if repair_item.get("status") != "open":
        raise MemoryRepairActionError(409, "memory_repair_item_is_not_open")
    if payload.expected_updated_at is not None and payload.expected_updated_at != repair_item.get("updated_at"):
        raise MemoryRepairActionError(409, "memory_repair_item_changed_refresh_and_retry")
    if payload.action not in {*OPERATIONS, "edit_memory"}:
        raise MemoryRepairActionError(400, "unsupported_memory_repair_action")
    if payload.level is not None and payload.level not in {1,2,3}:
        raise MemoryRepairActionError(400, "invalid_memory_level")
    if payload.importance is not None and (not math.isfinite(payload.importance) or not .1 <= payload.importance <= .99):
        raise MemoryRepairActionError(400, "invalid_memory_importance")
    if payload.action in {"edit_memory","merge_memories","insert_replacement_memory"}:
        content = _clean_text(payload.content)
        if not content:
            raise MemoryRepairActionError(400, f"{payload.action}_requires_content")
        _assert_within_injection_limit(content)
    if payload.action == "edit_memory":
        return await _edit_memory(repair_item,payload,admin_id)
    replacement = payload.action == "insert_replacement_memory"
    record = await _load_target_memory(repair_item,payload.memory_id,allow_archived=replacement) \
        if not replacement or payload.memory_id or repair_item.get("memory_id") else None
    try:
        return await apply_atomic_repair(repair_item=repair_item,payload=payload,admin_id=admin_id,record=record)
    except MemoryWriteError as exc:
        raise MemoryRepairActionError(exc.status_code,exc.detail) from exc


async def _edit_memory(
    repair_item: dict[str, Any],
    payload: MemoryRepairActionPayload,
    admin_id: str | None,
) -> dict[str, Any]:
    record = await _load_target_memory(repair_item, payload.memory_id)
    content = _clean_text(payload.content)
    if content is None:
        raise MemoryRepairActionError(400, "edit_memory_requires_content")
    _assert_within_injection_limit(content)

    try:
        _, resolved = await edit_memory_with_evidence(
            record=record, content=content, actor={"sub": admin_id, "role": "admin"},
            importance=_clamp_importance(payload.importance) if payload.importance is not None else None,
            level=payload.level, repair_item=repair_item,
            resolution_note=_resolution_note(payload), reason=payload.reason,
        )
    except MemoryEditError as exc:
        raise MemoryRepairActionError(exc.status_code, exc.detail) from exc
    return {"action": payload.action, "memory_id": record.id, "repair_item": resolved}


async def _load_target_memory(
    repair_item: dict[str, Any],
    requested_id: str | None,
    *,
    allow_archived: bool = False,
):
    repair_memory_id = repair_item.get("memory_id")
    memory_id = requested_id or repair_memory_id
    if not memory_id:
        raise MemoryRepairActionError(400, "memory_id_required")
    if repair_memory_id and requested_id and requested_id != repair_memory_id:
        raise MemoryRepairActionError(400, "memory_id_does_not_match_repair_item")
    record = await memory_repo.find_unique(memory_id)
    if not record or (record.isArchived and not allow_archived):
        raise MemoryRepairActionError(404, "memory_not_found")
    _assert_repair_scope(repair_item, record)
    return record


def _assert_repair_scope(repair_item: dict[str, Any], record) -> None:
    if repair_item.get("user_id") and repair_item["user_id"] != record.userId:
        raise MemoryRepairActionError(403, "memory_user_does_not_match_repair_item")
    if repair_item.get("workspace_id") and repair_item["workspace_id"] != record.workspaceId:
        raise MemoryRepairActionError(403, "memory_workspace_does_not_match_repair_item")
    if repair_item.get("memory_source") and repair_item["memory_source"] != record.source:
        raise MemoryRepairActionError(403, "memory_source_does_not_match_repair_item")


def _resolution_note(payload: MemoryRepairActionPayload) -> str:
    reason = _clean_text(payload.reason)
    if reason:
        return f"{payload.action}: {reason}"
    return payload.action


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _assert_within_injection_limit(content: str) -> None:
    """Reject repair edits/merges that would produce an unretrievable memory.

    reconciliation._llm_adjudicate already refuses to let the *LLM* merge two
    memories into one that exceeds the limit (it downgrades to keep_separate).
    The same act performed by hand through the repair tool had no such check —
    and this module exists specifically to *improve* memory quality, so it was
    the last place that should be able to mint a memory nothing can retrieve.
    """
    from app.services.memory.retrieval.context_selector import exceeds_injection_limit

    if exceeds_injection_limit(content):
        raise MemoryRepairActionError(400, "memory_content_too_long")


def _clamp_importance(value: float) -> float:
    return max(0.1, min(float(value), 0.99))
