"""Admin-reviewed repair actions for memory quality issues."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.storage.persistence import log_memory_changelog, store_memory
from app.services.memory.lifecycle.quality_state import mark_memory_superseded
from app.services.memory.repair_queue import (
    get_memory_repair_item,
    update_memory_repair_item_status,
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
    summary: str | None = None
    level: int | None = None
    importance: float | None = None
    memory_type: str | None = None
    main_category: str | None = None
    sub_category: str | None = None
    reason: str | None = None


async def apply_memory_repair_action(
    item_id: str,
    *,
    payload: MemoryRepairActionPayload,
    admin_id: str | None,
) -> dict[str, Any]:
    repair_item = await get_memory_repair_item(item_id)
    if repair_item is None:
        raise MemoryRepairActionError(404, "memory_repair_item_not_found")
    if repair_item.get("status") != "open":
        raise MemoryRepairActionError(409, "memory_repair_item_is_not_open")

    if payload.action == "archive_memory":
        result = await _archive_memory(repair_item, payload, admin_id)
    elif payload.action == "downgrade_memory":
        result = await _downgrade_memory(repair_item, payload, admin_id)
    elif payload.action == "edit_memory":
        result = await _edit_memory(repair_item, payload, admin_id)
    elif payload.action == "insert_replacement_memory":
        result = await _insert_replacement_memory(repair_item, payload, admin_id)
    elif payload.action == "mark_verified":
        result = await _mark_verified(repair_item, payload, admin_id)
    elif payload.action == "merge_memories":
        result = await _merge_memories(repair_item, payload, admin_id)
    else:
        raise MemoryRepairActionError(400, "unsupported_memory_repair_action")

    resolved = await update_memory_repair_item_status(
        item_id,
        status="resolved",
        resolution_note=_resolution_note(payload),
        resolved_by_id=admin_id,
    )
    result["repair_item"] = resolved
    return result


async def _archive_memory(
    repair_item: dict[str, Any],
    payload: MemoryRepairActionPayload,
    admin_id: str | None,
) -> dict[str, Any]:
    record = await _load_target_memory(repair_item, payload.memory_id)
    await memory_repo.update(record.id, source=record.source, record=record, isArchived=True)
    await _log_repair_changelog(
        record=record,
        operation="repair_archive",
        admin_id=admin_id,
        repair_item=repair_item,
        payload=payload,
        old_value=record.content,
        new_value=None,
    )
    return {"action": payload.action, "memory_id": record.id}


async def _downgrade_memory(
    repair_item: dict[str, Any],
    payload: MemoryRepairActionPayload,
    admin_id: str | None,
) -> dict[str, Any]:
    record = await _load_target_memory(repair_item, payload.memory_id)
    target_level = payload.level if payload.level is not None else 3
    if target_level not in {2, 3}:
        raise MemoryRepairActionError(400, "downgrade_level_must_be_2_or_3")
    if target_level < record.level:
        raise MemoryRepairActionError(400, "downgrade_cannot_increase_memory_level")

    await memory_repo.update(record.id, source=record.source, record=record, level=target_level)
    await _log_repair_changelog(
        record=record,
        operation="repair_downgrade",
        admin_id=admin_id,
        repair_item=repair_item,
        payload=payload,
        old_value=json.dumps({"level": record.level}, ensure_ascii=False),
        new_value=json.dumps({"level": target_level}, ensure_ascii=False),
    )
    return {"action": payload.action, "memory_id": record.id, "level": target_level}


async def _edit_memory(
    repair_item: dict[str, Any],
    payload: MemoryRepairActionPayload,
    admin_id: str | None,
) -> dict[str, Any]:
    record = await _load_target_memory(repair_item, payload.memory_id)
    content = _clean_text(payload.content)
    summary = _clean_text(payload.summary)
    if content is None and summary is None:
        raise MemoryRepairActionError(400, "edit_memory_requires_content_or_summary")

    update_data: dict[str, Any] = {}
    if content is not None:
        update_data["content"] = content
        update_data["summary"] = summary or content[:200]
        await store_embedding(record.id, await generate_embedding(content))
    elif summary is not None:
        update_data["summary"] = summary

    if payload.importance is not None:
        update_data["importance"] = _clamp_importance(payload.importance)
    if payload.level is not None:
        if payload.level not in {1, 2, 3}:
            raise MemoryRepairActionError(400, "invalid_memory_level")
        update_data["level"] = payload.level

    await memory_repo.update(record.id, source=record.source, record=record, **update_data)
    await _log_repair_changelog(
        record=record,
        operation="repair_edit",
        admin_id=admin_id,
        repair_item=repair_item,
        payload=payload,
        old_value=json.dumps(
            {"content": record.content, "summary": record.summary, "level": record.level, "importance": record.importance},
            ensure_ascii=False,
        ),
        new_value=json.dumps(update_data, ensure_ascii=False),
    )
    return {"action": payload.action, "memory_id": record.id}


async def _insert_replacement_memory(
    repair_item: dict[str, Any],
    payload: MemoryRepairActionPayload,
    admin_id: str | None,
) -> dict[str, Any]:
    content = _clean_text(payload.content)
    if not content:
        raise MemoryRepairActionError(400, "insert_replacement_memory_requires_content")

    old_record = None
    if payload.memory_id or repair_item.get("memory_id"):
        old_record = await _load_target_memory(
            repair_item,
            payload.memory_id,
            allow_archived=True,
        )

    user_id = payload.user_id or (old_record.userId if old_record else repair_item.get("user_id"))
    workspace_id = payload.workspace_id or (old_record.workspaceId if old_record else repair_item.get("workspace_id"))
    source = _coerce_source(payload.source or (old_record.source if old_record else repair_item.get("memory_source")) or "user")
    if not user_id:
        raise MemoryRepairActionError(400, "insert_replacement_memory_requires_user_id")

    level = payload.level if payload.level is not None else (old_record.level if old_record else 2)
    if level not in {1, 2, 3}:
        raise MemoryRepairActionError(400, "invalid_memory_level")

    new_id = await store_memory(
        user_id=user_id,
        content=content,
        summary=_clean_text(payload.summary) or content[:200],
        level=level,
        importance=_clamp_importance(payload.importance if payload.importance is not None else (old_record.importance if old_record else 0.7)),
        memory_type=payload.memory_type or (old_record.type if old_record else None),
        main_category=payload.main_category or (old_record.mainCategory if old_record else None),
        sub_category=payload.sub_category or (old_record.subCategory if old_record else None),
        source=source,
        workspace_id=workspace_id,
    )
    if not new_id:
        raise MemoryRepairActionError(409, "replacement_memory_was_deduped_or_blocked")

    await log_memory_changelog(
        user_id,
        new_id,
        "repair_insert_replacement",
        old_value=None,
        new_value=_audit_json(admin_id=admin_id, repair_item=repair_item, payload=payload, after={"content": content}),
        workspace_id=workspace_id,
    )
    if old_record is not None:
        try:
            await mark_memory_superseded(
                memory_id=old_record.id,
                source=old_record.source,
                superseded_by_memory_id=new_id,
                repair_item_id=repair_item.get("id"),
            )
        except Exception:
            pass
    return {"action": payload.action, "memory_id": new_id}


async def _mark_verified(
    repair_item: dict[str, Any],
    payload: MemoryRepairActionPayload,
    admin_id: str | None,
) -> dict[str, Any]:
    record = await _load_target_memory(repair_item, payload.memory_id)
    await _log_repair_changelog(
        record=record,
        operation="admin_verified",
        admin_id=admin_id,
        repair_item=repair_item,
        payload=payload,
        old_value=None,
        new_value=json.dumps({"verified": True}, ensure_ascii=False),
    )
    return {"action": payload.action, "memory_id": record.id}


async def _merge_memories(
    repair_item: dict[str, Any],
    payload: MemoryRepairActionPayload,
    admin_id: str | None,
) -> dict[str, Any]:
    content = _clean_text(payload.content)
    if not content:
        raise MemoryRepairActionError(400, "merge_memories_requires_content")
    target = await _load_target_memory(repair_item, payload.memory_id)
    merge_ids = list(dict.fromkeys(mid for mid in payload.memory_ids if mid and mid != target.id))
    if not merge_ids:
        raise MemoryRepairActionError(400, "merge_memories_requires_memory_ids")

    absorbed: list[str] = []
    for memory_id in merge_ids:
        record = await memory_repo.find_unique(memory_id)
        if not record or record.isArchived:
            raise MemoryRepairActionError(404, f"merge_memory_not_found:{memory_id}")
        _assert_same_scope(target, record)
        absorbed.append(record.id)

    summary = _clean_text(payload.summary) or content[:200]
    await store_embedding(target.id, await generate_embedding(content))
    await memory_repo.update(
        target.id,
        source=target.source,
        record=target,
        content=content,
        summary=summary,
        importance=_clamp_importance(payload.importance if payload.importance is not None else target.importance),
    )
    await _log_repair_changelog(
        record=target,
        operation="repair_merge",
        admin_id=admin_id,
        repair_item=repair_item,
        payload=payload,
        old_value=json.dumps({"content": target.content, "absorbed": absorbed}, ensure_ascii=False),
        new_value=json.dumps({"content": content, "summary": summary}, ensure_ascii=False),
    )

    for memory_id in absorbed:
        record = await memory_repo.find_unique(memory_id)
        if not record:
            continue
        await memory_repo.update(record.id, source=record.source, record=record, isArchived=True)
        await _log_repair_changelog(
            record=record,
            operation="repair_merge_archived",
            admin_id=admin_id,
            repair_item=repair_item,
            payload=payload,
            old_value=record.content,
            new_value=json.dumps({"merged_into": target.id}, ensure_ascii=False),
        )
    return {"action": payload.action, "memory_id": target.id, "absorbed_memory_ids": absorbed}


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


def _assert_same_scope(left, right) -> None:
    if left.source != right.source or left.userId != right.userId or left.workspaceId != right.workspaceId:
        raise MemoryRepairActionError(403, "merge_memories_must_share_source_user_workspace")


async def _log_repair_changelog(
    *,
    record,
    operation: str,
    admin_id: str | None,
    repair_item: dict[str, Any],
    payload: MemoryRepairActionPayload,
    old_value: str | None,
    new_value: str | None,
) -> None:
    await log_memory_changelog(
        record.userId,
        record.id,
        operation,
        old_value=old_value,
        new_value=_audit_json(
            admin_id=admin_id,
            repair_item=repair_item,
            payload=payload,
            after=new_value,
        ),
        workspace_id=record.workspaceId,
    )


def _audit_json(
    *,
    admin_id: str | None,
    repair_item: dict[str, Any],
    payload: MemoryRepairActionPayload,
    after: Any,
) -> str:
    return json.dumps(
        {
            "repair_item_id": repair_item.get("id"),
            "admin_id": admin_id,
            "action": payload.action,
            "reason": payload.reason,
            "after": after,
        },
        ensure_ascii=False,
    )


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


def _clamp_importance(value: float) -> float:
    return max(0.1, min(float(value), 0.99))


def _coerce_source(value: str) -> str:
    if value not in {"user", "ai"}:
        raise MemoryRepairActionError(400, "invalid_memory_source")
    return value
