from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.db import db
from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory.storage import repo as memory_repo
from app.services.offline.user_tags import derive_user_tags
from app.services.prompting.store import get_prompt_text
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

_MAX_TAGS = 9
_MIN_CONFIDENCE = 0.45
_ALLOWED_CATEGORIES = {
    "preference",
    "behavior",
    "personality",
    "lifestyle",
    "creative",
    "relationship",
    "work",
}
_BLOCKED_LABELS = {
    "姓名",
    "性别",
    "年龄",
    "职业",
    "职业与经济",
    "职业/与经济",
    "价值观",
    "身份",
    "生活",
    "思维",
    "情绪",
}
_SENSITIVE_LABEL_PARTS = {
    "轻生",
    "自杀",
    "辱骂",
    "冷战",
    "分手",
    "前女友",
    "台湾",
}


@dataclass(frozen=True)
class ProfileTag:
    label: str
    category: str
    confidence: float
    source_memory_ids: list[str]
    evidence_count: int
    source: str = "llm"


async def list_profile_tags(
    user_id: str,
    workspace_id: str | None,
    *,
    agent_id: str | None = None,
    limit: int = _MAX_TAGS,
) -> list[str]:
    rows = await db.query_raw(
        """
        SELECT label
        FROM user_profile_tags
        WHERE user_id = $1
          AND (($2::text IS NULL AND workspace_id IS NULL) OR workspace_id = $2)
          AND ($3::text IS NULL OR agent_id = $3)
          AND is_active = TRUE
        ORDER BY confidence DESC, evidence_count DESC, updated_at DESC
        LIMIT $4
        """,
        user_id,
        workspace_id,
        agent_id,
        limit,
    )
    labels: list[str] = []
    for row in rows or []:
        label = str(_field(row, "label") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


async def has_active_profile_tags(
    user_id: str,
    workspace_id: str | None,
    *,
    agent_id: str | None = None,
) -> bool:
    return bool(
        await list_profile_tags(
            user_id,
            workspace_id,
            agent_id=agent_id,
            limit=1,
        )
    )


async def refresh_profile_tags(
    user_id: str,
    agent_id: str,
    *,
    workspace_id: str | None = None,
    portrait: str | None = None,
) -> list[str]:
    workspace_id = workspace_id or await resolve_workspace_id(
        user_id=user_id,
        agent_id=agent_id,
    )
    memories = await memory_repo.find_many(
        source="user",
        where={
            "userId": user_id,
            "workspaceId": workspace_id,
            "isArchived": False,
            "level": {"in": [1, 2]},
        },
        order={"importance": "desc"},
        take=50,
    )
    if not memories:
        logger.info("No L1/L2 memories for profile tags user=%s", user_id)
        return []

    memory_lines = "\n".join(
        f"- id={m.id} [L{m.level}] [{m.mainCategory or '未分类'}/{m.subCategory or '其他'}] {m.content}"
        for m in memories
    )
    tags: list[ProfileTag] = []
    try:
        prompt = (await get_prompt_text("portrait.tags")).format(
            portrait=portrait or "暂无",
            memories=memory_lines,
        )
        raw = await invoke_json(get_utility_model(), prompt, profile="background")
        tags = _normalize_llm_tags(raw, valid_memory_ids={m.id for m in memories})
    except Exception as exc:
        logger.warning("Profile tag LLM generation failed for user=%s: %s", user_id, exc)

    if not tags:
        fallback_rows = [
            {
                "content": m.content,
                "main_category": m.mainCategory,
                "sub_category": m.subCategory,
            }
            for m in memories
        ]
        tags = [
            ProfileTag(
                label=label,
                category="preference",
                confidence=0.55,
                source_memory_ids=[],
                evidence_count=0,
                source="rules",
            )
            for label in derive_user_tags(fallback_rows, limit=_MAX_TAGS)
        ]

    if not tags:
        return []

    await _replace_active_tags(
        user_id,
        agent_id,
        workspace_id,
        tags[:_MAX_TAGS],
    )
    return [tag.label for tag in tags[:_MAX_TAGS]]


def _normalize_llm_tags(raw: Any, *, valid_memory_ids: set[str]) -> list[ProfileTag]:
    items = raw.get("tags") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    tags: list[ProfileTag] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        label = _clean_label(item.get("label"))
        if not label or label in seen:
            continue
        confidence = _coerce_confidence(item.get("confidence"))
        if confidence < _MIN_CONFIDENCE:
            continue
        category = str(item.get("category") or "preference").strip().lower()
        if category not in _ALLOWED_CATEGORIES:
            category = "preference"
        source_ids = _clean_source_ids(
            item.get("source_memory_ids"),
            valid_memory_ids=valid_memory_ids,
        )
        tags.append(
            ProfileTag(
                label=label,
                category=category,
                confidence=confidence,
                source_memory_ids=source_ids,
                evidence_count=len(source_ids),
            )
        )
        seen.add(label)
        if len(tags) >= _MAX_TAGS:
            break
    return tags


async def _replace_active_tags(
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    tags: list[ProfileTag],
) -> None:
    await db.execute_raw(
        """
        UPDATE user_profile_tags
        SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
          AND agent_id = $2
          AND (($3::text IS NULL AND workspace_id IS NULL) OR workspace_id = $3)
          AND is_active = TRUE
        """,
        user_id,
        agent_id,
        workspace_id,
    )
    for tag in tags:
        await db.execute_raw(
            """
            INSERT INTO user_profile_tags (
                id, user_id, agent_id, workspace_id, label, category,
                confidence, source_memory_ids, evidence_count, source,
                is_active, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8::jsonb, $9, $10,
                TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            )
            """,
            uuid4().hex,
            user_id,
            agent_id,
            workspace_id,
            tag.label,
            tag.category,
            tag.confidence,
            json.dumps(tag.source_memory_ids, ensure_ascii=False),
            tag.evidence_count,
            tag.source,
        )


def _clean_label(value: Any) -> str:
    label = str(value or "").strip(" \n\t\r。.!！?？；;，,、")
    if not (2 <= len(label) <= 12):
        return ""
    if label in _BLOCKED_LABELS:
        return ""
    if any(part in label for part in _SENSITIVE_LABEL_PARTS):
        return ""
    if label.endswith(("标签", "用户", "的人")):
        return ""
    return label


def _clean_source_ids(value: Any, *, valid_memory_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        item_id = str(item or "").strip()
        if item_id in valid_memory_ids and item_id not in ids:
            ids.append(item_id)
    return ids[:5]


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _field(row: Any, snake: str, camel: str | None = None) -> Any:
    if isinstance(row, dict):
        if snake in row:
            return row[snake]
        if camel and camel in row:
            return row[camel]
        return None
    if hasattr(row, snake):
        return getattr(row, snake)
    if camel and hasattr(row, camel):
        return getattr(row, camel)
    return None
