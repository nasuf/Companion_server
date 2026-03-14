"""Memory conflict detection and resolution.

Detects contradictions between new memories and existing L1 memories,
then applies resolution strategies (update, demote, ignore).
"""

import logging

from app.db import db
from app.services.llm.models import get_utility_model, invoke_json

logger = logging.getLogger(__name__)

CONFLICT_DETECTION_PROMPT = """你是一个记忆冲突检测系统。

请分析新记忆是否与现有核心记忆（L1）存在矛盾。

现有L1记忆列表：
{existing_memories}

新提取的记忆：
{new_memory}

请判断新记忆是否与某条现有记忆存在矛盾（例如：旧记忆说"用户喜欢咖啡"，新记忆说"用户不喝咖啡了"）。

返回JSON：
{{
  "has_conflict": true/false,
  "conflicting_memory_id": "冲突的旧记忆ID（如无冲突则为null）",
  "conflict_type": "update/correction/preference_change/null",
  "confidence": 0.0-1.0,
  "reason": "简要说明冲突原因",
  "resolution": "update_l1/demote_old/ignore"
}}

规则：
- update: 新信息明确替代旧信息（如改名、换工作）→ resolution=update_l1
- correction: 用户纠正之前的错误信息 → resolution=update_l1
- preference_change: 偏好发生变化（如不再喜欢某食物）→ resolution=update_l1，旧记忆降级L2
- 如果置信度<0.8，resolution=ignore
- 如果没有冲突，has_conflict=false"""


async def detect_conflicts(
    user_id: str,
    new_memory: dict,
) -> dict | None:
    """Detect if a new memory conflicts with existing L1 memories.

    Returns conflict info dict or None if no conflict.
    """
    existing = await db.memory.find_many(
        where={"userId": user_id, "level": 1, "isArchived": False},
        order={"importance": "desc"},
        take=20,
    )

    if not existing:
        return None

    existing_text = "\n".join(
        f"- [ID:{m.id}] {m.summary or m.content}" for m in existing
    )
    new_text = new_memory.get("summary", "")

    prompt = CONFLICT_DETECTION_PROMPT.format(
        existing_memories=existing_text,
        new_memory=new_text,
    )

    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"Conflict detection failed: {e}")
        return None

    if not result.get("has_conflict", False):
        return None

    if result.get("confidence", 0) < 0.8:
        logger.info(f"Conflict detected but low confidence ({result.get('confidence')}), ignoring")
        return None

    return result


async def resolve_conflict(
    user_id: str,
    conflict: dict,
    new_memory: dict,
) -> str:
    """Apply conflict resolution strategy.

    Returns the resolution action taken: 'updated', 'demoted', 'ignored'.
    """
    resolution = conflict.get("resolution", "ignore")
    old_id = conflict.get("conflicting_memory_id")

    if resolution == "ignore" or not old_id:
        return "ignored"

    if resolution == "update_l1":
        old_memory = await db.memory.find_unique(where={"id": old_id})
        if not old_memory:
            return "ignored"

        old_content = old_memory.content

        if conflict.get("conflict_type") == "preference_change":
            # Demote old memory to L2
            await db.memory.update(
                where={"id": old_id},
                data={"level": 2, "importance": max(0.3, old_memory.importance - 0.2)},
            )
            logger.info(f"Demoted conflicting memory {old_id} from L1 to L2")

            # Log the change
            await _log_changelog(user_id, old_id, "update", old_content, f"降级L2: {conflict.get('reason', '')}")
            return "demoted"
        else:
            # Update the old memory with new content
            new_content = new_memory.get("summary", "")
            await db.memory.update(
                where={"id": old_id},
                data={"content": new_content, "summary": new_content},
            )
            logger.info(f"Updated conflicting memory {old_id}: {old_content[:30]} -> {new_content[:30]}")

            await _log_changelog(user_id, old_id, "update", old_content, new_content)
            return "updated"

    return "ignored"


async def _log_changelog(
    user_id: str,
    memory_id: str,
    operation: str,
    old_value: str | None = None,
    new_value: str | None = None,
) -> None:
    """Write a memory changelog entry."""
    try:
        await db.memorychangelog.create(
            data={
                "user": {"connect": {"id": user_id}},
                "memory": {"connect": {"id": memory_id}},
                "operation": operation,
                "oldValue": old_value,
                "newValue": new_value,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to write changelog: {e}")
