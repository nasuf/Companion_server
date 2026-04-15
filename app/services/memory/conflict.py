"""Memory conflict detection and resolution.

Detects contradictions between new memories and existing L1 memories,
then applies resolution strategies (update, demote, ignore).
"""

import logging

from app.services.memory import memory_repo
from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory.config import LLM_INTENT_MIN_CONFIDENCE
from app.services.memory.storage import log_memory_changelog
from app.services.memory.taxonomy import conflict_candidate_scope
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

# Prompt is loaded from the prompt store at runtime via get_prompt_text("memory.conflict_detection")


async def detect_conflicts(
    user_id: str,
    new_memory: dict,
    workspace_id: str | None = None,
) -> dict | None:
    """Detect if a new memory conflicts with existing L1 memories.

    Returns conflict info dict or None if no conflict.
    """
    scope = conflict_candidate_scope(
        new_memory.get("main_category"),
        new_memory.get("sub_category"),
    )
    if not scope.get("should_check", True):
        return None

    where: dict = {"userId": user_id, "level": 1, "isArchived": False}
    if workspace_id:
        where["workspaceId"] = workspace_id
    if new_memory.get("main_category"):
        where["mainCategory"] = new_memory.get("main_category")
    if scope.get("prefer_same_sub_category") and new_memory.get("sub_category"):
        where["subCategory"] = new_memory.get("sub_category")

    # 搜索 user + ai 两种 source 的 L1 记忆, 确保对话记忆能与人生经历(ai)做冲突检测
    existing_user = await memory_repo.find_many(
        source="user", where=where, order={"importance": "desc"}, take=10,
    )
    existing_ai = await memory_repo.find_many(
        source="ai", where=where, order={"importance": "desc"}, take=10,
    )
    existing = existing_user + existing_ai

    if not existing:
        return None

    existing_text = "\n".join(
        f"- [ID:{m.id}] {m.summary or m.content}" for m in existing
    )
    new_text = new_memory.get("summary", "")

    prompt = (await get_prompt_text("memory.conflict_detection")).format(
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

    if result.get("confidence", 0) < LLM_INTENT_MIN_CONFIDENCE:
        logger.info(f"Conflict detected but low confidence ({result.get('confidence')}), ignoring")
        return None

    if not result.get("resolution") or result.get("resolution") == "ignore":
        default_resolution = scope.get("default_resolution")
        if default_resolution:
            result["resolution"] = default_resolution

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

    if resolution in ("update_l1", "demote_old"):
        old_memory = await memory_repo.find_unique(old_id)
        if not old_memory:
            return "ignored"

        old_content = old_memory.content

        if conflict.get("conflict_type") == "preference_change" or resolution == "demote_old":
            # Demote old memory to L2
            await memory_repo.update(
                old_id, source=old_memory.source, record=old_memory,
                level=2, importance=max(0.3, old_memory.importance - 0.2),
            )
            logger.info(f"Demoted conflicting memory {old_id} from L1 to L2")

            await log_memory_changelog(user_id, old_id, "update", old_content, f"降级L2: {conflict.get('reason', '')}")
            return "demoted"
        else:
            # Update the old memory with new content
            new_content = new_memory.get("summary", "")
            await memory_repo.update(
                old_id, source=old_memory.source, record=old_memory,
                content=new_content, summary=new_content,
            )
            logger.info(f"Updated conflicting memory {old_id}: {old_content[:30]} -> {new_content[:30]}")

            await log_memory_changelog(user_id, old_id, "update", old_content, new_content)
            return "updated"

    return "ignored"
