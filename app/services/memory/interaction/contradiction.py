"""Interactive contradiction handling (spec §4).

When a user's new message conflicts with an existing L1 memory, the AI should:
1. Detect the contradiction — memory.contradiction_detection (small model, §4.1)
2. Ask the user naturally — memory.contradiction_inquiry (big model, §4.2)
3. Analyze user's response — memory.contradiction_analysis (small model, §4.3)
4. Adjust memories (demote old L1 → L2, create new entry, §4.4)
5. Generate wrap-up reply — memory.contradiction_reply (big model, §4.5)

All 5 LLM calls go through the prompt registry; no inline prompts.
"""

from __future__ import annotations

import json
import logging

from app.redis_client import get_redis
from app.services.llm.models import (
    get_chat_model,
    get_utility_model,
    invoke_json,
    invoke_text,
)
from app.services.memory.storage import repo as memory_repo
from app.services.prompting.store import get_prompt_text
from app.services.prompting.utils import SafeDict, pad_params

_PENDING_KEY_PREFIX = "contradiction:pending:"
_PENDING_TTL = 1800  # 30 min — generous window for user to reply

logger = logging.getLogger(__name__)


async def detect_l1_contradiction(
    user_message: str,
    user_id: str,
    workspace_id: str | None = None,
) -> dict | None:
    """Spec §4.1: detect if user's message contradicts any L1 memory.

    Returns {"has_conflict": True, "old_memory_id": ..., "old_content": ...,
             "new_info": ..., "conflict_description": ...} or None.
    """
    # spec §4.1: 输入 = "关于该用户的核心记忆" + 用户当前提及内容.
    # 只取 source=user 的 L1 — AI 人设记忆 (e.g. "我在苏州长大" 描述的是 AI 自己)
    # 跟"用户提了什么事实"无关, 混入会导致 LLM 把 AI 第一人称叙述误读为用户说的话
    # (实测: 用户说"我是西安人", LLM 把 AI 的"在苏州长大"当成用户说过的, 误报矛盾).
    # 全量取 L1 (不按 mainCategory 过滤, 不限 take), 安全网 300 防极端情况.
    l1_user = await memory_repo.find_many(
        source="user",
        where={
            "userId": user_id, "workspaceId": workspace_id,
            "level": 1, "isArchived": False,
        },
        order={"importance": "desc"}, take=300,
    )
    if not l1_user:
        return None

    l1_text = "\n".join(f"[{m.id}] {m.summary or m.content}" for m in l1_user)

    try:
        template = await get_prompt_text("memory.contradiction_detection")
        prompt = template.format_map(SafeDict({
            "user_message": user_message,
            "existing_l1_memory": l1_text,
        }))
        result = await invoke_json(get_utility_model(), prompt)
        if isinstance(result, dict) and result.get("has_conflict"):
            return result
        return None
    except Exception as e:
        logger.warning(f"L1 contradiction detection failed: {e}")
        return None


async def generate_contradiction_inquiry(
    conflict: dict,
    agent_name: str = "AI",
    recent_context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
) -> str:
    """Spec §4.2: generate a natural inquiry about the contradiction.

    Uses `memory.contradiction_inquiry` (registry). Friendly, not accusatory.
    """
    try:
        template = await get_prompt_text("memory.contradiction_inquiry")
        params = {
            "user_message": conflict.get("new_info", ""),
            "original_memory": conflict.get("old_content", ""),
            "conflict_memory": conflict.get("conflict_description", ""),
            "recent_context": recent_context or "(无)",
            "personality_brief": personality_brief or agent_name,
            "user_portrait": user_portrait or "(未知)",
            **pad_params(user_emotion),
        }
        prompt = template.format_map(SafeDict(params))
        return (await invoke_text(get_chat_model(), prompt)).strip()
    except Exception as e:
        logger.warning(f"Contradiction inquiry generation failed: {e}")
        return "诶,我记得你之前说的不太一样,是情况有变化吗?"


async def analyze_contradiction_response(
    user_reply: str,
    conflict: dict,
    recent_context: str = "",
) -> dict:
    """Spec §4.3: analyze user's response → 变化 / 新增 / 错误 + 调整方案。"""
    try:
        template = await get_prompt_text("memory.contradiction_analysis")
        prompt = template.format_map(SafeDict({
            "user_reply": user_reply,
            "recent_context": recent_context or "(无)",
            "original_memory": conflict.get("old_content", ""),
            "conflict_memory": conflict.get("conflict_description", ""),
        }))
        result = await invoke_json(get_utility_model(), prompt)
        return result if isinstance(result, dict) else {"change_type": "新增", "reason": "解析失败"}
    except Exception as e:
        logger.warning(f"Contradiction response analysis failed: {e}")
        return {"change_type": "新增", "reason": str(e)[:20]}


async def generate_contradiction_reply(
    user_message: str,
    conflict: dict,
    analysis: dict,
    recent_context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
) -> str:
    """Spec §4.5: 用户解释清楚后，自然地把话题拉回正轨。"""
    try:
        template = await get_prompt_text("memory.contradiction_reply")
        params = {
            "user_message": user_message,
            "recent_context": recent_context or "(无)",
            "original_memory": conflict.get("old_content", ""),
            "conflict_memory": conflict.get("conflict_description", ""),
            "change_reason": analysis.get("reason", ""),
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            **pad_params(user_emotion),
        }
        prompt = template.format_map(SafeDict(params))
        return (await invoke_text(get_chat_model(), prompt)).strip()
    except Exception as e:
        logger.warning(f"Contradiction reply generation failed: {e}")
        return "好的，我记住了~"


async def apply_contradiction_resolution(
    conflict: dict,
    analysis: dict,
) -> None:
    """Spec §4.4: adjust memories based on analysis.

    - 变化/错误: demote old L1 → L2, create new entry at appropriate level
    - 新增: keep old L1, add new entry alongside
    """
    change_type = analysis.get("change_type", "新增")
    old_id = conflict.get("conflicting_memory_id")

    if change_type in ("变化", "错误") and old_id:
        # Demote old L1 → L2 with reduced importance
        old_mem = await memory_repo.find_unique(old_id)
        if old_mem:
            new_imp = max(0.30, (old_mem.importance or 0.5) - 0.20)
            await memory_repo.update(
                old_id,
                source=getattr(old_mem, "source", "user"),
                record=old_mem,
                level=2,
                importance=new_imp,
            )
            logger.info(f"Contradiction: demoted {old_id} L1→L2 ({change_type})")

    # New memory from analysis (if provided) will be created by the normal
    # memory extraction pipeline on the user's latest message — no need to
    # double-create here. The orchestrator ensures the message goes through
    # the standard memory recording flow after contradiction resolution.


# ── Conversation-level state management ──────────────────────────────────
# Stores a pending contradiction in Redis so the next user message can
# trigger steps 3-5. Expires after _PENDING_TTL (10 min).

async def save_pending_contradiction(conversation_id: str, conflict: dict) -> None:
    redis = await get_redis()
    await redis.set(
        f"{_PENDING_KEY_PREFIX}{conversation_id}",
        json.dumps(conflict, ensure_ascii=False),
        ex=_PENDING_TTL,
    )


async def load_pending_contradiction(conversation_id: str) -> dict | None:
    redis = await get_redis()
    raw = await redis.get(f"{_PENDING_KEY_PREFIX}{conversation_id}")
    if not raw:
        return None
    try:
        data = json.loads(raw if isinstance(raw, str) else raw.decode())
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


async def clear_pending_contradiction(conversation_id: str) -> None:
    redis = await get_redis()
    await redis.delete(f"{_PENDING_KEY_PREFIX}{conversation_id}")
