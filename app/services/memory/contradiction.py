"""Interactive contradiction handling (spec §4).

When a user's new message conflicts with an existing L1 memory, the AI should:
1. Detect the contradiction (small model)
2. Ask the user naturally about the discrepancy (big model)
3. Analyze user's response (small model → 变化/新增/错误)
4. Adjust memories (demote old L1 → L2, create new entry)
5. Generate a natural reply acknowledging the change

This module provides the detection + inquiry generation steps (1-2).
Steps 3-5 are handled in the orchestrator when the user replies to the inquiry.
"""

from __future__ import annotations

import json
import logging

from app.redis_client import get_redis
from app.services.llm.models import get_utility_model, get_chat_model, invoke_json
from app.services.memory import memory_repo

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
    # Only load identity + lifestyle L1 memories — these are the categories
    # where factual contradictions matter (name, age, location, job...).
    # Loading ALL L1 (170+) would blow up token cost per message.
    contradiction_categories = ("身份", "生活")
    l1_user = await memory_repo.find_many(
        source="user",
        where={
            "userId": user_id, "workspaceId": workspace_id,
            "level": 1, "isArchived": False,
            "mainCategory": {"in": list(contradiction_categories)},
        },
        order={"importance": "desc"},
        take=15,
    )
    l1_ai = await memory_repo.find_many(
        source="ai",
        where={
            "userId": user_id, "workspaceId": workspace_id,
            "level": 1, "isArchived": False,
            "mainCategory": {"in": list(contradiction_categories)},
        },
        order={"importance": "desc"},
        take=15,
    )
    all_l1 = l1_user + l1_ai
    if not all_l1:
        return None

    l1_text = "\n".join(f"[{m.id}] {m.summary or m.content}" for m in all_l1)

    prompt = f"""你是一个记忆矛盾检测器。判断用户新消息是否与已有 L1 核心记忆矛盾。

已有 L1 记忆:
{l1_text}

用户新消息: {user_message}

判断标准:
- 矛盾: 新消息中的事实与已有记忆直接冲突(如记忆说住北京,新消息说住上海)
- 非矛盾: 补充信息、无关话题、或与记忆一致

输出 JSON:
{{
  "has_conflict": true/false,
  "conflicting_memory_id": "记忆ID(若有矛盾)",
  "old_content": "原记忆内容",
  "new_info": "用户提到的新信息",
  "conflict_description": "矛盾描述(≤30字)"
}}
"""
    try:
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
) -> str:
    """Spec §4.2: generate a natural inquiry about the contradiction.

    The AI asks the user in a friendly way: "wait, I thought you lived in X,
    did something change?"
    """
    prompt = f"""你是 {agent_name}，用户刚才说的话似乎和你记忆中的信息不一致。

原来的记忆: {conflict.get('old_content', '')}
用户新说的: {conflict.get('new_info', '')}
矛盾描述: {conflict.get('conflict_description', '')}

请以朋友的语气温和地询问用户:
- 是情况发生了变化?
- 还是之前的记忆有误?
- 或者是你理解错了?

要求:
- 语气自然温暖,像朋友关心对方
- 不要用"根据我的记忆"这种机械表达
- 简短,1-2句话
- 用口语化的表达

直接输出询问文本,不要 JSON。
"""
    try:
        from app.services.llm.models import invoke_text
        return await invoke_text(get_chat_model(), prompt)
    except Exception as e:
        logger.warning(f"Contradiction inquiry generation failed: {e}")
        return f"诶,我记得你之前说的不太一样,是情况有变化吗?"


async def analyze_contradiction_response(
    user_reply: str,
    conflict: dict,
) -> dict:
    """Spec §4.3: analyze user's response to contradiction inquiry.

    Returns {"change_type": "变化"|"新增"|"错误",
             "reason": "...",
             "updated_memory": "...",
             "new_memory": "..."}
    """
    prompt = f"""用户回复了你的矛盾询问。请分析用户的意图:

原记忆: {conflict.get('old_content', '')}
矛盾内容: {conflict.get('new_info', '')}
用户回复: {user_reply}

输出 JSON:
{{
  "change_type": "变化" 或 "新增" 或 "错误",
  "reason": "变化原因(≤20字)",
  "updated_memory": "原记忆应更新为的内容(如适用)",
  "new_memory": "需新增的记忆内容(如适用)"
}}

- 变化: 用户确认情况发生了变化(搬家、换工作等)
- 新增: 新信息是补充而非替代(可以和原记忆共存)
- 错误: 原记忆有误,需要纠正
"""
    try:
        result = await invoke_json(get_utility_model(), prompt)
        return result if isinstance(result, dict) else {"change_type": "新增", "reason": "解析失败"}
    except Exception as e:
        logger.warning(f"Contradiction response analysis failed: {e}")
        return {"change_type": "新增", "reason": str(e)[:20]}


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
