"""Memory deletion by user request (spec §5).

4-step flow:
  1. Intent recognition (keyword + LLM)
  2. Find candidates → generate confirmation reply (show what would be deleted)
  3. User confirms → execute deletion
  4. Physical delete + audit log

Uses Redis pending state (same pattern as contradiction) to remember
deletion candidates across the confirmation round-trip.
"""

import json
import logging

from app.redis_client import get_redis
from app.services.memory.storage import repo as memory_repo
from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory.config import DELETION_SIMILARITY_THRESHOLD, LLM_INTENT_MIN_CONFIDENCE
from app.services.memory.storage.embedding import generate_embedding
from app.services.memory.retrieval.vector_search import search_by_embedding
from app.services.memory.storage.persistence import log_memory_changelog
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

_PENDING_DELETION_PREFIX = "deletion:pending:"
_PENDING_DELETION_TTL = 300  # 5 min for confirmation

# Keywords that may indicate deletion intent
# Part 5 §4.2: 用户说"不用提醒了/取消提醒"→ 走记忆删除机制 (针对 reminder 子类)
DELETION_KEYWORDS = [
    "忘了", "忘掉", "别记了", "不记得", "删除", "删掉",
    "不要记", "别提了", "忘记", "去掉", "移除",
    # Part 5 §4.2 提醒取消语义
    "不用提醒", "取消提醒", "不用记着", "不用再提",
    "forget", "delete", "remove", "don't remember",
]

DELETION_RESPONSE_TEMPLATES = [
    "好的，那件事我不会再提了。",
    "嗯，已经忘掉了~",
    "了解，以后不会再提起这个了。",
    "好吧，就当没发生过。",
]


async def detect_deletion_intent(message: str) -> dict | None:
    """Detect if user wants to delete a memory.

    Returns deletion intent info or None.
    """
    # Quick keyword check
    has_keyword = any(kw in message for kw in DELETION_KEYWORDS)
    if not has_keyword:
        return None

    # Confirm with LLM
    prompt = (await get_prompt_text("memory.deletion_intent")).format(message=message)
    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"Deletion intent detection failed: {e}")
        return None

    if not result.get("is_deletion_request", False):
        return None
    if result.get("confidence", 0) < LLM_INTENT_MIN_CONFIDENCE:
        return None

    return result


async def find_matching_memories(
    user_id: str,
    description: str,
    threshold: float = 0.7,
) -> list[dict]:
    """查找匹配的记忆但不删除，返回候选列表。"""
    embedding = await generate_embedding(description)
    results = await search_by_embedding(embedding, user_id, top_k=5)
    matches = []
    for r in results:
        sim = float(r.get("similarity", 0))
        if sim >= threshold:
            matches.append(r)
    return matches


async def generate_deletion_reply(
    agent_name: str,
    description: str,
    deleted_count: int,
) -> str:
    """删除记忆后的兜底回复 (静态模板).

    spec §5.3 主路径走 registry-backed `intent.deletion_reply` (intent_replies.
    deletion_done_reply), 这里仅在主 LLM 返回 None/空时承接, 因此不再二次调
    LLM——同一会话同一窗口再调一次内联简化 prompt 多半也会失败, 直接给模板更
    稳更快。模板池见 DELETION_RESPONSE_TEMPLATES。
    agent_name / description 仅留作签名兼容, 不再使用。
    """
    del agent_name, description  # 保留签名兼容
    if deleted_count == 0:
        return "嗯...我好像没有关于这个的记忆呢。"
    return get_deletion_response()


async def delete_memories_by_description(
    user_id: str,
    description: str,
) -> int:
    """Find and delete memories matching the description.

    Returns number of deleted memories.
    """
    # Generate embedding for the target description
    embedding = await generate_embedding(description)

    # Find similar memories
    results = await search_by_embedding(embedding, user_id, top_k=5)

    deleted = 0
    for r in results:
        sim = r.get("similarity", 0)
        if isinstance(sim, str):
            sim = float(sim)

        if sim < DELETION_SIMILARITY_THRESHOLD:
            continue

        memory_id = r.get("id")
        if not memory_id:
            continue

        # Audit log BEFORE delete (once deleted, memory row & content are gone)
        memory = await memory_repo.find_unique(memory_id)
        if memory:
            await log_memory_changelog(
                user_id, memory_id, "delete",
                old_value=memory.content,
            )

        # memory_repo.delete handles embedding cascade in the safe order
        # (memory row first → embedding row). Retrieval is guaranteed to
        # miss the record the moment the memory row disappears.
        try:
            await memory_repo.delete(memory_id)
            deleted += 1
            logger.info(f"Deleted memory {memory_id}: {r.get('content', '')[:50]}")
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id}: {e}")

    return deleted


def get_deletion_response() -> str:
    """Get a natural language response for memory deletion."""
    import random
    return random.choice(DELETION_RESPONSE_TEMPLATES)


# ── Spec §5.2-5.3: Confirmation state ────────────────────────────────────

async def save_pending_deletion(conversation_id: str, candidates: list[dict]) -> None:
    """Store deletion candidates in Redis so user can confirm."""
    redis = await get_redis()
    await redis.set(
        f"{_PENDING_DELETION_PREFIX}{conversation_id}",
        json.dumps(candidates, ensure_ascii=False),
        ex=_PENDING_DELETION_TTL,
    )


async def load_pending_deletion(conversation_id: str) -> list[dict] | None:
    redis = await get_redis()
    raw = await redis.get(f"{_PENDING_DELETION_PREFIX}{conversation_id}")
    if not raw:
        return None
    try:
        data = json.loads(raw if isinstance(raw, str) else raw.decode())
        return data if isinstance(data, list) else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


async def clear_pending_deletion(conversation_id: str) -> None:
    redis = await get_redis()
    await redis.delete(f"{_PENDING_DELETION_PREFIX}{conversation_id}")


_CONFIRM_KEYWORDS = {"对", "是", "是的", "确认", "删掉", "删吧", "好", "好的", "嗯", "ok", "yes"}


def is_deletion_confirmed(user_reply: str) -> bool:
    """Check if user's reply is a confirmation to proceed with deletion."""
    return user_reply.strip().lower() in _CONFIRM_KEYWORDS


async def generate_deletion_confirmation_prompt(
    agent_name: str,
    candidates: list[dict],
) -> str:
    """Spec §5.2 兜底确认提示 (静态模板).

    主路径走 registry-backed `intent.deletion_confirm` (intent_replies.
    deletion_confirm_reply), 这里仅在 None/空 时承接——同会话再调一次简化版
    LLM 没意义, 直接列候选更稳。agent_name 留着作签名兼容, 不再使用。
    """
    del agent_name  # 仅保留签名兼容
    previews = "\n".join(
        f"  {i + 1}. {c.get('content', c.get('summary', ''))[:60]}"
        for i, c in enumerate(candidates[:5])
    )
    return f"我找到了这些可能相关的记忆：\n{previews}\n\n你确定要我把这些都忘掉吗？"



async def execute_confirmed_deletion(
    user_id: str,
    candidates: list[dict],
) -> int:
    """Spec §5.3-5.4: execute physical deletion after confirmation."""
    deleted = 0
    for c in candidates:
        memory_id = c.get("id")
        if not memory_id:
            continue
        memory = await memory_repo.find_unique(memory_id)
        if memory:
            await log_memory_changelog(user_id, memory_id, "delete", old_value=memory.content)
        try:
            await memory_repo.delete(memory_id)
            deleted += 1
        except Exception as e:
            logger.warning(f"Confirmed deletion failed for {memory_id}: {e}")
    return deleted
