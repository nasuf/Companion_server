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
from app.services.llm.models import get_utility_model, get_chat_model, invoke_json, invoke_text
from app.services.memory.config import DELETION_SIMILARITY_THRESHOLD, LLM_INTENT_MIN_CONFIDENCE
from app.services.memory.storage.embedding import generate_embedding
from app.services.memory.retrieval.vector_search import search_by_embedding
from app.services.memory.storage.persistence import log_memory_changelog
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

_PENDING_DELETION_PREFIX = "deletion:pending:"
_PENDING_DELETION_TTL = 300  # 5 min for confirmation

# Keywords that may indicate deletion intent
DELETION_KEYWORDS = [
    "忘了", "忘掉", "别记了", "不记得", "删除", "删掉",
    "不要记", "别提了", "忘记", "去掉", "移除",
    "forget", "delete", "remove", "don't remember",
]

DELETION_INTENT_PROMPT = """判断用户是否在要求AI忘记/删除某条记忆。

用户消息：{message}

返回JSON：
{{
  "is_deletion_request": true/false,
  "target_description": "用户想删除的记忆描述（如无则为null）",
  "confidence": 0.0-1.0
}}

规则：
- 用户说"忘了吧"、"别记了"、"删掉那个"等属于删除请求
- 用户说"我忘了"（表达自己忘记）不是删除请求
- 只有明确要求AI删除/忘记时才返回true"""

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
    """用LLM生成删除记忆后的委婉回复。失败时用模板。"""
    if deleted_count == 0:
        return "嗯...我好像没有关于这个的记忆呢。"

    prompt = (
        f"你是{agent_name}。用户要求你忘掉关于「{description}」的事情，你已经忘掉了。\n"
        "请用1句话自然地回复用户，语气温和体贴，像朋友聊天一样。\n"
        "不要说「好的」开头，不要提及「记忆」或「删除」这类技术词汇。"
    )
    try:
        return await invoke_text(get_chat_model(), prompt)  # spec §5.3: 大模型生成删除回复
    except Exception:
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
    """Spec §5.2: show candidates and ask user to confirm."""
    previews = "\n".join(
        f"  {i+1}. {c.get('content', c.get('summary', ''))[:60]}"
        for i, c in enumerate(candidates[:5])
    )
    prompt = (
        f"你是{agent_name}。用户想让你忘掉一些事情，你找到了以下相关记忆：\n"
        f"{previews}\n\n"
        "请用1-2句温和的话确认用户是否真的要忘掉这些内容。"
        "语气像朋友确认一样，不要用技术词汇。"
    )
    try:
        return await invoke_text(get_chat_model(), prompt)  # spec §5.2: 大模型生成确认回复
    except Exception:
        return f"我找到了{len(candidates)}条相关的记忆，你确定要我忘掉吗？"


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
