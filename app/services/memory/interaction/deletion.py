"""Memory deletion / reschedule by user request (spec §5 + Phase 5 扩展).

4-step flow:
  1. Intent recognition (keyword + LLM) — 同时识别 delete / reschedule
  2. Find candidates → generate confirmation reply (show what would be deleted/moved)
  3. User confirms → execute deletion 或 apply_reschedule
  4. Physical delete + audit log; reschedule 还要更新对应 timetrigger

Uses Redis pending state (same pattern as contradiction) to remember
deletion / reschedule candidates across the confirmation round-trip.

Pending shape: 历史是 list[dict] (delete-only); Phase 5 起新写入是
`{"action": "delete"|"reschedule", "candidates": [...], "new_time": "ISO|None"}`.
load_pending_deletion 同时识别两种 shape 保持向后兼容.
"""

import json
import logging
import re
from datetime import datetime

from app.db import db
from app.observability.events import EVT_MEMORY_DELETED, EVT_MEMORY_DELETION_PENDING
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

# Keywords that may indicate deletion / reschedule intent
# Part 5 §4.2: 用户说"不用提醒了/取消提醒"→ 走记忆删除机制 (针对 reminder 子类)
# Phase 5 (改期): 用户说"挪到/改到/推迟到/提前到/调到 X" 也走同一 detect 路径,
# LLM intent 字段区分 delete vs reschedule.
DELETION_KEYWORDS = [
    "忘了", "忘掉", "别记了", "不记得", "删除", "删掉",
    "不要记", "别提了", "忘记", "去掉", "移除",
    # Part 5 §4.2 提醒取消语义
    "不用提醒", "取消提醒", "不用记着", "不用再提",
    # Phase 5 改期语义
    "挪到", "改到", "推迟到", "提前到", "调到", "改成",
    "forget", "delete", "remove", "don't remember", "reschedule",
]

DELETION_RESPONSE_TEMPLATES = [
    "好的，那件事我不会再提了。",
    "嗯，已经忘掉了~",
    "了解，以后不会再提起这个了。",
    "好吧，就当没发生过。",
]


_DEICTIC_DELETION_TERMS = (
    "这一点", "这点", "这个", "这件事", "刚才那个", "刚刚那个",
    "上面那个", "那一点", "那点", "那个",
)


def _ensure_deletion_prompt_has_context(prompt: str) -> str:
    """兼容数据库里尚未刷新 defaults 的旧 prompt.

    prompt registry 的默认模板改了不代表线上 DB 里的模板立刻同步；删除链路如果
    继续只给当前句子，"这一点/这个"这类请求就仍然无法解析。这里在运行时补
    上最近对话，保证能力不依赖 prompt 种子状态。
    """
    if "{context}" in prompt:
        return prompt
    return (
        "最近对话（旧→新）：\n{context}\n\n"
        f"{prompt}\n\n"
        "补充规则：如果用户说\"这一点/这个/刚才那个\"，必须根据最近对话还原 "
        "target_description，而不是返回 null。"
    )


def _last_prefixed_line(context: str, prefixes: tuple[str, ...]) -> str | None:
    """从 format_recent_context 文本里取最后一条指定说话人的内容."""
    for raw in reversed((context or "").splitlines()):
        line = raw.strip()
        for prefix in prefixes:
            marker = f"{prefix}:"
            if line.startswith(marker):
                value = line[len(marker):].strip()
                return value or None
    return None


def _clean_short_answer(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"[，。！？?~～]", " ", text)
    cleaned = re.sub(r"(对吧|是吧|应该是|我记得|你是|你叫)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:40] if cleaned else None


def _structured_target_from_recent_qa(last_user: str | None, last_ai: str | None) -> str | None:
    """对稳定身份事实做保守规则化兜底.

    这不是主要语义理解层；主要层是带 context 的 deletion_intent LLM。这里仅在
    LLM 没给 target_description 时处理高置信、短答案的事实问答。
    """
    if not (last_user and last_ai):
        return None

    age_match = re.search(r"(?<!\d)(\d{1,3})\s*岁", last_ai)
    if re.search(r"(我.*(多大|几岁)|年龄)", last_user) and age_match:
        return f"用户{age_match.group(1)}岁"

    if re.search(r"(我.*(叫什?么|名字)|怎么称呼我)", last_user):
        name_match = re.search(r"(?:叫|是)\s*([^，。！？?\s]{1,16})", last_ai)
        if name_match:
            return f"用户叫{name_match.group(1)}"
        answer = _clean_short_answer(last_ai)
        if answer and len(answer) <= 16:
            return f"用户叫{answer}"

    if re.search(r"(我.*(做什么|职业|工作)|我是干什么)", last_user):
        answer = _clean_short_answer(last_ai)
        if answer:
            return f"用户是{answer}"

    if re.search(r"(毕业|大学|学校)", last_user):
        school_match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]{2,30}(?:大学|学院|学校))", last_ai)
        if school_match:
            return f"用户毕业于{school_match.group(1)}"

    if re.search(r"(专业|大学学.*什么|学的.*什么)", last_user):
        major_match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]{2,30}专业)", last_ai)
        if major_match:
            return f"用户大学学的是{major_match.group(1)}"

    return None


def resolve_contextual_deletion_target(
    message: str,
    recent_context: str | None = None,
) -> str | None:
    """还原"忘了这一点吧"这类省略删除目标.

    删除链路不能只靠当前句子；用户常在 AI 刚回答某个事实后说"忘了这个"。
    这里先做低风险确定性还原，避免 LLM target_description 为空时整条删除
    intent 掉回主回复。
    """
    if not any(term in message for term in _DEICTIC_DELETION_TERMS):
        return None

    last_user = _last_prefixed_line(recent_context or "", ("用户", "User", "human"))
    last_ai = _last_prefixed_line(recent_context or "", ("AI", "assistant", "Hia"))

    structured_target = _structured_target_from_recent_qa(last_user, last_ai)
    if structured_target:
        return structured_target

    if last_ai:
        cleaned = re.sub(r"\s+", " ", last_ai).strip()
        if cleaned:
            return cleaned[:80]
    return None


async def detect_deletion_intent(
    message: str,
    recent_context: str | None = None,
) -> dict | None:
    """Detect if user wants to delete a memory.

    Returns deletion intent info or None.
    """
    # Quick keyword check
    has_keyword = any(kw in message for kw in DELETION_KEYWORDS)
    if not has_keyword:
        return None

    # Confirm with LLM
    prompt_template = _ensure_deletion_prompt_has_context(
        await get_prompt_text("memory.deletion_intent")
    )
    prompt = prompt_template.format(
        message=message,
        context=recent_context or "（无）",
    )
    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"Deletion intent detection failed: {e}")
        return None

    if not result.get("is_deletion_request", False):
        return None
    if result.get("confidence", 0) < LLM_INTENT_MIN_CONFIDENCE:
        return None

    if not result.get("target_description"):
        target = resolve_contextual_deletion_target(message, recent_context)
        if target:
            result["target_description"] = target

    return result


def _memory_to_candidate(record, *, similarity: float = 1.0) -> dict:
    return {
        "id": record.id,
        "content": record.content,
        "summary": record.summary,
        "level": record.level,
        "importance": record.importance,
        "type": record.type,
        "main_category": record.mainCategory,
        "sub_category": record.subCategory,
        "source": record.source,
        "similarity": similarity,
    }


def _literal_candidate_score(description: str, content: str) -> float:
    """给删除候选做保守字面兜底评分.

    只覆盖确定性强的事实型匹配；例如"用户28岁"和"用户年龄是28岁"。
    不用它扩大相似主题召回，避免误删。
    """
    desc = re.sub(r"\s+", "", description or "")
    text = re.sub(r"\s+", "", content or "")
    if not desc or not text:
        return 0.0
    if desc in text or text in desc:
        return 1.0

    desc_age = re.search(r"(?<!\d)(\d{1,3})岁", desc)
    text_age = re.search(r"(?<!\d)(\d{1,3})岁", text)
    if desc_age and text_age and desc_age.group(1) == text_age.group(1):
        desc_is_age = "用户" in desc and ("年龄" in desc or "岁" in desc)
        text_is_age = "用户" in text and ("年龄" in text or "岁" in text)
        if desc_is_age and text_is_age:
            return 0.99
    return 0.0


async def _find_literal_matching_memories(
    user_id: str,
    description: str,
    *,
    limit: int = 5,
) -> list[dict]:
    records = await memory_repo.find_many(
        source=None,
        where={"userId": user_id, "isArchived": False},
        take=200,
    )
    candidates: list[dict] = []
    for record in records:
        score = max(
            _literal_candidate_score(description, record.content or ""),
            _literal_candidate_score(description, record.summary or ""),
        )
        if score <= 0:
            continue
        candidates.append(_memory_to_candidate(record, similarity=score))
    candidates.sort(
        key=lambda c: (float(c.get("similarity") or 0), float(c.get("importance") or 0)),
        reverse=True,
    )
    return candidates[:limit]


async def find_matching_memories(
    user_id: str,
    description: str,
    threshold: float = 0.78,
) -> list[dict]:
    """查找匹配的记忆但不删除，返回候选列表.

    Phase 0.2 提高默认阈值 0.7 → 0.78: 0.7 太松, "忘了我喜欢咖啡" 召回了
    "喜欢茶/喜欢热饮" 等不该删的相似条目 (用户回 '嗯' 一刀切删全部 → 用户
    数据丢失). 0.78 是 bge-m3 上 "明确指代同一事实" 的近似下限, 既能召回真正
    要删的, 又能滤掉只是话题相关的.

    callers 仍可显式传 threshold 覆盖 (e.g. find_for_audit 用更松 0.6).
    """
    literal_matches = await _find_literal_matching_memories(user_id, description)

    embedding = await generate_embedding(description)
    results = await search_by_embedding(embedding, user_id, top_k=5)
    matches = list(literal_matches)
    seen_ids = {m.get("id") for m in matches}
    for r in results:
        sim = float(r.get("similarity", 0))
        if sim >= threshold and r.get("id") not in seen_ids:
            matches.append(r)
            seen_ids.add(r.get("id"))
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
    """读 pending. 兼容历史 list[dict] shape 与 Phase 5 dict shape; 后者由
    `load_pending_action` 解开. 这里只返回 candidates 列表给老调用方使用."""
    action = await load_pending_action(conversation_id)
    if not action:
        return None
    return action.get("candidates")


async def save_pending_action(
    conversation_id: str,
    *,
    action: str,
    candidates: list[dict] | None = None,
    new_time: datetime | str | None = None,
    summary: str | None = None,
) -> None:
    """统一跨消息 pending 写入. action ∈ {delete, reschedule, set_reminder}.

    - delete / reschedule: candidates 必填 (memory 候选), new_time 仅 reschedule 用
    - set_reminder: candidates 留空, summary 必填 (用户原话, 第二轮拿到时间后用它建 memory)
    """
    payload: dict = {"action": action}
    if candidates is not None:
        payload["candidates"] = candidates
    if new_time is not None:
        payload["new_time"] = (
            new_time.isoformat() if isinstance(new_time, datetime) else str(new_time)
        )
    if summary is not None:
        payload["summary"] = summary
    redis = await get_redis()
    await redis.set(
        f"{_PENDING_DELETION_PREFIX}{conversation_id}",
        json.dumps(payload, ensure_ascii=False),
        ex=_PENDING_DELETION_TTL,
    )
    logger.info(
        f"[PENDING] {action} saved n_candidates={len(candidates or [])}",
        extra={
            "event": EVT_MEMORY_DELETION_PENDING,
            "action": action,
            "n_candidates": len(candidates or []),
            "ttl_sec": _PENDING_DELETION_TTL,
        },
    )


async def load_pending_action(conversation_id: str) -> dict | None:
    """读 pending 并 normalize 成 dict shape `{action, candidates, new_time, summary}`.

    历史 list shape 视为 `{action: "delete", candidates: list}` (向后兼容).
    set_reminder 路径 candidates 为空 list, summary 持用户原话.
    """
    redis = await get_redis()
    raw = await redis.get(f"{_PENDING_DELETION_PREFIX}{conversation_id}")
    if not raw:
        return None
    try:
        data = json.loads(raw if isinstance(raw, str) else raw.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(data, list):
        return {
            "action": "delete", "candidates": data,
            "new_time": None, "summary": None,
        }
    if isinstance(data, dict):
        return {
            "action": str(data.get("action") or "delete"),
            "candidates": list(data.get("candidates") or []),
            "new_time": data.get("new_time"),
            "summary": data.get("summary"),
        }
    return None


async def apply_reschedule(
    user_id: str,
    candidates: list[dict],
    new_time: str,
    *,
    agent_id: str | None = None,
) -> int:
    """把候选 memory 的 occurTime + 对应 reminder timetrigger 全部挪到 new_time.

    - memory.occurTime ← new_time (按 candidate.source 路由 memories_user / _ai)
    - timetrigger.actionType=reminder 且 actionData.memory_id ∈ candidates → triggerTime ← new_time
    - 不删 memory, 不改 isArchived

    `agent_id` 可选但**强烈建议传入**: 多 agent 用户场景下, 不传则 trigger
    find_many 仅按 userId 过滤, 可能影响到其他 agent 的同名 memory_id 引用
    (理论上 memory_id 全局唯一所以不会真的 cross-bleed, 但加 agent 闸防御
    数据漂移). 主路径调用方 preflight 持有 agent_id, 应该传.

    candidate 必须含 'source' 字段 ('user' 或 'ai'); 缺失记 WARN 跳过, 不
    silently fallback (避免 AI memory 被错路由到 user 表).

    返回成功更新的 memory 数."""
    try:
        new_time_dt = datetime.fromisoformat(new_time)
    except (ValueError, TypeError):
        logger.warning(f"apply_reschedule: invalid new_time={new_time!r}")
        return 0

    candidate_ids = {c["id"] for c in candidates if c.get("id")}
    if not candidate_ids:
        return 0

    # 一次性拉所有 (user, agent) 的 active reminder triggers, 按 memory_id 索引.
    # Round-2 review #9: 用 services/reminder/scheduling 收口 (跟 cancel /
    # idempotency check 共享 helper). agent_id 闸防多 agent 数据漂移.
    from app.services.reminder.scheduling import find_active_reminder_triggers
    trigger_by_memory: dict = {}
    all_triggers = await find_active_reminder_triggers(
        user_id=user_id, agent_id=agent_id,
    )
    for t in all_triggers:
        mid = (t.actionData or {}).get("memory_id")
        if isinstance(mid, str) and mid in candidate_ids:
            trigger_by_memory[mid] = t

    updated = 0
    for c in candidates:
        memory_id = c.get("id")
        if not memory_id:
            continue
        # repo CRUD router 必须知道 source 才能选对表 (memories_user vs _ai).
        # candidate 来自 vector_search 的 UNION SELECT, 必带 'source' 字段;
        # 缺失则数据有问题, 不能 silently fallback 到 'user' (会误更 AI memory).
        source = c.get("source")
        if source not in ("user", "ai"):
            logger.warning(
                f"apply_reschedule: candidate {memory_id} missing/invalid source "
                f"({source!r}); skipping"
            )
            continue
        memory = await memory_repo.find_unique(memory_id)
        old_occur = memory.occurTime if memory else None
        try:
            await memory_repo.update(memory_id, source=source, occurTime=new_time_dt)
        except Exception as e:
            logger.warning(f"apply_reschedule: memory occurTime update failed {memory_id}: {e}")
            continue

        t = trigger_by_memory.get(memory_id)
        if t is not None:
            try:
                await db.timetrigger.update(
                    where={"id": t.id},  # type: ignore[attr-defined]
                    data={"triggerTime": new_time_dt},
                )
            except Exception as e:
                logger.warning(f"apply_reschedule: timetrigger update failed {memory_id}: {e}")

        try:
            await log_memory_changelog(
                user_id, memory_id, "reschedule",
                old_value=old_occur.isoformat() if old_occur else None,
                new_value=new_time_dt.isoformat(),
            )
        except Exception:
            pass
        updated += 1
    return updated


async def clear_pending_deletion(conversation_id: str) -> None:
    redis = await get_redis()
    await redis.delete(f"{_PENDING_DELETION_PREFIX}{conversation_id}")


_CONFIRM_KEYWORDS = {"对", "是", "是的", "确认", "删掉", "删吧", "好", "好的", "嗯", "ok", "yes"}
_CONFIRM_DELETE_ACTION_RE = re.compile(r"(忘掉|忘记|删掉|删除|去掉|移除|别记|不要记)")
_CONFIRM_DENY_RE = re.compile(r"(不删|别删|不要删|不用删|不删除|保留|算了|别动)")
_SELF_FORGOT_RE = re.compile(r"^我.{0,3}忘(记|了)")


def is_deletion_confirmed(user_reply: str) -> bool:
    """Check if user's reply is a confirmation to proceed with deletion."""
    reply = (user_reply or "").strip()
    if not reply:
        return False
    if reply.lower() in _CONFIRM_KEYWORDS:
        return True
    if _CONFIRM_DENY_RE.search(reply):
        return False
    if _SELF_FORGOT_RE.search(reply):
        return False
    return bool(_CONFIRM_DELETE_ACTION_RE.search(reply))


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



# ═══════════════════════════════════════════════════════════════════
# Phase 0.2: 删除 undo 机制 — 1h 内 snapshot 可恢复
#
# 当前架构无 deletedAt 字段, 删除是物理 delete + embedding cascade.
# 真正 soft delete 需要 schema migration (Phase 4 范畴), 这里用 Redis
# snapshot 提供 1h 撤销窗口: delete 前把完整 record 写 Redis, undo 时从
# Redis 还原 (重新 insert + 重新生成 embedding 异步).
#
# 局限 (vs 真 soft delete):
# - undo 后 ID 变了 (新 UUID) — 用户感知不到, 但跟外部引用 (e.g. 其他
#   memory 的 embedding 关联) 会断
# - 30 天后无法 undo (vs schema 方案的 30 天 grace)
# - Redis 失效 / 大对象超 512KB 截断风险 (mitigation: snapshot 只存关键字段)
# ═══════════════════════════════════════════════════════════════════

_DELETE_UNDO_PREFIX = "memory:delete_undo:"
_DELETE_UNDO_TTL = 3600  # 1 小时撤销窗口


async def save_delete_undo(
    *, conversation_id: str, snapshots: list[dict],
) -> None:
    """存被删 memory 的快照到 Redis, 1h 内可 undo.

    snapshots 每项: {id, userId, workspaceId, source, content, summary,
                     mainCategory, subCategory, level, importance, type,
                     occurTime, statementTime, recurrence}
    """
    from datetime import datetime, UTC
    redis = await get_redis()
    payload = {
        "snapshots": snapshots,
        "deleted_at": datetime.now(UTC).isoformat(),
    }
    await redis.set(
        f"{_DELETE_UNDO_PREFIX}{conversation_id}",
        json.dumps(payload, ensure_ascii=False, default=str),
        ex=_DELETE_UNDO_TTL,
    )


async def load_delete_undo(conversation_id: str) -> dict | None:
    """读 undo state."""
    redis = await get_redis()
    raw = await redis.get(f"{_DELETE_UNDO_PREFIX}{conversation_id}")
    if not raw:
        return None
    try:
        data = json.loads(raw if isinstance(raw, str) else raw.decode())
        if isinstance(data, dict) and isinstance(data.get("snapshots"), list):
            return data
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return None


async def clear_delete_undo(conversation_id: str) -> None:
    redis = await get_redis()
    await redis.delete(f"{_DELETE_UNDO_PREFIX}{conversation_id}")


def _snapshot_memory(record) -> dict:
    """提取 memory 关键字段做 snapshot, 给 undo 用. 排除会变的字段
    (createdAt/updatedAt/mentionCount). embedding 不在 snapshot 里 — undo
    时由 store_memory 重新生成 (cost ~50ms/条).
    """
    from datetime import datetime
    def _iso(dt):
        return dt.isoformat() if isinstance(dt, datetime) else None

    return {
        "id": record.id,
        "userId": record.userId,
        "workspaceId": getattr(record, "workspaceId", None),
        "source": record.source,
        "content": record.content,
        "summary": record.summary,
        "type": record.type,
        "mainCategory": getattr(record, "mainCategory", None),
        "subCategory": getattr(record, "subCategory", None),
        "level": record.level,
        "importance": record.importance,
        "isArchived": record.isArchived,
        "occurTime": _iso(getattr(record, "occurTime", None)),
        "statementTime": _iso(getattr(record, "statementTime", None)),
        "recurrence": getattr(record, "recurrence", None),
    }


async def restore_deleted_memories(snapshots: list[dict]) -> int:
    """从 snapshot 恢复被删 memory. 返成功数.

    重新 store_memory + 重新生成 embedding (异步). ID 会变 (新 UUID),
    但内容 + 元数据完整保留.
    """
    from app.services.memory.storage.persistence import store_memory
    from datetime import datetime

    def _parse_dt(s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    restored = 0
    for snap in snapshots:
        try:
            new_id = await store_memory(
                user_id=snap["userId"],
                content=snap["content"],
                summary=snap.get("summary") or snap["content"],
                level=snap.get("level", 3),
                importance=snap.get("importance", 0.5),
                memory_type=snap.get("type", "life"),
                main_category=snap.get("mainCategory"),
                sub_category=snap.get("subCategory"),
                source=snap.get("source", "user"),
                workspace_id=snap.get("workspaceId"),
                occur_time=_parse_dt(snap.get("occurTime")),
                statement_time=_parse_dt(snap.get("statementTime")),
                recurrence=snap.get("recurrence"),
            )
            if new_id:
                restored += 1
                logger.info(
                    f"[DELETE-UNDO] restored memory '{snap['content'][:40]}' "
                    f"as new_id={new_id[:8]} (was {snap.get('id', '?')[:8]})"
                )
        except Exception as e:
            logger.warning(
                f"[DELETE-UNDO] restore failed for '{snap.get('content', '')[:40]}': {e}"
            )
    return restored


async def execute_confirmed_deletion(
    user_id: str,
    candidates: list[dict],
    *,
    conversation_id: str | None = None,
) -> int:
    """Spec §5.3-5.4: execute physical deletion after confirmation.

    Phase 0.2: 删除前对每条 record 做完整 snapshot 存 Redis (1h TTL),
    用户说"撤回刚才的删除" 在 1h 内可全部 undo 还原. snapshot 包含所有
    关键字段, undo 走 store_memory 重新插入 (新 UUID, 内容完整).
    """
    deleted = 0
    snapshots: list[dict] = []
    for c in candidates:
        memory_id = c.get("id")
        if not memory_id:
            continue
        memory = await memory_repo.find_unique(memory_id)
        if memory:
            # snapshot 在 delete 前抓 — delete 后 record 已 gone
            snapshots.append(_snapshot_memory(memory))
            await log_memory_changelog(
                user_id, memory_id, "delete",
                old_value=memory.content,
            )
        try:
            await memory_repo.delete(memory_id)
            deleted += 1
        except Exception as e:
            logger.warning(f"Confirmed deletion failed for {memory_id}: {e}")

    # 存 undo state (即使部分 delete 失败, 成功的那部分仍可 undo)
    if conversation_id and snapshots:
        try:
            await save_delete_undo(
                conversation_id=conversation_id, snapshots=snapshots,
            )
        except Exception as e:
            logger.warning(f"[DELETE] save undo state failed: {e}")

    logger.info(
        f"execute_confirmed_deletion: {deleted}/{len(candidates)} memories deleted "
        f"(undo snapshot saved={bool(snapshots and conversation_id)})",
        extra={
            "event": EVT_MEMORY_DELETED,
            "n_candidates": len(candidates),
            "n_deleted": deleted,
        },
    )
    return deleted
