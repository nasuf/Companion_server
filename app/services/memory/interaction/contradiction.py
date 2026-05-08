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
import re

from app.observability.events import (
    EVT_LLM_FAIL,
    EVT_MEMORY_CONTRADICTION_STEP,
)
from app.redis_client import get_redis
from app.services.llm.models import (
    get_chat_model,
    get_utility_model,
    invoke_json,
    invoke_text,
)
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.persistence import log_memory_changelog, store_memory
from app.services.prompting.store import get_prompt_text
from app.services.prompting.utils import SafeDict

_PENDING_KEY_PREFIX = "contradiction:pending:"
_PENDING_TTL = 1800  # 30 min — generous window for user to reply

logger = logging.getLogger(__name__)


_CONFLICT_META_WORD_RE = re.compile(
    r"用户|本人|自己|对方|当前|刚才|新信息|提到|提及|表示|透露|声称|"
    r"称|说|为|是|的|了"
)


def _normalize_conflict_grounding_text(text: str) -> str:
    """Normalize role/meta wording while preserving literal fact content."""
    compact = re.sub(r"[\s，。！？!?~～…,.、:：；;\"'“”‘’（）()【】\[\]{}]+", "", text or "")
    compact = _CONFLICT_META_WORD_RE.sub("", compact)
    return compact.lower()


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _conflict_new_info_grounded(result: dict, user_message: str) -> bool:
    """Return true only if reported new_info is supported by current message.

    The detector compares current user message against all L1 memories. Small
    models sometimes report conflicts between two old L1 memories and place one
    of those old memories in `new_info`. That must not trigger the interactive
    contradiction flow.
    """
    new_info = str(result.get("new_info") or "").strip()
    if not new_info:
        return False
    user_text = _normalize_conflict_grounding_text(user_message)
    new_text = _normalize_conflict_grounding_text(new_info)
    if not user_text or not new_text:
        return False
    if new_text in user_text:
        return True

    new_bigrams = _char_bigrams(new_text)
    user_bigrams = _char_bigrams(user_text)
    if not new_bigrams:
        return False
    overlap = len(new_bigrams & user_bigrams) / len(new_bigrams)
    return overlap >= 0.6


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
        logger.debug(
            "contradiction.detect: no L1 user memory to compare against",
            extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "detect", "outcome": "no_l1"},
        )
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
            if not _conflict_new_info_grounded(result, user_message):
                logger.info(
                    "contradiction.detect: discarded ungrounded conflict",
                    extra={
                        "event": EVT_MEMORY_CONTRADICTION_STEP, "step": "detect",
                        "outcome": "ungrounded_conflict",
                        "n_l1_checked": len(l1_user),
                        "old_memory_id": result.get("conflicting_memory_id"),
                    },
                )
                return None
            logger.info(
                f"contradiction.detect: conflict found in {len(l1_user)} L1",
                extra={
                    "event": EVT_MEMORY_CONTRADICTION_STEP, "step": "detect",
                    "outcome": "conflict", "n_l1_checked": len(l1_user),
                    "old_memory_id": result.get("conflicting_memory_id"),
                },
            )
            return result
        logger.debug(
            f"contradiction.detect: no conflict (checked {len(l1_user)} L1)",
            extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "detect",
                   "outcome": "no_conflict", "n_l1_checked": len(l1_user)},
        )
        return None
    except Exception as e:
        logger.warning(
            f"L1 contradiction detection failed: {e}",
            extra={"event": EVT_LLM_FAIL, "stage": "contradiction_detect",
                   "error_type": type(e).__name__},
        )
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
        }
        prompt = template.format_map(SafeDict(params))
        inquiry = (await invoke_text(get_chat_model(), prompt)).strip()
        logger.info(
            f"contradiction.inquiry generated len={len(inquiry)}",
            extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "inquiry",
                   "inquiry_len": len(inquiry)},
        )
        return inquiry
    except Exception as e:
        logger.warning(
            f"Contradiction inquiry generation failed: {e}",
            extra={"event": EVT_LLM_FAIL, "stage": "contradiction_inquiry",
                   "error_type": type(e).__name__},
        )
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
        if isinstance(result, dict):
            logger.info(
                f"contradiction.analyze: change_type={result.get('change_type')}",
                extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "analyze",
                       "change_type": result.get("change_type"),
                       "reason": (result.get("reason") or "")[:60]},
            )
            return result
        return {"change_type": "新增", "reason": "解析失败"}
    except Exception as e:
        logger.warning(
            f"Contradiction response analysis failed: {e}",
            extra={"event": EVT_LLM_FAIL, "stage": "contradiction_analyze",
                   "error_type": type(e).__name__},
        )
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
        }
        prompt = template.format_map(SafeDict(params))
        return (await invoke_text(get_chat_model(), prompt)).strip()
    except Exception as e:
        logger.warning(f"Contradiction reply generation failed: {e}")
        return "好的，我记住了~"


# Phase 0.3: 矛盾解决 importance 差异化 + 老条目 archive (而非 demote).
#
# 历史 bug: 所有 change_type 都把新条目硬编 importance=0.95 (L1). 用户口误/
# 玩笑/被诱导 → 错信息直接污染 L1 永久. 老条目仅 demote 到 L2 (importance
# -0.20) 留在检索通路 → "AI 时而说苏州时而说上海" 的双重污染.
#
# 修复:
# - 变化 (生活演进, 搬家/换工作): new_imp = max(L1 base, old_imp - 0.05) →
#   保留 L1 级别 (用户正常生活更新, 直接接受); 老条目 archive (从检索消失)
# - 错误 (用户纠正过去说错的): new_imp = 0.7 (L2) → 不直接 L1, 等用户后续
#   反复提到/L2 dynamics 自然 promote (防止口误/玩笑污染 L1); 老条目 archive
# - 新增 (新维度无冲突): new_imp = 0.85 (L1, 跟 spec 一致); 老条目不动 (无冲突)
#
# Archive (isArchived=True) vs demote (level降): archive 完全从 retrieval
# 移除, 防双重事实污染. changelog 保留旧内容供审计/未来 rollback.

# 矛盾解决新条目的 importance 默认值, 按 change_type 分.
_CONTRADICTION_NEW_IMP_DEFAULT: dict[str, float] = {
    "变化": 0.85,  # 用户正常生活更新, 直接 L1
    "错误": 0.70,  # 用户纠正过去说错的, L2 等自然 promote (防口误污染)
    "新增": 0.85,  # 新维度无冲突, 走 L1
}


def _resolve_new_importance(change_type: str, old_importance: float) -> float:
    """根据 change_type 计算新条目 importance.

    - 变化: max(0.85, old - 0.05) — 沿用 L1 级别, 微降 0.05 标记是 transition
    - 错误: 0.70 (L2 区间) — 等用户反复提到自然 promote, 防口误/玩笑污染 L1
    - 新增 / 其他: 0.85 (L1 base)
    """
    base = _CONTRADICTION_NEW_IMP_DEFAULT.get(change_type, 0.85)
    if change_type == "变化":
        return max(base, old_importance - 0.05)
    return base


async def apply_contradiction_resolution(
    conflict: dict,
    analysis: dict,
) -> None:
    """Spec §4.4 + Phase 0.3 修复: differentiate importance by change_type.

    - 变化: archive 老条目 + 写新 L1 (importance ≈ old_imp, 接受演进)
    - 错误: archive 老条目 + 写 L2 (importance=0.70, 不直接 L1, 防口误污染)
    - 新增: 老条目不动 + 写新 L1 (importance=0.85, 新维度无冲突)

    新条目入库走 store_memory 正常路径 (embedding + dedup + taxonomy).
    """
    change_type = analysis.get("change_type", "新增")
    old_id = conflict.get("conflicting_memory_id")
    new_memory_text = (analysis.get("new_memory") or "").strip()
    new_main = (analysis.get("new_memory_main_category") or "").strip()
    new_sub = (analysis.get("new_memory_sub_category") or "").strip()

    # Step 1: 处理老条目 — archive (变化/错误) 或保留 (新增)
    old_mem = None
    if change_type in ("变化", "错误") and old_id:
        old_mem = await memory_repo.find_unique(old_id)
        if not old_mem:
            return  # 老条目已被删/找不到, 静默退出

        # archive 而非 demote — 防止旧错事实留在检索通路造成"双重事实"污染
        await memory_repo.update(
            old_id,
            source=getattr(old_mem, "source", "user"),
            record=old_mem,
            isArchived=True,
        )
        # audit log: 记录 contradiction-driven archive (跟正常 archive 区分)
        try:
            await log_memory_changelog(
                old_mem.userId, old_id,
                operation="contradiction_archived",
                old_value=old_mem.content,
                new_value=f"superseded by user correction (change_type={change_type})",
                workspace_id=getattr(old_mem, "workspaceId", None),
            )
        except Exception:
            pass  # audit 失败不阻塞主流程
        logger.info(
            f"contradiction.apply: archived {old_id[:8]} ({change_type}); "
            f"old content was '{(old_mem.content or '')[:40]}'",
            extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "apply",
                   "change_type": change_type, "old_memory_id": old_id,
                   "outcome": "archived"},
        )
    elif change_type == "新增" and old_id:
        # 新增 case: 不动老条目, 但需要老条目作 user/workspace/source 上下文
        old_mem = await memory_repo.find_unique(old_id)

    # Step 2: 写新条目 (3 个 change_type 都可能产生 new_memory)
    if not new_memory_text:
        logger.debug(
            f"contradiction.apply: no new_memory to write (change_type={change_type})",
            extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "apply",
                   "change_type": change_type, "outcome": "no_new"},
        )
        return

    if not old_mem:
        logger.warning(
            f"contradiction.apply: cannot write new_memory without old_mem context "
            f"(change_type={change_type}, old_id={old_id})",
            extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "apply",
                   "change_type": change_type, "outcome": "no_old_context"},
        )
        return

    # 类目策略 (沿用之前):
    # - 变化/错误: 优先 LLM 给的类目, fallback 复用老条目 (同一属性, 老的肯定对)
    # - 新增: 用 LLM 给的类目 (新维度, 老的可能完全不适用); 没给就 fallback 老的
    main_category = new_main or getattr(old_mem, "mainCategory", None)
    sub_category = new_sub or getattr(old_mem, "subCategory", None)

    # Phase 0.3: importance 按 change_type 分级 (不再硬编 0.95)
    new_imp = _resolve_new_importance(
        change_type, old_importance=getattr(old_mem, "importance", 0.5) or 0.5,
    )
    # level 跟 importance 同步: imp >= 0.85 → L1, >= 0.5 → L2, else L3
    new_level = 1 if new_imp >= 0.85 else 2 if new_imp >= 0.50 else 3

    try:
        new_id = await store_memory(
            user_id=old_mem.userId,
            content=new_memory_text,
            summary=new_memory_text,
            level=new_level,
            importance=new_imp,
            main_category=main_category,
            sub_category=sub_category,
            source=getattr(old_mem, "source", "user"),
            workspace_id=getattr(old_mem, "workspaceId", None),
        )
        if new_id:
            # audit log: 记录 contradiction-derived 新条目 (跟普通 insert 区分)
            try:
                await log_memory_changelog(
                    old_mem.userId, new_id,
                    operation="contradiction_new",
                    new_value=(
                        f"change_type={change_type} imp={new_imp:.2f} L{new_level}; "
                        f"replaces old={old_id}"
                    ),
                    workspace_id=getattr(old_mem, "workspaceId", None),
                )
            except Exception:
                pass
            logger.info(
                f"contradiction.apply: wrote new L{new_level} {new_id[:8]} "
                f"({change_type}, imp={new_imp:.2f}) "
                f"cat={main_category}/{sub_category} text='{new_memory_text[:40]}'",
                extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "apply",
                       "change_type": change_type, "new_memory_id": new_id,
                       "new_main": main_category, "new_sub": sub_category,
                       "new_importance": new_imp, "new_level": new_level,
                       "new_memory_text_len": len(new_memory_text)},
            )
        else:
            logger.warning(
                f"contradiction.apply: store_memory returned None for "
                f"new_memory='{new_memory_text[:40]}' (likely dedup or "
                f"taxonomy block); 新条目未入库",
                extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "apply",
                       "change_type": change_type, "outcome": "new_blocked",
                       "new_main": main_category, "new_sub": sub_category,
                       "new_memory_text_len": len(new_memory_text)},
            )
    except Exception as e:
        logger.warning(
            f"contradiction.apply: failed to write new memory: {e}",
            extra={"event": EVT_LLM_FAIL, "stage": "contradiction_apply_new",
                   "change_type": change_type, "error_type": type(e).__name__},
        )


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
