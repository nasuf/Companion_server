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
from app.services.memory.storage.persistence import store_memory
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
            **pad_params(user_emotion),
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

    - 变化/错误: demote old L1 → L2 + 写新 L1 (复用老条目类目)
    - 新增: 老 L1 不动 + 写新 L1/L2 (用 LLM 给的新类目)

    新条目入库逻辑: contradiction_analysis prompt 输出 new_memory 文本 +
    new_memory_main/sub_category 类目. 之前注释说"由 extraction pipeline
    创建", 但 extraction 跑在用户的"确认"消息上 (e.g. "说错了之前"), 文本
    不含新事实抽不到. 这里直接调 store_memory 写新条目, 走正常 embedding +
    dedup + taxonomy resolve (LLM 给的类目即便有偏差, taxonomy.SUBCATEGORY_
    ALIASES + fuzzy match 会 canonicalize).
    """
    change_type = analysis.get("change_type", "新增")
    old_id = conflict.get("conflicting_memory_id")
    new_memory_text = (analysis.get("new_memory") or "").strip()
    new_main = (analysis.get("new_memory_main_category") or "").strip()
    new_sub = (analysis.get("new_memory_sub_category") or "").strip()

    # Step 1: demote old L1 → L2 (变化/错误 才降; 新增 老条目保留)
    old_mem = None
    if change_type in ("变化", "错误") and old_id:
        old_mem = await memory_repo.find_unique(old_id)
        if not old_mem:
            return  # 老条目已被删/找不到, 静默退出

        new_imp = max(0.30, (old_mem.importance or 0.5) - 0.20)
        await memory_repo.update(
            old_id,
            source=getattr(old_mem, "source", "user"),
            record=old_mem,
            level=2,
            importance=new_imp,
        )
        logger.info(
            f"contradiction.apply: demoted {old_id[:8]} L1→L2 ({change_type})",
            extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "apply",
                   "change_type": change_type, "old_memory_id": old_id,
                   "new_importance": new_imp},
        )
    elif change_type == "新增" and old_id:
        # 新增 case: 不降级老条目, 但需要拿老条目作 user/workspace/source 上下文
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
        # 不该到这里 (上面 change_type=变化/错误 时 old_mem=None 已经 return),
        # 但 change_type 异常值时兜底, 防止后续 getattr 崩.
        logger.warning(
            f"contradiction.apply: cannot write new_memory without old_mem context "
            f"(change_type={change_type}, old_id={old_id})",
            extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "apply",
                   "change_type": change_type, "outcome": "no_old_context"},
        )
        return

    # 类目策略:
    # - 变化/错误: 优先 LLM 给的类目, fallback 复用老条目 (同一属性, 老的肯定对)
    # - 新增: 用 LLM 给的类目 (新维度, 老的可能完全不适用); 没给就 fallback 老的,
    #   resolve_taxonomy 至少能 canonicalize 到合法 sub
    main_category = new_main or getattr(old_mem, "mainCategory", None)
    sub_category = new_sub or getattr(old_mem, "subCategory", None)

    try:
        new_id = await store_memory(
            user_id=old_mem.userId,
            content=new_memory_text,
            summary=new_memory_text,
            level=1,
            importance=0.95,
            main_category=main_category,
            sub_category=sub_category,
            source=getattr(old_mem, "source", "user"),
            workspace_id=getattr(old_mem, "workspaceId", None),
        )
        if new_id:
            logger.info(
                f"contradiction.apply: wrote new L1 {new_id[:8]} "
                f"({change_type}) cat={main_category}/{sub_category} "
                f"text='{new_memory_text[:40]}'",
                extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "apply",
                       "change_type": change_type, "new_memory_id": new_id,
                       "new_main": main_category, "new_sub": sub_category,
                       "new_memory_text_len": len(new_memory_text)},
            )
        else:
            # store_memory 返回 None: dedup 命中 (sim ≥ 0.9) 或 taxonomy 不允许.
            # 矛盾解决场景属异常 — 用户已主动纠正, 应该入库.
            logger.warning(
                f"contradiction.apply: store_memory returned None for "
                f"new_memory='{new_memory_text[:40]}' (likely dedup or "
                f"taxonomy block); 新条目未入库",
                extra={"event": EVT_MEMORY_CONTRADICTION_STEP, "step": "apply",
                       "change_type": change_type, "outcome": "new_l1_blocked",
                       "new_main": main_category, "new_sub": sub_category,
                       "new_memory_text_len": len(new_memory_text)},
            )
    except Exception as e:
        # 不让写入失败炸掉 resolution 流程 — 老条目状态已确定, 对话可继续.
        # 新条目缺失是 degraded-but-recoverable (用户下次提及会走正常 extraction).
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
