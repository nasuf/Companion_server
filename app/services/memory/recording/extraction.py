"""Memory extraction service.

Spec 第二部分 §2.1 (用户侧) / §2.2 (AI 侧)：两条独立管线，**输入侧决定归属**，
不再让 LLM 从混合对话里推断 owner。由 `side="user"` / `side="ai"` 选择对应的
指令模版（memory.extraction_user / memory.extraction_ai）。

Output: {memories, entities, preferences, topics}
"""

import logging
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from app.config import settings
from app.observability.events import EVT_LLM_FAIL, EVT_MEMORY_EXTRACTED
from app.services.llm.models import get_chat_model, invoke_json
from app.services.memory.taxonomy import TAXONOMY_MATRIX
from app.services.prompting.store import PromptDisabledError, get_prompt_text

logger = logging.getLogger(__name__)

Side = Literal["user", "ai"]

_PROMPT_KEY_BY_SIDE: dict[Side, str] = {
    "user": "memory.extraction_user",
    "ai": "memory.extraction_ai",
}


def _taxonomy_list_text(side: Side) -> str:
    """Render the L1 taxonomy bullets for the given side's extraction prompt.

    Always uses the L1 superset — extraction assigns a tentative level and
    the downstream pipeline will demote/reject if the (source, level, main)
    slice turns out to forbid the sub-category.
    """
    return "\n".join(
        f"- {main}：{'、'.join(subs)}"
        for main, subs in TAXONOMY_MATRIX[side][1].items()
    )


async def extract_memories(
    new_conversation: str,
    *,
    context_conversation: str = "",
    side: Side = "user",
) -> dict:
    """Extract structured memory from a conversation string for one side.

    Args:
        new_conversation: Dialogue lines to extract from (`user:` / `assistant:`
            prefixed). LLM must only output memories sourced from this block.
        context_conversation: Prior dialogue lines for disambiguation (指代 /
            情境); LLM instructed NOT to extract from this block. Empty string
            when the pipeline has no history pre-watermark.
        side: Which speaker's memories to extract ("user" | "ai"). The owner
            is determined by this path, not inferred by the LLM.

    Returns dict with keys: memories, entities, preferences, topics.
    """
    model = get_chat_model()  # spec §2.1.3 / §2.2.3: 大模型精细处理
    now = datetime.now(ZoneInfo(settings.schedule_timezone))
    try:
        tpl = await get_prompt_text(_PROMPT_KEY_BY_SIDE[side])
    except PromptDisabledError:
        # admin 停用抽取模板 → 本条消息不抽记忆, 但不能让异常打断整条
        # _bg_memory_pipeline (否则时间解析/changelog/提醒 timetrigger 全部连坐).
        logger.info(f"memory extraction prompt disabled, skipping side={side}")
        return {"memories": [], "entities": [], "preferences": [], "topics": []}
    prompt = tpl.format(
        new_conversation=new_conversation,
        context_conversation=context_conversation or "(无)",
        current_time=now.strftime("%Y-%m-%d %H:%M %A"),
        taxonomy_list=_taxonomy_list_text(side),
    )

    import time as _time
    t0 = _time.perf_counter()
    try:
        # 用 memory_extract profile (streaming + idle_timeout 保护) 而非默认
        # chat_extract (45s 总 timeout). 解决用户一次发"画像 dump" 20+ 条事实
        # 时 LLM 输出 1500-2500 tokens 接近 chat_extract 上限 → wait_for cancel
        # → langsmith pending → silent failure 全丢. 生产 trace 019dd808.
        #
        # ⚠️ profile 必须传字符串 NAME, 不要传 get_profile(...) 拿到的 CallProfile
        # 对象 — invoke_json 内部 `_provider_and_profile` 自己会再调 get_profile,
        # 误传对象会变成 get_profile(get_profile("...")) → KeyError(CallProfile(...))
        # → except 吃掉返空 dict → DB 0 条. 生产 trace 019e0004 (2026-05-07).
        result = await invoke_json(model, prompt, profile="memory_extract")
        # Validate structure
        result.setdefault("memories", [])
        result.setdefault("entities", [])
        result.setdefault("preferences", [])
        result.setdefault("topics", [])
        sanitized_memories = []
        for mem in result["memories"]:
            if not isinstance(mem, dict):
                continue
            from app.services.memory.normalization import normalize_memory_category
            taxonomy = await normalize_memory_category(
                main_category=mem.get("main_category"),
                sub_category=mem.get("sub_category"),
                legacy_type=mem.get("type"),
                summary=mem.get("summary", ""),
            )
            mem["main_category"] = taxonomy.main_category
            mem["sub_category"] = taxonomy.sub_category
            mem["type"] = taxonomy.legacy_type
            sanitized_memories.append(mem)
        result["memories"] = sanitized_memories
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        logger.info(
            f"[MEM-EXTRACT-{side}] LLM returned {len(sanitized_memories)} memories "
            f"({elapsed_ms:.0f}ms)",
            extra={
                "event": EVT_MEMORY_EXTRACTED,
                "side": side, "n_extracted": len(sanitized_memories),
                "n_entities": len(result.get("entities", [])),
                "n_topics": len(result.get("topics", [])),
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )
        return result
    except Exception as e:
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        logger.error(
            f"Memory extraction failed: {e}",
            extra={"event": EVT_LLM_FAIL, "stage": "memory_extract", "side": side,
                   "error_type": type(e).__name__, "elapsed_ms": round(elapsed_ms, 1)},
        )
        return {"memories": [], "entities": [], "preferences": [], "topics": []}
