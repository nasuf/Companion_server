"""Async memory pipeline.

Orchestrates: extract -> score -> dedup -> store -> embed -> entity link.
Runs as FastAPI BackgroundTasks (non-blocking).

Spec 第二部分 §2.1 / §2.2：用户侧与 AI 侧各走一条独立管线，`side` 决定抽取 prompt 与
存储归属（B 库 / A 库）。owner 不再由 LLM 推断。

L1 冲突处理完全走 spec §4 交互矛盾机制（热路径询问用户确认）—
录入期不自动降级/修改 L1，避免绕过用户确认（spec §1.5.1）。
"""

import logging
from datetime import datetime
from typing import Literal

from app.services.memory.storage.entity_repo import (
    record_entities_for_memory,
    record_preferences_for_memory,
    record_topics_for_memory,
)
from app.services.memory.recording.extraction import extract_memories
from app.services.memory.recording.filter import should_extract_memory
from app.services.memory.recording.pre_filter import should_memorize
from app.services.memory.storage.persistence import store_memory, log_memory_changelog
from app.services.schedule_domain.time_parser import (
    has_explicit_time,
    parse_with_statement_time,
)
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

Side = Literal["user", "ai"]

# Spec §1.5.2: keywords indicating user expressed that information is important.
# Detected once per pipeline call (on new_conversation), tagged on all extracted
# memories so L2→L1 promotion can verify the condition.
_IMPORTANCE_EXPRESSIONS = (
    "很重要", "一定要记住", "千万别忘", "记住了", "别忘了",
    "这很关键", "非常重要", "特别重要", "务必记住",
)


async def process_memory_pipeline(
    user_id: str,
    new_conversation: str,
    *,
    context_conversation: str = "",
    statement_time: datetime | None = None,
    side: Side = "user",
) -> list[str]:
    """Run the full memory extraction and storage pipeline for one side.

    Args:
        new_conversation: Dialogue to extract from (post-watermark messages).
            Pre-filter + heuristic filter run against this block only.
        context_conversation: Prior dialogue for LLM disambiguation only (no
            extraction). Empty string when no pre-watermark history.
        side: "user" → spec §2.1，抽取用户记忆，存入 memories_user；
              "ai"   → spec §2.2，抽取 AI 自我记忆，存入 memories_ai。
        statement_time: Part 5 §3.1 说出这句话的时间（消息接收时刻）。
            未提供时由 store_memory 取 now() 作为最佳估计。

    Returns list of stored memory IDs.
    """
    workspace_id = await resolve_workspace_id(user_id=user_id)

    # Step 0: Heuristic filter — skip purely noise messages (no LLM call)
    # 只对 new_conversation 判定, 历史上下文已经抽过了
    if not should_extract_memory(new_conversation):
        logger.debug(f"[MEM-{side}] filtered by heuristic filter")
        return []

    # Step 1 (spec §2.1.2 / §2.2.2): Small model pre-filter — "记" or "不记"
    # Uses registered prompt `memory.judgement_{side}` via should_memorize.
    from app.config import settings
    if settings.enable_memory_prefilter:
        try:
            if not await should_memorize(new_conversation, side=side):
                logger.debug(f"[MEM-{side}] pre-filter: 不记")
                return []
        except Exception as e:
            logger.warning(f"[MEM-{side}] pre-filter failed ({e}), proceeding")

    # Step 2 (spec §2.1.3 / §2.2.3): Big model extraction
    extraction = await extract_memories(
        new_conversation,
        context_conversation=context_conversation,
        side=side,
    )
    memories = extraction.get("memories", [])

    if not memories:
        logger.info(f"[MEM-{side}] no memories extracted")
        return []

    # Spec Part 5 §3.1: 解析过程不调大模型。规则引擎作为权威源；只有当
    # 消息里唯一匹配到 1 个时间表述时才覆盖 LLM 的 occur_time — 否则
    # 无法把单个时间归因到多条记忆中的哪一条，回退到 LLM 抽取字段。
    rule_based_occur_time: datetime | None = None
    if has_explicit_time(new_conversation):
        parsed = parse_with_statement_time(new_conversation, now=statement_time)
        if len(parsed.event_times) == 1:
            rule_based_occur_time = parsed.event_times[0].start

    stored_ids: list[str] = []

    # Spec §1.5.2 user emphasis only drives user-side L2→L1 promotion.
    user_emphasized = side == "user" and any(
        kw in new_conversation for kw in _IMPORTANCE_EXPRESSIONS
    )

    # Step 3: Store each memory with dedup and conflict check
    for mem in memories:
        summary = mem.get("summary", "")
        importance = mem.get("importance", 0.5)
        memory_type = mem.get("type")
        main_category = mem.get("main_category")
        sub_category = mem.get("sub_category")

        # Per spec《产品手册·背景信息》§2.3 — level is derived from importance
        # score (0-100), not whatever level the LLM may have guessed:
        #   ≥ 0.85 → L1   |   0.50-0.84 → L2   |   0.10-0.49 → L3   |   < 0.10 → drop
        if importance < 0.10:
            logger.debug(f"Memory dropped (importance={importance:.2f} < 0.10): {summary[:40]}")
            continue
        elif importance >= 0.85:
            level = 1
        elif importance >= 0.50:
            level = 2
        else:
            level = 3

        # Rule engine wins when it's unambiguous; LLM occur_time is fallback.
        occur_time: datetime | None = rule_based_occur_time
        if occur_time is None:
            raw_time = mem.get("occur_time")
            if raw_time and isinstance(raw_time, str):
                try:
                    occur_time = datetime.fromisoformat(raw_time)
                except ValueError:
                    pass

        # Adjust importance based on emotion
        emotion = mem.get("emotion")
        if emotion:
            pleasure_abs = abs(emotion.get("pleasure", 0.0))
            importance = min(1.0, importance + pleasure_abs * 0.2)

        memory_id = await store_memory(
            user_id=user_id,
            content=summary,
            summary=summary,
            level=level,
            importance=importance,
            memory_type=memory_type,
            main_category=main_category,
            sub_category=sub_category,
            occur_time=occur_time,
            statement_time=statement_time,
            workspace_id=workspace_id,
            source=side,
        )

        if memory_id:
            stored_ids.append(memory_id)
            # Log user emphasis so L2→L1 promotion can verify the condition
            if user_emphasized:
                try:
                    await log_memory_changelog(
                        user_id, memory_id, "user_emphasized",
                        new_value="用户表达了该信息的重要性",
                        workspace_id=workspace_id,
                    )
                except Exception:
                    pass

            # Step 3: Link entities / topics / preferences to this memory.
            # Best-effort: failure here is advisory (retrieval still works
            # from the memory row + pgvector) so we log-and-continue.
            try:
                await record_entities_for_memory(
                    memory_id=memory_id,
                    memory_source=side,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    entities=extraction.get("entities", []),
                )
                await record_topics_for_memory(
                    memory_id=memory_id,
                    memory_source=side,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    topics=extraction.get("topics", []),
                )
                await record_preferences_for_memory(
                    memory_id=memory_id,
                    memory_source=side,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    preferences=extraction.get("preferences", []),
                )
            except Exception as e:
                logger.warning(f"Entity linking failed for memory {memory_id}: {e}")

    logger.info(f"Pipeline complete: {len(stored_ids)}/{len(memories)} memories stored")
    return stored_ids
