"""Memory storage service.

Stores memories to PostgreSQL with deduplication (cosine > 0.9).
Classifies into L1/L2/L3 levels.
"""

import logging
from datetime import datetime, timezone

from app.db import db
from app.services.memory.storage import repo as memory_repo
from app.services.memory.config import DEDUP_THRESHOLD
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.taxonomy import is_singleton, resolve_taxonomy
from app.services.memory.retrieval.vector_search import search_by_embedding
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

# Map legacy/mixed type values to the standard English enum
_TYPE_NORMALIZE_MAP: dict[str, str] = {
    # Standard values (passthrough)
    "identity": "identity",
    "emotion": "emotion",
    "preference": "preference",
    "life": "life",
    "thought": "thought",
    "consolidated": "consolidated",
    # Old English values
    "event": "life",
    "relationship": "life",
    "fact": "identity",
    # Old Chinese values (from self_memory)
    "感受": "emotion",
    "体验": "life",
    "思考": "thought",
    "生活": "life",
    "关系": "life",
    # System types
    "compressed": "consolidated",
}


def normalize_memory_type(memory_type: str | None) -> str | None:
    """Normalize a memory type value to the standard English enum.

    Standard types: identity, emotion, preference, life, thought, consolidated.
    Returns None if input is None, or the mapped value (original value if no mapping found).
    """
    if memory_type is None:
        return None
    return _TYPE_NORMALIZE_MAP.get(memory_type, memory_type)


def _normalize_singleton_text(text: str | None) -> str:
    """Normalize a singleton fact for cheap same-fact blocking.

    This is not semantic understanding. The semantic signal is the structured
    taxonomy tuple `(source, mainCategory, subCategory, level)`. Normalization
    only avoids churn when the extractor emits the exact same singleton string
    with minor punctuation or whitespace differences.
    """
    if not text:
        return ""
    return "".join(ch for ch in text if not ch.isspace() and ch not in "，。！？?~～,.、:：；;")


async def log_memory_changelog(
    user_id: str,
    memory_id: str,
    operation: str,
    old_value: str | None = None,
    new_value: str | None = None,
    workspace_id: str | None = None,
) -> None:
    """Write a memory changelog entry for portrait generation."""
    try:
        workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)
        await db.memorychangelog.create(
            data={
                "userId": user_id,
                "memoryId": memory_id,
                "operation": operation,
                "oldValue": old_value,
                "newValue": new_value,
                "workspaceId": workspace_id,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to write changelog: {e}")

async def find_duplicate_id(
    user_id: str,
    content: str,
    embedding: list[float],
    workspace_id: str | None = None,
) -> str | None:
    """If a near-duplicate memory already exists (cosine > DEDUP_THRESHOLD),
    return its id. Otherwise None.

    Phase 3.1: 极性校验防反义误判. bge-m3 反义对 cosine 0.84-0.89, 容易超
    DEDUP_THRESHOLD=0.85 → 后写的"我不住北京"被当成已存"我住北京"的重复 →
    数据丢失. 加 polarity 校验: 极性不一致 → 不算重复, 都存.

    用 case: RECORD_REQUEST handler dedup 命中时需要拿到 existing memory id,
    才能 update occurTime + 重建 timetrigger (旧 trigger 已 fired 完, 新一次
    的"提醒我X"必须建新 trigger).
    """
    from app.services.memory.polarity import semantic_conflict_reasons

    results = await search_by_embedding(embedding, user_id, top_k=5, workspace_id=workspace_id)
    for r in results:
        sim = r.get("similarity", 0)
        if isinstance(sim, str):
            sim = float(sim)
        if sim > DEDUP_THRESHOLD:
            mid = r.get("id")
            if not mid:
                continue
            # Phase 3.1/3.B: 极性/语义对立校验 — 反义对不算重复
            cand_text = r.get("summary") or r.get("content") or ""
            conflict_reasons = semantic_conflict_reasons(content, cand_text)
            if conflict_reasons:
                logger.info(
                    f"Dedup semantic conflict ({','.join(conflict_reasons)}, "
                    f"sim={sim:.3f}): "
                    f"new='{content[:30]}' vs existing='{cand_text[:30]}'; "
                    f"NOT treating as duplicate (saves both)"
                )
                continue  # 反义对, 检查下一个 candidate
            logger.info(
                f"Duplicate memory detected (similarity={sim:.3f}, "
                f"matched_id={str(mid)[:8]}): {content[:50]}"
            )
            return str(mid)
    return None


async def is_duplicate(
    user_id: str,
    content: str,
    embedding: list[float],
    workspace_id: str | None = None,
) -> bool:
    """向后兼容 wrapper. 新调用方应直接用 find_duplicate_id 拿 id."""
    matched = await find_duplicate_id(user_id, content, embedding, workspace_id=workspace_id)
    return matched is not None


async def store_memory(
    user_id: str,
    content: str,
    summary: str | None = None,
    level: int = 3,
    importance: float = 0.5,
    memory_type: str | None = None,
    main_category: str | None = None,
    sub_category: str | None = None,
    source: str = "user",
    occur_time: datetime | None = None,
    statement_time: datetime | None = None,
    workspace_id: str | None = None,
    recurrence: str | None = None,
) -> str | None:
    """Store a memory with deduplication.

    Returns memory_id if stored, None if duplicate.
    Args:
        source: "user" for memories about the user, "ai" for AI self-memories.
        recurrence: Part 5 §4.2 提醒重复规则 (once|yearly|monthly|weekly|daily).
            仅 sub_category="提醒" 时有效, 其他子类传 None.
    """
    # Source narrows to the literal Source type expected by the taxonomy
    repo_source = "ai" if source == "ai" else "user"
    taxonomy = resolve_taxonomy(
        main_category=main_category,
        sub_category=sub_category,
        legacy_type=normalize_memory_type(memory_type),
        source=repo_source,
        level=level,
    )
    if not taxonomy.allowed:
        # The (source, level, main) combination is forbidden by the spec —
        # e.g. trying to write an AI 身份/偏好/思维 memory at L2 or L3.
        # Refuse rather than silently mis-categorize. Caller can retry at
        # a different level (typically L1) if appropriate.
        logger.info(
            f"Refusing memory: ({repo_source}, L{level}, {taxonomy.main_category}) "
            f"is not allowed by the taxonomy spec. content={content[:60]}"
        )
        return None
    memory_type = normalize_memory_type(taxonomy.legacy_type)

    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)

    # spec §1.5.1 闸门: L1 SINGLETON 子类 (姓名/年龄/生日 等身份硬唯一字段) 永
    # 远只能 1 条 L1. 即便 LLM 把"我今年28岁"复述为"我今年28岁，生日是3月15号"
    # 这种合并表述, dedup 单看跟任一已有 L1 都 < 0.85 阈值拦不住, 这里硬拦.
    if level == 1 and is_singleton(taxonomy.main_category, taxonomy.sub_category):
        # find_many(take=1) 比 count() 快: Prisma count 生成 SELECT COUNT(*) 是
        # 全过滤行扫描, take=1 生成 LIMIT 1 命中第一行就停; 没建索引时差距更明显.
        existing = await memory_repo.find_many(
            source=repo_source,
            where={
                "userId": user_id,
                "workspaceId": workspace_id,
                "level": 1,
                "isArchived": False,
                "mainCategory": taxonomy.main_category,
                "subCategory": taxonomy.sub_category,
            },
            take=1,
        )
        if existing:
            # 用户姓名是对话里高频纠正/重设的当前称呼。新的姓名事实已经过
            # extraction 分类为 (user, 身份/姓名, L1)，因此应替换旧当前值，
            # 而不是被旧 L1 永久挡住；完全相同文本仍短路，避免重复写入。
            if repo_source == "user" and taxonomy.main_category == "身份" and taxonomy.sub_category == "姓名":
                old_record = existing[0]
                old_text = getattr(old_record, "summary", None) or getattr(old_record, "content", None)
                if _normalize_singleton_text(old_text) != _normalize_singleton_text(summary or content):
                    await memory_repo.update(
                        old_record.id,
                        source=repo_source,
                        record=old_record,
                        isArchived=True,
                    )
                    try:
                        await log_memory_changelog(
                            user_id,
                            old_record.id,
                            "singleton_replaced",
                            old_value=getattr(old_record, "content", None),
                            new_value=content,
                            workspace_id=workspace_id,
                        )
                    except Exception:
                        pass
                    logger.info(
                        f"L1 SINGLETON replaced: ({repo_source}, {taxonomy.main_category}/"
                        f"{taxonomy.sub_category}) old_id={old_record.id} "
                        f"new_content={content[:60]}"
                    )
                else:
                    logger.info(
                        f"L1 SINGLETON blocked: ({repo_source}, {taxonomy.main_category}/"
                        f"{taxonomy.sub_category}) 已有 L1 {existing[0].id}, 拒收新条目. "
                        f"new_content={content[:60]}"
                    )
                    return None
            else:
                logger.info(
                    f"L1 SINGLETON blocked: ({repo_source}, {taxonomy.main_category}/"
                    f"{taxonomy.sub_category}) 已有 L1 {existing[0].id}, 拒收新条目. "
                    f"new_content={content[:60]}"
                )
                return None

    # Generate embedding
    embedding = await generate_embedding(content)

    # Deduplication check
    if await is_duplicate(user_id, content, embedding, workspace_id=workspace_id):
        return None

    # Store in PostgreSQL (routed to memories_user or memories_ai)
    create_data = dict(
        userId=user_id,
        content=content,
        summary=summary or content[:200],
        level=level,
        importance=importance,
        type=memory_type,
        mainCategory=taxonomy.main_category,
        subCategory=taxonomy.sub_category,
        workspaceId=workspace_id,
    )
    if occur_time is not None:
        create_data["occurTime"] = occur_time
    # Part 5 §3.1: statement_time = 用户说出这句话的时间 (消息接收时刻)
    # 调用方未提供时, 用 now 作为最佳估计 (extraction 时机近似消息时刻)
    create_data["statementTime"] = statement_time or datetime.now(timezone.utc)
    # Part 5 §4.2: 提醒重复规则; 仅提醒子类有效, 其他子类强制清空避免脏数据.
    if recurrence and taxonomy.sub_category == "提醒":
        create_data["recurrence"] = recurrence
    memory = await memory_repo.create(source=source, **create_data)

    # Store embedding. If this fails the memory row exists but would never
    # be retrievable by vector search — delete the orphan to keep state
    # consistent. The caller sees the error and can retry.
    #
    # Phase 0.4: store_embedding 自身已有 retry (3 attempts), 走到 except 表示
    # PG 真实故障 (非 transient). Rollback 也 retry, rollback 失败就 emit
    # EVT_MEMORY_ORPHAN 让 admin 可查 (背景 cure 脚本扫这个事件).
    try:
        await store_embedding(memory.id, embedding)
    except Exception as embed_err:
        logger.error(
            f"Embedding store failed for memory {memory.id} after retries; "
            f"rolling back memory row. Error: {embed_err}"
        )
        rollback_succeeded = False
        for attempt in range(3):
            try:
                await memory_repo.delete(memory.id, source=source)  # type: ignore[arg-type]
                rollback_succeeded = True
                break
            except Exception as cleanup_err:
                if attempt == 2:
                    # Rollback 终极失败 — orphan 留在 DB, 标 observability
                    from app.observability.events import EVT_MEMORY_ORPHAN
                    logger.error(
                        f"[ORPHAN] memory {memory.id} rollback failed 3x: "
                        f"{cleanup_err}; orphan row retained in DB",
                        extra={
                            "event": EVT_MEMORY_ORPHAN,
                            "memory_id": memory.id,
                            "source": source,
                            "user_id": user_id,
                            "rollback_error": str(cleanup_err)[:100],
                            "embed_error": str(embed_err)[:100],
                        },
                    )
                else:
                    import asyncio as _asyncio
                    await _asyncio.sleep(0.1 * (attempt + 1))
        if not rollback_succeeded:
            # changelog 也不写 (orphan 已记 EVT_MEMORY_ORPHAN, 不再额外污染 changelog)
            pass
        raise

    # Changelog is advisory (portrait generation input); a failure here is
    # not worth rolling back the memory for.
    try:
        await log_memory_changelog(
            user_id,
            memory.id,
            "insert",
            new_value=content,
            workspace_id=workspace_id,
        )
    except Exception as e:
        logger.warning(f"Changelog write failed for memory {memory.id}: {e}")

    logger.info(f"Stored memory L{level} (importance={importance:.2f}): {content[:50]}")
    return memory.id
