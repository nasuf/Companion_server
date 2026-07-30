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
from app.services.memory.storage.reconciliation import resolve_memory_write
from app.services.memory.taxonomy import is_singleton, resolve_taxonomy
from app.services.memory.retrieval.vector_search import search_by_embedding
from app.services.runtime.distributed_lock import (
    DistributedLockNotAcquired,
    distributed_lock,
)
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


def _split_for_storage(content: str, taxonomy) -> list[str]:
    """写入前把超限内容拆成多条; 不需要拆或不能拆时返回单元素列表.

    单例类目 (姓名/性别/年龄…) 不拆 —— 那些子类每个 agent 只允许一行, 拆成两条会
    被 singleton 闸门拦下第二条, 反而丢内容。它们天然很短, 实际不会走到这。
    """
    from app.services.memory.recording.splitting import split_multi_fact
    from app.services.memory.retrieval.context_selector import (
        MAX_MEMORY_TOKENS_PER_ITEM,
        estimate_tokens,
    )

    if not content or estimate_tokens(content) <= MAX_MEMORY_TOKENS_PER_ITEM:
        return [content]
    if is_singleton(taxonomy.main_category, taxonomy.sub_category):
        return [content]
    return split_multi_fact(content)


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
        try:
            from app.services.memory.lifecycle.quality_state import refresh_quality_state_for_changelog

            # "access" can be emitted on every retrieval and should not turn the
            # recall hot path into a full quality recomputation. Backfill and
            # non-access writes keep the materialized state fresh enough for ops.
            if operation != "access":
                await refresh_quality_state_for_changelog(memory_id)
        except Exception as state_err:
            logger.debug(f"Memory quality state refresh skipped: {state_err}")
        try:
            if operation != "access":
                from app.services.achievements.service import handle_memory_changelog_event

                await handle_memory_changelog_event(
                    user_id=user_id,
                    memory_id=memory_id,
                    operation=operation,
                    workspace_id=workspace_id,
                )
        except Exception as achievement_err:
            logger.debug(f"Achievement memory hook skipped: {achievement_err}")
    except Exception as e:
        logger.warning(f"Failed to write changelog: {e}")

async def find_duplicate_id(
    user_id: str,
    content: str,
    embedding: list[float],
    workspace_id: str | None = None,
    source: str = "user",
) -> str | None:
    """If a near-duplicate memory already exists (cosine > DEDUP_THRESHOLD),
    return its id. Otherwise None.

    `source` scopes the search to the same owner table. Without it the UNION
    search could return an AI self-memory id for a user-side dedup — the caller
    would then update the wrong table (a non-existent id) and silently drop the
    reminder. Owner is a hard boundary here.

    Phase 3.1: 极性校验防反义误判. bge-m3 反义对 cosine 0.84-0.89, 容易超
    DEDUP_THRESHOLD=0.85 → 后写的"我不住北京"被当成已存"我住北京"的重复 →
    数据丢失. 加 polarity 校验: 极性不一致 → 不算重复, 都存.

    用 case: RECORD_REQUEST handler dedup 命中时需要拿到 existing memory id,
    才能 update occurTime + 重建 timetrigger (旧 trigger 已 fired 完, 新一次
    的"提醒我X"必须建新 trigger).
    """
    from app.services.memory.polarity import semantic_conflict_reasons

    results = await search_by_embedding(
        embedding, user_id, top_k=5, workspace_id=workspace_id, sources=[source],
    )
    for r in results:
        sim = r.get("similarity", 0)
        if isinstance(sim, str):
            sim = float(sim)
        if sim > DEDUP_THRESHOLD:
            mid = r.get("id")
            if not mid:
                continue
            # Phase 3.1/3.B: 极性/语义对立校验 — 反义对不算重复
            cand_text = r.get("content") or ""
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
    source: str = "user",
) -> bool:
    """向后兼容 wrapper. 新调用方应直接用 find_duplicate_id 拿 id."""
    matched = await find_duplicate_id(
        user_id, content, embedding, workspace_id=workspace_id, source=source,
    )
    return matched is not None


async def store_memory(
    user_id: str,
    content: str,
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
    entities: list[str] | None = None,
    topics: list[str] | None = None,
    provenance: str | None = None,
    skip_reconciliation: bool = False,
    _singleton_locked: bool = False,
    _split_done: bool = False,
) -> str | None:
    """Store a memory with deduplication.

    Returns memory_id if stored, None if duplicate.
    Args:
        source: "user" for memories about the user, "ai" for AI self-memories.
        recurrence: Part 5 §4.2 提醒重复规则 (once|yearly|monthly|weekly|daily).
            仅 sub_category="提醒" 时有效, 其他子类传 None.
        provenance: 记忆出处 (memory/provenance.py 常量); 非法值落 NULL.
            reconciliation 走 update/merge 时保留既有行的 provenance.
        skip_reconciliation: 跳过写入期矛盾/合并判定, 强制作为新行插入. 仅用于
            调用方已自证写入内容独立、且**不能**被 reconciliation 合并进无关既有
            行的场景 (如 L3 整合摘要 — 否则 digest 可能被 update_existing 并进一条
            非簇同类记忆, 连带把那条行内容覆盖). 常规写入切勿开启.
        _singleton_locked: 内部标志 — singleton 写锁已持有的重入调用, 勿外部使用.
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

    if not _split_done:
        pieces = _split_for_storage(content, taxonomy)
        if len(pieces) > 1:
            # 超过检索单条上限的记忆会被 context_selector 整条跳过 —— 存进去了, 但
            # 任何对话都不会用到它, 而且没有任何外部症状。这里是全部 11 个写入方的
            # 唯一收口, 在这拆一次就覆盖聊天抽取 / 每日总结 / L3 整合 / 矛盾纠正等
            # 全部路径 (2026-07 生产实测: 每日总结最长已到 171, 距上限只剩 9)。
            logger.info(
                "[STORE-SPLIT] content exceeds the per-item limit; storing as %d rows",
                len(pieces),
            )
            first_id: str | None = None
            for piece in pieces:
                mid = await store_memory(
                    user_id, piece, level=level, importance=importance,
                    memory_type=memory_type, main_category=taxonomy.main_category,
                    sub_category=taxonomy.sub_category, source=source,
                    occur_time=occur_time, statement_time=statement_time,
                    workspace_id=workspace_id, recurrence=recurrence,
                    entities=entities, topics=topics, provenance=provenance,
                    skip_reconciliation=skip_reconciliation, _split_done=True,
                )
                first_id = first_id or mid
            return first_id

    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)

    # TOCTOU 修复: singleton 的 find_many 检查与最终 create 之间隔着 embedding
    # 生成 + reconciliation 向量搜索 (秒级窗口), 并发 extraction 会双双通过检查
    # → L1 重复 (生产 case 2026-05-07). 用 per-(source, user, ws, 类目) Redis 锁
    # 把「检查→创建」整段串行化后重入自身; 等锁 10s 拿不到说明另一写入者正在写
    # 同类目 singleton, 本条按重复丢弃 (与 dedup drop 同语义, 后到的合并表述
    # 本来也会被 singleton 闸门拦下).
    if (
        level == 1
        and is_singleton(taxonomy.main_category, taxonomy.sub_category)
        and not _singleton_locked
    ):
        lock_name = (
            f"singleton_write:{repo_source}:{user_id}:{workspace_id}:"
            f"{taxonomy.main_category}:{taxonomy.sub_category}"
        )
        try:
            async with distributed_lock(
                lock_name, ttl_s=30, wait_timeout_s=10,
                retry_interval_s=0.2, fail_open=True,
            ):
                return await store_memory(
                    user_id, content, level=level,
                    importance=importance, memory_type=memory_type,
                    main_category=main_category, sub_category=sub_category,
                    source=source, occur_time=occur_time,
                    statement_time=statement_time, workspace_id=workspace_id,
                    recurrence=recurrence, entities=entities, topics=topics,
                    provenance=provenance,
                    _singleton_locked=True,
                )
        except DistributedLockNotAcquired:
            logger.info(
                f"L1 SINGLETON write lock contention: ({repo_source}, "
                f"{taxonomy.main_category}/{taxonomy.sub_category}) held >10s "
                f"by another writer; dropping as duplicate. content={content[:60]}"
            )
            return None

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
                old_text = getattr(old_record, "content", None)
                if _normalize_singleton_text(old_text) != _normalize_singleton_text(content):
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

    # Reconciliation check: duplicate detection, richer update, and recall-echo
    # suppression all happen here. This supersedes the old boolean dedup gate
    # for the main write path; is_duplicate/find_duplicate_id remain for legacy
    # callers that only need a yes/no answer.
    if skip_reconciliation:
        # Caller guarantees this row must be inserted standalone (never merged
        # into an unrelated existing row). Bypass update/merge/drop adjudication.
        from app.services.memory.storage.reconciliation import ReconciliationDecision

        decision = ReconciliationDecision(action="insert_new")
    else:
        decision = await resolve_memory_write(
            user_id=user_id,
            source=repo_source,
            workspace_id=workspace_id,
            content=content,
            embedding=embedding,
            main_category=taxonomy.main_category,
            sub_category=taxonomy.sub_category,
            entities=entities,
            topics=topics,
        )
    if decision.action == "drop_duplicate":
        try:
            from app.services.memory.lifecycle.decay import increment_mention_count

            if decision.existing_id:
                await increment_mention_count(decision.existing_id)
        except Exception:
            pass
        try:
            if decision.existing_id:
                await log_memory_changelog(
                    user_id,
                    decision.existing_id,
                    "dedup_drop",
                    old_value=getattr(decision.existing_record, "content", None),
                    new_value=content,
                    workspace_id=workspace_id,
                )
        except Exception:
            pass
        logger.info(
            f"Memory reconciliation dropped duplicate "
            f"(matched_id={str(decision.existing_id)[:8]}): {content[:50]}"
        )
        return None

    if decision.action in {"update_existing", "merge_existing"} and decision.existing_id and decision.existing_record:
        updated_content = decision.merged_content or content
        update_data = dict(
            content=updated_content,
            level=min(decision.existing_record.level, level),
            importance=max(float(decision.existing_record.importance or 0), float(importance)),
            type=memory_type,
            mainCategory=taxonomy.main_category,
            subCategory=taxonomy.sub_category,
            statementTime=statement_time or datetime.now(timezone.utc),
        )
        if occur_time is not None:
            update_data["occurTime"] = occur_time
        if recurrence and taxonomy.sub_category == "提醒":
            update_data["recurrence"] = recurrence

        # Keep vector and row consistent: update the embedding first, then row.
        updated_embedding = embedding
        if updated_content != content:
            updated_embedding = await generate_embedding(updated_content)
        await store_embedding(decision.existing_id, updated_embedding)
        await memory_repo.update(
            decision.existing_id,
            source=repo_source,
            record=decision.existing_record,
            **update_data,
        )
        try:
            await log_memory_changelog(
                user_id,
                decision.existing_id,
                "reconciliation_merge" if decision.action == "merge_existing" else "reconciliation_update",
                old_value=getattr(decision.existing_record, "content", None),
                new_value=update_data["content"],
                workspace_id=workspace_id,
            )
        except Exception as e:
            logger.warning(f"Changelog write failed for memory {decision.existing_id}: {e}")
        logger.info(
            f"Memory reconciliation updated existing "
            f"(id={decision.existing_id[:8]}): {content[:50]}"
        )
        return decision.existing_id

    # Store in PostgreSQL (routed to memories_user or memories_ai)
    from app.services.memory.provenance import normalize_provenance

    create_data = dict(
        userId=user_id,
        content=content,
        level=level,
        importance=importance,
        type=memory_type,
        mainCategory=taxonomy.main_category,
        subCategory=taxonomy.sub_category,
        workspaceId=workspace_id,
    )
    normalized_provenance = normalize_provenance(provenance)
    if normalized_provenance:
        create_data["provenance"] = normalized_provenance
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
