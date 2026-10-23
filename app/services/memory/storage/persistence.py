"""Memory storage service.

Stores memories to PostgreSQL with deduplication (cosine > 0.9).
Classifies into L1/L2/L3 levels.
"""

import logging
from datetime import datetime, timedelta, timezone

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
from app.services.schedule_domain.time_parser import resolve_occur_time
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
        exceeds_injection_limit,
    )

    if not content or not exceeds_injection_limit(content):
        return [content]
    if is_singleton(taxonomy.main_category, taxonomy.sub_category):
        return [content]

    pieces = split_multi_fact(content)
    # 拆分是"尽力而为"而不是"保证": 没有全角分号、整段单句、或段数超过 MAX_SEGMENTS
    # 时, split_multi_fact 原样返回, 于是超限内容照样落库 —— 不报错、不打日志。
    # 那正是这层收口最容易被误信的地方 (它看起来像个闸门, 实际是个尽力器)。
    # 这里不改成硬拒 (会让整条写入失败, 丢掉的信息比留一条难检索的更多), 但至少
    # 让失败可观测: 巡检看到的是结果, 这行日志能指出是哪条路径漏的。
    still_over = [p for p in pieces if exceeds_injection_limit(p)]
    if still_over:
        logger.warning(
            "[SPLIT-FAILED] %d/%d piece(s) still exceed the injection limit "
            "after splitting (%s/%s); stored anyway but unretrievable: %r",
            len(still_over), len(pieces),
            taxonomy.main_category, taxonomy.sub_category,
            still_over[0][:60],
        )
    return pieces


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


async def after_memory_changelog(*, user_id: str, memory_id: str, operation: str,
                                 workspace_id: str | None) -> None:
    """Run optional derived-state hooks only after a changelog has committed."""
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
        await after_memory_changelog(user_id=user_id, memory_id=memory_id,
            operation=operation, workspace_id=workspace_id)
    except Exception as e:
        logger.warning(f"Failed to write changelog: {e}")

async def _audit_write(database, user_id, workspace_id, memory_id, operation, *,
                       old_value=None, new_value=None):
    """Required audit for the primary write, using its owning transaction."""
    await database.memorychangelog.create(data={
        "userId": user_id, "workspaceId": workspace_id, "memoryId": memory_id,
        "operation": operation, "oldValue": old_value, "newValue": new_value,
    })


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


async def _generated_self_memory_allowed(user_id, workspace_id, content, provenance) -> bool:
    if provenance in {"profile_seed", "knowledge_seed"}:
        return True
    from app.services.persona_grounding import (
        generated_stable_self_claim, needs_location_verification,
        verify_generated_locations, workspace_agent,
    )
    if generated_stable_self_claim(content):
        logger.info("[persona-grounding] refusing generated stable identity claim")
        return False
    if needs_location_verification(content, implicit_self=True):
        # A location must be attributed to the exact generating Agent. Missing
        # scope is retryable, never resolved to the user's most recent Agent.
        agent = await workspace_agent(user_id, workspace_id)
        return not await verify_generated_locations(agent, [content], kind="memory")
    return True


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
    evidence_sources: tuple | None = None,
    extractor_version: str = "memory-write-v1",
    skip_reconciliation: bool = False,
    infer_time: bool = True,
    _singleton_locked: bool = False,
    _split_done: bool = False,
    _consolidation_staged: bool = False,
    write_receipt=None,
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
    from app.services.memory.recording import receipts
    receipt_hash = receipts.fingerprint({
        "content": content, "level": level, "importance": importance,
        "memory_type": memory_type, "main_category": main_category, "sub_category": sub_category,
        "occur_time": occur_time.isoformat() if occur_time else None,
        "statement_time": statement_time.isoformat() if statement_time else None,
        "recurrence": recurrence, "provenance": provenance, "infer_time": infer_time,
    }) if write_receipt is not None else ""
    if write_receipt is not None:
        if (user_id, workspace_id, source) != (write_receipt.scope.owner_user_id,
                write_receipt.scope.workspace_id, write_receipt.side):
            raise receipts.MemoryReceiptError("memory_plan_scope_changed")
        cached = await receipts.lookup(write_receipt, receipt_hash)
        if cached is not None:
            return cached["memory_id"]
    async def skipped():
        await receipts.record_skip(write_receipt, receipt_hash)
        return None

    # Source narrows to the literal Source type expected by the taxonomy
    repo_source = "ai" if source == "ai" else "user"
    if repo_source == "ai" and not await _generated_self_memory_allowed(
        user_id, workspace_id, content, provenance,
    ):
        return await skipped()
    if _consolidation_staged and (provenance != "consolidated" or not skip_reconciliation or level != 3):
        raise ValueError("Only independent L3 consolidation digests may be staged")
    if provenance == "daily_summary":
        level, importance = 3, min(0.49, max(0.1, importance))
    taxonomy = resolve_taxonomy(
        main_category=main_category,
        sub_category=sub_category,
        legacy_type=normalize_memory_type(memory_type),
        source=repo_source,
        level=level,
        provenance=provenance,
    )
    if (repo_source == "ai" and provenance in {"ai_authored", "daily_summary"}
            and taxonomy.main_category in {"身份", "偏好", "思维"}):
        logger.info("Refusing generated persona memory (%s)", taxonomy.main_category)
        return await skipped()
    if not taxonomy.allowed:
        # The (source, level, main) combination is forbidden by the spec —
        # e.g. trying to write an AI 身份/偏好/思维 memory at L2 or L3.
        # Refuse rather than silently mis-categorize. Caller can retry at
        # a different level (typically L1) if appropriate.
        logger.info(
            f"Refusing memory: ({repo_source}, L{level}, {taxonomy.main_category}) "
            f"is not allowed by the taxonomy spec. content={content[:60]}"
        )
        return await skipped()
    memory_type = normalize_memory_type(taxonomy.legacy_type)

    # occur_time 兜底 (2026-08): 事件类记忆 (生活/情绪) 若上游没给事件时间, 在这里
    # 解析 —— content 里有显性日期就用它 (相对说话时刻), 否则退到说话时刻本身
    # (event ≈ 说到它的时刻)。放在 store_memory 这个唯一收口, 让聊天抽取 / 每日
    # 总结 / 送礼红包 / L3 整合等**所有**写入路径都受益, 而不是各写一遍。分层与
    # 排除 (提醒/profile_seed/远过去词/非事件类) 见 time_parser.resolve_occur_time。
    # 放在拆分前 → 拆出的多行共享同一 occur_time。
    if occur_time is None and infer_time:
        occur_time = resolve_occur_time(
            content,
            statement_time=statement_time,
            main_category=taxonomy.main_category,
            sub_category=taxonomy.sub_category,
            provenance=provenance,
        )

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
            if not infer_time:
                from app.services.memory.recording.temporal_sources import uncertain_composite_sources
                evidence_sources = uncertain_composite_sources(evidence_sources or (), "split_unresolved")
                occur_time = statement_time = None
            first_id: str | None = None
            for piece_index, piece in enumerate(pieces):
                mid = await store_memory(
                    user_id, piece, level=level, importance=importance,
                    memory_type=memory_type, main_category=taxonomy.main_category,
                    sub_category=taxonomy.sub_category, source=source,
                    occur_time=occur_time, statement_time=statement_time,
                    workspace_id=workspace_id, recurrence=recurrence,
                    entities=entities, topics=topics, provenance=provenance,
                    evidence_sources=evidence_sources, extractor_version=extractor_version,
                    skip_reconciliation=skip_reconciliation, infer_time=infer_time, _split_done=True,
                    _consolidation_staged=_consolidation_staged,
                    write_receipt=write_receipt.piece(piece_index) if write_receipt is not None else None,
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
                    evidence_sources=evidence_sources, extractor_version=extractor_version,
                    skip_reconciliation=skip_reconciliation, infer_time=infer_time, _split_done=_split_done,
                    _consolidation_staged=_consolidation_staged,
                    _singleton_locked=True,
                    write_receipt=write_receipt,
                )
        except DistributedLockNotAcquired:
            logger.info(
                f"L1 SINGLETON write lock contention: ({repo_source}, "
                f"{taxonomy.main_category}/{taxonomy.sub_category}) held >10s "
                f"by another writer; dropping as duplicate. content={content[:60]}"
            )
            return await skipped()

    # spec §1.5.1 闸门: L1 SINGLETON 子类 (姓名/年龄/生日 等身份硬唯一字段) 永
    # 远只能 1 条 L1. 即便 LLM 把"我今年28岁"复述为"我今年28岁，生日是3月15号"
    # 这种合并表述, dedup 单看跟任一已有 L1 都 < 0.85 阈值拦不住, 这里硬拦.
    name_replacement = None
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
                    # Archive only inside the replacement transaction. A bad
                    # origin or failed insert must leave the current name live.
                    name_replacement = old_record
                else:
                    logger.info(
                        f"L1 SINGLETON blocked: ({repo_source}, {taxonomy.main_category}/"
                        f"{taxonomy.sub_category}) 已有 L1 {existing[0].id}, 拒收新条目. "
                        f"new_content={content[:60]}"
                    )
                    return await skipped()
            else:
                logger.info(
                    f"L1 SINGLETON blocked: ({repo_source}, {taxonomy.main_category}/"
                    f"{taxonomy.sub_category}) 已有 L1 {existing[0].id}, 拒收新条目. "
                    f"new_content={content[:60]}"
                )
                return await skipped()

    # Generate embedding
    embedding = await generate_embedding(content)

    # Reconciliation check: duplicate detection, richer update, and recall-echo
    # suppression all happen here. This supersedes the old boolean dedup gate
    # for the main write path; is_duplicate/find_duplicate_id remain for legacy
    # callers that only need a yes/no answer.
    if skip_reconciliation or name_replacement is not None:
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
        if write_receipt is not None:
            # A duplicate is a durable result too. Its mention and audit must
            # commit with its skip receipt, not repeat on a partial-batch retry.
            async with db.tx(timeout=timedelta(seconds=15)) as tx:
                cached = await receipts.claim(tx, write_receipt, receipt_hash)
                if cached is not None:
                    return cached["memory_id"]
                table = 'memories_ai' if repo_source == 'ai' else 'memories_user'
                updated = await tx.query_raw(f"""UPDATE {table} SET mention_count=mention_count+1
                    WHERE id=$1 AND user_id=$2 AND workspace_id=$3 AND NOT is_archived RETURNING id""",
                    decision.existing_id, user_id, workspace_id)
                if updated:
                    await _audit_write(tx, user_id, workspace_id, decision.existing_id,
                        "dedup_drop", old_value=getattr(decision.existing_record, 'content', None),new_value=content)
                await receipts.commit(tx, write_receipt, receipt_hash, None)
            return None
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
        return await skipped()

    if decision.action in {"update_existing", "merge_existing"} and decision.existing_id and decision.existing_record:
        updated_content = decision.merged_content or content
        if (repo_source == "ai" and updated_content != content
                and not await _generated_self_memory_allowed(user_id, workspace_id, updated_content, provenance)):
            return await skipped()
        update_data = dict(
            content=updated_content,
            level=min(decision.existing_record.level, level),
            importance=max(float(decision.existing_record.importance or 0), float(importance)),
            type=memory_type,
            mainCategory=taxonomy.main_category,
            subCategory=taxonomy.sub_category,
            statementTime=statement_time if not infer_time else statement_time or datetime.now(timezone.utc),
        )
        if not infer_time and updated_content != content:
            from app.services.memory.recording.temporal_sources import uncertain_composite_sources
            evidence_sources = uncertain_composite_sources(evidence_sources or (), "merge_unresolved")
            update_data["statementTime"] = None
            occur_time = None
        if occur_time is not None or not infer_time:
            update_data["occurTime"] = occur_time
        if recurrence and taxonomy.sub_category == "提醒":
            update_data["recurrence"] = recurrence

        # A rejected origin must roll back the vector as well as the row.
        updated_embedding = embedding
        if updated_content != content:
            updated_embedding = await generate_embedding(updated_content)
        from app.services.memory.manual_write import MemoryWriteError, lock_memories
        record = decision.existing_record
        if (record.userId, record.workspaceId, record.source) != (user_id, workspace_id, repo_source):
            raise MemoryWriteError(409, "memory_write_scope_changed")
        operation = "reconciliation_merge" if decision.action == "merge_existing" else "reconciliation_update"
        async with db.tx(timeout=timedelta(seconds=15)) as tx:
            cached = await receipts.claim(tx, write_receipt, receipt_hash)
            if cached is not None:
                return cached["memory_id"]
            # Preparation may call models. Lock and compare the exact persisted
            # snapshot only now; never overwrite a concurrent edit or deletion.
            await lock_memories(tx, [record])
            await memory_repo.update(record.id, source=repo_source, record=record, database=tx,
                evidence_sources=evidence_sources, extractor_version=extractor_version, **update_data)
            await store_embedding(record.id, updated_embedding, database=tx)
            await _audit_write(tx, user_id, workspace_id, record.id, operation,
                               old_value=record.content, new_value=updated_content)
            await receipts.commit(tx, write_receipt, receipt_hash, record.id, updated_content)
        await memory_repo.invalidate_scope(user_id, workspace_id)
        await after_memory_changelog(user_id=user_id, memory_id=record.id,
            operation=operation, workspace_id=workspace_id)
        logger.info(
            f"Memory reconciliation updated existing "
            f"(id={decision.existing_id[:8]}): {content[:50]}"
        )
        return decision.existing_id

    # Store in PostgreSQL (routed to memories_user or memories_ai)
    from app.services.memory.provenance import normalize_provenance

    create_data = dict(
        isArchived=_consolidation_staged,
        userId=user_id,
        content=content,
        level=level,
        importance=importance,
        type=memory_type,
        mainCategory=taxonomy.main_category,
        subCategory=taxonomy.sub_category,
        workspaceId=workspace_id,
        valueUpdatedAt=datetime.now(timezone.utc),
    )
    normalized_provenance = normalize_provenance(provenance)
    if normalized_provenance:
        create_data["provenance"] = normalized_provenance
    if occur_time is not None:
        create_data["occurTime"] = occur_time
    # Part 5 §3.1: statement_time = 用户说出这句话的时间 (消息接收时刻)
    # 调用方未提供时, 用 now 作为最佳估计 (extraction 时机近似消息时刻)
    create_data["statementTime"] = statement_time if not infer_time else statement_time or datetime.now(timezone.utc)
    # Part 5 §4.2: 提醒重复规则; 仅提醒子类有效, 其他子类强制清空避免脏数据.
    if recurrence and taxonomy.sub_category == "提醒":
        create_data["recurrence"] = recurrence
    async with db.tx(timeout=timedelta(seconds=15)) as tx:
        cached = await receipts.claim(tx, write_receipt, receipt_hash)
        if cached is not None:
            return cached["memory_id"]
        if level == 1 and is_singleton(taxonomy.main_category, taxonomy.sub_category):
            # Shared with lifecycle promotion and administrator replacement.
            from app.services.memory.lifecycle.lazy_update import scope_lock_key
            from app.services.memory.manual_write import lock_memories
            table = "memories_ai" if repo_source == "ai" else "memories_user"
            await tx.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)",
                                 scope_lock_key(repo_source, user_id, workspace_id))
            taken = await tx.query_raw(
                f"SELECT id FROM {table} WHERE user_id=$1 AND workspace_id IS NOT DISTINCT FROM $2::text "
                "AND main_category=$3 AND sub_category=$4 AND level=1 AND NOT is_archived LIMIT 1",
                user_id, workspace_id, taxonomy.main_category, taxonomy.sub_category,
            )
            if taken and (name_replacement is None or taken[0]["id"] != name_replacement.id):
                await receipts.commit(tx, write_receipt, receipt_hash, None)
                return None
            if name_replacement is not None:
                await lock_memories(tx, [name_replacement])
                await tx.usermemory.update(where={"id": name_replacement.id}, data={"isArchived": True})
                await _audit_write(tx, user_id, workspace_id, name_replacement.id, "singleton_replaced",
                                   old_value=name_replacement.content, new_value=content)
        memory = await memory_repo.create(source=repo_source, database=tx,
            evidence_sources=evidence_sources, extractor_version=extractor_version, **create_data)
        # All durable parts of this row commit together, including sources and
        # its vector. No post-commit compensating delete can erase a later edit.
        await store_embedding(memory.id, embedding, database=tx)
        await _audit_write(tx, user_id, workspace_id, memory.id, "insert", new_value=content)
        await receipts.commit(tx, write_receipt, receipt_hash, memory.id, content)

    # Derived-state/network work belongs after commit, outside database locks.
    await memory_repo.invalidate_scope(user_id, workspace_id)
    if name_replacement is not None:
        await after_memory_changelog(user_id=user_id, memory_id=name_replacement.id,
            operation="singleton_replaced", workspace_id=workspace_id)
    await after_memory_changelog(user_id=user_id, memory_id=memory.id,
        operation="insert", workspace_id=workspace_id)

    logger.info(f"Stored memory L{level} (importance={importance:.2f}): {content[:50]}")
    return memory.id
