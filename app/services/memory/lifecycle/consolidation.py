"""L3 cluster compression (Phase 2 记忆整合).

The daily-life summary cron adds 5-10 trivial L3 self-memories per agent per
day ("早上七点起床…"), and chat extraction adds more. L3 rows are only read by
L3 awakening, but they grow linearly and drown retrieval signal. This job
compresses same-topic L3 clusters into ONE summary memory and archives the
originals (recoverable — never physically deleted).

Complementary to lifecycle/hygiene.py: hygiene deduplicates near-identical
pairs via reconciliation; this job merges *many related-but-distinct* episodic
rows (5+ per cluster) into a digest.

Safety posture:
- Gated behind settings.memory_consolidation_enabled (default OFF).
- Originals are archived with a `consolidated_into:<id>` changelog trail.
- Only L3, only rows older than _MIN_AGE_DAYS. Candidate set = daily_summary
  rows (regardless of access history — trivia that has been injected before is
  still trivia) OR NULL-legacy rows that were never accessed. user_stated /
  ai_authored / consolidated rows are excluded (real grounding, stay separate).
- Per-workspace cluster cap bounds LLM cost per run.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

from app.db import db
from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory.normalization import cosine_similarity
from app.services.memory.provenance import (
    COMPRESSION_EXEMPT,
    CONSOLIDATED,
    DAILY_SUMMARY,
)
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.persistence import store_memory
from app.services.prompting.store import PromptDisabledError, get_prompt_text

logger = logging.getLogger(__name__)

_MIN_AGE_DAYS = 30
_MIN_CLUSTER_SIZE = 5

# 聚类阈值随嵌入模型走。0.75 是 bge-m3 时代定的, 换到 qwen3-embedding 后同题记忆
# 的相似度整体下移 —— 生产实测最大桶内最高两两相似度只有 0.567, 于是一簇都聚不
# 出来。它在上一轮阈值校准里被漏掉了, 因为当时整合是关闭的、不在校准范围内。
#
# 0.55 是在生产候选池上扫出来的: 它是最紧的、仍能聚出簇的阈值, 且抽出的簇确实
# 同题 (五条里四条讲午休)。再松到 0.40 会聚出 24 条的巨簇, 把不相干的日常混进
# 同一条摘要 —— 而原行归档后就找不回来了, 所以这里宁紧勿松。
# 重新标定: scripts/calibrate_cluster_threshold.py
_CLUSTER_SIMILARITY = 0.55
_MAX_CLUSTERS_PER_WORKSPACE = 10
_CANDIDATE_LIMIT = 300


async def _load_candidates(
    *, source: str, user_id: str, workspace_id: str | None,
) -> list[dict]:
    """L3 consolidation candidates with embeddings, oldest first.

    Prioritized: daily_summary provenance or legacy NULL rows that have never
    been accessed. user_stated/ai_authored rows are excluded — they carry real
    conversational grounding and stay individually recallable.
    """
    table = "memories_ai" if source == "ai" else "memories_user"
    cutoff = datetime.now(UTC) - timedelta(days=_MIN_AGE_DAYS)
    rows = await db.query_raw(
        f"""
        SELECT m.id, m.content, m.importance, m.main_category,
               m.sub_category, m.occur_time, m.created_at, m.provenance,
               e.embedding::text AS embedding_text
        FROM {table} m
        JOIN memory_embeddings e ON e.memory_id = m.id
        WHERE m.user_id = $1
          AND m.workspace_id = $2
          AND m.is_archived = false
          AND m.level = 3
          AND m.sub_category IS DISTINCT FROM '提醒'
          AND m.created_at < $3::timestamp
          AND (
            m.provenance = '{DAILY_SUMMARY}'
            OR (
              m.provenance IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM memory_changelogs cl
                WHERE cl.memory_id = m.id AND cl.operation = 'access'
              )
            )
          )
          -- 反思判断与既有摘要都豁免有损压缩 (见 provenance.COMPRESSION_EXEMPT)。
          -- 现在的候选白名单 (daily_summary / NULL) 本就把它们挡在外面, 显式写出来
          -- 是为了以后放宽白名单时不会顺手把它们放进来。
          AND COALESCE(m.provenance, '') <> ALL($4::text[])
          -- 已经被压缩过的行不再参与。批量归档虽然原子, 但"摘要已建、还没走到
          -- 归档"之间进程崩掉的话, 原行仍是未归档状态, 下一轮会把同一簇再压一
          -- 次、产出重复摘要。changelog 先于归档写入, 所以它能覆盖这个窗口。
          AND NOT EXISTS (
            SELECT 1 FROM memory_changelogs cl
            WHERE cl.memory_id = m.id AND cl.operation = 'consolidated_into'
          )
        ORDER BY m.created_at ASC
        LIMIT {_CANDIDATE_LIMIT}
        """,
        user_id,
        workspace_id,
        cutoff,
        sorted(COMPRESSION_EXEMPT),
    )
    out: list[dict] = []
    for r in rows:
        vec_text = r.get("embedding_text") or ""
        try:
            vec = [float(x) for x in vec_text.strip("[]").split(",") if x]
        except ValueError:
            continue
        if vec:
            r["_vec"] = vec
            out.append(r)
    return out


def _cluster(candidates: list[dict]) -> list[list[dict]]:
    """Greedy same-(main,sub) clustering by cosine similarity to cluster seed."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for c in candidates:
        key = (c.get("main_category") or "", c.get("sub_category") or "")
        buckets.setdefault(key, []).append(c)

    clusters: list[list[dict]] = []
    for rows in buckets.values():
        assigned: set[str] = set()
        for i, seed in enumerate(rows):
            if seed["id"] in assigned:
                continue
            cluster = [seed]
            for other in rows[i + 1:]:
                if other["id"] in assigned:
                    continue
                if cosine_similarity(seed["_vec"], other["_vec"]) >= _CLUSTER_SIMILARITY:
                    cluster.append(other)
            if len(cluster) >= _MIN_CLUSTER_SIZE:
                for c in cluster:
                    assigned.add(c["id"])
                clusters.append(cluster)
    return clusters


async def _compress_cluster(
    *, source: str, user_id: str, workspace_id: str | None, cluster: list[dict],
) -> str | None:
    """LLM-compress one cluster into a digest memory; archive the originals."""
    items = "\n".join(
        f"- {(c.get('content') or '').strip()}"
        for c in cluster
    )
    try:
        tpl = await get_prompt_text("memory.consolidation")
    except PromptDisabledError:
        logger.info("memory.consolidation prompt disabled; skipping cluster")
        return None
    prompt = tpl.format(
        owner="我" if source == "ai" else "用户",
        main_category=cluster[0].get("main_category") or "生活",
        sub_category=cluster[0].get("sub_category") or "其他",
        memory_items=items,
    )
    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"consolidation LLM failed: {e}")
        return None
    digest = str((result or {}).get("summary") or "").strip()
    if len(digest) < 8:
        return None

    importance = max(float(c.get("importance") or 0.3) for c in cluster)
    # query_raw returns timestamps as ISO strings — parse defensively.
    occur_times: list[datetime] = []
    for c in cluster:
        raw = c.get("occur_time")
        if isinstance(raw, datetime):
            occur_times.append(raw)
        elif isinstance(raw, str):
            try:
                occur_times.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except ValueError:
                pass
    occur_mid = sorted(occur_times)[len(occur_times) // 2] if occur_times else None

    new_id = await store_memory(
        user_id=user_id,
        content=digest,
        level=3,
        importance=min(0.49, importance),  # digests stay L3
        main_category=cluster[0].get("main_category"),
        sub_category=cluster[0].get("sub_category"),
        source=source,
        occur_time=occur_mid,
        workspace_id=workspace_id,
        provenance=CONSOLIDATED,
        # 强制插入独立行: 不走 reconciliation. 否则 digest 可能被 update_existing
        # 并进一条**非簇**同类记忆 (含 L2), 连带覆盖那条行内容并把它当作归档目标 —
        # 一次整合意外改写了不该动的记忆. digest 内容本就是新摘要, 无需去重.
        skip_reconciliation=True,
    )
    if not new_id:
        return None

    # skip_reconciliation 保证 new_id 是全新行, 不会等于任何簇内既有 id;
    # 这条防御仍保留以防未来 store_memory 语义变化.
    originals = [c for c in cluster if c["id"] != new_id]
    try:
        await _archive_originals(
            source=source, user_id=user_id, workspace_id=workspace_id,
            originals=originals, digest_id=new_id,
        )
    except Exception as e:
        # 归档没成 → 摘要必须撤销。留着它就等于"摘要已生成但原行还在", 下一轮
        # 同一簇会被再压一次, 产出重复摘要 —— 旧实现逐条归档 + 吞异常正是这个
        # 半失败状态的来源。
        logger.error(f"consolidation archive failed, rolling back digest {new_id}: {e}")
        await _rollback_digest(source=source, digest_id=new_id)
        return None
    return new_id


async def _archive_originals(
    *, source: str, user_id: str, workspace_id: str | None,
    originals: list[dict], digest_id: str,
) -> None:
    """把簇内原行一次性归档, 并留下可回滚的 changelog 轨迹。

    两条语句都是批量的, 各自天然原子 —— 旧实现逐条 update 且吞异常, 中途失败会
    留下"部分归档"的状态, 而部分归档意味着下一轮同一簇再被压一次。

    顺序是先写 changelog 再归档: 这样任何被归档的行一定有回滚线索。反过来的话,
    归档成功而 changelog 失败就产生了无从追溯的孤儿。
    """
    if not originals:
        return

    table = "memories_ai" if source == "ai" else "memories_user"
    ids = [c["id"] for c in originals]

    # old_value 存原文快照。原行只是归档不是删除, 所以数据本身不会丢 —— 但撤销
    # 脚本要靠它把"这次整合吞掉了什么"直接显示出来。少了它, 运维得先去两张表里捞
    # 原行才看得懂自己在撤什么, 那这份审计就等于没有。
    values = ",".join(
        f"(${i * 5 + 1}, ${i * 5 + 2}, ${i * 5 + 3}, ${i * 5 + 4}, "
        f"'consolidated_into', ${i * 5 + 5}, ${len(ids) * 5 + 1})"
        for i in range(len(ids))
    )
    args: list = []
    for c in originals:
        args.extend((
            str(uuid.uuid4()), user_id, workspace_id, c["id"],
            (c.get("content") or "")[:200],
        ))
    args.append(digest_id)
    await db.execute_raw(
        "INSERT INTO memory_changelogs "
        "(id, user_id, workspace_id, memory_id, operation, old_value, new_value) "
        f"VALUES {values}",
        *args,
    )

    archived = await db.execute_raw(
        f"UPDATE {table} SET is_archived = true "
        "WHERE id = ANY($1::text[]) AND is_archived = false",
        ids,
    )
    if archived != len(ids):
        # 少归档了几行 —— 可能是并发改动。让调用方回滚摘要, 下一轮重来。
        raise RuntimeError(
            f"archived {archived}/{len(ids)} rows; refusing partial consolidation"
        )


async def _rollback_digest(*, source: str, digest_id: str) -> None:
    """归档失败时撤销刚建的摘要。撤销本身失败只记日志 —— 再抛会掩盖真正的错因。"""
    try:
        await memory_repo.update(
            digest_id, source=source, isArchived=True,  # type: ignore[arg-type]
        )
    except Exception as e:
        logger.error(f"digest rollback failed for {digest_id}: {e}")


async def compress_l3_clusters_for_workspace(
    *, user_id: str, workspace_id: str | None,
    max_clusters: int = _MAX_CLUSTERS_PER_WORKSPACE,
    dry_run: bool = False,
) -> dict:
    """Compress eligible L3 clusters for one (user, workspace) scope.

    dry_run 只做聚簇和统计, 不调 LLM 也不写库 —— 开 flag 前先看它会动多少东西,
    比开了再后悔便宜。
    """
    stats: dict = {
        "clusters": 0, "compressed_rows": 0, "digests": 0,
        "failed": 0, "digest_ids": [], "dry_run": dry_run,
    }
    for source in ("ai", "user"):
        try:
            candidates = await _load_candidates(
                source=source, user_id=user_id, workspace_id=workspace_id,
            )
        except Exception as e:
            logger.warning(f"consolidation candidate load failed ({source}): {e}")
            continue
        if len(candidates) < _MIN_CLUSTER_SIZE:
            continue
        clusters = _cluster(candidates)[:max_clusters]
        for cluster in clusters:
            stats["clusters"] += 1
            if dry_run:
                stats["compressed_rows"] += len(cluster)
                continue
            new_id = await _compress_cluster(
                source=source, user_id=user_id,
                workspace_id=workspace_id, cluster=cluster,
            )
            if new_id:
                stats["digests"] += 1
                stats["compressed_rows"] += len(cluster)
                stats["digest_ids"].append(new_id)
            else:
                stats["failed"] += 1
    if stats["digests"] or stats["failed"]:
        logger.info(
            f"[CONSOLIDATION] ws={str(workspace_id)[:8]} "
            f"{stats['compressed_rows']} rows → {stats['digests']} digests "
            f"({stats['failed']} failed)"
        )
    if not dry_run and (stats["clusters"] or stats["failed"]):
        await _record_run(stats, user_id=user_id, workspace_id=workspace_id)
    return stats


async def _record_run(
    stats: dict, *, user_id: str | None, workspace_id: str | None,
) -> None:
    """记录一次整合的 run 级审计。

    这张表此前只有 hygiene 在写, 簇压缩完全没留痕 —— 而簇压缩才是会**归档原行**
    的那个, 出问题时最需要能回答"这条记忆去哪了"。job 列区分两者。

    审计写失败不冒泡: 它掩盖不了业务结果, 但让整合因为记不上账而失败更糟。
    """
    try:
        await db.execute_raw(
            """
            INSERT INTO memory_consolidation_runs (
                id, job, status, user_id, workspace_id,
                scopes, checked, archived, merged, errors, changes
            ) VALUES ($1, 'l3_compression', $2, $3, $4, 1, $5, $6, $7, $8, $9::jsonb)
            """,
            str(uuid.uuid4()),
            "succeeded" if not stats.get("failed") else "completed_with_errors",
            user_id,
            workspace_id,
            int(stats.get("clusters") or 0),
            int(stats.get("compressed_rows") or 0),
            int(stats.get("digests") or 0),
            int(stats.get("failed") or 0),
            json.dumps(
                {"digest_ids": stats.get("digest_ids") or []}, ensure_ascii=False,
            ),
        )
    except Exception as e:
        logger.warning(f"consolidation run audit skipped: {e}")
