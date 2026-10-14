"""L3 cluster compression (Phase 2 记忆整合).

The daily-life summary cron adds up to five L3 self-memories per agent per
day ("早上七点起床…"), and chat extraction adds more. L3 rows are only read by
L3 awakening, but they grow linearly and drown retrieval signal. This job
compresses same-topic L3 clusters into ONE summary memory and archives the
originals (recoverable — never physically deleted).

Complementary to lifecycle/hygiene.py: hygiene deduplicates near-identical
pairs via reconciliation; this job merges *many related-but-distinct* episodic
rows (5+ per cluster) into a digest.

Safety posture:
- Gated behind settings.memory_consolidation_enabled (currently default ON).
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
from app.services.memory.storage.persistence import _generated_self_memory_allowed
from app.services.memory.storage.embedding import generate_embedding
from app.services.memory.lifecycle.consolidation_write import (
    prepare_cluster, commit_cluster, MIN_AGE_DAYS as _MIN_AGE_DAYS, MAX_PARENTS as _CANDIDATE_LIMIT,
)
from app.services.prompting.store import PromptDisabledError, get_prompt_text

logger = logging.getLogger(__name__)

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
    try:
        inputs = await prepare_cluster(db, source=source, user_id=user_id,
                                       workspace_id=workspace_id, cluster=cluster)
    except Exception as exc:
        logger.warning("Consolidation input rejected: %s", type(exc).__name__)
        return None
    items = "\n".join(f"- {p['content'].strip()}" for p in inputs["parents"])
    try:
        tpl = await get_prompt_text("memory.consolidation")
    except PromptDisabledError:
        logger.info("memory.consolidation prompt disabled; skipping cluster")
        return None
    prompt = tpl.format(
        owner="我" if source == "ai" else "用户",
        main_category=inputs["parents"][0].get("main_category") or "生活",
        sub_category=inputs["parents"][0].get("sub_category") or "其他",
        memory_items=items,
    )
    model = get_utility_model()
    try:
        result = await invoke_json(model, prompt)
        digest = result.get("summary") if isinstance(result, dict) else None
        if not isinstance(digest, str) or len(digest.strip()) < 8:
            return None
        digest = digest.strip()
        from app.services.memory.retrieval.context_selector import estimate_tokens
        if estimate_tokens(digest) > 180:
            logger.warning("Consolidation digest exceeds per-memory context budget")
            return None
        if source == "ai" and not await _generated_self_memory_allowed(
                user_id, workspace_id, digest, CONSOLIDATED):
            return None
        embedding = await generate_embedding(digest)
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
        return await commit_cluster(db, source=source, user_id=user_id, workspace_id=workspace_id,
            inputs=inputs, digest=digest, embedding=embedding, prompt=prompt,
            model_name=model_name if isinstance(model_name, str) else None, output=result)
    except Exception as exc:
        # A lost COMMIT acknowledgment can leave a committed digest. The receipt
        # and archived parents fence replay; never compensate by deleting it.
        logger.warning("Consolidation cluster failed: %s", type(exc).__name__, exc_info=True)
        return None


async def compress_l3_clusters_for_workspace(
    *, user_id: str, workspace_id: str | None,
    max_clusters: int = _MAX_CLUSTERS_PER_WORKSPACE,
    dry_run: bool = False,
) -> dict:
    """Compress eligible L3 clusters for one (user, workspace) scope.

    dry_run 只做聚簇和统计, 不调 LLM 也不写库 —— 开 flag 前先看它会动多少东西,
    比开了再后悔便宜。
    """
    if type(max_clusters) is not int or not 1 <= max_clusters <= _MAX_CLUSTERS_PER_WORKSPACE:
        raise ValueError("Invalid consolidation cluster limit")
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
            logger.warning("Consolidation candidate load failed (%s): %s", source, type(e).__name__)
            stats["failed"] += 1
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
