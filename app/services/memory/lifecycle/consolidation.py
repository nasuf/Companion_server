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

import logging
from datetime import UTC, datetime, timedelta

from app.db import db
from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory.normalization import cosine_similarity
from app.services.memory.provenance import CONSOLIDATED, DAILY_SUMMARY
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.persistence import log_memory_changelog, store_memory
from app.services.prompting.store import PromptDisabledError, get_prompt_text

logger = logging.getLogger(__name__)

_MIN_AGE_DAYS = 30
_MIN_CLUSTER_SIZE = 5
_CLUSTER_SIMILARITY = 0.75
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
        SELECT m.id, m.content, m.summary, m.importance, m.main_category,
               m.sub_category, m.occur_time, m.created_at, m.provenance,
               e.embedding::text AS embedding_text
        FROM {table} m
        JOIN memory_embeddings e ON e.memory_id = m.id
        WHERE m.user_id = $1
          AND m.workspace_id = $2
          AND m.is_archived = false
          AND m.level = 3
          AND m.sub_category IS DISTINCT FROM '提醒'
          AND m.created_at < $3
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
        ORDER BY m.created_at ASC
        LIMIT {_CANDIDATE_LIMIT}
        """,
        user_id,
        workspace_id,
        cutoff,
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
        f"- {(c.get('summary') or c.get('content') or '').strip()}"
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
        summary=digest,
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

    for c in cluster:
        # skip_reconciliation 保证 new_id 是全新行, 不会等于任何簇内既有 id;
        # 这条防御仍保留以防未来 store_memory 语义变化.
        if c["id"] == new_id:
            continue
        try:
            await memory_repo.update(
                c["id"],
                source=source,  # type: ignore[arg-type]
                isArchived=True,
            )
            await log_memory_changelog(
                user_id, c["id"], "consolidated_into",
                old_value=(c.get("summary") or c.get("content") or "")[:200],
                new_value=new_id,
                workspace_id=workspace_id,
            )
        except Exception as e:
            logger.warning(f"consolidation archive failed for {c['id']}: {e}")
    return new_id


async def compress_l3_clusters_for_workspace(
    *, user_id: str, workspace_id: str | None,
    max_clusters: int = _MAX_CLUSTERS_PER_WORKSPACE,
) -> dict:
    """Compress eligible L3 clusters for one (user, workspace) scope."""
    stats = {"clusters": 0, "compressed_rows": 0, "digests": 0}
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
            new_id = await _compress_cluster(
                source=source, user_id=user_id,
                workspace_id=workspace_id, cluster=cluster,
            )
            if new_id:
                stats["digests"] += 1
                stats["compressed_rows"] += len(cluster)
    if stats["digests"]:
        logger.info(
            f"[CONSOLIDATION] ws={str(workspace_id)[:8]} "
            f"{stats['compressed_rows']} rows → {stats['digests']} digests"
        )
    return stats
