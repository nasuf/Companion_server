"""Pin compression inputs and commit a digest and its parent trail together.

No network operation runs inside the final transaction. Parent text is retained;
the input receipt does not turn a lossy summary into a verified fact.
"""

from datetime import UTC, datetime, timedelta
import json
import logging
from types import SimpleNamespace
import uuid

from app.services.memory.evidence import (
    EvidenceSource,
    bind_memory_evidence,
    content_version,
    _table,
)
from app.services.memory.input_evidence import (
    prepare_input_origin,
    persist_input_origin,
)
from app.services.memory.lifecycle.lazy_update import scope_lock_key
from app.services.memory.manual_write import memory_scope, MemoryWriteError
from app.services.memory.storage import repo
from app.services.memory.storage.embedding import store_embedding
from app.services.memory.storage.persistence import after_memory_changelog
from app.services.memory.taxonomy import resolve_taxonomy

logger = logging.getLogger(__name__)

EXTRACTOR = "l3-compression-v1"
MIN_AGE_DAYS = 30
MAX_PARENTS = 300
MAX_INPUT_BYTES = 524288


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda v: v.isoformat() if isinstance(v, datetime) else str(v),
        allow_nan=False,
    )


async def pin_scope(database, *, user_id, workspace_id, lock=False):
    scope = await memory_scope(
        database, SimpleNamespace(userId=user_id, workspaceId=workspace_id), lock=lock
    )
    suffix = " FOR SHARE" if lock else ""
    takeovers = await database.query_raw(
        "SELECT status FROM human_takeovers WHERE user_id=$1" + suffix, user_id
    )
    if any(row["status"] == "active" for row in takeovers):
        raise MemoryWriteError(409, "consolidation_human_takeover")
    agents = await database.query_raw(
        "SELECT name,city,occupation,age FROM ai_agents WHERE id=$1" + suffix,
        scope["agent_id"],
    )
    return {**scope, "agent": agents[0]}


async def pin_parents(
    database, *, source, user_id, workspace_id, ids, cutoff, lock=False
):
    table = _table(source)
    if (
        not 5 <= len(ids) <= MAX_PARENTS
        or len(set(ids)) != len(ids)
        or any(not isinstance(mid, str) or not mid or len(mid) > 200 for mid in ids)
    ):
        raise ValueError("consolidation_invalid_parents")
    other = _table("user" if source == "ai" else "ai")
    rows = await database.query_raw(
        f"""SELECT m.id,m.content,m.importance,m.main_category,m.sub_category,m.occur_time,
          m.created_at,m.provenance,m.xmin::text AS row_version
        FROM {table} m WHERE m.id=ANY($1::text[]) AND m.user_id=$2 AND m.workspace_id=$3
          AND NOT m.is_archived AND m.level=3 AND m.created_at<$4::timestamptz AT TIME ZONE 'UTC'
          AND m.sub_category IS DISTINCT FROM '提醒'
          AND (m.provenance='daily_summary' OR (m.provenance IS NULL AND NOT EXISTS (
            SELECT 1 FROM memory_changelogs c WHERE c.memory_id=m.id AND c.operation='access')))
          AND NOT EXISTS (SELECT 1 FROM memory_changelogs c WHERE c.memory_id=m.id AND c.operation='consolidated_into')
          AND NOT EXISTS (SELECT 1 FROM {other} a WHERE a.id=m.id)
        ORDER BY m.id"""
        + (" FOR UPDATE OF m" if lock else ""),
        sorted(ids),
        user_id,
        workspace_id,
        cutoff,
    )
    if len(rows) != len(ids):
        raise MemoryWriteError(409, "consolidation_parents_changed_or_ineligible")
    if len({(r["main_category"], r["sub_category"]) for r in rows}) != 1:
        raise ValueError("consolidation_mixed_categories")
    if len(canonical(rows).encode()) > MAX_INPUT_BYTES:
        raise ValueError("consolidation_input_too_large")
    return rows


async def prepare_cluster(database, *, source, user_id, workspace_id, cluster):
    _table(source)
    if not workspace_id:
        raise ValueError("consolidation_workspace_required")
    cutoff = (datetime.now(UTC) - timedelta(days=MIN_AGE_DAYS)).isoformat()
    async with database.tx(timeout=timedelta(seconds=15)) as tx:
        await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        scope = await pin_scope(
            tx, user_id=user_id, workspace_id=workspace_id, lock=True
        )
        parents = await pin_parents(
            tx,
            source=source,
            user_id=user_id,
            workspace_id=workspace_id,
            ids=[r["id"] for r in cluster],
            cutoff=cutoff,
        )
    # The caller's clustering decision must refer to the same text we summarize.
    if {r["id"]: r["content"] for r in parents} != {
        r["id"]: r["content"] for r in cluster
    }:
        raise MemoryWriteError(409, "consolidation_candidate_content_changed")
    return {
        "scope": scope,
        "parents": parents,
        "cutoff": cutoff,
        "sampled_at": datetime.now(UTC).isoformat(),
        "side": source,
    }


async def commit_cluster(
    database,
    *,
    source,
    user_id,
    workspace_id,
    inputs,
    digest,
    embedding,
    prompt,
    model_name,
    output,
):
    """All durable effects succeed together; ambiguous commit errors never delete.

    The workspace/side lock serializes compression with lifecycle updates.
    Row locks protect against manual edits and writers without that scope lock.
    """
    parents = inputs["parents"]
    ids = sorted(r["id"] for r in parents)
    if inputs["side"] != source:
        raise ValueError("consolidation_side_mismatch")
    taxonomy = resolve_taxonomy(
        main_category=parents[0]["main_category"],
        sub_category=parents[0]["sub_category"],
        source=source,
        level=3,
        provenance="consolidated",
    )
    if not taxonomy.allowed:
        raise ValueError("consolidation_invalid_taxonomy")
    importance = min(
        0.49, max(0.1, max(float(r["importance"] or 0.3) for r in parents))
    )
    times = [
        (
            datetime.fromisoformat(r["occur_time"].replace("Z", "+00:00"))
            if isinstance(r["occur_time"], str)
            else r["occur_time"]
        )
        for r in parents
        if r["occur_time"]
    ]
    times = sorted(
        t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC) for t in times
    )
    # The receipt survives target deletion; restored originals cannot replay it.
    cluster_key = content_version(
        canonical([[r["id"], content_version(r["content"])] for r in parents])
    )
    origin = prepare_input_origin(
        kind="consolidation",
        payload={
            "format": EXTRACTOR,
            "side": source,
            "cluster_key": cluster_key,
            "summary": {
                "parents": len(parents),
                "side": source,
                "sampled_at": inputs["sampled_at"],
            },
            "inputs": inputs,
            "prompt": prompt,
            "model": model_name,
            "output": output,
        },
    )
    async with database.tx(timeout=timedelta(seconds=30)) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock($1::bigint)",
            scope_lock_key(source, user_id, workspace_id),
        )
        scope = await pin_scope(
            tx, user_id=user_id, workspace_id=workspace_id, lock=True
        )
        if scope != inputs["scope"]:
            raise MemoryWriteError(409, "consolidation_scope_changed")
        locked = await pin_parents(
            tx,
            source=source,
            user_id=user_id,
            workspace_id=workspace_id,
            ids=ids,
            cutoff=inputs["cutoff"],
            lock=True,
        )
        if locked != parents:
            raise MemoryWriteError(409, "consolidation_parent_version_changed")
        if await tx.query_raw(
            """SELECT id FROM memory_input_origins WHERE workspace_id=$1
            AND source_status='consolidation' AND payload_text::jsonb->'input'->>'side'=$2
            AND payload_text::jsonb->'input'->>'cluster_key'=$3 LIMIT 1""",
            workspace_id,
            source,
            cluster_key,
        ):
            raise MemoryWriteError(409, "consolidation_already_recorded")
        receipt = await persist_input_origin(
            tx,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=scope["agent_id"],
            origin=origin,
        )
        row = await repo.create(
            source=source,
            database=tx,
            userId=user_id,
            workspaceId=workspace_id,
            content=digest,
            level=3,
            importance=importance,
            currentScore=importance,
            type=taxonomy.legacy_type,
            mainCategory=taxonomy.main_category,
            subCategory=taxonomy.sub_category,
            provenance="consolidated",
            occurTime=times[len(times) // 2] if times else None,
            statementTime=datetime.now(UTC),
            valueUpdatedAt=datetime.now(UTC),
        )
        await store_embedding(row.id, embedding, database=tx)
        sources = (receipt,) + tuple(
            EvidenceSource(
                "memory",
                p["id"],
                side=source,
                relation="derived_from",
                expected_version=content_version(p["content"]),
            )
            for p in parents
        )
        for start in range(0, len(sources), 100):
            await bind_memory_evidence(
                database=tx,
                memory_id=row.id,
                side=source,
                user_id=user_id,
                workspace_id=workspace_id,
                sources=sources[start : start + 100],
                extractor_version=EXTRACTOR,
            )
        await tx.memorychangelog.create(
            data={
                "userId": user_id,
                "workspaceId": workspace_id,
                "memoryId": row.id,
                "operation": "insert",
                "newValue": digest,
            }
        )
        await tx.execute_raw(
            """INSERT INTO memory_changelogs(id,user_id,workspace_id,memory_id,operation,old_value,new_value)
            SELECT unnest($1::text[]),$2,$3,unnest($4::text[]),'consolidated_into',unnest($5::text[]),$6""",
            [str(uuid.uuid4()) for _ in ids],
            user_id,
            workspace_id,
            ids,
            [p["content"][:200] for p in parents],
            row.id,
        )
        await tx.execute_raw(
            f"UPDATE {_table(source)} SET is_archived=true WHERE id=ANY($1::text[])",
            ids,
        )
    await repo.invalidate_scope(user_id, workspace_id)
    try:
        await after_memory_changelog(
            user_id=user_id,
            memory_id=row.id,
            operation="insert",
            workspace_id=workspace_id,
        )
    except Exception:
        logger.warning("Consolidation postcommit hook failed", exc_info=True)
    return row.id
