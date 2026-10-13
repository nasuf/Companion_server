"""Commit a daily review, its memories and private inputs as one unit."""

from datetime import datetime, timedelta, timezone
import json
import logging
import math
from zoneinfo import ZoneInfo

from app.services.memory.config import DEDUP_THRESHOLD
from app.services.memory.evidence import bind_memory_evidence
from app.services.memory.input_evidence import (
    prepare_input_origin,
    persist_input_origin,
)
from app.services.memory.lifecycle import capacity
from app.services.memory.lifecycle.lazy_update import scope_lock_key
from app.services.memory.manual_write import MemoryWriteError
from app.services.memory.polarity import semantic_conflict_reasons
from app.services.memory.recording.splitting import split_multi_fact
from app.services.memory.storage import repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.storage.persistence import (
    after_memory_changelog,
    _generated_self_memory_allowed,
)
from app.services.memory.taxonomy import resolve_taxonomy
from .review_inputs import collect_inputs, input_version

logger = logging.getLogger(__name__)
EXTRACTOR = "daily-review-v1"


def candidates(result, budget):
    if not isinstance(result, dict) or not isinstance(result.get("memories", []), list):
        raise ValueError("daily_review_invalid_classification")
    selected = []
    for mem in result.get("memories", [])[:50]:
        if isinstance(mem, str):
            content, main, score = mem, "生活", 30
        elif isinstance(mem, dict):
            content = mem.get("content", "")
            main = str(mem.get("type") or "生活").removesuffix("记忆")
            score = mem.get(
                "score",
                (
                    mem.get("importance", 0.3) * 100
                    if type(mem.get("importance", 0.3)) in (int, float)
                    else None
                ),
            )
        else:
            continue
        if (
            main not in {"生活", "情绪"}
            or not isinstance(content, str)
            or not content.strip()
            or len(content) > 10000
            or isinstance(score, bool)
            or not isinstance(score, (float, int))
            or not math.isfinite(score)
            or score < 10
        ):
            continue
        for piece in split_multi_fact(content):
            if len(selected) >= budget:
                return selected
            if piece.strip() and not any(s["content"] == piece for s in selected):
                selected.append(
                    {
                        "content": piece,
                        "main": main,
                        "importance": min(0.49, score / 100),
                    }
                )
    return selected


async def prepare_candidates(result, *, user_id, workspace_id, budget):
    prepared = []
    for item in candidates(result, budget):
        if await _generated_self_memory_allowed(
            user_id, workspace_id, item["content"], "daily_summary"
        ):
            prepared.append(
                {**item, "embedding": await generate_embedding(item["content"])}
            )
    return prepared


async def commit_review(
    database, *, review_id, user_id, workspace_id, agent_id, inputs, payload, prepared
):
    origin = prepare_input_origin(kind="daily_summary", payload=payload)
    stored = []
    async with database.tx(timeout=timedelta(seconds=30)) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock($1::bigint)",
            scope_lock_key("ai", user_id, workspace_id),
        )
        current = await collect_inputs(
            tx,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            start=datetime.fromisoformat(inputs["start"]).astimezone(
                ZoneInfo(inputs["timezone"])
            ),
            end=datetime.fromisoformat(inputs["end"]).astimezone(
                ZoneInfo(inputs["timezone"])
            ),
            lock=True,
        )
        if input_version(current) != input_version(inputs):
            raise MemoryWriteError(409, "daily_review_inputs_changed")
        claims = await tx.query_raw(
            """SELECT status FROM memory_daily_reviews WHERE id=$1
            AND workspace_id=$2 AND local_date=$3::date FOR UPDATE""",
            review_id,
            workspace_id,
            inputs["local_date"],
        )
        if not claims or claims[0]["status"] != "running":
            raise MemoryWriteError(409, "daily_review_claim_changed")
        if await tx.query_raw(
            """SELECT id FROM memory_input_origins WHERE workspace_id=$1
            AND source_status='daily_summary' AND payload_text::jsonb->'input'->>'local_date'=$2 LIMIT 1""",
            workspace_id,
            inputs["local_date"],
        ):
            # The durable receipt remains even if results or old claims were deleted.
            await tx.execute_raw(
                "UPDATE memory_daily_reviews SET status='completed',updated_at=NOW() WHERE id=$1",
                review_id,
            )
            return []
        count = await tx.aimemory.count(
            where={
                "userId": user_id,
                "workspaceId": workspace_id,
                "provenance": "daily_summary",
            }
        )
        if count + len(prepared) > capacity.DAILY_SUMMARY_ROW_BUDGET:
            raise MemoryWriteError(409, "daily_review_capacity_changed")
        source = await persist_input_origin(
            tx,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            origin=origin,
        )
        for item in prepared:
            vector = "[" + ",".join(str(float(v)) for v in item["embedding"]) + "]"
            matches = await tx.query_raw(
                """SELECT m.content,1-(e.embedding OPERATOR(extensions.<=>) $3::extensions.vector) AS similarity
                FROM memory_embeddings e JOIN memories_ai m ON m.id=e.memory_id
                WHERE m.user_id=$1 AND m.workspace_id=$2 AND NOT m.is_archived
                ORDER BY e.embedding OPERATOR(extensions.<=>) $3::extensions.vector LIMIT 5""",
                user_id,
                workspace_id,
                vector,
            )
            if any(
                float(r["similarity"]) > DEDUP_THRESHOLD
                and not semantic_conflict_reasons(item["content"], r["content"])
                for r in matches
            ):
                continue
            taxonomy = resolve_taxonomy(
                main_category=item["main"],
                legacy_type="life",
                source="ai",
                level=3,
                provenance="daily_summary",
            )
            if not taxonomy.allowed:
                raise ValueError("daily_review_invalid_taxonomy")
            row = await repo.create(
                source="ai",
                database=tx,
                userId=user_id,
                workspaceId=workspace_id,
                content=item["content"],
                level=3,
                importance=item["importance"],
                type=taxonomy.legacy_type,
                mainCategory=taxonomy.main_category,
                subCategory=taxonomy.sub_category,
                provenance="daily_summary",
                occurTime=datetime.fromisoformat(inputs["start"]),
                statementTime=datetime.fromisoformat(inputs["end"]),
                valueUpdatedAt=datetime.now(timezone.utc),
            )
            await store_embedding(row.id, item["embedding"], database=tx)
            await bind_memory_evidence(
                database=tx,
                memory_id=row.id,
                side="ai",
                user_id=user_id,
                workspace_id=workspace_id,
                sources=(source,),
                extractor_version=EXTRACTOR,
            )
            await tx.memorychangelog.create(
                data={
                    "userId": user_id,
                    "workspaceId": workspace_id,
                    "memoryId": row.id,
                    "operation": "insert",
                    "newValue": item["content"],
                }
            )
            stored.append(row.id)
        await tx.execute_raw(
            """UPDATE memory_daily_reviews SET status='completed',memory_ids=$2::jsonb,
            updated_at=NOW() WHERE id=$1""",
            review_id,
            json.dumps(stored),
        )
    await repo.invalidate_scope(user_id, workspace_id)
    for mid in stored:
        try:
            await after_memory_changelog(
                user_id=user_id,
                memory_id=mid,
                operation="insert",
                workspace_id=workspace_id,
            )
        except Exception:
            logger.warning("Daily review postcommit hook failed", exc_info=True)
    return stored
