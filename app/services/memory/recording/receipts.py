"""Frozen, scoped extraction plans and transactional memory-write receipts.

The existing chat trigger resumes a plan; this module is not a new scheduler.
The model never chooses identity, scope, receipt keys, or deletion permissions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from uuid import uuid4

from app.db import db
from app.services.memory.evidence import snapshot_message_sources
from app.services.runtime.execution_scope import (
    ExecutionScope,
    bind_conversation_scope,
    _read_scope,
    ExecutionScopeExpired,
    ExecutionScopeUnavailable,
)

MAX_FACTS = 100
PLAN_VERSION = 1


class MemoryReceiptError(RuntimeError):
    def __init__(self, reason="memory_receipt_conflict"):
        self.reason = reason
        super().__init__(reason)


def _json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value):
    return sha256(_json(value).encode()).hexdigest()


def _source_records(origins):
    return [
        {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in asdict(s).items()
            if k not in {"input_text", "temporal_basis"}
        }
        for s in origins
    ]


def _key(value):
    return int.from_bytes(sha256(value.encode()).digest()[:8], "big", signed=True)


@dataclass(frozen=True)
class WriteToken:
    batch_id: str
    fact_index: int
    epoch: int
    scope: ExecutionScope
    side: str
    piece_index: int = -1

    def piece(self, index):
        if not 0 <= index < 100:
            raise MemoryReceiptError("memory_plan_invalid")
        return replace(self, piece_index=index)


async def _guard(tx, token):
    current = await _read_scope(
        tx, token.scope.actor_user_id, token.scope.conversation_id, lock=True
    )
    if current != token.scope:
        raise ExecutionScopeExpired()
    rows = await tx.query_raw(
        "SELECT generation FROM memory_erasure_epochs WHERE workspace_id=$1 FOR SHARE",
        token.scope.workspace_id,
    )
    if not rows or rows[0]["generation"] != token.epoch:
        raise MemoryReceiptError("memory_erasure_changed")
    plans = await tx.query_raw(
        "SELECT scope,side,extraction,status FROM memory_extraction_batches WHERE id=$1 FOR SHARE",
        token.batch_id,
    )
    if (
        not plans
        or plans[0]["scope"] != token.scope.to_record()
        or plans[0]["side"] != token.side
        or plans[0]["status"] not in {"processing", "completed"}
    ):
        raise MemoryReceiptError("memory_plan_invalid")
    plan = plans[0]
    if plan["extraction"] is not None:
        facts = plan["extraction"]["memories"]
        if token.fact_index >= len(facts):
            raise MemoryReceiptError("memory_plan_invalid")
        if facts[token.fact_index].get("_erased"):
            return False
    return plan["status"]


async def claim(tx, token: WriteToken | None, input_hash: str):
    """Serialize one piece and return its committed receipt, if any."""
    if token is None:
        return None
    allowed = await _guard(tx, token)
    await tx.execute_raw(
        "SELECT pg_advisory_xact_lock($1::bigint)",
        _key(f"memory-receipt:{token.batch_id}:{token.fact_index}:{token.piece_index}"),
    )
    rows = await tx.query_raw(
        """SELECT * FROM memory_write_receipts
        WHERE batch_id=$1 AND fact_index=$2 AND piece_index=$3 FOR UPDATE""",
        token.batch_id,
        token.fact_index,
        token.piece_index,
    )
    if not allowed:
        return {"status": "erased", "memory_id": None}
    if rows:
        if rows[0]["input_hash"] != input_hash:
            raise MemoryReceiptError("memory_plan_input_changed")
        return rows[0]
    if allowed == "completed":
        raise MemoryReceiptError("memory_plan_already_completed")
    return None


async def commit(
    tx, token: WriteToken | None, input_hash: str, memory_id, content=None
):
    if token is None:
        return
    revisions = (
        (
            await tx.query_raw(
                "SELECT id FROM memory_temporal_versions WHERE memory_id=$1 AND side=$2 AND user_id=$3 AND workspace_id=$4 ORDER BY id DESC LIMIT 1",
                memory_id,
                token.side,
                token.scope.owner_user_id,
                token.scope.workspace_id,
            )
        )
        if memory_id
        else []
    )
    await tx.execute_raw(
        """INSERT INTO memory_write_receipts
        (batch_id,fact_index,piece_index,input_hash,status,memory_id,memory_version,memory_revision)
        VALUES($1,$2,$3,$4,$5,$6,$7,$8)""",
        token.batch_id,
        token.fact_index,
        token.piece_index,
        input_hash,
        "written" if memory_id else "skipped",
        memory_id,
        sha256(content.encode()).hexdigest() if content is not None else None,
        revisions[0]["id"] if revisions else None,
    )


async def lookup(token, input_hash):
    if token is None:
        return None
    async with db.tx() as tx:
        return await claim(tx, token, input_hash)


async def record_skip(token, input_hash):
    if token is not None:
        async with db.tx() as tx:
            receipt = await claim(tx, token, input_hash)
            if receipt is None:
                await commit(tx, token, input_hash, None)


def _sanitize(extraction, source_count):
    if not isinstance(extraction, dict) or not isinstance(
        extraction.get("memories"), list
    ):
        raise MemoryReceiptError("memory_plan_invalid")
    facts = extraction["memories"]
    if len(facts) > MAX_FACTS or any(not isinstance(f, dict) for f in facts):
        raise MemoryReceiptError("memory_plan_invalid")
    payload = {
        key: extraction.get(key, [])
        for key in ("memories", "entities", "topics", "preferences")
    }
    for key in ("entities", "topics", "preferences"):
        if not isinstance(payload[key], list):
            payload[key] = []
    payload["plan_version"] = PLAN_VERSION
    # The field is an attribution proposal, not identity. Invalid proposals stay
    # unknown; normalize here so erasure never depends on arbitrary JSON types.
    payload = json.loads(_json(payload))
    for f in payload["memories"]:
        f.pop("_erased", None)
        f.pop("_clock", None)
        if not isinstance(f.get("content", ""), str):
            raise MemoryReceiptError("memory_plan_invalid")
        for key in ("main_category", "sub_category", "type", "recurrence"):
            if f.get(key) is not None and not isinstance(f[key], str):
                f[key] = None
        for key in ("entities", "topics"):
            if not isinstance(f.get(key, []), list):
                f[key] = []
        indices = f.get("source_indices")
        if not isinstance(indices, list) or any(
            type(i) is not int or not 1 <= i <= source_count for i in indices
        ):
            f["source_indices"] = []
    if len(_json(payload).encode()) > 262144:
        raise MemoryReceiptError("memory_plan_invalid")
    return payload


async def freeze(batch, extraction, origins):
    from app.services.memory.recording.temporal_sources import resolve_fact_clock
    from app.services.memory.taxonomy import SUBCATEGORY_ALIASES

    payload = _sanitize(extraction, len(batch["sources"]))
    for fact in payload["memories"]:
        sub = fact.get("sub_category")
        sub = SUBCATEGORY_ALIASES.get(sub, sub) if isinstance(sub, str) else None
        clock = resolve_fact_clock({**fact, "sub_category": sub}, origins)
        fact["_clock"] = {
            "statement_time": (
                clock.statement_time.isoformat() if clock.statement_time else None
            ),
            "occur_time": clock.occur_time.isoformat() if clock.occur_time else None,
            "basis": clock.basis,
            "indices": list(clock.indices),
        }
    if len(_json(payload).encode()) > 262144:
        raise MemoryReceiptError("memory_plan_invalid")
    async with db.tx() as tx:
        current = await _read_scope(
            tx, batch["scope"]["actor_user_id"], batch["conversation_id"], lock=True
        )
        if current.to_record() != batch["scope"]:
            raise ExecutionScopeExpired()
        epoch = await tx.query_raw(
            "SELECT generation FROM memory_erasure_epochs WHERE workspace_id=$1 FOR SHARE",
            batch["workspace_id"],
        )
        if not epoch or epoch[0]["generation"] != batch["epoch"]:
            raise MemoryReceiptError("memory_erasure_changed")
        await _inputs(batch, database=tx, lock=True)
        stored = await tx.query_raw(
            "SELECT sources FROM memory_extraction_batches WHERE id=$1 FOR UPDATE",
            batch["id"],
        )
        if not stored or stored[0]["sources"] != batch["sources"]:
            raise MemoryReceiptError("memory_plan_source_changed")
        rows = await tx.query_raw(
            """UPDATE memory_extraction_batches
            SET extraction=COALESCE(extraction,$2::jsonb),fact_count=COALESCE(fact_count,jsonb_array_length($2::jsonb->'memories')),status='processing',updated_at=CURRENT_TIMESTAMP
            WHERE id=$1 AND status IN ('preparing','processing') RETURNING extraction""",
            batch["id"],
            _json(payload),
        )
        if not rows:
            raise MemoryReceiptError("memory_plan_invalid")
        return rows[0]["extraction"]


async def _admit(scope, side, origins):
    async with db.tx() as tx:
        if (
            await _read_scope(tx, scope.actor_user_id, scope.conversation_id, lock=True)
            != scope
        ):
            raise ExecutionScopeExpired()
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock($1::bigint)",
            _key(f"memory-plan:{scope.conversation_id}:{side}"),
        )
        await tx.execute_raw(
            "INSERT INTO memory_erasure_epochs(workspace_id) VALUES($1) ON CONFLICT DO NOTHING",
            scope.workspace_id,
        )
        await tx.query_raw(
            "SELECT generation FROM memory_erasure_epochs WHERE workspace_id=$1 FOR SHARE",
            scope.workspace_id,
        )
        claims = await tx.query_raw(
            """SELECT s.message_id,s.batch_id FROM memory_extraction_sources s
            JOIN memory_extraction_batches b ON b.id=s.batch_id
            WHERE s.message_id=ANY($1::text[]) AND s.side=$2 AND b.conversation_id=$3
            AND b.user_id=$4 AND b.workspace_id=$5""",
            [s.ref for s in origins],
            side,
            scope.conversation_id,
            scope.owner_user_id,
            scope.workspace_id,
        )
        claimed = {r["message_id"] for r in claims}
        batch_ids = list(dict.fromkeys(r["batch_id"] for r in claims))
        erased = await tx.query_raw(
            """SELECT message_id,source_version FROM memory_erased_sources
            WHERE workspace_id=$1 AND side=$2 AND message_id=ANY($3::text[])""",
            scope.workspace_id,
            side,
            [s.ref for s in origins],
        )
        blocked = {(r["message_id"], r["source_version"]) for r in erased}
        fresh = [
            s
            for s in origins
            if s.ref not in claimed and (s.ref, s.expected_version) not in blocked
        ]
        if fresh:
            bid = str(uuid4())
            await tx.execute_raw(
                """INSERT INTO memory_extraction_batches
                (id,user_id,workspace_id,conversation_id,side,scope,sources)
                VALUES($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb)""",
                bid,
                scope.owner_user_id,
                scope.workspace_id,
                scope.conversation_id,
                side,
                _json(scope.to_record()),
                _json(_source_records(fresh)),
            )
            for source in fresh:
                await tx.execute_raw(
                    "INSERT INTO memory_extraction_sources(message_id,side,batch_id) VALUES($1,$2,$3)",
                    source.ref,
                    side,
                    bid,
                )
            batch_ids.append(bid)
        return batch_ids


async def _inputs(batch, *, database=None, lock=False):
    from app.services.memory.recording.chat_sources import extraction_text

    ids = [s["ref"] for s in batch["sources"]]
    database = database if database is not None else db
    rows = await database.query_raw(
        "SELECT id,role,content,metadata FROM messages WHERE id=ANY($1::text[]) ORDER BY id"
        + (" FOR SHARE" if lock else ""),
        ids,
    )
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(ids):
        raise MemoryReceiptError("memory_plan_source_changed")
    text = "\n".join(f"{by_id[i]['role']}: {extraction_text(by_id[i])}" for i in ids)
    try:
        origins = await snapshot_message_sources(
            user_id=batch["user_id"],
            workspace_id=batch["workspace_id"],
            side=batch["side"],
            message_ids=ids,
            extraction_input=text,
            conversation_id=batch["conversation_id"],
            database=database,
        )
    except ValueError as exc:
        raise MemoryReceiptError("memory_plan_source_changed") from exc
    if _source_records(origins) != batch["sources"]:
        raise MemoryReceiptError("memory_plan_source_changed")
    return ids, text


async def _load_batch(bid, scope):
    async with db.tx() as tx:
        if (
            await _read_scope(tx, scope.actor_user_id, scope.conversation_id, lock=True)
            != scope
        ):
            raise ExecutionScopeExpired()
        epochs = await tx.query_raw(
            "SELECT generation FROM memory_erasure_epochs WHERE workspace_id=$1 FOR SHARE",
            scope.workspace_id,
        )
        rows = await tx.query_raw(
            "SELECT * FROM memory_extraction_batches WHERE id=$1 FOR UPDATE", bid
        )
        if not rows:
            raise MemoryReceiptError("memory_plan_scope_changed")
        batch = rows[0]
        if batch["scope"] != scope.to_record():
            if (batch["user_id"], batch["workspace_id"], batch["conversation_id"]) != (
                scope.owner_user_id,
                scope.workspace_id,
                scope.conversation_id,
            ):
                raise MemoryReceiptError("memory_plan_scope_changed")
            # The same owner returned after a lifecycle change. A stale plan
            # must stop, without blocking unrelated future messages forever.
            await tx.execute_raw(
                "UPDATE memory_extraction_batches SET status='blocked',extraction=NULL,last_error='source_changed',updated_at=CURRENT_TIMESTAMP WHERE id=$1",
                bid,
            )
            batch.update(status="blocked", extraction=None)
            return batch
        batch["epoch"] = epochs[0]["generation"]
        if batch["status"] == "preparing":
            blocked = await tx.query_raw(
                "SELECT message_id,source_version FROM memory_erased_sources WHERE workspace_id=$1 AND side=$2 AND message_id=ANY($3::text[])",
                scope.workspace_id,
                batch["side"],
                [s["ref"] for s in batch["sources"]],
            )
            erased = {(r["message_id"], r["source_version"]) for r in blocked}
            remaining = [
                s
                for s in batch["sources"]
                if (s["ref"], s["expected_version"]) not in erased
            ]
            if not remaining:
                await tx.execute_raw(
                    "UPDATE memory_extraction_batches SET status='completed',extraction=NULL,result_ids='[]',updated_at=CURRENT_TIMESTAMP WHERE id=$1",
                    bid,
                )
                batch["status"] = "completed"
                batch["result_ids"] = []
            elif remaining != batch["sources"]:
                await tx.execute_raw(
                    "UPDATE memory_extraction_batches SET sources=$2::jsonb,updated_at=CURRENT_TIMESTAMP WHERE id=$1",
                    bid,
                    _json(remaining),
                )
                batch["sources"] = remaining
        return batch


async def _execute_batch(bid, scope, execute, kwargs):
    batch = await _load_batch(bid, scope)
    side = batch["side"]
    if (
        batch["extraction"] is not None
        and batch["extraction"].get("plan_version") != PLAN_VERSION
    ):
        raise MemoryReceiptError("memory_plan_version_unsupported")
    if batch["status"] == "blocked":
        return []
    if batch["status"] == "completed":
        table = "memories_user" if side == "user" else "memories_ai"
        rows = await db.query_raw(
            f"SELECT id FROM {table} WHERE user_id=$1 AND workspace_id=$2 AND id=ANY($3::text[]) AND NOT is_archived",
            scope.owner_user_id,
            scope.workspace_id,
            batch["result_ids"],
        )
        return [r["id"] for r in rows]
    if batch["status"] != "processing" and batch["status"] != "preparing":
        raise MemoryReceiptError("memory_plan_invalid")
    try:
        ids, text = await _inputs(batch)
        stored = await execute(
            user_id=scope.owner_user_id,
            workspace_id=scope.workspace_id,
            side=side,
            evidence_conversation_id=scope.conversation_id,
            evidence_message_ids=ids,
            new_conversation=text,
            _durable_batch=batch,
            **kwargs,
        )
        # Includes heuristic/no-op outcomes, where no write transaction had a
        # chance to bind the source. Never acknowledge an edited input.
        await _inputs(batch)
    except Exception as exc:
        # Preserve only controlled categories, never provider errors or text.
        permanent = (
            isinstance(exc, MemoryReceiptError)
            and exc.reason == "memory_plan_source_changed"
        )
        reason = "source_changed" if permanent else "processing_failed"
        await db.execute_raw(
            """UPDATE memory_extraction_batches
            SET last_error=$2,status=CASE WHEN $3 THEN 'blocked' ELSE status END,
                extraction=CASE WHEN $3 THEN NULL ELSE extraction END,updated_at=CURRENT_TIMESTAMP
            WHERE id=$1 AND status IN ('preparing','processing')""",
            bid,
            reason,
            permanent,
        )
        raise
    async with db.tx() as tx:
        if (
            await _read_scope(tx, scope.actor_user_id, scope.conversation_id, lock=True)
            != scope
        ):
            raise ExecutionScopeExpired()
        epochs = await tx.query_raw(
            "SELECT generation FROM memory_erasure_epochs WHERE workspace_id=$1 FOR SHARE",
            scope.workspace_id,
        )
        if not epochs or epochs[0]["generation"] != batch["epoch"]:
            raise MemoryReceiptError("memory_erasure_changed")
        await _inputs(batch, database=tx, lock=True)
        changed = await tx.execute_raw(
            "UPDATE memory_extraction_batches SET status='completed',extraction=NULL,fact_count=COALESCE(fact_count,0),result_ids=$2::jsonb,last_error=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=$1 AND status IN ('preparing','processing','completed')",
            bid,
            _json(stored),
        )
        if changed != 1:
            raise MemoryReceiptError("memory_plan_source_changed")
    return stored


async def process(
    *,
    execute,
    user_id,
    workspace_id,
    side,
    evidence_conversation_id,
    evidence_message_ids,
    new_conversation,
    **kwargs,
):
    """Resume the oldest plan before admitting new inputs or advancing progress."""
    try:
        origins = await snapshot_message_sources(
            user_id=user_id,
            workspace_id=workspace_id,
            side=side,
            message_ids=evidence_message_ids,
            extraction_input=new_conversation,
            conversation_id=evidence_conversation_id,
        )
    except ValueError as exc:
        raise MemoryReceiptError("memory_plan_source_changed") from exc
    scope = await bind_conversation_scope(
        actor_user_id=user_id, conversation_id=evidence_conversation_id
    )
    if scope.workspace_id != workspace_id:
        raise ExecutionScopeExpired()
    # One side is serialized by the existing conversation job lock. Bound this
    # recovery pass even for a legacy/test caller that bypasses that lock.
    pending = await db.query_raw(
        "SELECT id FROM memory_extraction_batches WHERE conversation_id=$1 AND side=$2 AND status IN ('preparing','processing') ORDER BY prepared_at,id LIMIT 5",
        scope.conversation_id,
        side,
    )
    if len(pending) > 4:
        raise MemoryReceiptError("memory_recovery_page_required")
    result = []
    seen = set()
    for row in pending:
        result.extend(await _execute_batch(row["id"], scope, execute, kwargs))
        seen.add(row["id"])
    for bid in await _admit(scope, side, origins):
        if bid not in seen:
            result.extend(await _execute_batch(bid, scope, execute, kwargs))
    return result


def saved_clock(fact):
    from app.services.memory.recording.temporal_sources import FactClock

    try:
        value = fact["_clock"]
        return FactClock(
            (
                datetime.fromisoformat(value["statement_time"])
                if value["statement_time"]
                else None
            ),
            (
                datetime.fromisoformat(value["occur_time"])
                if value["occur_time"]
                else None
            ),
            value["basis"],
            tuple(value["indices"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise MemoryReceiptError("memory_plan_invalid") from exc
